"""Language modeling experiment runner for LSTM with InnerNet activation.

Supports WikiText-2 dataset via HuggingFace datasets, multi-seed training.
"""
import os
import math
import random
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger('exp_logger')


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for next-token prediction."""

    def __init__(self, split='train', context_size=32, vocab=None):
        from datasets import load_dataset
        from collections import Counter

        self.context_size = context_size
        logger.info(f"Loading WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
        text = " ".join(dataset['text'])
        tokens = text.split()

        if vocab is None:
            counts = Counter(tokens)
            self.vocab = {word: i for i, (word, c) in enumerate(counts.most_common(9999))}
            self.vocab['<UNK>'] = len(self.vocab)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        self.data = torch.tensor(indices, dtype=torch.long)
        logger.info(f"Dataset ({split}) ready. {len(self.data)} tokens, vocab={self.vocab_size}")

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx):
        return self.data[idx: idx + self.context_size], self.data[idx + self.context_size]


def pretrain_inner_net_gaussian(inner_net, device, num_steps=300, lr=1e-2):
    """Pretrain InnerNet on Gaussian target function."""
    optimizer = optim.Adam(inner_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    xv, yv = np.meshgrid(x, y)
    inputs = torch.tensor(
        np.vstack([xv.reshape(-1), yv.reshape(-1)]).T,
        dtype=torch.float32
    ).to(device)
    targets = torch.exp(-(inputs[:, 0]**2 + inputs[:, 1]**2)).view(-1, 1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = criterion(inner_net(inputs), targets)
        loss.backward()
        optimizer.step()

    return inner_net.state_dict()


class LMRunner:
    """Runner for LSTM language modeling experiments."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        # LM hyperparameters
        lm = config.lm
        self.context_size = lm.get('context_size', 32)
        self.embed_dim = lm.get('embed_dim', 64)
        self.hidden_dim = lm.get('hidden_dim', 128)
        self.batch_size = lm.get('batch_size', 128)
        self.epochs = lm.get('epochs', 10)
        self.lr = lm.get('lr', 1e-3)
        self.grad_clip = lm.get('grad_clip', 1.0)
        self.num_seeds = lm.get('num_seeds', 5)
        self.num_workers = lm.get('num_workers', 4)

        # Model config
        self.model_name = config.model.name
        self.is_innernet = self.model_name in ('InnerNetLSTMModel', 'InnerNetTransformer')
        self.is_transformer = self.model_name in ('InnerNetTransformer', 'StandardTransformer', 'SwiGLUTransformer')

    def _make_model(self, vocab_size):
        if self.is_transformer:
            from model.transformer import InnerNetTransformer, StandardTransformer, SwiGLUTransformer
            d_model = self.config.model.get('d_model', 128)
            n_heads = self.config.model.get('n_heads', 4)
            d_ff = self.config.model.get('d_ff', 512)
            n_layers = self.config.model.get('n_layers', 4)
            max_len = self.config.lm.get('context_size', 64)
            dropout = self.config.model.get('dropout', 0.1)
            if self.model_name == 'InnerNetTransformer':
                inner_hidden = self.config.model.get('inner_hidden', 32)
                return InnerNetTransformer(vocab_size, d_model, n_heads, d_ff,
                                           n_layers, max_len, inner_hidden, dropout)
            elif self.model_name == 'SwiGLUTransformer':
                return SwiGLUTransformer(vocab_size, d_model, n_heads, d_ff,
                                         n_layers, max_len, dropout)
            else:
                return StandardTransformer(vocab_size, d_model, n_heads, d_ff,
                                           n_layers, max_len, dropout)
        else:
            from model.lstm import InnerNetLSTMModel, StandardLSTMModel
            if self.model_name == 'InnerNetLSTMModel':
                inner_hidden = self.config.model.get('inner_hidden', 32)
                return InnerNetLSTMModel(vocab_size, self.embed_dim, self.hidden_dim, inner_hidden)
            else:
                return StandardLSTMModel(vocab_size, self.embed_dim, self.hidden_dim)

    def train(self):
        """Train LSTM across multiple seeds."""
        # Load data once
        train_ds = WikiTextDataset(split='train', context_size=self.context_size)
        val_ds = WikiTextDataset(split='validation', context_size=self.context_size,
                                 vocab=train_ds.vocab)
        vocab_size = train_ds.vocab_size

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        # Pretrain InnerNet once
        gaussian_weights = None
        if self.is_innernet:
            logger.info("Pretraining InnerNet on Gaussian target...")
            if self.is_transformer:
                from model.transformer import InnerNetFFNActivation
                temp_inner = InnerNetFFNActivation(
                    hidden_dim=self.config.model.get('inner_hidden', 32)
                ).to(self.device)
            else:
                from model.lstm import InnerNetLSTMActivation
                temp_inner = InnerNetLSTMActivation(
                    hidden_dim=self.config.model.get('inner_hidden', 32)
                ).to(self.device)
            gaussian_weights = pretrain_inner_net_gaussian(temp_inner, self.device)
            logger.info("InnerNet pretrained.")

        seeds = list(range(42, 42 + self.num_seeds))
        all_ppl_histories = []

        for si, seed in enumerate(seeds):
            logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)}) Training {self.model_name} "
                       f"for {self.epochs} epochs...")
            ppl_history = self._train_single_seed(
                seed, train_loader, val_loader, vocab_size, gaussian_weights
            )
            all_ppl_histories.append(ppl_history)
            logger.info(f"[Seed {seed}] Done. Best PPL: {min(ppl_history):.2f}")

        # Save results
        data = np.array(all_ppl_histories)  # [num_seeds, epochs]
        results = {
            'model_name': self.model_name,
            'seeds': seeds,
            'all_ppl': all_ppl_histories,
            'mean_ppl': np.mean(data, axis=0).tolist(),
            'std_ppl': np.std(data, axis=0).tolist(),
            'best_mean_ppl': float(np.min(np.mean(data, axis=0))),
        }
        results_path = os.path.join(self.save_dir, 'lm_results.p')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        self._mark_stage('COMPLETED')
        logger.info(f"All seeds done. Best mean PPL: {results['best_mean_ppl']:.2f}")
        logger.info(f"Results saved to {results_path}")

    def _train_single_seed(self, seed, train_loader, val_loader, vocab_size, gaussian_weights):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = self._make_model(vocab_size).to(self.device)

        # Load pretrained InnerNet
        if self.is_innernet and gaussian_weights is not None:
            if self.is_transformer:
                # Load into each block's FFN inner_net
                for block in model.blocks:
                    block.ffn.inner_net.load_state_dict(gaussian_weights)
            else:
                model.cell.inner_net.load_state_dict(gaussian_weights)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        ppl_history = []

        for epoch in range(self.epochs):
            # Train
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

            # Validate
            model.eval()
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    out = model(x)
                    total_loss += criterion(out, y).item()
                    num_batches += 1

            ppl = math.exp(total_loss / num_batches)
            ppl_history.append(ppl)
            logger.info(f"  Seed {seed} Ep {epoch+1}/{self.epochs}: PPL = {ppl:.2f}")

        return ppl_history

    def test(self):
        """Test is not applicable separately. Results are saved during train."""
        logger.info("LM test: results were saved during training.")

    def _mark_stage(self, stage_name):
        marker = os.path.join(self.save_dir, stage_name)
        with open(marker, 'w') as f:
            f.write('')
