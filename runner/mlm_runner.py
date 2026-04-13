"""Masked Language Model runner — BERT-style pretraining.

Reuses existing Transformer blocks with bidirectional attention.
Masks 15% of tokens and predicts them. Compares GELU/InnerNet/SwiGLU FFN.
"""
import os
import math
import pickle
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("exp")


class MLMDataset(Dataset):
    """Dataset for masked language modeling. Returns fixed-length token sequences."""

    def __init__(self, split='train', seq_len=64, vocab=None, dataset_name='wikitext'):
        from collections import Counter

        self.seq_len = seq_len

        if dataset_name == 'ptb':
            split_map = {'train': 'train', 'validation': 'valid', 'test': 'test'}
            ptb_split = split_map.get(split, split)
            ptb_path = os.path.join('./data/ptb', f'{ptb_split}.txt')
            if not os.path.exists(ptb_path):
                ptb_path = os.path.join('./data/ptb', f'ptb.{ptb_split}.txt')
            with open(ptb_path) as f:
                text = f.read()
            tokens = text.split()
        else:
            from datasets import load_dataset
            name = 'wikitext-103-v1' if dataset_name == 'wikitext103' else 'wikitext-2-v1'
            logger.info(f"Loading {name} ({split})...")
            dataset = load_dataset('wikitext', name, split=split)
            text = " ".join(dataset['text'])
            tokens = text.split()

        if vocab is None:
            counts = Counter(tokens)
            self.vocab = {word: i for i, (word, _) in enumerate(counts.most_common(9999))}
            self.vocab['<UNK>'] = len(self.vocab)
            self.vocab['<MASK>'] = len(self.vocab)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)
        self.mask_token_id = self.vocab['<MASK>']
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        self.data = torch.tensor(indices, dtype=torch.long)
        logger.info(f"MLM Dataset ({split}): {len(self.data)} tokens, vocab={self.vocab_size}")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return self.data[idx: idx + self.seq_len]


class MLMWrapper(nn.Module):
    """Wraps a decoder Transformer for bidirectional MLM.

    Removes causal mask and returns logits for ALL positions.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        x = m.pos_enc(m.embedding(x) * math.sqrt(m.d_model))
        for block in m.blocks:
            x = block(x, mask=None)  # bidirectional — no causal mask
        x = m.ln_f(x)
        return m.head(x)  # [B, S, vocab_size]


class MLMRunner:
    """Runner for Masked Language Modeling experiments."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        lm = config.lm
        self.seq_len = lm.get('context_size', 64)
        self.batch_size = lm.get('batch_size', 128)
        self.epochs = lm.get('epochs', 10)
        self.lr = lm.get('lr', 5e-4)
        self.grad_clip = lm.get('grad_clip', 1.0)
        self.num_seeds = lm.get('num_seeds', 5)
        self.num_workers = lm.get('num_workers', 4)
        self.mask_prob = lm.get('mask_prob', 0.15)
        self.dataset_name = lm.get('dataset', 'wikitext')

        self.model_name = config.model.name
        self.is_innernet = 'InnerNet' in self.model_name

    def _make_model(self, vocab_size):
        """Build Transformer and wrap for MLM."""
        from model.transformer import (InnerNetTransformer, ClassicInnerNetTransformer,
                                       StandardTransformer, SwiGLUTransformer)

        d_model = self.config.model.get('d_model', 128)
        n_heads = self.config.model.get('n_heads', 4)
        d_ff = self.config.model.get('d_ff', 512)
        n_layers = self.config.model.get('n_layers', 4)
        max_len = self.seq_len
        dropout = self.config.model.get('dropout', 0.1)
        inner_hidden = self.config.model.get('inner_hidden', 32)

        if self.model_name == 'InnerNetTransformer':
            base = InnerNetTransformer(vocab_size, d_model, n_heads, d_ff,
                                       n_layers, max_len, inner_hidden, dropout)
        elif self.model_name == 'ClassicInnerNetTransformer':
            base = ClassicInnerNetTransformer(vocab_size, d_model, n_heads, d_ff,
                                              n_layers, max_len, inner_hidden, dropout)
        elif self.model_name == 'SwiGLUTransformer':
            base = SwiGLUTransformer(vocab_size, d_model, n_heads, d_ff,
                                     n_layers, max_len, dropout)
        else:
            base = StandardTransformer(vocab_size, d_model, n_heads, d_ff,
                                       n_layers, max_len, dropout)

        return MLMWrapper(base)

    def _mask_tokens(self, x, mask_token_id):
        """Apply BERT-style masking: 15% masked, of which 80% [MASK], 10% random, 10% keep."""
        labels = x.clone()
        mask = torch.rand(x.shape, device=x.device) < self.mask_prob
        labels[~mask] = -100  # only compute loss on masked tokens

        # 80% replace with [MASK]
        replace_mask = mask & (torch.rand(x.shape, device=x.device) < 0.8)
        x[replace_mask] = mask_token_id

        # 10% replace with random token
        random_mask = mask & ~replace_mask & (torch.rand(x.shape, device=x.device) < 0.5)
        x[random_mask] = torch.randint(0, mask_token_id, x[random_mask].shape, device=x.device)

        # 10% keep original (already in x)
        return x, labels

    def train(self):
        train_ds = MLMDataset(split='train', seq_len=self.seq_len,
                              dataset_name=self.dataset_name)
        val_ds = MLMDataset(split='validation', seq_len=self.seq_len,
                            vocab=train_ds.vocab, dataset_name=self.dataset_name)
        vocab_size = train_ds.vocab_size
        mask_token_id = train_ds.mask_token_id

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        # Pretrain InnerNet
        gaussian_weights = None
        if self.is_innernet:
            from runner.lm_runner import pretrain_inner_net_gaussian
            from model.transformer import InnerNetFFNActivation
            logger.info("Pretraining InnerNet on Gaussian target...")
            temp_inner = InnerNetFFNActivation(
                hidden_dim=self.config.model.get('inner_hidden', 32)
            ).to(self.device)
            gaussian_weights = pretrain_inner_net_gaussian(temp_inner, self.device)
            logger.info("InnerNet pretrained.")

        seeds = list(range(42, 42 + self.num_seeds))
        all_ppl_histories = []

        for si, seed in enumerate(seeds):
            logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)}) Training MLM {self.model_name} "
                       f"for {self.epochs} epochs...")
            ppl_history = self._train_single_seed(
                seed, train_loader, val_loader, vocab_size,
                mask_token_id, gaussian_weights
            )
            all_ppl_histories.append(ppl_history)
            logger.info(f"[Seed {seed}] Done. Best PPL: {min(ppl_history):.2f}")

        data = np.array(all_ppl_histories)
        results = {
            'model_name': self.model_name,
            'task': 'mlm',
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

    def _train_single_seed(self, seed, train_loader, val_loader, vocab_size,
                           mask_token_id, gaussian_weights):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = self._make_model(vocab_size).to(self.device)

        # Load pretrained InnerNet
        if self.is_innernet and gaussian_weights is not None:
            # Shared inner_net — loading once applies to all layers
            model.model.blocks[0].ffn.inner_net.load_state_dict(gaussian_weights)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        ppl_history = []

        for epoch in range(self.epochs):
            model.train()
            for x in train_loader:
                x = x.to(self.device, non_blocking=True)
                masked_x, labels = self._mask_tokens(x.clone(), mask_token_id)
                optimizer.zero_grad(set_to_none=True)
                logits = model(masked_x)  # [B, S, vocab]
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

            # Validate
            model.eval()
            total_loss = 0
            total_masked = 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(self.device, non_blocking=True)
                    masked_x, labels = self._mask_tokens(x.clone(), mask_token_id)
                    logits = model(masked_x)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    n_masked = (labels != -100).sum().item()
                    total_loss += loss.item() * n_masked
                    total_masked += n_masked

            avg_loss = total_loss / max(total_masked, 1)
            ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
            ppl_history.append(ppl)
            logger.info(f"  Seed {seed} Ep {epoch+1}/{self.epochs}: MLM PPL = {ppl:.2f}")

        return ppl_history

    def test(self):
        logger.info("MLM test: results were saved during training.")

    def _mark_stage(self, stage_name):
        marker = os.path.join(self.save_dir, stage_name)
        with open(marker, 'w') as f:
            f.write('')
