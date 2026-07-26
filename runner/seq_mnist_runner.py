"""Sequential MNIST runner — pixel-by-pixel classification (784 steps).

Tests whether InnerNet can discover gate mechanisms in RNNs.
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
from torchvision import datasets, transforms

logger = logging.getLogger('exp_logger')

CKPT_DIR = '/user_data/yizhouc3/xor_checkpoints' if os.path.isdir('/user_data/yizhouc3') else None


def save_ckpt(model, exp_name, cond, seed, epoch, optimizer=None, metrics=None):
    if CKPT_DIR is None:
        return
    d = os.path.join(CKPT_DIR, exp_name, f'{cond}_seed{seed}')
    os.makedirs(d, exist_ok=True)
    state = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        state['metrics'] = metrics
    torch.save(state, os.path.join(d, f'ep{epoch:03d}.pth'))


class SeqMNISTDataset(Dataset):
    """MNIST as sequential pixel-by-pixel input (784 steps × 1 feature)."""
    def __init__(self, train=True, data_path='./data'):
        self.dataset = datasets.MNIST(
            data_path, train=train, download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        seq = img.view(-1, 1)  # [784, 1]
        return seq, label


class SeqMNISTRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        sm = config.seq_mnist
        self.hidden_size = sm.get('hidden_size', 128)
        self.batch_size = sm.get('batch_size', 128)
        self.epochs = sm.get('epochs', 50)
        self.lr = sm.get('lr', 1e-3)
        self.grad_clip = sm.get('grad_clip', 1.0)
        self.num_seeds = sm.get('num_seeds', 5)
        self.num_workers = sm.get('num_workers', 4)
        self.inner_hidden = sm.get('inner_hidden', 32)
        self.cell_tanh = sm.get('cell_tanh', False)
        self.ortho_init = sm.get('ortho_init', False)
        self.warmup_epochs = sm.get('warmup_epochs', 0)

        self.model_name = config.model.name

    def _make_model(self):
        from model.seq_rnn import (SeqRNN, SeqLSTM, SeqGRU, SeqInnerNetRNN,
                                    SeqGatedRNN, SeqMinGatedRNN)
        if self.model_name == 'SeqRNN':
            return SeqRNN(1, self.hidden_size, 10)
        elif self.model_name == 'SeqLSTM':
            return SeqLSTM(1, self.hidden_size, 10)
        elif self.model_name == 'SeqGRU':
            return SeqGRU(1, self.hidden_size, 10)
        elif self.model_name == 'SeqInnerNetRNN':
            return SeqInnerNetRNN(1, self.hidden_size, 10, self.inner_hidden)
        elif self.model_name == 'SeqGatedRNN':
            return SeqGatedRNN(1, self.hidden_size, 10, self.inner_hidden, self.cell_tanh,
                               self.ortho_init)
        elif self.model_name == 'SeqMinGatedRNN':
            return SeqMinGatedRNN(1, self.hidden_size, 10, self.inner_hidden,
                                  self.ortho_init)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _get_loaders(self):
        data_path = self.config.get('data_path', './data')
        train_ds = SeqMNISTDataset(train=True, data_path=data_path)
        test_ds = SeqMNISTDataset(train=False, data_path=data_path)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        return train_loader, test_loader

    def train(self):
        train_loader, test_loader = self._get_loaders()
        all_results = []
        seeds = list(range(42, 42 + self.num_seeds))
        exp_name = os.path.basename(self.save_dir)

        for si, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = self._make_model().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)
            criterion = nn.CrossEntropyLoss()

            params = sum(p.numel() for p in model.parameters())
            logger.info(f"[Seed {seed}] ({si+1}/{self.num_seeds}) Training {self.model_name} "
                        f"({params} params) for {self.epochs} epochs...")

            best_acc = 0.0
            history = {'train_loss': [], 'test_acc': []}

            warmup_steps = self.warmup_epochs * len(train_loader)
            global_step = 0

            for epoch in range(1, self.epochs + 1):
                # Train
                model.train()
                total_loss = 0
                n_batches = 0
                for seqs, labels in train_loader:
                    # Linear LR warmup: ramp 0 -> base_lr over warmup_steps to let
                    # unstable inits settle before taking large optimizer steps.
                    if warmup_steps > 0 and global_step < warmup_steps:
                        warm_lr = self.lr * (global_step + 1) / warmup_steps
                        for pg in optimizer.param_groups:
                            pg['lr'] = warm_lr
                    global_step += 1

                    seqs, labels = seqs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    logits = model(seqs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / n_batches
                history['train_loss'].append(avg_loss)

                # Eval
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for seqs, labels in test_loader:
                        seqs, labels = seqs.to(self.device), labels.to(self.device)
                        logits = model(seqs)
                        preds = logits.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                acc = correct / total
                history['test_acc'].append(acc)

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(),
                               os.path.join(self.save_dir, f'best_model_seed{seed}.pth'))

                scheduler.step(acc)

                logger.info(f"  Seed {seed} Ep {epoch}/{self.epochs}: "
                            f"Loss={avg_loss:.4f} Acc={acc:.4f} (best={best_acc:.4f})")

                save_ckpt(model, exp_name, self.model_name, seed, epoch,
                          optimizer, {'loss': avg_loss, 'acc': acc, 'best_acc': best_acc})

                if math.isnan(avg_loss):
                    logger.info(f"  Seed {seed} NaN detected, stopping this seed.")
                    break

            logger.info(f"[Seed {seed}] Done. Best Acc: {best_acc:.4f}")
            all_results.append({
                'seed': seed, 'best_acc': best_acc, 'history': history
            })

        accs = [r['best_acc'] for r in all_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        logger.info(f"FINAL: {self.model_name} Acc = {mean_acc:.4f} ± {std_acc:.4f} "
                    f"({self.num_seeds} seeds)")

        pickle.dump(all_results, open(os.path.join(self.save_dir, 'results.p'), 'wb'))

    def test(self):
        _, test_loader = self._get_loaders()
        logger.info("Test mode not implemented for SeqMNIST — use train() which includes eval.")
