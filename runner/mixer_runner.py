"""MLP-Mixer experiment runner for image classification.

Supports InnerNetMLPMixer and StandardMLPMixer on CIFAR-10.
Multi-seed training with pretrain support for InnerNet variant.
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
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

logger = logging.getLogger('exp_logger')


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


class MixerRunner:
    """Runner for MLP-Mixer classification experiments."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        mc = config.mixer
        self.batch_size = mc.get('batch_size', 128)
        self.epochs = mc.get('epochs', 50)
        self.lr = mc.get('lr', 1e-3)
        self.num_seeds = mc.get('num_seeds', 5)
        self.num_workers = mc.get('num_workers', 4)

        self.model_name = config.model.name
        self.is_innernet = self.model_name == 'InnerNetMLPMixer'

    def _make_model(self):
        from model.mlp_mixer import InnerNetMLPMixer, StandardMLPMixer
        mc = self.config.model
        kwargs = dict(
            image_size=mc.get('image_size', 32),
            patch_size=mc.get('patch_size', 4),
            in_channels=mc.get('in_channels', 3),
            d_model=mc.get('d_model', 128),
            token_hidden=mc.get('token_hidden', 64),
            channel_hidden=mc.get('channel_hidden', 512),
            num_layers=mc.get('num_layers', 4),
            num_classes=mc.get('num_classes', 10),
            dropout=mc.get('dropout', 0.1),
        )
        if self.model_name == 'InnerNetMLPMixer':
            kwargs['inner_hidden'] = mc.get('inner_hidden', 32)
            return InnerNetMLPMixer(**kwargs)
        else:
            return StandardMLPMixer(**kwargs)

    def _get_loaders(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True)
        return train_loader, test_loader

    def train(self):
        train_loader, test_loader = self._get_loaders()

        # Pretrain InnerNet once
        gaussian_weights = None
        if self.is_innernet:
            logger.info("Pretraining InnerNet on Gaussian target...")
            from model.mlp_mixer import InnerNetMixerActivation
            temp_inner = InnerNetMixerActivation(
                hidden_dim=self.config.model.get('inner_hidden', 32)
            ).to(self.device)
            gaussian_weights = pretrain_inner_net_gaussian(temp_inner, self.device)
            logger.info("InnerNet pretrained.")

        seeds = list(range(42, 42 + self.num_seeds))
        all_acc_histories = []

        for si, seed in enumerate(seeds):
            logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)}) Training {self.model_name} "
                       f"for {self.epochs} epochs...")
            acc_history = self._train_single_seed(
                seed, train_loader, test_loader, gaussian_weights
            )
            all_acc_histories.append(acc_history)
            logger.info(f"[Seed {seed}] Done. Best Acc: {max(acc_history):.4f}")

        data = np.array(all_acc_histories)
        results = {
            'model_name': self.model_name,
            'seeds': seeds,
            'all_acc': all_acc_histories,
            'mean_acc': np.mean(data, axis=0).tolist(),
            'std_acc': np.std(data, axis=0).tolist(),
            'best_mean_acc': float(np.max(np.mean(data, axis=0))),
        }
        results_path = os.path.join(self.save_dir, 'mixer_results.p')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        self._mark_stage('COMPLETED')
        logger.info(f"All seeds done. Best mean acc: {results['best_mean_acc']:.4f}")

    def _train_single_seed(self, seed, train_loader, test_loader, gaussian_weights):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = self._make_model().to(self.device)

        if self.is_innernet and gaussian_weights is not None:
            model.inner_net.load_state_dict(gaussian_weights)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        acc_history = []

        for epoch in range(self.epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    out = model(imgs)
                    _, pred = torch.max(out, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

            acc = correct / total
            acc_history.append(acc)
            logger.info(f"  Seed {seed} Ep {epoch+1}/{self.epochs}: Acc = {acc:.4f}")

        return acc_history

    def test(self):
        logger.info("Mixer test: results were saved during training.")

    def _mark_stage(self, stage_name):
        marker = os.path.join(self.save_dir, stage_name)
        with open(marker, 'w') as f:
            f.write('')
