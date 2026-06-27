"""Deploy step for the non-SwiGLU case: CNN on CIFAR-10 (PROJECT_STATUS P1, case 2).

The CNN InnerNet distills to a genuinely non-SwiGLU operator (poly3 R^2=0.974,
pure SwiGLU form only 0.91 — it carries real b, b^2 terms). This deploys that
fixed closed-form operator in a fresh CNN and compares quality (test acc) and
speed (images/sec) against ReLU / SwiGLU / InnerNet.

Conditions:
  relu      standard CNN (reference floor)
  swiglu    silu(a)*b gating
  innernet  per-element MLP(a,b)  (discovery tool, slow)
  distilled fixed poly3 op distilled from the trained CNN InnerNet (fast)

Usage:
  python scripts/deploy_distilled_cnn.py --save_dir exp/deploy_cnn \
      --distill_json results/figures/distill_ws_cnn.json --distill_family poly3 \
      --ops relu,swiglu,innernet,distilled --epochs 100 --num_seeds 5
"""
import os
import sys
import json
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import _eval_distilled

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class InnerNetAct2d(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        C = x.size(1) // 2
        a, b = x[:, :C], x[:, C:]
        B, C, H, W = a.shape
        pairs = torch.stack([a, b], dim=-1)
        return self.net(pairs.reshape(-1, 2)).view(B, C, H, W)


class SwiGLUAct2d(nn.Module):
    def forward(self, x):
        C = x.size(1) // 2
        return F.silu(x[:, :C]) * x[:, C:]


class DistilledAct2d(nn.Module):
    """Fixed closed-form operator distilled from the CNN InnerNet."""
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = dict(coeffs)

    def forward(self, x):
        C = x.size(1) // 2
        return _eval_distilled(x[:, :C], x[:, C:], self.coeffs)


def make_gated_cnn(act_module, in_ch=3, channels=[60, 120, 120, 120], num_classes=10):
    """Conv -> out_ch*2 -> BN -> act (halves to out_ch) -> pool -> dropout."""
    layers, c = [], in_ch
    for out_ch in channels:
        layers += [nn.Conv2d(c, out_ch * 2, 3, padding=1), nn.BatchNorm2d(out_ch * 2),
                   act_module, nn.MaxPool2d(2), nn.Dropout(0.5)]
        c = out_ch
    return nn.Sequential(*layers), nn.Linear(channels[-1] * 2 * 2, num_classes)


def make_relu_cnn(in_ch=3, channels=[60, 120, 120, 120], num_classes=10):
    layers, c = [], in_ch
    for out_ch in channels:
        layers += [nn.Conv2d(c, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                   nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5)]
        c = out_ch
    return nn.Sequential(*layers), nn.Linear(channels[-1] * 2 * 2, num_classes)


class CNN(nn.Module):
    def __init__(self, op, coeffs=None, num_classes=10):
        super().__init__()
        if op == 'relu':
            self.features, self.fc = make_relu_cnn(num_classes=num_classes)
        elif op == 'swiglu':
            self.features, self.fc = make_gated_cnn(SwiGLUAct2d(), num_classes=num_classes)
        elif op == 'innernet':
            self.features, self.fc = make_gated_cnn(InnerNetAct2d(32), num_classes=num_classes)
        elif op == 'distilled':
            self.features, self.fc = make_gated_cnn(DistilledAct2d(coeffs), num_classes=num_classes)
        else:
            raise ValueError(op)

    def forward(self, x):
        out = self.features(x)
        return self.fc(out.view(out.size(0), -1))


def get_loaders(batch_size=100):
    tf_tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    tf_te = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    tr = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tf_tr)
    te = torchvision.datasets.CIFAR10('./data', train=False, transform=tf_te)
    return (DataLoader(tr, batch_size, shuffle=True, num_workers=4),
            DataLoader(te, batch_size, shuffle=False, num_workers=4))


def train_epoch(model, loader, opt, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def measure_throughput(model, loader, device, n_batches=30):
    model.train()
    crit = nn.CrossEntropyLoss()
    it = iter(loader)
    batches = [b for _, b in zip(range(n_batches + 5), it)]
    for x, y in batches[:5]:
        x, y = x.to(device), y.to(device)
        crit(model(x), y).backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0, imgs = time.perf_counter(), 0
    for x, y in batches[5:]:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        imgs += x.size(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return imgs / dt if dt > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='exp/deploy_cnn')
    ap.add_argument('--distill_json', default='results/figures/distill_ws_cnn.json')
    ap.add_argument('--distill_family', default='poly3')
    ap.add_argument('--ops', default='relu,swiglu,innernet,distilled')
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--num_seeds', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    with open(args.distill_json) as fh:
        dj = json.load(fh)
    fam = next(r for r in dj['results'] if r['family'] == args.distill_family)
    coeffs = fam['terms']
    logger.info(f"Distilled op ({args.distill_family}, R2={fam['r2']:.4f}): {coeffs}")

    train_loader, test_loader = get_loaders(args.batch_size)
    ops = args.ops.split(',')
    results = {op: {'acc': [], 'tput': []} for op in ops}

    for op in ops:
        for seed in range(42, 42 + args.num_seeds):
            set_seed(seed)
            model = CNN(op, coeffs=coeffs).to(device)
            nparam = sum(p.numel() for p in model.parameters())
            opt = optim.Adam(model.parameters(), lr=args.lr)
            best = 0.0
            for ep in range(args.epochs):
                train_epoch(model, train_loader, opt, device)
                acc = evaluate(model, test_loader, device)
                best = max(best, acc)
                if (ep + 1) % 10 == 0:
                    logger.info(f"[{op} seed{seed}] ep{ep+1}/{args.epochs} "
                                f"acc={acc*100:.2f}% (best={best*100:.2f}%)")
            tput = measure_throughput(model, train_loader, device)
            results[op]['acc'].append(best)
            results[op]['tput'].append(tput)
            logger.info(f"[{op} seed{seed}] DONE best={best*100:.2f}% "
                        f"tput={tput:,.0f} img/s params={nparam:,}")
            with open(os.path.join(args.save_dir, 'results.json'), 'w') as fh:
                json.dump({'args': vars(args), 'coeffs': coeffs, 'results': results}, fh, indent=2)

    logger.info("=" * 70)
    for op in ops:
        a = np.array(results[op]['acc']) * 100; t = np.array(results[op]['tput'])
        logger.info(f"{op:<10} acc={a.mean():.2f}±{a.std():.2f}%  tput={t.mean():,.0f} img/s")


if __name__ == '__main__':
    main()
