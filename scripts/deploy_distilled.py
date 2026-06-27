"""Deploy step of the discover -> distill -> deploy loop (PROJECT_STATUS P1).

Trains fresh-from-scratch Transformers on WikiText-2 under several FFN operators
and compares both quality (val PPL) and speed (tokens/sec):

  gelu      StandardTransformer  (baseline)
  swiglu    SwiGLUTransformer    (hand-designed gating)
  innernet  InnerNetTransformer  (the discovery tool — per-element inner MLP, slow)
  distilled DistilledTransformer (fixed closed-form op distilled from InnerNet)

The claim: the distilled fixed operator matches InnerNet quality while running at
SwiGLU-like speed (no inner MLP at inference). Coefficients come from a JSON
produced by scripts/distill_innernet.py.

Usage:
  python scripts/deploy_distilled.py --save_dir exp/deploy_ffn \
      --distill_json results/figures/distill_ivs_d128.json --distill_family poly3 \
      --ops gelu,swiglu,innernet,distilled --epochs 20 --num_seeds 5
"""
import os
import sys
import json
import math
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            tot += crit(model(x), y).item(); n += 1
    return math.exp(tot / n)


def train_one_epoch(model, loader, opt, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def measure_throughput(model, loader, device, n_batches=30):
    """Tokens/sec over fwd+bwd, after warmup, with cuda sync for honest timing."""
    model.train()
    crit = nn.CrossEntropyLoss()
    it = iter(loader)
    batches = []
    for _ in range(n_batches + 5):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    # warmup
    for x, y in batches[:5]:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y); loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = 0
    for x, y in batches[5:]:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = crit(model(x), y); loss.backward()
        tokens += x.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tokens / dt if dt > 0 else float('nan')


def build_model(op, vocab_size, args, coeffs):
    from model.transformer import (StandardTransformer, SwiGLUTransformer,
                                   InnerNetTransformer, DistilledTransformer)
    kw = dict(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
              d_ff=args.d_ff, n_layers=args.n_layers, max_len=args.context_size,
              dropout=0.1)
    if op == 'gelu':
        return StandardTransformer(**kw)
    if op == 'swiglu':
        return SwiGLUTransformer(**kw)
    if op == 'innernet':
        return InnerNetTransformer(inner_hidden=32, **kw)
    if op == 'distilled':
        return DistilledTransformer(coeffs=coeffs, **kw)
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='exp/deploy_ffn')
    ap.add_argument('--distill_json', default='results/figures/distill_ivs_d128.json')
    ap.add_argument('--distill_family', default='poly3')
    ap.add_argument('--ops', default='gelu,swiglu,innernet,distilled')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--d_ff', type=int, default=512)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--context_size', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--num_seeds', type=int, default=5)
    ap.add_argument('--dataset', default='wikitext')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load distilled coefficients
    with open(args.distill_json) as fh:
        dj = json.load(fh)
    fam = next(r for r in dj['results'] if r['family'] == args.distill_family)
    coeffs = fam['terms']
    logger.info(f"Distilled op ({args.distill_family}, R2={fam['r2']:.4f}) "
                f"from {args.distill_json}: {coeffs}")

    from runner.lm_runner import WikiTextDataset
    train_ds = WikiTextDataset(split='train', context_size=args.context_size,
                               dataset_name=args.dataset)
    val_ds = WikiTextDataset(split='validation', context_size=args.context_size,
                             vocab=train_ds.vocab, dataset_name=args.dataset)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    logger.info(f"vocab={vocab_size}, train={len(train_ds)}, val={len(val_ds)}")

    ops = args.ops.split(',')
    results = {op: {'ppl': [], 'tput': []} for op in ops}

    for op in ops:
        for seed in range(42, 42 + args.num_seeds):
            set_seed(seed)
            model = build_model(op, vocab_size, args, coeffs).to(device)
            nparam = sum(p.numel() for p in model.parameters())
            opt = optim.Adam(model.parameters(), lr=args.lr)
            best = float('inf')
            for ep in range(args.epochs):
                train_one_epoch(model, train_loader, opt, device)
                ppl = evaluate(model, val_loader, device)
                best = min(best, ppl)
                logger.info(f"[{op} seed{seed}] ep{ep+1}/{args.epochs} "
                            f"PPL={ppl:.2f} (best={best:.2f})")
            tput = measure_throughput(model, train_loader, device)
            results[op]['ppl'].append(best)
            results[op]['tput'].append(tput)
            logger.info(f"[{op} seed{seed}] DONE best PPL={best:.2f} "
                        f"tput={tput:,.0f} tok/s params={nparam:,}")
            with open(os.path.join(args.save_dir, 'results.json'), 'w') as fh:
                json.dump({'args': vars(args), 'coeffs': coeffs,
                           'results': results}, fh, indent=2)

    logger.info("=" * 70)
    logger.info(f"{'op':<10} {'PPL (mean±std)':>20} {'tok/s (mean)':>16}")
    for op in ops:
        p = np.array(results[op]['ppl']); t = np.array(results[op]['tput'])
        logger.info(f"{op:<10} {p.mean():>10.2f} ± {p.std():<6.2f} "
                    f"{t.mean():>15,.0f}")


if __name__ == '__main__':
    main()
