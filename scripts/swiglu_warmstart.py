"""SwiGLU warm-start experiment for InnerNet.

Self-contained: trains SwiGLU, fits InnerNet to SiLU(a)*b, copies weights, finetunes.
Tests if InnerNet can surpass SwiGLU when starting from SwiGLU's solution.

Usage:
  python scripts/swiglu_warmstart.py --save_dir exp/warmstart_test
  python scripts/swiglu_warmstart.py --d_model 128 --epochs 10 --finetune_epochs 5
"""
import os
import sys
import math
import pickle
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


def fit_innernet_to_swiglu(inner_net, device, num_steps=2000, lr=1e-3):
    """Train InnerNet to match SiLU(a)*b on a grid."""
    optimizer = optim.Adam(inner_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (torch.nn.functional.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)

    for step in range(num_steps):
        optimizer.zero_grad()
        pred = inner_net(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            logger.info(f"  Fitting InnerNet→SwiGLU: step {step+1}/{num_steps}, MSE={loss.item():.6f}")

    logger.info(f"  Final fitting MSE={loss.item():.6f}")
    return inner_net


def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            n += 1
    return math.exp(total_loss / n)


def train_one_epoch(model, train_loader, optimizer, device, grad_clip=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='exp/warmstart_innernet')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--context_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10, help='SwiGLU training epochs')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='wikitext')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
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

    from model.transformer import (SwiGLUTransformer, InnerNetTransformer,
                                    InnerNetFFNActivation)

    # Fit InnerNet to SwiGLU function (once, reuse for all seeds)
    logger.info("=== Fitting InnerNet to SiLU(a)*b ===")
    inner_template = InnerNetFFNActivation(hidden_dim=32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_inner_weights = inner_template.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)})")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Phase 1: Train SwiGLU
        logger.info("--- Phase 1: Train SwiGLU ---")
        swiglu_model = SwiGLUTransformer(
            vocab_size, args.d_model, args.n_heads, args.d_ff,
            args.n_layers, args.context_size, args.lr
        ).to(device)
        optimizer = optim.Adam(swiglu_model.parameters(), lr=args.lr)

        swiglu_ppl = []
        for epoch in range(args.epochs):
            train_one_epoch(swiglu_model, train_loader, optimizer, device)
            ppl = evaluate(swiglu_model, val_loader, device)
            swiglu_ppl.append(ppl)
            logger.info(f"  SwiGLU Ep {epoch+1}/{args.epochs}: PPL={ppl:.2f}")

        swiglu_state = swiglu_model.state_dict()
        best_swiglu = min(swiglu_ppl)
        logger.info(f"  SwiGLU best: {best_swiglu:.2f}")

        # Phase 2: Build InnerNet, copy weights, replace activation
        logger.info("--- Phase 2: Copy weights to InnerNet ---")

        # Reset seed for fair comparison
        torch.manual_seed(seed)
        shared_inner = InnerNetFFNActivation(hidden_dim=32).to(device)
        shared_inner.load_state_dict(fitted_inner_weights)

        innernet_model = InnerNetTransformer(
            vocab_size, args.d_model, args.n_heads, args.d_ff,
            args.n_layers, args.context_size, 32, args.lr
        ).to(device)

        # Copy matching weights from SwiGLU
        inn_dict = innernet_model.state_dict()
        copied = 0
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v
                copied += 1
        innernet_model.load_state_dict(inn_dict, strict=False)
        logger.info(f"  Copied {copied} weight tensors from SwiGLU")

        # Set fitted InnerNet weights
        for block in innernet_model.blocks:
            block.ffn.inner_net.load_state_dict(fitted_inner_weights)

        # Eval after swap (should match SwiGLU)
        ppl_after_swap = evaluate(innernet_model, val_loader, device)
        logger.info(f"  PPL after swap: {ppl_after_swap:.2f} (SwiGLU was {swiglu_ppl[-1]:.2f})")

        # Phase 3: Finetune InnerNet
        logger.info("--- Phase 3: Finetune InnerNet ---")
        optimizer = optim.Adam(innernet_model.parameters(), lr=args.finetune_lr)

        innernet_ppl = [ppl_after_swap]
        for epoch in range(args.finetune_epochs):
            train_one_epoch(innernet_model, train_loader, optimizer, device)
            ppl = evaluate(innernet_model, val_loader, device)
            innernet_ppl.append(ppl)
            logger.info(f"  Finetune Ep {epoch+1}/{args.finetune_epochs}: PPL={ppl:.2f}")

        best_innernet = min(innernet_ppl)
        logger.info(f"  InnerNet best: {best_innernet:.2f} (SwiGLU was {best_swiglu:.2f})")

        all_results.append({
            'seed': seed,
            'swiglu_ppl': swiglu_ppl,
            'ppl_after_swap': ppl_after_swap,
            'innernet_ppl': innernet_ppl,
            'best_swiglu': best_swiglu,
            'best_innernet': best_innernet,
        })

    # Summary
    swiglu_bests = [r['best_swiglu'] for r in all_results]
    innernet_bests = [r['best_innernet'] for r in all_results]
    swap_ppls = [r['ppl_after_swap'] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY ({len(seeds)} seeds)")
    logger.info(f"  SwiGLU best:      {np.mean(swiglu_bests):.2f} ± {np.std(swiglu_bests):.2f}")
    logger.info(f"  After swap:       {np.mean(swap_ppls):.2f} ± {np.std(swap_ppls):.2f}")
    logger.info(f"  InnerNet best:    {np.mean(innernet_bests):.2f} ± {np.std(innernet_bests):.2f}")
    logger.info(f"  Improvement:      {np.mean(swiglu_bests) - np.mean(innernet_bests):.2f}")

    results = {
        'all_results': all_results,
        'mean_swiglu': float(np.mean(swiglu_bests)),
        'mean_innernet': float(np.mean(innernet_bests)),
        'mean_swap': float(np.mean(swap_ppls)),
    }
    with open(os.path.join(args.save_dir, 'warmstart_results.p'), 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved to {args.save_dir}/warmstart_results.p")


if __name__ == '__main__':
    main()
