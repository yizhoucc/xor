"""Frozen network experiment: find InnerNet's ceiling.

Freeze the entire network (from trained SwiGLU), only train InnerNet.
Compare shared (1 InnerNet for all layers) vs non-shared (each layer gets its own).
Train until convergence (early stopping).

Usage:
  python scripts/frozen_innernet.py --save_dir exp/frozen_innernet
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

CKPT_DIR = '/user_data/yizhouc3/xor_checkpoints'

def save_ckpt(model, exp_name, cond, seed, epoch):
    d = os.path.join(CKPT_DIR, exp_name, f'{cond}_seed{seed}')
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, f'ep{epoch:03d}.pth'))



def fit_innernet_to_swiglu(inner_net, device, num_steps=2000, lr=1e-3):
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
    logger.info(f"  Fitted InnerNet to SwiGLU, MSE={loss.item():.6f}")


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


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def make_innernet_from_swiglu(swiglu_state, vocab_size, d_model, n_heads, d_ff,
                               n_layers, context_size, fitted_weights, device, shared=True):
    """Build InnerNet Transformer and load SwiGLU weights."""
    from model.transformer import InnerNetTransformer, InnerNetFFN, InnerNetFFNActivation, TransformerBlock, MultiHeadAttention, PositionalEncoding

    if shared:
        # Standard shared InnerNet
        model = InnerNetTransformer(
            vocab_size, d_model, n_heads, d_ff,
            n_layers, context_size, 32, 0.1
        ).to(device)
        # Load SwiGLU weights
        model_dict = model.state_dict()
        for k, v in swiglu_state.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
        # Load fitted InnerNet
        for block in model.blocks:
            block.ffn.inner_net.load_state_dict(fitted_weights)
    else:
        # Non-shared: each layer gets its own InnerNet
        model = InnerNetTransformer(
            vocab_size, d_model, n_heads, d_ff,
            n_layers, context_size, 32, 0.1
        ).to(device)
        # Load SwiGLU weights
        model_dict = model.state_dict()
        for k, v in swiglu_state.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
        # Give each layer its own InnerNet (break sharing)
        for block in model.blocks:
            own_inner = InnerNetFFNActivation(hidden_dim=32).to(device)
            own_inner.load_state_dict(fitted_weights)
            block.ffn.inner_net = own_inner

    return model


def run_frozen(model, train_loader, val_loader, device, max_epochs=50, patience=10, lr=5e-4, cond='frozen', seed=0):
    """Freeze network, train only InnerNet until convergence."""
    # Freeze everything except inner_net
    for name, param in model.named_parameters():
        if 'inner_net' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable} / {total} params")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    ppl_history = []
    best_ppl = float('inf')
    no_improve = 0

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        ppl = evaluate(model, val_loader, device)
        ppl_history.append(ppl)
        save_ckpt(model, os.path.basename(args.save_dir), cond, seed, epoch+1)

        if ppl < best_ppl:
            best_ppl = ppl
            no_improve = 0
        else:
            no_improve += 1

        logger.info(f"  Ep {epoch+1}/{max_epochs}: PPL={ppl:.2f} (best={best_ppl:.2f}, patience={no_improve}/{patience})")

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    return ppl_history, best_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='exp/frozen_innernet')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--context_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--swiglu_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--frozen_lr', type=float, default=5e-4)
    parser.add_argument('--max_frozen_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='wikitext')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    train_ds = WikiTextDataset(split='train', context_size=args.context_size, dataset_name=args.dataset)
    val_ds = WikiTextDataset(split='validation', context_size=args.context_size, vocab=train_ds.vocab, dataset_name=args.dataset)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    from model.transformer import SwiGLUTransformer, InnerNetFFNActivation

    # Fit InnerNet to SwiGLU
    logger.info("=== Fitting InnerNet to SiLU(a)*b ===")
    inner_template = InnerNetFFNActivation(hidden_dim=32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.state_dict()

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

        # Train SwiGLU
        logger.info("--- Train SwiGLU ---")
        swiglu = SwiGLUTransformer(vocab_size, args.d_model, args.n_heads, args.d_ff,
                                    args.n_layers, args.context_size, 0.1).to(device)
        opt = optim.Adam(swiglu.parameters(), lr=args.lr)
        swiglu_ppl = []
        for ep in range(args.swiglu_epochs):
            train_one_epoch(swiglu, train_loader, opt, device)
            ppl = evaluate(swiglu, val_loader, device)
            swiglu_ppl.append(ppl)
            save_ckpt(swiglu, os.path.basename(args.save_dir), 'swiglu', seed, ep+1)
            logger.info(f"  SwiGLU Ep {ep+1}/{args.swiglu_epochs}: PPL={ppl:.2f}")
        best_swiglu = min(swiglu_ppl)
        swiglu_state = swiglu.state_dict()

        # Shared frozen
        logger.info("--- Frozen SHARED InnerNet ---")
        model_shared = make_innernet_from_swiglu(
            swiglu_state, vocab_size, args.d_model, args.n_heads, args.d_ff,
            args.n_layers, args.context_size, fitted_weights, device, shared=True)
        ppl_swap = evaluate(model_shared, val_loader, device)
        logger.info(f"  After swap: PPL={ppl_swap:.2f}")
        shared_history, best_shared = run_frozen(
            model_shared, train_loader, val_loader, device,
            args.max_frozen_epochs, args.patience, args.frozen_lr,
            cond='frozen_shared', seed=seed)

        # Non-shared frozen
        logger.info("--- Frozen NON-SHARED InnerNet ---")
        model_nonshared = make_innernet_from_swiglu(
            swiglu_state, vocab_size, args.d_model, args.n_heads, args.d_ff,
            args.n_layers, args.context_size, fitted_weights, device, shared=False)
        nonshared_history, best_nonshared = run_frozen(
            model_nonshared, train_loader, val_loader, device,
            args.max_frozen_epochs, args.patience, args.frozen_lr,
            cond='frozen_nonshared', seed=seed)

        logger.info(f"  SwiGLU: {best_swiglu:.2f} | Shared: {best_shared:.2f} | Non-shared: {best_nonshared:.2f}")

        all_results.append({
            'seed': seed,
            'best_swiglu': best_swiglu,
            'ppl_swap': ppl_swap,
            'shared_history': shared_history,
            'best_shared': best_shared,
            'nonshared_history': nonshared_history,
            'best_nonshared': best_nonshared,
        })

    # Summary
    sw = [r['best_swiglu'] for r in all_results]
    sh = [r['best_shared'] for r in all_results]
    ns = [r['best_nonshared'] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY ({len(seeds)} seeds)")
    logger.info(f"  SwiGLU:           {np.mean(sw):.2f} ± {np.std(sw):.2f}")
    logger.info(f"  Frozen shared:    {np.mean(sh):.2f} ± {np.std(sh):.2f} ({sum(p.numel() for p in inner_template.parameters())} params)")
    logger.info(f"  Frozen non-shared:{np.mean(ns):.2f} ± {np.std(ns):.2f} ({sum(p.numel() for p in inner_template.parameters()) * args.n_layers} params)")

    with open(os.path.join(args.save_dir, 'frozen_results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results, 'mean_swiglu': float(np.mean(sw)),
                     'mean_shared': float(np.mean(sh)), 'mean_nonshared': float(np.mean(ns))}, f)
    logger.info(f"Saved to {args.save_dir}")


if __name__ == '__main__':
    main()
