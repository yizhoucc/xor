"""InnerNet vs SwiGLU: two clean experiments.

Experiment 1 (Fair comparison):
  - SwiGLU trains 10ep (makes sense)
  - Fit InnerNet to SwiGLU, replace
  - Fork: SwiGLU continues 10ep vs InnerNet (full unfreeze) continues 10ep
  - Compare at total 20ep each

Experiment 2 (InnerNet ceiling):
  - SwiGLU trains 20ep (converged), then stops
  - Fit InnerNet to SwiGLU, replace
  - Freeze network, InnerNet NON-SHARED, train to convergence
  - If InnerNet wins → its ceiling is higher than SwiGLU

Usage:
  python scripts/innernet_vs_swiglu.py --save_dir exp/inner_vs_swiglu
"""
import os
import sys
import math
import copy
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
    optimizer = optim.Adam(inner_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (torch.nn.functional.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = criterion(inner_net(inputs), targets)
        loss.backward()
        optimizer.step()
    logger.info(f"  Fitted InnerNet to SwiGLU, MSE={loss.item():.6f}")


def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
            n += 1
    return math.exp(total_loss / n)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def build_innernet_from_swiglu(swiglu_state, fitted_weights, vocab_size,
                                d_model, n_heads, d_ff, n_layers, context_size,
                                device, shared=True):
    from model.transformer import InnerNetTransformer, InnerNetFFNActivation

    model = InnerNetTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, context_size, 32, 0.1
    ).to(device)

    # Copy all matching weights from SwiGLU
    model_dict = model.state_dict()
    copied = 0
    for k, v in swiglu_state.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
            copied += 1
    model.load_state_dict(model_dict, strict=False)
    logger.info(f"  Copied {copied} weight tensors from SwiGLU")

    if shared:
        # All layers already share the same inner_net, just load weights
        for block in model.blocks:
            block.ffn.inner_net.load_state_dict(fitted_weights)
    else:
        # Give each layer its own independent InnerNet
        for block in model.blocks:
            own_inner = InnerNetFFNActivation(hidden_dim=32).to(device)
            own_inner.load_state_dict(fitted_weights)
            block.ffn.inner_net = own_inner

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='exp/inner_vs_swiglu')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--context_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='wikitext')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    from model.transformer import SwiGLUTransformer, InnerNetFFNActivation

    train_ds = WikiTextDataset(split='train', context_size=args.context_size, dataset_name=args.dataset)
    val_ds = WikiTextDataset(split='validation', context_size=args.context_size, vocab=train_ds.vocab, dataset_name=args.dataset)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Fit InnerNet to SwiGLU once
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

        # ============================================================
        # EXPERIMENT 1: Fair comparison
        # ============================================================
        logger.info("========== EXPERIMENT 1: Fair Comparison ==========")

        # Phase 1: SwiGLU trains 10ep
        logger.info("--- SwiGLU Phase 1: 10 epochs ---")
        swiglu = SwiGLUTransformer(
            vocab_size, args.d_model, args.n_heads, args.d_ff,
            args.n_layers, args.context_size, 0.1
        ).to(device)
        opt_sw = optim.Adam(swiglu.parameters(), lr=args.lr)
        sw_ppl_phase1 = []
        for ep in range(10):
            train_one_epoch(swiglu, train_loader, opt_sw, device)
            ppl = evaluate(swiglu, val_loader, device)
            sw_ppl_phase1.append(ppl)
            logger.info(f"  SwiGLU Ep {ep+1}/10: PPL={ppl:.2f}")

        # Save state at ep 10 for forking
        swiglu_state_ep10 = copy.deepcopy(swiglu.state_dict())
        opt_state_ep10 = copy.deepcopy(opt_sw.state_dict())

        # Branch A: SwiGLU continues 10ep
        logger.info("--- Branch A: SwiGLU continues 10 more epochs ---")
        sw_ppl_phase2 = []
        for ep in range(10):
            train_one_epoch(swiglu, train_loader, opt_sw, device)
            ppl = evaluate(swiglu, val_loader, device)
            sw_ppl_phase2.append(ppl)
            logger.info(f"  SwiGLU Ep {ep+11}/20: PPL={ppl:.2f}")
        best_swiglu_20 = min(sw_ppl_phase1 + sw_ppl_phase2)

        # Branch B: InnerNet replaces, continues 10ep (full unfreeze)
        logger.info("--- Branch B: InnerNet replaces, continues 10 epochs ---")
        innernet_model = build_innernet_from_swiglu(
            swiglu_state_ep10, fitted_weights, vocab_size,
            args.d_model, args.n_heads, args.d_ff, args.n_layers,
            args.context_size, device, shared=True
        )
        ppl_swap = evaluate(innernet_model, val_loader, device)
        logger.info(f"  After swap: PPL={ppl_swap:.2f}")

        opt_in = optim.Adam(innernet_model.parameters(), lr=args.lr)
        in_ppl_phase2 = []
        for ep in range(10):
            train_one_epoch(innernet_model, train_loader, opt_in, device)
            ppl = evaluate(innernet_model, val_loader, device)
            in_ppl_phase2.append(ppl)
            logger.info(f"  InnerNet Ep {ep+11}/20: PPL={ppl:.2f}")
        best_innernet_20 = min(in_ppl_phase2)

        logger.info(f"  EXP1 RESULT: SwiGLU@20ep={best_swiglu_20:.2f} vs InnerNet@20ep={best_innernet_20:.2f}")

        # ============================================================
        # EXPERIMENT 2: InnerNet ceiling (frozen, non-shared)
        # ============================================================
        logger.info("========== EXPERIMENT 2: InnerNet Ceiling ==========")

        # SwiGLU already trained 20ep from Exp1, use its final state
        swiglu_final_ppl = sw_ppl_phase2[-1]
        best_swiglu_final = best_swiglu_20
        swiglu_state_final = swiglu.state_dict()

        # Build non-shared InnerNet from converged SwiGLU
        logger.info("--- Frozen network, non-shared InnerNet ---")
        frozen_model = build_innernet_from_swiglu(
            swiglu_state_final, fitted_weights, vocab_size,
            args.d_model, args.n_heads, args.d_ff, args.n_layers,
            args.context_size, device, shared=False
        )

        # Freeze everything except inner_net
        for name, param in frozen_model.named_parameters():
            if 'inner_net' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
        logger.info(f"  Trainable: {trainable} params (non-shared InnerNet)")

        ppl_swap2 = evaluate(frozen_model, val_loader, device)
        logger.info(f"  After swap: PPL={ppl_swap2:.2f}")

        opt_frozen = optim.Adam(filter(lambda p: p.requires_grad, frozen_model.parameters()), lr=args.lr)
        frozen_ppl = []
        best_frozen = float('inf')
        no_improve = 0
        for ep in range(50):
            train_one_epoch(frozen_model, train_loader, opt_frozen, device)
            ppl = evaluate(frozen_model, val_loader, device)
            frozen_ppl.append(ppl)
            if ppl < best_frozen:
                best_frozen = ppl
                no_improve = 0
            else:
                no_improve += 1
            logger.info(f"  Frozen Ep {ep+1}: PPL={ppl:.2f} (best={best_frozen:.2f}, patience={no_improve}/10)")
            if no_improve >= 10:
                logger.info(f"  Early stopping at epoch {ep+1}")
                break

        logger.info(f"  EXP2 RESULT: SwiGLU@20ep={best_swiglu_final:.2f} vs Frozen InnerNet={best_frozen:.2f}")

        all_results.append({
            'seed': seed,
            # Exp 1
            'sw_ppl_phase1': sw_ppl_phase1,
            'sw_ppl_phase2': sw_ppl_phase2,
            'best_swiglu_20': best_swiglu_20,
            'ppl_swap': ppl_swap,
            'in_ppl_phase2': in_ppl_phase2,
            'best_innernet_20': best_innernet_20,
            # Exp 2
            'best_swiglu_final': best_swiglu_final,
            'frozen_ppl': frozen_ppl,
            'best_frozen': best_frozen,
        })

    # Summary
    sw20 = [r['best_swiglu_20'] for r in all_results]
    in20 = [r['best_innernet_20'] for r in all_results]
    frz = [r['best_frozen'] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY ({len(seeds)} seeds)")
    logger.info(f"")
    logger.info(f"EXP 1 (Fair, both 20ep total):")
    logger.info(f"  SwiGLU 20ep:     {np.mean(sw20):.2f} ± {np.std(sw20):.2f}")
    logger.info(f"  InnerNet 20ep:   {np.mean(in20):.2f} ± {np.std(in20):.2f}")
    logger.info(f"  Diff:            {np.mean(sw20) - np.mean(in20):.2f}")
    logger.info(f"")
    logger.info(f"EXP 2 (Ceiling, frozen non-shared):")
    logger.info(f"  SwiGLU final:    {np.mean(sw20):.2f} ± {np.std(sw20):.2f}")
    logger.info(f"  Frozen InnerNet: {np.mean(frz):.2f} ± {np.std(frz):.2f}")
    logger.info(f"  Diff:            {np.mean(sw20) - np.mean(frz):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results,
                     'exp1_swiglu': float(np.mean(sw20)), 'exp1_innernet': float(np.mean(in20)),
                     'exp2_swiglu': float(np.mean(sw20)), 'exp2_frozen': float(np.mean(frz))}, f)
    logger.info(f"Saved to {args.save_dir}")


if __name__ == '__main__':
    main()
