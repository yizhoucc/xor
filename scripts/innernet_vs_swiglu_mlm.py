"""InnerNet vs SwiGLU warm-start for MLM (BERT-style).

Same design as innernet_vs_swiglu.py but for masked language modeling.
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
        loss = criterion(inner_net(inputs), targets)
        loss.backward()
        optimizer.step()
    logger.info(f"  Fitted InnerNet to SwiGLU, MSE={loss.item():.6f}")


def evaluate_mlm(model, val_loader, device, mask_token_id, mask_prob=0.15):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_masked = 0, 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            masked_x = x.clone()
            labels = x.clone()
            mask = torch.rand(x.shape, device=device) < mask_prob
            labels[~mask] = -100
            masked_x[mask] = mask_token_id
            logits = model(masked_x)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = (labels != -100).sum().item()
            total_loss += loss.item() * n
            total_masked += n
    return math.exp(min(total_loss / max(total_masked, 1), 20))


def train_mlm_one_epoch(model, train_loader, optimizer, device, mask_token_id, mask_prob=0.15):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    for x in train_loader:
        x = x.to(device)
        masked_x = x.clone()
        labels = x.clone()
        mask = torch.rand(x.shape, device=device) < mask_prob
        labels[~mask] = -100
        replace = mask & (torch.rand(x.shape, device=device) < 0.8)
        masked_x[replace] = mask_token_id
        rand_mask = mask & ~replace & (torch.rand(x.shape, device=device) < 0.5)
        masked_x[rand_mask] = torch.randint(0, mask_token_id, masked_x[rand_mask].shape, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(masked_x)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='exp/ivs_mlm')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--context_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.mlm_runner import MLMDataset, MLMWrapper
    from model.transformer import SwiGLUTransformer, InnerNetTransformer, InnerNetFFNActivation

    train_ds = MLMDataset(split='train', seq_len=args.context_size)
    val_ds = MLMDataset(split='validation', seq_len=args.context_size, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    mask_token_id = train_ds.mask_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

        # SwiGLU 10ep
        logger.info("--- SwiGLU Phase 1: 10 epochs ---")
        swiglu_base = SwiGLUTransformer(vocab_size, args.d_model, args.n_heads, args.d_ff,
                                         args.n_layers, args.context_size, 0.1).to(device)
        swiglu_model = MLMWrapper(swiglu_base)
        opt_sw = optim.Adam(swiglu_model.parameters(), lr=args.lr)
        sw_ppl = []
        for ep in range(10):
            train_mlm_one_epoch(swiglu_model, train_loader, opt_sw, device, mask_token_id)
            ppl = evaluate_mlm(swiglu_model, val_loader, device, mask_token_id)
            sw_ppl.append(ppl)
            save_ckpt(swiglu_model, 'ivs_mlm', 'swiglu_phase1', seed, ep+1)
            logger.info(f"  SwiGLU Ep {ep+1}/10: PPL={ppl:.2f}")

        swiglu_state = swiglu_base.state_dict()
        opt_state = copy.deepcopy(opt_sw.state_dict())

        # Branch A: SwiGLU continues 10ep
        logger.info("--- Branch A: SwiGLU continues ---")
        sw_ppl2 = []
        for ep in range(10):
            train_mlm_one_epoch(swiglu_model, train_loader, opt_sw, device, mask_token_id)
            ppl = evaluate_mlm(swiglu_model, val_loader, device, mask_token_id)
            sw_ppl2.append(ppl)
            save_ckpt(swiglu_model, 'ivs_mlm', 'swiglu_phase2', seed, ep+1)
            logger.info(f"  SwiGLU Ep {ep+11}/20: PPL={ppl:.2f}")
        best_swiglu = min(sw_ppl + sw_ppl2)

        # Branch B: InnerNet replaces, continues 10ep
        logger.info("--- Branch B: InnerNet replaces ---")
        shared_inner = InnerNetFFNActivation(hidden_dim=32).to(device)
        shared_inner.load_state_dict(fitted_weights)
        innernet_base = InnerNetTransformer(vocab_size, args.d_model, args.n_heads, args.d_ff,
                                             args.n_layers, args.context_size, 32, 0.1).to(device)
        inn_dict = innernet_base.state_dict()
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v
        innernet_base.load_state_dict(inn_dict, strict=False)
        for block in innernet_base.blocks:
            block.ffn.inner_net.load_state_dict(fitted_weights)
        innernet_model = MLMWrapper(innernet_base)

        ppl_swap = evaluate_mlm(innernet_model, val_loader, device, mask_token_id)
        logger.info(f"  After swap: PPL={ppl_swap:.2f}")

        opt_in = optim.Adam(innernet_model.parameters(), lr=args.lr)
        in_ppl = []
        for ep in range(10):
            train_mlm_one_epoch(innernet_model, train_loader, opt_in, device, mask_token_id)
            ppl = evaluate_mlm(innernet_model, val_loader, device, mask_token_id)
            in_ppl.append(ppl)
            save_ckpt(innernet_model, 'ivs_mlm', 'innernet', seed, ep+1)
            logger.info(f"  InnerNet Ep {ep+11}/20: PPL={ppl:.2f}")
        best_innernet = min(in_ppl)
        try:
            torch.save(shared_inner.state_dict(), os.path.join(args.save_dir, f"inner_weights_seed{seed}.pth"))
            logger.info(f"  Saved InnerNet weights")
        except: pass

        logger.info(f"  RESULT: SwiGLU={best_swiglu:.2f} vs InnerNet={best_innernet:.2f}")

        all_results.append({
            'seed': seed, 'best_swiglu': best_swiglu, 'best_innernet': best_innernet,
            'ppl_swap': ppl_swap, 'sw_ppl': sw_ppl + sw_ppl2, 'in_ppl': in_ppl,
        })

    sw_bests = [r['best_swiglu'] for r in all_results]
    in_bests = [r['best_innernet'] for r in all_results]
    logger.info(f"\nSUMMARY ({len(seeds)} seeds)")
    logger.info(f"  SwiGLU:   {np.mean(sw_bests):.2f} ± {np.std(sw_bests):.2f}")
    logger.info(f"  InnerNet: {np.mean(in_bests):.2f} ± {np.std(in_bests):.2f}")
    logger.info(f"  Diff:     {np.mean(sw_bests) - np.mean(in_bests):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
