"""Non-shared InnerNet warm-start on best 3 tasks.

Each layer gets its own InnerNet (not shared). Compare with shared results.
Tests: MLM, CNN CIFAR-10, PTB (d=128).
"""
import os, sys, math, copy, pickle, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class InnerNetAct(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x)


def fit_innernet_to_swiglu(inner_net, device, steps=2000):
    opt = optim.Adam(inner_net.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (F.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet, MSE={loss.item():.6f}")


def build_nonshared_innernet_transformer(swiglu_state, fitted_weights, vocab_size,
                                          d_model, n_heads, d_ff, n_layers,
                                          context_size, device):
    """Build InnerNet Transformer with NON-SHARED InnerNet (each layer independent)."""
    from model.transformer import (InnerNetTransformer, InnerNetFFN,
                                    InnerNetFFNActivation, TransformerBlock,
                                    MultiHeadAttention, PositionalEncoding)

    # Build standard InnerNetTransformer first (shared)
    model = InnerNetTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, context_size, 32, 0.1
    ).to(device)

    # Copy SwiGLU weights
    model_dict = model.state_dict()
    for k, v in swiglu_state.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)

    # Break sharing: give each layer its own InnerNet
    for block in model.blocks:
        own_inner = InnerNetFFNActivation(hidden_dim=32).to(device)
        own_inner.load_state_dict(fitted_weights)
        block.ffn.inner_net = own_inner

    return model


# ============ Task runners ============

def run_transformer_task(dataset_name, d_model, n_heads, d_ff, n_layers,
                          context_size, fitted_weights, device, args):
    """Run warm-start for Transformer LM task (WikiText-2 or PTB)."""
    from runner.lm_runner import WikiTextDataset
    from model.transformer import SwiGLUTransformer

    train_ds = WikiTextDataset(split='train', context_size=context_size, dataset_name=dataset_name)
    val_ds = WikiTextDataset(split='validation', context_size=context_size,
                              vocab=train_ds.vocab, dataset_name=dataset_name)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    def evaluate_ppl(model):
        model.eval()
        total, n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total += nn.CrossEntropyLoss()(model(x), y).item()
                n += 1
        return math.exp(total / n)

    def train_ep(model, opt):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    results = []
    for si, seed in enumerate(range(42, 42 + args.num_seeds)):
        logger.info(f"\n[Seed {seed}] ({si+1}/{args.num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # SwiGLU 10ep
        sw = SwiGLUTransformer(vocab_size, d_model, n_heads, d_ff, n_layers, context_size, 0.1).to(device)
        opt_sw = optim.Adam(sw.parameters(), lr=5e-4)
        for ep in range(10):
            train_ep(sw, opt_sw)
            ppl = evaluate_ppl(sw)
            save_ckpt(sw, os.path.basename(args.save_dir), f'swiglu_phase1_{dataset_name}', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+1}: PPL={ppl:.2f}")
        sw_state = copy.deepcopy(sw.state_dict())

        # SwiGLU continues 10ep
        sw_ppl = []
        for ep in range(10):
            train_ep(sw, opt_sw)
            ppl = evaluate_ppl(sw)
            sw_ppl.append(ppl)
            save_ckpt(sw, os.path.basename(args.save_dir), f'swiglu_phase2_{dataset_name}', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+11}: PPL={ppl:.2f}")
        best_sw = min(sw_ppl)

        # Non-shared InnerNet
        in_model = build_nonshared_innernet_transformer(
            sw_state, fitted_weights, vocab_size,
            d_model, n_heads, d_ff, n_layers, context_size, device)
        opt_in = optim.Adam(in_model.parameters(), lr=5e-4)
        in_ppl = []
        for ep in range(10):
            train_ep(in_model, opt_in)
            ppl = evaluate_ppl(in_model)
            in_ppl.append(ppl)
            save_ckpt(in_model, os.path.basename(args.save_dir), f'nonshared_{dataset_name}', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  NonShared Ep {ep+11}: PPL={ppl:.2f}")
        best_in = min(in_ppl)

        logger.info(f"  RESULT: SwiGLU={best_sw:.2f} vs NonShared={best_in:.2f}")
        results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

        # Save per-layer weights
        for i, block in enumerate(in_model.blocks):
            torch.save(block.ffn.inner_net.state_dict(),
                       os.path.join(args.save_dir, f'inner_layer{i}_seed{seed}.pth'))

    return results


def run_mlm_task(fitted_weights, device, args):
    """Run warm-start for MLM task."""
    from runner.mlm_runner import MLMDataset, MLMWrapper
    from model.transformer import SwiGLUTransformer, InnerNetFFNActivation

    train_ds = MLMDataset(split='train', seq_len=64)
    val_ds = MLMDataset(split='validation', seq_len=64, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    mask_id = train_ds.mask_token_id
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    def mask_tokens(x, mask_id, p=0.15):
        labels = x.clone()
        mask = torch.rand(x.shape, device=x.device) < p
        labels[~mask] = -100
        x_masked = x.clone()
        x_masked[mask & (torch.rand(x.shape, device=x.device) < 0.8)] = mask_id
        return x_masked, labels

    def eval_mlm(model):
        model.eval()
        total, n_masked = 0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                mx, labels = mask_tokens(x, mask_id)
                logits = model(mx)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
                nm = (labels != -100).sum().item()
                total += loss.item() * nm
                n_masked += nm
        return math.exp(min(total / max(n_masked, 1), 20))

    def train_mlm(model, opt):
        model.train()
        for x in train_loader:
            x = x.to(device)
            mx, labels = mask_tokens(x, mask_id)
            opt.zero_grad()
            logits = model(mx)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    results = []
    for si, seed in enumerate(range(42, 42 + args.num_seeds)):
        logger.info(f"\n[Seed {seed}] ({si+1}/{args.num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sw_base = SwiGLUTransformer(vocab_size, 128, 4, 512, 4, 64, 0.1).to(device)
        sw_model = MLMWrapper(sw_base)
        opt_sw = optim.Adam(sw_model.parameters(), lr=5e-4)
        for ep in range(10):
            train_mlm(sw_model, opt_sw)
            ppl = eval_mlm(sw_model)
            save_ckpt(sw_model, os.path.basename(args.save_dir), 'swiglu_phase1_mlm', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+1}: PPL={ppl:.2f}")
        sw_state = sw_base.state_dict()

        sw_ppl2 = []
        for ep in range(10):
            train_mlm(sw_model, opt_sw)
            ppl = eval_mlm(sw_model)
            sw_ppl2.append(ppl)
            save_ckpt(sw_model, os.path.basename(args.save_dir), 'swiglu_phase2_mlm', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+11}: PPL={ppl:.2f}")
        best_sw = min(sw_ppl2)

        in_base = build_nonshared_innernet_transformer(
            sw_state, fitted_weights, vocab_size, 128, 4, 512, 4, 64, device)
        in_model = MLMWrapper(in_base)
        opt_in = optim.Adam(in_model.parameters(), lr=5e-4)
        in_ppl = []
        for ep in range(10):
            train_mlm(in_model, opt_in)
            ppl = eval_mlm(in_model)
            in_ppl.append(ppl)
            save_ckpt(in_model, os.path.basename(args.save_dir), 'nonshared_mlm', seed, ep+1)
            if (ep+1) % 5 == 0: logger.info(f"  NonShared Ep {ep+11}: PPL={ppl:.2f}")
        best_in = min(in_ppl)

        logger.info(f"  MLM RESULT: SwiGLU={best_sw:.2f} vs NonShared={best_in:.2f}")
        results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

        for i, block in enumerate(in_base.blocks):
            torch.save(block.ffn.inner_net.state_dict(),
                       os.path.join(args.save_dir, f'mlm_inner_layer{i}_seed{seed}.pth'))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/warmstart_nonshared')
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    inner_template = InnerNetAct(32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.state_dict()

    all_results = {}

    # Task 1: PTB
    logger.info("\n" + "="*60)
    logger.info("TASK: PTB d=128")
    all_results['ptb'] = run_transformer_task(
        'ptb', 128, 4, 512, 4, 64, fitted_weights, device, args)

    # Task 2: MLM
    logger.info("\n" + "="*60)
    logger.info("TASK: MLM WikiText-2")
    all_results['mlm'] = run_mlm_task(fitted_weights, device, args)

    # Task 3: WikiText-2 d=128
    logger.info("\n" + "="*60)
    logger.info("TASK: WikiText-2 d=128")
    all_results['wiki'] = run_transformer_task(
        'wikitext', 128, 4, 512, 4, 64, fitted_weights, device, args)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    for task, results in all_results.items():
        sw = [r['best_sw'] for r in results]
        inn = [r['best_in'] for r in results]
        logger.info(f"  {task}: SwiGLU={np.mean(sw):.2f}±{np.std(sw):.2f} vs NonShared={np.mean(inn):.2f}±{np.std(inn):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
