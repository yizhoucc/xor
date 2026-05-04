"""Multiply-init warm-start across tasks.

Same as innernet_vs_swiglu.py but InnerNet initialized to f(a,b)=a*b
instead of SwiGLU. Tests if the endpoint is the same regardless of init.

Tasks: d=64, d=128, PTB, MLM, CNN
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


def fit_to_multiply(inner_net, device, steps=2000):
    opt = optim.Adam(inner_net.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (inputs[:, 0] * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet to a*b, MSE={loss.item():.6f}")


def make_innernet_from_swiglu(swiglu_state, inner_net, vocab_size, d_model, n_heads,
                               d_ff, n_layers, context_size, device):
    from model.transformer import InnerNetTransformer
    model = InnerNetTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, context_size, 32, 0.1
    ).to(device)
    model_dict = model.state_dict()
    for k, v in swiglu_state.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)
    for block in model.blocks:
        block.ffn.inner_net = inner_net
    return model


def eval_ppl(model, val_loader, device):
    model.eval()
    total, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            total += nn.CrossEntropyLoss()(model(x), y).item()
            n += 1
    return math.exp(total / n)


def train_ep(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def run_lm_task(name, dataset_name, d_model, n_heads, d_ff, n_layers,
                 fitted_weights, device, num_seeds=5, lr=5e-4):
    from runner.lm_runner import WikiTextDataset
    from model.transformer import SwiGLUTransformer

    train_ds = WikiTextDataset(split='train', context_size=64, dataset_name=dataset_name)
    val_ds = WikiTextDataset(split='validation', context_size=64, vocab=train_ds.vocab, dataset_name=dataset_name)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    results = []
    for si, seed in enumerate(range(42, 42 + num_seeds)):
        logger.info(f"\n  [{name}] Seed {seed} ({si+1}/{num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sw = SwiGLUTransformer(vocab_size, d_model, n_heads, d_ff, n_layers, 64, 0.1).to(device)
        opt_sw = optim.Adam(sw.parameters(), lr=lr)
        for ep in range(10):
            train_ep(sw, train_loader, opt_sw, device)
        sw_state = copy.deepcopy(sw.state_dict())

        # SwiGLU continues
        sw_ppls = []
        for ep in range(10):
            train_ep(sw, train_loader, opt_sw, device)
            ppl = eval_ppl(sw, val_loader, device)
            sw_ppls.append(ppl)
        best_sw = min(sw_ppls)

        # Multiply-init InnerNet
        inner = InnerNetAct(32).to(device)
        inner.load_state_dict(fitted_weights)
        model = make_innernet_from_swiglu(sw_state, inner, vocab_size, d_model, n_heads,
                                           d_ff, n_layers, 64, device)
        opt_in = optim.Adam(model.parameters(), lr=lr)
        in_ppls = []
        for ep in range(10):
            train_ep(model, train_loader, opt_in, device)
            ppl = eval_ppl(model, val_loader, device)
            in_ppls.append(ppl)
        best_in = min(in_ppls)

        logger.info(f"    SwiGLU={best_sw:.2f} vs MultInit={best_in:.2f}")
        results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw'] for r in results]
    in_b = [r['best_in'] for r in results]
    logger.info(f"  {name} SUMMARY: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs MultInit={np.mean(in_b):.2f}±{np.std(in_b):.2f}")
    return results


def run_mlm_task(fitted_weights, device, num_seeds=5):
    from runner.mlm_runner import MLMDataset, MLMWrapper
    from model.transformer import SwiGLUTransformer

    train_ds = MLMDataset(split='train', seq_len=64)
    val_ds = MLMDataset(split='validation', seq_len=64, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    mask_id = train_ds.mask_token_id
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    def mask_tokens(x, p=0.15):
        labels = x.clone()
        mask = torch.rand(x.shape, device=x.device) < p
        labels[~mask] = -100
        mx = x.clone()
        mx[mask & (torch.rand(x.shape, device=x.device) < 0.8)] = mask_id
        return mx, labels

    def eval_mlm(model):
        model.eval()
        total, nm = 0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                mx, labels = mask_tokens(x)
                logits = model(mx)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
                n = (labels != -100).sum().item()
                total += loss.item() * n; nm += n
        return math.exp(min(total / max(nm, 1), 20))

    def train_mlm(model, opt):
        model.train()
        for x in train_loader:
            x = x.to(device)
            mx, labels = mask_tokens(x)
            opt.zero_grad()
            logits = model(mx)
            nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    results = []
    for si, seed in enumerate(range(42, 42 + num_seeds)):
        logger.info(f"\n  [MLM] Seed {seed} ({si+1}/{num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        from runner.mlm_runner import MLMWrapper
        sw_base = SwiGLUTransformer(vocab_size, 128, 4, 512, 4, 64, 0.1).to(device)
        sw_model = MLMWrapper(sw_base)
        opt_sw = optim.Adam(sw_model.parameters(), lr=5e-4)
        for ep in range(10):
            train_mlm(sw_model, opt_sw)
        sw_state = sw_base.state_dict()

        sw_ppls = []
        for ep in range(10):
            train_mlm(sw_model, opt_sw)
            ppl = eval_mlm(sw_model)
            sw_ppls.append(ppl)
        best_sw = min(sw_ppls)

        inner = InnerNetAct(32).to(device)
        inner.load_state_dict(fitted_weights)
        in_base = make_innernet_from_swiglu(sw_state, inner, vocab_size, 128, 4, 512, 4, 64, device)
        in_model = MLMWrapper(in_base)
        opt_in = optim.Adam(in_model.parameters(), lr=5e-4)
        in_ppls = []
        for ep in range(10):
            train_mlm(in_model, opt_in)
            ppl = eval_mlm(in_model)
            in_ppls.append(ppl)
        best_in = min(in_ppls)

        logger.info(f"    SwiGLU={best_sw:.2f} vs MultInit={best_in:.2f}")
        results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw'] for r in results]
    in_b = [r['best_in'] for r in results]
    logger.info(f"  MLM SUMMARY: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs MultInit={np.mean(in_b):.2f}±{np.std(in_b):.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/multiply_init')
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    inner_template = InnerNetAct(32).to(device)
    fit_to_multiply(inner_template, device)
    fitted_weights = inner_template.state_dict()

    all_results = {}

    logger.info("\n" + "="*60)
    all_results['d64'] = run_lm_task('d=64', 'wikitext', 64, 4, 256, 4, fitted_weights, device, args.num_seeds)

    logger.info("\n" + "="*60)
    all_results['d128'] = run_lm_task('d=128', 'wikitext', 128, 4, 512, 4, fitted_weights, device, args.num_seeds)

    logger.info("\n" + "="*60)
    all_results['ptb'] = run_lm_task('PTB', 'ptb', 128, 4, 512, 4, fitted_weights, device, args.num_seeds)

    logger.info("\n" + "="*60)
    all_results['mlm'] = run_mlm_task(fitted_weights, device, args.num_seeds)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    for task, results in all_results.items():
        sw = [r['best_sw'] for r in results]
        inn = [r['best_in'] for r in results]
        logger.info(f"  {task:8s}: SwiGLU={np.mean(sw):.2f}±{np.std(sw):.2f} vs MultInit={np.mean(inn):.2f}±{np.std(inn):.2f}  diff={np.mean(sw)-np.mean(inn):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
