"""Free-init warm-start: good network weights, free InnerNet.

Network (w1a, w1b, w2) from trained SwiGLU. InnerNet initialized
differently to avoid SwiGLU bias.

4 InnerNet initializations:
  1. SwiGLU-fitted (current baseline)
  2. Simple multiply: f(a,b) ≈ a*b
  3. Random (untrained MLP)
  4. Identity-like: f(a,b) ≈ a

All get same SwiGLU network weights. Compare after training.
Tests on d=128 WikiText-2 and MLM.
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


class InnerNetAct(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x)


def fit_to_target(inner_net, device, target_fn, steps=2000, lr=1e-3):
    opt = optim.Adam(inner_net.net.parameters(), lr=lr)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = target_fn(inputs[:, 0], inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    return loss.item()


def make_innernet_model(swiglu_state, inner_net, vocab_size, d_model, n_heads,
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


def evaluate_ppl(model, val_loader, device):
    model.eval()
    total, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            total += nn.CrossEntropyLoss()(model(x), y).item()
            n += 1
    return math.exp(total / n)


def train_ep(model, train_loader, optimizer, device):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def run_lm_experiment(dataset_name, swiglu_state, vocab_size, init_weights_dict,
                       d_model, n_heads, d_ff, n_layers, context_size,
                       train_loader, val_loader, device, epochs=10, lr=5e-4):
    """Run one seed with multiple InnerNet initializations."""
    results = {}
    for init_name, inner_state in init_weights_dict.items():
        inner = InnerNetAct(32).to(device)
        inner.load_state_dict(inner_state)

        model = make_innernet_model(
            swiglu_state, inner, vocab_size, d_model, n_heads,
            d_ff, n_layers, context_size, device)

        ppl_swap = evaluate_ppl(model, val_loader, device)
        logger.info(f"    {init_name}: after swap PPL={ppl_swap:.2f}")

        opt = optim.Adam(model.parameters(), lr=lr)
        ppls = []
        for ep in range(epochs):
            train_ep(model, train_loader, opt, device)
            ppl = evaluate_ppl(model, val_loader, device)
            ppls.append(ppl)
            if (ep+1) % 5 == 0:
                logger.info(f"    {init_name} Ep {ep+1}: PPL={ppl:.2f}")

        best = min(ppls)
        results[init_name] = {'swap_ppl': ppl_swap, 'ppls': ppls, 'best': best}
        logger.info(f"    {init_name}: best={best:.2f}")

        # Save weights
        torch.save(inner.state_dict(),
                   os.path.join(args.save_dir, f'inner_{init_name}_{dataset_name}.pth'))

    return results


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/free_init')
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    from model.transformer import SwiGLUTransformer

    # Prepare 4 InnerNet initializations
    logger.info("=== Preparing InnerNet initializations ===")

    # 1. SwiGLU-fitted
    inner_swiglu = InnerNetAct(32).to(device)
    mse = fit_to_target(inner_swiglu, device,
                         lambda a, b: F.silu(a) * b)
    logger.info(f"  swiglu_fitted: MSE={mse:.6f}")

    # 2. Simple multiply: f(a,b) = a*b
    inner_mult = InnerNetAct(32).to(device)
    mse = fit_to_target(inner_mult, device,
                         lambda a, b: a * b)
    logger.info(f"  multiply: MSE={mse:.6f}")

    # 3. Random (no fitting)
    inner_random = InnerNetAct(32).to(device)
    logger.info(f"  random: no fitting")

    # 4. Identity-like: f(a,b) = a
    inner_identity = InnerNetAct(32).to(device)
    mse = fit_to_target(inner_identity, device,
                         lambda a, b: a)
    logger.info(f"  identity: MSE={mse:.6f}")

    init_weights = {
        'swiglu_fitted': copy.deepcopy(inner_swiglu.state_dict()),
        'multiply': copy.deepcopy(inner_mult.state_dict()),
        'random': copy.deepcopy(inner_random.state_dict()),
        'identity': copy.deepcopy(inner_identity.state_dict()),
    }

    # ==================== WikiText-2 d=128 ====================
    logger.info("\n" + "="*60)
    logger.info("TASK: WikiText-2 d=128")

    train_ds = WikiTextDataset(split='train', context_size=64)
    val_ds = WikiTextDataset(split='validation', context_size=64, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    all_wiki = []
    for si, seed in enumerate(range(42, 42 + args.num_seeds)):
        logger.info(f"\n[Seed {seed}] ({si+1}/{args.num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Train SwiGLU 10ep
        sw = SwiGLUTransformer(vocab_size, 128, 4, 512, 4, 64, 0.1).to(device)
        opt_sw = optim.Adam(sw.parameters(), lr=5e-4)
        for ep in range(10):
            train_ep(sw, train_loader, opt_sw, device)
        sw_state = sw.state_dict()

        # SwiGLU continues 10ep
        sw_ppls = []
        for ep in range(10):
            train_ep(sw, train_loader, opt_sw, device)
            ppl = evaluate_ppl(sw, val_loader, device)
            sw_ppls.append(ppl)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+11}: PPL={ppl:.2f}")
        best_sw = min(sw_ppls)
        logger.info(f"  SwiGLU best: {best_sw:.2f}")

        # Run all InnerNet inits
        res = run_lm_experiment(
            'wiki', sw_state, vocab_size, init_weights,
            128, 4, 512, 4, 64, train_loader, val_loader, device)
        res['swiglu_best'] = best_sw
        all_wiki.append(res)

    # ==================== MLM ====================
    logger.info("\n" + "="*60)
    logger.info("TASK: MLM WikiText-2")

    from runner.mlm_runner import MLMDataset, MLMWrapper

    mlm_train = MLMDataset(split='train', seq_len=64)
    mlm_val = MLMDataset(split='validation', seq_len=64, vocab=mlm_train.vocab)
    mlm_vocab = mlm_train.vocab_size
    mask_id = mlm_train.mask_token_id
    mlm_train_loader = DataLoader(mlm_train, batch_size=128, shuffle=True, num_workers=4)
    mlm_val_loader = DataLoader(mlm_val, batch_size=128, num_workers=4)

    def mask_tokens(x, mask_id, p=0.15):
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
            for x in mlm_val_loader:
                x = x.to(device)
                mx, labels = mask_tokens(x, mask_id)
                logits = model(mx)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
                n = (labels != -100).sum().item()
                total += loss.item() * n; nm += n
        return math.exp(min(total / max(nm, 1), 20))

    def train_mlm_ep(model, opt):
        model.train()
        for x in mlm_train_loader:
            x = x.to(device)
            mx, labels = mask_tokens(x, mask_id)
            opt.zero_grad()
            logits = model(mx)
            nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    all_mlm = []
    for si, seed in enumerate(range(42, 42 + args.num_seeds)):
        logger.info(f"\n[Seed {seed}] ({si+1}/{args.num_seeds})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sw_base = SwiGLUTransformer(mlm_vocab, 128, 4, 512, 4, 64, 0.1).to(device)
        sw_model = MLMWrapper(sw_base)
        opt_sw = optim.Adam(sw_model.parameters(), lr=5e-4)
        for ep in range(10):
            train_mlm_ep(sw_model, opt_sw)
        sw_state = sw_base.state_dict()

        sw_ppls = []
        for ep in range(10):
            train_mlm_ep(sw_model, opt_sw)
            ppl = eval_mlm(sw_model)
            sw_ppls.append(ppl)
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+11}: PPL={ppl:.2f}")
        best_sw = min(sw_ppls)

        # InnerNet inits for MLM
        mlm_results = {}
        for init_name, inner_state in init_weights.items():
            inner = InnerNetAct(32).to(device)
            inner.load_state_dict(inner_state)
            in_base = make_innernet_model(
                sw_state, inner, mlm_vocab, 128, 4, 512, 4, 64, device)
            in_model = MLMWrapper(in_base)

            opt_in = optim.Adam(in_model.parameters(), lr=5e-4)
            ppls = []
            for ep in range(10):
                train_mlm_ep(in_model, opt_in)
                ppl = eval_mlm(in_model)
                ppls.append(ppl)
                if (ep+1) % 5 == 0: logger.info(f"    {init_name} Ep {ep+11}: PPL={ppl:.2f}")
            best = min(ppls)
            mlm_results[init_name] = {'best': best, 'ppls': ppls}
            logger.info(f"    {init_name}: best={best:.2f}")

            torch.save(inner.state_dict(),
                       os.path.join(args.save_dir, f'inner_{init_name}_mlm_seed{seed}.pth'))

        mlm_results['swiglu_best'] = best_sw
        all_mlm.append(mlm_results)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    for task_name, all_res in [('WikiText-2', all_wiki), ('MLM', all_mlm)]:
        logger.info(f"\n{task_name}:")
        sw_bests = [r['swiglu_best'] for r in all_res]
        logger.info(f"  SwiGLU:        {np.mean(sw_bests):.2f} ± {np.std(sw_bests):.2f}")
        for init_name in init_weights:
            bests = [r[init_name]['best'] for r in all_res]
            logger.info(f"  {init_name:15s}: {np.mean(bests):.2f} ± {np.std(bests):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'wiki': all_wiki, 'mlm': all_mlm}, f)


if __name__ == '__main__':
    main()
