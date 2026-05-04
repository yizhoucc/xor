"""From-scratch training with different InnerNet initializations.

NOT warm-start. Main network starts from random. Only InnerNet gets a head start.
Tests whether InnerNet init can solve co-adaptation without warm-start.

4 conditions:
  1. SwiGLU (baseline, fixed activation)
  2. InnerNet random init (current from-scratch)
  3. InnerNet gaussian pretrain (paper's approach)
  4. InnerNet multiply init (f=a*b)

All on WikiText-2 d=128.
"""
import os, sys, math, pickle, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def fit_to_target(inner_net, device, target_fn, steps=2000):
    opt = optim.Adam(inner_net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = target_fn(inputs[:, 0], inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net(inputs), targets)
        loss.backward()
        opt.step()
    return loss.item()


def gaussian_pretrain(inner_net, device, steps=300):
    """Paper's approach: fit to Gaussian-blurred random function."""
    from scipy.signal import convolve2d
    from scipy.stats import multivariate_normal

    nb = 101
    x = np.linspace(-5, 5, nb)
    y = np.linspace(-5, 5, nb)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T

    mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
    gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
    gaussian_kernel /= gaussian_kernel.sum()

    npr = np.random.RandomState(seed=42)
    init_unif = npr.uniform(-1, 1, size=(nb, nb))
    targets_np = convolve2d(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)

    inputs = torch.tensor(xy, dtype=torch.float32).to(device)
    targets = torch.tensor(targets_np, dtype=torch.float32).to(device)

    opt = optim.Adam(inner_net.parameters(), lr=1e-2)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net(inputs), targets)
        loss.backward()
        opt.step()

    logger.info(f"  Gaussian pretrain done, MSE={loss.item():.6f}")
    return loss.item()


CKPT_DIR = '/user_data/yizhouc3/xor_checkpoints'


def save_ckpt(model, exp_name, cond, seed, epoch, optimizer=None, metrics=None):
    """Save full checkpoint to user_data."""
    d = os.path.join(CKPT_DIR, exp_name, f'{cond}_seed{seed}')
    os.makedirs(d, exist_ok=True)
    state = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        state['metrics'] = metrics
    torch.save(state, os.path.join(d, f'ep{epoch:03d}.pth'))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/from_scratch_init')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    from model.transformer import (SwiGLUTransformer, InnerNetTransformer,
                                    InnerNetFFNActivation)

    train_ds = WikiTextDataset(split='train', context_size=64)
    val_ds = WikiTextDataset(split='validation', context_size=64, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    # Prepare InnerNet initializations
    logger.info("=== Preparing initializations ===")

    # Gaussian pretrain (paper's approach)
    inner_gauss = InnerNetFFNActivation(hidden_dim=32).to(device)
    gaussian_pretrain(inner_gauss, device)
    gauss_weights = {k: v.clone() for k, v in inner_gauss.state_dict().items()}

    # Multiply init
    inner_mult = InnerNetFFNActivation(hidden_dim=32).to(device)
    mse = fit_to_target(inner_mult, device, lambda a, b: a * b)
    logger.info(f"  Multiply init, MSE={mse:.6f}")
    mult_weights = {k: v.clone() for k, v in inner_mult.state_dict().items()}

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)})")

        seed_results = {}

        # 1. SwiGLU baseline
        logger.info("--- SwiGLU (baseline) ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        sw = SwiGLUTransformer(vocab_size, 128, 4, 512, 4, 64, 0.1).to(device)
        opt = optim.Adam(sw.parameters(), lr=args.lr)
        sw_ppls = []
        for ep in range(args.epochs):
            train_ep(sw, train_loader, opt, device)
            ppl = eval_ppl(sw, val_loader, device)
            sw_ppls.append(ppl)
            save_ckpt(sw, 'from_scratch_init', 'swiglu', seed, ep+1, opt, {'ppl': ppl})
            if (ep+1) % 5 == 0: logger.info(f"  SwiGLU Ep {ep+1}: PPL={ppl:.2f}")
        seed_results['swiglu'] = {'ppls': sw_ppls, 'best': min(sw_ppls)}
        logger.info(f"  SwiGLU best: {min(sw_ppls):.2f}")

        # 2. InnerNet random init
        logger.info("--- InnerNet random init ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model_rand = InnerNetTransformer(vocab_size, 128, 4, 512, 4, 64, 32, 0.1).to(device)
        opt = optim.Adam(model_rand.parameters(), lr=args.lr)
        rand_ppls = []
        for ep in range(args.epochs):
            train_ep(model_rand, train_loader, opt, device)
            ppl = eval_ppl(model_rand, val_loader, device)
            rand_ppls.append(ppl)
            save_ckpt(model_rand, 'from_scratch_init', 'random', seed, ep+1, opt, {'ppl': ppl})
            if (ep+1) % 5 == 0: logger.info(f"  Random Ep {ep+1}: PPL={ppl:.2f}")
        seed_results['random'] = {'ppls': rand_ppls, 'best': min(rand_ppls)}
        logger.info(f"  Random best: {min(rand_ppls):.2f}")

        # 3. InnerNet gaussian pretrain (paper's approach)
        logger.info("--- InnerNet gaussian pretrain ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model_gauss = InnerNetTransformer(vocab_size, 128, 4, 512, 4, 64, 32, 0.1).to(device)
        # Load gaussian pretrained InnerNet
        for block in model_gauss.blocks:
            block.ffn.inner_net.load_state_dict(gauss_weights)
        opt = optim.Adam(model_gauss.parameters(), lr=args.lr)
        gauss_ppls = []
        for ep in range(args.epochs):
            train_ep(model_gauss, train_loader, opt, device)
            ppl = eval_ppl(model_gauss, val_loader, device)
            gauss_ppls.append(ppl)
            save_ckpt(model_gauss, 'from_scratch_init', 'gaussian', seed, ep+1, opt, {'ppl': ppl})
            if (ep+1) % 5 == 0: logger.info(f"  Gaussian Ep {ep+1}: PPL={ppl:.2f}")
        seed_results['gaussian'] = {'ppls': gauss_ppls, 'best': min(gauss_ppls)}
        logger.info(f"  Gaussian best: {min(gauss_ppls):.2f}")

        # 4. InnerNet multiply init
        logger.info("--- InnerNet multiply init ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model_mult = InnerNetTransformer(vocab_size, 128, 4, 512, 4, 64, 32, 0.1).to(device)
        for block in model_mult.blocks:
            block.ffn.inner_net.load_state_dict(mult_weights)
        opt = optim.Adam(model_mult.parameters(), lr=args.lr)
        mult_ppls = []
        for ep in range(args.epochs):
            train_ep(model_mult, train_loader, opt, device)
            ppl = eval_ppl(model_mult, val_loader, device)
            mult_ppls.append(ppl)
            save_ckpt(model_mult, 'from_scratch_init', 'multiply', seed, ep+1, opt, {'ppl': ppl})
            if (ep+1) % 5 == 0: logger.info(f"  Multiply Ep {ep+1}: PPL={ppl:.2f}")
        seed_results['multiply'] = {'ppls': mult_ppls, 'best': min(mult_ppls)}
        logger.info(f"  Multiply best: {min(mult_ppls):.2f}")

        # Save InnerNet weights
        for name, model in [('random', model_rand), ('gaussian', model_gauss), ('multiply', model_mult)]:
            torch.save(model.blocks[0].ffn.inner_net.state_dict(),
                       os.path.join(args.save_dir, f'inner_{name}_seed{seed}.pth'))

        all_results.append(seed_results)
        logger.info(f"\n  Seed {seed}: SwiGLU={seed_results['swiglu']['best']:.2f} "
                    f"Random={seed_results['random']['best']:.2f} "
                    f"Gaussian={seed_results['gaussian']['best']:.2f} "
                    f"Multiply={seed_results['multiply']['best']:.2f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY ({len(seeds)} seeds)")
    for cond in ['swiglu', 'random', 'gaussian', 'multiply']:
        bests = [r[cond]['best'] for r in all_results]
        logger.info(f"  {cond:12s}: {np.mean(bests):.2f} ± {np.std(bests):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
