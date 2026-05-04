"""MLP-Mixer warm-start: SwiGLU vs InnerNet on CIFAR-10.

Simple Mixer: patch embed → mixer blocks (token-mix + channel-mix) → cls head.
InnerNet replaces activation in channel-mixing MLP.
"""
import os, sys, math, copy, pickle, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

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
    def forward(self, a, b):
        pairs = torch.stack([a, b], dim=-1)
        shape = pairs.shape[:-1]
        return self.net(pairs.reshape(-1, 2)).view(*shape)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1a = nn.Linear(dim, hidden)
        self.w1b = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w2(F.silu(self.w1a(x)) * self.w1b(x))


class InnerNetMLP(nn.Module):
    def __init__(self, dim, hidden, inner_net):
        super().__init__()
        self.w1a = nn.Linear(dim, hidden)
        self.w1b = nn.Linear(dim, hidden)
        self.inner_net = inner_net
        self.w2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w2(self.inner_net(self.w1a(x), self.w1b(x)))


class MixerBlock(nn.Module):
    def __init__(self, num_patches, d_model, token_hidden, channel_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(nn.Linear(num_patches, token_hidden), nn.GELU(), nn.Linear(token_hidden, num_patches))
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = channel_mlp
    def forward(self, x):
        x = x + self.token_mix(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(self.ln2(x))
        return x


class SimpleMixer(nn.Module):
    def __init__(self, channel_mlp_fn, d_model=256, n_layers=4, patch_size=4, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, d_model, num_patches * 2, channel_mlp_fn())
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x.mean(dim=1)))


def get_loaders(bs=128):
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    return (DataLoader(torchvision.datasets.CIFAR10('./data', True, download=True, transform=t_train), bs, shuffle=True, num_workers=4),
            DataLoader(torchvision.datasets.CIFAR10('./data', False, transform=t_test), bs, num_workers=4))


def train_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); nn.CrossEntropyLoss()(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


def fit_innernet(inner, device, steps=2000):
    opt = optim.Adam(inner.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (F.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad(); nn.MSELoss()(inner.net(inputs), targets).backward(); opt.step()
    logger.info(f"  Fitted InnerNet, MSE={nn.MSELoss()(inner.net(inputs), targets).item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/warmstart_mixer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fork_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, test_loader = get_loaders()

    inner_template = InnerNetAct(32).to(device)
    fit_innernet(inner_template, device)
    fitted_weights = inner_template.net.state_dict()
    d_model, d_ff = args.d_model, args.d_model * 4

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        sw_model = SimpleMixer(lambda: SwiGLUMLP(d_model, d_ff), d_model=d_model).to(device)
        opt_sw = optim.Adam(sw_model.parameters(), lr=args.lr)
        sw_acc = []
        for ep in range(args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            acc = evaluate(sw_model, test_loader, device)
            sw_acc.append(acc)
            save_ckpt(sw_model, 'warmstart_mixer', 'swiglu_phase1', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {ep+1}: {acc*100:.2f}%")
        swiglu_state = copy.deepcopy(sw_model.state_dict())

        sw_acc2 = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            acc = evaluate(sw_model, test_loader, device)
            sw_acc2.append(acc)
            save_ckpt(sw_model, 'warmstart_mixer', 'swiglu_phase2', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: {acc*100:.2f}%")
        best_sw = max(sw_acc + sw_acc2)

        shared_inner = InnerNetAct(32).to(device)
        shared_inner.net.load_state_dict(fitted_weights)
        in_model = SimpleMixer(lambda: InnerNetMLP(d_model, d_ff, shared_inner), d_model=d_model).to(device)
        inn_dict = in_model.state_dict()
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape: inn_dict[k] = v
        in_model.load_state_dict(inn_dict, strict=False)
        for block in in_model.blocks:
            block.channel_mix.inner_net = shared_inner

        opt_in = optim.Adam(in_model.parameters(), lr=args.lr)
        in_acc = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(in_model, train_loader, opt_in, device)
            acc = evaluate(in_model, test_loader, device)
            in_acc.append(acc)
            save_ckpt(in_model, 'warmstart_mixer', 'innernet', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  InnerNet Ep {args.fork_epoch+ep+1}: {acc*100:.2f}%")
        best_in = max(in_acc)

        logger.info(f"  RESULT: SwiGLU={best_sw*100:.2f}% vs InnerNet={best_in*100:.2f}%")
        all_results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw']*100 for r in all_results]
    in_b = [r['best_in']*100 for r in all_results]
    logger.info(f"\nSUMMARY: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs InnerNet={np.mean(in_b):.2f}±{np.std(in_b):.2f}")
    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)

if __name__ == '__main__':
    main()
