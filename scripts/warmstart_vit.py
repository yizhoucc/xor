"""ViT warm-start: SwiGLU vs InnerNet on CIFAR-10.

Simple ViT: patch embed → transformer blocks (with SwiGLU/InnerNet FFN) → cls head.
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


class InnerNetAct(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, a, b):
        pairs = torch.stack([a, b], dim=-1)
        shape = pairs.shape[:-1]
        return self.net(pairs.reshape(-1, 2)).view(*shape)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)
        self.w1b = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1a(x)) * self.w1b(x)))


class InnerNetFFN(nn.Module):
    def __init__(self, d_model, d_ff, inner_net, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)
        self.w1b = nn.Linear(d_model, d_ff)
        self.inner_net = inner_net
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        a, b = self.w1a(x), self.w1b(x)
        return self.w2(self.dropout(self.inner_net(a, b)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x2 = self.ln1(x)
        x = x + self.drop(self.attn(x2, x2, x2)[0])
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class SimpleViT(nn.Module):
    def __init__(self, ffn_fn, d_model=192, n_heads=4, n_layers=4, patch_size=4, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn_fn()) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x[:, 0]))


def get_loaders(batch_size=128):
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
    test = torchvision.datasets.CIFAR10('./data', train=False, transform=t_test)
    return DataLoader(train, batch_size, shuffle=True, num_workers=4), DataLoader(test, batch_size, num_workers=4)


def train_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    c = t = 0
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
    parser.add_argument('--save_dir', default='exp/warmstart_vit')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fork_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=192)
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

        sw_model = SimpleViT(lambda: SwiGLUFFN(d_model, d_ff), d_model=d_model).to(device)
        opt_sw = optim.Adam(sw_model.parameters(), lr=args.lr)
        sw_acc = []
        for ep in range(args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            acc = evaluate(sw_model, test_loader, device)
            sw_acc.append(acc)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {ep+1}: {acc*100:.2f}%")
        swiglu_state = copy.deepcopy(sw_model.state_dict())

        # Branch A
        sw_acc2 = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            acc = evaluate(sw_model, test_loader, device)
            sw_acc2.append(acc)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: {acc*100:.2f}%")
        best_sw = max(sw_acc + sw_acc2)

        # Branch B
        shared_inner = InnerNetAct(32).to(device)
        shared_inner.net.load_state_dict(fitted_weights)
        in_model = SimpleViT(lambda: InnerNetFFN(d_model, d_ff, shared_inner), d_model=d_model).to(device)
        inn_dict = in_model.state_dict()
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape: inn_dict[k] = v
        in_model.load_state_dict(inn_dict, strict=False)
        for block in in_model.blocks:
            block.ffn.inner_net = shared_inner

        opt_in = optim.Adam(in_model.parameters(), lr=args.lr)
        in_acc = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(in_model, train_loader, opt_in, device)
            acc = evaluate(in_model, test_loader, device)
            in_acc.append(acc)
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
