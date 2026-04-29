"""CNN warm-start with non-shared InnerNet (each layer has its own)."""
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


class InnerNetAct2d(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        C = x.size(1) // 2
        a, b = x[:, :C], x[:, C:]
        B, C, H, W = a.shape
        pairs = torch.stack([a, b], dim=-1)
        return self.net(pairs.reshape(-1, 2)).view(B, C, H, W)


class SwiGLUAct2d(nn.Module):
    def forward(self, x):
        C = x.size(1) // 2
        return F.silu(x[:, :C]) * x[:, C:]


class CNN_NonShared(nn.Module):
    """CNN with non-shared InnerNet (each layer gets its own)."""
    def __init__(self, inner_nets, num_classes=10):
        super().__init__()
        channels = [60, 120, 120, 120]
        layers = []
        in_ch = 3
        self.inner_nets = nn.ModuleList(inner_nets)
        for i, out_ch in enumerate(channels):
            layers += [nn.Conv2d(in_ch, out_ch*2, 3, padding=1), nn.BatchNorm2d(out_ch*2),
                       self.inner_nets[i], nn.MaxPool2d(2), nn.Dropout(0.5)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1]*2*2, num_classes)
    def forward(self, x):
        out = self.features(x)
        return self.fc(out.view(out.size(0), -1))


class CNN_SwiGLU(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        act = SwiGLUAct2d()
        channels = [60, 120, 120, 120]
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch*2, 3, padding=1), nn.BatchNorm2d(out_ch*2),
                       act, nn.MaxPool2d(2), nn.Dropout(0.5)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1]*2*2, num_classes)
    def forward(self, x):
        out = self.features(x)
        return self.fc(out.view(out.size(0), -1))


def get_loaders(bs=100):
    t1 = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    t2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    return (DataLoader(torchvision.datasets.CIFAR10('./data', True, download=True, transform=t1), bs, shuffle=True, num_workers=4),
            DataLoader(torchvision.datasets.CIFAR10('./data', False, transform=t2), bs, num_workers=4))


def train_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); nn.CrossEntropyLoss()(model(x), y).backward(); opt.step()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/warmstart_cnn_nonshared')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--fork_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, test_loader = get_loaders()

    # Fit template
    inner_template = InnerNetAct2d(32).to(device)
    fit_innernet(inner_template, device)
    fitted_weights = inner_template.net.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        # SwiGLU train to fork
        sw_model = CNN_SwiGLU().to(device)
        opt_sw = optim.Adam(sw_model.parameters(), lr=args.lr)
        for ep in range(args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            if (ep+1) % 20 == 0:
                acc = evaluate(sw_model, test_loader, device)
                logger.info(f"  SwiGLU Ep {ep+1}: {acc*100:.2f}%")
        swiglu_state = copy.deepcopy(sw_model.state_dict())

        # Branch A: SwiGLU continues
        sw_accs = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            acc = evaluate(sw_model, test_loader, device)
            sw_accs.append(acc)
            if (ep+1) % 20 == 0: logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: {acc*100:.2f}%")
        best_sw = max(sw_accs)

        # Branch B: Non-shared InnerNet
        inner_nets = [InnerNetAct2d(32).to(device) for _ in range(4)]
        for inet in inner_nets:
            inet.net.load_state_dict(fitted_weights)
        in_model = CNN_NonShared(inner_nets).to(device)

        # Copy weights from SwiGLU
        inn_dict = in_model.state_dict()
        copied = 0
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v; copied += 1
        in_model.load_state_dict(inn_dict, strict=False)
        logger.info(f"  Copied {copied} tensors")

        opt_in = optim.Adam(in_model.parameters(), lr=args.lr)
        in_accs = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(in_model, train_loader, opt_in, device)
            acc = evaluate(in_model, test_loader, device)
            in_accs.append(acc)
            if (ep+1) % 20 == 0: logger.info(f"  NonShared Ep {args.fork_epoch+ep+1}: {acc*100:.2f}%")
        best_in = max(in_accs)

        # Save per-layer weights
        for i, inet in enumerate(inner_nets):
            torch.save(inet.net.state_dict(), os.path.join(args.save_dir, f'inner_layer{i}_seed{seed}.pth'))

        logger.info(f"  RESULT: SwiGLU={best_sw*100:.2f}% vs NonShared={best_in*100:.2f}%")
        all_results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw']*100 for r in all_results]
    in_b = [r['best_in']*100 for r in all_results]
    logger.info(f"\nSUMMARY: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs NonShared={np.mean(in_b):.2f}±{np.std(in_b):.2f}")
    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
