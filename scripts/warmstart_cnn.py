"""CNN warm-start: SwiGLU vs InnerNet on CIFAR-10.

Self-contained: builds both models with matching architecture (split first/second half).
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


class InnerNetAct2d(nn.Module):
    """InnerNet for CNN: split channels first/second half, apply MLP(a,b)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        C = x.size(1) // 2
        a, b = x[:, :C], x[:, C:]
        B, C, H, W = a.shape
        pairs = torch.stack([a, b], dim=-1)  # [B,C,H,W,2]
        out = self.net(pairs.reshape(-1, 2)).view(B, C, H, W)
        return out


class SwiGLUAct2d(nn.Module):
    def forward(self, x):
        C = x.size(1) // 2
        return F.silu(x[:, :C]) * x[:, C:]


def make_cnn(act_module, in_ch=3, channels=[60,120,120,120], num_classes=10):
    layers = []
    for out_ch in channels:
        layers += [nn.Conv2d(in_ch, out_ch*2, 3, padding=1), nn.BatchNorm2d(out_ch*2), act_module, nn.MaxPool2d(2), nn.Dropout(0.5)]
        in_ch = out_ch
    return nn.Sequential(*layers), nn.Linear(channels[-1]*2*2, num_classes)


class CNN(nn.Module):
    def __init__(self, act_module, num_classes=10):
        super().__init__()
        self.features, self.fc = make_cnn(act_module, num_classes=num_classes)
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, x):
        out = self.features(x)
        return self.fc(out.view(out.size(0), -1))


def get_loaders(batch_size=100):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test)
    return DataLoader(train_ds, batch_size, shuffle=True, num_workers=4), DataLoader(test_ds, batch_size, shuffle=False, num_workers=4)


def train_epoch(model, loader, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def fit_innernet_to_swiglu(inner, device, steps=2000):
    opt = optim.Adam(inner.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (F.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet, MSE={loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/warmstart_cnn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--fork_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, test_loader = get_loaders()

    inner_template = InnerNetAct2d(32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.net.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        # SwiGLU train to fork point
        swiglu_act = SwiGLUAct2d()
        swiglu_model = CNN(swiglu_act).to(device)
        opt_sw = optim.Adam(swiglu_model.parameters(), lr=args.lr)
        sw_acc = []
        for ep in range(args.fork_epoch):
            train_epoch(swiglu_model, train_loader, opt_sw, device)
            acc = evaluate(swiglu_model, test_loader, device)
            sw_acc.append(acc)
            if (ep+1) % 20 == 0: logger.info(f"  SwiGLU Ep {ep+1}: acc={acc*100:.2f}%")

        swiglu_state = copy.deepcopy(swiglu_model.state_dict())

        # Branch A: SwiGLU continues
        logger.info("--- Branch A: SwiGLU continues ---")
        sw_acc2 = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(swiglu_model, train_loader, opt_sw, device)
            acc = evaluate(swiglu_model, test_loader, device)
            sw_acc2.append(acc)
            if (ep+1) % 20 == 0: logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: acc={acc*100:.2f}%")
        best_sw = max(sw_acc + sw_acc2)

        # Branch B: InnerNet replaces
        logger.info("--- Branch B: InnerNet replaces ---")
        inner_act = InnerNetAct2d(32).to(device)
        inner_act.net.load_state_dict(fitted_weights)
        inner_model = CNN(inner_act).to(device)
        # Copy matching weights
        inn_dict = inner_model.state_dict()
        copied = 0
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v; copied += 1
        inner_model.load_state_dict(inn_dict, strict=False)
        logger.info(f"  Copied {copied} tensors")

        acc_swap = evaluate(inner_model, test_loader, device)
        logger.info(f"  After swap: acc={acc_swap*100:.2f}%")

        opt_in = optim.Adam(inner_model.parameters(), lr=args.lr)
        in_acc = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(inner_model, train_loader, opt_in, device)
            acc = evaluate(inner_model, test_loader, device)
            in_acc.append(acc)
            if (ep+1) % 20 == 0: logger.info(f"  InnerNet Ep {args.fork_epoch+ep+1}: acc={acc*100:.2f}%")
        best_in = max(in_acc)

        logger.info(f"  RESULT: SwiGLU={best_sw*100:.2f}% vs InnerNet={best_in*100:.2f}%")
        all_results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_bests = [r['best_sw']*100 for r in all_results]
    in_bests = [r['best_in']*100 for r in all_results]
    logger.info(f"\nSUMMARY: SwiGLU={np.mean(sw_bests):.2f}±{np.std(sw_bests):.2f} vs InnerNet={np.mean(in_bests):.2f}±{np.std(in_bests):.2f}")
    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)

if __name__ == '__main__':
    main()
