"""AE warm-start: SwiGLU vs InnerNet on MNIST."""
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



class InnerNetAct1d(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        # x: [B, 2*D] → split → InnerNet → [B, D]
        D = x.size(1) // 2
        a, b = x[:, :D], x[:, D:]
        pairs = torch.stack([a, b], dim=-1)  # [B, D, 2]
        return self.net(pairs.reshape(-1, 2)).view(x.size(0), D)


class SwiGLUAct1d(nn.Module):
    def forward(self, x):
        D = x.size(1) // 2
        return F.silu(x[:, :D]) * x[:, D:]


class AE(nn.Module):
    def __init__(self, act, input_dim=784, enc_dims=[512, 256], latent=32):
        super().__init__()
        # Encoder: each layer outputs 2× for gating
        enc_layers = []
        in_d = input_dim
        for d in enc_dims:
            enc_layers += [nn.Linear(in_d, d*2), nn.LayerNorm(d*2), act, nn.Dropout(0.5)]
            in_d = d
        enc_layers.append(nn.Linear(in_d, latent))
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder: mirror, uses ReLU (no gating)
        dec_layers = []
        in_d = latent
        for d in reversed(enc_dims):
            dec_layers += [nn.Linear(in_d, d), nn.ReLU(), nn.Dropout(0.5)]
            in_d = d
        dec_layers.append(nn.Linear(in_d, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def get_loaders(batch_size=128):
    transform = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(train_ds, batch_size, shuffle=True, num_workers=4), DataLoader(test_ds, batch_size, shuffle=False, num_workers=4)


def train_epoch(model, loader, optimizer, device):
    model.train()
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), x)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)
            total_loss += nn.MSELoss()(model(x), x).item()
            n += 1
    return total_loss / n


def fit_innernet_to_swiglu(inner, device, steps=2000):
    opt = optim.Adam(inner.net.parameters(), lr=1e-3)
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
    parser.add_argument('--save_dir', default='exp/warmstart_ae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fork_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, test_loader = get_loaders()

    inner_template = InnerNetAct1d(32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.net.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        swiglu_model = AE(SwiGLUAct1d()).to(device)
        opt_sw = optim.Adam(swiglu_model.parameters(), lr=args.lr)
        sw_mse = []
        for ep in range(args.fork_epoch):
            train_epoch(swiglu_model, train_loader, opt_sw, device)
            mse = evaluate(swiglu_model, test_loader, device)
            sw_mse.append(mse)
            save_ckpt(swiglu_model, 'warmstart_ae', 'swiglu_phase1', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {ep+1}: MSE={mse:.6f}")

        swiglu_state = copy.deepcopy(swiglu_model.state_dict())

        # Branch A
        sw_mse2 = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(swiglu_model, train_loader, opt_sw, device)
            mse = evaluate(swiglu_model, test_loader, device)
            sw_mse2.append(mse)
            save_ckpt(swiglu_model, 'warmstart_ae', 'swiglu_phase2', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: MSE={mse:.6f}")
        best_sw = min(sw_mse + sw_mse2)

        # Branch B
        inner_act = InnerNetAct1d(32).to(device)
        inner_act.net.load_state_dict(fitted_weights)
        inner_model = AE(inner_act).to(device)
        inn_dict = inner_model.state_dict()
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v
        inner_model.load_state_dict(inn_dict, strict=False)

        mse_swap = evaluate(inner_model, test_loader, device)
        logger.info(f"  After swap: MSE={mse_swap:.6f}")

        opt_in = optim.Adam(inner_model.parameters(), lr=args.lr)
        in_mse = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(inner_model, train_loader, opt_in, device)
            mse = evaluate(inner_model, test_loader, device)
            in_mse.append(mse)
            save_ckpt(inner_model, 'warmstart_ae', 'innernet', seed, ep+1)
            if (ep+1) % 10 == 0: logger.info(f"  InnerNet Ep {args.fork_epoch+ep+1}: MSE={mse:.6f}")
        best_in = min(in_mse)

        logger.info(f"  RESULT: SwiGLU={best_sw:.6f} vs InnerNet={best_in:.6f}")
        all_results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw'] for r in all_results]
    in_b = [r['best_in'] for r in all_results]
    logger.info(f"\nSUMMARY: SwiGLU={np.mean(sw_b):.6f}±{np.std(sw_b):.6f} vs InnerNet={np.mean(in_b):.6f}±{np.std(in_b):.6f}")
    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)

if __name__ == '__main__':
    main()
