"""LSTM warm-start: SwiGLU vs InnerNet on WikiText-2."""
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


class SwiGLULSTMCell(nn.Module):
    """LSTM cell with SwiGLU for cell candidate: SiLU(W_a@[x,h]) * W_b@[x,h]."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.cell_a = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_b = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        gates = self.gate_linear(combined)
        i, f, o = gates.chunk(3, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = F.silu(self.cell_a(combined)) * self.cell_b(combined)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class InnerNetLSTMCell(nn.Module):
    """LSTM cell with InnerNet for cell candidate."""
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.cell_a = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.inner_net = nn.Sequential(nn.Linear(2, inner_hidden), nn.ReLU(), nn.Linear(inner_hidden, 1))

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        gates = self.gate_linear(combined)
        i, f, o = gates.chunk(3, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        a = self.cell_a(combined)
        b = self.cell_b(combined)
        pairs = torch.stack([a, b], dim=-1)  # [B, H, 2]
        g = self.inner_net(pairs.reshape(-1, 2)).view(x.size(0), self.hidden_size)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, cell):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = cell
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.fc.weight = self.embedding.weight

    def forward(self, x):
        emb = self.embedding(x)  # [B, S, E]
        B, S, E = emb.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(S):
            h, (h, c) = self.cell(emb[:, t], (h, c))
        return self.fc(h)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, n = 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
            n += 1
    return math.exp(total_loss / n)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def fit_innernet_to_swiglu(inner_net, device, steps=2000):
    opt = optim.Adam(inner_net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (F.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet, MSE={loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/warmstart_lstm')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--fork_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--context_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    train_ds = WikiTextDataset(split='train', context_size=args.context_size)
    val_ds = WikiTextDataset(split='validation', context_size=args.context_size, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    # Fit InnerNet to SwiGLU
    inner_template = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # SwiGLU LSTM
        sw_cell = SwiGLULSTMCell(args.embed_dim, args.hidden_dim)
        sw_model = LSTMModel(vocab_size, args.embed_dim, args.hidden_dim, sw_cell).to(device)
        opt_sw = optim.Adam(sw_model.parameters(), lr=args.lr)
        sw_ppl = []
        for ep in range(args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            ppl = evaluate(sw_model, val_loader, device)
            sw_ppl.append(ppl)
            logger.info(f"  SwiGLU Ep {ep+1}: PPL={ppl:.2f}")

        swiglu_state = copy.deepcopy(sw_model.state_dict())

        # Branch A: SwiGLU continues
        sw_ppl2 = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(sw_model, train_loader, opt_sw, device)
            ppl = evaluate(sw_model, val_loader, device)
            sw_ppl2.append(ppl)
            logger.info(f"  SwiGLU Ep {args.fork_epoch+ep+1}: PPL={ppl:.2f}")
        best_sw = min(sw_ppl + sw_ppl2)

        # Branch B: InnerNet replaces
        in_cell = InnerNetLSTMCell(args.embed_dim, args.hidden_dim)
        in_model = LSTMModel(vocab_size, args.embed_dim, args.hidden_dim, in_cell).to(device)
        # Copy weights
        inn_dict = in_model.state_dict()
        for k, v in swiglu_state.items():
            if k in inn_dict and v.shape == inn_dict[k].shape:
                inn_dict[k] = v
        in_model.load_state_dict(inn_dict, strict=False)
        in_model.cell.inner_net.load_state_dict(fitted_weights)

        ppl_swap = evaluate(in_model, val_loader, device)
        logger.info(f"  After swap: PPL={ppl_swap:.2f}")

        opt_in = optim.Adam(in_model.parameters(), lr=args.lr)
        in_ppl = []
        for ep in range(args.epochs - args.fork_epoch):
            train_epoch(in_model, train_loader, opt_in, device)
            ppl = evaluate(in_model, val_loader, device)
            in_ppl.append(ppl)
            logger.info(f"  InnerNet Ep {args.fork_epoch+ep+1}: PPL={ppl:.2f}")
        best_in = min(in_ppl)

        logger.info(f"  RESULT: SwiGLU={best_sw:.2f} vs InnerNet={best_in:.2f}")
        all_results.append({'seed': seed, 'best_sw': best_sw, 'best_in': best_in})

    sw_b = [r['best_sw'] for r in all_results]
    in_b = [r['best_in'] for r in all_results]
    logger.info(f"\nSUMMARY: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs InnerNet={np.mean(in_b):.2f}±{np.std(in_b):.2f}")
    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)

if __name__ == '__main__':
    main()
