"""Finetune a pretrained model with InnerNet activation replacement.

Takes a HuggingFace model (e.g., DistilBERT), replaces GELU/SwiGLU
in FFN with InnerNet, and finetunes on a downstream task.
Compares: original activation vs InnerNet warm-start.

Usage:
  python scripts/finetune_pretrained.py --save_dir exp/finetune_distilbert
"""
import os, sys, pickle, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class InnerNetAct(nn.Module):
    """Small MLP activation: f(a, b) -> scalar."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x)


def fit_innernet_to_gelu(inner_net, device, steps=2000):
    """Fit InnerNet to approximate GELU(a) (1D, b ignored)."""
    opt = optim.Adam(inner_net.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 500, device=device)
    # For GELU: InnerNet(a, 0) ≈ GELU(a), but we need 2 inputs
    # Strategy: fit f(a, b) = GELU(a) * sigmoid(b) as a soft gate
    # Actually simpler: just fit f(a, b) = GELU(a) for all b
    b = torch.zeros_like(a)
    inputs = torch.stack([a, b], dim=1)
    targets = F.gelu(a).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet to GELU, MSE={loss.item():.6f}")


def replace_ffn_activations(model, inner_net, model_type='distilbert'):
    """Replace GELU activations in FFN layers with InnerNet.

    For DistilBERT: each layer has ffn.lin1 (d->4d) and ffn.lin2 (4d->d)
    with GELU in between. We split lin1 output into pairs and apply InnerNet.
    """
    replaced = 0
    if model_type == 'distilbert':
        for layer in model.distilbert.transformer.layer:
            ffn = layer.ffn
            orig_lin1 = ffn.lin1  # Linear(768, 3072)
            orig_lin2 = ffn.lin2  # Linear(3072, 768)

            # Replace lin1 to output 2x width, add InnerNet
            d_model = orig_lin1.in_features  # 768
            d_ff = orig_lin1.out_features     # 3072

            # New lin1: d_model -> 2*d_ff (for pairing)
            new_lin1 = nn.Linear(d_model, d_ff * 2).to(orig_lin1.weight.device)
            # Initialize: first half from original, second half copy
            with torch.no_grad():
                new_lin1.weight[:d_ff] = orig_lin1.weight
                new_lin1.weight[d_ff:] = orig_lin1.weight
                new_lin1.bias[:d_ff] = orig_lin1.bias
                new_lin1.bias[d_ff:] = orig_lin1.bias

            ffn.lin1 = new_lin1
            ffn.inner_net = inner_net
            ffn.d_ff = d_ff

            # New lin2 stays the same (d_ff -> d_model)
            # Override forward
            original_forward = ffn.forward

            def make_new_forward(ffn_ref):
                def new_forward(x):
                    h = ffn_ref.lin1(x)  # [B, S, 2*d_ff]
                    B, S, D2 = h.shape
                    d_ff = D2 // 2
                    a, b = h[:, :, :d_ff], h[:, :, d_ff:]
                    pairs = torch.stack([a, b], dim=-1)  # [B, S, d_ff, 2]
                    activated = ffn_ref.inner_net.net(pairs.reshape(-1, 2)).view(B, S, d_ff)
                    return ffn_ref.dropout(ffn_ref.lin2(activated))
                return new_forward

            ffn.forward = make_new_forward(ffn)
            replaced += 1

    logger.info(f"  Replaced {replaced} FFN layers with InnerNet")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/finetune_distilbert')
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--task', default='sst2', help='sst2 or ag_news')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from datasets import load_dataset

    # Load data
    logger.info(f"Loading {args.task}...")
    if args.task == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        num_labels = 2
        text_key = 'sentence'
    elif args.task == 'ag_news':
        dataset = load_dataset('ag_news')
        num_labels = 4
        text_key = 'text'

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

    def tokenize(split):
        texts = dataset[split][text_key][:5000]  # limit for speed
        labels = dataset[split]['label'][:5000]
        enc = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=args.max_length, return_tensors='pt')
        return TensorDataset(enc['input_ids'], enc['attention_mask'],
                            torch.tensor(labels))

    train_ds = tokenize('train')
    val_ds = tokenize('validation' if args.task == 'sst2' else 'test')
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)

    # Fit InnerNet
    inner_template = InnerNetAct(32).to(device)
    fit_innernet_to_gelu(inner_template, device)
    fitted_weights = inner_template.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Baseline: standard finetune
        logger.info("--- Baseline: original GELU ---")
        model_base = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels).to(device)
        opt_base = optim.AdamW(model_base.parameters(), lr=args.lr)

        base_accs = []
        for ep in range(args.epochs):
            model_base.train()
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                opt_base.zero_grad()
                out = model_base(ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                opt_base.step()

            model_base.eval()
            correct = total = 0
            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    logits = model_base(ids, attention_mask=mask).logits
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            base_accs.append(acc)
            logger.info(f"  GELU Ep {ep+1}: acc={acc*100:.2f}%")
        best_base = max(base_accs)

        # InnerNet: replace and finetune
        logger.info("--- InnerNet: replace GELU ---")
        torch.manual_seed(seed)
        model_inner = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels).to(device)

        shared_inner = InnerNetAct(32).to(device)
        shared_inner.load_state_dict(fitted_weights)
        model_inner = replace_ffn_activations(model_inner, shared_inner, 'distilbert')

        opt_inner = optim.AdamW(model_inner.parameters(), lr=args.lr)

        inner_accs = []
        for ep in range(args.epochs):
            model_inner.train()
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                opt_inner.zero_grad()
                out = model_inner(ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                opt_inner.step()

            model_inner.eval()
            correct = total = 0
            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    logits = model_inner(ids, attention_mask=mask).logits
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            inner_accs.append(acc)
            logger.info(f"  InnerNet Ep {ep+1}: acc={acc*100:.2f}%")
        best_inner = max(inner_accs)

        logger.info(f"  RESULT: GELU={best_base*100:.2f}% vs InnerNet={best_inner*100:.2f}%")
        all_results.append({'seed': seed, 'best_base': best_base, 'best_inner': best_inner})

    base_b = [r['best_base']*100 for r in all_results]
    inner_b = [r['best_inner']*100 for r in all_results]
    logger.info(f"\nSUMMARY: GELU={np.mean(base_b):.2f}±{np.std(base_b):.2f} vs InnerNet={np.mean(inner_b):.2f}±{np.std(inner_b):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
