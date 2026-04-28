"""Finetune pretrained DistilBERT with InnerNet activation.

Proper warm-start: duplicate lin1 → lin1a + lin1b, fit InnerNet to
f(a,a) ≈ GELU(a), replace. lin2 dimensions unchanged.

Phase 1: Finetune with GELU (baseline)
Phase 2: Finetune with InnerNet (warm-start from GELU checkpoint)

Usage:
  python scripts/finetune_pretrained.py --save_dir exp/finetune_distilbert
"""
import os, sys, copy, pickle, random, logging, argparse
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
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def fit_innernet_to_gelu_paired(inner_net, device, steps=2000):
    """Fit InnerNet so f(a, a) ≈ GELU(a)."""
    opt = optim.Adam(inner_net.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 500, device=device)
    inputs = torch.stack([a, a], dim=1)  # (a, a) pairs
    targets = F.gelu(a).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet: f(a,a)≈GELU(a), MSE={loss.item():.6f}")


def replace_ffn_with_innernet(model, inner_net):
    """Replace GELU in DistilBERT FFN with InnerNet.

    For each FFN layer:
    - Keep lin1 as lin1a
    - Add lin1b = copy of lin1 (same init)
    - Replace GELU with InnerNet(lin1a(x), lin1b(x))
    - lin2 unchanged (d_ff → d_model, same dims)
    """
    replaced = 0
    for layer in model.distilbert.transformer.layer:
        ffn = layer.ffn
        d_ff = ffn.lin1.out_features  # 3072

        # Duplicate lin1 → lin1a (original) + lin1b (copy)
        ffn.lin1a = ffn.lin1  # keep original
        ffn.lin1b = copy.deepcopy(ffn.lin1)  # copy
        ffn.inner_net = inner_net
        ffn._d_ff = d_ff

        def make_forward(ffn_ref):
            def new_forward(x):
                a = ffn_ref.lin1a(x)  # [B, S, d_ff]
                b = ffn_ref.lin1b(x)  # [B, S, d_ff]
                pairs = torch.stack([a, b], dim=-1)  # [B, S, d_ff, 2]
                B, S, D, _ = pairs.shape
                activated = ffn_ref.inner_net.net(pairs.reshape(-1, 2)).view(B, S, D)
                return ffn_ref.dropout(ffn_ref.lin2(activated))
            return new_forward

        ffn.forward = make_forward(ffn)
        replaced += 1

    logger.info(f"  Replaced {replaced} FFN layers")
    return model


def train_epoch(model, loader, optimizer, device):
    model.train()
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, attention_mask=mask).logits
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/finetune_distilbert_v2')
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--task', default='sst2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from datasets import load_dataset

    logger.info(f"Loading {args.task}...")
    if args.task == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        num_labels = 2
        text_key = 'sentence'
        val_split = 'validation'
    elif args.task == 'ag_news':
        dataset = load_dataset('ag_news')
        num_labels = 4
        text_key = 'text'
        val_split = 'test'

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

    def tokenize(split):
        texts = dataset[split][text_key][:args.max_samples]
        labels = dataset[split]['label'][:args.max_samples]
        enc = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=args.max_length, return_tensors='pt')
        return TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels))

    train_ds = tokenize('train')
    val_ds = tokenize(val_split)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)

    # Fit InnerNet: f(a,a) ≈ GELU(a)
    inner_template = InnerNetAct(32).to(device)
    fit_innernet_to_gelu_paired(inner_template, device)
    fitted_weights = inner_template.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Phase 1: GELU baseline
        logger.info("--- GELU baseline ---")
        model_gelu = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels).to(device)
        opt_gelu = optim.AdamW(model_gelu.parameters(), lr=args.lr)

        gelu_accs = []
        for ep in range(args.epochs):
            train_epoch(model_gelu, train_loader, opt_gelu, device)
            acc = evaluate(model_gelu, val_loader, device)
            gelu_accs.append(acc)
            logger.info(f"  GELU Ep {ep+1}: {acc*100:.2f}%")
        best_gelu = max(gelu_accs)

        # Save GELU checkpoint for warm-start
        gelu_state = copy.deepcopy(model_gelu.state_dict())

        # Phase 2: InnerNet warm-start from GELU checkpoint
        logger.info("--- InnerNet warm-start ---")
        torch.manual_seed(seed)
        model_inner = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels).to(device)
        model_inner.load_state_dict(gelu_state)  # start from finetuned GELU

        shared_inner = InnerNetAct(32).to(device)
        shared_inner.load_state_dict(fitted_weights)
        model_inner = replace_ffn_with_innernet(model_inner, shared_inner)

        # Verify: should be close to GELU performance
        acc_swap = evaluate(model_inner, val_loader, device)
        logger.info(f"  After swap: {acc_swap*100:.2f}% (GELU was {gelu_accs[-1]*100:.2f}%)")

        opt_inner = optim.AdamW(model_inner.parameters(), lr=args.lr)
        inner_accs = [acc_swap]
        for ep in range(args.epochs):
            train_epoch(model_inner, train_loader, opt_inner, device)
            acc = evaluate(model_inner, val_loader, device)
            inner_accs.append(acc)
            logger.info(f"  InnerNet Ep {ep+1}: {acc*100:.2f}%")
        best_inner = max(inner_accs)

        logger.info(f"  RESULT: GELU={best_gelu*100:.2f}% vs InnerNet={best_inner*100:.2f}%")
        all_results.append({'seed': seed, 'best_gelu': best_gelu, 'best_inner': best_inner,
                           'acc_swap': acc_swap, 'gelu_accs': gelu_accs, 'inner_accs': inner_accs})

    g = [r['best_gelu']*100 for r in all_results]
    i = [r['best_inner']*100 for r in all_results]
    logger.info(f"\nSUMMARY: GELU={np.mean(g):.2f}±{np.std(g):.2f} vs InnerNet={np.mean(i):.2f}±{np.std(i):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
