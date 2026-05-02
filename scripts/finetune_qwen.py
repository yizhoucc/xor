"""Finetune Qwen2.5-0.5B with InnerNet replacing SwiGLU.

Qwen uses SwiGLU with gate_proj + up_proj + down_proj, same structure
as our InnerNet FFN. Direct weight copy, no workaround needed.

Phase 1: Finetune with SwiGLU (baseline) on SST-2
Phase 2: InnerNet replaces SwiGLU (warm-start), continue finetune
Also: evaluate WikiText-2 PPL for both

Usage:
  python scripts/finetune_qwen.py --save_dir exp/finetune_qwen
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


def fit_innernet_to_swiglu(inner_net, device, steps=3000):
    """Fit InnerNet to SiLU(a)*b."""
    opt = optim.Adam(inner_net.net.parameters(), lr=1e-3)
    a = torch.linspace(-5, 5, 200, device=device)
    b = torch.linspace(-5, 5, 200, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    targets = (F.silu(inputs[:, 0]) * inputs[:, 1]).unsqueeze(1)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(inner_net.net(inputs), targets)
        loss.backward()
        opt.step()
    logger.info(f"  Fitted InnerNet to SwiGLU, MSE={loss.item():.6f}")


def replace_swiglu_with_innernet(model, inner_net):
    """Replace SwiGLU in Qwen MLP with InnerNet.

    Qwen MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    InnerNet:  down_proj(InnerNet(gate_proj(x), up_proj(x)))
    """
    replaced = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        mlp.inner_net = inner_net

        def make_forward(mlp_ref):
            def new_forward(x):
                a = mlp_ref.gate_proj(x)
                b = mlp_ref.up_proj(x)
                pairs = torch.stack([a, b], dim=-1)
                B, S, D = a.shape
                activated = mlp_ref.inner_net.net(pairs.reshape(-1, 2)).view(B, S, D)
                return mlp_ref.down_proj(activated)
            return new_forward

        mlp.forward = make_forward(mlp)
        replaced += 1

    logger.info(f"  Replaced {replaced} MLP layers with InnerNet")
    return model


def eval_ppl(model, tokenizer, device, text_data, max_length=512, stride=256):
    """Evaluate perplexity on text data."""
    model.eval()
    encodings = tokenizer('\n\n'.join(text_data[:500]), return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    nlls = []
    for i in range(0, input_ids.size(1) - max_length, stride):
        begin = i
        end = min(i + max_length, input_ids.size(1))
        target_len = end - begin
        input_chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            nlls.append(outputs.loss.item())
        if len(nlls) >= 50:
            break
    import math
    return math.exp(np.mean(nlls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/finetune_qwen')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SST-2
    logger.info("Loading SST-2...")
    dataset = load_dataset('glue', 'sst2')

    def tokenize(split, max_n):
        texts = dataset[split]['sentence'][:max_n]
        labels = dataset[split]['label'][:max_n]
        enc = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=args.max_length, return_tensors='pt')
        return TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels))

    train_ds = tokenize('train', args.max_samples)
    val_ds = tokenize('validation', 872)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)

    # Load WikiText-2 for PPL eval
    logger.info("Loading WikiText-2 for PPL...")
    wiki = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    wiki_texts = [t for t in wiki['text'] if len(t.strip()) > 50]

    # Fit InnerNet to SwiGLU
    inner_template = InnerNetAct(32).to(device)
    fit_innernet_to_swiglu(inner_template, device)
    fitted_weights = inner_template.state_dict()

    seeds = list(range(42, 42 + args.num_seeds))
    all_results = []

    for si, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)})")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Phase 1: SwiGLU baseline finetune
        logger.info("--- Phase 1: SwiGLU baseline ---")
        model_sw = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(device)
        model_sw.config.pad_token_id = tokenizer.pad_token_id

        # Higher lr for classification head, lower for pretrained body
        head_params = [p for n, p in model_sw.named_parameters() if 'score' in n]
        body_params = [p for n, p in model_sw.named_parameters() if 'score' not in n]
        opt_sw = optim.AdamW([
            {'params': body_params, 'lr': args.lr},
            {'params': head_params, 'lr': args.lr * 10},
        ])
        sw_accs = []
        for ep in range(args.epochs):
            model_sw.train()
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                opt_sw.zero_grad()
                out = model_sw(ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                nn.utils.clip_grad_norm_(model_sw.parameters(), 1.0)
                opt_sw.step()

            model_sw.eval()
            correct = total = 0
            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    logits = model_sw(ids, attention_mask=mask).logits
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            sw_accs.append(acc)
            logger.info(f"  SwiGLU Ep {ep+1}: {acc*100:.2f}%")
        best_sw = max(sw_accs)
        sw_state = copy.deepcopy(model_sw.state_dict())

        # Eval SwiGLU PPL
        model_sw_lm = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            torch_dtype=torch.float16).to(device)
        sw_ppl = eval_ppl(model_sw_lm, tokenizer, device, wiki_texts)
        logger.info(f"  SwiGLU WikiText PPL: {sw_ppl:.2f}")
        del model_sw_lm

        # Phase 2: InnerNet warm-start
        logger.info("--- Phase 2: InnerNet warm-start ---")
        torch.manual_seed(seed)
        model_in = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(device)
        model_in.config.pad_token_id = tokenizer.pad_token_id
        model_in.load_state_dict(sw_state)

        shared_inner = InnerNetAct(32).to(device)
        shared_inner.load_state_dict(fitted_weights)
        shared_inner = shared_inner.to(torch.bfloat16)
        model_in = replace_swiglu_with_innernet(model_in, shared_inner)

        acc_swap = evaluate(model_in, val_loader, device)
        logger.info(f"  After swap: {acc_swap*100:.2f}% (SwiGLU was {sw_accs[-1]*100:.2f}%)")

        head_params_in = [p for n, p in model_in.named_parameters() if 'score' in n]
        body_params_in = [p for n, p in model_in.named_parameters() if 'score' not in n]
        opt_in = optim.AdamW([
            {'params': body_params_in, 'lr': args.lr},
            {'params': head_params_in, 'lr': args.lr * 10},
        ])
        in_accs = [acc_swap]
        for ep in range(args.epochs):
            model_in.train()
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                opt_in.zero_grad()
                out = model_in(ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                nn.utils.clip_grad_norm_(model_in.parameters(), 1.0)
                opt_in.step()

            model_in.eval()
            correct = total = 0
            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    logits = model_in(ids, attention_mask=mask).logits
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            in_accs.append(acc)
            logger.info(f"  InnerNet Ep {ep+1}: {acc*100:.2f}%")
        best_in = max(in_accs)

        logger.info(f"  SST-2: SwiGLU={best_sw*100:.2f}% vs InnerNet={best_in*100:.2f}%")
        logger.info(f"  WikiText PPL: SwiGLU={sw_ppl:.2f}")

        # Save InnerNet weights
        torch.save(shared_inner.state_dict(),
                   os.path.join(args.save_dir, f'inner_weights_seed{seed}.pth'))

        all_results.append({
            'seed': seed, 'best_sw': best_sw, 'best_in': best_in,
            'acc_swap': acc_swap, 'sw_ppl': sw_ppl,
            'sw_accs': sw_accs, 'in_accs': in_accs,
        })

    sw_b = [r['best_sw']*100 for r in all_results]
    in_b = [r['best_in']*100 for r in all_results]
    logger.info(f"\nSUMMARY SST-2: SwiGLU={np.mean(sw_b):.2f}±{np.std(sw_b):.2f} vs InnerNet={np.mean(in_b):.2f}±{np.std(in_b):.2f}")

    with open(os.path.join(args.save_dir, 'results.p'), 'wb') as f:
        pickle.dump({'all_results': all_results}, f)


if __name__ == '__main__':
    main()
