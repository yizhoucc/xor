"""Network-independence probe for the SwiGLU attractor.

warmstart_free_init.py warm-starts the FFN network from a trained *SwiGLU*
model, leaving the confound: maybe the InnerNet converges to silu(a)*b only
because the host projections were optimised for SwiGLU. This script instead
warm-starts from a Bilinear-GLU network (gate = a*b, no activation) and drops in
a non-SwiGLU InnerNet. If the InnerNet still converges to silu(a)*b after
co-training, the SwiGLU form is a network-independent attractor.

Pipeline per seed:
  1. Train a BilinearGLUTransformer (a*b gate) for 20 epochs on WikiText-2.
  2. Take its network state (w1a/w1b/w2/attn/embed) and drop in an InnerNet FFN,
     InnerNet initialised as random / multiply (both non-SwiGLU).
  3. Co-train 10 epochs; save the resulting InnerNet weights for distillation.

Distill the saved weights with scripts/distill_innernet.py: silu(a)*b R^2 near
the free-init (SwiGLU-base) values => network-independent SwiGLU attractor.

Usage:
  .venv/bin/python scripts/warmstart_bilinear_probe.py --save_dir exp/bilinear_probe --num_seeds 3
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse the free-init helpers (InnerNetAct, fit_to_target, make_innernet_model,
# train_ep, evaluate_ppl, save_ckpt) and only swap the warm-start base model.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "warmstart_free_init", os.path.join(os.path.dirname(__file__), "warmstart_free_init.py"))
_wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="exp/bilinear_probe")
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--base_epochs", type=int, default=20)
    ap.add_argument("--probe_epochs", type=int, default=10)
    args = ap.parse_args()
    _wf.args = args  # save_ckpt reads args.save_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    from runner.lm_runner import WikiTextDataset
    from model.transformer import BilinearGLUTransformer

    train_ds = WikiTextDataset(split="train", context_size=64)
    val_ds = WikiTextDataset(split="validation", context_size=64, vocab=train_ds.vocab)
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    # Non-SwiGLU InnerNet inits (fitted once, deterministic given global seed below).
    torch.manual_seed(0)
    inner_random = _wf.InnerNetAct(32).to(device)
    inner_mult = _wf.InnerNetAct(32).to(device)
    _wf.fit_to_target(inner_mult, device, lambda a, b: a * b)
    init_weights = {
        "random": copy.deepcopy(inner_random.state_dict()),
        "multiply": copy.deepcopy(inner_mult.state_dict()),
    }

    for si, seed in enumerate(range(args.seed_start, args.seed_start + args.num_seeds)):
        print(f"\n[Seed {seed}] ({si+1}/{args.num_seeds})", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)

        # 1. Train Bilinear-GLU base (gate = a*b).
        base = BilinearGLUTransformer(vocab_size, 128, 4, 512, 4, 64, 0.1).to(device)
        opt = optim.Adam(base.parameters(), lr=5e-4)
        for ep in range(args.base_epochs):
            _wf.train_ep(base, train_loader, opt, device)
            if (ep + 1) % 5 == 0:
                print(f"  Bilinear base Ep {ep+1}: PPL={_wf.evaluate_ppl(base, val_loader, device):.2f}", flush=True)
        base_state = base.state_dict()

        # 2-3. Drop in each non-SwiGLU InnerNet init, co-train, save inner weights.
        res = _wf.run_lm_experiment(
            "wiki", base_state, vocab_size, init_weights,
            128, 4, 512, 4, 64, train_loader, val_loader, device,
            seed=seed, epochs=args.probe_epochs)
        for init_name in init_weights:
            print(f"  [{init_name}] best PPL after co-train: {res[init_name]['best']:.2f}", flush=True)

    print("\nDone. Distill exp/bilinear_probe/inner_*_wiki.pth with scripts/distill_innernet.py")


if __name__ == "__main__":
    main()
