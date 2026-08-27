"""Unified causal experiment for the SwiGLU attractor (Codex proposals #1 + #2).

One script covers:
  #1 network-independence + frozen/joint causality: host = Bilinear-GLU (a*b),
     frozen vs joint outer training, InnerNet init random/multiply.
  #2 cross-init x cross-seed: host = SwiGLU, joint, inits random/identity/
     multiply/swiglu, one seed per job (per-seed files, no overwrite).

For each (host, freeze, init) it warm-starts the host network, drops in the
InnerNet, trains, and saves the InnerNet weights per epoch (trajectory) and a
final copy tagged with the seed. Distill the saved weights with
scripts/distill_innernet.py.

Causal reading (host=bilinear):
  - frozen  -> InnerNet fits the fixed a*b-tuned projections (expect ~a*b).
  - joint   -> if it moves to silu(a)*b with better PPL, SwiGLU is discovered by
               task optimization, not imitation of the host gate.

Usage (one seed per job, fan out with --seed):
  python scripts/warmstart_causal.py --host bilinear --freeze 1 --inits random,multiply \
      --seed 42 --save_dir exp/causal/bilinear_frozen_s42
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "warmstart_free_init", os.path.join(os.path.dirname(__file__), "warmstart_free_init.py"))
_wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wf)


INIT_SEED_OFFSETS = {
    "random": 101,
    "identity": 211,
    "multiply": 307,
    "swiglu": 401,
}


def build_inits(names, device, seed):
    """Build each InnerNet init from a dedicated deterministic RNG stream.

    This keeps a condition identical whether it is run alone or grouped with
    other initializations in the same Slurm job.
    """
    import torch.nn.functional as F
    out = {}
    for n in names:
        torch.manual_seed(seed * 1000 + INIT_SEED_OFFSETS[n])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 1000 + INIT_SEED_OFFSETS[n])
        net = _wf.InnerNetAct(32).to(device)
        if n == "multiply":
            _wf.fit_to_target(net, device, lambda a, b: a * b)
        elif n == "identity":
            _wf.fit_to_target(net, device, lambda a, b: a)
        elif n == "swiglu":
            _wf.fit_to_target(net, device, lambda a, b: F.silu(a) * b)
        elif n == "random":
            pass
        else:
            raise ValueError(n)
        out[n] = copy.deepcopy(net.state_dict())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", choices=["swiglu", "bilinear"], required=True)
    ap.add_argument("--freeze", type=int, default=0, help="1 = freeze host network, train only InnerNet")
    ap.add_argument("--inits", default="random,multiply")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_epochs", type=int, default=20)
    ap.add_argument("--probe_epochs", type=int, default=10)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--host_checkpoint", default=None,
                    help="Load/save reusable host checkpoint (.pth).")
    ap.add_argument("--host_only", action="store_true",
                    help="Train/save the host and exit before InnerNet probes.")
    args = ap.parse_args()
    _wf.args = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    from runner.lm_runner import WikiTextDataset
    from model.transformer import SwiGLUTransformer, BilinearGLUTransformer

    train_ds = WikiTextDataset(split="train", context_size=64)
    val_ds = WikiTextDataset(split="validation", context_size=64, vocab=train_ds.vocab)
    V = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=4)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 1. Train or load the reusable host network.
    Host = SwiGLUTransformer if args.host == "swiglu" else BilinearGLUTransformer
    host = Host(V, 128, 4, 512, 4, 64, 0.1).to(device)
    if args.host_checkpoint and os.path.exists(args.host_checkpoint):
        payload = torch.load(args.host_checkpoint, map_location=device, weights_only=False)
        host.load_state_dict(payload["host_state_dict"])
        host_ppl = float(payload["host_ppl"])
        print(f"  loaded host checkpoint {args.host_checkpoint} PPL={host_ppl:.2f}", flush=True)
    else:
        opt = optim.Adam(host.parameters(), lr=5e-4)
        for ep in range(args.base_epochs):
            _wf.train_ep(host, train_loader, opt, device)
            if (ep + 1) % 5 == 0:
                ppl = _wf.evaluate_ppl(host, val_loader, device)
                print(f"  {args.host} host Ep {ep+1}: PPL={ppl:.2f}", flush=True)
        host_ppl = _wf.evaluate_ppl(host, val_loader, device)
        print(f"  host final PPL={host_ppl:.2f}", flush=True)
        if args.host_checkpoint:
            os.makedirs(os.path.dirname(args.host_checkpoint), exist_ok=True)
            torch.save({
                "host_state_dict": host.state_dict(),
                "host_ppl": host_ppl,
                "host": args.host,
                "seed": args.seed,
                "base_epochs": args.base_epochs,
            }, args.host_checkpoint)
            print(f"  saved host checkpoint {args.host_checkpoint}", flush=True)

    if args.host_only:
        print("Host-only job done.", flush=True)
        return

    host_state = host.state_dict()
    inits = build_inits(args.inits.split(","), device, args.seed)

    # 2. Per init: drop in InnerNet, train (frozen or joint), save per-epoch + final.
    results = {
        "host": args.host,
        "freeze": bool(args.freeze),
        "seed": args.seed,
        "host_checkpoint": args.host_checkpoint,
        "host_ppl": host_ppl,
        "probe_epochs": args.probe_epochs,
        "conditions": {},
    }
    for init_name, inner_state in inits.items():
        inner = _wf.InnerNetAct(32).to(device)
        inner.load_state_dict(inner_state)
        model = _wf.make_innernet_model(host_state, inner, V, 128, 4, 512, 4, 64, device)

        if args.freeze:
            for p in model.parameters():
                p.requires_grad = False
            for blk in model.blocks:
                for p in blk.ffn.inner_net.parameters():
                    p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = model.parameters()
        opt2 = optim.Adam(params, lr=5e-4)

        tag = f"{args.host}_{'frozen' if args.freeze else 'joint'}_{init_name}"
        final_path = os.path.join(args.save_dir, f"inner_{tag}_seed{args.seed}.pth")
        if os.path.exists(final_path):
            print(f"  [{tag}] final checkpoint exists; skipping", flush=True)
            results["conditions"][init_name] = {"status": "existing", "checkpoint": final_path}
            continue
        ppls = []
        for ep in range(args.probe_epochs):
            _wf.train_ep(model, train_loader, opt2, device)
            ppl = _wf.evaluate_ppl(model, val_loader, device)
            ppls.append(ppl)
            # per-epoch InnerNet weights for the a*b -> silu trajectory
            torch.save(model.blocks[0].ffn.inner_net.state_dict(),
                       os.path.join(args.save_dir, f"inner_{tag}_seed{args.seed}_ep{ep+1:02d}.pth"))
        torch.save(model.blocks[0].ffn.inner_net.state_dict(), final_path)
        results["conditions"][init_name] = {
            "status": "completed",
            "ppl": ppls,
            "best_ppl": min(ppls),
            "checkpoint": final_path,
        }
        print(f"  [{tag}] best PPL={min(ppls):.2f} (host {host_ppl:.2f})", flush=True)

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
