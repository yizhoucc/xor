"""Cross-seed gate-rediscovery analysis for Sequential-MNIST SeqGatedRNN.

The Plan-B recurrent cell has two InnerNets:
  - inner_net1(a, b): a = recurrent signal, b = input   -> hypothesised input gate
  - inner_net2(c, a): c = cell-state projection, a = recurrent -> hypothesised output gate

We sample each learned 2D surface f(x0, x1) (inputs are LayerNorm'd, so we use a
[-3, 3]^2 grid matching their operating range) and least-squares fit gate-shaped
operators to test whether the network converged on sigmoid/silu gating rather
than a plain product or a linear map — and whether it does so consistently
across the independently trained successful seeds.

Usage:
  .venv/bin/python scripts/gate_crossseed.py \
      exp/seq_mnist_gated_rnn_long_20260515_174406_cc6a1c88 42 43 46 \
      --out results/figures/gate_crossseed_seqmnist.json
"""
import argparse
import json
import os

import numpy as np
import torch

from model.seq_rnn import InnerNetActivation


def load_inner(state, prefix, hidden=32):
    net = InnerNetActivation(hidden_dim=hidden)
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    net.load_state_dict(sub)
    net.eval()
    return net


def sample(net, lo=-3.0, hi=3.0, n=200):
    g = torch.linspace(lo, hi, n)
    A, B = torch.meshgrid(g, g, indexing="ij")
    inp = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    with torch.no_grad():
        z = net(inp).reshape(-1).numpy()
    return A.numpy().reshape(-1), B.numpy().reshape(-1), z


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x / (1.0 + np.exp(-x))


FAMILIES = {
    "linear":   lambda a, b: ([np.ones_like(a), a, b], ["1", "x0", "x1"]),
    "mult":     lambda a, b: ([np.ones_like(a), a * b], ["1", "x0*x1"]),
    "gate01":   lambda a, b: ([np.ones_like(a), sig(a) * b], ["1", "sig(x0)*x1"]),
    "gate10":   lambda a, b: ([np.ones_like(a), sig(b) * a], ["1", "sig(x1)*x0"]),
    "silu01":   lambda a, b: ([np.ones_like(a), silu(a) * b], ["1", "silu(x0)*x1"]),
    "silu10":   lambda a, b: ([np.ones_like(a), silu(b) * a], ["1", "silu(x1)*x0"]),
}


def fit(fam, a, b, z):
    cols, names = FAMILIES[fam](a, b)
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r2": r2, "terms": {n: float(c) for n, c in zip(names, coef)}}


def analyse(net):
    a, b, z = sample(net)
    fits = {fam: fit(fam, a, b, z) for fam in FAMILIES}
    # best sigmoid-gate orientation
    best_gate = max(["gate01", "gate10"], key=lambda f: fits[f]["r2"])
    return fits, best_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {"run_dir": args.run_dir, "seeds": args.seeds, "inner_net1": [], "inner_net2": []}
    for which, prefix in [("inner_net1", "cell.inner_net1."),
                          ("inner_net2", "cell.inner_net2.")]:
        print(f"\n=== {which} ({'input gate' if which.endswith('1') else 'output gate'} hypothesis) ===")
        print(f"{'seed':>5} {'linear':>7} {'mult':>7} {'gate':>7} {'silu':>7}  best-gate term (coef)")
        print("-" * 74)
        gate_r2s = []
        for s in args.seeds:
            sd = torch.load(os.path.join(args.run_dir, f"best_model_seed{s}.pth"),
                            map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            net = load_inner(sd, prefix)
            fits, bg = analyse(net)
            gate_r2 = fits[bg]["r2"]
            gate_r2s.append(gate_r2)
            silu_best = max(fits["silu01"]["r2"], fits["silu10"]["r2"])
            term = list(fits[bg]["terms"].items())[1]
            print(f"{s:>5} {fits['linear']['r2']:>7.3f} {fits['mult']['r2']:>7.3f} "
                  f"{gate_r2:>7.3f} {silu_best:>7.3f}  {term[0]} ({term[1]:+.3f})")
            out[which].append({"seed": s, "fits": fits, "best_gate": bg})
        g = np.array(gate_r2s)
        print(f"gate R^2: mean={g.mean():.4f} sd={g.std(ddof=1):.4f} range=[{g.min():.3f},{g.max():.3f}]")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
