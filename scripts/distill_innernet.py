"""Distill a trained InnerNet (2 -> h -> 1 MLP) into closed-form operators.

Part of the "discover -> distill -> deploy" loop (PROJECT_STATUS P1).
A trained InnerNet IS the discovery; this script reads its checkpoint, samples
its learned 2D surface f(a, b), and fits several candidate closed-form operators
by least squares, reporting R^2 for each. The best fit becomes a fast fixed
operator that can be deployed in a fresh model (see deploy step).

Candidate families:
  - poly2 / poly3 : full bivariate polynomials (generic basis)
  - swiglu        : c0 + c1 * silu(a) * b          (tests SwiGLU rediscovery)
  - swiglu_sym    : c0 + c1*silu(a)*b + c2*silu(b)*a
  - mult          : c0 + c1 * a * b                (pure multiplicative gate)

Usage:
  .venv/bin/python scripts/distill_innernet.py \
      results/figures/v2_inner_weights_seed42_ivs_d128_v2.pth
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn


def load_innernet(path, hidden=32):
    """Rebuild a 2->hidden->1 ReLU MLP from a state dict (tolerates 'net.' prefix)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("net.", ""): v for k, v in sd.items()}
    hidden = sd["0.weight"].shape[0]
    net = nn.Sequential(
        nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
    )
    net.load_state_dict(sd)
    net.eval()
    return net


def sample_surface(net, lo=-5.0, hi=5.0, n=200):
    a = torch.linspace(lo, hi, n)
    b = torch.linspace(lo, hi, n)
    A, B = torch.meshgrid(a, b, indexing="ij")
    inp = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    with torch.no_grad():
        z = net(inp).reshape(-1).numpy()
    return A.numpy().reshape(-1), B.numpy().reshape(-1), z


def silu(x):
    return x / (1.0 + np.exp(-x))


def basis(name, a, b):
    """Return (design matrix columns, human-readable term names) for a family."""
    if name == "mult":
        cols = [np.ones_like(a), a * b]
        names = ["1", "a*b"]
    elif name == "swiglu":
        cols = [np.ones_like(a), silu(a) * b]
        names = ["1", "silu(a)*b"]
    elif name == "swiglu_sym":
        cols = [np.ones_like(a), silu(a) * b, silu(b) * a]
        names = ["1", "silu(a)*b", "silu(b)*a"]
    elif name == "poly2":
        cols = [np.ones_like(a), a, b, a * a, a * b, b * b]
        names = ["1", "a", "b", "a^2", "a*b", "b^2"]
    elif name == "poly3":
        cols = [np.ones_like(a), a, b, a * a, a * b, b * b,
                a**3, a*a*b, a*b*b, b**3]
        names = ["1", "a", "b", "a^2", "a*b", "b^2",
                 "a^3", "a^2*b", "a*b^2", "b^3"]
    else:
        raise ValueError(name)
    return np.stack(cols, axis=1), names


def fit(name, a, b, z):
    X, names = basis(name, a, b)
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    return {
        "family": name,
        "r2": r2,
        "rmse": rmse,
        "terms": {n: float(c) for n, c in zip(names, coef)},
    }


def formula(res):
    parts = []
    for term, c in res["terms"].items():
        if abs(c) < 1e-4:
            continue
        parts.append(f"{c:+.4f}*{term}" if term != "1" else f"{c:+.4f}")
    return " ".join(parts) or "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="InnerNet state-dict .pth")
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    net = load_innernet(args.ckpt)
    a, b, z = sample_surface(net, -args.range, args.range, args.n)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Sampled f(a,b) on [{-args.range},{args.range}]^2, "
          f"{args.n}x{args.n} grid | z: mean={z.mean():.3f} std={z.std():.3f}")
    print(f"{'family':<12} {'R^2':>8} {'RMSE':>8}   formula")
    print("-" * 78)

    results = []
    for fam in ["mult", "swiglu", "swiglu_sym", "poly2", "poly3"]:
        r = fit(fam, a, b, z)
        results.append(r)
        print(f"{fam:<12} {r['r2']:>8.4f} {r['rmse']:>8.4f}   f = {formula(r)}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"ckpt": args.ckpt, "results": results}, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
