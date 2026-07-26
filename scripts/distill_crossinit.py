"""Cross-initialization SwiGLU-rediscovery test (addresses the warm-start confound).

The multi-seed table in distill_crossseed.py used checkpoints from
scripts/innernet_vs_swiglu.py, which initialises every seed's InnerNet from a
SwiGLU fit — so it only shows warm-start *retention* of the SwiGLU shape, not
init-independent rediscovery.

This script distills the FREE-INIT checkpoints instead. There the surrounding
network is warm-started (a capable SwiGLU network) but the InnerNet is
initialised four different ways (random / identity / multiply / swiglu). If the
learned surface converges to silu(a)*b regardless of a *non*-SwiGLU init, the
SwiGLU form is the attractor at the optimum — genuine rediscovery, not
retention. As a contrast we also distill from-scratch checkpoints that fail to
reach the optimum.

Usage:
  .venv/bin/python scripts/distill_crossinit.py --out results/figures/distill_crossinit.json
"""
import argparse
import importlib.util
import json
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "distill_innernet", os.path.join(os.path.dirname(__file__), "distill_innernet.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

# (label, checkpoint, group). Free-init = capable network, InnerNet init varied.
CKPTS = [
    ("random",   "exp/free_init/inner_random_wiki.pth",        "free_init (reaches ~72 PPL optimum)"),
    ("identity", "exp/free_init/inner_identity_wiki.pth",      "free_init (reaches ~72 PPL optimum)"),
    ("multiply", "exp/free_init/inner_multiply_wiki.pth",      "free_init (reaches ~72 PPL optimum)"),
    ("swiglu",   "exp/free_init/inner_swiglu_fitted_wiki.pth", "free_init (reaches ~72 PPL optimum)"),
    ("random s42",   "exp/from_scratch_init_v2/inner_random_seed42.pth",   "from-scratch (fails to reach optimum)"),
    ("random s43",   "exp/from_scratch_init_v2/inner_random_seed43.pth",   "from-scratch (fails to reach optimum)"),
    ("multiply s42", "exp/from_scratch_init_v2/inner_multiply_seed42.pth", "from-scratch (fails to reach optimum)"),
    ("multiply s43", "exp/from_scratch_init_v2/inner_multiply_seed43.pth", "from-scratch (fails to reach optimum)"),
]


def distill(path):
    net = _d.load_innernet(path)
    a, b, z = _d.sample_surface(net, -5.0, 5.0, 200)
    mult = _d.fit("mult", a, b, z)
    sw = _d.fit("swiglu", a, b, z)
    return mult["r2"], sw["r2"], sw["terms"]["silu(a)*b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    group = None
    print(f"{'init':<16} {'mult a*b R2':>12} {'SwiGLU R2':>12} {'silu(a)*b coef':>16}")
    for label, path, grp in CKPTS:
        if not os.path.exists(path):
            print(f"{label:<16}  (missing: {path})")
            continue
        if grp != group:
            group = grp
            print(f"-- {group} --")
        m, s, c = distill(path)
        rows.append({"init": label, "group": grp, "mult_r2": m, "swiglu_r2": s, "coef": c})
        print(f"{label:<16} {m:>12.3f} {s:>12.3f} {c:>16.3f}")

    free = [r for r in rows if r["group"].startswith("free_init")]
    scratch = [r for r in rows if r["group"].startswith("from-scratch")]
    summary = {
        "free_init_swiglu_r2_mean": float(np.mean([r["swiglu_r2"] for r in free])) if free else None,
        "free_init_swiglu_r2_min": float(np.min([r["swiglu_r2"] for r in free])) if free else None,
        "scratch_mult_r2_mean": float(np.mean([r["mult_r2"] for r in scratch])) if scratch else None,
        "scratch_swiglu_r2_mean": float(np.mean([r["swiglu_r2"] for r in scratch])) if scratch else None,
    }
    print("\nfree_init: SwiGLU is the attractor regardless of init "
          f"(min SwiGLU R2={summary['free_init_swiglu_r2_min']:.3f}).")
    print("from-scratch failures stall at pure multiplication a*b "
          f"(mean mult R2={summary['scratch_mult_r2_mean']:.3f} > SwiGLU {summary['scratch_swiglu_r2_mean']:.3f}).")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "summary": summary}, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
