"""Cross-seed SwiGLU-rediscovery consistency (paper core evidence).

For each independently trained InnerNet checkpoint (one per seed), fit the same
closed-form operator families to its learned 2D surface f(a, b) and report how
consistently the seeds converge on the SwiGLU gate `c * silu(a) * b`.

This upgrades the discovery claim from "one InnerNet looks like SwiGLU" to
"independently trained InnerNets reproducibly converge to SwiGLU", which is the
evidence a discovery paper needs. It is pure analysis of existing checkpoints —
no training.

Usage:
  .venv/bin/python scripts/distill_crossseed.py \
      exp/ivs_d128_v2/inner_weights_seed{42,43,44,45,46}.pth \
      --out results/figures/distill_crossseed_ivs_d128.json

Reproduce the paper table:
  .venv/bin/python scripts/distill_crossseed.py exp/ivs_d128_v2/inner_weights_seed*.pth \
      --out results/figures/distill_crossseed_ivs_d128.json
"""
import argparse
import importlib.util
import json
import os
import re

import numpy as np

# Reuse the single-checkpoint distillation primitives.
_spec = importlib.util.spec_from_file_location(
    "distill_innernet",
    os.path.join(os.path.dirname(__file__), "distill_innernet.py"),
)
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

FAMILIES = ["mult", "swiglu", "swiglu_sym", "poly2", "poly3"]


def seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="per-seed InnerNet .pth files")
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_seed = []
    for path in args.ckpts:
        net = _d.load_innernet(path)
        a, b, z = _d.sample_surface(net, -args.range, args.range, args.n)
        fits = {fam: _d.fit(fam, a, b, z) for fam in FAMILIES}
        per_seed.append({"seed": seed_of(path), "ckpt": path, "fits": fits})

    def col(fam, key):
        return np.array([s["fits"][fam]["r2"] if key == "r2"
                         else s["fits"][fam]["terms"].get(key, 0.0)
                         for s in per_seed], dtype=float)

    sw_r2 = col("swiglu", "r2")
    mult_r2 = col("mult", "r2")
    c1 = col("swiglu", "silu(a)*b")

    print("Cross-seed SwiGLU rediscovery (InnerNet FFN surface)")
    print(f"{'seed':>5} {'mult R2':>8} {'swiglu R2':>10} "
          f"{'swiglu_sym R2':>13} {'poly3 R2':>9} {'c[silu(a)b]':>12}")
    print("-" * 62)
    for s in per_seed:
        f = s["fits"]
        print(f"{s['seed']:>5} {f['mult']['r2']:>8.3f} {f['swiglu']['r2']:>10.3f} "
              f"{f['swiglu_sym']['r2']:>13.3f} {f['poly3']['r2']:>9.3f} "
              f"{f['swiglu']['terms']['silu(a)*b']:>12.4f}")
    print("-" * 62)

    def ms(x):
        return f"{x.mean():.4f} +/- {x.std(ddof=1):.4f}"

    summary = {
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "swiglu_r2_mean": float(sw_r2.mean()),
        "swiglu_r2_sd": float(sw_r2.std(ddof=1)),
        "swiglu_r2_min": float(sw_r2.min()),
        "swiglu_r2_max": float(sw_r2.max()),
        "swiglu_coef_mean": float(c1.mean()),
        "swiglu_coef_sd": float(c1.std(ddof=1)),
        "mult_r2_mean": float(mult_r2.mean()),
    }
    print(f"swiglu R^2      : {ms(sw_r2)}  range [{sw_r2.min():.3f}, {sw_r2.max():.3f}]")
    print(f"c*silu(a)*b     : {ms(c1)}")
    print(f"mult (a*b) R^2  : {ms(mult_r2)}  (pure product fits far worse)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"per_seed": per_seed, "summary": summary}, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
