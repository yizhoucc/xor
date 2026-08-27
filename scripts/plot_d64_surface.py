"""Visualize the post-sharing d=64 InnerNet and its best SwiGLU fit."""
import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_spec = importlib.util.spec_from_file_location(
    "distill_innernet",
    os.path.join(os.path.dirname(__file__), "distill_innernet.py"),
)
_distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill)


def build_surfaces(checkpoint, n=200, surface_range=4.0):
    net = _distill.load_innernet(checkpoint)
    a, b, learned = _distill.sample_surface(net, -surface_range, surface_range, n)
    fit = _distill.fit("swiglu", a, b, learned)
    design, names = _distill.basis("swiglu", a, b)
    coefficients = np.asarray([fit["terms"][name] for name in names])
    fitted = design @ coefficients
    return {
        "a": a.reshape(n, n),
        "b": b.reshape(n, n),
        "learned": learned.reshape(n, n),
        "fitted": fitted.reshape(n, n),
        "residual": (learned - fitted).reshape(n, n),
        "fit": fit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/figures/v2_inner_weights_seed42_ivs_d64_v2.pth",
    )
    parser.add_argument("--output", default="results/figures/fig_d64_swiglu_surface")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--range", type=float, default=5.0)
    args = parser.parse_args()

    surfaces = build_surfaces(args.checkpoint, args.n, args.range)
    a, b = surfaces["a"], surfaces["b"]
    vmax = max(abs(surfaces["learned"]).max(), abs(surfaces["fitted"]).max())
    residual_max = abs(surfaces["residual"]).max()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panels = [
        ("Learned InnerNet", surfaces["learned"], vmax),
        (
            f"Scaled SwiGLU fit (R²={surfaces['fit']['r2']:.3f})",
            surfaces["fitted"],
            vmax,
        ),
        ("Residual", surfaces["residual"], residual_max),
    ]
    for axis, (title, values, limit) in zip(axes, panels):
        contour = axis.contourf(
            a, b, values, levels=30, cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        fig.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xlabel("a")
        axis.set_ylabel("b")
        axis.set_aspect("equal")
    coefficient = surfaces["fit"]["terms"]["silu(a)*b"]
    fig.suptitle(
        f"WikiText-2 d=64, seed 42: f(a,b) ≈ {coefficient:.3f}·SiLU(a)·b",
        fontsize=11,
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}.png/.pdf")


if __name__ == "__main__":
    main()
