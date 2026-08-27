"""Plot ReLU, SwiGLU, and an actual task-trained InnerNet surface."""
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


def load_learned_surface(checkpoint, n=200, surface_range=3.0):
    net = _distill.load_innernet(checkpoint)
    a, b, z = _distill.sample_surface(net, -surface_range, surface_range, n)
    return a.reshape(n, n), b.reshape(n, n), z.reshape(n, n)


def generate(checkpoint, output, n=200, surface_range=3.0):
    axis = np.linspace(-surface_range, surface_range, n)
    a, b = np.meshgrid(axis, axis)
    relu = np.maximum(0, a)
    swiglu = _distill.silu(a) * b
    learned_a, learned_b, learned = load_learned_surface(
        checkpoint, n=n, surface_range=surface_range
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("(a) ReLU: f(a) = max(0, a)", a, b, relu),
        ("(b) SwiGLU: SiLU(a) × b", a, b, swiglu),
        ("(c) Task-trained InnerNet (CNN)", learned_a, learned_b, learned),
    ]
    for plot_axis, (title, x, y, values) in zip(axes, panels):
        image = plot_axis.contourf(x, y, values, levels=30, cmap="RdBu_r")
        plot_axis.set_title(title, fontsize=12)
        plot_axis.set_xlabel("a")
        plot_axis.set_ylabel("b")
        fig.colorbar(image, ax=plot_axis, shrink=0.8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/figures/inner_weights_cnn_seed42.pth",
    )
    parser.add_argument("--output", default="results/figures/fig2_2d_activation_surfaces")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--range", type=float, default=3.0)
    args = parser.parse_args()
    generate(args.checkpoint, args.output, args.n, args.range)
    print(f"Wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
