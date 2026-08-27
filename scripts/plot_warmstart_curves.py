"""Plot the ivs_d128 warm-start fork and frozen-capacity trajectories."""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_curves(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    runs = data["all_results"]
    phase1 = np.asarray([run["sw_ppl_phase1"] for run in runs], dtype=float)
    swiglu = np.asarray([run["sw_ppl_phase2"] for run in runs], dtype=float)
    innernet = np.asarray([run["in_ppl_phase2"] for run in runs], dtype=float)
    swap = np.asarray([run["ppl_swap"] for run in runs], dtype=float)
    frozen_length = min(len(run["frozen_ppl"]) for run in runs)
    frozen = np.asarray([run["frozen_ppl"][:frozen_length] for run in runs], dtype=float)
    baseline = np.asarray([run["best_swiglu_final"] for run in runs], dtype=float)
    return {
        "seeds": [run["seed"] for run in runs],
        "phase1": phase1,
        "swiglu": swiglu,
        "innernet": innernet,
        "swap": swap,
        "frozen": frozen,
        "baseline": baseline,
    }


def _plot_mean_sd(ax, x, curves, label, color):
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1)
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="exp/ivs_d128_v2/results.p")
    parser.add_argument("--output", default="results/figures/fig_warmstart_curves")
    args = parser.parse_args()
    curves = load_curves(args.input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    phase1_epochs = np.arange(1, curves["phase1"].shape[1] + 1)
    phase2_epochs = np.arange(phase1_epochs[-1] + 1, phase1_epochs[-1] + curves["swiglu"].shape[1] + 1)
    _plot_mean_sd(ax1, phase1_epochs, curves["phase1"], "Shared SwiGLU pre-fork", "#777777")
    _plot_mean_sd(ax1, phase2_epochs, curves["swiglu"], "Continue SwiGLU", "#2ca02c")
    _plot_mean_sd(ax1, phase2_epochs, curves["innernet"], "Trainable InnerNet", "#1f77b4")
    ax1.errorbar(
        [phase1_epochs[-1] + 0.25], [curves["swap"].mean()],
        yerr=[curves["swap"].std(ddof=1)], marker="x", markersize=7,
        color="#d62728", capsize=3, label="Immediate swap",
    )
    ax1.axvline(phase1_epochs[-1] + 0.5, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation PPL (lower is better)")
    ax1.set_title("Warm-start fork (5 seeds)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.3)

    frozen_epochs = np.arange(1, curves["frozen"].shape[1] + 1)
    _plot_mean_sd(ax2, frozen_epochs, curves["frozen"], "Frozen outer net; train InnerNet", "#1f77b4")
    baseline_mean = curves["baseline"].mean()
    baseline_sd = curves["baseline"].std(ddof=1)
    ax2.axhline(baseline_mean, color="#2ca02c", linewidth=2, label="SwiGLU baseline")
    ax2.fill_between(frozen_epochs, baseline_mean - baseline_sd, baseline_mean + baseline_sd,
                     color="#2ca02c", alpha=0.18)
    ax2.set_xlabel("InnerNet-only epoch")
    ax2.set_ylabel("Validation PPL (lower is better)")
    ax2.set_title(f"Frozen-capacity test (first {curves['frozen'].shape[1]} epochs)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("InnerNet warm-start dynamics on WikiText-2 (d=128)", fontsize=11, y=1.02)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}.png/.pdf")


if __name__ == "__main__":
    main()
