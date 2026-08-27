"""Plot the Housing regression width sweep from canonical audit artifacts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WIDTHS = (32, 64, 120, 256, 512)


def load_housing_scaling(summary_path, comparisons_path):
    with open(summary_path, newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with open(comparisons_path, newline="") as handle:
        comparisons = {row["name"]: row for row in csv.DictReader(handle)}
    by_name = {
        row["exp_name"]: row
        for row in summaries
        if row["metric"] == "test_mse" and row["run_status"] == "success"
    }

    output = {}
    for width in WIDTHS:
        inner = by_name[f"mlp_housing_scale_2arg_w{width}"]
        relu = by_name[f"mlp_housing_scale_relu_w{width}"]
        comparison = comparisons[f"housing_w{width}_innernet_vs_relu"]
        inner_mean = float(inner["mean"])
        relu_mean = float(relu["mean"])
        output[width] = {
            "innernet_mean": inner_mean,
            "innernet_population_sd": float(inner["population_sd"]),
            "relu_mean": relu_mean,
            "relu_population_sd": float(relu["population_sd"]),
            "mse_reduction_pct": 100.0 * (relu_mean - inner_mean) / relu_mean,
            "paired_t_p": float(comparison["parametric_p"]),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/audit/grouped_metric_summary.csv")
    parser.add_argument("--comparisons", default="results/audit/core_comparisons.csv")
    parser.add_argument("--output", default="results/figures/fig_housing_scaling")
    args = parser.parse_args()

    data = load_housing_scaling(args.summary, args.comparisons)
    widths = list(WIDTHS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(
        widths, [data[w]["innernet_mean"] for w in widths],
        yerr=[data[w]["innernet_population_sd"] for w in widths],
        marker="o", capsize=3, linewidth=2, label="InnerNet",
    )
    ax1.errorbar(
        widths, [data[w]["relu_mean"] for w in widths],
        yerr=[data[w]["relu_population_sd"] for w in widths],
        marker="o", capsize=3, linewidth=2, label="ReLU", color="#777777",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(widths, labels=[str(w) for w in widths])
    ax1.set_xlabel("Hidden width")
    ax1.set_ylabel("Test MSE (lower is better)")
    ax1.set_title("Housing regression (3 seeds)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    improvements = [data[w]["mse_reduction_pct"] for w in widths]
    ax2.plot(widths, improvements, "D-", linewidth=2, color="#9467bd")
    for width, improvement in zip(widths, improvements):
        align = "left" if width == widths[0] else "right" if width == widths[-1] else "center"
        ax2.annotate(
            f"{improvement:+.1f}%\np={data[width]['paired_t_p']:.3g}",
            (width, improvement), textcoords="offset points", xytext=(0, 8),
            ha=align, fontsize=8,
        )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(widths, labels=[str(w) for w in widths])
    ax2.set_xlabel("Hidden width")
    ax2.set_ylabel("MSE reduction vs ReLU (%)")
    ax2.set_title("Benefit disappears at large width")
    ax2.set_ylim(min(improvements) - 0.8, max(improvements) + 1.6)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}.png/.pdf")
    print("MSE reduction %:", [round(value, 3) for value in improvements])


if __name__ == "__main__":
    main()
