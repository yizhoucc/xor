"""Plot the post-parameter-sharing Transformer scaling sweep.

All values are loaded from canonical audit artifacts; no result is hard-coded.
The comparison is same-width (not total-parameter-matched).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCALES = {
    64: {
        "gelu": "transformer_wikitext_baseline_small",
        "swiglu": "transformer_wikitext_swiglu_small",
        "innernet": "transformer_wikitext_2arg_small",
        "comparison": "transformer_d64_innernet_vs_gelu",
    },
    128: {
        "gelu": "transformer_wikitext_baseline",
        "swiglu": "transformer_wikitext_swiglu",
        "innernet": "transformer_wikitext_2arg",
        "comparison": "transformer_d128_innernet_vs_gelu",
    },
    192: {
        "gelu": "transformer_wikitext_scale_gelu_d192",
        "swiglu": "transformer_wikitext_swiglu_d192",
        "innernet": "transformer_wikitext_scale_2arg_d192",
        "comparison": "transformer_d192_innernet_vs_gelu",
    },
    256: {
        "gelu": "transformer_wikitext_baseline_large",
        "swiglu": "transformer_wikitext_swiglu_large",
        "innernet": "transformer_wikitext_2arg_large",
        "comparison": "transformer_d256_innernet_vs_gelu",
    },
}


def load_scaling_data(summary_path, comparisons_path):
    with open(summary_path, newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with open(comparisons_path, newline="") as handle:
        comparisons = {row["name"]: row for row in csv.DictReader(handle)}

    by_name = {}
    for row in summaries:
        if row["metric"] == "best_val_ppl" and row["run_status"] == "success":
            by_name.setdefault(row["exp_name"], []).append(row)

    output = {}
    for width, spec in SCALES.items():
        output[width] = {}
        for method in ("gelu", "swiglu", "innernet"):
            rows = by_name.get(spec[method], [])
            if len(rows) != 1:
                raise ValueError(f"expected one canonical row for {spec[method]}, found {len(rows)}")
            output[width][method] = {
                "mean": float(rows[0]["mean"]),
                "population_sd": float(rows[0]["population_sd"]),
            }
        comparison = comparisons[spec["comparison"]]
        output[width]["paired_t_p"] = float(comparison["parametric_p"])
        gelu = output[width]["gelu"]["mean"]
        innernet = output[width]["innernet"]["mean"]
        output[width]["improvement_pct"] = 100.0 * (gelu - innernet) / gelu
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/audit/grouped_metric_summary.csv")
    parser.add_argument("--comparisons", default="results/audit/core_comparisons.csv")
    parser.add_argument("--output", default="results/figures/fig_scaling_law")
    args = parser.parse_args()

    data = load_scaling_data(args.summary, args.comparisons)
    widths = sorted(data)
    colors = {"gelu": "#777777", "swiglu": "#2ca02c", "innernet": "#1f77b4"}
    labels = {"gelu": "GELU", "swiglu": "SwiGLU", "innernet": "InnerNet"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for method in ("gelu", "swiglu", "innernet"):
        means = [data[width][method]["mean"] for width in widths]
        errors = [data[width][method]["population_sd"] for width in widths]
        ax1.errorbar(widths, means, yerr=errors, marker="o", capsize=3,
                     linewidth=2, color=colors[method], label=labels[method])
    ax1.set_xlabel("d_model")
    ax1.set_ylabel("Validation PPL (lower is better)")
    ax1.set_title("WikiText-2 Transformer FFN (same width)")
    ax1.set_xticks(widths)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    improvements = [data[width]["improvement_pct"] for width in widths]
    p_values = [data[width]["paired_t_p"] for width in widths]
    ax2.plot(widths, improvements, "D-", color="#9467bd", linewidth=2)
    for width, improvement, p_value in zip(widths, improvements, p_values):
        align = "left" if width == widths[0] else "right" if width == widths[-1] else "center"
        ax2.annotate(
            f"{improvement:.1f}%\np={p_value:.3g}",
            (width, improvement), textcoords="offset points", xytext=(0, 8),
            ha=align, fontsize=8,
        )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("d_model")
    ax2.set_ylabel("PPL reduction vs GELU (%)")
    ax2.set_title("Benefit is positive but non-monotonic")
    ax2.set_xticks(widths)
    ax2.set_xlim(min(widths) - 12, max(widths) + 12)
    ax2.set_ylim(min(0.0, min(improvements) - 0.25), max(improvements) + 0.45)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}.png/.pdf")
    print("InnerNet improvement %:", [round(value, 3) for value in improvements])


if __name__ == "__main__":
    main()
