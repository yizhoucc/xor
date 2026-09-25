#!/usr/bin/env python3
"""Generate appendix figures from existing audited result files.

This script is intentionally read-only with respect to experiment data. It
does not run training or contact the cluster; it only reads files under
results/audit and writes figure files under results/figures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "audit"
FIGURE_DIR = ROOT / "results" / "figures"

CAUSAL_JSON = AUDIT_DIR / "causal_matrix_summary.json"
CAUSAL_CSV = AUDIT_DIR / "causal_matrix_conditions.csv"
INNER_HIDDEN_JSON = AUDIT_DIR / "inner_hidden_ablation.json"
COMPUTE_COST_JSON = AUDIT_DIR / "compute_cost_profile.json"

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
    "light_gray": "#E5E5E5",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "ytick.direction": "out",
            "xtick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [FIGURE_DIR / f"{stem}.png", FIGURE_DIR / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=400)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_causal_conditions() -> tuple[list[dict[str, Any]], str]:
    """Load per-condition causal matrix rows from JSON, falling back to CSV."""
    if CAUSAL_JSON.exists():
        payload = load_json(CAUSAL_JSON)
        conditions = payload.get("conditions", [])
        if conditions:
            return conditions, str(CAUSAL_JSON.relative_to(ROOT))

    with CAUSAL_CSV.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in ("seed", "host_ppl", "best_ppl", "final_ppl", "n_epochs",
                        "operator_r2_margin", "mult_r2", "swiglu_r2", "poly3_r2"):
                if key in parsed:
                    parsed[key] = to_float(parsed[key])
            if parsed.get("seed") is not None:
                parsed["seed"] = int(parsed["seed"])
            rows.append(parsed)
    return rows, str(CAUSAL_CSV.relative_to(ROOT))


def assert_fields(rows: list[dict[str, Any]], fields: tuple[str, ...], name: str) -> None:
    missing = []
    for field in fields:
        if any(row.get(field) in (None, "") for row in rows):
            missing.append(field)
    if missing:
        raise ValueError(f"{name} has missing fields: {', '.join(missing)}")


def plot_causal_matrix() -> dict[str, Any]:
    rows, source = load_causal_conditions()
    rows = [
        row
        for row in rows
        if row.get("host") in {"bilinear", "swiglu"}
        and row.get("operator") in {"multiply", "swiglu"}
    ]
    assert_fields(rows, ("host", "mode", "init", "seed", "operator", "mult_r2", "swiglu_r2"),
                  "causal matrix")

    host_targets = {"bilinear": "multiply", "swiglu": "swiglu"}
    host_labels = {"bilinear": "Bilinear host", "swiglu": "SwiGLU host"}
    operator_labels = {"multiply": "Multiply", "swiglu": "SwiGLU"}
    op_index = {"multiply": 0, "swiglu": 1}

    mode_order = {"frozen": 0, "joint": 1}
    init_order = {"random": 0, "identity": 1, "multiply": 2, "swiglu": 3}
    group_keys = sorted(
        {(row["host"], row["mode"], row["init"]) for row in rows},
        key=lambda item: (
            0 if item[0] == "bilinear" else 1,
            mode_order.get(item[1], 99),
            init_order.get(item[2], 99),
        ),
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    cell = np.full((len(group_keys), len(seeds)), np.nan)
    lookup = {(row["host"], row["mode"], row["init"], int(row["seed"])): row for row in rows}
    for y, (host, mode, init_name) in enumerate(group_keys):
        for x, seed in enumerate(seeds):
            row = lookup.get((host, mode, init_name, seed))
            if row is not None:
                cell[y, x] = op_index[row["operator"]]

    apply_style()
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.45], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    cmap = ListedColormap([COLORS["blue"], COLORS["orange"]])
    ax.imshow(cell, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    row_labels = [
        f"{host_labels[host].replace(' host', '')}, {mode}, init={init_name}"
        for host, mode, init_name in group_keys
    ]
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("a  Per-condition recovery", loc="left", pad=3)
    ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(group_keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in range(cell.shape[0]):
        for x in range(cell.shape[1]):
            if not np.isnan(cell[y, x]):
                ax.text(x, y, "M" if cell[y, x] == 0 else "S",
                        ha="center", va="center", color="white", fontsize=7.5)

    ax.legend(
        handles=[
            Patch(facecolor=COLORS["blue"], label="Multiply"),
            Patch(facecolor=COLORS["orange"], label="SwiGLU"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )

    count_ax = fig.add_subplot(gs[0, 1])
    summary: dict[str, dict[str, int]] = {}
    for host in ("bilinear", "swiglu"):
        host_rows = [row for row in rows if row["host"] == host]
        target = host_targets[host]
        summary[host] = {
            "target": sum(row["operator"] == target for row in host_rows),
            "total": len(host_rows),
            "off_target": sum(row["operator"] != target for row in host_rows),
        }
    y_pos = np.array([1, 0])
    target_counts = [summary["bilinear"]["target"], summary["swiglu"]["target"]]
    off_counts = [summary["bilinear"]["off_target"], summary["swiglu"]["off_target"]]
    target_colors = [COLORS["blue"], COLORS["orange"]]
    count_ax.barh(y_pos, target_counts, color=target_colors, height=0.48)
    count_ax.barh(y_pos, off_counts, left=target_counts, color=COLORS["light_gray"], height=0.48)
    for y, host in zip(y_pos, ("bilinear", "swiglu")):
        target = host_targets[host]
        count_ax.text(
            summary[host]["total"] + 0.5,
            y,
            f"{summary[host]['target']}/{summary[host]['total']} {operator_labels[target]}",
            va="center",
            fontsize=7.5,
        )
    count_ax.set_yticks(y_pos)
    count_ax.set_yticklabels([host_labels["bilinear"], host_labels["swiglu"]])
    count_ax.set_xlim(0, max(summary[host]["total"] for host in summary) + 8)
    count_ax.set_xlabel("Conditions")
    count_ax.set_title("b  Target recovery", loc="left", pad=3)
    count_ax.spines["bottom"].set_visible(True)

    outputs = save_figure(fig, "fig_causal_matrix")
    return {"outputs": outputs, "source": source, "summary": summary}


def elapsed_to_hours(value: str | None) -> float | None:
    if not value:
        return None
    day_split = value.split("-")
    if len(day_split) == 2:
        days = int(day_split[0])
        time_part = day_split[1]
    else:
        days = 0
        time_part = value
    hms = [int(part) for part in time_part.split(":")]
    if len(hms) != 3:
        return None
    hours, minutes, seconds = hms
    return days * 24 + hours + minutes / 60 + seconds / 3600


def plot_inner_hidden_tradeoff() -> dict[str, Any]:
    payload = load_json(INNER_HIDDEN_JSON)
    conditions = payload.get("conditions", {})
    rows = []
    missing = []
    for key in sorted(conditions, key=lambda item: int(item.lstrip("h"))):
        item = conditions[key]
        hidden = int(key.lstrip("h"))
        required = ("mean_ppl", "population_sd")
        for field in required:
            if item.get(field) is None:
                missing.append(f"{key}.{field}")
        elapsed_hours = elapsed_to_hours(item.get("elapsed"))
        if elapsed_hours is None:
            missing.append(f"{key}.elapsed")
        rows.append({
            "hidden": hidden,
            "mean_ppl": item.get("mean_ppl"),
            "population_sd": item.get("population_sd"),
            "elapsed_hours": elapsed_hours,
        })

    valid_ppl = [row for row in rows if row["mean_ppl"] is not None and row["population_sd"] is not None]
    valid_runtime = [row for row in valid_ppl if row["elapsed_hours"] is not None]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    ax.errorbar(
        [row["hidden"] for row in valid_ppl],
        [row["mean_ppl"] for row in valid_ppl],
        yerr=[row["population_sd"] for row in valid_ppl],
        color=COLORS["blue"],
        marker="o",
        capsize=3,
        linewidth=1.3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels(["8", "16", "32", "64"])
    ax.set_xlabel("InnerNet hidden units")
    ax.set_ylabel("Validation PPL (lower is better)")
    ax.set_title("a  PPL by hidden width", loc="left", pad=3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax = axes[1]
    ax.scatter(
        [row["elapsed_hours"] for row in valid_runtime],
        [row["mean_ppl"] for row in valid_runtime],
        s=34,
        color=COLORS["green"],
        zorder=3,
    )
    if len(valid_runtime) > 1:
        ordered = sorted(valid_runtime, key=lambda row: row["hidden"])
        ax.plot(
            [row["elapsed_hours"] for row in ordered],
            [row["mean_ppl"] for row in ordered],
            color=COLORS["green"],
            alpha=0.6,
            linewidth=1.0,
        )
    if valid_runtime:
        y_values = [row["mean_ppl"] for row in valid_runtime]
        y_pad = max(0.12, (max(y_values) - min(y_values)) * 0.25)
        ax.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)
    for row in valid_runtime:
        ax.annotate(
            f"h={row['hidden']}",
            (row["elapsed_hours"], row["mean_ppl"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    missing_runtime = [row for row in valid_ppl if row["elapsed_hours"] is None]
    if missing_runtime:
        label = ", ".join(f"h={row['hidden']}" for row in missing_runtime)
        ax.text(
            0.02,
            0.04,
            f"{label}: runtime not recorded",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color=COLORS["gray"],
        )
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("Validation PPL")
    ax.set_title("b  Runtime vs PPL", loc="left", pad=3)
    ax.grid(alpha=0.25, linewidth=0.5)

    outputs = save_figure(fig, "fig_inner_hidden_tradeoff")
    return {
        "outputs": outputs,
        "source": str(INNER_HIDDEN_JSON.relative_to(ROOT)),
        "missing": missing,
    }


def plot_compute_cost_breakdown() -> dict[str, Any]:
    payload = load_json(COMPUTE_COST_JSON)
    arch_specs = [
        ("cnn", "CNN", "relu"),
        ("transformer_ffn_d128", "Transformer FFN", "gelu"),
    ]
    method_specs = [
        ("innernet", "InnerNet", COLORS["red"]),
        ("distilled", "Poly3", COLORS["blue"]),
        ("swiglu", "SwiGLU", COLORS["green"]),
    ]
    metric_specs = [
        ("forward_ms_per_batch", "Inference"),
        ("train_step_ms_per_batch", "Training"),
        ("peak_training_memory_mb", "Memory"),
    ]

    missing = []
    ratios: dict[str, dict[str, dict[str, float]]] = {}
    for arch_key, _, baseline_key in arch_specs:
        arch = payload.get(arch_key)
        if arch is None:
            missing.append(arch_key)
            continue
        baseline = arch.get(baseline_key)
        if baseline is None:
            missing.append(f"{arch_key}.{baseline_key}")
            continue
        ratios[arch_key] = {}
        for method_key, _, _ in method_specs:
            method = arch.get(method_key)
            if method is None:
                missing.append(f"{arch_key}.{method_key}")
                continue
            ratios[arch_key][method_key] = {}
            for metric_key, _ in metric_specs:
                base_value = baseline.get(metric_key)
                method_value = method.get(metric_key)
                if base_value in (None, 0) or method_value is None:
                    missing.append(f"{arch_key}.{method_key}.{metric_key}")
                    continue
                ratios[arch_key][method_key][metric_key] = method_value / base_value

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), sharey=True, gridspec_kw={"wspace": 0.14})
    width = 0.22
    x = np.arange(len(metric_specs))
    for ax, (arch_key, arch_label, baseline_key) in zip(axes, arch_specs):
        for offset, (method_key, method_label, color) in zip((-width, 0, width), method_specs):
            values = [ratios.get(arch_key, {}).get(method_key, {}).get(metric_key, np.nan)
                      for metric_key, _ in metric_specs]
            bars = ax.bar(x + offset, values, width=width, label=method_label, color=color)
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.08,
                    f"{value:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    rotation=90 if value >= 10 else 0,
                )
        ax.axhline(1.0, color="#333333", linewidth=0.7, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 40)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metric_specs])
        ax.set_title(f"{arch_label} (baseline: {baseline_key.upper()})", loc="left", pad=3)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Relative cost (baseline = 1x)")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    outputs = save_figure(fig, "fig_compute_cost_breakdown")
    return {
        "outputs": outputs,
        "source": str(COMPUTE_COST_JSON.relative_to(ROOT)),
        "missing": missing,
        "ratios": ratios,
    }


def main() -> None:
    reports = [
        ("fig_causal_matrix", plot_causal_matrix()),
        ("fig_inner_hidden_tradeoff", plot_inner_hidden_tradeoff()),
        ("fig_compute_cost_breakdown", plot_compute_cost_breakdown()),
    ]
    for name, report in reports:
        outputs = ", ".join(str(path.relative_to(ROOT)) for path in report["outputs"])
        print(f"{name}: wrote {outputs}")
        print(f"  source: {report['source']}")
        missing = report.get("missing")
        if missing:
            print(f"  missing fields: {', '.join(missing)}")
        if name == "fig_causal_matrix":
            summary = report["summary"]
            print(
                "  recovery: "
                f"bilinear {summary['bilinear']['target']}/{summary['bilinear']['total']} multiply, "
                f"swiglu {summary['swiglu']['target']}/{summary['swiglu']['total']} swiglu"
            )


if __name__ == "__main__":
    main()
