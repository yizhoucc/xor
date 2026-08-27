"""Plot CIFAR-10 MLP accuracy against measured model parameter count."""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model


WIDTHS = (32, 64, 128, 256, 512)


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def count_parameters(config_path):
    with open(config_path) as handle:
        config = _namespace(yaml.safe_load(handle))
    instance = getattr(model, config.model.name)(config)
    return sum(parameter.numel() for parameter in instance.parameters())


def load_parameter_efficiency(summary_path, config_root="config/experiments"):
    with open(summary_path, newline="") as handle:
        summary = {
            row["exp_name"]: row
            for row in csv.DictReader(handle)
            if row["metric"] == "test_accuracy" and row["run_status"] == "success"
        }
    data = {"innernet": [], "relu": []}
    for method, suffix in (("innernet", "2arg"), ("relu", "relu")):
        for width in WIDTHS:
            exp_name = f"mlp_cifar_scale_{suffix}_w{width}"
            row = summary[exp_name]
            config = Path(config_root) / f"{exp_name}.yaml"
            data[method].append({
                "width": width,
                "parameters": count_parameters(config),
                "accuracy": float(row["mean"]),
                "population_sd": float(row["population_sd"]),
                "n": int(row["seed_count"]),
            })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/audit/grouped_metric_summary.csv")
    parser.add_argument("--config-root", default="config/experiments")
    parser.add_argument("--output", default="results/figures/fig_parameter_efficiency")
    parser.add_argument("--json", default="results/audit/parameter_efficiency.json")
    args = parser.parse_args()

    data = load_parameter_efficiency(args.summary, args.config_root)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    styles = {
        "innernet": ("InnerNet", "#1f77b4", "o"),
        "relu": ("ReLU", "#777777", "s"),
    }
    for method, rows in data.items():
        label, color, marker = styles[method]
        ax.errorbar(
            [row["parameters"] / 1e3 for row in rows],
            [100 * row["accuracy"] for row in rows],
            yerr=[100 * row["population_sd"] for row in rows],
            label=label, color=color, marker=marker, linewidth=2, capsize=3,
        )
        for row in rows:
            ax.annotate(
                f"w={row['width']}",
                (row["parameters"] / 1e3, 100 * row["accuracy"]),
                textcoords="offset points", xytext=(4, 5), fontsize=7,
            )

    inner128 = next(row for row in data["innernet"] if row["width"] == 128)
    relu256 = next(row for row in data["relu"] if row["width"] == 256)
    saving = 100.0 * (1.0 - inner128["parameters"] / relu256["parameters"])
    ax.plot(
        [inner128["parameters"] / 1e3, relu256["parameters"] / 1e3],
        [100 * inner128["accuracy"], 100 * relu256["accuracy"]],
        linestyle="--", color="#9467bd", linewidth=1.2,
    )
    ax.text(
        610, 52.5,
        f"Comparable accuracy\n{saving:.1f}% fewer parameters",
        color="#9467bd", fontsize=9, ha="center",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (thousands, log scale)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("MLP parameter efficiency (5 seeds)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    payload = {
        "data": data,
        "matched_cross_width": {
            "innernet_width": 128,
            "innernet_parameters": inner128["parameters"],
            "innernet_accuracy": inner128["accuracy"],
            "relu_width": 256,
            "relu_parameters": relu256["parameters"],
            "relu_accuracy": relu256["accuracy"],
            "parameter_saving_pct": saving,
        },
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {output}.png/.pdf and {json_path}")


if __name__ == "__main__":
    main()
