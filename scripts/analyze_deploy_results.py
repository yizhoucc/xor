"""Summarize deploy quality, throughput, parameters, and paired differences."""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_stats import paired_comparison


def _finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_payload(data, parameter_counts, primary_metric, expected_seeds=5):
    results = data["results"]
    summary = {"primary_metric": primary_metric, "expected_seeds": expected_seeds, "conditions": {}}
    for condition, payload in results.items():
        metric_values = _finite(payload.get(primary_metric, []))
        throughput = _finite(payload.get("tput", []))
        summary["conditions"][condition] = {
            "completed_seeds": len(metric_values),
            "complete": len(metric_values) == expected_seeds,
            "parameters": parameter_counts.get(condition),
            f"mean_{primary_metric}": float(np.mean(metric_values)) if metric_values else None,
            f"sample_sd_{primary_metric}": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else None,
            "mean_throughput": float(np.mean(throughput)) if throughput else None,
        }

    comparisons = {}
    pairs = [(condition, "innernet") for condition in results if condition != "innernet"]
    if "distilled" in results:
        pairs.extend(
            ("distilled", reference)
            for reference in ("relu", "gelu", "swiglu")
            if reference in results
        )
    for condition, reference_name in pairs:
        payload = results[condition]
        reference = results[reference_name]
        values = _finite(payload.get(primary_metric, []))
        throughput = _finite(payload.get("tput", []))
        ref_metric = _finite(reference.get(primary_metric, []))
        ref_tput = _finite(reference.get("tput", []))
        paired_n = min(len(values), len(ref_metric))
        comparison = None
        if paired_n >= 2:
            comparison = paired_comparison(
                np.asarray(values[:paired_n]), np.asarray(ref_metric[:paired_n]),
                n_bootstrap=10000, rng=np.random.default_rng(0),
            )
        speedup = None
        if throughput and ref_tput:
            speedup = float(np.mean(throughput) / np.mean(ref_tput))
        comparisons[f"{condition}_vs_{reference_name}"] = {
            "paired_seed_count": paired_n,
            "metric_difference_condition_minus_reference": (
                comparison["mean_difference"] if comparison else None
            ),
            "metric_difference_ci": (
                [comparison["ci_low"], comparison["ci_high"]] if comparison else None
            ),
            "paired_t_p": comparison["parametric_p"] if comparison else None,
            "wilcoxon_p": comparison["nonparametric_p"] if comparison else None,
            "cohen_dz": comparison["effect_size"] if comparison else None,
            "throughput_ratio_condition_vs_reference": speedup,
        }
    summary["comparisons"] = comparisons
    return summary


def _parameter_counts(kind, data):
    coeffs = data.get("coeffs", {})
    if kind == "cnn":
        from scripts.deploy_distilled_cnn import CNN
        return {
            op: sum(parameter.numel() for parameter in CNN(op, coeffs).parameters())
            for op in data["results"]
        }

    from scripts.deploy_distilled import build_model
    args = SimpleNamespace(**data["args"])
    vocab_size = 10000
    return {
        op: sum(parameter.numel() for parameter in build_model(op, vocab_size, args, coeffs).parameters())
        for op in data["results"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", default="exp/deploy_cnn_cifar10/results.json")
    parser.add_argument("--ffn", default="exp/deploy_ffn_d128/results.json")
    parser.add_argument("--output", default="results/audit/deploy_analysis.json")
    args = parser.parse_args()

    analyses = {}
    for kind, path, metric in (("cnn", args.cnn, "acc"), ("ffn", args.ffn, "ppl")):
        with open(path) as handle:
            data = json.load(handle)
        expected = int(data.get("args", {}).get("num_seeds", 5))
        analyses[kind] = summarize_payload(
            data, _parameter_counts(kind, data), metric, expected,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(analyses, handle, indent=2, sort_keys=True)
    print(json.dumps(analyses, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
