"""Build reproducible statistics for the paper's registered core comparisons."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_stats import independent_comparison, paired_comparison


OUTPUT_FIELDS = (
    "name",
    "condition_exp",
    "reference_exp",
    "metric",
    "mode",
    "lower_better",
    "condition_n",
    "reference_n",
    "comparison_n",
    "common_seeds",
    "condition_values",
    "reference_values",
    "condition_mean",
    "reference_mean",
    "mean_difference",
    "ci_low",
    "ci_high",
    "parametric_test",
    "parametric_p",
    "nonparametric_test",
    "nonparametric_p",
    "effect_name",
    "effect_size",
    "direction",
    "status",
    "notes",
)


def _select(rows, spec):
    selected = [
        row for row in rows
        if row["exp_name"] == spec["exp_name"]
        and row["metric"] == spec["metric"]
        and row.get("run_status", "success") == spec.get("run_status", "success")
        and (not spec.get("selection") or row["selection"] == spec["selection"])
    ]
    signatures = {row["config_signature"] for row in selected}
    if spec.get("config_signature"):
        selected = [row for row in selected if row["config_signature"] == spec["config_signature"]]
        signatures = {row["config_signature"] for row in selected}
    if not selected:
        raise ValueError(f"no manifest rows for {spec}")
    if len(signatures) != 1:
        raise ValueError(f"ambiguous scientific configs for {spec}: {sorted(signatures)}")

    by_seed = {}
    for row in selected:
        seed = str(row["seed"])
        value = float(row["value"])
        if seed in by_seed and not np.isclose(by_seed[seed], value, rtol=1e-12, atol=1e-12):
            raise ValueError(f"conflicting values for {spec}, seed {seed}")
        by_seed[seed] = value
    return by_seed


def _seed_sort(seed):
    return (not seed.isdigit(), int(seed) if seed.isdigit() else seed)


def build_comparisons(rows, specs, bootstrap=10000, random_seed=0):
    output = []
    for index, spec in enumerate(specs):
        condition = _select(rows, spec["condition"])
        reference = _select(rows, spec["reference"])
        paired = bool(spec.get("paired", False))
        lower_better = bool(spec.get("lower_better", False))
        notes = []

        if paired:
            common = sorted(set(condition) & set(reference), key=_seed_sort)
            missing_condition = sorted(set(reference) - set(condition), key=_seed_sort)
            missing_reference = sorted(set(condition) - set(reference), key=_seed_sort)
            if missing_condition:
                notes.append(f"missing condition seeds: {','.join(missing_condition)}")
            if missing_reference:
                notes.append(f"missing reference seeds: {','.join(missing_reference)}")
            x = np.asarray([condition[seed] for seed in common], dtype=float)
            ref = np.asarray([reference[seed] for seed in common], dtype=float)
            comparison = paired_comparison(
                x, ref, n_bootstrap=bootstrap,
                rng=np.random.default_rng(random_seed + index),
            )
            mode = "paired"
            parametric_test = "paired_t"
            nonparametric_test = "wilcoxon"
        else:
            common = []
            x = np.asarray(list(condition.values()), dtype=float)
            ref = np.asarray(list(reference.values()), dtype=float)
            comparison = independent_comparison(
                x, ref, n_bootstrap=bootstrap,
                rng=np.random.default_rng(random_seed + index),
            )
            mode = "independent"
            parametric_test = "welch_t"
            nonparametric_test = "mann_whitney_u"

        difference = comparison["mean_difference"]
        condition_better = difference < 0 if lower_better else difference > 0
        output.append({
            "name": spec["name"],
            "condition_exp": spec["condition"]["exp_name"],
            "reference_exp": spec["reference"]["exp_name"],
            "metric": spec["condition"]["metric"],
            "mode": mode,
            "lower_better": lower_better,
            "condition_n": len(condition),
            "reference_n": len(reference),
            "comparison_n": comparison["n"],
            "common_seeds": ";".join(common),
            "condition_values": ";".join(f"{value:.12g}" for value in x),
            "reference_values": ";".join(f"{value:.12g}" for value in ref),
            "condition_mean": float(np.mean(x)),
            "reference_mean": float(np.mean(ref)),
            "mean_difference": difference,
            "ci_low": comparison["ci_low"],
            "ci_high": comparison["ci_high"],
            "parametric_test": parametric_test,
            "parametric_p": comparison["parametric_p"],
            "nonparametric_test": nonparametric_test,
            "nonparametric_p": comparison["nonparametric_p"],
            "effect_name": comparison["effect_name"],
            "effect_size": comparison["effect_size"],
            "direction": "condition_better" if condition_better else "reference_better",
            "status": "complete",
            "notes": " | ".join(notes),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/audit/metric_manifest.csv")
    parser.add_argument("--comparisons", default="config/audit/core_comparisons.yaml")
    parser.add_argument("--output", default="results/audit/core_comparisons.csv")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    with open(args.comparisons) as handle:
        specs = yaml.safe_load(handle)["comparisons"]
    output = build_comparisons(rows, specs, args.bootstrap, args.random_seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"comparison_count": len(output), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
