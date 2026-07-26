"""Build an auditable manifest from local experiment artifacts.

The script never infers metrics from free-form logs. It parses only known,
structured result files and records completed experiments without such files as
missing sources for follow-up. Pickle inputs are trusted project artifacts.
"""
import argparse
import csv
import json
import math
import numbers
import pickle
from collections import Counter
from pathlib import Path

import yaml


KNOWN_RESULT_FILES = (
    "test_results.p",
    "lm_results.p",
    "mixer_results.p",
    "rl_results.p",
)
STAGE_MARKERS = (
    "PRETRAIN_DONE",
    "PHASE1_DONE",
    "PHASE2_DONE",
    "TEST_DONE",
    "COMPLETED",
)
INVENTORY_FIELDS = (
    "experiment_dir",
    "exp_name",
    "task_type",
    "model",
    "dataset",
    "config_seed",
    "config_hash",
    "markers",
    "result_files",
    "audit_status",
    "notes",
)
METRIC_FIELDS = (
    "experiment_dir",
    "exp_name",
    "task_type",
    "model",
    "dataset",
    "condition",
    "seed",
    "metric",
    "value",
    "selection",
    "selected_epoch",
    "num_epochs",
    "source_file",
    "config_hash",
    "audit_status",
)


def _nested(mapping, *keys, default=""):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _base_metadata(config, experiment_dir, config_hash):
    model = _nested(config, "model", "name")
    dataset = _nested(config, "dataset", "name") or config.get("dataset_name", "")
    condition = model or config.get("condition", "") or config.get("exp_name", "")
    return {
        "experiment_dir": str(experiment_dir),
        "exp_name": config.get("exp_name", experiment_dir.name),
        "task_type": config.get("task_type", "classification"),
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "config_hash": config_hash,
    }


def _metric_row(metadata, seed, metric, value, selection, epoch, num_epochs, source_file):
    return {
        **metadata,
        "seed": seed,
        "metric": metric,
        "value": float(value),
        "selection": selection,
        "selected_epoch": epoch,
        "num_epochs": num_epochs,
        "source_file": str(source_file),
        "audit_status": "raw-verified",
    }


def parse_test_results(data, metadata, config_seed, source_file):
    rows = []
    if not isinstance(data, dict):
        return rows
    for metric, value in sorted(data.items()):
        if _finite_number(value):
            rows.append(
                _metric_row(
                    metadata,
                    config_seed,
                    metric,
                    value,
                    "reported_scalar",
                    "",
                    "",
                    source_file,
                )
            )
    return rows


def _curve_spec(data):
    if "all_ppl" in data:
        return "all_ppl", "val_ppl", min
    if "all_acc" in data:
        return "all_acc", "val_accuracy", max
    if "all_scores" in data:
        return "all_scores", "eval_score", max
    return None


def parse_multiseed_curves(data, metadata, source_file):
    rows = []
    if not isinstance(data, dict) or "seeds" not in data:
        return rows
    spec = _curve_spec(data)
    if spec is None:
        return rows
    curve_key, metric_prefix, selector = spec
    seeds = data["seeds"]
    curves = data[curve_key]
    if len(seeds) != len(curves):
        raise ValueError(f"{curve_key} has {len(curves)} curves for {len(seeds)} seeds")

    for seed, curve in zip(seeds, curves):
        finite_curve = [(index, float(value)) for index, value in enumerate(curve) if _finite_number(value)]
        if not finite_curve:
            continue
        values = [value for _, value in finite_curve]
        selected_value = selector(values)
        selected_index = next(index for index, value in finite_curve if value == selected_value)
        final_index, final_value = finite_curve[-1]
        rows.append(
            _metric_row(
                metadata,
                seed,
                f"best_{metric_prefix}",
                selected_value,
                "best_over_recorded_epochs",
                selected_index + 1,
                len(curve),
                source_file,
            )
        )
        rows.append(
            _metric_row(
                metadata,
                seed,
                f"final_{metric_prefix}",
                final_value,
                "final_recorded_epoch",
                final_index + 1,
                len(curve),
                source_file,
            )
        )
    return rows


def parse_result_file(path, metadata, config_seed):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if path.name == "test_results.p":
        return parse_test_results(data, metadata, config_seed, path)
    return parse_multiseed_curves(data, metadata, path)


def audit_experiment(config_path, root):
    experiment_dir = config_path.parent
    relative_dir = experiment_dir.relative_to(root.parent)
    notes = []
    metric_rows = []
    try:
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as error:
        config = {}
        notes.append(f"config parse error: {error}")

    hash_path = experiment_dir / "config_hash.txt"
    config_hash = hash_path.read_text().strip() if hash_path.exists() else ""
    metadata = _base_metadata(config, relative_dir, config_hash)
    config_seed = config.get("seed", "")
    markers = [marker for marker in STAGE_MARKERS if (experiment_dir / marker).exists()]
    result_paths = [experiment_dir / name for name in KNOWN_RESULT_FILES if (experiment_dir / name).exists()]

    for result_path in result_paths:
        relative_result = result_path.relative_to(root.parent)
        try:
            rows = parse_result_file(result_path, metadata, config_seed)
            for row in rows:
                row["source_file"] = str(relative_result)
            metric_rows.extend(rows)
            if not rows:
                notes.append(f"no supported metrics in {result_path.name}")
        except Exception as error:
            notes.append(f"{result_path.name} parse error: {error}")

    if metric_rows:
        status = "raw-verified"
    elif notes and result_paths:
        status = "result-unparsed"
    elif "COMPLETED" in markers:
        status = "completed-no-result"
    else:
        status = "incomplete"

    inventory_row = {
        "experiment_dir": str(relative_dir),
        "exp_name": metadata["exp_name"],
        "task_type": metadata["task_type"],
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "config_seed": config_seed,
        "config_hash": config_hash,
        "markers": ";".join(markers),
        "result_files": ";".join(path.name for path in result_paths),
        "audit_status": status,
        "notes": " | ".join(notes),
    }
    return inventory_row, metric_rows


def build_manifest(exp_root):
    inventory = []
    metrics = []
    for config_path in sorted(exp_root.rglob("config.yaml")):
        inventory_row, metric_rows = audit_experiment(config_path, exp_root)
        inventory.append(inventory_row)
        metrics.extend(metric_rows)
    return inventory, metrics


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--output-dir", default="results/audit")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    output_dir = Path(args.output_dir)
    if not exp_root.is_dir():
        parser.error(f"experiment root does not exist: {exp_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory, metrics = build_manifest(exp_root)
    inventory_path = output_dir / "experiment_inventory.csv"
    metrics_path = output_dir / "metric_manifest.csv"
    summary_path = output_dir / "audit_summary.json"
    _write_csv(inventory_path, INVENTORY_FIELDS, inventory)
    _write_csv(metrics_path, METRIC_FIELDS, metrics)

    status_counts = Counter(row["audit_status"] for row in inventory)
    result_file_counts = Counter()
    for row in inventory:
        for name in filter(None, row["result_files"].split(";")):
            result_file_counts[name] += 1
    summary = {
        "exp_root": str(exp_root),
        "experiment_count": len(inventory),
        "metric_row_count": len(metrics),
        "status_counts": dict(sorted(status_counts.items())),
        "result_file_counts": dict(sorted(result_file_counts.items())),
        "inventory": str(inventory_path),
        "metrics": str(metrics_path),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
