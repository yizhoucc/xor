"""Build an auditable manifest from local experiment artifacts.

The script never infers metrics from free-form logs. It parses only known,
structured result files and records completed experiments without such files as
missing sources for follow-up. Pickle inputs are trusted project artifacts.
"""
import argparse
import csv
import hashlib
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
    "results.p",
    "results.json",
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
    "config_signature",
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
    "run_status",
    "selected_epoch",
    "num_epochs",
    "source_file",
    "config_hash",
    "config_signature",
    "audit_status",
)

VOLATILE_CONFIG_KEYS = {
    "best_model",
    "exp_dir",
    "gpus",
    "resume_model",
    "run_id",
    "save_dir",
    "seed",
    "test_model",
    "use_gpu",
}


def _nested(mapping, *keys, default=""):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _normalized_config(value):
    if isinstance(value, dict):
        return {
            key: _normalized_config(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_normalized_config(item) for item in value]
    return value


def _config_signature(config):
    payload = json.dumps(_normalized_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _base_metadata(config, experiment_dir, config_hash, config_signature):
    model = _nested(config, "model", "name")
    dataset = (
        _nested(config, "dataset", "name")
        or _nested(config, "lm", "dataset")
        or config.get("dataset_name", "")
    )
    if not dataset and config.get("task_type") == "language_model":
        dataset = "wikitext"
    condition = model or config.get("condition", "") or config.get("exp_name", "")
    return {
        "experiment_dir": str(experiment_dir),
        "exp_name": config.get("exp_name", experiment_dir.name),
        "task_type": config.get("task_type", "classification"),
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "config_hash": config_hash,
        "config_signature": config_signature,
    }


def _metric_row(
    metadata, seed, metric, value, selection, epoch, num_epochs, source_file,
    run_status="success",
):
    return {
        **metadata,
        "seed": seed,
        "metric": metric,
        "value": float(value),
        "selection": selection,
        "run_status": run_status,
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
        if curve_key == "all_scores":
            tail_values = values[-20:]
            rows.append(
                _metric_row(
                    metadata,
                    seed,
                    "mean_last20_eval_score",
                    sum(tail_values) / len(tail_values),
                    "mean_last_20_recorded_epochs",
                    final_index + 1,
                    len(curve),
                    source_file,
                )
            )
    return rows


def parse_script_results(data, metadata, source_file):
    """Parse structured outputs produced by standalone experiment scripts."""
    rows = []

    # Sequential-MNIST runners: [{seed, best_acc, history: {test_acc: [...]}}]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        for item in data:
            seed = item.get("seed", "")
            train_curve = item.get("history", {}).get("train_loss", [])
            run_status = "nan" if any(
                isinstance(value, numbers.Real) and not math.isfinite(value)
                for value in train_curve
            ) else "success"
            if _finite_number(item.get("best_acc")):
                rows.append(_metric_row(
                    metadata, seed, "best_test_accuracy", item["best_acc"],
                    "best_over_recorded_epochs", "", len(item.get("history", {}).get("test_acc", [])),
                    source_file, run_status,
                ))
            curve = item.get("history", {}).get("test_acc", [])
            finite_curve = [(index, value) for index, value in enumerate(curve) if _finite_number(value)]
            if finite_curve:
                index, value = finite_curve[-1]
                rows.append(_metric_row(
                    metadata, seed, "final_test_accuracy", value,
                    "final_recorded_epoch", index + 1, len(curve), source_file, run_status,
                ))
        return rows

    if not isinstance(data, dict):
        return rows

    # InnerNet-vs-SwiGLU warm-start scripts: {all_results: [{seed, ...}, ...]}.
    if isinstance(data.get("all_results"), list):
        scalar_conditions = {
            "best_swiglu_20": ("swiglu_continued", "best_val_ppl"),
            "best_innernet_20": ("innernet_joint", "best_val_ppl"),
            "best_swiglu_final": ("swiglu_capacity_baseline", "best_val_ppl"),
            "best_frozen": ("frozen_innernet", "best_val_ppl"),
            "ppl_swap": ("immediate_swap", "val_ppl"),
        }
        for item in data["all_results"]:
            seed = item.get("seed", "")
            for key, (condition, metric) in scalar_conditions.items():
                value = item.get(key)
                if not _finite_number(value):
                    continue
                row_metadata = {**metadata, "condition": condition}
                rows.append(_metric_row(
                    row_metadata, seed, metric, value,
                    "reported_scalar", "", "", source_file,
                ))
        return rows

    # Deploy scripts: {args, results: {condition: {metric: [seed values]}}}
    if isinstance(data.get("results"), dict):
        for condition, payload in data["results"].items():
            if not isinstance(payload, dict):
                continue
            for metric, values in payload.items():
                if not isinstance(values, list):
                    continue
                metric_name = {"acc": "test_accuracy", "ppl": "test_ppl", "tput": "throughput"}.get(metric, metric)
                for index, value in enumerate(values):
                    if _finite_number(value):
                        row_metadata = {**metadata, "condition": condition}
                        rows.append(_metric_row(
                            row_metadata, 42 + index, metric_name, value,
                            "reported_vector", "", "", source_file,
                        ))
        return rows

    # Causal probes: {host, freeze, seed, host_ppl, conditions: {init: {...}}}
    if isinstance(data.get("conditions"), dict) and "host" in data:
        seed = data.get("seed", "")
        if _finite_number(data.get("host_ppl")):
            host_metadata = {**metadata, "condition": f"{data['host']}_host"}
            rows.append(_metric_row(
                host_metadata, seed, "best_val_ppl", data["host_ppl"],
                "host_final", "", "", source_file,
            ))
        mode = "frozen" if data.get("freeze") else "joint"
        for init_name, payload in data["conditions"].items():
            if not isinstance(payload, dict) or not _finite_number(payload.get("best_ppl")):
                continue
            condition_metadata = {**metadata, "condition": f"{data['host']}_{mode}_{init_name}"}
            rows.append(_metric_row(
                condition_metadata, seed, "best_val_ppl", payload["best_ppl"],
                "best_over_probe_epochs", "", data.get("probe_epochs", ""), source_file,
            ))
        return rows

    # Warm-start scripts: {dataset: [{seed, best_sw, best_in}, ...], ...}
    if data and all(isinstance(records, list) for records in data.values()):
        for dataset, records in data.items():
            for item in records:
                if not isinstance(item, dict):
                    continue
                seed = item.get("seed", "")
                for key, value in item.items():
                    if not key.startswith("best_") or not _finite_number(value):
                        continue
                    condition = {"best_sw": "swiglu", "best_in": "innernet"}.get(key, key[5:])
                    row_metadata = {**metadata, "dataset": dataset, "condition": condition}
                    rows.append(_metric_row(
                        row_metadata, seed, "best_val_ppl", value,
                        "reported_scalar", "", "", source_file,
                    ))
    return rows


def parse_result_file(path, metadata, config_seed):
    if path.suffix == ".json":
        with path.open() as handle:
            data = json.load(handle)
    else:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    if path.name == "test_results.p":
        return parse_test_results(data, metadata, config_seed, path)
    if path.name in {"lm_results.p", "mixer_results.p", "rl_results.p"}:
        return parse_multiseed_curves(data, metadata, path)
    return parse_script_results(data, metadata, path)


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
    config_signature = _config_signature(config)
    metadata = _base_metadata(config, relative_dir, config_hash, config_signature)
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
        "config_signature": config_signature,
        "markers": ";".join(markers),
        "result_files": ";".join(path.name for path in result_paths),
        "audit_status": status,
        "notes": " | ".join(notes),
    }
    return inventory_row, metric_rows


def build_manifest(exp_root):
    inventory = []
    metrics = []
    config_paths = sorted(exp_root.rglob("config.yaml"))
    configured_dirs = {path.parent for path in config_paths}
    for config_path in config_paths:
        inventory_row, metric_rows = audit_experiment(config_path, exp_root)
        inventory.append(inventory_row)
        metrics.extend(metric_rows)

    # Standalone scripts often write results without a config.yaml. Preserve
    # their raw values while explicitly marking the missing config provenance.
    orphan_paths = []
    for name in ("results.p", "results.json"):
        orphan_paths.extend(
            path for path in exp_root.rglob(name) if path.parent not in configured_dirs
        )
    for result_path in sorted(orphan_paths):
        relative_dir = result_path.parent.relative_to(exp_root.parent)
        if result_path.suffix == ".json":
            with result_path.open() as handle:
                data = json.load(handle)
        else:
            with result_path.open("rb") as handle:
                data = pickle.load(handle)
        args = data.get("args", {}) if isinstance(data, dict) else {}
        signature_source = args if args else {"orphan_result": str(relative_dir)}
        metadata = {
            "experiment_dir": str(relative_dir),
            "exp_name": result_path.parent.name,
            "task_type": "standalone_script",
            "model": "",
            "dataset": args.get("dataset", "") if isinstance(args, dict) else "",
            "condition": result_path.parent.name,
            "config_hash": "",
            "config_signature": _config_signature(signature_source),
        }
        relative_result = result_path.relative_to(exp_root.parent)
        metric_rows = parse_script_results(data, metadata, relative_result)
        metrics.extend(metric_rows)
        inventory.append({
            "experiment_dir": str(relative_dir),
            "exp_name": metadata["exp_name"],
            "task_type": metadata["task_type"],
            "model": "",
            "dataset": metadata["dataset"],
            "config_seed": "",
            "config_hash": "",
            "config_signature": metadata["config_signature"],
            "markers": "",
            "result_files": result_path.name,
            "audit_status": "raw-verified" if metric_rows else "result-unparsed",
            "notes": "standalone structured result; config.yaml unavailable",
        })
    return inventory, metrics


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
