"""Generate a single results summary file from all experiment directories.

Usage: python scripts/generate_summary.py
Output: exp/SUMMARY.json — machine-readable summary of all results.

Run this after experiments complete to update the summary.
"""
import json
import os
import glob
import pickle
import numpy as np
from collections import defaultdict


def safe_load(path):
    try:
        return pickle.load(open(path, "rb"))
    except:
        return None


def scan_experiments():
    results = {}

    # Classification / regression experiments
    for d in sorted(glob.glob("exp/*/test_results.p")):
        exp_dir = os.path.dirname(d)
        name = os.path.basename(exp_dir)
        r = safe_load(d)
        if r is None:
            continue
        # Extract experiment type from dir name
        # e.g., cnn_cifar_2arg_20260305_... → cnn_cifar_2arg
        parts = name.split("_202")
        exp_type = parts[0] if parts else name

        if exp_type not in results:
            results[exp_type] = {"type": "classification", "runs": []}

        run = {}
        if "test_accuracy" in r:
            run["accuracy"] = r["test_accuracy"]
            results[exp_type]["type"] = "classification"
        if "test_mse" in r:
            run["mse"] = r["test_mse"]
            results[exp_type]["type"] = "regression"
        run["dir"] = name
        results[exp_type]["runs"].append(run)

    # LM experiments
    for d in sorted(glob.glob("exp/*/lm_results.p")):
        exp_dir = os.path.dirname(d)
        name = os.path.basename(exp_dir)
        r = safe_load(d)
        if r is None:
            continue
        parts = name.split("_202")
        exp_type = parts[0] if parts else name

        if exp_type not in results:
            results[exp_type] = {"type": "language_model", "runs": []}

        ppls = r.get("all_ppl", [])
        if ppls:
            vals = [x[-1] if isinstance(x, list) else x for x in ppls]
            results[exp_type]["runs"].append({
                "ppl_mean": float(np.mean(vals)),
                "ppl_std": float(np.std(vals)),
                "n_seeds": len(vals),
                "dir": name,
            })

    # RL experiments
    for d in sorted(glob.glob("exp/*/rl_results.p")):
        exp_dir = os.path.dirname(d)
        name = os.path.basename(exp_dir)
        r = safe_load(d)
        if r is None:
            continue
        parts = name.split("_202")
        exp_type = parts[0] if parts else name

        if exp_type not in results:
            results[exp_type] = {"type": "rl", "runs": []}

        ms = r.get("mean_scores", [])
        last = float(np.mean(ms[-20:])) if len(ms) >= 20 else float(np.mean(ms)) if ms else 0
        results[exp_type]["runs"].append({
            "last20_mean": last,
            "dir": name,
        })

    # Mixer/ViT experiments
    for d in sorted(glob.glob("exp/*/mixer_results.p")):
        exp_dir = os.path.dirname(d)
        name = os.path.basename(exp_dir)
        r = safe_load(d)
        if r is None:
            continue
        parts = name.split("_202")
        exp_type = parts[0] if parts else name

        if exp_type not in results:
            results[exp_type] = {"type": "mixer", "runs": []}

        aa = r.get("all_acc", [])
        if aa:
            vals = [x[-1] if isinstance(x, list) else x for x in aa]
            results[exp_type]["runs"].append({
                "acc_mean": float(np.mean(vals)),
                "acc_std": float(np.std(vals)),
                "n_seeds": len(vals),
                "dir": name,
            })

    # Compute aggregates
    summary = {}
    for exp_type, data in sorted(results.items()):
        entry = {"type": data["type"], "n_runs": len(data["runs"])}

        if data["type"] == "classification":
            accs = [r["accuracy"] for r in data["runs"] if "accuracy" in r]
            if accs:
                entry["accuracy_mean"] = round(float(np.mean(accs)) * 100, 2)
                entry["accuracy_std"] = round(float(np.std(accs)) * 100, 2)
        elif data["type"] == "regression":
            mses = [r["mse"] for r in data["runs"] if "mse" in r]
            if mses:
                entry["mse_mean"] = round(float(np.mean(mses)), 6)
                entry["mse_std"] = round(float(np.std(mses)), 6)
        elif data["type"] == "language_model":
            if data["runs"]:
                entry["ppl_mean"] = data["runs"][0].get("ppl_mean")
                entry["ppl_std"] = data["runs"][0].get("ppl_std")
                entry["n_seeds"] = data["runs"][0].get("n_seeds")
        elif data["type"] == "rl":
            if data["runs"]:
                entry["last20_mean"] = data["runs"][0].get("last20_mean")
        elif data["type"] == "mixer":
            if data["runs"]:
                entry["acc_mean"] = round(data["runs"][0].get("acc_mean", 0) * 100, 2)
                entry["acc_std"] = round(data["runs"][0].get("acc_std", 0) * 100, 2)

        summary[exp_type] = entry

    return summary


def print_summary(summary):
    """Print human-readable summary."""
    # Group by prefix
    groups = defaultdict(list)
    for name, data in sorted(summary.items()):
        prefix = name.rsplit("_", 1)[0] if any(name.endswith(s) for s in ["_2arg", "_relu", "_1arg", "_gelu", "_swiglu", "_matched"]) else name
        groups[prefix].append((name, data))

    for prefix, items in sorted(groups.items()):
        print("\n%s:" % prefix)
        for name, data in items:
            tag = name.split("_")[-1]
            n = data["n_runs"]
            if data["type"] == "classification":
                print("  %-8s %.2f+/-%.2f (n=%d)" % (tag, data.get("accuracy_mean", 0), data.get("accuracy_std", 0), n))
            elif data["type"] == "regression":
                print("  %-8s MSE=%.4f+/-%.4f (n=%d)" % (tag, data.get("mse_mean", 0), data.get("mse_std", 0), n))
            elif data["type"] == "language_model":
                print("  %-8s PPL=%.2f+/-%.2f (n=%d)" % (tag, data.get("ppl_mean", 0), data.get("ppl_std", 0), data.get("n_seeds", 0)))
            elif data["type"] == "rl":
                print("  %-8s reward=%.1f (n=%d)" % (tag, data.get("last20_mean", 0), n))
            elif data["type"] == "mixer":
                print("  %-8s %.2f+/-%.2f" % (tag, data.get("acc_mean", 0), data.get("acc_std", 0)))


if __name__ == "__main__":
    summary = scan_experiments()

    # Save JSON
    out_path = "exp/SUMMARY.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved to %s (%d experiments)" % (out_path, len(summary)))

    # Print readable
    print_summary(summary)
