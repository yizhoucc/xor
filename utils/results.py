"""Result loading utilities for XOR Neuron experiments.

Usage:
    from utils.results import load_all_results
    df = load_all_results()              # load from local exp/
    df = load_all_results("path/to/exp") # custom path

    # Filter and analyze
    cnn = df[df.arch == "cnn"]
    cnn.groupby(["dataset", "variant"])["accuracy"].agg(["mean", "std", "count"])

The returned DataFrame has one row per experiment run with columns:
    exp_type, arch, dataset, variant, seed, accuracy, mse, ppl, reward, params, exp_dir
"""
import os
import glob
import pickle
import re
import numpy as np
import pandas as pd
import yaml


def _safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None


def _safe_load_yaml(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except:
        return None


def _parse_exp_name(dirname):
    """Parse experiment directory name into components.

    e.g. 'cnn_cifar_2arg_20260305_085311_e1252721'
      → exp_type='cnn_cifar_2arg', arch='cnn', dataset='cifar', variant='2arg'

    e.g. 'mlp_cifar_scale_relu_w128_20260407_...'
      → exp_type='mlp_cifar_scale_relu_w128', arch='mlp', dataset='cifar', variant='relu_w128'
    """
    # Strip timestamp suffix: _YYYYMMDD_HHMMSS_hash
    m = re.match(r"(.+?)_(\d{8}_\d{6}_[a-f0-9]+)$", dirname)
    exp_type = m.group(1) if m else dirname

    parts = exp_type.split("_")

    # Detect architecture
    arch = parts[0] if parts else "unknown"
    if arch in ("mlp", "cnn", "resnet", "ae", "vit", "mixer", "dqn", "ppo", "lstm", "transformer"):
        pass
    else:
        arch = "unknown"

    # Detect variant (last meaningful part)
    variant = "unknown"
    for v in ["2arg", "1arg", "relu", "gelu", "swiglu", "baseline", "matched", "tanh"]:
        if v in parts:
            # Include everything after dataset up to variant
            idx = parts.index(v)
            variant = "_".join(parts[idx:])
            break

    # Detect dataset
    dataset = "unknown"
    for ds in ["mnist", "cifar", "cifar10", "cifar100", "fmnist", "fashionmnist", "svhn",
                "stl10", "adult", "wine", "housing", "diabetes", "sst2", "agnews",
                "speechcmd", "ecg", "wikitext", "ptb", "cartpole", "acrobot",
                "mountaincar", "lunarlander"]:
        if ds in parts:
            dataset = ds
            break
    # Normalize
    if dataset == "cifar" and "cifar100" not in exp_type:
        dataset = "cifar10"
    if dataset == "fmnist" or dataset == "fashionmnist":
        dataset = "fashionmnist"

    return exp_type, arch, dataset, variant


def _count_params_from_config(config):
    """Estimate parameter count from config (without building model)."""
    # This is approximate; for exact counts use the param_efficiency script
    return None


def load_all_results(exp_dir="exp", include_training_stats=False):
    """Load all experiment results into a pandas DataFrame.

    Args:
        exp_dir: path to experiment directory (default: 'exp')
        include_training_stats: if True, also load train_stats for convergence analysis

    Returns:
        DataFrame with columns:
            exp_type, arch, dataset, variant, seed,
            accuracy, mse, ppl, reward,
            n_seeds (for LM/RL with internal multi-seed),
            exp_dir, config_path
    """
    rows = []

    for entry in sorted(os.listdir(exp_dir)):
        entry_path = os.path.join(exp_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("_") or entry == "pretrain":
            continue

        exp_type, arch, dataset, variant = _parse_exp_name(entry)

        # Load config for seed
        config = _safe_load_yaml(os.path.join(entry_path, "config.yaml"))
        seed = config.get("seed", None) if config else None

        row = {
            "exp_type": exp_type,
            "arch": arch,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "accuracy": None,
            "mse": None,
            "ppl": None,
            "ppl_std": None,
            "reward": None,
            "n_seeds": 1,
            "exp_dir": entry,
        }

        # Classification / regression results
        r = _safe_load_pickle(os.path.join(entry_path, "test_results.p"))
        if r:
            if "test_accuracy" in r:
                row["accuracy"] = r["test_accuracy"]
            if "test_mse" in r:
                row["mse"] = r["test_mse"]

        # Language model results (internal multi-seed)
        r = _safe_load_pickle(os.path.join(entry_path, "lm_results.p"))
        if r:
            ppls = r.get("all_ppl", [])
            if ppls:
                vals = [x[-1] if isinstance(x, list) else x for x in ppls]
                row["ppl"] = float(np.mean(vals))
                row["ppl_std"] = float(np.std(vals))
                row["n_seeds"] = len(vals)

        # RL results (internal multi-seed)
        r = _safe_load_pickle(os.path.join(entry_path, "rl_results.p"))
        if r:
            ms = r.get("mean_scores", [])
            if ms:
                row["reward"] = float(np.mean(ms[-20:])) if len(ms) >= 20 else float(np.mean(ms))
                row["n_seeds"] = len(r.get("seeds", []))

        # Mixer/ViT results (internal multi-seed)
        r = _safe_load_pickle(os.path.join(entry_path, "mixer_results.p"))
        if r:
            aa = r.get("all_acc", [])
            if aa:
                vals = [x[-1] if isinstance(x, list) else x for x in aa]
                row["accuracy"] = float(np.mean(vals))
                row["n_seeds"] = len(vals)

        # Robustness results
        r = _safe_load_pickle(os.path.join(entry_path, "autoattack_results.p"))
        if r:
            row["clean_accuracy"] = r.get("clean_accuracy")
            row["robust_accuracy"] = r.get("robust_accuracy")

        r = _safe_load_pickle(os.path.join(entry_path, "cifar10c_results.p"))
        if r:
            row["cifar10c_overall"] = r.get("overall_mean")

        # Training stats (optional)
        if include_training_stats:
            for phase in ["phase1", "phase2"]:
                ts = _safe_load_pickle(os.path.join(entry_path, f"train_stats_{phase}.p"))
                if ts:
                    row[f"train_stats_{phase}"] = ts

        # Only add if we have some result
        has_result = any(row[k] is not None for k in ["accuracy", "mse", "ppl", "reward"])
        if has_result:
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summary_table(df, metric="accuracy", groupby=["dataset", "variant"]):
    """Generate a summary table with mean±std for each group.

    Args:
        df: DataFrame from load_all_results()
        metric: column to aggregate ('accuracy', 'mse', 'ppl', 'reward')
        groupby: columns to group by

    Returns:
        DataFrame with mean, std, count per group
    """
    valid = df[df[metric].notna()].copy()
    if metric == "accuracy":
        valid[metric] = valid[metric] * 100  # Convert to percentage

    agg = valid.groupby(groupby)[metric].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_mean", f"{metric}_std", "n"]
    agg = agg.round(4)
    return agg


# Convenience functions

def cnn_results(df=None, exp_dir="exp"):
    """Get CNN classification results summary."""
    if df is None:
        df = load_all_results(exp_dir)
    cnn = df[df.arch == "cnn"]
    return summary_table(cnn, "accuracy", ["dataset", "variant"])


def ae_results(df=None, exp_dir="exp"):
    """Get autoencoder results summary."""
    if df is None:
        df = load_all_results(exp_dir)
    ae = df[df.arch == "ae"]
    return summary_table(ae, "mse", ["dataset", "variant"])


def lm_results(df=None, exp_dir="exp"):
    """Get language model results summary."""
    if df is None:
        df = load_all_results(exp_dir)
    lm = df[(df.arch == "transformer") | (df.arch == "lstm")]
    return summary_table(lm, "ppl", ["exp_type"])


def rl_results(df=None, exp_dir="exp"):
    """Get RL results summary."""
    if df is None:
        df = load_all_results(exp_dir)
    rl = df[(df.arch == "dqn") | (df.arch == "ppo")]
    return summary_table(rl, "reward", ["dataset", "variant"])
