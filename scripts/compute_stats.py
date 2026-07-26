"""Statistical comparison of multi-seed experiment results.

The input JSON must contain metric arrays under each condition::

    {"results": {"<condition>": {"<metric>": [seed values...]}, ...}}

Use ``--paired`` when conditions share aligned seeds or branch from the same
checkpoint. Paired mode reports a paired t-test, Wilcoxon signed-rank test,
Cohen's dz, and a bootstrap confidence interval over within-seed differences.
The default remains an independent-samples comparison for compatibility.

Examples::

    .venv/bin/python scripts/compute_stats.py results.json \
        --metric ppl --ref innernet --lower_better --paired
    .venv/bin/python scripts/compute_stats.py results.json \
        --metric acc --ref innernet
"""
import argparse
import json

import numpy as np
from scipy import stats


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def cohens_d(x, y):
    """Pooled-standard-deviation effect size for independent samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = (
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)) if pooled_var > 0 else float("nan")


def cohens_dz(differences):
    """Within-pair effect size: mean difference / SD of differences."""
    if len(differences) < 2:
        return float("nan")
    sd = np.std(differences, ddof=1)
    return float(np.mean(differences) / sd) if sd > 0 else float("nan")


def _bootstrap_ci(x, y, paired, confidence, n_bootstrap, rng):
    """Bootstrap CI for mean(x) - mean(y)."""
    if n_bootstrap <= 0:
        return float("nan"), float("nan")

    if paired:
        differences = x - y
        indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
        samples = differences[indices].mean(axis=1)
    else:
        x_indices = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
        y_indices = rng.integers(0, len(y), size=(n_bootstrap, len(y)))
        samples = x[x_indices].mean(axis=1) - y[y_indices].mean(axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def prepare_paired(x, y):
    """Drop non-finite observations pairwise and report how many were dropped."""
    if len(x) != len(y):
        raise ValueError(f"paired comparison requires equal lengths, got {len(x)} and {len(y)}")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], int((~keep).sum())


def prepare_independent(x, y):
    """Drop non-finite observations independently and report both counts."""
    x_keep = np.isfinite(x)
    y_keep = np.isfinite(y)
    return x[x_keep], y[y_keep], int((~x_keep).sum()), int((~y_keep).sum())


def paired_comparison(x, ref, confidence=0.95, n_bootstrap=10000, rng=None):
    x, ref, dropped = prepare_paired(np.asarray(x, dtype=float), np.asarray(ref, dtype=float))
    if len(x) < 2:
        raise ValueError("paired comparison needs at least two finite pairs")
    differences = x - ref
    rng = np.random.default_rng() if rng is None else rng
    try:
        wilcoxon_p = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    ci_low, ci_high = _bootstrap_ci(x, ref, True, confidence, n_bootstrap, rng)
    return {
        "n": len(x),
        "dropped": dropped,
        "mean_difference": float(np.mean(differences)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "parametric_p": float(stats.ttest_rel(x, ref).pvalue),
        "nonparametric_p": wilcoxon_p,
        "effect_size": cohens_dz(differences),
        "effect_name": "Cohen dz",
    }


def independent_comparison(x, ref, confidence=0.95, n_bootstrap=10000, rng=None):
    x, ref, dropped_x, dropped_ref = prepare_independent(
        np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    )
    if len(x) < 2 or len(ref) < 2:
        raise ValueError("independent comparison needs at least two finite values per group")
    rng = np.random.default_rng() if rng is None else rng
    ci_low, ci_high = _bootstrap_ci(x, ref, False, confidence, n_bootstrap, rng)
    return {
        "n": len(x),
        "n_ref": len(ref),
        "dropped": dropped_x,
        "dropped_ref": dropped_ref,
        "mean_difference": float(np.mean(x) - np.mean(ref)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "parametric_p": float(stats.ttest_ind(x, ref, equal_var=False).pvalue),
        "nonparametric_p": float(stats.mannwhitneyu(x, ref, alternative="two-sided").pvalue),
        "effect_size": cohens_d(x, ref),
        "effect_name": "Cohen d",
    }


def _format_verdict(mean_difference, lower_better, significant):
    condition_better = mean_difference < 0 if lower_better else mean_difference > 0
    verdict = "condition better" if condition_better else "reference better"
    return verdict + (" (p<0.05)" if significant else " (n.s.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("--metric", default="ppl")
    parser.add_argument("--ref", default="innernet", help="reference condition")
    parser.add_argument("--lower_better", action="store_true")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="arrays are aligned observations from the same seeds/checkpoints",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--show_values", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")

    with open(args.json) as handle:
        data = json.load(handle)
    results = data["results"]
    values = {
        condition: np.asarray(result[args.metric], dtype=float)
        for condition, result in results.items()
    }
    if args.ref not in values:
        parser.error(f"reference condition '{args.ref}' is not present")

    mode = "paired" if args.paired else "independent"
    direction = "lower" if args.lower_better else "higher"
    print(f"File: {args.json}  metric={args.metric}  {direction} better  mode={mode}")
    print(f"{'condition':<14} {'n':>3} {'mean':>10} {'sample SD':>10}")
    print("-" * 43)
    for condition, raw in values.items():
        finite = raw[np.isfinite(raw)]
        print(f"{condition:<14} {len(finite):>3} {finite.mean():>10.4f} {_sample_std(finite):>10.4f}")
        if args.show_values:
            print(f"  values={raw.tolist()}")

    reference = values[args.ref]
    ci_pct = 100.0 * args.confidence
    test_names = "paired-t / Wilcoxon" if args.paired else "Welch-t / MWU"
    print(f"\nComparisons against '{args.ref}' ({test_names}; {ci_pct:g}% bootstrap CI):")
    print(
        f"{'condition':<14} {'n':>3} {'mean diff':>10} {'CI low':>10} {'CI high':>10} "
        f"{'t p':>9} {'nonpar p':>10} {'effect':>9}  verdict"
    )
    print("-" * 112)

    rng = np.random.default_rng(args.random_seed)
    for condition, raw in values.items():
        if condition == args.ref:
            continue
        comparison = paired_comparison(
            raw, reference, args.confidence, args.bootstrap, rng
        ) if args.paired else independent_comparison(
            raw, reference, args.confidence, args.bootstrap, rng
        )
        dropped = comparison["dropped"]
        if args.paired and dropped:
            print(f"note: {condition}: dropped {dropped} non-finite pair(s)")
        if not args.paired and (dropped or comparison["dropped_ref"]):
            print(
                f"note: {condition}: dropped {dropped} condition and "
                f"{comparison['dropped_ref']} reference non-finite value(s)"
            )
        verdict = _format_verdict(
            comparison["mean_difference"],
            args.lower_better,
            comparison["parametric_p"] < 0.05,
        )
        print(
            f"{condition:<14} {comparison['n']:>3} "
            f"{comparison['mean_difference']:>10.4f} "
            f"{comparison['ci_low']:>10.4f} {comparison['ci_high']:>10.4f} "
            f"{comparison['parametric_p']:>9.4f} "
            f"{comparison['nonparametric_p']:>10.4f} "
            f"{comparison['effect_size']:>9.2f}  {verdict}"
        )


if __name__ == "__main__":
    main()
