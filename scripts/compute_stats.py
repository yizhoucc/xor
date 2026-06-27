"""Statistical comparison of multi-seed results (PROJECT_STATUS P3).

Reads a deploy/experiment results.json of the form
  {"results": {"<op>": {"<metric>": [seed values...]}, ...}}
and, for the chosen metric, reports per-op mean±std plus pairwise comparisons
against a reference op: difference, Welch's t-test p-value, Mann-Whitney U
p-value, and Cohen's d effect size.

Usage:
  .venv/bin/python scripts/compute_stats.py exp/deploy_ffn_d128/results.json \
      --metric ppl --ref innernet --lower_better
  .venv/bin/python scripts/compute_stats.py exp/deploy_cnn_cifar10/results.json \
      --metric acc --ref innernet
"""
import argparse
import json

import numpy as np
from scipy import stats


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
                 / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    ap.add_argument('--metric', default='ppl')
    ap.add_argument('--ref', default='innernet', help='reference op to compare others to')
    ap.add_argument('--lower_better', action='store_true',
                    help='metric where lower is better (e.g. PPL)')
    args = ap.parse_args()

    data = json.load(open(args.json))
    res = data['results']
    ops = list(res.keys())
    vals = {op: np.array(res[op][args.metric], dtype=float) for op in ops}

    print(f"File: {args.json}  metric={args.metric}  "
          f"({'lower' if args.lower_better else 'higher'} better)")
    print(f"{'op':<12} {'n':>3} {'mean':>10} {'std':>8}")
    print("-" * 40)
    for op in ops:
        v = vals[op]
        print(f"{op:<12} {len(v):>3} {v.mean():>10.4f} {v.std(ddof=1):>8.4f}")

    if args.ref not in vals:
        print(f"\n(ref op '{args.ref}' not present; skipping pairwise tests)")
        return

    ref = vals[args.ref]
    print(f"\nPairwise vs '{args.ref}' (mean {ref.mean():.4f}):")
    cohen_hdr = 'Cohen d'
    print(f"{'op':<12} {'dmean':>10} {'t p':>10} {'MWU p':>10} "
          f"{cohen_hdr:>9}  verdict")
    print("-" * 62)
    for op in ops:
        if op == args.ref:
            continue
        v = vals[op]
        dmean = v.mean() - ref.mean()
        # Welch t-test (unequal variance) + non-parametric Mann-Whitney
        tp = stats.ttest_ind(v, ref, equal_var=False).pvalue
        try:
            mwu = stats.mannwhitneyu(v, ref, alternative='two-sided').pvalue
        except ValueError:
            mwu = float('nan')
        d = cohens_d(v, ref)
        better = (dmean < 0) if args.lower_better else (dmean > 0)
        sig = tp < 0.05
        verdict = ("ref better" if not better else "op better")
        verdict += " (sig)" if sig else " (n.s.)"
        print(f"{op:<12} {dmean:>10.4f} {tp:>10.4f} {mwu:>10.4f} "
              f"{d:>9.2f}  {verdict}")


if __name__ == '__main__':
    main()
