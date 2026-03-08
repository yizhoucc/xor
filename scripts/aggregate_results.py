#!/usr/bin/env python3
"""Aggregate multi-seed experiment results.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --filter mlp_mnist
"""
import os
import pickle
import math
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default='exp', help='Experiment base directory')
    parser.add_argument('--filter', default=None, help='Filter experiment names (substring match)')
    args = parser.parse_args()

    # Group experiments by name (strip timestamp and hash suffix)
    groups = defaultdict(list)

    for entry in sorted(os.listdir(args.exp_dir)):
        entry_path = os.path.join(args.exp_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if not os.path.exists(os.path.join(entry_path, 'COMPLETED')):
            continue

        # Extract experiment name: e.g. "mlp_mnist_2arg" from "mlp_mnist_2arg_20260301_145935_59238531"
        parts = entry.split('_')
        # Find the timestamp part (8 digits)
        name_parts = []
        for p in parts:
            if len(p) == 8 and p.isdigit():
                break
            name_parts.append(p)
        exp_name = '_'.join(name_parts)

        if args.filter and args.filter not in exp_name:
            continue

        # Read seed from config
        config_path = os.path.join(entry_path, 'config.yaml')
        seed = None
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            seed = cfg.get('seed', '?')

        # Read test result
        test_result_path = os.path.join(entry_path, 'test_results.p')
        lm_result_path = os.path.join(entry_path, 'lm_results.p')
        rl_result_path = os.path.join(entry_path, 'rl_results.p')

        result = None
        if os.path.exists(test_result_path):
            with open(test_result_path, 'rb') as f:
                r = pickle.load(f)
            if 'test_accuracy' in r:
                result = {'type': 'acc', 'value': r['test_accuracy']}
            elif 'test_loss' in r:
                result = {'type': 'ppl', 'value': math.exp(r['test_loss'])}
        elif os.path.exists(lm_result_path):
            with open(lm_result_path, 'rb') as f:
                r = pickle.load(f)
            if 'best_mean_ppl' in r:
                result = {'type': 'ppl', 'value': r['best_mean_ppl']}
        elif os.path.exists(rl_result_path):
            with open(rl_result_path, 'rb') as f:
                r = pickle.load(f)
            if 'all_scores' in r:
                seed_avgs = []
                for scores in r['all_scores']:
                    last100 = scores[-100:]
                    seed_avgs.append(sum(last100) / len(last100))
                result = {'type': 'reward', 'value': sum(seed_avgs) / len(seed_avgs)}

        if result:
            groups[exp_name].append({'seed': seed, 'result': result, 'dir': entry})

    # Print results
    print(f"{'Experiment':<30} {'Seeds':>6} {'Mean':>10} {'Std':>8} {'Values'}")
    print('=' * 90)

    for exp_name in sorted(groups.keys()):
        entries = groups[exp_name]
        values = [e['result']['value'] for e in entries]
        result_type = entries[0]['result']['type']

        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0

        if result_type == 'acc':
            val_str = ', '.join(f"{v*100:.2f}%" for v in values)
            print(f"{exp_name:<30} {n:>6} {mean*100:>9.2f}% {std*100:>7.2f}% {val_str}")
        elif result_type == 'ppl':
            val_str = ', '.join(f"{v:.2f}" for v in values)
            print(f"{exp_name:<30} {n:>6} {mean:>9.2f}  {std:>7.2f}  {val_str}")
        elif result_type == 'reward':
            val_str = ', '.join(f"{v:.1f}" for v in values)
            print(f"{exp_name:<30} {n:>6} {mean:>9.1f}  {std:>7.1f}  {val_str}")


if __name__ == '__main__':
    main()
