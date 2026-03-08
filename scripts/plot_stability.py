#!/usr/bin/env python3
"""
Stability analysis: box/strip plots showing seed-to-seed variance.

Usage:
    python scripts/plot_stability.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    apply_paper_style, load_all_experiments, group_experiments,
    compute_stats, COLORS, get_model_type, save_fig,
)


def main():
    experiments = load_all_experiments('exp')
    clf_exps = [e for e in experiments
                if e.test_accuracy is not None
                and any(k in e.exp_name for k in ['mlp_', 'cnn_'])
                and 'dqn' not in e.exp_name]
    groups = group_experiments(clf_exps)
    stats = compute_stats(groups, metric='accuracy')

    apply_paper_style()

    # Focus on MLP experiments that have multi-seed data
    panels = [
        ('MLP — MNIST', ['mlp_mnist_2arg', 'mlp_mnist_1arg', 'mlp_mnist_relu_ln', 'mlp_mnist_relu']),
        ('MLP — CIFAR-10', ['mlp_cifar_2arg', 'mlp_cifar_1arg', 'mlp_cifar_relu_ln', 'mlp_cifar_relu']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, (title, exp_names) in zip(axes, panels):
        positions = []
        data = []
        labels = []
        colors = []

        for i, name in enumerate(exp_names):
            if name not in stats:
                continue
            s = stats[name]
            vals = [v * 100 for v in s.values]
            data.append(vals)
            positions.append(i)
            mt = get_model_type(name)
            labels.append(mt)
            colors.append(COLORS.get(mt, '#999'))

        if not data:
            continue

        # Strip plot (individual points) + box
        bp = ax.boxplot(data, positions=positions, widths=0.4,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Overlay individual seed points with jitter
        for i, (pos, vals) in enumerate(zip(positions, data)):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter([pos] * len(vals) if len(vals) == 1 else pos + jitter,
                       vals, color=colors[i], s=30, zorder=5,
                       edgecolors='white', linewidths=0.5)
            # Add mean ± std text
            m = np.mean(vals)
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0
            n = len(vals)
            label = f'{m:.1f}%'
            if s > 0:
                label += f'\n±{s:.1f} (n={n})'
            else:
                label += f'\n(n={n})'
            ax.text(pos, max(vals) + 1.0, label,
                    ha='center', va='bottom', fontsize=6.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(title)

        # Adjust y limits
        all_vals = [v for d in data for v in d]
        y_min = min(all_vals) - 5
        y_max = max(all_vals) + 5
        ax.set_ylim(y_min, y_max)

    fig.suptitle('Seed Stability Analysis', fontsize=11)
    fig.tight_layout()
    save_fig(fig, 'seed_stability')
    plt.show()


if __name__ == '__main__':
    main()
