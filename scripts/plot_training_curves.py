#!/usr/bin/env python3
"""
Plot training curves: validation accuracy vs epoch.

Shows mean ± std shade for multi-seed experiments.
Layout: 2×2 grid (MLP-MNIST, MLP-CIFAR, CNN-MNIST, CNN-CIFAR).

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --phase phase1
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    apply_paper_style, load_all_experiments, group_experiments,
    plot_training_curves, save_fig,
)


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--exp-dir', default='exp')
    parser.add_argument('--output', default='results/training_curves')
    parser.add_argument('--phase', default='phase1', choices=['phase1', 'phase2'])
    parser.add_argument('--show-ln', action='store_true', help='Include ReLU+LN')
    args = parser.parse_args()

    experiments = load_all_experiments(args.exp_dir)
    clf_exps = [e for e in experiments
                if e.test_accuracy is not None
                and any(k in e.exp_name for k in ['mlp_', 'cnn_'])
                and 'dqn' not in e.exp_name]
    groups = group_experiments(clf_exps)

    apply_paper_style()

    # 2×2 grid
    panels = [
        ('MLP — MNIST', 'mlp_mnist'),
        ('MLP — CIFAR-10', 'mlp_cifar'),
        ('CNN — MNIST', 'cnn_mnist'),
        ('CNN — CIFAR-10', 'cnn_cifar'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    axes = axes.flatten()

    for ax, (title, prefix) in zip(axes, panels):
        exp_names = [f'{prefix}_2arg', f'{prefix}_1arg', f'{prefix}_relu']
        if args.show_ln:
            exp_names.append(f'{prefix}_relu_ln')

        plot_training_curves(ax, groups, exp_names,
                             phase=args.phase,
                             key='val_acc',
                             ylabel='Val. Accuracy',
                             alpha_fill=0.15)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Set reasonable y-limits per dataset
        if 'mnist' in prefix:
            ax.set_ylim(0.85, 1.0)
        else:
            ax.set_ylim(0.3, 0.85)

    fig.tight_layout()
    save_fig(fig, os.path.basename(args.output),
             output_dir=os.path.dirname(args.output) or 'results')
    plt.show()


if __name__ == '__main__':
    main()
