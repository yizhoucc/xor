#!/usr/bin/env python3
"""
Early convergence comparison: zoomed-in first N epochs.

Shows that XorNeuron models converge faster in early training,
which is a key claim of the paper.

Usage:
    python scripts/plot_early_convergence.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    apply_paper_style, load_all_experiments, group_experiments,
    get_training_curves, COLORS, get_model_type, LABELS, save_fig,
)


def main():
    experiments = load_all_experiments('exp')
    clf_exps = [e for e in experiments
                if e.test_accuracy is not None
                and any(k in e.exp_name for k in ['mlp_', 'cnn_'])
                and 'dqn' not in e.exp_name]
    groups = group_experiments(clf_exps)

    apply_paper_style()

    # Zoom into first 50 epochs
    zoom_epochs = 50

    panels = [
        ('MLP — MNIST (first 50 epochs)', 'mlp_mnist', (0.5, 1.0)),
        ('MLP — CIFAR-10 (first 50 epochs)', 'mlp_cifar', (0.1, 0.55)),
        ('CNN — MNIST (first 50 epochs)', 'cnn_mnist', (0.9, 1.0)),
        ('CNN — CIFAR-10 (first 50 epochs)', 'cnn_cifar', (0.3, 0.82)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    for ax, (title, prefix, ylim) in zip(axes, panels):
        exp_names = [f'{prefix}_2arg', f'{prefix}_1arg',
                     f'{prefix}_relu_ln', f'{prefix}_relu']

        for exp_name in exp_names:
            if exp_name not in groups:
                continue

            mean, std = get_training_curves(groups[exp_name],
                                             phase='phase1', key='val_acc')
            if mean is None:
                continue

            # Truncate to zoom window
            n = min(zoom_epochs, len(mean))
            mean = mean[:n]
            epochs = np.arange(1, n + 1)

            mt = get_model_type(exp_name)
            color = COLORS.get(mt, '#333')
            label = LABELS.get(exp_name, exp_name)
            # Shorten labels
            label = label.split(' ')[-1]  # just "2-arg", "1-arg", etc.

            ax.plot(epochs, mean, color=color, label=label, linewidth=1.5)
            if std is not None and n <= len(std):
                std_z = std[:n]
                if np.any(std_z > 0):
                    ax.fill_between(epochs, mean - std_z, mean + std_z,
                                    color=color, alpha=0.15)

        ax.set_xlim(1, zoom_epochs)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val. Accuracy')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    save_fig(fig, 'early_convergence')
    plt.show()


if __name__ == '__main__':
    main()
