#!/usr/bin/env python3
"""
Reproduce Figure 4d from Yoon et al. (2021):
  Classification accuracy bar chart — MLP/CNN × MNIST/CIFAR × ReLU/1-arg/2-arg

Also includes ReLU+LayerNorm baselines for fair comparison.

Usage:
    python scripts/plot_fig4d.py
    python scripts/plot_fig4d.py --exp-dir exp --output results/figure_4d.pdf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    apply_paper_style, load_all_experiments, group_experiments,
    compute_stats, COLORS, HATCHES, save_fig, add_value_labels,
)


# Paper reference values (estimated from Figure 4d)
PAPER_REFERENCE = {
    'mlp_mnist_2arg': 98.0,
    'mlp_mnist_1arg': 97.5,
    'mlp_mnist_relu': 97.0,
    'mlp_cifar_2arg': 52.5,
    'mlp_cifar_1arg': 50.0,
    'mlp_cifar_relu': 48.5,
    'cnn_mnist_2arg': 99.0,
    'cnn_mnist_1arg': 98.8,
    'cnn_mnist_relu': 98.5,
    'cnn_cifar_2arg': 72.5,
    'cnn_cifar_1arg': 70.0,
    'cnn_cifar_relu': 68.5,
}


def plot_figure_4d(stats, show_paper_ref=True, show_ln=True):
    """Create Figure 4d bar chart.

    Layout: 4 groups (MLP-MNIST, MLP-CIFAR, CNN-MNIST, CNN-CIFAR)
    Each group has 3-4 bars: 2-arg, 1-arg, ReLU, [ReLU+LN]
    """
    apply_paper_style()

    # Define groups and bar order
    group_defs = [
        ('MLP\nMNIST', 'mlp_mnist'),
        ('MLP\nCIFAR-10', 'mlp_cifar'),
        ('CNN\nMNIST', 'cnn_mnist'),
        ('CNN\nCIFAR-10', 'cnn_cifar'),
    ]

    model_types = ['2-arg', '1-arg', 'ReLU']
    suffixes = ['2arg', '1arg', 'relu']
    if show_ln:
        model_types.append('ReLU+LN')
        suffixes.append('relu_ln')

    n_groups = len(group_defs)
    n_bars = len(model_types)
    bar_width = 0.18
    group_width = n_bars * bar_width + 0.1

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.0))

    x_centers = np.arange(n_groups) * (group_width + 0.3)

    for j, (mtype, suffix) in enumerate(zip(model_types, suffixes)):
        x_positions = x_centers + (j - (n_bars - 1) / 2) * bar_width
        heights = []
        errors = []
        valid_x = []

        for i, (group_label, prefix) in enumerate(group_defs):
            exp_name = f'{prefix}_{suffix}'
            if exp_name in stats:
                s = stats[exp_name]
                heights.append(s.mean * 100)
                errors.append(s.std * 100)
                valid_x.append(x_positions[i])
            else:
                heights.append(0)
                errors.append(0)
                valid_x.append(x_positions[i])

        color = COLORS.get(mtype, '#999999')
        hatch = HATCHES.get(mtype, '')

        bars = ax.bar(valid_x, heights, bar_width,
                      yerr=errors if any(e > 0 for e in errors) else None,
                      capsize=2, color=color, hatch=hatch,
                      edgecolor='white' if not hatch else color,
                      label=mtype, alpha=0.85,
                      error_kw={'linewidth': 0.8})

    # Paper reference markers
    if show_paper_ref:
        for i, (group_label, prefix) in enumerate(group_defs):
            for j, suffix in enumerate(['2arg', '1arg', 'relu']):
                exp_name = f'{prefix}_{suffix}'
                if exp_name in PAPER_REFERENCE:
                    x = x_centers[i] + (j - (n_bars - 1) / 2) * bar_width
                    ax.plot(x, PAPER_REFERENCE[exp_name], '_',
                            color='black', markersize=8, markeredgewidth=1.5,
                            zorder=10)

        # Dummy marker for legend
        ax.plot([], [], '_', color='black', markersize=8,
                markeredgewidth=1.5, label='Paper ref.')

    # Labels and formatting
    ax.set_xticks(x_centers)
    ax.set_xticklabels([g[0] for g in group_defs])
    ax.set_ylabel('Test Accuracy (%)')

    # Break y-axis: start from a reasonable baseline
    all_vals = [s.mean * 100 - s.std * 100
                for s in stats.values() if s.metric == 'accuracy']
    y_min = max(0, min(all_vals) - 10)
    # Round down to nearest 10
    y_min = int(y_min // 10) * 10
    ax.set_ylim(bottom=y_min, top=105)

    ax.legend(loc='lower left', ncol=2 if show_ln else 1)
    ax.set_title('Classification Accuracy (Figure 4d)')

    fig.tight_layout()
    return fig


def print_summary_table(stats):
    """Print a formatted results table."""
    print(f"\n{'Experiment':<25} {'Seeds':>5} {'Mean%':>8} {'Std%':>7} {'Paper%':>7}")
    print('=' * 55)

    order = [
        'mlp_mnist_2arg', 'mlp_mnist_1arg', 'mlp_mnist_relu', 'mlp_mnist_relu_ln',
        'mlp_cifar_2arg', 'mlp_cifar_1arg', 'mlp_cifar_relu', 'mlp_cifar_relu_ln',
        'cnn_mnist_2arg', 'cnn_mnist_1arg', 'cnn_mnist_relu', 'cnn_mnist_relu_ln',
        'cnn_cifar_2arg', 'cnn_cifar_1arg', 'cnn_cifar_relu', 'cnn_cifar_relu_ln',
    ]

    for name in order:
        if name not in stats:
            continue
        s = stats[name]
        paper = PAPER_REFERENCE.get(name, None)
        paper_str = f'{paper:.1f}' if paper else '  —'
        print(f"{name:<25} {s.n_seeds:>5} {s.mean*100:>7.2f}% {s.std*100:>6.2f}% {paper_str:>7}")


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure 4d')
    parser.add_argument('--exp-dir', default='exp')
    parser.add_argument('--output', default='results/figure_4d')
    parser.add_argument('--no-paper-ref', action='store_true')
    parser.add_argument('--no-ln', action='store_true', help='Hide ReLU+LN bars')
    args = parser.parse_args()

    # Load data
    experiments = load_all_experiments(args.exp_dir)
    if not experiments:
        print(f"No completed experiments found in {args.exp_dir}/")
        return

    # Filter to classification experiments only
    clf_exps = [e for e in experiments
                if e.test_accuracy is not None
                and any(k in e.exp_name for k in ['mlp_', 'cnn_'])
                and 'dqn' not in e.exp_name]

    groups = group_experiments(clf_exps)
    stats = compute_stats(groups, metric='accuracy')

    print_summary_table(stats)

    fig = plot_figure_4d(stats,
                         show_paper_ref=not args.no_paper_ref,
                         show_ln=not args.no_ln)
    save_fig(fig, os.path.basename(args.output),
             output_dir=os.path.dirname(args.output) or 'results')
    plt.show()


if __name__ == '__main__':
    main()
