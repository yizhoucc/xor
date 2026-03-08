#!/usr/bin/env python3
"""
Exploratory visualizations beyond the paper.

Generates a multi-panel figure with:
  1. Convergence speed: epochs to reach accuracy threshold
  2. Phase1 vs Phase2 improvement (XorNeuron only)
  3. LayerNorm effect: ReLU vs ReLU+LN
  4. All experiments overview (extended experiments included)

Usage:
    python scripts/plot_explore.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    apply_paper_style, load_all_experiments, group_experiments,
    compute_stats, get_training_curves, COLORS, save_fig,
)


def plot_convergence_speed(ax, groups):
    """Bar chart: epochs to reach accuracy threshold."""
    # MLP experiments show the most interesting convergence differences
    configs = [
        ('mlp_mnist_2arg', '2-arg'),
        ('mlp_mnist_1arg', '1-arg'),
        ('mlp_mnist_relu_ln', 'ReLU+LN'),
        ('mlp_mnist_relu', 'ReLU'),
    ]
    thresholds = [0.93, 0.95, 0.97]
    colors_t = ['#a6cee3', '#1f78b4', '#08306b']

    x = np.arange(len(configs))
    width = 0.22

    for j, t in enumerate(thresholds):
        epochs = []
        for exp_name, label in configs:
            if exp_name not in groups:
                epochs.append(0)
                continue
            mean, _ = get_training_curves(groups[exp_name], phase='phase1', key='val_acc')
            if mean is None:
                epochs.append(0)
                continue
            idx = np.where(mean >= t)[0]
            epochs.append(idx[0] + 1 if len(idx) > 0 else len(mean))

        bars = ax.bar(x + (j - 1) * width, epochs, width,
                      label=f'{t*100:.0f}%', color=colors_t[j])

        # Add value labels
        for bar, val in zip(bars, epochs):
            if val > 0 and val < 400:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        str(val), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in configs])
    ax.set_ylabel('Epochs')
    ax.set_title('Convergence Speed (MLP MNIST)')
    ax.legend(title='Threshold', fontsize=7)


def plot_phase_improvement(ax, groups):
    """Phase2 improvement over Phase1 for XorNeuron models."""
    xor_names = []
    deltas = []
    p1_accs = []
    p2_accs = []

    order = [
        'mlp_mnist_2arg', 'mlp_mnist_1arg',
        'mlp_cifar_2arg', 'mlp_cifar_1arg',
        'cnn_mnist_2arg', 'cnn_mnist_1arg',
        'cnn_cifar_2arg', 'cnn_cifar_1arg',
    ]

    display = {
        'mlp_mnist_2arg': 'MLP\nMNIST\n2-arg',
        'mlp_mnist_1arg': 'MLP\nMNIST\n1-arg',
        'mlp_cifar_2arg': 'MLP\nCIFAR\n2-arg',
        'mlp_cifar_1arg': 'MLP\nCIFAR\n1-arg',
        'cnn_mnist_2arg': 'CNN\nMNIST\n2-arg',
        'cnn_mnist_1arg': 'CNN\nMNIST\n1-arg',
        'cnn_cifar_2arg': 'CNN\nCIFAR\n2-arg',
        'cnn_cifar_1arg': 'CNN\nCIFAR\n1-arg',
    }

    for name in order:
        if name not in groups:
            continue
        for e in groups[name]:
            p1 = e.train_stats_phase1
            p2 = e.train_stats_phase2
            if p1 and p2 and 'best_val_acc' in p1 and 'best_val_acc' in p2:
                d = (p2['best_val_acc'][0] - p1['best_val_acc'][0]) * 100
                xor_names.append(display.get(name, name))
                deltas.append(d)
                p1_accs.append(p1['best_val_acc'][0] * 100)
                p2_accs.append(p2['best_val_acc'][0] * 100)

    if not deltas:
        ax.text(0.5, 0.5, 'No Phase2 data', transform=ax.transAxes,
                ha='center', va='center')
        return

    x = np.arange(len(deltas))
    bar_colors = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]
    bars = ax.bar(x, deltas, color=bar_colors, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, deltas):
        offset = 0.05 if val >= 0 else -0.05
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2., val + offset,
                f'{val:+.2f}%', ha='center', va=va, fontsize=6.5)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xor_names, fontsize=6.5)
    ax.set_ylabel('Accuracy Change (%)')
    ax.set_title('Phase 2 Improvement (frozen InnerNet)')


def plot_layernorm_effect(ax, groups, stats):
    """Side-by-side comparison: ReLU vs ReLU+LN."""
    pairs = [
        ('mlp_mnist_relu', 'mlp_mnist_relu_ln', 'MLP\nMNIST'),
        ('mlp_cifar_relu', 'mlp_cifar_relu_ln', 'MLP\nCIFAR'),
    ]

    x = np.arange(len(pairs))
    width = 0.3

    relu_vals = []
    relu_errs = []
    ln_vals = []
    ln_errs = []

    for relu_name, ln_name, label in pairs:
        if relu_name in stats:
            s = stats[relu_name]
            relu_vals.append(s.mean * 100)
            relu_errs.append(s.std * 100)
        else:
            relu_vals.append(0)
            relu_errs.append(0)

        if ln_name in stats:
            s = stats[ln_name]
            ln_vals.append(s.mean * 100)
            ln_errs.append(s.std * 100)
        else:
            ln_vals.append(0)
            ln_errs.append(0)

    bars1 = ax.bar(x - width/2, relu_vals, width, yerr=relu_errs,
                   capsize=3, label='ReLU', color=COLORS['ReLU'], alpha=0.85)
    bars2 = ax.bar(x + width/2, ln_vals, width, yerr=ln_errs,
                   capsize=3, label='ReLU+LN', color=COLORS['ReLU+LN'], alpha=0.85)

    # Add value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    # Add delta annotations
    for i in range(len(pairs)):
        delta = ln_vals[i] - relu_vals[i]
        mid_x = x[i]
        mid_y = max(relu_vals[i], ln_vals[i]) + 3
        ax.annotate(f'{delta:+.1f}%',
                    xy=(mid_x, mid_y), ha='center', fontsize=8,
                    fontweight='bold', color='#d62728')

    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('LayerNorm Effect on Baseline')
    ax.legend()

    y_min = min(relu_vals + ln_vals) - 10
    ax.set_ylim(bottom=max(0, y_min))


def plot_all_experiments_overview(ax, stats):
    """Horizontal bar chart of ALL experiment results."""
    # Sort by metric type, then by value
    acc_exps = [(n, s) for n, s in stats.items() if s.metric == 'accuracy']
    acc_exps.sort(key=lambda x: x[1].mean, reverse=True)

    if not acc_exps:
        return

    names = [n for n, _ in acc_exps]
    means = [s.mean * 100 for _, s in acc_exps]
    stds = [s.std * 100 for _, s in acc_exps]

    # Color by model type
    from plot_utils import get_model_type
    colors = []
    for n, _ in acc_exps:
        mt = get_model_type(n)
        colors.append(COLORS.get(mt, '#999999'))

    y = np.arange(len(names))
    bars = ax.barh(y, means, xerr=stds, capsize=2,
                   color=colors, alpha=0.85, height=0.7,
                   error_kw={'linewidth': 0.8})

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        label = f'{m:.1f}%'
        if s > 0:
            label += f' ±{s:.1f}'
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                label, va='center', fontsize=6.5)

    # Clean up names for display
    display_names = []
    for n in names:
        dn = n.replace('_', ' ').replace('relu ln', 'ReLU+LN').replace('relu', 'ReLU')
        dn = dn.replace('1arg', '1-arg').replace('2arg', '2-arg')
        display_names.append(dn.upper() if len(dn) < 20 else dn)

    ax.set_yticks(y)
    ax.set_yticklabels(display_names, fontsize=6.5)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('All Classification Experiments')
    ax.invert_yaxis()


def main():
    experiments = load_all_experiments('exp')
    clf_exps = [e for e in experiments
                if e.test_accuracy is not None
                and any(k in e.exp_name for k in ['mlp_', 'cnn_'])
                and 'dqn' not in e.exp_name]
    groups = group_experiments(clf_exps)
    stats = compute_stats(groups, metric='accuracy')

    apply_paper_style()

    fig = plt.figure(figsize=(10, 8))

    # 2x2 layout
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_convergence_speed(ax1, groups)
    plot_phase_improvement(ax2, groups)
    plot_layernorm_effect(ax3, groups, stats)
    plot_all_experiments_overview(ax4, stats)

    fig.suptitle('XOR Neuron — Exploratory Analysis', fontsize=12, y=1.01)
    fig.tight_layout()

    save_fig(fig, 'exploratory_analysis')
    plt.show()


if __name__ == '__main__':
    main()
