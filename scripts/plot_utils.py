"""
Visualization utilities for XOR Neuron paper reproduction.

Provides:
  - Paper-quality matplotlib styling (apply_paper_style)
  - Experiment data loading and grouping (load_all_experiments, group_experiments)
  - Multi-seed statistics (compute_stats)
  - Standard color/label mappings for model types

Usage:
    from plot_utils import apply_paper_style, load_all_experiments, COLORS

    apply_paper_style()
    experiments = load_all_experiments('exp/')
    groups = group_experiments(experiments)
"""

import os
import pickle
import math
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# 1. Paper-quality style
# ============================================================

def apply_paper_style():
    """Apply publication-quality matplotlib defaults (NeurIPS / ICML style)."""
    mpl.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 4,

        # Axes
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # PDF text as text (not paths), for editability
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
    })


# ============================================================
# 2. Color and label mappings
# ============================================================

# Primary model colors (consistent across all figures)
COLORS = {
    '2-arg':      '#1f77b4',  # blue
    '1-arg':      '#ff7f0e',  # orange
    'ReLU':       '#2ca02c',  # green
    'ReLU+LN':    '#d62728',  # red
    'tanh':       '#9467bd',  # purple
    'GELU':       '#8c564b',  # brown
    'SwiGLU':     '#e377c2',  # pink
    'InnerNet':   '#1f77b4',  # blue (same as 2-arg)
    'Baseline':   '#2ca02c',  # green (same as ReLU)
}

# Hatching patterns for bar charts
HATCHES = {
    '2-arg':    '',
    '1-arg':    '//',
    'ReLU':     '',
    'ReLU+LN':  'xx',
}

# Display labels (for legends)
LABELS = {
    'mlp_mnist_2arg':     'MLP MNIST 2-arg',
    'mlp_mnist_1arg':     'MLP MNIST 1-arg',
    'mlp_mnist_relu':     'MLP MNIST ReLU',
    'mlp_mnist_relu_ln':  'MLP MNIST ReLU+LN',
    'mlp_cifar_2arg':     'MLP CIFAR 2-arg',
    'mlp_cifar_1arg':     'MLP CIFAR 1-arg',
    'mlp_cifar_relu':     'MLP CIFAR ReLU',
    'mlp_cifar_relu_ln':  'MLP CIFAR ReLU+LN',
    'cnn_mnist_2arg':     'CNN MNIST 2-arg',
    'cnn_mnist_1arg':     'CNN MNIST 1-arg',
    'cnn_mnist_relu':     'CNN MNIST ReLU',
    'cnn_mnist_relu_ln':  'CNN MNIST ReLU+LN',
    'cnn_cifar_2arg':     'CNN CIFAR 2-arg',
    'cnn_cifar_1arg':     'CNN CIFAR 1-arg',
    'cnn_cifar_relu':     'CNN CIFAR ReLU',
    'cnn_cifar_relu_ln':  'CNN CIFAR ReLU+LN',
}


def get_model_type(exp_name: str) -> str:
    """Extract model type from experiment name for color/hatch lookup."""
    if 'relu_ln' in exp_name:
        return 'ReLU+LN'
    elif 'relu' in exp_name:
        return 'ReLU'
    elif '2arg' in exp_name:
        return '2-arg'
    elif '1arg' in exp_name:
        return '1-arg'
    elif 'tanh' in exp_name:
        return 'tanh'
    elif 'swiglu' in exp_name:
        return 'SwiGLU'
    elif 'gelu' in exp_name or 'baseline' in exp_name:
        return 'GELU'
    return 'unknown'


def get_arch_dataset(exp_name: str) -> Tuple[str, str]:
    """Extract (architecture, dataset) from experiment name."""
    parts = exp_name.split('_')
    arch = parts[0].upper() if parts else '?'
    dataset = parts[1].upper() if len(parts) > 1 else '?'
    return arch, dataset


# ============================================================
# 3. Data loading
# ============================================================

@dataclass
class ExperimentResult:
    """Container for one experiment run."""
    exp_name: str           # e.g. "mlp_mnist_2arg"
    exp_dir: str            # full path to experiment directory
    seed: int = 0
    test_accuracy: Optional[float] = None
    test_loss: Optional[float] = None    # for LM experiments → PPL = exp(loss)
    train_stats_phase1: Optional[dict] = None
    train_stats_phase2: Optional[dict] = None
    config: Optional[dict] = None

    @property
    def test_ppl(self) -> Optional[float]:
        if self.test_loss is not None:
            return math.exp(self.test_loss)
        return None


def _parse_exp_name(dirname: str) -> str:
    """Extract experiment name from directory name.

    e.g. "mlp_mnist_2arg_20260301_145935_59238531" → "mlp_mnist_2arg"
    """
    parts = dirname.split('_')
    name_parts = []
    for p in parts:
        if len(p) == 8 and p.isdigit():
            break
        name_parts.append(p)
    return '_'.join(name_parts)


def load_experiment(exp_dir: str) -> Optional[ExperimentResult]:
    """Load a single experiment result from its directory."""
    dirname = os.path.basename(exp_dir)
    exp_name = _parse_exp_name(dirname)

    # Must be completed
    if not os.path.exists(os.path.join(exp_dir, 'COMPLETED')):
        return None

    result = ExperimentResult(exp_name=exp_name, exp_dir=exp_dir)

    # Config (seed)
    config_path = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            result.config = yaml.safe_load(f)
        result.seed = result.config.get('seed', 0)

    # Test results
    test_path = os.path.join(exp_dir, 'test_results.p')
    if os.path.exists(test_path):
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        if 'test_accuracy' in test_data:
            result.test_accuracy = test_data['test_accuracy']
        if 'test_loss' in test_data:
            result.test_loss = test_data['test_loss']

    # Training stats
    for phase in ['phase1', 'phase2']:
        stats_path = os.path.join(exp_dir, f'train_stats_{phase}.p')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            if phase == 'phase1':
                result.train_stats_phase1 = stats
            else:
                result.train_stats_phase2 = stats

    return result


def load_all_experiments(exp_base: str = 'exp',
                         filter_name: Optional[str] = None) -> List[ExperimentResult]:
    """Load all completed experiments from the experiment directory.

    Args:
        exp_base: base experiment directory
        filter_name: substring filter on experiment name

    Returns:
        list of ExperimentResult
    """
    results = []
    if not os.path.isdir(exp_base):
        return results

    for entry in sorted(os.listdir(exp_base)):
        entry_path = os.path.join(exp_base, entry)
        if not os.path.isdir(entry_path):
            continue

        exp = load_experiment(entry_path)
        if exp is None:
            continue

        if filter_name and filter_name not in exp.exp_name:
            continue

        results.append(exp)

    return results


def group_experiments(experiments: List[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """Group experiments by name (across seeds)."""
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.exp_name].append(exp)
    return dict(groups)


# ============================================================
# 4. Statistics
# ============================================================

@dataclass
class ExpStats:
    """Aggregated statistics for one experiment type across seeds."""
    exp_name: str
    n_seeds: int
    values: List[float]
    mean: float
    std: float
    metric: str  # 'accuracy', 'ppl', 'reward'


def compute_stats(groups: Dict[str, List[ExperimentResult]],
                  metric: str = 'auto') -> Dict[str, ExpStats]:
    """Compute mean ± std for each experiment group.

    Args:
        groups: output of group_experiments()
        metric: 'accuracy', 'ppl', or 'auto' (detect from data)

    Returns:
        dict mapping exp_name → ExpStats
    """
    stats = {}
    for exp_name, exps in groups.items():
        values = []
        detected_metric = metric

        for exp in exps:
            if metric == 'auto':
                if exp.test_accuracy is not None:
                    detected_metric = 'accuracy'
                elif exp.test_loss is not None:
                    detected_metric = 'ppl'
                else:
                    continue

            if detected_metric == 'accuracy' and exp.test_accuracy is not None:
                values.append(exp.test_accuracy)
            elif detected_metric == 'ppl' and exp.test_loss is not None:
                values.append(math.exp(exp.test_loss))

        if not values:
            continue

        arr = np.array(values)
        stats[exp_name] = ExpStats(
            exp_name=exp_name,
            n_seeds=len(values),
            values=values,
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            metric=detected_metric,
        )

    return stats


# ============================================================
# 5. Plotting helpers
# ============================================================

def add_value_labels(ax, bars, fmt='{:.1f}', fontsize=7, offset=0.5):
    """Add value labels on top of bar chart bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                fmt.format(height),
                ha='center', va='bottom', fontsize=fontsize)


def save_fig(fig, name: str, output_dir: str = 'results', formats=('png', 'pdf')):
    """Save figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f'{name}.{fmt}')
        fig.savefig(path)
        print(f'Saved: {path}')


def get_training_curves(exps: List[ExperimentResult],
                        phase: str = 'phase1',
                        key: str = 'val_acc') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract training curves from multiple runs, return (mean, std) arrays.

    Handles unequal lengths by truncating to the shortest.

    Args:
        exps: list of experiments (same exp_name, different seeds)
        phase: 'phase1' or 'phase2'
        key: 'val_acc', 'val_loss', 'train_loss'

    Returns:
        (mean_curve, std_curve) as numpy arrays, or (None, None)
    """
    curves = []
    for exp in exps:
        stats = exp.train_stats_phase1 if phase == 'phase1' else exp.train_stats_phase2
        if stats is None or key not in stats:
            continue
        curves.append(np.array(stats[key]))

    if not curves:
        return None, None

    # Truncate to shortest
    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    return curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(min_len)


def plot_training_curves(ax, groups: Dict[str, List[ExperimentResult]],
                         exp_names: List[str],
                         phase: str = 'phase1',
                         key: str = 'val_acc',
                         ylabel: str = 'Validation Accuracy',
                         alpha_fill: float = 0.2):
    """Plot mean ± std training curves on a given axes.

    Args:
        ax: matplotlib axes
        groups: grouped experiments
        exp_names: which experiment names to plot
        phase: 'phase1' or 'phase2'
        key: 'val_acc' or 'val_loss'
        ylabel: y-axis label
        alpha_fill: opacity for std shading
    """
    for exp_name in exp_names:
        if exp_name not in groups:
            continue

        mean, std = get_training_curves(groups[exp_name], phase=phase, key=key)
        if mean is None:
            continue

        model_type = get_model_type(exp_name)
        color = COLORS.get(model_type, '#333333')
        label = LABELS.get(exp_name, exp_name)
        epochs = np.arange(1, len(mean) + 1)

        ax.plot(epochs, mean, color=color, label=label)
        if std is not None and np.any(std > 0):
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=alpha_fill)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()
