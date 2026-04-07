"""Evaluate corruption robustness on CIFAR-10-C for trained CNN models.

Usage:
    python scripts/eval_cifar10c.py --exp_dir exp/cnn_cifar_2arg_20260305_...

Downloads CIFAR-10-C if needed and evaluates across all 15 corruption types × 5 severities.
"""
import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_helper import load_model
import yaml


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


def load_config_and_model(exp_dir):
    """Load config and instantiate model from experiment directory."""
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    sys.path.insert(0, '.')
    from easydict import EasyDict
    config = EasyDict(config_dict)
    config.save_dir = exp_dir

    import model as model_module
    model_cls = getattr(model_module, config.model.name)
    model = model_cls(config)

    has_inner = config.model.name not in ('BaselineMLP', 'BaselineCNN', 'BaselineRNN')
    if has_inner:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase2.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')
    else:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')

    load_model(model, model_path)
    return config, model


def download_cifar10c(data_path):
    """Download CIFAR-10-C dataset if not present."""
    c10c_dir = os.path.join(data_path, 'CIFAR-10-C')
    if os.path.exists(os.path.join(c10c_dir, 'labels.npy')):
        return c10c_dir

    os.makedirs(c10c_dir, exist_ok=True)
    import urllib.request
    url = 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar'
    tar_path = os.path.join(data_path, 'CIFAR-10-C.tar')

    if not os.path.exists(tar_path):
        print(f"Downloading CIFAR-10-C (~2.7GB)...")
        urllib.request.urlretrieve(url, tar_path)

    import tarfile
    print("Extracting...")
    with tarfile.open(tar_path) as t:
        t.extractall(data_path)

    return c10c_dir


def evaluate_corruption(model, x_corrupt, labels, device, batch_size=100):
    """Evaluate accuracy on a corruption set."""
    model.eval()
    correct = 0
    total = 0

    normalize = transforms.Normalize([0.5]*3, [0.5]*3)

    for i in range(0, len(x_corrupt), batch_size):
        batch_x = x_corrupt[i:i+batch_size]
        batch_y = labels[i:i+batch_size]

        # Convert to tensor: (N, H, W, C) uint8 → (N, C, H, W) float [0,1] → normalized
        x_t = torch.from_numpy(batch_x).permute(0, 3, 1, 2).float() / 255.0
        x_t = torch.stack([normalize(img) for img in x_t])
        y_t = torch.from_numpy(batch_y).long()

        x_t = x_t.to(device)
        y_t = y_t.to(device)

        with torch.no_grad():
            dummy_labels = y_t
            out, _, _ = model(x_t, dummy_labels)
            _, pred = torch.max(out, 1)
            correct += (pred == y_t).sum().item()
            total += len(batch_y)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config, model = load_config_and_model(args.exp_dir)
    model = model.to(device)
    model.eval()

    c10c_dir = download_cifar10c(args.data_path)
    labels = np.load(os.path.join(c10c_dir, 'labels.npy'))

    results = {}
    for corruption in CORRUPTIONS:
        filepath = os.path.join(c10c_dir, f'{corruption}.npy')
        if not os.path.exists(filepath):
            print(f"  {corruption}: file not found, skipping")
            continue

        x_corrupt = np.load(filepath)  # (50000, 32, 32, 3)

        severity_accs = []
        for severity in range(5):
            start = severity * 10000
            end = (severity + 1) * 10000
            acc = evaluate_corruption(model, x_corrupt[start:end],
                                       labels[start:end], device, args.batch_size)
            severity_accs.append(acc)

        mean_acc = np.mean(severity_accs)
        results[corruption] = {
            'severity_accs': severity_accs,
            'mean_acc': mean_acc
        }
        print(f"  {corruption}: {mean_acc:.4f}  ({[f'{a:.3f}' for a in severity_accs]})")

    # Overall mean
    all_means = [r['mean_acc'] for r in results.values()]
    overall = np.mean(all_means) if all_means else 0
    results['overall_mean'] = overall
    print(f"\nOverall mean accuracy: {overall:.4f}")

    out_path = os.path.join(args.exp_dir, 'cifar10c_results.p')
    pickle.dump(results, open(out_path, 'wb'))
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
