"""Evaluate adversarial robustness using AutoAttack on trained CNN CIFAR-10 models.

Usage:
    python scripts/eval_autoattack.py --exp_dir exp/cnn_cifar_2arg_20260305_... --eps 8/255

Loads the best model from the experiment directory and runs AutoAttack (L∞).
"""
import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_helper import load_model
from easydict import EasyDict
import yaml


def load_config_and_model(exp_dir):
    """Load config and instantiate model from experiment directory."""
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)
    config.save_dir = exp_dir

    # Import model classes
    from model import *

    model_cls = eval(config.model.name)
    model = model_cls(config)

    # Determine which checkpoint to load
    has_inner = config.model.name not in ('BaselineMLP', 'BaselineCNN', 'BaselineRNN')
    if has_inner:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase2.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')
    else:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')

    load_model(model, model_path)
    return config, model


class ModelWrapper(nn.Module):
    """Wrapper that takes (x) and returns logits only (for AutoAttack)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Create dummy labels
        dummy_labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out, _, _ = self.model(x, dummy_labels)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--eps', type=str, default='8/255',
                        help='Perturbation budget (e.g., 8/255)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_examples', type=int, default=1000,
                        help='Number of test examples to evaluate')
    args = parser.parse_args()

    eps = eval(args.eps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    config, model = load_config_and_model(args.exp_dir)
    model = model.to(device)
    model.eval()

    wrapped = ModelWrapper(model).to(device)

    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])
    test_dataset = datasets.CIFAR10(root=config.dataset.data_path,
                                     train=False, transform=transform, download=True)

    # Subsample
    indices = list(range(min(args.n_examples, len(test_dataset))))
    subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size)

    # Collect all data
    x_test = []
    y_test = []
    for x, y in loader:
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test).to(device)
    y_test = torch.cat(y_test).to(device)

    # Clean accuracy
    with torch.no_grad():
        clean_out = wrapped(x_test)
        clean_acc = (clean_out.argmax(1) == y_test).float().mean().item()
    print(f"Clean accuracy: {clean_acc:.4f}")

    # AutoAttack
    from autoattack import AutoAttack
    adversary = AutoAttack(wrapped, norm='Linf', eps=eps, version='standard',
                           verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

    # Robust accuracy
    with torch.no_grad():
        rob_out = wrapped(x_adv)
        rob_acc = (rob_out.argmax(1) == y_test).float().mean().item()
    print(f"Robust accuracy (eps={eps:.4f}): {rob_acc:.4f}")

    # Save results
    results = {
        'clean_accuracy': clean_acc,
        'robust_accuracy': rob_acc,
        'eps': eps,
        'n_examples': len(x_test),
        'exp_dir': args.exp_dir,
    }
    out_path = os.path.join(args.exp_dir, 'autoattack_results.p')
    pickle.dump(results, open(out_path, 'wb'))
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
