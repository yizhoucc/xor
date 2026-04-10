"""Analyze CNN representation quality: does InnerNet's channel halving lose information?

Runs on cluster (needs .pth model weights).
Compares feature map statistics between InnerNet CNN and ReLU CNN.

Usage: python scripts/analyze_representation.py --exp2arg exp/cnn_cifar_2arg_... --exprelu exp/cnn_cifar_relu_...
"""
import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_helper import load_model
from easydict import EasyDict
import yaml
import model as model_module


def load_model_from_exp(exp_dir):
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))
    config.save_dir = exp_dir

    model_cls = getattr(model_module, config.model.name)
    m = model_cls(config)

    has_inner = config.model.name not in ('BaselineMLP', 'BaselineCNN', 'BaselineRNN')
    if has_inner:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase2.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')
    else:
        model_path = os.path.join(exp_dir, 'model_snapshot_best_phase1.pth')

    load_model(m, model_path)
    return m, config


def get_feature_maps(model, dataloader, device, n_batches=10):
    """Extract intermediate feature maps from CNN."""
    model.eval()
    all_features = []

    hooks = []
    features = {}

    def make_hook(name):
        def hook(module, input, output):
            features[name] = output.detach().cpu()
        return hook

    # Hook into conv layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            if i >= n_batches:
                break
            features.clear()
            imgs = imgs.to(device)
            labels = labels.to(device)
            model(imgs, labels)
            all_features.append({k: v.clone() for k, v in features.items()})

    for h in hooks:
        h.remove()

    return all_features


def analyze_features(features_list, label):
    """Compute feature map statistics."""
    print(f"\n=== {label} ===")
    # Use first batch
    features = features_list[0]
    for name, feat in features.items():
        B, C, H, W = feat.shape
        # Channel-wise statistics
        channel_means = feat.mean(dim=(0, 2, 3))  # per-channel mean
        channel_vars = feat.var(dim=(0, 2, 3))     # per-channel variance

        # Dead channels (always zero or near-zero)
        dead = (channel_vars < 1e-6).sum().item()
        dead_pct = dead / C * 100

        # Effective rank (how many channels carry information)
        # Using singular values of the channel correlation matrix
        feat_flat = feat.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
        try:
            S = torch.linalg.svdvals(feat_flat.float())
            S_norm = S / S.sum()
            entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
            eff_rank = np.exp(entropy)
        except:
            eff_rank = C

        print(f"  {name}: channels={C}, dead={dead} ({dead_pct:.1f}%), "
              f"eff_rank={eff_rank:.1f}/{C}, mean_var={channel_vars.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp2arg', type=str, required=True)
    parser.add_argument('--exprelu', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    model_2arg, config_2arg = load_model_from_exp(args.exp2arg)
    model_relu, config_relu = load_model_from_exp(args.exprelu)
    model_2arg = model_2arg.to(device)
    model_relu = model_relu.to(device)

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False,
                                     transform=transform, download=True)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Compare
    feat_2arg = get_feature_maps(model_2arg, loader, device)
    feat_relu = get_feature_maps(model_relu, loader, device)

    p_2arg = sum(p.numel() for p in model_2arg.parameters())
    p_relu = sum(p.numel() for p in model_relu.parameters())
    print(f"InnerNet params: {p_2arg:,} | ReLU params: {p_relu:,} | ratio: {p_2arg/p_relu:.2f}x")

    analyze_features(feat_2arg, "InnerNet (2-arg)")
    analyze_features(feat_relu, "ReLU baseline")

    # Save
    results = {'exp2arg': args.exp2arg, 'exprelu': args.exprelu}
    pickle.dump(results, open(os.path.join(args.exp2arg, 'representation_analysis.p'), 'wb'))
    print("\nDone.")


if __name__ == '__main__':
    main()
