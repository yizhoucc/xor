"""Extract training curves (epoch vs val_acc/val_loss) from experiment logs.

Usage: python scripts/extract_training_curves.py
Outputs CSV files for plotting convergence speed comparison.
"""
import os
import re
import glob
import csv


def extract_curves(exp_dir):
    """Extract epoch, val_loss, val_acc from log files."""
    logs = sorted(glob.glob(os.path.join(exp_dir, "log_exp_*.txt")))
    if not logs:
        return []

    curves = []
    for log in logs:
        with open(log) as f:
            for line in f:
                # Match: Epoch 5 | Val Loss = 0.123456 | Val Acc = 0.9800
                m = re.search(r'Epoch\s+(\d+)\s+\|\s+Val Loss\s+=\s+([\d.]+)\s+\|\s+Val Acc\s+=\s+([\d.]+)', line)
                if m:
                    epoch, loss, acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    curves.append((epoch, loss, acc))
                # Match: Epoch 5 | Val MSE = 0.123456
                m2 = re.search(r'Epoch\s+(\d+)\s+\|\s+Val MSE\s+=\s+([\d.]+)', line)
                if m2:
                    epoch, mse = int(m2.group(1)), float(m2.group(2))
                    curves.append((epoch, mse, 0))
    return curves


def main():
    output_dir = "results/training_curves"
    os.makedirs(output_dir, exist_ok=True)

    # Key experiments for convergence comparison
    experiments = {
        "CNN CIFAR-10": [
            ("2arg", "exp/cnn_cifar_2arg_20260305_*"),
            ("relu", "exp/cnn_cifar_relu_20260304_*"),
        ],
        "Big MLP MNIST": [
            ("2arg", "exp/mlp_mnist_big_2arg_*"),
            ("relu", "exp/mlp_mnist_big_relu_*"),
        ],
        "AE MNIST": [
            ("2arg", "exp/ae_mnist_2arg_*"),
            ("relu", "exp/ae_mnist_relu_*"),
        ],
    }

    for exp_name, variants in experiments.items():
        print(f"\n=== {exp_name} ===")
        for variant_name, pattern in variants:
            dirs = sorted(glob.glob(pattern))
            if not dirs:
                print(f"  {variant_name}: no dirs found for {pattern}")
                continue
            # Use first dir
            curves = extract_curves(dirs[0])
            if curves:
                fname = f"{output_dir}/{exp_name.replace(' ', '_')}_{variant_name}.csv"
                with open(fname, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['epoch', 'val_loss', 'val_acc'])
                    w.writerows(curves)
                print(f"  {variant_name}: {len(curves)} epochs → {fname}")
                # Show first/last few
                if curves:
                    print(f"    epoch 1: loss={curves[0][1]:.4f} acc={curves[0][2]:.4f}")
                    mid = len(curves) // 4
                    print(f"    epoch {curves[mid][0]}: loss={curves[mid][1]:.4f} acc={curves[mid][2]:.4f}")
                    print(f"    epoch {curves[-1][0]}: loss={curves[-1][1]:.4f} acc={curves[-1][2]:.4f}")
            else:
                print(f"  {variant_name}: no val data in logs")


if __name__ == "__main__":
    main()
