#!/bin/bash
# Run classification experiments with multiple seeds (paper uses 4 seeds).
# Usage:
#   bash scripts/run_multiseed.sh config/experiments/mlp_mnist_2arg.yaml
#   bash scripts/run_multiseed.sh config/experiments/mlp_mnist_2arg.yaml --seeds "42 43 44 45"
#   bash scripts/run_multiseed.sh all   # run all 12 paper experiments x 4 seeds

set -e

SEEDS="${SEEDS:-42 43 44 45}"
CONFIG="$1"

# Parse optional --seeds argument
if [[ "$2" == "--seeds" ]]; then
    SEEDS="$3"
fi

run_config() {
    local cfg="$1"
    echo "=========================================="
    echo "Config: $cfg"
    echo "Seeds: $SEEDS"
    echo "=========================================="
    for seed in $SEEDS; do
        echo "--- Seed $seed ---"
        python run.py -c "$cfg" --seed "$seed"
    done
}

if [[ "$CONFIG" == "all" ]]; then
    # Ordered fast→slow: baselines (no pretrain) first, then 1-arg, then 2-arg (3 phases)
    # Within each group: MLP before CNN, MNIST before CIFAR
    CONFIGS=(
        # --- Baselines (fastest: phase1 only) ---
        config/experiments/mlp_mnist_relu.yaml
        config/experiments/mlp_mnist_relu_ln.yaml
        config/experiments/mlp_cifar_relu.yaml
        config/experiments/mlp_cifar_relu_ln.yaml
        config/experiments/cnn_mnist_relu.yaml
        config/experiments/cnn_mnist_relu_ln.yaml
        config/experiments/cnn_cifar_relu.yaml
        config/experiments/cnn_cifar_relu_ln.yaml
        # --- 1-arg (pretrain + phase1 + phase2) ---
        config/experiments/mlp_mnist_1arg.yaml
        config/experiments/mlp_cifar_1arg.yaml
        config/experiments/cnn_mnist_1arg.yaml
        config/experiments/cnn_cifar_1arg.yaml
        # --- 2-arg (pretrain + phase1 + phase2, slowest) ---
        config/experiments/mlp_mnist_2arg.yaml
        config/experiments/mlp_cifar_2arg.yaml
        config/experiments/cnn_mnist_2arg.yaml
        config/experiments/cnn_cifar_2arg.yaml
    )
    for cfg in "${CONFIGS[@]}"; do
        run_config "$cfg"
    done
else
    run_config "$CONFIG"
fi
