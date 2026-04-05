#!/bin/bash
# Submit all paper experiments x 4 seeds as independent Slurm jobs
# Usage: bash scripts/slurm_submit_all.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SEEDS="42 43 44 45"

CONFIGS=(
    # Baselines (fastest)
    config/experiments/mlp_mnist_relu.yaml
    config/experiments/mlp_mnist_relu_ln.yaml
    config/experiments/mlp_cifar_relu.yaml
    config/experiments/mlp_cifar_relu_ln.yaml
    config/experiments/cnn_mnist_relu.yaml
    config/experiments/cnn_mnist_relu_ln.yaml
    config/experiments/cnn_cifar_relu.yaml
    config/experiments/cnn_cifar_relu_ln.yaml
    # 1-arg
    config/experiments/mlp_mnist_1arg.yaml
    config/experiments/mlp_cifar_1arg.yaml
    config/experiments/cnn_mnist_1arg.yaml
    config/experiments/cnn_cifar_1arg.yaml
    # 2-arg (slowest)
    config/experiments/mlp_mnist_2arg.yaml
    config/experiments/mlp_cifar_2arg.yaml
    config/experiments/cnn_mnist_2arg.yaml
    config/experiments/cnn_cifar_2arg.yaml
)

count=0
for cfg in "${CONFIGS[@]}"; do
    # Extract experiment name for job naming
    exp_name=$(basename "$cfg" .yaml)
    for seed in $SEEDS; do
        job_name="xor_${exp_name}_s${seed}"
        if $DRY_RUN; then
            echo "[dry-run] sbatch --job-name=$job_name scripts/slurm_run.sh $cfg $seed"
        else
            sbatch --job-name="$job_name" scripts/slurm_run.sh "$cfg" "$seed"
        fi
        count=$((count + 1))
    done
done

echo "Submitted $count jobs"
