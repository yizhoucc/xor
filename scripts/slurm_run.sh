#!/bin/bash
#SBATCH --job-name=xor
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Usage: sbatch scripts/slurm_run.sh <config_yaml> [seed]
# Example: sbatch scripts/slurm_run.sh config/experiments/mlp_mnist_2arg.yaml 42

CONFIG=${1:?"Usage: sbatch scripts/slurm_run.sh <config.yaml> [seed]"}
SEED=${2:-1234}

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4

eval "$(conda shell.bash hook)"
conda activate xor

cd /home/yizhouc3/xor

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Config: $CONFIG, Seed: $SEED"

python run.py -c "$CONFIG" --seed "$SEED"
