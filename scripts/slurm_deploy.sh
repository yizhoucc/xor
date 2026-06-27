#!/bin/bash
#SBATCH --job-name=xor_deploy
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# P1 deploy: discover -> distill -> deploy loop, all FFN ops on ONE GPU
# so throughput (tok/s) is measured fairly on the same device.
# Usage: sbatch scripts/slurm_deploy.sh <python args...>

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4
eval "$(conda shell.bash hook)"
conda activate xor
cd /home/yizhouc3/xor

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "Args: $@"

python scripts/deploy_distilled.py "$@"
