#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=14-12:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Usage: sbatch scripts/slurm_eval.sh <eval_script.py> <exp_dir> [extra_args...]
# Example: sbatch scripts/slurm_eval.sh scripts/eval_autoattack.py exp/cnn_cifar_2arg_...

SCRIPT=${1:?"Usage: sbatch scripts/slurm_eval.sh <script.py> <exp_dir> [args...]"}
EXP_DIR=${2:?"Usage: sbatch scripts/slurm_eval.sh <script.py> <exp_dir> [args...]"}
shift 2

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4

eval "$(conda shell.bash hook)"
conda activate xor

cd /home/yizhouc3/xor

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Script: $SCRIPT, Exp: $EXP_DIR"

python "$SCRIPT" --exp_dir "$EXP_DIR" "$@"
