#!/bin/bash
#SBATCH --job-name=xor_bilin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Usage: sbatch scripts/slurm_bilinear_probe.sh <save_dir> <seed_start>
SAVE_DIR=${1:?"need save_dir"}
SEED_START=${2:-42}

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4
eval "$(conda shell.bash hook)"
conda activate xor
cd /home/yizhouc3/xor

echo "Node: $(hostname), save_dir=$SAVE_DIR, seed_start=$SEED_START"
python scripts/warmstart_bilinear_probe.py --save_dir "$SAVE_DIR" --num_seeds 1 --seed_start "$SEED_START"
