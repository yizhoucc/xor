#!/bin/bash
#SBATCH --job-name=xor_causal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Usage: sbatch scripts/slurm_causal.sh <host> <freeze> <seed> <save_dir> <inits>
HOST=${1:?}; FREEZE=${2:?}; SEED=${3:?}; SAVE_DIR=${4:?}; INITS=${5:?}
source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4
eval "$(conda shell.bash hook)"; conda activate xor
cd /home/yizhouc3/xor
echo "Node: $(hostname) host=$HOST freeze=$FREEZE seed=$SEED dir=$SAVE_DIR inits=$INITS"
python scripts/warmstart_causal.py --host "$HOST" --freeze "$FREEZE" --seed "$SEED" \
    --save_dir "$SAVE_DIR" --inits "$INITS"
