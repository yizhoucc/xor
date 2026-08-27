#!/bin/bash
#SBATCH --job-name=xor_host
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --exclude=mind-1-19-[1-2],mind-1-24
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Usage: sbatch scripts/slurm_causal_host.sh <host> <seed> <host_ckpt> <save_dir>
HOST=${1:?}; SEED=${2:?}; HOST_CKPT=${3:?}; SAVE_DIR=${4:?}
source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4
eval "$(conda shell.bash hook)"; conda activate xor
export HF_HOME="/tmp/xor_hf_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
cd "${XOR_CODE_DIR:-/home/yizhouc3/xor}"
echo "Node: $(hostname) host=$HOST seed=$SEED host_ckpt=$HOST_CKPT"
python scripts/warmstart_causal.py --host "$HOST" --seed "$SEED" \
    --save_dir "$SAVE_DIR" --host_checkpoint "$HOST_CKPT" --host_only
