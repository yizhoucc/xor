#!/bin/bash
#SBATCH --job-name=xor
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=14-12:00:00
#SBATCH --exclude=mind-1-19-[1-2],mind-1-24
#SBATCH --output=/home/yizhouc3/xor/logs/slurm_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/slurm_%j.err

# Same as slurm_run.sh but requests --mem=200GB to force allocation
# on nodes with large GPUs (A5000 24GB, TITAN RTX 24GB, L40S 48GB).
# Use for InnerNet CNN models that OOM on 10-12GB GPUs.

CONFIG=${1:?"Usage: sbatch scripts/slurm_run_largegpu.sh <config.yaml> [seed]"}
SEED=${2:-1234}

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4

eval "$(conda shell.bash hook)"
conda activate xor

CODE_DIR="${XOR_CODE_DIR:-/home/yizhouc3/xor}"
RUN_DIR="${XOR_RUN_DIR:-$CODE_DIR}"
if [[ "$CONFIG" = /* ]]; then
    CONFIG_PATH="$CONFIG"
else
    CONFIG_PATH="$CODE_DIR/$CONFIG"
fi
export HF_HOME="/tmp/xor_hf_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
cd "$RUN_DIR"

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Code: $CODE_DIR, Run dir: $RUN_DIR"
echo "Config: $CONFIG_PATH, Seed: $SEED"

python "$CODE_DIR/run.py" -c "$CONFIG_PATH" --seed "$SEED"
