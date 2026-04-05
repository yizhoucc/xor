#!/bin/bash
#SBATCH --job-name=xor-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/yizhouc3/xor/logs/setup_%j.out
#SBATCH --error=/home/yizhouc3/xor/logs/setup_%j.err

# Setup conda environment for XOR Neuron project on Mind cluster

source /usr/share/Modules/init/bash
module load anaconda3-2023.03 cuda-12.4

eval "$(conda shell.bash hook)"

cd /home/yizhouc3/xor

# Create conda env
conda create -n xor python=3.10 -y
conda activate xor

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install numpy scipy pyyaml easydict tqdm tensorboard matplotlib pillow gymnasium datasets

echo "Setup complete!"
conda list
