# XOR Neuron

Reproducing and extending **"Two-argument activation functions learn soft XOR operations like cortical neurons"** (Yoon, Orhan, Kim, Pitkow, 2021, [arXiv:2110.06871v2](https://arxiv.org/abs/2110.06871v2)).

## Core Idea

Replace scalar activation functions (ReLU) with a small learned **InnerNet** that takes 2 inputs → 1 output. The learned function converges to a **soft XOR / multiplicative gating** pattern, similar to dendritic computation in biological neurons.

## Key Results

InnerNet provides consistent benefits in **feedforward networks without skip connections**:

| Task | InnerNet | Baseline | Gain |
|------|----------|----------|------|
| CNN CIFAR-10 | 78.29% | 73.99% | **+4.30%** |
| CNN SVHN | 95.01% | 92.63% | **+2.38%** |
| Autoencoder MNIST (MSE) | 0.0039 | 0.0068 | **-43%** |
| Transformer FFN PPL | 95.26 | 96.82 | **-1.6%** |
| LSTM PPL | 105.30 | 108.39 | **-2.9%** |

**Parameter efficiency**: CNN InnerNet uses only **60% of parameters** while outperforming ReLU. MLP InnerNet w=128 matches ReLU w=256 (55% parameter savings).

Full results: [`RESULTS_EN.md`](RESULTS_EN.md) | [`RESULTS_CN.md`](RESULTS_CN.md)

## Architecture

```
Input → Linear → LayerNorm → InnerNet(x1, x2) → Dropout → ... → Output
                                ↑
                    Small MLP: 2 inputs → 64 hidden → 1 output
                    Shared across all neurons (like a fixed activation)
```

**Training**: 3 phases — (I) pretrain InnerNet on random smooth functions, (II) joint training, (III) freeze InnerNet + retrain outer network.

## Models

| Model | Description |
|-------|-------------|
| `XorNeuronMLP` / `XorNeuronConv` | Paper's core: shared InnerNet MLP/CNN |
| `BaselineMLP` / `BaselineCNN` | ReLU baselines |
| `BaselineResNet` / `InnerNetResNet` | ResNet-18 with InnerNet |
| `BaselineAE` / `InnerNetAE` | Autoencoder with InnerNet |
| `InnerNetTransformer` / `StandardTransformer` | Transformer FFN with InnerNet |
| `InnerNetLSTMModel` / `StandardLSTMModel` | LSTM with InnerNet |
| `InnerNetDQN` / `BaselineDQN` / `SwiGLUDQN` | DQN for RL |
| `InnerNetPPO` / `BaselinePPO` / `SwiGLUPPO` | PPO for RL |

## Quick Start

```bash
# Setup
conda env create -f condaenv.yml && conda activate xor
# or: pip install -r requirements.txt

# Run a single experiment
python run.py -c config/experiments/cnn_cifar_2arg.yaml

# Validate config before submitting to cluster
python run.py -c config/experiments/cnn_cifar_2arg.yaml --validate

# Run all paper experiments
for cfg in config/experiments/cnn_{mnist,cifar}_{2arg,1arg,relu}.yaml; do
  python run.py -c "$cfg"
done

# Multi-seed (Slurm cluster)
for seed in 1234 42 43 44 45; do
  sbatch scripts/slurm_run.sh config/experiments/cnn_cifar_2arg.yaml "$seed"
done
```

## Project Structure

```
xor/
├── run.py                      # Unified entry point
├── model/
│   ├── xorneuron.py            # InnerNet models (XorNeuronMLP, XorNeuronConv, ...)
│   ├── baseline.py             # ReLU baselines (BaselineMLP, BaselineCNN, BaselineRNN)
│   ├── resnet.py               # ResNet-18 with/without InnerNet
│   ├── autoencoder.py          # Autoencoder with/without InnerNet
│   ├── transformer.py          # Transformer (InnerNet/GELU/SwiGLU FFN)
│   ├── lstm.py                 # LSTM with/without InnerNet
│   ├── dqn.py                  # DQN models
│   └── ppo.py                  # PPO models
├── runner/
│   ├── experiment_runner.py    # Classification/regression/AE runner
│   ├── lm_runner.py            # Language model runner (LSTM/Transformer)
│   ├── rl_runner.py            # DQN runner
│   └── ppo_runner.py           # PPO runner
├── dataset/
│   ├── innernet_data.py        # InnerNet pretraining data
│   └── tabular.py              # Tabular/text/audio/timeseries datasets
├── config/experiments/         # All experiment configs (YAML)
├── scripts/
│   ├── slurm_run.sh            # Slurm job submission
│   ├── slurm_eval.sh           # Slurm eval submission
│   ├── eval_autoattack.py      # AutoAttack robustness eval
│   └── eval_cifar10c.py        # CIFAR-10-C corruption eval
├── exp/                        # Experiment outputs
├── PROJECT_STATUS.md           # Current status + TODO
├── RESULTS_EN.md               # Full results (English)
├── RESULTS_CN.md               # Full results (Chinese)
└── CLAUDE.md                   # Development guidelines
```

## Datasets

Auto-downloaded: MNIST, FashionMNIST, CIFAR-10/100, SVHN, STL-10, WikiText-2, UCI Adult/Wine, California Housing, Diabetes, SST-2, AG News.

Manual: `bash scripts/download_ptb.sh` (PTB for RNN experiments).

## Citation

```bibtex
@article{yoon2021two,
  title={Two-argument activation functions learn soft XOR operations like cortical neurons},
  author={Yoon, Kijung and Orhan, Emin and Kim, Juhyun and Pitkow, Xaq},
  journal={arXiv preprint arXiv:2110.06871},
  year={2021}
}
```
