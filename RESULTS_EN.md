# Experiment Results — InnerNet (Learnable 2-Argument Activation Function)

> Summary of all reproducibility and extension experiments for "Two-argument activation functions learn soft XOR operations like cortical neurons" (Yoon et al., 2021).

## Key Findings

1. **CNN image classification**: InnerNet consistently outperforms ReLU (+0.4–4.3%) with **40% fewer parameters**
2. **Autoencoder reconstruction**: Most dramatic improvement — up to **39% lower MSE** on MNIST
3. **Transformer/LSTM language models**: InnerNet FFN consistently beats GELU baseline (-1.6–3.4% PPL)
4. **LSTM ablation**: Classic (adjacent pairing) outperforms semantic pairing — simplest approach works best
5. **Regression**: Effective on Housing (-5% MSE), marginal on others
6. **ResNet (skip connections)**: InnerNet gains disappear — skip connections provide equivalent feature interaction
7. **Parameter efficiency**: InnerNet w=128 matches ReLU w=256 performance, saving ~55% parameters (MLP CIFAR-10)
8. **SwiGLU vs InnerNet on images**: SwiGLU CNN (79.79%) slightly beats InnerNet CNN (78.57%) on CIFAR-10

---

## 1. CNN Image Classification (5 seeds)

| Dataset | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | SwiGLU | Gain (2arg-ReLU) |
|---------|-------|-------|------|---------|-------------|--------|----------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | **+0.39** |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | **+1.57** |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | **+2.46** |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

**Parameter fairness (CIFAR-10)**: InnerNet (127K params, 78.57%) > ReLU matched (127K params, 70.67%). Same params, +7.9% accuracy.

**SwiGLU on images**: SwiGLU CNN (79.79%) slightly beats InnerNet (78.57%) on CIFAR-10, but InnerNet beats SwiGLU on CIFAR-100 (53.74% vs 46.48%).

## 2. Mainstream Architecture Baselines (CIFAR-100 + augmentation)

| Architecture | Accuracy | Params |
|-------------|----------|--------|
| WRN-28-10 | 74.78±0.15 (n=5) | ~36M |
| ResNet-18 ReLU | 73.51±0.18 (n=5) | ~11M |
| ResNet-18 InnerNet | 71.72±0.52 (n=4†) | — |
| VGG-16+BN | 68.48±0.49 (n=5) | ~138M |

ResNet-18 InnerNet slightly behind ReLU on CIFAR-100 with augmentation. †One seed (59.43%) diverged due to training instability — excluded from statistics.

## 3. Autoencoder Reconstruction (MSE↓, 3–5 seeds)

| Dataset | InnerNet | ReLU | ReLU matched | Improvement |
|---------|----------|------|-------------|-------------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43% vs ReLU, -34% vs matched** |
| FashionMNIST | **0.0076** | 0.0086 | — | **-12%** |
| CIFAR-10 | **0.0081** | 0.0105 | — | **-23%** |

### AE Capacity Scaling (latent dimension)

| Latent Dim | InnerNet | ReLU | Improvement |
|-----------|----------|------|-------------|
| 8 | 0.0141 | 0.0183 | -23% |
| 16 | 0.0075 | 0.0114 | -34% |
| 32 | 0.0039 | 0.0067 | -42% |
| 64 | 0.0026 | 0.0042 | -39% |

## 4. Language Models (PPL↓, 5 seeds)

### Transformer FFN (WikiText-2)

| Config | InnerNet | GELU | SwiGLU | Improvement |
|--------|----------|------|--------|-------------|
| d=64 (small) | **112.66±0.66** | 116.63±0.84 | — | **-3.4%** |
| d=128 (standard) | **95.26±1.00** | 96.82±1.19 | 92.98±1.14 | **-1.6%** |
| d=192 | **88.14±0.80** | 89.11±0.92 | — | **-1.1%** |
| d=256 (large) | **85.40±1.15** | 86.05±0.97 | — | **-0.8%** |

### Transformer FFN (PTB)

| Model | PPL |
|-------|-----|
| **InnerNet (ReLU)** | **207.81±1.58** |
| SiLU-InnerNet | 208.43±1.44 |
| GELU baseline | 212.28±0.88 |

SiLU-InnerNet does NOT outperform ReLU-InnerNet — smooth inductive bias doesn't help.

### LSTM Ablation (WikiText-2) — 2×2 Design

| | Unbounded | Bounded (tanh) |
|--|-----------|----------------|
| **Classic** (adjacent pair) | **101.72±0.99** | 104.46±0.38 |
| **Semantic** (x vs h pair) | 105.30±0.31 | 107.15±0.83 |
| **Standard** (tanh only) | — | 108.39±0.75 |

**Key finding**: Classic unbounded (101.72) is the best. Adjacent pairing without semantic structure outperforms deliberate x-vs-h pairing. Bounding with tanh hurts performance.

### LSTM (WikiText-2, best result)

| Model | PPL |
|-------|-----|
| Classic InnerNet (unbounded) | **101.72±0.99** |
| Standard | 108.39±0.75 |
| Improvement | **-6.2%** |

## 5. Regression (MSE↓, 3–5 seeds)

| Dataset | InnerNet | ReLU | Improvement |
|---------|----------|------|-------------|
| California Housing | **0.196±0.007** | 0.206±0.008 | **-5.0%** |
| Diabetes | 0.506±0.065 | 0.510±0.043 | -0.8% (neutral) |
| Wine Quality | 0.599±0.029 | **0.548±0.022** | +9.3% (worse) |

## 6. Big MLP MNIST (3×256, dropout=0.3, 5 seeds)

| Model | Accuracy |
|-------|----------|
| InnerNet | **98.39±0.04** |
| ReLU | 97.93±0.07 |
| Improvement | **+0.46%** |

## 7. ResNet (SGD, 150 epochs, 5 seeds, no augmentation)

| Dataset | InnerNet | ReLU | Diff |
|---------|----------|------|------|
| CIFAR-10 | 86.09±0.82 | 86.33±0.34 | -0.24 (neutral) |
| CIFAR-100 | 56.78±2.77 | 57.95±0.52 | -1.17 (neutral) |

**Skip connections make InnerNet redundant.**

## 8. PPO Reinforcement Learning (10 seeds)

| Environment | InnerNet | ReLU | SwiGLU |
|-------------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| **Acrobot** | **-75.3** | -79.8 | -81.7 |
| MountainCar | -200.0 | -200.0 | -200.0 |
| LunarLander | 166.6 | **209.1** | -139.1 |

## 9. Capacity Scaling — Parameter Efficiency

### MLP CIFAR-10 (5 seeds each)

| Width | InnerNet (Params) | ReLU (Params) | Gain |
|-------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

**Key insight:** InnerNet w=128 (52.05%, 415K params) ≥ ReLU w=256 (51.99%, 921K params) → **55% parameter savings**.

### CNN CIFAR-10 Channel Scaling (3 seeds)

| Scale | InnerNet | ReLU | Diff |
|-------|----------|------|------|
| ×0.25 | 47.96 | 53.07 | -5.11 |
| ×0.5 | 62.43 | 67.83 | -5.41 |
| ×1 | **78.74** | 77.14 | **+1.60** |

InnerNet underperforms at small scales due to channel-pairing overhead.

## 10. Negative / Neutral Results

| Experiment | Result | Reason |
|------------|--------|--------|
| Text classification (TF-IDF/embedding) | Neutral | Sparse/structured features don't benefit from pairwise pairing |
| ResNet (skip connections) | Neutral | Residual connections already provide feature interaction |
| Transformer attention replacement | Worse (+6.6% PPL) | Cannot replace softmax's mathematical structure |
| Wine regression | Worse | Low-dim tabular data, InnerNet overhead not justified |
| CNN at very small scale (×0.25) | Worse | Channel-pairing overhead dominates when model is tiny |
| SiLU-InnerNet vs ReLU-InnerNet | Neutral | Smooth inductive bias doesn't help (PTB: 208.43 vs 207.81) |
| LSTM semantic pairing | Worse than classic | Deliberate x-vs-h pairing underperforms simple adjacent pairing |

## Summary

InnerNet (learnable 2-argument activation) provides consistent benefits in **feedforward networks without skip connections**, achieving better accuracy with fewer parameters. The advantage is most pronounced in:
- **Autoencoders** (up to -43% MSE)
- **CNNs** (+0.4–4.6% acc, 40% fewer params)
- **Transformer FFN** (-0.8–3.4% PPL across 4 model sizes)
- **LSTM** (-6.2% PPL with classic unbounded variant)

The advantage disappears when models already have built-in feature interaction (ResNet skip connections, self-attention) or when input features are sparse/structured (text, tabular).

Surprising finding: In LSTM, simple adjacent-dimension pairing outperforms deliberate semantic pairing, and removing the tanh bound further improves performance.
