# Experiment Results — InnerNet (Learnable 2-Argument Activation Function)

> Summary of all reproducibility and extension experiments for "Two-argument activation functions learn soft XOR operations like cortical neurons" (Yoon et al., 2021).

## Key Findings

1. **CNN image classification**: InnerNet consistently outperforms ReLU (+0.4–4.3%) with **40% fewer parameters**
2. **Autoencoder reconstruction**: Most dramatic improvement — up to **43% lower MSE** on MNIST
3. **Transformer/LSTM language models**: InnerNet FFN consistently beats GELU baseline (-1.6–3.4% PPL)
4. **Regression**: Effective on Housing (-5% MSE), marginal on others
5. **ResNet (skip connections)**: InnerNet gains disappear — skip connections provide equivalent feature interaction
6. **Parameter efficiency**: InnerNet w=128 matches ReLU w=256 performance, saving ~55% parameters (MLP CIFAR-10)

---

## 1. CNN Image Classification (5 seeds)

| Dataset | InnerNet | ReLU | Gain | InnerNet Params | ReLU Params |
|---------|----------|------|------|----------------|-------------|
| MNIST | 99.41±0.04 | 99.02±0.03 | **+0.39** | 127K | 211K (0.60×) |
| CIFAR-10 | 78.29±0.54 | 73.99±0.49 | **+4.30** | 128K | 212K (0.60×) |
| FashionMNIST | 90.87 | 89.46 | **+1.41** | 127K | 211K (0.60×) |
| SVHN | 95.01 | 92.63 | **+2.38** | 128K | 212K (0.60×) |
| CIFAR-100 (big CNN) | 54.12±0.29 | 50.00±0.83 | **+4.12** | — | — |

InnerNet achieves better accuracy with only **60% of the parameters**.

## 2. Autoencoder Reconstruction (MSE↓, 3–5 seeds)

| Dataset | InnerNet | ReLU | Improvement |
|---------|----------|------|-------------|
| MNIST | **0.0039** | 0.0068 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | **-12%** |
| CIFAR-10 | **0.0081** | 0.0105 | **-23%** |

Consistent improvement across all datasets. MNIST shows nearly 2× better reconstruction quality.

### AE Capacity Scaling (latent dimension)

| Latent Dim | InnerNet | ReLU | Improvement |
|-----------|----------|------|-------------|
| 8 | 0.0141 | 0.0183 | -23% |
| 16 | 0.0075 | 0.0114 | -34% |
| 32 | 0.0039 | 0.0067 | -42% |
| 64 | 0.0026 | 0.0042 | -39% |

InnerNet advantage is consistent across all latent dimensions.

## 3. Language Models (PPL↓, 5 seeds)

### Transformer FFN (WikiText-2)

| Config | InnerNet | GELU | Improvement |
|--------|----------|------|-------------|
| d=64 (small) | **112.66±0.66** | 116.63±0.84 | **-3.4%** |
| d=128 (standard) | **95.26±1.00** | 96.82±1.19 | **-1.6%** |
| d=192 | ⏳ | 89.11±0.92 | pending |
| d=256 (large) | ⏳ | 86.05±0.97 | pending |
| SwiGLU d=128 | — | 92.98±1.14 | hand-crafted best |

### LSTM (WikiText-2)

| Model | PPL |
|-------|-----|
| InnerNet | **105.30±0.31** |
| Standard | 108.39±0.75 |
| Improvement | **-2.9%** |

## 4. Regression (MSE↓, 3–5 seeds)

| Dataset | InnerNet | ReLU | Improvement |
|---------|----------|------|-------------|
| California Housing | **0.196±0.007** | 0.206±0.008 | **-5.0%** |
| Diabetes | 0.506±0.065 | 0.510±0.043 | -0.8% (neutral) |
| Wine Quality | 0.599±0.029 | **0.548±0.022** | +9.3% (worse) |

Effective on Housing, marginal on Diabetes, negative on Wine. InnerNet's pairwise pairing may not suit all low-dimensional tabular data.

## 5. Big MLP MNIST (3×256, dropout=0.3, 5 seeds)

| Model | Accuracy |
|-------|----------|
| InnerNet | **98.39±0.04** |
| ReLU | 97.93±0.07 |
| Improvement | **+0.46%** |

With a properly-sized MLP, InnerNet still shows consistent improvement.

## 6. ResNet (SGD, 150 epochs, 5 seeds)

| Dataset | InnerNet | ReLU | Diff |
|---------|----------|------|------|
| CIFAR-10 | 86.09±0.82 | 86.33±0.34 | -0.24 (neutral) |
| CIFAR-100 | 56.78±2.77 | 57.95±0.52 | -1.17 (neutral) |

**Skip connections make InnerNet redundant.** ResNet's residual paths already provide sufficient feature interaction, eliminating InnerNet's advantage.

## 7. PPO Reinforcement Learning (10 seeds)

| Environment | InnerNet | ReLU | SwiGLU |
|-------------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| **Acrobot** | **-75.3** | -79.8 | -81.7 |
| MountainCar | -200.0 | -200.0 | -200.0 |
| LunarLander | 166.6 | **209.1** | -139.1 |

InnerNet best on Acrobot (+5.6%), ReLU best on LunarLander. MountainCar unsolved by all.

## 8. Capacity Scaling — Parameter Efficiency

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
| ×0.25 | 49.61 | 53.07 | -3.47 |
| ×0.5 | 62.43 | 67.83 | -5.41 |
| ×1 | **78.74** | 77.14 | **+1.60** |
| ×2 | pending | 80.31 | — |

InnerNet **underperforms at very small scales** but overtakes at standard scale and above. The channel-pairing overhead is relatively larger for small models.

### Housing Regression Width Scaling (3 seeds)

| Width | InnerNet (MSE) | ReLU (MSE) |
|-------|---------------|------------|
| 32 | **0.237** | 0.251 |
| 64 | **0.214** | 0.218 |
| 120 | **0.197** | 0.207 |
| 256 | 0.200 | **0.196** |
| 512 | 0.205 | **0.196** |

InnerNet advantage diminishes at larger widths. Optimal at small-to-medium capacity.

## 9. Negative / Neutral Results

| Experiment | Result | Reason |
|------------|--------|--------|
| Text classification (TF-IDF/embedding) | Neutral | Sparse/structured features don't benefit from pairwise pairing |
| ResNet (skip connections) | Neutral | Residual connections already provide feature interaction |
| Transformer attention replacement | Worse (+6.6% PPL) | Cannot replace softmax's mathematical structure |
| Wine regression | Worse | Low-dim tabular data, InnerNet overhead not justified |
| CNN at very small scale (×0.25) | Worse | Channel-pairing overhead dominates when model is tiny |

## Summary

InnerNet (learnable 2-argument activation) provides consistent benefits in **feedforward networks without skip connections**, achieving better accuracy with fewer parameters. The advantage is most pronounced in:
- **Autoencoders** (up to -43% MSE)
- **CNNs** (+0.4–4.3% acc, 40% fewer params)
- **Transformer FFN** (-1.6–3.4% PPL)

The advantage disappears when models already have built-in feature interaction (ResNet skip connections, self-attention) or when input features are sparse/structured (text, tabular).
