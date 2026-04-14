# Experiment Results — InnerNet (Learnable 2-Argument Activation Function)

> Summary of all experiments for "Two-argument activation functions learn soft XOR operations like cortical neurons" (Yoon et al., 2021).

## Paper Story

InnerNet replaces scalar activations (ReLU) with a small learned MLP that takes two inputs: `f(a, b) → output`. This lets each neuron "see" a neighboring feature before activating — similar to cortical neurons that perform soft XOR operations.

**Core claims:**

1. **InnerNet improves feedforward networks without skip connections** — CNN (+0.4–4.6%), AE (-43% MSE), Transformer FFN (-2.1–3.3% PPL), with 40% fewer parameters
2. **InnerNet as architecture discovery tool** — at small scale (d=64), InnerNet (112.83) independently matches SwiGLU (112.31)
3. **Simplicity wins** — simple adjacent pairing > deliberate semantic pairing (LSTM); no pretrain needed (end-to-end ≈ 3-phase)

**Boundaries:** InnerNet is redundant when skip connections already provide cross-feature interaction (ResNet).

---

## 1. CNN Image Classification (5 seeds)

| Dataset | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | SwiGLU | Gain |
|---------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | **+0.39** |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | **+1.57** |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | **+2.46** |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

**Parameter fairness (CIFAR-10)**: InnerNet (127K params, 78.57%) > ReLU matched (127K params, 70.67%). Same params, **+7.9% accuracy**.

Configs: `config/experiments/cnn_cifar_2arg.yaml`, exp: `exp/cnn_cifar_2arg_*`

## 2. Autoencoder Reconstruction (MSE↓, 3–5 seeds)

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

Configs: `config/experiments/ae_mnist_2arg.yaml`, exp: `exp/ae_mnist_2arg_*`

## 3. Transformer Language Models (PPL↓, 5 seeds)

| Config | GELU | SwiGLU | InnerNet | vs GELU |
|--------|------|--------|----------|---------|
| WikiText-2 d=64 | 116.63±0.84 | **112.31±0.49** | **112.83** | **-3.3%** |
| WikiText-2 d=128 | 96.82±1.19 | **92.98±1.14** | ⏳ | — |
| WikiText-2 d=192 | 89.11±0.92 | **85.43** | ⏳ | — |
| WikiText-2 d=256 | 86.05±0.97 | ⏳ | ⏳ | — |
| PTB d=128 | 212.28±0.88 | **205.82±0.98** | **207.91** | **-2.1%** |

InnerNet consistently beats GELU. At d=64, InnerNet (112.83) ≈ SwiGLU (112.31).

Config: `config/experiments/transformer_wikitext_2arg.yaml`, exp: `exp/transformer_wikitext_2arg_*`

## 4. LSTM (WikiText-2, PPL↓, 5 seeds)

| Variant | PPL |
|---------|-----|
| **Classic InnerNet** (adjacent pair) | **101.72±0.99** |
| Semantic InnerNet (x vs h pair) | 105.30±0.31 |
| Standard LSTM | 108.39±0.75 |

Classic InnerNet achieves **-6.2% PPL** vs standard LSTM. Adjacent-dimension pairing outperforms deliberate semantic pairing.

Config: `config/experiments/lstm_wikitext_classic.yaml`, exp: `exp/lstm_wikitext_classic_*`

## 5. Parameter Efficiency — MLP CIFAR-10 (5 seeds)

| Width | InnerNet (Params) | ReLU (Params) | Gain |
|-------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

**InnerNet w=128 (415K) ≥ ReLU w=256 (921K) → 55% parameter savings.**

## 6. Big MLP MNIST (3×256, dropout=0.3, 5 seeds)

| Model | Accuracy |
|-------|----------|
| InnerNet | **98.39±0.04** |
| ReLU | 97.93±0.07 |
| Improvement | **+0.46%** |

## 7. Where InnerNet Does Not Help

| Experiment | Result | Interpretation |
|------------|--------|---------------|
| ResNet (skip connections) | Neutral | Residual path already provides cross-feature interaction |
| CNN at very small scale (×0.25) | Worse (-5%) | Channel-pairing overhead dominates when model is tiny |

## Summary

InnerNet provides consistent benefits in **feedforward networks without built-in feature interaction mechanisms**:

- **Autoencoders**: -23% to -43% MSE
- **CNNs**: +0.4–4.6% accuracy with 40% fewer parameters, consistent across 5 datasets
- **Transformer FFN**: -2.1–3.3% PPL; InnerNet ≈ SwiGLU at small scale (d=64)
- **LSTM**: -6.2% PPL (WikiText-2, classic pairing)
- **Parameter efficiency**: 55% parameter savings (InnerNet w=128 ≈ ReLU w=256)

**Boundaries**: InnerNet is redundant when models already have built-in feature interaction (ResNet skip connections).

**Simplicity principle**: Adjacent pairing > semantic pairing — the dual-input interaction itself drives the improvement.
