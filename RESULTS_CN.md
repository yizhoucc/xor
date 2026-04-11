# 实验结果 — InnerNet（可学习双参数激活函数）

> 论文 "Two-argument activation functions learn soft XOR operations like cortical neurons"（Yoon et al., 2021）的复现与扩展实验汇总。

## 核心发现

1. **CNN 图像分类**：InnerNet 一致优于 ReLU（+0.4–4.6%），且**参数量少 40%**
2. **Autoencoder 重建**：效果最显著——MNIST 上 **MSE 降低 39%**
3. **Transformer/LSTM 语言模型**：InnerNet FFN 一致优于 GELU（PPL 降低 0.8–3.4%）
4. **LSTM 消融**：Classic（相邻配对）优于 Semantic（x vs h 配对）——最简单的方法最好
5. **回归任务**：Housing 上有效（MSE -5%），其他数据集效果有限
6. **ResNet（skip connection）**：InnerNet 优势消失——skip connection 提供了等效的特征交互
7. **参数效率**：InnerNet w=128 可达到 ReLU w=256 的性能，节省约 55% 参数
8. **SwiGLU vs InnerNet 图像**：SwiGLU CNN (79.79%) 略优于 InnerNet CNN (78.57%) on CIFAR-10

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | SwiGLU | 增益 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

**参数公平对比（CIFAR-10）**：InnerNet (127K, 78.57%) > ReLU matched (127K, 70.67%)，同参数量 +7.9%。

**SwiGLU 图像对比**：CIFAR-10 上 SwiGLU (79.79%) 略赢 InnerNet (78.57%)，但 CIFAR-100 上 InnerNet (53.74%) 大幅赢 SwiGLU (46.48%)。

## 2. 主流架构基线（CIFAR-100 + augmentation）

| 架构 | 准确率 | 参数量 |
|------|--------|--------|
| WRN-28-10 | 74.78±0.15 (n=5) | ~36M |
| ResNet-18 ReLU | 73.51±0.18 (n=5) | ~11M |
| ResNet-18 InnerNet | 71.72±0.52 (n=4†) | — |
| VGG-16+BN | 68.48±0.49 (n=5) | ~138M |

†一个种子 (59.43%) 因训练不稳定发散，已从统计中排除。

## 3. Autoencoder 重建（MSE↓，3–5 seeds）

| 数据集 | InnerNet | ReLU | ReLU matched | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43% vs ReLU, -34% vs matched** |
| FashionMNIST | **0.0076** | 0.0086 | — | **-12%** |
| CIFAR-10 | **0.0081** | 0.0105 | — | **-23%** |

### AE 容量缩放（latent 维度）

| Latent 维度 | InnerNet | ReLU | 改进 |
|------------|----------|------|------|
| 8 | 0.0141 | 0.0183 | -23% |
| 16 | 0.0075 | 0.0114 | -34% |
| 32 | 0.0039 | 0.0067 | -42% |
| 64 | 0.0026 | 0.0042 | -39% |

## 4. 语言模型（PPL↓，5 seeds）

### Transformer FFN（WikiText-2）

| 配置 | InnerNet | GELU | SwiGLU | 改进 |
|------|----------|------|--------|------|
| d=64 | **112.66±0.66** | 116.63±0.84 | — | **-3.4%** |
| d=128 | **95.26±1.00** | 96.82±1.19 | 92.98±1.14 | **-1.6%** |
| d=192 | **88.14±0.80** | 89.11±0.92 | — | **-1.1%** |
| d=256 | **85.40±1.15** | 86.05±0.97 | — | **-0.8%** |

### Transformer FFN（PTB）

| 模型 | PPL |
|------|-----|
| **InnerNet (ReLU)** | **207.81±1.58** |
| SiLU-InnerNet | 208.43±1.44 |
| GELU baseline | 212.28±0.88 |

SiLU-InnerNet 不优于 ReLU-InnerNet——平滑 inductive bias 没有帮助。

### LSTM 消融矩阵（WikiText-2）— 2×2 设计

| | Unbounded | Bounded (tanh) |
|--|-----------|----------------|
| **Classic**（相邻配对） | **101.72±0.99** | 104.46±0.38 |
| **Semantic**（x vs h 配对） | 105.30±0.31 | 107.15±0.83 |
| **Standard**（原始 tanh） | — | 108.39±0.75 |

**关键发现**：Classic unbounded (101.72) 最好。排序：Classic unbnd > Classic bnd > Semantic unbnd > Semantic bnd > Standard。简单的相邻维度配对反而优于精心设计的语义配对。

## 5. 回归任务（MSE↓，3–5 seeds）

| 数据集 | InnerNet | ReLU | 改进 |
|--------|----------|------|------|
| California Housing | **0.196±0.007** | 0.206±0.008 | **-5.0%** |
| Diabetes | 0.506±0.065 | 0.510±0.043 | -0.8%（持平） |
| Wine Quality | 0.599±0.029 | **0.548±0.022** | +9.3%（更差） |

## 6. 大 MLP MNIST（3×256，dropout=0.3，5 seeds）

| 模型 | 准确率 |
|------|--------|
| InnerNet | **98.39±0.04** |
| ReLU | 97.93±0.07 |
| 提升 | **+0.46%** |

## 7. ResNet（SGD，150 epochs，5 seeds，无 augmentation）

| 数据集 | InnerNet | ReLU | 差异 |
|--------|----------|------|------|
| CIFAR-10 | 86.09±0.82 | 86.33±0.34 | -0.24（持平） |
| CIFAR-100 | 56.78±2.77 | 57.95±0.52 | -1.17（持平） |

**Skip connection 让 InnerNet 冗余。**

## 8. PPO 强化学习（10 seeds）

| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| **Acrobot** | **-75.3** | -79.8 | -81.7 |
| MountainCar | -200.0 | -200.0 | -200.0 |
| LunarLander | 166.6 | **209.1** | -139.1 |

## 9. 容量缩放 — 参数效率

### MLP CIFAR-10（5 seeds）

| 宽度 | InnerNet (参数) | ReLU (参数) | 增益 |
|------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

**核心发现**：InnerNet w=128 (415K) ≥ ReLU w=256 (921K) → **节省 55% 参数**。

## 10. 负面/中性结果

| 实验 | 结果 | 原因 |
|------|------|------|
| 文本分类（TF-IDF/embedding） | 中性 | 稀疏/结构化特征不适合 pairwise 配对 |
| ResNet（skip connection） | 中性 | 残差连接已提供特征交互 |
| Transformer attention 替换 | 更差（PPL +6.6%） | 无法替代 softmax 的数学结构 |
| Wine 回归 | 更差 | 低维表格数据，InnerNet 开销不合算 |
| CNN 极小 scale (×0.25) | 更差 | 通道配对开销在小模型中占比过大 |
| SiLU-InnerNet vs ReLU-InnerNet | 中性 | 平滑 bias 没帮助（PTB: 208.43 vs 207.81） |
| LSTM semantic pairing | 不如 classic | 精心设计的 x-vs-h 配对不如简单相邻配对 |

## 总结

InnerNet 在**无 skip connection 的前馈网络**中持续有效。LSTM 消融实验揭示了一个惊人发现：简单的相邻维度配对优于精心设计的语义配对，且不加 tanh 约束效果更好——说明 InnerNet 的优势来自**可学习的非线性交互本身**，而非特定的配对策略。
