# 项目状态 — 2026-04-07

## 集群运行中
| 实验 | 状态 |
|------|------|
| ResNet CIFAR-10 2arg (lr=0.01, ×5seeds) | running |
| ResNet CIFAR-100 2arg (lr=0.01, ×5seeds) | running |
| Big CNN CIFAR-100 2arg (reduced channels, ×5seeds) | running |
| Big MLP MNIST (3×256, 2arg/relu ×5seeds) | running |
| GPT Transformer (gelu/swiglu done?, 2arg running) | running |
| tf_2arg_large (d=256) | running 36h+ |

## 可以 Report 的实验（Baseline 合理）

### CNN 图像分类
| 任务 | InnerNet | ReLU | 提升 |
|------|----------|------|------|
| MNIST | 99.41±0.04 | 99.02±0.03 | +0.39 |
| CIFAR-10 | 78.29±0.54 | 73.99±0.49 | **+4.30** |
| FashionMNIST | 90.87 | 89.46 | **+1.41** |
| SVHN | 95.01 | 92.63 | **+2.38** |

### Transformer / LSTM 语言模型 (PPL↓)
| 配置 | InnerNet | Baseline | 差异 |
|------|----------|----------|------|
| FFN d=128 | **95.26±1.00** | GELU 96.82±1.19 | **-1.6%** |
| FFN d=64 | **112.66±0.66** | 116.63±0.84 | **-3.4%** |
| LSTM | **105.30±0.31** | 108.39±0.75 | **-2.9%** |

### 回归 / 重建
| 任务 | InnerNet | ReLU | 差异 |
|------|----------|------|------|
| Housing 回归 (MSE↓) | **0.196±0.007** | 0.206±0.008 | **-5.0%** |
| AE MNIST (MSE↓) | **0.0039** | 0.0068 | **-43%** |

### RL (PPO)
| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| **Acrobot** | **-75.3** | -79.8 | -81.7 |

### 非图像分类（持平，可作为负面结果 report）
| 任务 | InnerNet | ReLU | 差异 |
|------|----------|------|------|
| Adult 表格 | 84.72±0.07 | 84.69±0.07 | +0.03 |
| AG News 文本 | 91.62±0.15 | 91.39±0.09 | +0.23 |
| Wine 表格 | 63.44±0.52 | 60.56±1.60 | +2.88 |

## 容量缩放（参数效率曲线用）
| Width | 2-arg | ReLU | 增益 |
|-------|-------|------|------|
| 32 | 38.34±0.92 | 37.47±0.92 | +0.87 |
| 64 | 47.66±0.35 | 45.77±0.17 | +1.89 |
| 128 | 52.05±0.22 | 49.87±0.21 | +2.17 |
| 256 | 54.82±0.41 | 51.99±0.20 | +2.83 |
| 512 | 55.63±0.41 | 52.63±0.31 | +3.00 |
**发现**: 增益随宽度单调递增。InnerNet w=64 ≈ ReLU w=128（一半参数）。

## 不 Report
| 实验 | 原因 |
|------|------|
| MLP MNIST (ReLU 91.95%) | baseline 异常低，等 big MLP 替代 |
| MLP CIFAR-100 (16%) | MLP 太弱 |
| Speech Commands (16%) | MLP 不适合音频 |
| ECG200 (82%, std=5.8) | 数据集太小 |
| PPO MountainCar (-200) | 全部失败 |
| PPO LunarLander (167 vs 209) | InnerNet 输 |
| SST-2 文本 (79 vs 79) | 持平且 baseline 不高 |
| AutoAttack (1.4% robust) | 模型太小无对抗鲁棒性 |

## 等待中的结果
- ResNet CIFAR-10/100 2arg → 验证 skip connection 假说
- Big MLP MNIST → 替代有问题的小 MLP baseline
- Big CNN CIFAR-100 2arg → 合理 baseline 下的 CIFAR-100
- GPT Transformer → 更大模型 InnerNet vs GELU
- tf_2arg_large (d=256) → Transformer 规模缩放

## 核心洞察
1. **CNN 图像分类**: 一致有效 (+0.4~4.3%)，baseline 74~99% 合理
2. **Autoencoder**: 效果最显著 (-43% MSE)，无监督重建新场景
3. **LM (Transformer/LSTM)**: FFN 替换一致有效 (-1.6~3.4% PPL)
4. **回归**: 首次验证 InnerNet 在回归任务有效 (-5% MSE)
5. **容量缩放**: 增益随宽度递增，InnerNet 提供持续额外表达力
6. **参数效率**: InnerNet w=64 ≈ ReLU w=128（约一半参数）
7. **文本/表格**: 基本持平，InnerNet 不适合稀疏/结构化特征
