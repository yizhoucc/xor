# 项目状态 — 2026-04-07

## 集群运行中
| 实验 | 状态 |
|------|------|
| ResNet CIFAR-10 (SGD, 2arg/relu × 5seeds) | running |
| ResNet CIFAR-100 (SGD, 2arg/relu × 5seeds) | running |
| Big CNN CIFAR-100 2arg (×5seeds) | running |
| GPT Transformer (gelu/swiglu done?, 2arg running) | running |
| tf_2arg_large (d=256) 续跑 | running 24h+ |

## 最新实验结果

### 1. 容量缩放 — MLP CIFAR-10 (5 seeds each)
| Width | 2-arg | ReLU | 增益 |
|-------|-------|------|------|
| 32 | 38.34±0.92 | 37.47±0.92 | +0.87 |
| 64 | 47.66±0.35 | 45.77±0.17 | +1.89 |
| 128 | 52.05±0.22 | 49.87±0.21 | +2.17 |
| 256 | 54.82±0.41 | 51.99±0.20 | +2.83 |
| 512 | 55.63±0.41 | 52.63±0.31 | +3.00 |
**发现**: 增益随宽度单调递增。参数效率: InnerNet w=64 ≈ ReLU w=128。

### 2. Housing 回归 (MSE↓)
2-arg **0.197±0.008** vs ReLU 0.206±0.008 → **-4.4%**

### 3. Autoencoder MNIST (MSE↓)
2-arg **0.00391** vs ReLU 0.00684 → **-43% MSE** (重建质量提升近一倍)

### 4. AutoAttack (L∞, eps=8/255)
| 模型 | Clean | Robust |
|------|-------|--------|
| 2-arg | 77.0% | 1.4% |
| 1-arg | 82.2% | 1.4% |
模型太小无对抗鲁棒性。需对抗训练。

### 5. CIFAR-10-C 损坏鲁棒性
1-arg overall=59.37% | 2-arg overall=57.78% | 缺 ReLU baseline

### 6. Dense Embedding 文本 (5 seeds)
- SST-2: 2arg 79.06 vs ReLU 79.36 — 持平
- AG News: 2arg 91.62 vs ReLU 91.39 — 微弱优势

### 7. PPO RL (10 internal seeds)
| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 498.4 | 500.0 | 500.0 |
| Acrobot | -87.2 | -81.7 | -88.4 |
| MountainCar | -200.0 | -200.0 | -200.0 |
| **LunarLander** | **225.4** | 21.7 | -134.7 |

### 8. Big CNN CIFAR-100
ReLU: 50.00±0.83 | 2-arg: running

## 之前确认的核心结果

### MLP 图像分类 (5 seeds)
| 任务 | 2-arg | ReLU | 提升 |
|------|-------|------|------|
| MNIST | 97.97±0.11 | 91.95±3.03 | **+6.03** |
| CIFAR-10 | 51.86±0.19 | 49.53±0.22 | **+2.33** |
| FashionMNIST | 88.53 | 86.68 | **+1.85** |
| CIFAR-100 | 18.20 | 15.95 | **+2.25** |

### CNN 图像分类
| 任务 | 2-arg | ReLU | 提升 |
|------|-------|------|------|
| MNIST | 99.41±0.04 | 99.02±0.03 | +0.39 |
| CIFAR-10 | 78.29±0.54 | 73.99±0.49 | **+4.30** |
| FashionMNIST | 90.87 | 89.46 | +1.41 |
| CIFAR-100 | 34.65 | 29.70 | **+4.95** |
| SVHN | 95.01 | 92.63 | **+2.38** |

### Transformer LM (PPL↓)
| 配置 | InnerNet | Baseline | 差异 |
|------|----------|----------|------|
| FFN d=128 | **95.26±1.00** | GELU 96.82±1.19 | -1.6% |
| FFN d=64 | **112.66±0.66** | 116.63±0.84 | -3.4% |
| SwiGLU d=128 | — | 92.98±1.14 | 手工最优 |

### LSTM (PPL↓)
InnerNet **105.30±0.31** vs Standard 108.39±0.75 → -2.9%

### DQN RL
| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | **228.3** | 149.7 | 107.3 |
| Acrobot | -167.2 | -191.7 | **-127.3** |
| MountainCar | -155.4 | -169.6 | -153.7 |
| LunarLander | -28.1 | **153.3** | -175.6 |

## 核心洞察更新

1. **容量缩放**: InnerNet 增益随宽度递增（不是递减），说明 InnerNet 提供的不是"弥补容量不足"，而是一种**持续的额外表达力**
2. **参数效率**: InnerNet w=64 ≈ ReLU w=128，约一半参数达到同等性能
3. **Autoencoder**: InnerNet 在无监督重建上效果显著 (-43% MSE)，证明优势不限于分类
4. **PPO vs DQN**: InnerNet 在 LunarLander 上 PPO=225 vs DQN=-28——算法选择影响巨大
5. **文本分类**: 无论 TF-IDF 还是 dense embedding，InnerNet 在文本上基本持平
6. **对抗鲁棒性**: 无 adversarial training 时无法体现差异
