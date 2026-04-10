# 项目状态 — 2026-04-09

## 集群运行中 (~64 jobs)

| 类别 | 实验 | 状态 |
|------|------|------|
| Critical seeds | CNN SVHN 2arg ×3, FMNIST 1arg ×4, CIFAR 2arg ×1 | running |
| Urgent | ResNet-18+aug CIFAR-100 (2arg/relu ×5) | running |
| Urgent | VGG-16+BN CIFAR-100 (×5) | running |
| Urgent | WideResNet-28-10 CIFAR-100 (×5) | running |
| Urgent | SwiGLU CNN CIFAR-10/100 (×5 each) | running |
| Urgent | Representation 分析 (U1) | running |
| LSTM 消融 | Bounded ×2, Classic ×2, BoundedClassic ×2 (Wiki+PTB) | running |
| Seeds 补齐 | Diabetes ×4, Wine_reg ×4 | running |
| GPT | Transformer 2arg + swiglu (d=256) | running (2d+) |

---

## TODO — 按优先级

### 🔴 Critical

| # | 项目 | 状态 |
|---|------|------|
| C1 | 补齐 5 seeds | ⏳ SVHN 2arg/FMNIST 1arg/CIFAR 2arg 最后几个在跑 |
| C2 | CNN 参数公平对比 | ✅ ReLU matched 70.67% vs InnerNet 78.29% (同 127K) |
| C3 | AE 参数匹配 | ✅ ReLU matched 0.0059 vs InnerNet 0.0039 (同 ~660K) |
| C4 | 1-arg 系统对比 | ⏳ SVHN 1arg 5/5 ✅, FMNIST 1arg 在跑 |
| C5 | ReLU+LN ablation | ✅ 4 数据集完成 |
| M1 | 训练曲线 | ⏳ 数据已有，需出图 |

### 🔴 Urgent

| # | 项目 | 状态 |
|---|------|------|
| U1 | CNN 60% params representation 分析 | ⏳ 脚本在跑 |
| U2 | 主流 CNN (ResNet-18+aug, VGG-16+BN, WRN-28-10) | ⏳ 全部已提交 |
| U3 | SwiGLU CNN 图像对比 | ⏳ CIFAR-10/100 已提交 |
| U4 | LSTM 消融 (2×2: semantic/classic × bounded/unbounded) | ⏳ 全部已提交 |
| U5 | SiLU-InnerNet Transformer (WikiText-2 + PTB) | ⏳ 已提交 | InnerNet 内部 ReLU→SiLU，看能否靠近 SwiGLU |
| U6 | InnerGate（备选，如 SiLU 不行）| TODO | `b × sigmoid(InnerNet(a,b))`：b 保留直达通路 + gate 双向感知。上限最高但有点"作弊"（结构太接近 SwiGLU） |

### 🟡 Major

| # | 项目 | 状态 |
|---|------|------|
| M2 | XOR 可视化（2D 激活函数表面） | TODO |
| M3 | CNN 小 scale 反转解释 | TODO（论文讨论） |
| M4 | 回归 inconsistency 解释 | TODO |
| M5 | RL inconsistency | TODO（降级 preliminary） |
| M6 | PReLU/Swish baseline 对比 | TODO |

### 🟢 Minor

| # | 项目 | 状态 |
|---|------|------|
| m1 | ResNet baseline 提升 | 被 U2 覆盖 |
| m2 | 论文原始数字复现对比表 | TODO |
| m3 | 更多 LM dataset | ✅ PTB 已完成 |
| m4 | 计算开销分析（FLOPs + wall-clock） | TODO |
| m5 | 显著性检验 p-value | TODO |
| 24 | 参数效率出图 | 数据已有 |

---

## 最新确认结果

### CNN 图像分类（完整 ablation）
| Dataset | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | Gain |
|---------|-------|-------|------|---------|-------------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | +0.39 |
| CIFAR-10 | 78.29±0.54 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | +4.30 |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | +1.57 |
| SVHN | ⏳(1 seed) | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | +3.74 |

### Transformer LM (PPL↓)
| 配置 | InnerNet | Baseline | 差异 |
|------|----------|----------|------|
| d=64 | **112.66** | 116.63 | -3.4% |
| d=128 | **95.26** | 96.82 (GELU) | -1.6% |
| d=192 | **88.14** | 89.11 | -1.1% |
| d=256 | **85.40** | 86.05 | -0.8% |
| PTB d=128 | **207.81** | 212.28 | -2.1% |
| GPT d=256 | ⏳ | 72.54 | — |

### LSTM 消融矩阵 (WikiText-2)
| | Unbounded | Bounded |
|--|-----------|---------|
| Semantic | **105.30** ✅ | ⏳ |
| Classic | ⏳ | ⏳ |
| Standard | — | 108.39 ✅ |

### 其他已确认结果
- AE: MNIST -39%, FashionMNIST -12%, CIFAR-10 -26%
- Housing 回归: -5% MSE
- Big MLP MNIST: +0.46%
- PPO Acrobot: +5.6%
- 参数效率: MLP w=128 ≈ ReLU w=256 (55% savings)
- ResNet: InnerNet ≈ ReLU (skip connection 消除优势)

---

## 不再追踪

| 实验 | 原因 |
|------|------|
| MLP CIFAR-100 (16%) | baseline 太低 |
| Speech Commands MLP (16%) | MLP 不适合音频 |
| ECG200 (82%, std=5.8) | 数据集太小 |
| PPO MountainCar (-200) | 全部失败 |
| Cora / S4/Mamba | 改动太大 |
| STL-10 (57% vs 59%) | InnerNet 输，方差大 |
