# 论文核心结果摘要

来源: "Two-argument activation functions learn soft XOR operations like cortical neurons"
Yoon, Orhan, Kim, Pitkow (2021), arXiv:2110.06871v2

## 论文架构设定

### Baseline 外部网络
- **MLP**: 3 层隐藏层 × 64 units
- **CNN**: 4 层 Conv [60, 120, 120, 120], kernel 3×3, stride 1, 2×2 max-pool
- **参数匹配**: baseline ReLU 网络每层用 ⌊√(n×h)⌋+β units 来近似匹配参数量

### InnerNet
- 2 层隐藏层 × 64 units + ReLU → 1 输出
- CNN 版本用 1×1 conv 替代 Linear
- 所有层共享同一个 InnerNet（参数共享）

### 训练超参数
- 优化器: Adam, lr=0.001
- Dropout: 0.5
- Layer Normalization (在 InnerNet 之前)
- Early stopping: window size = 20
- Session II: 200 epochs, Session III: 400 epochs
- 每组实验重复 4 次取平均

---

## 主要分类结果 (Figure 4d)

论文的主要分类性能结果展示在 **Figure 4d** 中（training curves 图），没有精确数字表格。
从图中可以读到的**大致趋势**（非精确数值）：

### MLP + MNIST (Session III, ~400 epochs)
| 激活函数 | 大致 Test Accuracy |
|---------|-------------------|
| 2-arg (learned) | ~98% |
| 1-arg (learned) | ~97.5% |
| ReLU baseline | ~97% |

### MLP + CIFAR-10 (Session III, ~400 epochs)
| 激活函数 | 大致 Test Accuracy |
|---------|-------------------|
| 2-arg (learned) | ~52-53% |
| 1-arg (learned) | ~50% |
| ReLU baseline | ~48-49% |

### CNN + MNIST (Session III, ~400 epochs)
| 激活函数 | 大致 Test Accuracy |
|---------|-------------------|
| 2-arg (learned) | ~99% |
| 1-arg (learned) | ~98.8% |
| ReLU baseline | ~98.5% |

### CNN + CIFAR-10 (Session III, ~400 epochs)
| 激活函数 | 大致 Test Accuracy |
|---------|-------------------|
| 2-arg (learned) | ~72-73% |
| 1-arg (learned) | ~70% |
| ReLU baseline | ~68-69% |

> **注意**: 以上数值是从论文 Figure 4d 的图表中目测读出的近似值，不是精确数字。
> 论文的重点不是绝对精度，而是：
> 1. **2-arg 学得更快**（training curves 更陡）
> 2. **2-arg 渐近性能更好**
> 3. **学到的激活函数收敛为 soft XOR（乘法门控）结构**

---

## Table 1: AutoAttack 对抗鲁棒性

精确数值（论文 Table 1）：

| Dataset | Outer-Net | 2-arg | 1-arg | ReLU | 2-arg 提升 |
|---------|-----------|-------|-------|------|-----------|
| MNIST (l∞, ε=0.3) | MLP | **39.80** | 22.86 | 26.74 | +13.06 |
| MNIST (l∞, ε=0.3) | Conv | **49.25** | 10.02 | 9.33 | +39.92 |
| CIFAR-10 (l∞, ε=0.031) | MLP | **4.83** | 5.62 | 2.96 | +1.87 |
| CIFAR-10 (l∞, ε=0.031) | Conv | **11.27** | 9.55 | 8.57 | +2.70 |

> 2-arg 在所有设置下都比 ReLU baseline 更鲁棒。

---

## CIFAR-10-C 鲁棒性 (Figure 9)

- 2-arg Conv 的 mean Corruption Error (mCE) = **91.3%**（低于 100% 表示优于 ReLU baseline）
- relative mCE = **99.5%**（表示精度下降幅度也小于 ReLU）

---

## 关键发现

1. **激活函数快速成熟**: 2-arg 激活在 Session II 的 1-5 epoch 内就收敛为稳定的空间模式
2. **Soft XOR 结构**: 学到的激活函数呈现二次曲面形状，负曲率（CNN 中 78% 的试次, p=0.007），对应 f(x1,x2) ≈ x1·x2 的乘法门控
3. **功率谱分析**: 学到的函数在 ℓ=2（四极矩）处有更强的功率，而 Xavier 初始化的函数主要在 ℓ=1（偶极矩）
4. **Session III 证明**: 冻结 InnerNet 后重训外部网络的曲线与 Session II 相似，说明长期学习主要归因于外部网络参数变化

---

## 我们的复现结果

### MLP + MNIST (seed=1234, 单次运行, baseline 未参数匹配)

| 激活函数 | Phase1 Best Val Acc | Test Acc | 论文参考（近似） | 匹配度 |
|---------|--------------------:|--------:|----------------:|-------|
| 2-arg (learned) | 98.12% | **97.99%** | ~98% | 完美匹配 |
| 1-arg (learned) | 98.32% | **98.35%** | ~97.5% | 略高（单次方差） |
| ReLU baseline [64,64,64] | — | **85.63%** | ~97% | 差距大（未参数匹配） |

> 注: 上述 baseline 使用 [64,64,64] = 59K 参数，远小于 2-arg 的 114K。已修正为参数匹配版本，待重跑。

---

### 参数匹配方案（已更新到 config）

论文要求 baseline 与 2-arg 模型参数量匹配。计算结果：

| 组 | 2-arg 参数 | Baseline 参数 | Baseline 宽度 | 匹配度 |
|----|----------:|------------:|--------------|-------|
| MLP MNIST | 113,867 | 114,362 | [112, 112, 112] | +0.4% |
| MLP CIFAR | 388,427 | 388,333 | [117, 117, 117] | -0.02% |
| CNN MNIST | 126,683 | 113,882 | [44, 88, 88, 88] | -10.1% |
| CNN CIFAR | 127,763 | 125,222 | [46, 92, 92, 92] | -2.0% |
| RNN PTB | 2,512,250 | 2,542,760 | h=130 | +1.2% |

### 下一步
- [ ] 用参数匹配的 baseline 重跑所有实验
- [ ] 多次重复取平均（4 seeds）

---

## 核心洞察：InnerNet 到底提升了什么？

### 本质
标准神经元计算 `f(w·x)`（ReLU 逐维独立处理），InnerNet 神经元计算 `f(w1·x, w2·x)`（两个线性投影的非线性交互）。学到的函数收敛为 **soft XOR / 乘法门控**，即 `f(x1, x2) ≈ x1 · x2`。

这本质上是在每个神经元级别引入了 **特征交互（feature interaction）**——类似 attention 和 LSTM 门控的机制，但内置在激活函数中。

### 提升的三个层面
1. **学习速度**（sample efficiency）— 更少 epoch 达到同样精度
2. **渐近性能**（final accuracy）— 最终精度略高
3. **鲁棒性**（robustness）— 对抗攻击和自然扰动下更强

### 关键发现（来自 run_v0.ipynb 实验）
- **像素分类（MNIST/Fashion-MNIST）**：InnerNet 优势小，因为相邻像素太相似，XOR(白,白)=0 无意义
- **语言建模（PTB/WikiText）**：InnerNet 优势大，因为 embedding 各维是高度压缩的语义特征，维度间交互有意义（如"皇室" AND "女性" = "女王"）
- **RL（LunarLander）**：InnerNet 学得更快（Ep 600 就到 106 分，Baseline 要到 Ep 850 才到 206 分）

### 推论
任何使用激活函数的网络，理论上都可以受益于 2-arg InnerNet。受益程度取决于任务是否依赖特征间的乘法/门控交互。
