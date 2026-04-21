# 项目状态 — 2026-04-14

## 已修复的问题

- **U20 param sharing bug**：之前 Transformer/ResNet/WRN 的 InnerNet 每层各一个没共享。已修复，重跑。修复后结果和之前差不多（d=64: 112.66→112.83），说明影响不大，但 sharing 是论文基本设计。CNN/MLP/AE/VGG/LSTM/PPO 不受影响。

## 集群运行中

**优先级：Transformer > ResNet > LSTM**

| 类别 | 实验 | 状态 |
|------|------|------|
| U20 | GPT InnerNet v4 | running |
| U19 | ResNet C10/C100 internal ×6 | pending |
| U14 | LSTM Wiki-103 ×15 | running |
| U14 | ~~LSTM CNN/DM~~ | 暂停（占太多 GPU，以后再跑） |

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
| U2 | 主流 CNN (aug fix 后重跑) | ✅ ResNet 73.51%, VGG 68.69%, WRN 74.80% — 合理 baseline。ResNet InnerNet 71.72% (n=4, 1 outlier) |
| U3 | SwiGLU CNN 图像 | ✅ CIFAR-10: SwiGLU 79.79% > InnerNet 78.57%. CIFAR-100: InnerNet 53.74% > SwiGLU 46.48% |
| U4 | LSTM 消融 (2×2) | ✅ 全部完成: Classic unbnd **99.33** > Classic bnd 101.76 > Semantic unbnd 103.41 > Standard 104.38 > Semantic bnd 105.59 |
| U5 | SiLU-InnerNet Transformer | ✅ PTB 208.43, WikiText-2 94.90。SiLU 和 ReLU InnerNet 差不多 |
| U6 | InnerGate（备选，如 SiLU 不行）| TODO | `b × sigmoid(InnerNet(a,b))`：b 保留直达通路 + gate 双向感知。上限最高但有点"作弊"（结构太接近 SwiGLU） |
| U7 | 训练阶段消融 | ✅ 4/4 完成 | AE/ResNet/MLP: e2e ≈ 3-phase。CNN e2e 77.35% vs 3-phase 78.57% 稍差 -1.2%。pretrain 不是必须的 |
| U8 | ResNet InnerNet 训练不稳定 | ⏳ 5 seeds 完成，seed 44=59.43% outlier。n=4 均值 71.72±0.52 vs ReLU 73.51±0.18。需要 lr warmup 或更低 lr |
| U9 | Small CNN CIFAR-100 参数不公平 | TODO | SwiGLU 46.48% 远超 InnerNet 34.65%/ReLU 29.70%。SwiGLU 训练 400ep 不分阶段 vs InnerNet 3-phase。需要公平对比（同 epoch 或 end-to-end InnerNet） |
| U10 | LSTM PTB vs WikiText 结论不一致 | TODO | WikiText: classic>semantic。PTB: semantic>classic。需要解释或更多数据集验证 |
| U11 | VGG-16 SwiGLU | ❌ lr=0.01+grad_clip 仍 1% 准确率。SwiGLU 与 VGG 深层 conv+SGD 不兼容，记为负面结果 |
| U12 | GPT Transformer 2arg 卡死 | ⏳ 已取消重启 |
| U13 | Transformer 全规模 SwiGLU 对比 | ⏳ d=64/128/192/PTB ✅。d=256 pending。GPT(d=256): GELU 72.54 > SwiGLU 75.30，大模型趋势不同 |
| U14 | LSTM 2×2 消融多数据集 | ⏳ PTB ✅。Wiki-103 在跑(单 seed 并行)。CNN/DM 暂停(占太多 GPU，Transformer 优先) |
| U15 | RL 加 seeds + 只报 PPO | ✅ LunarLander 30s: InnerNet 187.6 > ReLU 158.8 > SwiGLU -249.7。InnerNet 赢（10 seeds 时输，30 seeds 翻了） |
| U16 | Masked LM（类 BERT） | ⏳ SwiGLU 93.83, GELU 101.39 完成。InnerNet 在跑 |
| U17 | Transformer Classic InnerNet FFN | ✅ Wiki 95.49 ≈ Semantic 95.26, PTB 208.81 ≈ Semantic 207.81。TF 上 Classic ≈ Semantic |
| U18 | d=64 InnerNet 学到了什么 | TODO | d=64 时 InnerNet ≈ SwiGLU，可视化看是不是真的像 SwiGLU |
| U19 | ResNet/WRN 只换内部 ReLU | ✅ **有效果** | C100+aug **74.97%** vs ReLU 73.51% (+1.5%), C10 **87.7%** vs 86.33% (+1.4%, 2/5 done) |
| U21 | InnerNet vs SwiGLU 对比 | ⏳ 多配置在跑 | d=128: InnerNet 76.85 vs SwiGLU 77.04 (4/5 赢)。**MLM: InnerNet 37-39 vs SwiGLU 52-57 大幅赢**。PTB: 3/3 赢。d=64: 差不多但 frozen 赢。d=256: 在跑 |
| U22 | AE warm-start | ⏳ 已提交 | AE SwiGLU vs InnerNet，5 seeds |
| U23 | CNN CIFAR-10 warm-start | ⏳ 已提交 | SwiGLU 79.79 赢了，试反转 |
| U24 | LSTM warm-start | ⏳ 已提交 | LSTM SwiGLU vs InnerNet |
| U25 | TF d=192 warm-start | ⏳ 已提交 | 5 seeds |
| **U20** | **修复 InnerNet parameter sharing** | ✅ TF 全完成 | d=64 112.83, d=128 95.23, d=192 88.42, **d=256 84.62**, PTB 207.91。全部赢 GELU。ResNet full 持平, internal +1.5%。MLM 124.82 差 |

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

### LSTM 消融 (WikiText-2) ✅
| 变体 | Best PPL | Last PPL |
|------|---------|----------|
| Classic（相邻配对） | **99.33** | 101.72±0.99 |
| Semantic（x vs h） | 103.41 | 105.30±0.31 |
| Standard (baseline) | 104.38 | 108.39±0.75 |

*Bounded (tanh) 变体已测试并归档——加 tanh 约束一致更差。*

### 其他已确认结果
- AE: MNIST -39%, FashionMNIST -12%, CIFAR-10 -26%
- Housing 回归: -5% MSE
- Big MLP MNIST: +0.46%
- PPO Acrobot: +5.6%
- 参数效率: MLP w=128 ≈ ReLU w=256 (55% savings)
- ResNet: InnerNet ≈ ReLU (skip connection 消除优势)

---

## 不再追踪（已归档至 `archive/`）

| 实验 | 原因 | 归档位置 |
|------|------|---------|
| MLP CIFAR-100 (16%) | baseline 太低 | archive/configs/, archive/exp/ |
| Speech Commands MLP (16%) | MLP 不适合音频 | archive/configs/, archive/exp/ |
| ECG200 (82%, std=5.8) | 数据集太小 | archive/configs/, archive/exp/ |
| STL-10 (57% vs 59%) | InnerNet 输 | archive/configs/, archive/exp/ |
| 文本分类 SST-2/AG News/Wine/Adult | 稀疏特征不适合 | archive/configs/, archive/exp/ |
| DQN RL (全部) | 改用 PPO | archive/models/dqn.py, archive/runners/rl_runner.py |
| Transformer attention 替换 | +6.6% PPL，失败 | archive/configs/ |
| 旧版小 CNN CIFAR-100 | 被 big 版本取代 | archive/configs/, archive/exp/ |
| PPO MountainCar (-200) | 全部失败 | 仍在 exp/（PPO 本身不归档） |
| LSTM Bounded (tanh) 变体 | 加 tanh 一致更差 | archive/configs/ |
| Cora / S4/Mamba | 改动太大 | 未实现 |
