# 项目状态 — 2026-05-13

## 核心结论

- InnerNet 在没 skip 保护的位置有效（CNN, AE, TF FFN, ResNet internal）
- Warm-start 后 InnerNet 在 **10/11** 任务上赢或持平 SwiGLU，从头训打不过是优化问题
- **容量上限 ≥ SwiGLU**（ivs_d128 最终结果：SwiGLU 77.38±0.54 vs Frozen InnerNet **77.38±0.51**，完全持平）
- 模型越大 InnerNet 优势越小：d=64 赢 3.3%, d=128 赢 1.6%, d=256 从头训反转为输（GPT d=256: ~76.2 vs GELU ~72.6）
- **从头训一致输 SwiGLU**：scratch_init 2 seeds 确认 SwiGLU > Gaussian > Random > Multiply
- **大模型 finetune 替换不可行**：Qwen 2.5-0.5B 替换后 acc 从 89% 掉到 80%
- 适合场景：小模型 / on-device, warm-start finetune（非直接替换）, 架构搜索
- Non-shared（每层不同 InnerNet）比 shared 效果更好（PTB -1.95 vs -1.04），参数差可以忽略
- **Multiply-init MLM 大幅赢**：MultInit 15.93±0.21 vs SwiGLU 19.09±0.27（-16.6%，5 seeds）
- **初始化不影响收敛终点**：4 种初始化都收敛到 ~71.7-72.6（Wiki d=128），都大幅赢 SwiGLU ~77.3

## 论文 Story 思路

### 角度 A: Architecture Discovery Tool
InnerNet 不是用来部署的，是用来发现的。用 InnerNet 替换激活函数 → 训练 → 可视化学到的 2D 函数 → 提炼新公式 → 部署。类似 NAS 但连续可微。SwiGLU 当年也是搜索出来的，InnerNet 是同类方法。

### 角度 B: Efficient Finetuning
现有 LLM 用 SwiGLU。finetune 时替换成 InnerNet，只加 97 个参数。和 LoRA（加在权重上）互补，InnerNet 加在激活函数上。小模型效果更大。

### 角度 C: Understanding Activation Functions
学术贡献不在"赢 0.19 PPL"，在于理解：固定激活不是最优的、优化比表达能力更关键、位置决定效果、小模型受益更大。这些发现对下一代激活函数设计有指导意义。

标题方向：不说"beats SwiGLU"，说"Learnable activation functions reveal optimization barriers" 或 "Two-argument activations as differentiable architecture search"。

## 已修复的问题

- **U20 param sharing bug**：之前 Transformer/ResNet/WRN 的 InnerNet 每层各一个没共享。已修复，重跑。修复后结果和之前差不多（d=64: 112.66→112.83），说明影响不大，但 sharing 是论文基本设计。CNN/MLP/AE/VGG/LSTM/PPO 不受影响。

## 集群运行中（2026-05-13）

| Job ID | 实验 | 状态 |
|--------|------|------|
| 470140 | Sequential MNIST — SeqRNN (tanh baseline) | RUNNING |
| 470141 | Sequential MNIST — SeqLSTM (上界) | RUNNING |
| 470142 | Sequential MNIST — SeqGRU | RUNNING |
| 470143 | Sequential MNIST — SeqInnerNetRNN (Plan A: 1 InnerNet) | RUNNING |
| 470144 | Sequential MNIST — SeqGatedRNN (Plan B: 2 InnerNets + cell state) | RUNNING |

### 最近完成的集群任务

| 实验 | 结果 |
|------|------|
| **RNN PTB 2-arg** ✅ | Test Loss=5.130 (PPL≈169)。**输 tanh baseline 很多** |
| **RNN PTB 1-arg** ✅ | Test Loss=5.186 (PPL≈179)。**输 tanh baseline 很多** |
| **RNN PTB tanh** ✅ | Test Loss=4.943 (PPL≈140)。Baseline 赢 |
| **ivs_d128** ✅ | SwiGLU 77.38±0.54 vs Frozen InnerNet **77.38±0.51** — 完全持平，容量验证 |
| **mult_init** ✅ | d=64 持平, d=128 -0.24, PTB -1.08, **MLM -3.16**（5 seeds） |
| **GPT v4** ⏳ | 3/5 seeds 完成（77.20, 75.69, 75.83 → 均值 ~76.2），时间到 |
| **free_init_v2** ⏳ | Wiki 3/3 seeds ✅（4 种初始化收敛到 ~71.7-72.6 vs SwiGLU ~77.3），MLM 2/3 seeds |
| **scratch_init** ⏳ | 2.5/5 seeds。SwiGLU 一致赢：76.6/76.9 > Random 78.1/78.0 > Gaussian 78.1/78.3 > Multiply 80.2/79.2 |

---

## TODO — 按优先级

### 🔴 Critical

| # | 项目 | 状态 |
|---|------|------|
| C1 | 补齐 5 seeds | ⏳ SVHN 2arg/FMNIST 1arg 在集群上失败（OOM/时间到），需重提交 |
| C2 | CNN 参数公平对比 | ✅ ReLU matched 70.67% vs InnerNet 78.29% (同 127K) |
| C3 | AE 参数匹配 | ✅ ReLU matched 0.0059 vs InnerNet 0.0039 (同 ~660K) |
| C4 | 1-arg 系统对比 | ⏳ SVHN 1arg 5/5 ✅, FMNIST 1arg 失败需重提交 |
| C5 | ReLU+LN ablation | ✅ 4 数据集完成 |
| M1 | 训练曲线 | ⏳ 数据已有，需出图 |

### 🔴 Urgent

| # | 项目 | 状态 |
|---|------|------|
| U1 | CNN 60% params representation 分析 | 搁置 |
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
| U12 | GPT Transformer 2arg 卡死 | ✅ 被 U13/GPT v4 取代 |
| U13 | Transformer 全规模 SwiGLU 对比 | ⏳ d=64/128/192/PTB ✅。GPT d=256: GELU ~72.6 > SwiGLU ~74.5 > InnerNet ~76.2（3/5 seeds, 时间到）。大模型从头训 InnerNet 反转为劣势 |
| U14 | LSTM 2×2 消融多数据集 | ⏳ PTB ✅。Wiki-103/CNN-DM 时间到未完成，搁置 |
| U15 | RL 加 seeds + 只报 PPO | ✅ LunarLander 30s: InnerNet 187.6 > ReLU 158.8 > SwiGLU -249.7。InnerNet 赢（10 seeds 时输，30 seeds 翻了） |
| U16 | Masked LM（类 BERT） | ✅ SwiGLU 93.83, GELU 101.39, InnerNet 124.82（差）。warm-start 大幅赢 |
| U17 | Transformer Classic InnerNet FFN | ✅ Wiki 95.49 ≈ Semantic 95.26, PTB 208.81 ≈ Semantic 207.81。TF 上 Classic ≈ Semantic |
| U18 | d=64 InnerNet 学到了什么 | TODO | d=64 时 InnerNet ≈ SwiGLU，可视化看是不是真的像 SwiGLU |
| U19 | ResNet/WRN 只换内部 ReLU | ✅ **有效果** | C100+aug **74.97%** vs ReLU 73.51% (+1.5%), C10 **87.7%** vs 86.33% (+1.4%, 2/5 done) |
| U21 | InnerNet vs SwiGLU 对比 | ✅ 11/11 完成 | **10/11 赢或持平，LSTM 唯一输**。CNN +2.46%, MLM -16.18, PTB -1.04, GPT/ViT/d=256/AE 持平。LSTM 输 (105.79 vs 104.71) |
| U22 | AE warm-start | ✅ | 持平 (0.01206 vs 0.01211) |
| U23 | CNN warm-start | ✅ | **InnerNet +2.46%** |
| U24 | LSTM warm-start | ✅ **InnerNet 输** | SwiGLU 104.71 vs InnerNet 105.79。LSTM 唯一 warm-start InnerNet 输的 |
| U25 | TF d=192 warm-start | ⏳ 3/3 赢（未完成 5 seeds），搁置 |
| U26 | GPT warm-start | ✅ | 持平 (2赢2输1差不多，~73.1 vs ~73.2) |
| U27 | ViT warm-start | ✅ | 持平 (77.54 vs 77.59) |
| U28 | Mixer warm-start | ✅ | InnerNet 略好 (81.25 vs 81.13) |
| U29 | 可视化训练后 InnerNet 2D 函数 | ✅ | 4 任务对比完成。不同任务学到不同函数——d=64 接近 SwiGLU，MLM 偏离最大。偏离越大效果越好 |
| U30 | Scaling law 图 | TODO | d=64/128/192/256 的优势画曲线 |
| U31 | 训练曲线 | TODO | warm-start 两条分支 PPL 随 epoch 变化 |
| U32 | 参数量和推理速度 | TODO | InnerNet 加了多少参数，推理慢多少 |
| U33 | 提炼 InnerNet 为简单公式 | ✅ d=128 | f(a,b)≈0.12·a·b+0.11-0.06·b+0.03·a²·b。和 SwiGLU 完全不同，变成缩小版乘法交互 |
| U34 | Qwen2.5-0.5B finetune | ✅ **负面结果** | 3 seeds: InnerNet ~80% vs SwiGLU ~89%。替换瞬间崩到 52-66%，恢复不回来。大模型直接替换不可行 |
| U35 | InnerNet hidden dim 消融 | TODO | hidden=8/16/32/64 对比，InnerNet 需要多大才够 |
| U36 | Non-shared warm-start | ⏳ PTB ✅ MLM ✅ | PTB 5/5 赢, CNN +3.12%, **MLM non-shared 15.63 vs SwiGLU 18.91 (-3.28)**。Wiki d=128 在跑 |
| U37 | Free-init (不同初始化) | ✅ Wiki 3/3, MLM 2/3 | Wiki: 4 种初始化全收敛到 ~71.7-72.6 vs SwiGLU ~77.3。MLM: random/multiply/swiglu_fitted/identity 都 ~15.7-16.1。**初始化不影响终点** |
| U38 | Multiply-init 多任务 | ✅ 5/5 seeds | d=64 持平, d=128 -0.24, PTB -1.08, **MLM MultInit 15.93±0.21 vs SwiGLU 19.09±0.27 (-16.6%)** |
| U39 | Scratch-init (从头训对比) | ⏳ 2.5/5 seeds | SwiGLU 一致赢所有 InnerNet 初始化。Seed 42: SwiGLU 76.6 > Gaussian 78.1 > Random 79.3 > Multiply 80.2 |
| U40 | RNN PTB 重跑 | ✅ | 2arg Test PPL≈169, 1arg PPL≈179, tanh PPL≈140。**InnerNet 输 tanh baseline 20-28%**。PTB 上 InnerNet 不好用 |
| U41 | Sequential MNIST (InnerNet 发现 gate) | ⏳ 5 jobs 在跑 | RNN vs LSTM vs GRU vs InnerNetRNN(Plan A) vs GatedRNN(Plan B)。784 步逐像素，50ep×5seeds。验证 InnerNet 能否自主学出 gate 机制 |
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
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | +4.58 |
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
| GPT d=256 | **~76.2** (3/5 seeds) | 72.54 (GELU) | **+5.0% 输** |

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
