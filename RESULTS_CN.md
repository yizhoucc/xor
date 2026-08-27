# 实验结果笔记 — InnerNet

> 内部文档。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

### 定位：可微的架构探针（NOT "更好的激活函数"）

把 ReLU 换成一个小 MLP（两输入一输出），让每个神经元能看到隔壁特征。但**我们不卖"它是更强的激活函数"**——因为从头训在大模型上打不过 SwiGLU、直接替换又慢又掉点（Qwen -9%）。如果按"better activation"写，reviewer 会说"和原论文（IEEE Access 2022）一样、marginal、推理慢、没价值"。

**定位：用它探测给定优化 basin 中的二元架构基元。** SwiGLU-host 内，random / identity / multiply 等 InnerNet 初值都会走到 SwiGLU-like 表面；但新的 Bilinear-host 因果实验中，4 个完成的 joint seeds 全部保持纯 `a·b`（mult R²≥0.998），没有转成 SwiGLU，同时 PPL 也达到约 73.5。结论因此不是“存在唯一、网络无关的 SwiGLU 最优形态”，而是 **InnerNet 能从不同初值优化/恢复与外围网络 basin 相容的高性能二元算子**。这是更可信也更一般的 architecture-probe 结论。

功能性结果（不作机制性 claim）：Sequential MNIST 约束模型以 21K 参数在 8/9 个实际训练 seed 达到 98.44±0.22%，但单个 InnerNet 表面仍不可辨识为标准 gate。

### 相对原论文（Yoon et al., IEEE Access 2022）的新意

原论文只测 MLP+CNN、只在 MNIST/CIFAR 上、结论温和（简单任务略有提升 + 鲁棒 + 生物动机）。我们超出的：**架构广度**（Transformer/LSTM/RNN/AE/ResNet/ViT/Mixer/PPO/MLM/GPT 10+ 个）+ **系统性边界**（位置效应、优化壁垒≠容量上限、scaling 反转、可解释表面提炼，以及加法记忆 + 二元交互的功能性结果）。

### 实验事实速查

确认有效的：CNN、AE、Transformer FFN（d=64~256 全赢 GELU）、LSTM WikiText-2、PPO、ResNet internal-only、Seq-MNIST 加法记忆 + InnerNet（成功 runs ~98%）。
不好用的：ResNet 全换（skip connection 冗余）、LSTM PTB、大模型从头训（GPT d=256 反转）、大模型直接替换（Qwen -9%）。

**关键发现**：InnerNet 容量上限 ≥ SwiGLU（warm-start 10/11 赢或持平，ivs_d128 追平），从头训大模型输是优化问题。模型越大差距越大（d=64 赢 3.3%, d=256 输 5.2%）。适合"发现 + warm-start finetune"，不适合大模型从头训或直接部署替换。

### 投稿前剩余工作

1. **结果源审计**：canonical manifest 已建立；继续清理修复前后结果混用、缺原始来源和不完整 seeds。
2. **统计严谨性**：paired/unpaired 工具已完成；继续为正文核心结果生成 raw seeds、差值区间、效应量和配对/非配对检验。
3. **gate 稳定性必须透明报告**：原 Plan B 为3/5成功、2 NaN；约束版本去掉 `W_c` 后为8/9实际训练成功、1 NaN，另有1次基础设施OOM。两套分母都应报告，不能只选成功runs。

部署不是投稿前置条件。CNN deploy 已完成，FFN deploy 因时限中止；它们只作为扩展结果，不阻塞以“发现”为核心的论文。

**部署结果**：CNN 上 InnerNet 为 **84.95±0.57%**，提炼后的固定 poly3 为 **81.32±0.32%**，ReLU/SwiGLU 约为79.9%（5 seeds，sample SD）。固定算子比 InnerNet 快 **2.68×**，但仍比 SwiGLU 慢2.46×，并损失3.63个准确率点；它仍显著优于 ReLU/SwiGLU约1.4点。说明“发现→提炼”能回收部分效率和收益，但现有手写多项式没有完成无损部署。FFN deploy 超时，只有 InnerNet 4 seeds且 distilled 0 seeds；已完成分支显示 InnerNet训练吞吐约比SwiGLU慢6.03×，不据此声称FFN部署成功。详见 `results/audit/deploy_analysis.json`。

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | 95.016±0.005 (n=3) | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

参数公平对比：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

2026-08-27 原始结果补拉：FashionMNIST 2-arg 已达到 5 seeds（90.91±0.29%，population SD），SVHN 1-arg 已达到 5 seeds（95.16±0.23%）；SVHN 2-arg 当前 3 seeds 为 95.016±0.005%，仍需补 seeds 44/45。对应 config：`config/experiments/cnn_fmnist_2arg.yaml`、`cnn_svhn_1arg.yaml`、`cnn_svhn_2arg.yaml`；exp 模式：`exp/cnn_{fmnist,svhn}_{1arg,2arg}_*`。

多 seed 训练曲线见 `results/figures/fig_training_curves.pdf`（MLP/CNN × MNIST/CIFAR-10，mean±sample SD）。图中 MLP-MNIST ReLU 使用参数匹配的 width-112 配置，已排除同名旧 width-64 run，避免把两套架构混入方差带。

## 2. 自编码器

| 数据集 | InnerNet | ReLU | ReLU 参数匹配 | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | — | -12% |
| CIFAR-10 | **0.0081** | 0.0105 | — | -23% |

效果最好的。容量缩放 latent=32 改进最大 (-42%)。

Configs: `config/experiments/ae_mnist_2arg.yaml`，exp: `exp/ae_mnist_2arg_*`

## 3. Transformer FFN

| 配置 | GELU | SwiGLU | InnerNet | vs GELU |
|------|------|--------|----------|---------|
| Wiki d=64 | 116.63 | 112.31 | **112.83** | **-3.3%** |
| Wiki d=128 | 96.82 | 92.98 | **95.26** | **-1.6%** |
| Wiki d=192 | 89.11 | 85.43 | **88.42** | **-0.8%** |
| Wiki d=256 | 86.05 | 81.56 | **84.62** | **-1.7%** |
| PTB d=128 | 212.28 | 205.82 | **207.91** | **-2.1%** |
| **GPT d=256** | **~72.6** | ~74.5 | **~76.2** (3/5) | **+5.0% 输** |

d=64 到 d=256 InnerNet 一直赢 GELU（-3.3% → -1.7%）。但 GPT d=256（更大规模）**反转为输** +5.0%。

四个规模都是 same-width 比较，并非参数匹配。相对 GELU 的 paired-t p 值依次为 d=64: 0.00022、d=128: 0.05095、d=192: 0.361、d=256: 0.173；均值方向一致，但统计证据主要集中在最小规模。英文表中的 `±` 统一使用 population SD（ddof=0）。

修复 parameter-sharing 后的 scaling 图由 canonical audit 数据直接生成：`results/figures/fig_scaling_law.pdf`。旧硬编码 pre-fix 图已被替换；当前趋势是正向但**非单调**，不能再写成“规模越大优势单调缩小”。

Scaling 趋势：d=64 赢 3.3% → d=128 赢 1.6% → d=192 赢 0.8% → d=256 赢 1.7% → **GPT d=256 输 5.0%**。

GPT 训练曲线显示 InnerNet 在 epoch 20 时还没收敛（仍在下降），GELU 每 epoch 28 分钟而 InnerNet 3 小时。同样 20 epochs，InnerNet 优化负担更重。但 warm-start 实验已证明容量足够（ivs_d128: 77.47 追平 SwiGLU 77.50）。

SwiGLU 是 InnerNet 的子集。从头训 InnerNet 打不过 SwiGLU，做了两个对比实验：

**公平对比（各 20ep，5 seeds）**：SwiGLU 训 10ep → fork → 各续 10ep

| | SwiGLU 20ep | InnerNet 20ep |
|--|-------------|---------------|
| 均值 | 77.04±0.79 | **76.85±0.63** |
| 赢的 seeds | 1/5 | **4/5** |

InnerNet 从 SwiGLU 初始化后继续训，4/5 seeds 赢了，均值好 0.19 PPL。

**Frozen（只动 InnerNet，5 seeds）**：SwiGLU 20ep 收敛 → freeze network → non-shared InnerNet 训到收敛

| | SwiGLU | Frozen InnerNet |
|--|--------|-----------------|
| 均值 | 77.04 | 77.11 |

持平。只动 InnerNet 参数超不过 SwiGLU。改进主要来自 InnerNet 和 network 一起调整。

**Warm-start 全配置对比（SwiGLU 训到 make sense → InnerNet 替换 → 各自继续训）**：

| 实验 | SwiGLU | Shared InnerNet (97p) | 差 | Non-shared (388p) | 差 |
|------|--------|----------------------|-----|-------------------|-----|
| **CNN CIFAR-10** | 83.33% | **85.79%** | **+2.46%** | — | — |
| **MLM** | 54.92 | **38.74** | **-16.18** | ⏳ | — |
| PTB d=128 | 162.22 | **161.18** | -1.04 | **162.49** | **-2.11**† |
| TF d=128 | 77.04 | **76.85** | -0.19 | ⏳ | — |
| MLM (non-shared) | 18.91 | 38.74 (shared) | -16.18 | **15.63** | **-3.28** |
| TF d=64 | 92.08 | **91.93** | -0.15 | — | — |
| TF d=192 | ⏳ | ⏳ 3/3 赢 | — | — | — |
| Mixer | 81.13% | **81.25%** | +0.12% | — | — |
| ViT | 77.54% | 77.59% | 持平 | — | — |
| TF d=256 | — | — | 持平 | — | — |
| AE | 0.01206 | 0.01211 | 持平 | — | — |
| GPT d=256 | ~73.1 | ~73.2 | 持平 | — | — |
| **LSTM** | **104.71** | 105.79 | **+1.08 输** | — | — |

Shared: **10/11 赢或持平，LSTM 唯一输。**
Non-shared PTB 赢的幅度约是 shared 的 2 倍（-2.11 vs -1.04）。每层学到了不同的函数（fig11）。

†注意：non-shared PTB 是配对 warm-start，比的是它自己单独训的 SwiGLU baseline（InnerNet 162.49 vs SwiGLU 164.59，seeds 42–46，InnerNet 5/5 全赢，paired-t p=0.037，dz=-1.38）。这个 164.59 跟上表 SwiGLU 列的 162.22 不是同一个 run，不能直接比。之前表里写 162.64/-1.95 是把两个不同 baseline 混用了，已按原始 `exp/warmstart_nonshared/results.p` 更正。

Non-shared 更符合生物学（不同区域的神经元激活特性不同）。参数差 291 个，可以忽略。

**ivs_d128 容量验证** ✅：最终结果（5 seeds）：SwiGLU **77.38±0.54** vs Frozen InnerNet **77.38±0.51**。**完全持平**，证明 InnerNet 容量 ≥ SwiGLU。

对应的逐 epoch 训练曲线已生成：`results/figures/fig_warmstart_curves.{png,pdf}`。左图显示共同 SwiGLU 预训练后分叉为继续 SwiGLU 与可训练 InnerNet，右图显示冻结外部网络、只训练 InnerNet 的容量验证；两图均汇总 5 seeds 的 mean±sample-SD。原始数据：`exp/ivs_d128_v2/results.p`。

### InnerNet 初始化不重要

Free-init 实验 ✅ Wiki d=128 3 seeds 完成，MLM 2/3 seeds：

**Wiki d=128**（3 seeds）：

| 初始化 | Seed 42 | Seed 43 | Seed 44 | 均值 |
|--------|---------|---------|---------|------|
| swiglu_fitted | 71.98 | 71.98 | 72.60 | ~72.2 |
| multiply | 71.91 | 71.91 | 72.54 | ~72.1 |
| random | 71.72 | 71.72 | 72.23 | ~71.9 |
| identity | 71.99 | 71.99 | 72.33 | ~72.1 |
| **SwiGLU baseline** | **77.29** | **77.29** | **77.24** | **~77.3** |

**MLM**（2/3 seeds）：random 15.86, multiply 15.74-16.04, swiglu_fitted 15.91-16.05, identity 15.96

4 种初始化全收敛到同一水平，都大幅赢 SwiGLU（Wiki: ~72 vs ~77, MLM: ~16 vs ~19）。**初始化不影响终点，只影响收敛速度。** 之前担心的"warm-start 让 InnerNet 太像 SwiGLU"不是问题。

CNN 和 MLM 效果最大。MLM 从头训 InnerNet 124.82 完全不行，warm-start 后 38.74 大幅赢 SwiGLU 54.92。从头训不动是优化问题。

结论：给个好初始化，InnerNet 在多数任务上赢 SwiGLU。上限更高。

模型越大差距越小（d=64 赢 0.15, d=128 赢 0.19, d=256 差不多）。因为大模型架构本身足够复杂，单个激活函数的边际贡献小。InnerNet 更适合小模型 / on-device 场景和 finetune 阶段。

### Qwen2.5-0.5B finetune（真实预训练模型）— 负面结果

✅ 3 seeds 完成。大模型直接替换 SwiGLU 为 InnerNet **不可行**：

| Seed | SwiGLU (原始) | InnerNet 替换后 best | 替换瞬间 acc |
|------|---------------|---------------------|-------------|
| 42 | **89.3%** | 79.2% | 65.7% |
| 43 | **88.6%** | 80.0% | 51.7% |
| 44 | **88.5%** | 79.4% | 51.7% |

替换瞬间 acc 崩到 52-66%，finetune 后恢复到 ~80% 但远不及原始 89%。和 ivs_d128 不同——Qwen 0.5B 太大，InnerNet 优化跟不上。

### Multiply 初始化 — 多任务对比 ✅

5/5 seeds 完成。Multiply-init 在 MLM 上大幅赢，其他任务持平或小赢：

| 任务 | SwiGLU | MultInit | 差 |
|------|--------|----------|-----|
| d=64 | 92.01±0.34 | 92.25±0.38 | 持平 |
| d=128 | 77.45±0.29 | **77.21±0.34** | -0.24 |
| PTB | 164.19±1.12 | **163.10±2.14** | -1.08 |
| **MLM** | **19.09±0.27** | **15.93±0.21** | **-3.16 (-16.6%)** |

MLM 大幅赢的原因：MLM 任务和简单乘法交互高度匹配。

### 从头训 vs SwiGLU — scratch_init（2.5/5 seeds）

**从头训 InnerNet 一致输 SwiGLU**，无论什么初始化：

| Seed | SwiGLU | Gaussian | Random | Multiply |
|------|--------|----------|--------|----------|
| 42 | **76.60** | 78.08 | 79.25 | 80.17 |
| 43 | **76.93** | 78.27 | 77.99 | 79.24 |
| 44 | **77.84** | 78.42 | 78.06 | ⏳ |

SwiGLU 一直赢 1-3 PPL。Gaussian pretrain 略好于 random/multiply。确认从头训的差距是优化问题。

### 提炼 InnerNet 为简单闭式算子（distill — P1）✅ 定量版

用 `scripts/distill_innernet.py` 把训练好的 InnerNet checkpoint 在 [-5,5]² 上采样，用最小二乘拟合到几种闭式算子，报 R²。注意：下表旗舰 FFN checkpoint 来自显式 SwiGLU 拟合后的 warm-start，不是无信息初始化的发现实验。

| checkpoint | mult `a·b` | **swiglu `silu(a)·b`** | poly3 | 提炼出的算子 |
|-----------|-----------|----------------------|-------|------------|
| **ivs_d128（FFN，SwiGLU warm-start）** | 0.658 | **0.942** | 0.997 | **0.24·silu(a)·b**（缩放版 SwiGLU） |
| CNN CIFAR-10 | 0.674 | **0.908** | 0.974 | 0.35·silu(a)·b |
| fit-to-SwiGLU（自检）| 0.542 | **0.992** | 0.984 | 0.98·silu(a)·b ✓ |

**可支持的结论**：FFN InnerNet 任务训练后的函数仍有 **94% 由单个 SwiGLU 项解释**，而纯乘法 `a·b` 只有 66%。由于它在训练前已被显式拟合成 SwiGLU，这证明的是 SwiGLU 形状在后续优化中被保留并缩放，不是自主发现。自检行确认拟合方法能还原已知的 SwiGLU 表面。

**跨 seed 一致性（warm-start retention）**：5 个 seeds 的后续任务优化彼此独立，但它们共享同一份显式拟合到 SwiGLU 的 InnerNet 初值。逐个提炼得到 SwiGLU 拟合 **R²=0.947±0.010**（范围 0.931–0.956），系数 **0.238±0.005**，纯乘法约 0.66。这个结果排除了单个 seed 的偶然漂移，但不能升级成“独立再发现”；要支持后者，必须分析从 Gaussian/random 等非 SwiGLU 初值训练并保存的多 seed checkpoint。

**Bilinear-host 因果结果（2026-07-30）**：4 个完成的 joint seeds（42/43/45/46）中，host PPL 为 **79.60±0.31**，换入 random/multiply InnerNet 并联合训练 10ep 后为 **73.51±0.37 / 73.72±0.35**。但表面没有变成 SwiGLU：random 的 mult R² **0.9982±0.0006**、multiply 的 mult R² **0.9989±0.0005**，SwiGLU R² 只有约 0.53/0.55。已有 frozen controls 同样恢复 `a·b`（mult R²≈0.999）。因此 SwiGLU-host 的 cross-init 结论只在该 host basin 内成立，不能写成 network-independent attractor。PPL 改善也不能全归因于 activation，因为缺少 Bilinear host 继续训练 10ep 的对照。

| | swiglu R² | silu(a)·b 系数 | mult R² |
|--|:---:|:---:|:---:|
| 42/43/44/45/46 | 0.942/0.931/0.956/0.949/0.955 | 0.238/0.231/0.241/0.243/0.237 | 0.66 全部 |
| **均值±SD** | **0.947±0.010** | **0.238±0.005** | 0.661±0.003 |

脚本 `scripts/distill_crossseed.py`，结果 `results/figures/distill_crossseed_ivs_d128.json`。

提炼出的算子是**缩放版 SwiGLU** `c·silu(a)·b`（旗舰 c≈0.24）。在当前训练来源下，它是 warm-start 后的可压缩表示；若要把它作为新发现部署，须先补足无 SwiGLU 初始化的发现证据。

（旧的多项式提炼 `f≈0.12·a·b+...` 来自更早的范围；现统一用上表的 silu 基拟合，便于量化 warm-start 后的 SwiGLU 形状保留。）

Configs: `scripts/distill_innernet.py`（提炼）, `scripts/innernet_vs_swiglu.py`（发现）。结果 JSON: `results/figures/distill_ivs_d128.json`

---

### MLM 掩码预测（BERT 式）

| 模型 | PPL |
|------|-----|
| SwiGLU | **93.83** |
| GELU | 101.39 |
| InnerNet | 124.82 |

从头训 InnerNet 不行。但 warm-start 后 InnerNet **37-39** 大幅赢 SwiGLU **52-57**（2/5 seeds 完成）。从头训不动是优化问题。

### GPT (d=256) — 大模型反转

| 模型 | Seeds | Best PPL |
|------|-------|----------|
| **GELU** | 4/5 | **72.05, 72.73, 72.82, 72.85 → ~72.6** |
| SwiGLU | 2/5 | 73.82, 75.14 → ~74.5 |
| InnerNet | 3/5 | 75.69, 75.83, 77.20 → **~76.2 输** |

InnerNet 在 GPT d=256 从头训落后 GELU 约 3.6 PPL（+5.0%）。3/5 seeds 后时间到，剩余 seeds 未完成。epoch 20 时 InnerNet 仍在下降，GELU 每 epoch 28min vs InnerNet 3h——同样 epochs InnerNet 优化负担更重。

Configs: `config/experiments/transformer_wikitext_*.yaml`

## 4. LSTM

### WikiText-2 好用

| 变体 | PPL |
|------|-----|
| **Classic** | **101.72** |
| Semantic | 105.30 |
| Standard | 108.39 |

### PTB 不好用

| 变体 | PPL |
|------|-----|
| **Standard** | **183.02** |
| Classic | 186.54 |
| Semantic | 187.52 |

WikiText-2 上好用，PTB 上不好用。WikiText-103 和 CNN/DailyMail 开跑但时间到未完成，搁置。

Configs: `lstm_wikitext_classic.yaml`, `lstm_ptb_classic.yaml`, `lstm_wikitext103_*.yaml`, `lstm_cnndm_*.yaml`

## 4b. RNN PTB（ComplexNeuronRNN）— 负面结果

论文原始的 RNN 实验，3-phase 训练。

| 模型 | Test Loss | Test PPL |
|------|-----------|----------|
| **tanh baseline** | **4.943** | **≈140** |
| 2-arg InnerNet | 5.130 | ≈169 |
| 1-arg InnerNet | 5.186 | ≈179 |

InnerNet 两个变体都**输 tanh 约 20-28%**。PTB 上 InnerNet 一致不好用（RNN 和 LSTM 都输）。

Configs: `config/experiments/rnn_ptb_*.yaml`

## 4c. Sequential MNIST — 加法记忆 + InnerNet 的功能性结果

**实验问题**：InnerNet 是双输入激活函数，能够表示 σ(a)·b 一类门控形式。给 RNN 加一个加法记忆通道后，二元交互是否足以解决需要长记忆的任务？性能可以回答功能性问题；是否真的学出标准 gate，必须另做表面与机制分析。

Sequential MNIST：逐像素读 MNIST（784 步），最后分类。RNN 完全记不住（11%），LSTM/GRU 轻松。

### 结果

| 模型 | 参数 | Best Acc | Seeds | 说明 |
|------|------|----------|-------|------|
| SeqRNN (tanh) | 18K | **11.36% ± 0.02%** | 5/5 ✅ | 随机水平，完全记不住 |
| SeqLSTM | 68K | **77.03% ± 15.66%** | 5/5 ✅ | 不稳定（49%~94%） |
| SeqGRU | 52K | **98.72% ± 0.14%** | 5/5 ✅ | 最强，非常稳定 |
| SeqInnerNetRNN (Plan A) | 19K | **11.04% ± 0.62%** | 5/5 ✅ | **失败**，和 RNN 一样 |
| **SeqGatedRNN (Plan B, 150ep)** | 37K | **成功 seeds ~98.36%**（98.42/98.15/98.51） | 3/5 ✅, 2 NaN | **成功！逼近 GRU** |
| **SeqMinGatedRNN（去 `W_c`）** | 21K | **98.44% ± 0.22%** | 8/9 实际训练成功，1 NaN；另1 infra OOM | 更小、更稳定，逼近 GRU |

LSTM 每个 seed：87.6%, 71.2%, 49.4%, 83.1%, 93.8%——方差极大。
GRU 每个 seed：98.8%, 98.9%, 98.5%, 98.7%, 98.7%——非常稳定。
Plan A 每个 seed：11.4%, 9.8%, 11.4%, 11.4%, 11.4%——一致失败。
Plan B（150ep）成功 seed：98.42%, 98.15%, 98.51%（seed 44/46 NaN 训炸）。
SeqMinGatedRNN 成功 seed 42/44/45/46/47/48/49/50：98.43/98.10/98.50/98.25/98.83/98.25/98.47/98.65%，均值 **98.435%**、popSD **0.220%**；seed 51 在 ep17 NaN。seed 43 的节点没有暴露 GPU，52 秒后 host-memory OOM，属于基础设施失败。

### 分析

**Plan A 失败**：单个 InnerNet 替换 tanh 不够。没有 cell state 提供加法记忆通道，InnerNet 无法学到 forget gate——信息在 784 步的传播中衰减殆尽。

**Plan B 的功能性结果**：给模型加法 cell state 和两个 InnerNet 后，3 个成功 seed 达到约 98%。这证明该组合足以解决长序列记忆任务，但不证明 InnerNet1/2 分别成为标准 input/output gate。

约束版本去掉 `W_c` 后只有 21K 参数，比 LSTM 少约69%、比 GRU 少约59%，仍达到98.44%。这显著加强功能性结论：只保留加法记忆、归一化和两个二元交互网络，已经足以逼近GRU。

**⚠️ 机制性 claim 的定量核查（2026-07-26，内部）**：用和 SwiGLU 同样的跨 seed distillation 方法（`scripts/gate_crossseed.py`，成功 seed 42/43/46，输入经 LN 故取 [-3,3]² 网格）去测"InnerNet1/2 是不是干净的 sigmoid 门"，**结果不支持**：
- inner_net1 门控 R² = **0.29 ± 0.27**（0.004~0.529），inner_net2 = **0.43 ± 0.31**（0.18~0.77）——远不及 SwiGLU 的 0.947±0.010。
- 跨 seed 极不一致：朝向、符号都在变；seed 43 的 inner_net1 门控 R²≈0（几乎无门控结构）却照样 98.15%；有的 seed 线性基线拟合还更好。
- **结论**：功能层面 gate-like 行为成立（存在性证明：加法记忆 + 可学交互足以解 Seq-MNIST），但"每个 InnerNet 各自收敛成一个干净的 input/output gate"这个机制性读法**站不住**——网络靠 cell-state 递归 + 线性映射 + InnerNet 的分布式配合解决，不是单个 InnerNet = 单个门。
- **进一步排除（2026-07-26）**：换 LSTM 精确门形式 σ(a)·tanh(b)、以及在**真实 Seq-MNIST 数据流形**上（跑真实前向抓 InnerNet 实际输入）重测，仍然乱——inner_net1 on-manifold 门控 R²=0.14/0.00/0.00，seed43 的 inner_net1 对任何门形式都 ≈0（full_gate 0.02）却照样 98.15%。**确认是非可辨识性**（gating 被 cell 递归 + W_h/W_x/W_c + LayerNorm + InnerNet 分摊），不是家族/范围选错，也不是 seed 数不够——**加 seed 不会变干净**。
- **约束 cell 已完成（2026-07-28）**：去掉 `W_c`、保留 `ln_c` 后，8/9 个实际训练 seed 成功。但 grid 分析仍不干净：inner_net1 gate R²=**0.245±0.223**，inner_net2=**0.447±0.263**，与旧模型0.29/0.43基本相同，朝向和符号仍跨 seed 变化。`W_c` 不是非可辨识性的主因。
- **对外写作建议**：gate 只按功能性存在证明写（98.44% vs plain RNN 11%，21K参数），并公开8/9实际训练成功、1 NaN和额外1次基础设施OOM。机制分析留档 `results/figures/gate_crossseed_seqmin_constrained.json`。

### 训练稳定性 — 3 种修复全部失败 ❌（2026-06）

Plan B 的痛点是 5 seeds 里 2 个训出 NaN。原假设：NaN 来自加法 cell `c_t = c_prev + update` 在 784 步无界增长。为此试了 3 个修复，各跑满 150ep × 5 seeds：

| 变体 | 改动 | 成功 seeds | NaN seeds |
|------|------|-----------|-----------|
| clip | lr 5e-4 + grad_clip 0.25 | 42/43/45 (~97.6–98.1%) | **44, 46** |
| tanh | cell update 加 tanh 约束 | 42/43/46 (~97.5–98.2%) | **44, 45** |
| all | tanh + lr 5e-4 + clip 0.25 | 42/43/45 (~97.9–98.2%) | **44, 46** |

三种修复全部仍 2/5 NaN。关键观察：**seed 44 在三个变体里全炸，而且第 1 个 epoch 就炸**。如果 NaN 真来自无界增长，tanh-bounded cell 应该能救——但没救。所以推断 **NaN 是初始化敏感（特定 seed 的初始权重一开始就发散），不是数值随步数累积**。下一步该往 lr warmup / 换初始化方案 / 跳过坏 seed 方向试，梯度裁剪和 tanh 约束这条路走不通。成功 seeds 性能稳定在 ~98%，逼近 GRU。

Configs: `config/experiments/seq_mnist_*.yaml`（稳定性变体：`seq_mnist_gated_clip/tanh/all.yaml`）

## 5. 训练阶段消融

| 任务 | 3-phase | End-to-end | 差异 |
|------|---------|-----------|------|
| AE MNIST | 0.0039 | 0.0039 | 没差 |
| ResNet CIFAR-10 | 86.09% | 86.00% | 没差 |
| CNN CIFAR-10 | 78.57% | 77.35% | -1.2% |
| MLP MNIST | 98.0% | 97.95% | 没差 |

pretrain 不是必须的。以后默认 end-to-end。

Configs: `config/experiments/*_e2e.yaml`

## 6. 参数效率

InnerNet w=128 (415K) ≈ ReLU w=256 (921K) → **省 55% 参数**。

## 7. ResNet

全换没用（CIFAR-10 86.10% vs ReLU 86.33%，CIFAR-100+aug 73.00% vs 73.51%）。

**Internal-only（只换 block 内部）有效果**：

| 数据集 | Internal InnerNet | ReLU | 全换 |
|--------|------------------|------|------|
| CIFAR-10 | **87.7%** (2/5 done) | 86.33% | 86.10% |
| CIFAR-100+aug | **74.97%** (5 seeds) | 73.51% | 73.00% |

只换没 skip 保护的位置就有效果（+1.4~1.5%），全换反而没用。CIFAR-10 还有 3 seed OOM 重提交了。

Configs: `config/experiments/resnet_cifar_internal_2arg.yaml`

## 8. PPO RL

| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| Acrobot | **-75.3** | -79.8 | -81.7 |
| LunarLander 30s | **187.6** | 158.8 | -249.7 |

30 seeds 后 InnerNet 赢了 LunarLander。

## 9. 不好用的

| 实验 | 情况 |
|------|------|
| ResNet 全换 | skip connection 下没用（但 internal-only 有效果） |
| MLM InnerNet 从头训 | 比 GELU 差（124.82 vs 101.39），但 warm-start 大幅赢 |
| LSTM PTB | Standard 赢（WikiText-2 上反过来） |
| RNN PTB | InnerNet PPL≈169/179 vs tanh PPL≈140，输 20-28% |
| CNN ×0.25 | 太小了 |
| GPT d=256 从头训 | InnerNet ~76.2 vs GELU ~72.6，输 5% |
| Qwen 0.5B 直接替换 | InnerNet ~80% vs SwiGLU ~89%，大模型替换不可行 |
| 从头训 vs SwiGLU | scratch_init 2.5 seeds 全输（SwiGLU 赢 1-3 PPL） |

## 当前运行（2026-08-27）

| 实验 | 进度 |
|------|------|
| P1 causal matrix v2 | 10 个可复用 host checkpoint jobs + 20 个依赖 probe jobs 已提交；排除不兼容 RTX Pro 6000，输出到 `/user_data/yizhouc3/xor_causal_v2/` |
| Critical CNN seeds | SVHN 2-arg seeds 44/45（664241/664242）与 FashionMNIST 1-arg seeds 42–45（664243–664246）已提交；从已有 pretrain checkpoint 续跑，完成通知 664247 |

GPT v4 (3/5 seeds)、free_init_v2 (Wiki 3/3, MLM 2/3)、scratch_init (2.5/5) 时间到未完成，数据已够用。

## 统计与可追溯性（2026-08-27）

- `scripts/build_result_manifest.py` 已扫描 474 个实验目录，得到 1132 行结构化指标：399 个实验 raw-verified，75 个 incomplete，0 个 completed-no-result；deploy、Seq-MNIST 和 warm-start 脚本自产的 `results.p/results.json` 也已纳入，包括 `ivs_d128_v2` 的逐 epoch 分支。
- `scripts/summarize_result_manifest.py` 按科学配置、condition 与 run status 自动去重并汇总 mean、sample SD、population SD、raw seeds/values；当前 209 个指标组全部可汇报，0 个同配置 seed 数值冲突。
- NaN run 不再与成功 run 混算：SeqMinGatedRNN 成功组 8 seeds 自动复算为 **98.435%**（sample SD 0.236%，population SD 0.220%），另保留 1 个 NaN seed 的独立记录。
- 自动发现 1 个同名配置碰撞：`mlp_mnist_relu` 同时指 64-width 未参数匹配版（seed1234=85.63%）和 112-width 参数匹配版（seed1234=91.27%，其余 seeds 同组）。两者现在按配置签名分开，不再混算。
- 15 项核心比较已由 `config/audit/core_comparisons.yaml` 注册并自动复算。Transformer d=64/128/192/256 的 InnerNet-vs-GELU paired-t p 分别为 **0.00022 / 0.05095 / 0.361 / 0.173**；d=128 SwiGLU-vs-InnerNet p=**0.00917**。CNN、AE、Big-MLP headline 的 paired-t 均 <0.014。
- 小样本解释：n=5 时双侧 Wilcoxon 的离散最小值通常是 0.0625，因此不单看“p<0.05”；正式报告同时给 raw seeds、bootstrap CI、paired-t/Wilcoxon 和 Cohen's dz。
- `scripts/check_document_claims.py` 已把 RESULTS_CN/EN 的40个已注册 headline table cells 与 canonical summary 自动对照，当前 **40/40 match**。
- 产物：`results/audit/grouped_metric_summary.csv`、`metric_conflicts.csv`、`experiment_variant_collisions.csv`、`core_comparisons.csv`、`document_consistency.csv`、`deploy_analysis.json`；24 个审计/统计单元测试全部通过。

## 总结

有效的：**CNN (+0.4~4.6%)，AE (-43%)，TF FFN (-0.8~3.3% d=64~256)，LSTM WikiText-2 (-6.2%)，ResNet internal-only (+1.5%)，参数省 55%，PPO LunarLander (+18%)，Warm-start 10/11 赢或持平，Multiply-init MLM -16.6%**。

没用的：ResNet 全换（持平），MLM 从头训（差），LSTM/RNN PTB（差），GPT d=256 从头训（输 5%），Qwen 0.5B 直接替换（-9%），从头训一致输 SwiGLU。

关键发现：
- InnerNet 放在没 skip 保护的位置有效果
- **容量上限 ≥ SwiGLU**（warm-start 10/11 赢或持平，ivs_d128: 77.38=77.38 完全持平）
- 从头训打不过是**优化问题**，不是容量问题（scratch_init 证实）
- **初始化不影响终点**：4 种初始化都收敛到同一水平（free_init 证实）
- Scaling: d=64 赢 3.3% → d=256 赢 1.7% → GPT d=256 输 5%。模型越大优化负担越重
- 大模型直接替换不可行（Qwen -9%），但 warm-start + 继续训可以（ivs_d128 追平）
- InnerNet 适合：(1) 小模型 / on-device (2) warm-start finetune (3) 架构搜索工具
- 不适合：大模型从头训、大模型直接替换
- **Sequential MNIST 功能性结果**：21K约束版在8/9实际训练runs达到98.44±0.22%，接近GRU 98.72%；Plan A失败。表面分析仍不支持单个InnerNet=标准gate
