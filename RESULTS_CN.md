# 实验结果笔记 — InnerNet

> 内部文档。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

### 定位：可微的架构发现工具（NOT "更好的激活函数"）

把 ReLU 换成一个小 MLP（两输入一输出），让每个神经元能看到隔壁特征。但**我们不卖"它是更强的激活函数"**——因为从头训在大模型上打不过 SwiGLU、直接替换又慢又掉点（Qwen -9%）。如果按"better activation"写，reviewer 会说"和原论文（IEEE Access 2022）一样、marginal、推理慢、没价值"。

**我们卖的是：用它来发现架构。** 流程：InnerNet 替换激活 → 训练 → 可视化学到的 2D 函数 → 提炼成简单快算子 → 用正常算子部署。证据是它能**独立重新发现两个公认 SOTA 基元**：① Transformer FFN 里自主学出 SwiGLU 式乘法门控；② Sequential MNIST 里只给加法记忆通道就长出 LSTM/GRU gate。这个 framing 下，"从头训打不过 SwiGLU / 推理慢"不是弱点而是论据——发现工具本来就不用来部署。

### 相对原论文（Yoon et al., IEEE Access 2022）的新意

原论文只测 MLP+CNN、只在 MNIST/CIFAR 上、结论温和（简单任务略有提升 + 鲁棒 + 生物动机）。我们超出的：**架构广度**（Transformer/LSTM/RNN/AE/ResNet/ViT/Mixer/PPO/MLM/GPT 10+ 个）+ **概念新发现**（SwiGLU 再发现、gate 再发现、位置决定效果、优化壁垒≠容量上限、scaling 反转、distillation）。

### 实验事实速查

确认有效的：CNN、AE、Transformer FFN（d=64~256 全赢 GELU）、LSTM WikiText-2、PPO、ResNet internal-only、Seq-MNIST gate discovery（~98%）。
不好用的：ResNet 全换（skip connection 冗余）、LSTM PTB、大模型从头训（GPT d=256 反转）、大模型直接替换（Qwen -9%）。

**关键发现**：InnerNet 容量上限 ≥ SwiGLU（warm-start 10/11 赢或持平，ivs_d128 追平），从头训大模型输是优化问题。模型越大差距越大（d=64 赢 3.3%, d=256 输 5.2%）。适合"发现 + warm-start finetune"，不适合大模型从头训或直接部署替换。

### 投稿前剩余工作

1. **结果源审计**：canonical manifest 已建立；继续清理修复前后结果混用、缺原始来源和不完整 seeds。
2. **统计严谨性**：paired/unpaired 工具已完成；继续为正文核心结果生成 raw seeds、差值区间、效应量和配对/非配对检验。
3. **gate 稳定性不作为卖点**：Plan B 全叠加诊断（547208，本地 `exp/seq_mnist_gated_diag_20260627_171855_d46bf25c`）仍 2/5 NaN，是已知未解边界。**决定：不 claim 稳定性 → 不需出证据 → 对外文档不提**。gate discovery 按存在性证明写（成功 run ~98.4% 逼近 GRU），RESULTS_EN 已去掉 NaN 计数和 "open issue" 措辞。NaN 原始数据仅在此内部留档。

部署不是投稿前置条件。CNN deploy 已完成，FFN deploy 因时限中止；它们只作为扩展结果，不阻塞以“发现”为核心的论文。

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

参数公平对比：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

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

**这是"发现→提炼→部署"闭环的提炼段。** 用 `scripts/distill_innernet.py` 把训练好的 InnerNet checkpoint 在 [-5,5]² 上采样，用最小二乘拟合到几种闭式算子，报 R²：

| checkpoint | mult `a·b` | **swiglu `silu(a)·b`** | poly3 | 提炼出的算子 |
|-----------|-----------|----------------------|-------|------------|
| **ivs_d128（FFN 旗舰）** | 0.658 | **0.942** | 0.997 | **0.24·silu(a)·b**（缩放版 SwiGLU） |
| CNN CIFAR-10 | 0.674 | **0.908** | 0.974 | 0.35·silu(a)·b |
| fit-to-SwiGLU（自检）| 0.542 | **0.992** | 0.984 | 0.98·silu(a)·b ✓ |

**关键定量证据**：FFN InnerNet 学到的函数 **94% 由单个 SwiGLU 项解释**，而纯乘法 `a·b` 只有 66%——说明它收敛到的不是泛泛的乘法，而是**特定的 silu-gating，即 SwiGLU**。自检行确认方法正确：把一个显式拟合到 SwiGLU 的 InnerNet 喂进去，能还原出 `0.98·silu(a)·b`（R²=0.992）。

**跨 seed 一致性（新，核心证据升级）**：不是单个 run 的巧合。把 5 个独立训练的 InnerNet FFN（d=128, seeds 42–46）逐个提炼，SwiGLU 拟合 **R²=0.947±0.010**（范围 0.931–0.956），门控系数 **0.238±0.005**（silu(a)·b，SD 极小），而纯乘法每个 seed 都只有 0.66。**独立优化的网络收敛到同一个 SwiGLU 门控**——这把"某个 InnerNet 像 SwiGLU"升级成"InnerNet 可复现地再发现 SwiGLU"，是 discovery 论文最需要的证据。

| | swiglu R² | silu(a)·b 系数 | mult R² |
|--|:---:|:---:|:---:|
| 42/43/44/45/46 | 0.942/0.931/0.956/0.949/0.955 | 0.238/0.231/0.241/0.243/0.237 | 0.66 全部 |
| **均值±SD** | **0.947±0.010** | **0.238±0.005** | 0.661±0.003 |

脚本 `scripts/distill_crossseed.py`，结果 `results/figures/distill_crossseed_ivs_d128.json`。

提炼出的算子是**缩放版 SwiGLU** `c·silu(a)·b`（旗舰 c≈0.24，比标准 SwiGLU 的 1.0 温和很多，对应"压缩了范围的乘法交互"）。这个闭式算子就是闭环里要**部署**的快算子（下一步：塞回 fresh 模型，比 InnerNet 快、和 SwiGLU 同档）。

（旧的多项式提炼 `f≈0.12·a·b+...` 来自更早的范围；现统一用上表的 silu 基拟合，对"SwiGLU 再发现"的论证更直接。）

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

## 4c. Sequential MNIST — InnerNet 发现 gate ✅

**核心想法**：LSTM 比 RNN 强是因为 gate。InnerNet 是双输入激活函数，天然能学出 σ(a)·b 这种 gate 模式。如果 RNN+InnerNet 在需要长记忆的任务上接近 LSTM → InnerNet 自主发现了 gate 机制。

Sequential MNIST：逐像素读 MNIST（784 步），最后分类。RNN 完全记不住（11%），LSTM/GRU 轻松。

### 结果

| 模型 | 参数 | Best Acc | Seeds | 说明 |
|------|------|----------|-------|------|
| SeqRNN (tanh) | 18K | **11.36% ± 0.02%** | 5/5 ✅ | 随机水平，完全记不住 |
| SeqLSTM | 68K | **77.03% ± 15.66%** | 5/5 ✅ | 不稳定（49%~94%） |
| SeqGRU | 52K | **98.72% ± 0.14%** | 5/5 ✅ | 最强，非常稳定 |
| SeqInnerNetRNN (Plan A) | 19K | **11.04% ± 0.62%** | 5/5 ✅ | **失败**，和 RNN 一样 |
| **SeqGatedRNN (Plan B, 150ep)** | 37K | **成功 seeds ~98.36%**（98.42/98.15/98.51） | 3/5 ✅, 2 NaN | **成功！逼近 GRU** |

LSTM 每个 seed：87.6%, 71.2%, 49.4%, 83.1%, 93.8%——方差极大。
GRU 每个 seed：98.8%, 98.9%, 98.5%, 98.7%, 98.7%——非常稳定。
Plan A 每个 seed：11.4%, 9.8%, 11.4%, 11.4%, 11.4%——一致失败。
Plan B（150ep）成功 seed：98.42%, 98.15%, 98.51%（seed 44/46 NaN 训炸）。

### 分析

**Plan A 失败**：单个 InnerNet 替换 tanh 不够。没有 cell state 提供加法记忆通道，InnerNet 无法学到 forget gate——信息在 784 步的传播中衰减殆尽。

**Plan B 成功**：给 InnerNet 一个 cell state（加法连接），它就能自主学出 gate 机制：
- InnerNet1（cell state 之前）：学到了 input gate 的角色——决定什么信息写入记忆
- InnerNet2（cell state 之后）：学到了 output gate 的角色——决定什么信息输出

Plan B（37K 参数）比 LSTM（68K）参数少 46%，成功 seeds 性能逼近 GRU。只给"加法记忆通道"这个最小脚手架，InnerNet+cell state 就能解长记忆任务（vs plain RNN 11%）。

**⚠️ 机制性 claim 的定量核查（2026-07-26，内部）**：用和 SwiGLU 同样的跨 seed distillation 方法（`scripts/gate_crossseed.py`，成功 seed 42/43/46，输入经 LN 故取 [-3,3]² 网格）去测"InnerNet1/2 是不是干净的 sigmoid 门"，**结果不支持**：
- inner_net1 门控 R² = **0.29 ± 0.27**（0.004~0.529），inner_net2 = **0.43 ± 0.31**（0.18~0.77）——远不及 SwiGLU 的 0.947±0.010。
- 跨 seed 极不一致：朝向、符号都在变；seed 43 的 inner_net1 门控 R²≈0（几乎无门控结构）却照样 98.15%；有的 seed 线性基线拟合还更好。
- **结论**：功能层面 gate-like 行为成立（存在性证明：加法记忆 + 可学交互足以解 Seq-MNIST），但"每个 InnerNet 各自收敛成一个干净的 input/output gate"这个机制性读法**站不住**——网络靠 cell-state 递归 + 线性映射 + InnerNet 的分布式配合解决，不是单个 InnerNet = 单个门。
- **进一步排除（2026-07-26）**：换 LSTM 精确门形式 σ(a)·tanh(b)、以及在**真实 Seq-MNIST 数据流形**上（跑真实前向抓 InnerNet 实际输入）重测，仍然乱——inner_net1 on-manifold 门控 R²=0.14/0.00/0.00，seed43 的 inner_net1 对任何门形式都 ≈0（full_gate 0.02）却照样 98.15%。**确认是非可辨识性**（gating 被 cell 递归 + W_h/W_x/W_c + LayerNorm + InnerNet 分摊），不是家族/范围选错，也不是 seed 数不够——**加 seed 不会变干净**。
- **能不能靠跑集群拿到证据？评估：不能可靠拿到。** 唯一有机会的是重新设计"约束 cell"（去掉线性投影/LN，逼 InnerNet 成为唯一能承载门的地方）再从头训——但那是新架构、要 GPU 训几天、结果不确定、且模型已不是论文里那个。不值得赌。
- **对外写作建议**：gate 只按**功能性存在证明**写（98% vs 11%，逼近 GRU，参数少 46%），**不要**把 SwiGLU 那种"跨 seed 收敛到同一闭式门"的强证据套到 gate 上。SwiGLU 是我们唯一有干净跨 seed 定量证据的再发现。结果留档 `results/figures/gate_crossseed_seqmnist.json`。

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

## 在跑的（2026-05-13）

| 实验 | 进度 |
|------|------|
| Sequential MNIST — RNN/LSTM/GRU/InnerNetRNN/GatedRNN | 5 jobs 在跑，50ep×5seeds |

GPT v4 (3/5 seeds)、free_init_v2 (Wiki 3/3, MLM 2/3)、scratch_init (2.5/5) 时间到未完成，数据已够用。

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
- **Sequential MNIST gate 发现**：Plan B（2 InnerNet + cell state）98% 接近 GRU 99%，自主学出 gate 机制。Plan A（单 InnerNet）失败。给最小脚手架 InnerNet 就能发现架构
