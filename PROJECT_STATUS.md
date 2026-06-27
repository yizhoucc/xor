# 项目状态 — 2026-06-27

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

## 论文定位（已定 — 2026-06-27）

### 相对原论文（Yoon/Kim/Orhan/Pitkow, IEEE Access 2022）的新意

**原论文范围/结论**：只测 MLP+CNN，只在 MNIST/CIFAR-10 上，结论温和——参数匹配下比 ReLU 学得快一点、略好、更鲁棒 + 生物学动机。本质是一篇"受生物启发的更好激活函数"。

**我们超出的部分**：
- **架构广度**（原论文完全没碰）：Transformer FFN、LSTM、RNN(Seq-MNIST)、AE、ResNet/VGG/WRN、ViT、MLP-Mixer、PPO、Masked LM、GPT —— 10+ 架构。
- **概念新发现**：① SwiGLU 再发现（FFN 里自主学出 SwiGLU 式乘法门控）；② Gate 再发现（Seq-MNIST 只给加法记忆通道就长出 LSTM/GRU gate）；③ 位置决定效果（skip connection 抹掉优势）；④ 优化壁垒 ≠ 容量上限（warm-start 10/11 赢，ivs_d128 持平）；⑤ scaling 反转（越大越没优势）；⑥ distillation（学到的函数提炼成闭式算子）。

### 定下来的 Story：**可微的架构发现工具**（NOT "a better activation function"）

⚠️ **绝不卖"更强的激活函数"**。理由：我们自己的数据显示 InnerNet 从头训在现代/大模型上普遍打不过 SwiGLU，直接替换又慢又掉点（Qwen -9%）。若按"better activation"写，reviewer 会说"和 IEEE Access 2022 一样、marginal、不 scale、推理慢、没价值"。

✅ **卖"architecture discovery"**：用 InnerNet 替换激活 → 训练 → 可视化学到的 2D 函数 → 提炼成简单算子 → 用正常快算子部署。它能**独立重新发现 SwiGLU 和 gate** 这两个公认 SOTA 基元，就是它作为发现工具有效的证据。这个 framing 下"从头训打不过 SwiGLU / 推理慢"**不是弱点而是论据**——发现工具本来就不用来部署，提炼出的快算子才用来部署。reviewer 买这个账：你用它发现了好架构，再换成近似的快算子，效率高。

标题方向："Two-argument activations as differentiable architecture search" / "Learnable activations rediscover SwiGLU and gating"。

（辅助角度，备用，不作主线）Efficient finetuning（替换 +97 参数，和 LoRA 互补）；Understanding activation functions（优化比表达能力更关键 / 位置决定效果 / 小模型受益更大）。

## 已修复的问题

- **U20 param sharing bug**：之前 Transformer/ResNet/WRN 的 InnerNet 每层各一个没共享。已修复，重跑。修复后结果和之前差不多（d=64: 112.66→112.83），说明影响不大，但 sharing 是论文基本设计。CNN/MLP/AE/VGG/LSTM/PPO 不受影响。

## 集群状态（2026-06-27）

### 运行中

| Job ID | 实验 | 配置 | 状态 |
|--------|------|------|------|
| 547111 | **P1 部署段 — FFN deploy** | `deploy_distilled.py`，4 op (gelu/swiglu/innernet/distilled-poly3)，WikiText-2 d=128 d_ff=512 4层 20ep×5seeds，全部同一 GPU 测吞吐 | 提交于 6-27，PD |

目标：证明 distilled 固定算子（poly3，从 ivs_d128 InnerNet 提炼）≈ InnerNet 质量但 ≈SwiGLU 速度。输出 `exp/deploy_ffn_d128/results.json`（PPL + tok/s）。

### 已完成

上一批 3 个 Plan B 稳定性实验（510929/510930/510931）已于 6-06~6-12 全部跑完 5 seeds（见下方稳定性表，3 修复全失败）。

### Plan B 稳定性实验结果 — 3 种 NaN 修复全部失败 ❌

| Job ID | 变体 | 改动 | 成功 seeds | NaN seeds | 结果 |
|--------|------|------|-----------|-----------|------|
| 510929 | clip | lr 5e-4 + grad_clip 0.25 | 42/43/45 (~97.6–98.1%) | **44, 46** | 仍 2/5 NaN |
| 510930 | tanh | cell update 加 tanh 约束 | 42/43/46 (~97.5–98.2%) | **44, 45** | 仍 2/5 NaN |
| 510931 | all | tanh + lr 5e-4 + clip 0.25 | 42/43/45 (~97.9–98.2%) | **44, 46** | 仍 2/5 NaN |

**关键发现**：三种修复都没解决 NaN，全部稳定 2/5 seeds 炸。⭐ **seed 44 在三个变体里全炸** → NaN 不是来自"加法 cell `c_t=c_prev+update` 在 784 步无界增长"假设（否则 tanh-bounded 应该能救），而更像是**初始化 / 特定 seed 敏感**问题，且第 1 个 epoch 就炸。下一步方向：lr warmup / 重新初始化（换 init scheme 或 init scale）/ 跳过坏 seed。成功 seeds 性能仍 ~98%，逼近 GRU。

### Sequential MNIST 结果

| 模型 | 参数 | Best Acc | Seeds | 说明 |
|------|------|----------|-------|------|
| SeqRNN (tanh) | 18K | **11.36% ± 0.02%** | 5/5 ✅ | 随机水平 |
| SeqLSTM | 68K | **77.03% ± 15.66%** | 5/5 ✅ | 不稳定（49%~94%） |
| SeqGRU | 52K | **98.72% ± 0.14%** | 5/5 ✅ | 最强最稳 |
| SeqInnerNetRNN (Plan A) | 19K | **11.04% ± 0.62%** | 5/5 ✅ | 失败，和 RNN 一样 |
| Plan B (50ep) | 37K | 成功 seeds ~97.75% | 3/5 ✅, 2 NaN | 接近 GRU，参数少 46% |
| **Plan B (150ep)** | 37K | **成功 seeds ~98.36%**（98.42/98.15/98.51） | 3/5 ✅, 2 NaN | **逼近 GRU 98.72%**，但仍 2 NaN |
| Plan B + clip | 37K | 成功 seeds ~97.6–98.1% | 3/5 ✅, 2 NaN | NaN 未解决 |
| Plan B + tanh-bounded | 37K | 成功 seeds ~97.5–98.2% | 3/5 ✅, 2 NaN | NaN 未解决 |
| Plan B + all | 37K | 成功 seeds ~97.9–98.2% | 3/5 ✅, 2 NaN | NaN 未解决 |

**关键发现**：Plan A（单 InnerNet 替换 tanh）完全失败。Plan B（2 InnerNet + cell state）成功——给一个加法记忆通道，InnerNet 就能自主发现 gate 机制。长训（150ep）成功 seeds 提升到 98.36%，逼近 GRU。**训练不稳定（5 seeds 里 2 个 NaN）目前未解决：3 种修复（梯度裁剪 / tanh 约束 / 组合）全部无效，且同一 seed 反复炸 → 推断是初始化敏感而非数值增长。**

### 其他最近完成

| 实验 | 结果 |
|------|------|
| **RNN PTB 2-arg** ✅ | Test PPL≈169，输 tanh baseline (PPL≈140) |
| **RNN PTB 1-arg** ✅ | Test PPL≈179，输 tanh baseline |
| **ivs_d128** ✅ | SwiGLU 77.38±0.54 vs Frozen InnerNet **77.38±0.51** — 完全持平，容量验证 |
| **mult_init** ✅ | d=64 持平, d=128 -0.24, PTB -1.08, **MLM -3.16**（5 seeds） |
| **GPT v4** ⏳ | 3/5 seeds 完成（77.20, 75.69, 75.83 → 均值 ~76.2），时间到 |
| **free_init_v2** ⏳ | Wiki 3/3 seeds ✅（4 种初始化收敛到 ~71.7-72.6 vs SwiGLU ~77.3），MLM 2/3 seeds |
| **scratch_init** ⏳ | 2.5/5 seeds。SwiGLU 一致赢：76.6/76.9 > Random 78.1/78.0 > Gaussian 78.1/78.3 > Multiply 80.2/79.2 |

---

## TODO — 按优先级

### 🔴🔴 论文主线缺口（"架构发现工具" framing 必须补的 3 件事 — 2026-06-27 定）

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| **P1** | **合上"发现→提炼→部署"闭环** | ⏳ 进行中 | **提炼段 ✅ 已做实**：`scripts/distill_innernet.py` 把 InnerNet checkpoint 拟合到闭式算子并报 R²。FFN 旗舰 = **0.24·silu(a)·b, R²=0.942**（纯 a·b 只 0.66）→ 定量证明 SwiGLU 再发现；CNN R²=0.91；自检（fit-to-SwiGLU）还原 0.98·silu(a)·b R²=0.992。**剩部署段 ❌**：把缩放版 SwiGLU `c·silu(a)·b` 当固定快算子塞回 fresh Transformer，证明 (a) 质量≈InnerNet/SwiGLU (b) 比 InnerNet 快（无 inner-MLP）。需 cluster GPU |
| **P2** | **稳定 gate discovery（Plan B NaN）** | 🔜 开始 | 当前 2/5 NaN、3 个修复全失败（初始化敏感，见 U42）。"自主发现 gate"作为论文证据需要稳定，否则被质疑 cherry-pick 3/5 seeds。下一步：lr warmup / 换 init scheme / init scale / 跳坏 seed |
| **P3** | **统计严谨性** | 🔜 开始 | p-value 显著性检验（见 m5）、scaling-law 图（U30/U31）、训练曲线图（M1/U31）。把"赢"从单点数字升级到带统计检验 + 趋势图 |

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
| U41 | Sequential MNIST (InnerNet 发现 gate) | ⏳ **Plan B 成功!** | Plan A 11.04%(失败) / **Plan B 150ep 成功 seeds ~98.36%**(逼近 GRU 98.72%)。给 cell state 脚手架 InnerNet 自主学出 gate |
| U42 | Plan B 训练稳定性 | ❌ 3 修复全失败 | 3 实验已跑完（clip / tanh-bounded / all），全部仍 2/5 NaN。seed 44 在三个变体里全炸、第 1 epoch 就炸 → **NaN 来自初始化敏感而非加法 cell 无界增长**。下一步：lr warmup / 换 init / 跳坏 seed |
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
