# 项目状态 — 2026-08-27

## 执行环境规则

- **本项目本地不运行测试或实验。** 本地仅用于代码/文档编辑、静态查看和结果整理；所有测试、验证与训练必须通过 Mind Cluster 的 Slurm 调度执行，严禁在 cluster 登录节点直接运行。

## Codex 发表前审计（2026-07-26）

- 工作分支：`codex/publication-audit`；开始前快照：`e4d4e9a`（已推送 `main`）。
- 当前范围：先核对原论文公平性协议，再做结果源审计、canonical manifest 和配对统计；不把部署闭环、长实验或新增任务作为投稿前置条件。
- 原论文协议已核对：原论文按总参数量调整 baseline 宽度；本项目 MLP/CNN 参数匹配思路一致。Transformer 现有比较是 same-width 而非 total-parameter-matched，这只限制性能增益措辞，不影响 SwiGLU-like interaction 的发现结论。
- 原论文同样没有部署闭环，只通过函数拟合和结构统计汇报发现；当前不新增部署实验。
- 配对统计工具已完成：新增 paired t-test、Wilcoxon、Cohen's dz、bootstrap CI 和非有限 pair 报告，并保留独立样本模式。5-seed same-width Transformer 示例中，`GELU-InnerNet=+1.558 PPL`（paired-t p=0.05095），`SwiGLU-InnerNet=-2.280 PPL`（p=0.00917）；正式结果需同时报告 raw seeds 和非参数检验。
- 本地结果 inventory/manifest 已生成：477 个实验、1602 行指标；**402 raw-verified、75 incomplete、0 completed-no-result**。已纳入统一 runner 之外的 deploy、Seq-MNIST、warm-start 与 PPO 结果；PPO 现在显式记录每个 seed 的最后20个评估点均值。自动分组得到 238 个可汇报指标组、0 个同配置 seed 冲突；另检测到 1 个实验命名碰撞（`mlp_mnist_relu` 同名但分别为 64-width 未匹配版与 112-width 参数匹配版）。
- ✅ **SSH 取证已完成（Claude，2026-07-26）**：
  - CNN CIFAR-10 2-arg seed 42 已从集群拉回，`test_accuracy=0.7969`（**79.69%**，非之前反推的 79.68%），日志确认，现为 **raw-verified**。五个 seed 齐全，mean=78.57%（popSD 0.74 / sampleSD 0.82）。
  - 4 个旧 job 终态：547111 FFN deploy **TIMEOUT**（未完成，innernet 仅 4 seed、distilled 空）；547112 CNN deploy COMPLETED；547208 Seq-MNIST 诊断 COMPLETED；547209 bark COMPLETED。队列现已空。
  - PTB non-shared 冲突已用原始 `exp/warmstart_nonshared/results.p` 解决：InnerNet 162.49 vs 自身 SwiGLU baseline 164.59（5/5 赢，-2.11，paired-t p=0.037）。旧表的 162.64/-1.95 是混用了 shared 实验的 SwiGLU(162.22) baseline，已更正。
  - 6/27 后集群新结果仅 3 个（两个 deploy json + 一个 tanh test_results，均非核心分类）。
- 详细计划、证据和逐步改动记录：`docs/CODEX_PUBLICATION_AUDIT.md`。
- ⚠️ **2026-07-30 发现证据更新**：SwiGLU-host cross-init 显示四种 InnerNet 初值均得到 SwiGLU R²≥0.98，但 Bilinear-host 因果实验给出相反边界：4 个完成的 joint seeds 无论 random/multiply init 均保持纯乘法（mult R²≈0.998–0.999，SwiGLU R²≈0.51–0.56），同时 PPL 从 host 79.60±0.31 降至 73.51±0.37 / 73.72±0.35。结论是 learned surface **依赖 host/optimization basin**，现有证据不支持 network-independent SwiGLU attractor。cross-init 多 seed补跑因 GPU 架构不兼容失败，尚待在兼容节点完成。

## 核心结论

- InnerNet 在没 skip 保护的位置有效（CNN, AE, TF FFN, ResNet internal）
- Warm-start 后 InnerNet 在 **10/11** 任务上赢或持平 SwiGLU，从头训打不过是优化问题
- **容量上限 ≥ SwiGLU**（ivs_d128 最终结果：SwiGLU 77.38±0.54 vs Frozen InnerNet **77.38±0.51**，完全持平）
- 模型越大 InnerNet 优势越小：d=64 赢 3.3%, d=128 赢 1.6%, d=256 从头训反转为输（GPT d=256: ~76.2 vs GELU ~72.6）
- **从头训一致输 SwiGLU**：scratch_init 2 seeds 确认 SwiGLU > Gaussian > Random > Multiply
- **大模型 finetune 替换不可行**：Qwen 2.5-0.5B 替换后 acc 从 89% 掉到 80%
- 适合场景：小模型 / on-device, warm-start finetune（非直接替换）, 架构搜索
- Non-shared（每层不同 InnerNet）比 shared 效果更好（PTB -2.11 vs -1.04，配对 5/5 赢），参数差可以忽略
- **Multiply-init MLM 大幅赢**：MultInit 15.93±0.21 vs SwiGLU 19.09±0.27（-16.6%，5 seeds）
- **初始化不影响收敛终点**：4 种初始化都收敛到 ~71.7-72.6（Wiki d=128），都大幅赢 SwiGLU ~77.3
- **约束 SeqMinGatedRNN**：去掉 `W_c` 后，8/9 个实际训练 seed 成功，最佳准确率 **98.44±0.22%**（popSD，范围 98.10–98.83%）；1 个 NaN，另有 1 个未进入训练的 infrastructure OOM。参数降至 21K，但标准 gate 表面仍不可辨识。

## 论文定位（已定 — 2026-06-27）

### 相对原论文（Yoon/Kim/Orhan/Pitkow, IEEE Access 2022）的新意

**原论文范围/结论**：只测 MLP+CNN，只在 MNIST/CIFAR-10 上，结论温和——参数匹配下比 ReLU 学得快一点、略好、更鲁棒 + 生物学动机。本质是一篇"受生物启发的更好激活函数"。

**我们超出的部分**：
- **架构广度**（原论文完全没碰）：Transformer FFN、LSTM、RNN(Seq-MNIST)、AE、ResNet/VGG/WRN、ViT、MLP-Mixer、PPO、Masked LM、GPT —— 10+ 架构。
- **新增结论**：① 显式 SwiGLU warm-start 在任务优化后保持为缩放 SwiGLU；② 加法记忆 + 二元交互在 Seq-MNIST 成功 runs 上达到约 98%，但标准 gate 表面不可辨识；③ 位置决定效果；④ 优化壁垒 ≠ 容量上限；⑤ scaling 反转；⑥ learned-surface distillation。当前证据不支持“从无信息初始化独立再发现 SwiGLU/gate”。

### 当前证据支持的 Story：**可微的架构探针**（NOT "a better activation function"）

⚠️ **绝不卖"更强的激活函数"**。理由：我们自己的数据显示 InnerNet 从头训在现代/大模型上普遍打不过 SwiGLU，直接替换又慢又掉点（Qwen -9%）。若按"better activation"写，reviewer 会说"和 IEEE Access 2022 一样、marginal、不 scale、推理慢、没价值"。

✅ **现有证据支持“architecture probe / hypothesis generator”**：用 InnerNet 替换激活，训练后分析二维表面和功能边界。要升级成“architecture discovery”，至少需要多 seed、非 SwiGLU 初值的 checkpoint 显示同一闭式算子，并排除 warm-start 先验。部署不是投稿前置，但也不能用 warm-start retention 代替 discovery 证据。

标题方向暂改为："Two-argument activations as probes of learned feature interactions"；补足无信息初始化证据后再考虑 "rediscover"。

（辅助角度，备用，不作主线）Efficient finetuning（替换 +97 参数，和 LoRA 互补）；Understanding activation functions（优化比表达能力更关键 / 位置决定效果 / 小模型受益更大）。

## 已修复的问题

- **U20 param sharing bug**：之前 Transformer/ResNet/WRN 的 InnerNet 每层各一个没共享。已修复，重跑。修复后结果和之前差不多（d=64: 112.66→112.83），说明影响不大，但 sharing 是论文基本设计。CNN/MLP/AE/VGG/LSTM/PPO 不受影响。

## 集群状态（2026-08-27）

### 当前运行：P1 causal matrix v2（30 jobs + 依赖）

旧矩阵因单个 job 重复训练 host、串行执行多个 init，出现 24h timeout；另有 8 个 job 落到 PyTorch 不支持的 RTX Pro 6000 节点。现已改成**共享 host checkpoint + probe 拆分 + 可续跑 + 结构化 `results.json`**，并用独立 worktree `/home/yizhouc3/xor-codex-audit` 保护 cluster 上有未提交改动的 `~/xor`。

| Job IDs | 阶段 | 数量 | 状态/约束 | 输出 |
|---------|------|------|-----------|------|
| **664213/231/219/189/192** | Bilinear host seeds 42–46 | 5 | 4 COMPLETED / 1 RUNNING | `/user_data/yizhouc3/xor_causal_v2/hosts/bilinear_seed*.pth` |
| **664224/198/201/204/207** | SwiGLU host seeds 42–46 | 5 | 4 COMPLETED / 1 RUNNING | `/user_data/yizhouc3/xor_causal_v2/hosts/swiglu_seed*.pth` |
| **664214/238/232/233/220/221/190/191/193/194** | Bilinear joint/frozen × random/multiply × 5 seeds | 10 | 8 RUNNING / 2 PENDING dependency | `/user_data/yizhouc3/xor_causal_v2/probes/bilinear_*` |
| **664225/226/236/200/202/203/205/206/208/209** | SwiGLU joint × 4 init × 5 seeds（每 job 2 init） | 10 | 8 RUNNING / 2 PENDING dependency | `/user_data/yizhouc3/xor_causal_v2/probes/swiglu_*` |

提交清单：`/user_data/yizhouc3/xor_causal_v2/submitted_20260827.tsv`。代码提交：`e801536`（host cache/resume/JSON）+ `963ea13`（隔离 worktree 支持）+ `9f82434`（每-job HF cache）。预期在调度后约 12–24h 完成；实际 wall time 取决于兼容 GPU 排队。

Bark：启动/异常通知已发送；完成通知 job **664237** 依赖当前矩阵全部有效 jobs，结束后自动提醒拉取和分析结果。

启动异常与处置：首批 host 664180/183/186 在 L40S 上因共享 HuggingFace cache 的 NFS `Stale file handle` 失败；已在 `9f82434` 改为每-job `/tmp` cache。664195/664216、probe 664199 和 frozen probe 664215 在 Titan RTX 节点 `mind-1-24` 出现 CUDA illegal/misaligned address；664236 替代664199，**664238 替代664215**。两个 causal Slurm 脚本现默认排除已知故障节点 `mind-1-19-[1-2],mind-1-24`，失败 job 的依赖均已替换，不影响实验设计。

通用 `slurm_run*.sh` 也已支持 `XOR_CODE_DIR`/`XOR_RUN_DIR`、per-job HuggingFace cache 和故障节点排除；后续补 seed 可用隔离 worktree 的代码，同时复用 `~/xor/exp` 中已有阶段 checkpoint。

### 当前运行：Critical CNN seed 补齐（6 jobs + Bark）

| Job IDs | 实验 | Seeds | 状态 | 续跑位置 |
|---------|------|-------|------|----------|
| **664241/664242** | CNN SVHN 2-arg | 44/45 | PENDING（Resources/Priority） | `~/xor/exp/cnn_svhn_2arg_*` |
| **664243–664246** | CNN FashionMNIST 1-arg | 42–45 | PENDING（Priority） | `~/xor/exp/cnn_fmnist_1arg_*` |
| **664254–664258** | CNN CIFAR-10 PReLU+LN | 1234/42–45 | PENDING；afterany 664241–664246 | `~/xor/exp/cnn_cifar_prelu_ln_*` |
| **664259–664263** | CNN CIFAR-10 Swish+LN | 1234/42–45 | PENDING；afterany 664241–664246 | `~/xor/exp/cnn_cifar_swish_ln_*` |

提交清单：`/user_data/yizhouc3/xor_cnn_seed_completion_20260827.tsv` 与 `/user_data/yizhouc3/xor_m6_baselines_20260827.tsv`。使用隔离代码 `/home/yizhouc3/xor-codex-audit`；Critical seed补齐完成通知 job **664247**，M6 baseline完成通知 job **664264**。

### 当前验证：cluster-only

- **664267**：❌ FAILED（4秒），尝试在 CPU 分区运行完整 pytest；原因是 cluster conda `xor` 环境未安装 pytest，未进入测试执行。
- **664268**：✅ COMPLETED（36秒），CPU 分区的 `run.py --validate` 已确认两份 M6 配置均可构建并完成一次前向；PReLU/Swish 参数量分别为 125,226 / 125,222。

### 上一轮（2026-07-26）已终止

| Job | 实验 | 状态 | 目的 |
|-----|------|------|------|
| 613846–613855 | `SeqMinGatedRNN` seeds 42–51 | 8 success / 1 NaN / 1 infra OOM | Seeds 42/44–50 完成；seed 51 ep17 NaN；seed 43 节点无 GPU后 host-memory OOM。成功结果 98.44±0.22%（popSD）。inner1/2 gate R²=0.245±0.223 / 0.447±0.263，约束未改善机制可辨识性。 |
| 613861–613870 | Bilinear causal matrix | 5 completed / 2 timeout / 3 incompatible-GPU failures | Joint seeds 42/43/45/46 完成并保持 `a*b`；joint seed44 timeout。Frozen seed42完成、seed43完成 random 后 timeout；seeds44–46落到 RTX Pro 6000，CUDA build不支持。部分结果已拉回。 |
| 613871–613875 | SwiGLU cross-init×cross-seed | 5 incompatible-GPU failures | 全部落到 RTX Pro 6000 (`mind-1-19`)，启动时 `no kernel image`；没有训练结果。需要在 L40S/兼容GPU重跑。 |
| 613857–613859 | 旧 Bilinear probe | CANCELLED by user | 约29分钟时被取消，未进入 InnerNet probe；已被 causal matrix 取代。 |

### 已终止（原 2026-06-27 提交的 job，终态已核实）

| Job ID | 实验 | 配置 | 终态 |
|--------|------|------|------|
| 547111 | **P1 部署段 — FFN deploy（case 1，主打速度）** | `deploy_distilled.py`，4 op (gelu/swiglu/innernet/distilled-poly3)，WikiText-2 d=128 d_ff=512 4层 20ep×5seeds，全部同一 GPU 测吞吐 | ❌ **TIMEOUT**（跑满 2 天被砍；`deploy_ffn_d128/results.json` innernet 仅 4 seed、distilled 空——部署闭环非投稿前置，不重跑） |
| 547112 | **P1 部署段 — CNN deploy（case 2，主打非-SwiGLU 新算子）** | `deploy_distilled_cnn.py`，4 op (relu/swiglu/innernet/distilled-poly3)，CIFAR-10 100ep×5seeds | ✅ COMPLETED (10h)；`deploy_cnn_cifar10/results.json`：InnerNet 84.6–85.7% vs ReLU/SwiGLU ~80%，distilled 81% 且快 3× |
| 547208 | **P2 — Plan B 稳定性诊断（5ep 快测）** | `seq_mnist_gated_diag.yaml`：全叠加稳定手段（ortho + warmup3 + cell_tanh + clip0.25），只 5ep×5seeds 快看 NaN 模式 | ✅ COMPLETED (6h) |
| 547209 | bark 通知 | afterany 依赖 547111/112/208，cluster 自己 curl Bark | ✅ COMPLETED |

**时间估算**：FFN deploy ~1.5 天（4min/epoch×400，innernet 更慢）；CNN deploy ~十几小时；P2 RNN 极慢（~18min/epoch，784 步 BPTT），诊断 ~7h，若过则全跑 150ep×5seeds ~一周。

**P2 进展**：第 2 轮（ortho+warmup，job 547114 已取消）**回归了**——seed 42（原本最稳）在 ep1 NaN，seed 43 却正常。本地复现确认 **init 时前向不 NaN**（所有 seed max|out|≈0.06-0.2），所以 **NaN 是训练中（ep1 内）的不稳定，不是初始化前向问题，也不是纯梯度爆炸**（warmup lr≈0 仍炸）。**第 3 轮诊断 547208 结果已拉回本地（`exp/seq_mnist_gated_diag_20260627_171855_d46bf25c/results.p`）：全叠加稳定手段（ortho+warmup3+cell_tanh+clip0.25）后仍 2/5 NaN——seed 42 ep4 炸、seed 46 ep1 炸；43/44/45 正常（5ep 快测只到 0.56/0.65/0.67，未收敛）。结论：稳定性仍是未解问题，全叠加也没根治。**

**部署分析（`results/audit/deploy_analysis.json`）**：
- CNN 5 seeds：InnerNet **84.95±0.57%**、distilled poly3 **81.32±0.32%**、SwiGLU **79.97±0.37%**、ReLU **79.93±0.17%**（sample SD）。distilled 比 InnerNet 低 **3.63 points**（paired-t p=0.00015），但仍比 ReLU/SwiGLU 高 1.39/1.35 points（p=0.0010/0.0046）。
- CNN 吞吐：InnerNet 1,622 img/s；distilled 4,345（**2.68×** InnerNet）；SwiGLU 10,680（6.59×）；ReLU 15,059（9.29×）。Distilled 与 SwiGLU 参数相同（658,570），InnerNet只多129参数；瓶颈是逐元素小 MLP 的执行成本，不是参数量。
- FFN deploy 不完整：InnerNet 4/5、distilled 0/5。已完成分支中 InnerNet约99.6k tok/s，SwiGLU约600.1k（**6.03×**），GELU约677.9k（6.81×）。不能对 distilled FFN 质量/速度作结论。
- 结论：CNN 闭式蒸馏追回部分速度并保留相对 fixed baselines 的优势，但没有保持完整 InnerNet 精度，且手写 poly3 仍比 SwiGLU慢2.46×；因此部署闭环是有限正面结果，不是论文主 claim。

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
| **SeqMinGatedRNN（无 `W_c`）** | **21K** | **98.44±0.22%** | **8/9实际训练成功，1 NaN；另1 infra OOM** | 功能稳定性提高，gate表面仍不可辨识 |

**关键发现**：Plan A（单InnerNet替换tanh）完全失败。Plan B证明加法记忆+可学二元交互在功能上足以解决任务；约束版本进一步以21K参数在8/9实际训练seed达到98.44%。但跨seed表面分析仍不支持单个InnerNet收敛为标准gate，说明性能发现与机制可辨识性必须分开。

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

### 🔴🔴 论文主线与审计状态（2026-07-26 更新）

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| **P1** | **表面定量提炼** | ⏳ causal matrix v2 运行中 | 初步结果显示 host-dependent：SwiGLU host → SwiGLU-like，Bilinear host → pure `a*b`。有效矩阵现为 5 seeds × Bilinear joint/frozen × random/multiply，以及 5 seeds × SwiGLU joint × 4 init；共享 host checkpoint、可续跑，并排除已知不兼容节点。当前 8/10 host 已完成，16/20 probes 已启动；664215 的节点级 CUDA 失败已由 664238 替代，无未处理失败。`scripts/analyze_causal_matrix.py` 已就绪，结果落盘后自动检查 40/40 conditions 并汇总 PPL、surface R² 与 operator votes。 |
| **P2** | **Seq-MNIST 功能与稳定性边界** | ✅ 约束实验完成 | 去掉 `W_c` 后 8/9 实际训练成功，98.44±0.22%，1 NaN（另 1 infra OOM），参数从37K降至21K。inner2 gate R² 0.447±0.263，与旧设计约0.43±0.31相同：功能结果增强，机制仍不可辨识。 |
| **P3** | **结果审计与统计严谨性** | ✅ 工具链完成；等待 P1 新结果 | canonical manifest 已覆盖统一 runner + deploy/Seq-MNIST/warm-start/PPO script-native 结果；自动分组/冲突报告、24项预注册核心比较、文档一致性检查及deploy trade-off分析完成（31 tests PASS）。当前238/238科学配置/状态组可汇报，0个同配置seed冲突；RESULTS_CN/EN 的58个已注册 headline cells 与 manifest **58/58一致**。NaN runs 与 success runs 显式分组。causal v2 完成并拉回后只需重跑生成链并补注册项。 |

### 🔴 Critical

| # | 项目 | 状态 |
|---|------|------|
| C1 | 补齐 5 seeds | ⏳ FMNIST 2arg 已拉回并验证 5/5；SVHN 2arg 已拉回 3/5，seeds 44/45 已提交（664241/664242） |
| C2 | CNN 参数公平对比 | ✅ ReLU matched 70.67% vs InnerNet 78.29% (同 127K) |
| C3 | AE 参数匹配 | ✅ ReLU matched 0.0059 vs InnerNet 0.0039 (同 ~660K) |
| C4 | 1-arg 系统对比 | ⏳ SVHN 1arg 原始结果已拉回并验证 5/5；FMNIST 1arg 仍为 1/5，seeds 42–45 已提交（664243–664246） |
| C5 | ReLU+LN ablation | ✅ 4 数据集完成 |
| M1 | 训练曲线 | ✅ `results/figures/fig_training_curves.{png,pdf}`；4-panel mean±sample-SD 图已生成，并排除同名的旧 MLP-MNIST width-64 baseline，使用参数匹配 width-112 版本 |

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
| U15 | RL 加 seeds + 只报 PPO | ✅ 已按30-seed原始曲线纠正 | LunarLander 最后20个评估点均值：InnerNet 98.8±94.7 vs ReLU 101.3±57.8（paired-t p=0.897，持平），两者均显著优于 SwiGLU -210.4±84.0。旧187.6/158.8/-249.7只是各自最后一个 seed 的日志值，已撤回。 |
| U16 | Masked LM（类 BERT） | ✅ SwiGLU 93.83, GELU 101.39, InnerNet 124.82（差）。warm-start 大幅赢 |
| U17 | Transformer Classic InnerNet FFN | ✅ Wiki 95.49 ≈ Semantic 95.26, PTB 208.81 ≈ Semantic 207.81。TF 上 Classic ≈ Semantic |
| U18 | d=64 InnerNet 学到了什么 | ✅ | post-sharing seed42 的 SwiGLU fit R²=0.977（mult R²=0.568，poly3 R²=0.995），主项 `0.321·SiLU(a)·b`；`fig_d64_swiglu_surface.{png,pdf}` 展示 learned / fitted / residual |
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
| U30 | Scaling law 图 | ✅ | `fig_scaling_law.{png,pdf}` 已从 canonical post-sharing 数据自动生成：3.3%→1.6%→0.8%→1.7%，明确为正向但非单调，并标注 paired-t p 值 |
| U31 | 训练曲线 | ✅ | `fig_warmstart_curves.{png,pdf}`：5-seed warm-start fork 与 frozen-capacity 轨迹，mean±sample-SD；原始逐 epoch 数据来自 `exp/ivs_d128_v2/results.p` |
| U32 | 参数量和推理速度 | ✅ | `deploy_analysis.json`：CNN InnerNet只比SwiGLU多129参数但慢6.59×；distilled快2.68×但仍比SwiGLU慢2.46×。FFN InnerNet约比SwiGLU慢6.03×（4 seeds；distilled未跑到） |
| U33 | 提炼 InnerNet 为简单公式 | ✅ | d=128 poly3 R²=0.997；SwiGLU family R²=0.942。CNN poly3 R²=0.974、SwiGLU family R²=0.908。causal结果说明具体算子依赖host/basin，不能称普适SwiGLU吸引子 |
| U34 | Qwen2.5-0.5B finetune | ✅ **负面结果** | 3 seeds: InnerNet ~80% vs SwiGLU ~89%。替换瞬间崩到 52-66%，恢复不回来。大模型直接替换不可行 |
| U35 | InnerNet hidden dim 消融 | TODO | hidden=8/16/32/64 对比，InnerNet 需要多大才够 |
| U36 | Non-shared warm-start | ⏳ PTB ✅ MLM ✅ | PTB 5/5 赢, CNN +3.12%, **MLM non-shared 15.63 vs SwiGLU 18.91 (-3.28)**。Wiki d=128 不在当前队列，本地无最终原始结果，不再标记为运行中。 |
| U37 | Free-init (不同初始化) | ✅ Wiki 3/3, MLM 2/3 | Wiki: 4 种初始化全收敛到 ~71.7-72.6 vs SwiGLU ~77.3。MLM: random/multiply/swiglu_fitted/identity 都 ~15.7-16.1。**初始化不影响终点** |
| U38 | Multiply-init 多任务 | ✅ 5/5 seeds | d=64 持平, d=128 -0.24, PTB -1.08, **MLM MultInit 15.93±0.21 vs SwiGLU 19.09±0.27 (-16.6%)** |
| U39 | Scratch-init (从头训对比) | ⏳ 2.5/5 seeds | SwiGLU 一致赢所有 InnerNet 初始化。Seed 42: SwiGLU 76.6 > Gaussian 78.1 > Random 79.3 > Multiply 80.2 |
| U40 | RNN PTB 重跑 | ✅ | 2arg Test PPL≈169, 1arg PPL≈179, tanh PPL≈140。**InnerNet 输 tanh baseline 20-28%**。PTB 上 InnerNet 不好用 |
| U41 | Sequential MNIST（加法记忆 + InnerNet） | ✅ 约束实验完成 | Plan A 11.04%失败；原Plan B 3/5成功；约束版21K、8/9实际训练成功、98.44±0.22%。功能结果增强，但单个InnerNet gate表面仍不可辨识。 |
| U42 | Plan B 训练稳定性 | ✅ 诊断收口：全叠加仍 2/5 NaN | 前 3 个稳定性变体各 2/5 NaN。第 3 轮全叠加（ortho+warmup3+cell_tanh+clip0.25）诊断 547208 结果已拉回：seed 42 ep4 NaN、seed 46 ep1 NaN，43/44/45 正常。全叠加未根治，稳定性是已知边界。 |
| **U20** | **修复 InnerNet parameter sharing** | ✅ TF 全完成 | d=64 112.83, d=128 95.26, d=192 88.42, **d=256 84.62**, PTB 207.91。全部赢 GELU。ResNet full 持平, internal +1.5%。MLM 124.82 差 |

### 🟡 Major

| # | 项目 | 状态 |
|---|------|------|
| M2 | 2D 激活函数表面可视化 | ✅ `fig2_2d_activation_surfaces.{png,pdf}` 已改为真实 CNN checkpoint；移除原来手写的“learned”示意面，避免把合成函数误作实验结果 |
| M3 | CNN 小 scale 反转解释 | ✅ n=3 不支持稳定反转：paired差值 -11.04/+0.74/-0.10pp，均值-3.47pp但 p=0.457；由单个 seed1234 崩落驱动，按高方差边界报告，不归因于固定机制 |
| M4 | 回归 inconsistency 解释 | ✅ Housing width sweep 显示容量交叉：w32/64/120 改善5.6/1.6/4.7%，w256/512 反转为-2.2/-4.5%；n=3，仅w64/120 paired-t<0.05，解释为小模型 inductive-bias 收益随容量消失，不作普适回归增益 claim |
| M5 | RL inconsistency | ✅ 已统一指标并降级 | 全部按每 seed 最后20个 recorded eval 的均值汇总；Acrobot InnerNet 显著优于 ReLU，LunarLander 与 ReLU 持平且方差很大，RL 仅作扩展证据 |
| M6 | PReLU/Swish baseline 对比 | ⏳ jobs 664254–664263 已提交；cluster验证通过 | CIFAR-10 CNN，参数匹配宽度46/92/92/92 + LayerNorm，PReLU/Swish各5 seeds；664268 已验证两份配置，训练依赖 Critical CNN jobs 完成后启动，Bark job 664264 |

### 🟢 Minor

| # | 项目 | 状态 |
|---|------|------|
| m1 | ResNet baseline 提升 | 被 U2 覆盖 |
| m2 | 论文原始数字复现对比表 | TODO |
| m3 | 更多 LM dataset | ✅ PTB 已完成 |
| m4 | 计算开销分析（FLOPs + wall-clock） | TODO |
| m5 | 显著性检验 p-value | ✅ 19项预注册比较，paired/Welch t、Wilcoxon/Mann–Whitney、bootstrap CI 与效应量均自动生成 |
| 24 | 参数效率出图 | ✅ `fig_parameter_efficiency.{png,pdf}`；模型实测参数量 + canonical 5-seed CIFAR-10准确率，InnerNet w128 415,051参数与ReLU w256 920,842参数表现相当，少54.9% |

---

## 最新确认结果

### CNN 图像分类（完整 ablation）
| Dataset | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | Gain |
|---------|-------|-------|------|---------|-------------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | +4.58 |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | +1.57 |
| SVHN | 95.016±0.005 (n=3) | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | +3.74 |

### Transformer LM (PPL↓)
| 配置 | InnerNet | Baseline | 差异 |
|------|----------|----------|------|
| d=64 | **112.83** | 116.63 | -3.3% |
| d=128 | **95.26** | 96.82 (GELU) | -1.6% |
| d=192 | **88.42** | 89.11 | -0.8% |
| d=256 | **84.62** | 86.05 | -1.7% |
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
- PPO Acrobot: 最后20个评估点均值 -90.1 vs ReLU -111.8（+21.7 return points，paired-t p=0.0036）
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
