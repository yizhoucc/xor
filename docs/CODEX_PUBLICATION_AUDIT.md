# Codex 发表前结果审计工作记录

## 文档用途

这份文档是 `codex/publication-audit` 分支的详细工作日志。它记录每一步做了什么、为什么做、依据是什么、改了哪些文件、如何验证、对应提交，以及仍未解决的问题。它不替代 Git diff，而是保存 diff 无法表达的研究判断和证据链。

## 基线与隔离

- 开始时间：2026-07-26 10:16 PDT
- 开始前分支：`main`
- 开始前快照：`e4d4e9a` (`Snapshot workspace before Codex publication audit`)
- 快照已推送：`origin/main`
- Codex 工作分支：`codex/publication-audit`
- 工作分支已推送并跟踪：`origin/codex/publication-audit`
- 回退方式：任何 Codex 修改都可以与 `e4d4e9a` 比较，或从该提交重新创建分支。

## 用户确认的论文范围

1. 论文的核心贡献是“发现”：InnerNet 学到可解释的二元交互结构，包括 SwiGLU-like interaction 和 recurrent gate-like behavior。
2. “发现 -> 提炼 -> 部署”完整工业闭环不是投稿前置条件。部署、跨任务迁移和推理优化只有在耗时很短且直接增强论文证据时才做。
3. 核心对比是否公平，先以原论文的实验协议和本项目实际协议为依据审计；读完原论文后再提出异议，不能只按现代 Transformer 的常规做法推断。
4. 当前优先工作是结果源审计和统计协议，避免继续增加数据集、模型或长时间训练。

## 审计目标

### A. 原论文协议核对

- 核对 2-arg、1-arg 与 baseline 的参数匹配方法。
- 核对作者比较的是相同外部宽度、有效宽度、参数量还是训练预算。
- 核对三阶段训练、随机函数预训练、共享 InnerNet 和重复次数。
- 明确哪些本项目对比是在复现原协议，哪些是扩展实验中的新协议。

### B. 唯一结果清单

为每个可汇报结论建立机器可读 manifest，至少包含：

- claim / experiment family
- config path
- exp directory
- model / dataset / condition
- seed
- metric name / metric direction
- selected epoch 与选模 split
- final value
- result source file
- config hash / Git commit（能恢复时）
- 状态：`raw-verified`、`derived`、`doc-only`、`conflicted`、`incomplete`

证据优先级：原始 per-seed 文件和日志 > checkpoint 可复算结果 > 自动聚合文件 > `RESULTS_*.md` / `PROJECT_STATUS.md` 手工数字。

### C. 统计协议

- 同 seed 或共同 checkpoint 分叉的实验使用配对检验。
- 独立训练且 seed 无法对齐的实验使用独立样本检验。
- 报告均值、样本标准差、差值、95% confidence interval、效应量和原始 seed 数值。
- 小样本同时报告 parametric 与 non-parametric 结果，但不以单一 `p < 0.05` 替代效应大小。
- 不完整 seed、NaN 和提前终止单独标注，不静默删除。

## 当前短任务顺序

| 顺序 | 任务 | 状态 | 产出 |
|---|---|---|---|
| 0 | 冻结开始前状态并创建独立分支 | 完成 | `e4d4e9a`，`codex/publication-audit` |
| 1 | 阅读原论文并核对公平性协议 | 完成 | 本文中的协议结论与引用位置 |
| 2 | 盘点本地和集群结果源 | 本地完成；远端阻塞 | source inventory、缺失清单 |
| 3 | 建立 canonical manifest | 本地初版完成 | 机器可读结果清单和冲突报告 |
| 4 | 修正 paired/unpaired 统计工具 | 完成 | 脚本、测试、示例输出 |
| 5 | 用 manifest 自动复核核心表格 | 待开始 | 可重复生成的核心结果表 |

## 暂缓事项

以下事项不作为当前审计的前置条件：

- 重跑 Transformer、CNN、Sequential MNIST 长实验
- fixed-op fresh-model deployment
- empirical-distribution distillation
- operator transfer matrix
- CUDA / 推理性能优化
- 新数据集或新架构
- 全仓库发布包装

审计结束后只选择能直接修补论文证据且成本明确的工作。

## 已知待核实冲突

1. Transformer parameter-sharing 修复前后两组 scaling 数字混用；现有 scaling 图使用旧数字，结果表使用新数字。
2. PTB non-shared 表格中的 `162.64` 与相对 `162.22` 的 `-1.95` 差值无法同时成立。
3. 部分 warm-start、free-init、scratch-init 和 GPT 实验 seed 未完成，需要明确哪些数字可作为正式结论。
4. `PROJECT_STATUS.md` 的 Slurm 状态停留在 2026-06-27，需要与集群真实状态核对。

## 详细执行日志

### 2026-07-26 10:16 PDT - 建立工作基线

操作：

- 检查 `git status`、新文件大小与 `git diff --check`。
- 将用户已有的 session log 更新、旧结果文件清理、新 Figure 13 和 4 个小型权重文件提交到 `main`。
- 推送 `main`。
- 从 `e4d4e9a` 创建 `codex/publication-audit` 并推送远端。

结果：

- `main` 和 Codex 分支都以 `e4d4e9a` 为共同、可恢复基线。
- Codex 后续修改不进入 Claude 使用的 `main`，除非用户明确合并。

验证：

- `git status --short --branch`
- `git log -2 --oneline --decorate`
- `git push origin main`
- `git push -u origin codex/publication-audit`

### 2026-07-26 10:25 PDT - 原论文实验协议核对

来源：`docs/2110.06871v2.pdf`，重点阅读 Section 3、4.1、4.2、4.3、4.4、4.5 和 Discussion。

原论文明确事实：

- InnerNet 是跨所有层和节点共享的 canonical activation；MLP InnerNet 为两个 64-unit hidden layers，CNN 使用等价的 1x1 convolutions。
- ReLU 和 1-arg baseline 保持相同的网络类型与层数，但不是保持相同 hidden width。Section 4.2 / Figure 6 明确通过调整每层 hidden units 或 feature maps，使总 learnable parameter counts comparable。
- Figure 4d 报告 4 次重复的 mean test accuracy 与 ±1 SD；论文没有对性能曲线报告 hypothesis-test p-value。
- 论文的发现证据包括二次多项式拟合、curvature 符号和 spectral analysis；其中 curvature 跨 48 trials 使用 binomial null，得到 CNN negative curvature `p=0.007`。
- 论文没有把提炼函数放入 fresh model 做部署闭环。Discussion 只指出 polynomial approximation 可以降低实际应用的参数和内存需求。因此部署不是沿用原论文 discovery claim 所必需的证据。

本项目协议核对：

- MLP/CNN 复现实验已有 parameter-matched ReLU 配置，其思路与原论文 Section 4.2 一致。
- 新增 Transformer 在相同 `d_ff=512` 下比较 GELU、SwiGLU、InnerNet，不是原论文意义的 total-parameter matching。本地按 `vocab_size=10000, d_model=128, n_layers=4` 实例化得到：GELU `2,083,344`，SwiGLU `2,347,536`，InnerNet `2,347,665`。
- 近似参数匹配组合为：GELU `d_ff=768` (`2,346,512`) 对 InnerNet `d_ff=512` (`2,347,665`)；或 GELU `d_ff=512` (`2,083,344`) 对 InnerNet `d_ff=341` (`2,083,641`)。

研究判断：

- 参数差不影响“训练后的 InnerNet surface 接近 SwiGLU-like interaction”这一发现，因为该结论来自所学函数本身。
- 参数差影响“同等模型规模下优于 GELU”和 scaling 性能增益的严格表述。当前不为此新增长实验；正文应把现有比较标明为 same-width，并避免把它表述成 parameter-matched superiority。
- “发现 -> 提炼 -> 部署”不再列为投稿前置条件。已有 distillation 结果可以作为发现的定量解释，部署脚本留作扩展工作。

验证命令：

- `pdftotext -layout docs/2110.06871v2.pdf -`
- 使用 `.venv/bin/python` 实例化三种 Transformer 并统计 `sum(p.numel())`

### 2026-07-26 10:40 PDT - 配对统计工具

改动：

- 重构 `scripts/compute_stats.py`，保留默认 independent mode，并新增显式 `--paired`。
- paired mode：paired t-test、Wilcoxon signed-rank、Cohen's dz、within-pair mean difference bootstrap confidence interval。
- independent mode：Welch t-test、Mann-Whitney U、pooled Cohen's d、independent bootstrap confidence interval。
- 非有限值不再静默忽略：paired mode 按 pair 删除并报告数量；independent mode 分组删除并分别报告。
- 新增 confidence level、bootstrap iterations、random seed 和 raw-value display 参数。
- 新增 `tests/test_compute_stats.py`，覆盖配对差值、NaN/Inf pair、长度错位和独立样本统计。

验证：

- `.venv/bin/python -m unittest tests/test_compute_stats.py -v`：4 tests passed。
- `.venv/bin/python -m py_compile scripts/compute_stats.py tests/test_compute_stats.py`：通过。

现有 Transformer 5-seed 示例（每个 seed 取其 10 epoch 最低 validation PPL；same-width，不是 total-parameter-matched）：

| Condition | Mean | Sample SD | Per-seed values |
|---|---:|---:|---|
| InnerNet | 95.2608 | 1.1130 | 93.8331, 95.5599, 94.5241, 96.7153, 95.6717 |
| GELU | 96.8187 | 1.3267 | 95.6541, 95.1574, 97.3564, 97.8701, 98.0554 |
| SwiGLU | 92.9806 | 1.2703 | 92.3048, 91.5118, 93.0462, 94.9354, 93.1047 |

配对结果：

- `GELU - InnerNet = +1.5579 PPL`，95% bootstrap CI `[+0.4869, +2.4506]`，paired-t `p=0.05095`，Wilcoxon `p=0.125`，Cohen's dz `1.23`。
- `SwiGLU - InnerNet = -2.2803 PPL`，95% bootstrap CI `[-3.2379, -1.5585]`，paired-t `p=0.00917`，Wilcoxon `p=0.0625`，Cohen's dz `-2.11`。

解释约束：

- InnerNet 相对 GELU 的方向和效应量较大，但 n=5 下 paired-t 略高于 0.05，Wilcoxon 也不显著；不能只引用 bootstrap interval 或只引用一个检验。
- SwiGLU 相对 InnerNet 的方向在 5 个 seeds 上一致。n=5 的双侧 Wilcoxon 最小可达 p-value 受离散性限制，因此同时报告原始 seed 值更重要。
- 这些数值用于验证统计协议，不改变前述 same-width / parameter-count 限制。

### 2026-07-26 11:00 PDT - 本地结果源 inventory 与 manifest

外部状态：

- 尝试通过 `ssh ... mind.cs.cmu.edu squeue ...` 核对集群，连接被当前执行沙箱以 `Operation not permitted` 阻止。不是 SSH key 或账号认证失败。
- 因此 2026-06-27 的集群任务状态仍不能更新，远端结果也尚未拉回。

实现：

- 新增 `scripts/build_result_manifest.py`。
- 递归扫描 `exp/**/config.yaml`，解析 config metadata、config hash、阶段标记和已知结构化结果文件。
- 支持 `test_results.p` scalar metrics，以及 `lm_results.p` / `mixer_results.p` / `rl_results.p` 的多 seed curves。
- 多 seed curve 同时保存 best 和 final 值、选择规则、selected epoch、总 epochs、seed 和 source file。
- 不从自由文本日志猜测指标；缺结果与不完整目录单独标记。
- 新增 4 个 manifest 单元测试，与统计测试合计 8 个 tests 全部通过。

生成结果：

- `results/audit/experiment_inventory.csv`：452 个带 config 的实验目录。
- `results/audit/metric_manifest.csv`：919 行可追溯指标。
- `results/audit/audit_summary.json`：368 `raw-verified`、83 `incomplete`、1 `completed-no-result`。
- 结构化来源计数：329 `test_results.p`、10 `lm_results.p`、9 `mixer_results.p`、20 `rl_results.p`。

核心实验追溯：

- Transformer WikiText-2 InnerNet/GELU/SwiGLU 各 5 seeds，均可直接从 `lm_results.p` 复算。
- CNN CIFAR-10 1-arg 和 ReLU 各 5 个 test accuracy，均可直接追溯。
- CNN CIFAR-10 2-arg 只有 seed 1234/43/44/45 四个结构化 test accuracy。seed 42 目录 `exp/cnn_cifar_2arg_20260404_172455_9a5b0541` 虽有 `TEST_DONE` 和 `COMPLETED`，但没有 `test_results.p`；日志停在 `Starting Test`。
- 文档汇总 `78.57±0.74` 可以由四个 raw values 加 `79.68%` 精确还原，但 `79.68%` 当前只能标为 `doc-derived`，不能标为 raw-verified。需从远端原始结果或旧备份恢复。

验证：

- `.venv/bin/python -m unittest tests/test_build_result_manifest.py tests/test_compute_stats.py -v`：8 tests passed。
- `.venv/bin/python scripts/build_result_manifest.py`：成功生成三份 audit artifacts。

### 2026-07-26 (Claude) - SSH 远端取证与文档冲突修复

背景：Codex 的 SSH 取证被其执行沙箱阻止（`Operation not permitted`）。Claude 环境可正常 SSH 到 Mind Cluster，故由 Claude 完成 SSH-1/2/3 并接手 LOCAL-2/3/4 中已有证据支持的部分。所有远端操作只读/复制，未在登录节点跑训练或计算。

SSH-1（旧 job 终态，`sacct`）：

- `547111` FFN distilled deploy：**TIMEOUT**（2 天上限被砍，batch CANCELLED）。`deploy_ffn_d128/results.json` 中 innernet 仅 4 seed、distilled 为空。部署闭环非投稿前置，不重跑。
- `547112` CNN distilled deploy：COMPLETED（10h）。
- `547208` Sequential MNIST 诊断：COMPLETED（6h）。
- `547209` bark notify：COMPLETED。
- 当前 `squeue` 队列为空。

SSH-2（结构化结果恢复）：

- **CNN CIFAR-10 2-arg seed 42 恢复成功**。集群目录 `cnn_cifar_2arg_20260404_172455_9a5b0541` 存在 `test_results.p`（本地此前未同步），内容 `{'test_accuracy': 0.7969}`，日志 `Test Accuracy = 0.7969`。已 scp 回本地同相对路径。重跑 manifest 后该 seed 由 `completed-no-result` 变为 `raw-verified`。
- 修正：文档此前反推的 79.68% 应为 **79.69%**。五个 seed（78.68/79.69/78.94/77.93/77.62）mean=78.57%，population SD=0.74，sample SD=0.82。原 `78.57±0.74` 均值正确，用的是 population SD。
- PTB non-shared 冲突原始来源定位为 `exp/warmstart_nonshared/results.p`（已 scp 回本地）。这是配对 warm-start 实验：PTB 每 seed InnerNet(non-shared) vs 其自身 SwiGLU baseline。InnerNet mean=162.49（SD 3.44），SwiGLU mean=164.59（SD 4.07），InnerNet 5/5 全赢，paired diff=-2.11，paired-t p=0.037，Wilcoxon p=0.0625，Cohen's dz=-1.38。
- 冲突根因：结果文档 warm-start 表把 non-shared 的 Δ 写成相对 shared 实验的 SwiGLU baseline（162.22），但实际是相对 non-shared 自身 baseline（164.59）算的。旧值 `162.64 / -1.95` 因此自相矛盾。已按原始 pickle 更正为 `162.49 / -2.11` 并在三份文档加注说明两个 SwiGLU baseline 来自不同 run、不可直接比较。

SSH-3（6/27 后新结果）：仅 3 个结构化文件更新——`deploy_cnn_cifar10/results.json`、`deploy_ffn_d128/results.json`、`rnn_ptb_tanh_.../test_results.p`（后者与本地内容一致，仅 mtime 被 touch）。均非核心分类实验。

文档修复（LOCAL-3/4，仅改有原始证据支持的项）：

- Transformer d=128 InnerNet PPL：`95.23` → `95.26`（`RESULTS_EN.md`、`RESULTS_CN.md`、`PROJECT_STATUS.md`）。原始 `lm_results.p` 每 seed 取最低 val PPL 得 mean=95.2608；`-1.6%` 相对 GELU 96.82 与 95.26 一致。
- Transformer "consistently beats GELU" 措辞改为 same-width 限定 + 报告 d=128 配对统计（paired-t p=0.05，95% CI [0.49, 2.45]，dz=1.23），核心卖点改述为"自动再发现 SwiGLU 式门控"而非性能碾压。
- CNN §1 加 SD 约定说明（population SD，headline 2-arg sample SD=0.82）。
- `PROJECT_STATUS.md`：审计小结更新为 SSH 已完成；集群"运行中"表改为"已终止"并填入四个 job 终态；manifest 计数更新为 369 raw-verified / 83 incomplete / 0 completed-no-result；PTB `-1.95` → `-2.11`。

验证：

- `sacct -j 547111,547112,547208,547209`、`squeue -u yizhouc3`
- 本地 `test_results.p` 读回确认 `{'test_accuracy': 0.7969}`
- `scripts/compute_stats.py /tmp/ptb_ns.json --ref swiglu --lower_better --paired` 与 Transformer 5-seed 配对统计
- `scripts/build_result_manifest.py` 重生成，seed 42 行为 `raw-verified`

遗留：`deploy_ffn_d128`（547111 TIMEOUT）innernet 仅 4 seed、distilled 空——属部署闭环，非投稿前置，按用户范围不重跑。

### 2026-07-26 (Codex) - SSH 交付复核

已独立验证：

- 两个新提交 `90542da` / `c0e4115` 均已推送，工作区开始复核时干净。
- CNN CIFAR-10 2-arg seed 42 pickle 内容为 `{'test_accuracy': 0.7969}`；重跑 manifest 后为 369 raw-verified / 83 incomplete / 0 completed-no-result。
- `exp/warmstart_nonshared/results.p` 的 PTB 5-seed 原始值可复算 InnerNet 162.49、SwiGLU 164.59、paired difference -2.11；文档修正正确。
- 8 个 audit/statistics tests 全部通过，重新生成 manifest 后 Git 无差异。

复核发现并已修正的残留冲突：

- `RESULTS_CN.md` / `RESULTS_EN.md` 和 P3 仍声称 scaling improvement 单调缩小，但 parameter-sharing 修复后的表格为 3.3%→1.6%→0.8%→1.7%，并不单调。
- `PROJECT_STATUS.md` 的 P2/U42 仍写诊断“即将开始/在跑”，与 job 547208 已 COMPLETED 冲突。
- 论文范围已确认不要求部署，但中英文 story 仍把 deploy 写成投稿前置步骤。
- `scripts/plot_scaling_law.py` 使用修复前的 112.66/95.26/88.14/85.40；已标为 archived，不再用于论文。

仍缺本地原始文件：

- `exp/deploy_ffn_d128/results.json`
- `exp/deploy_cnn_cifar10/results.json`
- job 547208 对应的 Sequential MNIST 诊断结果目录/日志
- parameter-sharing 修复后 d=64/d=192/d=256 Transformer per-seed 原始结果（已由 `f93986c` 恢复并进入 manifest）

前三项不阻塞发现主线，但在本地文件补齐前，其数值只能视为 SSH-verified，不能进入 canonical manifest。最后一项 raw-traceability 缺口已经关闭。

### 2026-07-26 (Claude) - 复核 Codex 的 scaling 修正并恢复 post-fix 原始数据

触发：核对 Codex 提交 `8252ae1`（scaling 单调性修正 + 状态对账）。初看疑似 Codex 把数字搞反——本地 `lm_results.p` 显示 d=64=112.66、d=256=85.40，恰好匹配它标记为“作废”的 `plot_scaling_law.py`，而非它写进正文的 112.83/84.62。进一步 SSH 核实后确认 **Codex 是对的**：

- 本地 `transformer_wikitext_2arg_small/large_20260405`（d=64/d=256）是 **parameter-sharing 修复前的旧 run**（112.66 / 85.40），与旧 plot 脚本一致。
- 集群上有更新的 post-fix reruns（04-13/04-14），从其 `lm_results.p` 复算（每 seed 取最低 val PPL，seeds 42–46）：
  - d=64 InnerNet **112.83**（SwiGLU 112.31）
  - d=192 InnerNet **88.42**（SwiGLU 85.43）
  - d=256 InnerNet **84.62**（SwiGLU **81.56**）
  - 与正文 U20 / RESULTS 表完全一致，与 GELU 的百分比 3.3% / 0.8% / 1.7% 亦自洽（非单调，Codex 修正成立）。
- 已将这 6 个 post-fix scale dir 的 `config.yaml` + `lm_results.p` 拉回本地对应路径，manifest 重生成后 raw-verified 由 369 增至 **375**。Codex 之前标记“本地缺 post-sharing d=64/d=192/d=256 per-seed”这一 raw-traceability 缺口现已补齐。
- 顺带补全 RESULTS 表中 d=256 SwiGLU 的 ⏳ → **81.56±1.05**（来自 `transformer_wikitext_swiglu_large_20260413`）。

Codex 独立复算修正：上一行的 `±1.05` 是 sample SD；`RESULTS_EN.md` 既定约定为 population SD，因此正式表修正为 **81.56±0.94**。同时补齐所有新恢复规模的 population SD：d=64 InnerNet 0.94；d=192 InnerNet/SwiGLU 0.94/0.41；d=256 InnerNet/SwiGLU 1.47/0.94。

四规模 paired statistics（InnerNet - GELU，seeds 42–46）：

| d_model | Mean difference | 95% bootstrap CI | paired-t p | Wilcoxon p | Cohen's dz |
|---:|---:|---:|---:|---:|---:|
| 64 | -3.8039 | [-4.3320, -3.2976] | 0.00022 | 0.0625 | -5.702 |
| 128 | -1.5579 | [-2.4506, -0.4869] | 0.05095 | 0.1250 | -1.233 |
| 192 | -0.6843 | [-1.6623, 0.6514] | 0.36142 | 0.6250 | -0.460 |
| 256 | -1.4276 | [-2.9535, 0.0983] | 0.17321 | 0.3125 | -0.740 |

结论：四个均值方向一致，但不能称四个规模都显著；推断证据集中在 d=64，d=128 位于 0.05 边界，d=192/256 的 interval 包含 0。发现主张依赖学到的函数结构，不依赖每个性能点显著。

后续处理（已完成）：本地 04-05 的两个 InnerNet 2arg 旧 dir（small/large）是 pre-fix 归档产物（112.66 / 85.40），已 `git mv` 到 `archive/exp/`，从 manifest 移除（raw-verified 375 → 373），避免与 post-fix run 混淆。注意 04-05 的 GELU baseline dir（116.63 / 86.05）**未动**——parameter-sharing 修复只影响 InnerNet，GELU/SwiGLU baseline 不受影响，其值仍是正文 canonical。

### 2026-07-26 (Claude) - 补齐剩余非核心远端文件

将之前列为“仍缺本地原始文件”的三项全部 SSH 拉回，缺失清单清零：

- `exp/deploy_ffn_d128/results.json`、`exp/deploy_cnn_cifar10/results.json`：部署段结果（547111 TIMEOUT、547112 COMPLETED），仅作扩展结果，非投稿前置。
- job 547208 的 Sequential MNIST 全叠加稳定性诊断：`exp/seq_mnist_gated_diag_20260627_171855_d46bf25c/`（config + results.p + log）。原始 `results.p` 显示全叠加稳定手段（ortho+warmup3+cell_tanh+clip0.25）后**仍 2/5 NaN**：seed 42 在 ep4 NaN（best 0.1126）、seed 46 在 ep1 NaN（0.098）；seed 43/44/45 正常但 5-epoch 快测只到 0.5577 / 0.6492 / 0.6677（未收敛，仅诊断用）。据此 `PROJECT_STATUS.md` 的 P2/U42 从“结果待取”改为收口：gate 可被发现（成功 seed ~98% @150ep），但训练稳定性仍是未解边界，全叠加也没根治。论文按 §投稿前剩余工作 的约定同时报告成功率与失败率。

注：`seq_mnist_gated_diag` 的 `results.p` 结构（list of per-seed dict）不在 `build_result_manifest.py` 当前支持的四种结果文件之内，故不进 manifest；作为原始证据留档即可。

## 提交记录

| Commit | 分支 | 内容 | 验证 |
|---|---|---|---|
| `e4d4e9a` | `main` | Codex 开始前工作区快照 | push 成功，工作区干净 |
| `cee495c` | `codex/publication-audit` | 建立审计范围和详细记录 | push 成功，工作区干净 |
| `485f896` | `codex/publication-audit` | 原论文协议与公平性审计 | push 成功，工作区干净 |
| `2b35740` | `codex/publication-audit` | 配对/独立样本统计工具与测试 | 8 项中的统计 4 tests 通过，push 成功 |
| `5a9705a` | `codex/publication-audit` | 本地结果 inventory、canonical manifest 与测试 | 8 tests 通过，452 experiments / 919 metric rows，push 成功 |

## 后续工作分工

### 用户负责：SSH 与远端结果取证

所有操作只查询状态、读取日志或复制结果，不在 Mind 登录节点启动训练、Python 分析或其他计算任务。

#### SSH-1：核对 2026-06-27 记录的任务

查询以下 Slurm jobs 的最终状态、退出码和运行时间：

- `547111`：FFN distilled deployment
- `547112`：CNN distilled deployment
- `547208`：Sequential MNIST 5-epoch stability diagnostic
- `547209`：依赖任务的 Bark notification

建议只读命令：

```bash
ssh -Y -C -l yizhouc3 mind.cs.cmu.edu
squeue -u yizhouc3
sacct -j 547111,547112,547208,547209 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
```

需要返回的信息：每个 job 的 `State`、`ExitCode`、`Elapsed`，以及失败任务对应的 `logs/slurm_<jobid>.err` 最后 50 行。

#### SSH-2：恢复结构化结果文件

优先检查并复制回本地相同相对路径：

- `/home/yizhouc3/xor/exp/deploy_ffn_d128/results.json`
- `/home/yizhouc3/xor/exp/deploy_cnn_cifar10/results.json`
- job 547208 对应的 Sequential MNIST 结果目录和结果文件
- `/home/yizhouc3/xor/exp/cnn_cifar_2arg_20260404_172455_9a5b0541/test_results.p`

前两个 deployment 文件只用于补齐项目状态；即使不存在，也不阻塞论文发现主线。

CNN seed 42 是最高优先级。如果远端同名目录没有 `test_results.p`，继续检查旧备份、同步目录或 `/user_data/yizhouc3/xor_checkpoints` 中是否有对应 config hash `9a5b0541` 的记录。只需复制原始文件，不要根据文档数字手工创建 pickle。

#### SSH-3：列出 2026-06-27 之后的新结果

需要列出 `/home/yizhouc3/xor/exp/` 下 2026-06-27 之后更新的结构化结果文件：

```bash
find /home/yizhouc3/xor/exp -type f \
  \( -name 'results.json' -o -name 'test_results.p' -o -name 'lm_results.p' \
     -o -name 'mixer_results.p' -o -name 'rl_results.p' \) \
  -newermt '2026-06-27' -print
```

这些文件复制回本地 `exp/` 的同一相对路径即可；`exp/` 已被 Git 忽略，不会污染 Codex 分支提交。

#### SSH 完成标准

- 4 个旧 job 不再留在 `PROJECT_STATUS.md` 的运行中列表。
- CNN CIFAR-10 2-arg seed 42 被标记为 `raw-verified`，或明确记录为“原始结果已丢失”。
- 6 月 27 日后的远端结果都有本地副本或一份明确的缺失清单。

### Codex 负责：本地短任务

#### LOCAL-1：自动分组汇总

从 `metric_manifest.csv` 生成 group summary，包含：

- experiment name / model / dataset / metric
- raw seed values
- n 和 unique seed count
- mean
- population SD（复现现有文档数字）
- sample SD（正式统计建议）
- duplicate seed 和 missing seed 标记

遇到同实验、同 seed 的重复目录时不自动选择“最好”或“最新”结果，先输出冲突。

#### LOCAL-2：核心统计表

只计算具有可对齐 seeds 的现有核心比较：

- Transformer WikiText-2：InnerNet / GELU / SwiGLU
- CNN CIFAR-10：2-arg / 1-arg / ReLU（seed 42 恢复后）
- 其他已有完整 seed 数据且正文确实使用的比较

每项报告 raw values、配对差值、95% CI、paired-t、Wilcoxon 和 Cohen's dz。没有对齐 seed 的实验改用 independent mode，并明确标注。

#### LOCAL-3：文档冲突报告

逐项比较 `PROJECT_STATUS.md`、`RESULTS_CN.md`、`RESULTS_EN.md` 与 manifest，输出：

- 一致且 raw-verified
- 数值一致但标准差定义不同
- doc-derived / 缺原始来源
- 修复前后结果混用
- 算术不一致
- incomplete seeds 被写成完整结论

先生成报告，再修改结果文档；没有原始证据的数字不凭推断覆盖。

#### LOCAL-4：同步正式文档

只有 LOCAL-3 和远端取证完成后才更新三份正式文档：

- `PROJECT_STATUS.md`
- `RESULTS_CN.md`
- `RESULTS_EN.md`

所有正式表格注明 n、标准差定义、config 或 exp folder，以及 paired/unpaired protocol。

#### Codex 完成标准

- 所有正文 headline results 都能追溯到 manifest 行或明确标成 doc-only。
- 三份正式文档不再存在同一实验的不同数字。
- 表格和统计输出可以由命令重建。
- 详细审计文档记录每个冲突的处理依据和对应 commit。

## 明确不做

- 不启动新的 Transformer、CNN 或 Sequential MNIST 训练。
- 不把 fixed-op deployment 设为投稿门槛。
- 不做 empirical-distribution distillation 或 operator transfer matrix。
- 不增加数据集、模型或 CUDA 优化。
- 不在证据审计完成前大规模清理仓库或改写论文 story。
