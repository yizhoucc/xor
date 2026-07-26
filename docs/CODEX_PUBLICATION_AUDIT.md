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
| 2 | 盘点本地和集群结果源 | 待开始 | source inventory、缺失清单 |
| 3 | 建立 canonical manifest | 待开始 | 机器可读结果清单和冲突报告 |
| 4 | 修正 paired/unpaired 统计工具 | 进行中 | 脚本、测试、示例输出 |
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

## 提交记录

| Commit | 分支 | 内容 | 验证 |
|---|---|---|---|
| `e4d4e9a` | `main` | Codex 开始前工作区快照 | push 成功，工作区干净 |
| `cee495c` | `codex/publication-audit` | 建立审计范围和详细记录 | push 成功，工作区干净 |
