# 新实验 TODO

跟踪非图像分类 + SwiGLU RL 实验的开发和部署进度。

## 进度状态说明
- [ ] 未开始
- [x] 已完成

---

## 1. SwiGLU DQN (RL)
> 目的: 对比 InnerNet vs SwiGLU 在 RL 任务上的表现

- [x] 写代码 (SwiGLUDQN model + rl_runner 支持)
- [x] 创建 config (CartPole, Acrobot, MountainCar, LunarLander)
- [ ] validate 通过
- [x] 提交 cluster (job 384496-384499)

## 2. 文本分类 — SST-2 (情感分析)
> 目的: 验证 InnerNet 在 NLP 分类任务上的效果。SST-2 是 GLUE benchmark 标准。
> 架构: MLP / small Transformer encoder on 预训练 embeddings

- [ ] 数据加载 (dataset/)
- [ ] 模型 (model/)
- [ ] Runner 或复用 experiment_runner
- [ ] Config: sst2_2arg.yaml, sst2_relu.yaml, sst2_swiglu.yaml
- [ ] 本地 validate
- [ ] 提交 cluster

## 3. 文本分类 — AG News (新闻4分类)
> 目的: 更大规模文本分类，4类。
> 架构: 同 SST-2

- [ ] 数据加载
- [ ] Config: agnews_2arg.yaml, agnews_relu.yaml, agnews_swiglu.yaml
- [ ] 本地 validate
- [ ] 提交 cluster

## 4. 表格数据 — UCI Adult (收入预测)
> 目的: MLP 在表格数据上仍是强 baseline，正好匹配论文 MLP 架构。
> 架构: MLP (复用现有 XorNeuronMLP / BaselineMLP)

- [ ] 数据加载 (混合特征预处理: one-hot + 标准化)
- [ ] Config: adult_2arg.yaml, adult_relu.yaml, adult_swiglu.yaml
- [ ] 本地 validate
- [ ] 提交 cluster

## 5. 表格数据 — UCI Wine Quality
> 目的: 小规模表格回归/分类。
> 架构: MLP

- [ ] 数据加载
- [ ] Config: wine_2arg.yaml, wine_relu.yaml
- [ ] 本地 validate
- [ ] 提交 cluster

## 6. 音频分类 — Speech Commands v2
> 目的: 1秒音频片段 → 35类关键词分类。
> 架构: CNN (on mel-spectrogram) 或 MLP

- [ ] 数据加载 (torchaudio, mel-spectrogram 提取)
- [ ] 模型适配
- [ ] Config: speechcmd_2arg.yaml, speechcmd_relu.yaml
- [ ] 本地 validate
- [ ] 提交 cluster

## 7. 时间序列 — UCR Archive (选一个代表性数据集)
> 目的: 1D 时间序列分类。
> 架构: 1D-CNN 或 MLP

- [ ] 选择数据集 (如 ECG200 或 FordA)
- [ ] 数据加载
- [ ] Config
- [ ] 本地 validate
- [ ] 提交 cluster

## 8. 图分类 — Cora (节点分类)
> 目的: 图结构数据上的分类。
> 架构: 需要 GNN，复杂度较高，优先级低。

- [ ] 评估可行性
- [ ] 数据加载
- [ ] 模型
- [ ] Config
- [ ] 提交 cluster

---

## 优先级排序
1. **SwiGLU DQN** — 已提交 ✅
2. **UCI Adult** — 最快上手，复用现有 MLP，只需数据加载
3. **SST-2** — NLP 标准，价值高
4. **AG News** — 复用 SST-2 基础设施
5. **Wine Quality** — 小数据集，快
6. **Speech Commands** — 需要音频处理
7. **UCR 时间序列** — 需要 1D 处理
8. **Cora 图分类** — 最复杂，优先级低
