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
- [x] 提交 cluster (job 384496-384499, 3 running, 1 pending)

## 2. 文本分类 — SST-2 (情感分析)
> TF-IDF (5000 dim) → MLP

- [x] 数据加载 (dataset/tabular.py: load_sst2)
- [x] Config: mlp_sst2_2arg.yaml, mlp_sst2_relu.yaml
- [x] 提交 cluster (job 384533-384534, running)

## 3. 文本分类 — AG News (新闻4分类)
> TF-IDF (5000 dim) → MLP

- [x] 数据加载 (dataset/tabular.py: load_agnews)
- [x] Config: mlp_agnews_2arg.yaml, mlp_agnews_relu.yaml
- [x] 提交 cluster (job 384535-384536, running)

## 4. 表格数据 — UCI Adult (收入预测)
> 14 features → MLP, 二分类

- [x] 数据加载 (dataset/tabular.py: load_adult, auto-download)
- [x] Config: mlp_adult_2arg.yaml, mlp_adult_relu.yaml
- [x] 提交 cluster (job 384532 + 384527, running)

## 5. 表格数据 — UCI Wine Quality
> 11 features → MLP, 6类

- [x] 数据加载 (dataset/tabular.py: load_wine, auto-download)
- [x] Config: mlp_wine_2arg.yaml, mlp_wine_relu.yaml
- [x] 提交 cluster (job 384528-384529, running)

## 6. 音频分类 — Speech Commands v2
> mel-spectrogram (40×32=1280 dim) → MLP, 35类

- [x] 数据加载 (dataset/tabular.py: load_speech_commands)
- [x] Config: mlp_speechcmd_2arg.yaml, mlp_speechcmd_relu.yaml
- [x] 提交 cluster (job 384539-384540, pending)

## 7. 时间序列 — ECG200 (UCR Archive)
> 96-dim 心电图, 二分类

- [x] 数据加载 (dataset/tabular.py: load_ecg, auto-download)
- [x] Config: mlp_ecg_2arg.yaml, mlp_ecg_relu.yaml
- [x] 提交 cluster (job 384537-384538, running)

## 8. 图分类 — Cora (节点分类)
> 需要 GNN，优先级低，暂不实现

- [ ] 评估可行性（需要 PyG 或 DGL，改动大）

---

## 集群状态总览 (2026-04-05 ~18:30)

| 类别 | 实验 | 状态 |
|------|------|------|
| SwiGLU DQN | CartPole/Acrobot/MountainCar | running |
| SwiGLU DQN | LunarLander | pending |
| 文本 | SST-2 2arg/relu | running |
| 文本 | AG News 2arg/relu | running |
| 表格 | Adult 2arg/relu | running |
| 表格 | Wine 2arg/relu | running |
| 音频 | Speech Commands 2arg/relu | pending |
| 时序 | ECG200 2arg/relu | running |
| Transformer 规模 | tf_2arg_small/large | running |
| 补 seeds | mlp/cnn 2arg (11 jobs) | running |
