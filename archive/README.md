# Archive — 废弃/被取代的实验

这些实验已完成但不再使用。结果要么不佳、要么被更好的版本取代。文件保留供参考。

## 归档内容

### 1. Transformer Attention 替换（失败）
- `transformer_wikitext_attn_2arg.yaml` — InnerNet 替换 softmax，PPL +6.6%
- 结论：2→1 MLP 无法替代 softmax 的概率归一化结构
- 注：`model/transformer.py` 中的 `InnerNetAttention` 和 `InnerNetAttnTransformer` 类仍在源文件中（标记为 deprecated）

### 2. DQN 强化学习（改用 PPO）
- 12 个 config（CartPole/LunarLander/Acrobot/MountainCar × 2arg/relu/swiglu）
- `models/dqn.py` — DQN 模型代码
- `runners/rl_runner.py` — DQN runner
- 结论：DQN 太旧，结果不稳定。PPO 结果已取代（见 `runner/ppo_runner.py`）

### 3. 低精度图像/文本/音频实验
| 实验 | 精度 | 废弃原因 |
|------|------|---------|
| MLP CIFAR-100 | ~16% | baseline 太低，无意义 |
| MLP Speech Commands | ~16% | MLP 不适合音频任务 |
| ECG200 | 82±5.8% | 数据集太小（200 样本），方差过大 |
| CNN STL-10 | 57% vs 59% | InnerNet 输给 ReLU |
| MLP SST-2 / AG News (TF-IDF + emb) | 中性 | 稀疏/结构化特征不适合 pairwise 配对 |
| MLP Wine 分类 + 回归 | 中性/更差 | 低维表格数据，开销不合算 |
| MLP Adult 分类 | 中性 | 同上 |

### 4. 旧版小 CNN CIFAR-100
- `cnn_cifar100_2arg.yaml`, `cnn_cifar100_relu.yaml`
- 被 `cnn_cifar100_big_*.yaml`（更大通道数）取代

## 目录结构
```
archive/
  configs/    — 37 个 YAML 配置文件
  models/     — dqn.py
  runners/    — rl_runner.py
  exp/        — ~118 个实验输出目录
```

归档日期：2026-04-11
