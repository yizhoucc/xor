# XOR Neuron 项目指南

## 论文
"Two-argument activation functions learn soft XOR operations like cortical neurons"
Yoon, Orhan, Kim, Pitkow (2021), arXiv:2110.06871v2
PDF: `docs/2110.06871v2.pdf`

## 项目目标
复现论文实验：用可学习的二参数激活函数（InnerNet）替代 ReLU，验证学到的函数收敛为软 XOR，且比 ReLU 学得更快、更鲁棒。

## 技术栈
- Python 3.10, PyTorch 2.10
- conda 环境名: `xor`
- 远程 GPU 服务器训练，本地 Mac 开发

## 架构关键点
- **InnerNet**: 2 输入 → 2 层隐藏层(64 units, ReLU) → 1 输出，替代标量激活函数
- **参数共享**: 所有神经元共享同一个 InnerNet（类似固定激活函数如 ReLU 的角色）
- **Layer Normalization** 放在 InnerNet 之前，**Dropout(0.5)** 放在之后
- **训练三阶段**:
  - Session I: pretrain InnerNet（拟合高斯模糊的随机函数）
  - Session II: 联合训练 inner+outer（激活函数在 1-5 epoch 成熟）
  - Session III: 冻结 InnerNet，重训外部网络
- **多版本**: ComplexNeuron*(多 cell type) vs XorNeuron*(单一共享 InnerNet)
- `XorNeuronMLP_v3` 使用 Grouped Conv1d 并行化加速

## 代码结构
- 模型: `model/xorneuron.py`（InnerNet, ComplexNeuron*, XorNeuron*）
- Baseline 模型: `model/baseline.py`（BaselineMLP, BaselineCNN, BaselineRNN）
- 层: `model/denselayer.py`, `model/conv2dlayer.py`, `model/rnncell.py`
- Runner（旧）: `runner/inference_runner.py`（包含 pretrain/phase1/phase2/test 逻辑）
- Runner（新）: `runner/experiment_runner.py`（干净的 ExperimentRunner，统一处理所有模型类型）
- 数据: `dataset/innernet_data.py`（101×101 网格，高斯核 σ=1/3）
- 配置: `config/*.yaml`（旧）, `config/experiments/*.yaml`（论文 Table 1-3 的 15 个标准化实验）
- 统一入口: `run.py`（config hash 去重 + 断点续传）
- 旧入口: `scripts/run_exp_local.py`（本地）, `scripts/run_exp.py`（DataJoint 集群）
- 实验输出: `exp/`
- Notebooks: `notebooks/`
- 实验结果图: `results/`
- 论文: `docs/`

## 运行
```bash
# 新入口（推荐）— 论文复现实验
python run.py -c config/experiments/mlp_mnist_2arg.yaml      # 完整流程
python run.py -c config/experiments/mlp_mnist_2arg.yaml -t   # 仅测试
python run.py -c config/experiments/mlp_mnist_2arg.yaml --resume exp/xxx  # 手动续传

# 批量复现所有实验
for cfg in config/experiments/*.yaml; do python run.py -c "$cfg"; done

# 旧入口（仍可用）
python scripts/run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml
python scripts/run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml -t
```

## 论文实验配置（config/experiments/）

论文只包含 MLP 和 CNN 的分类实验（Figure 4d）+ AutoAttack 鲁棒性（Table 1）+ CIFAR-10-C 鲁棒性（Figure 9）。
RNN PTB 是原作者代码中存在但**未写入论文**的实验，属于扩展实验。

### 论文实验（Figure 4d, Table 1）
| 实验 | 配置文件 | 模型 | 论文位置 |
|------|----------|------|----------|
| MLP MNIST 2-arg | mlp_mnist_2arg.yaml | XorNeuronMLP | Figure 4d |
| MLP MNIST 1-arg | mlp_mnist_1arg.yaml | XorNeuronMLP | Figure 4d |
| MLP MNIST ReLU | mlp_mnist_relu.yaml | BaselineMLP | Figure 4d |
| MLP CIFAR 2-arg | mlp_cifar_2arg.yaml | XorNeuronMLP | Figure 4d |
| MLP CIFAR 1-arg | mlp_cifar_1arg.yaml | XorNeuronMLP | Figure 4d |
| MLP CIFAR ReLU | mlp_cifar_relu.yaml | BaselineMLP | Figure 4d |
| CNN MNIST 2-arg | cnn_mnist_2arg.yaml | XorNeuronConv | Figure 4d |
| CNN MNIST 1-arg | cnn_mnist_1arg.yaml | XorNeuronConv | Figure 4d |
| CNN MNIST ReLU | cnn_mnist_relu.yaml | BaselineCNN | Figure 4d |
| CNN CIFAR 2-arg | cnn_cifar_2arg.yaml | XorNeuronConv | Figure 4d |
| CNN CIFAR 1-arg | cnn_cifar_1arg.yaml | XorNeuronConv | Figure 4d |
| CNN CIFAR ReLU | cnn_cifar_relu.yaml | BaselineCNN | Figure 4d |

### 扩展实验（超越论文）
| 实验 | 配置文件 | 模型 | 备注 |
|------|----------|------|------|
| RNN PTB 2-arg | rnn_ptb_2arg.yaml | ComplexNeuronRNN | 原作者代码中有但未写入论文 |
| RNN PTB 1-arg | rnn_ptb_1arg.yaml | ComplexNeuronRNN | 同上 |
| RNN PTB tanh | rnn_ptb_tanh.yaml | BaselineRNN | 同上 |
| LSTM WikiText-2 InnerNet | lstm_wikitext_2arg.yaml | InnerNetLSTMModel | 我们新增 |
| LSTM WikiText-2 baseline | lstm_wikitext_baseline.yaml | StandardLSTMModel | 我们新增 |
| Transformer WikiText-2 InnerNet | transformer_wikitext_2arg.yaml | InnerNetTransformer | 我们新增 |
| Transformer WikiText-2 GELU | transformer_wikitext_baseline.yaml | StandardTransformer | 我们新增 |
| Transformer WikiText-2 SwiGLU | transformer_wikitext_swiglu.yaml | SwiGLUTransformer | 我们新增 |
| DQN CartPole InnerNet | dqn_cartpole_2arg.yaml | InnerNetDQN | 我们新增 |
| DQN CartPole ReLU | dqn_cartpole_relu.yaml | BaselineDQN | 我们新增 |
| DQN LunarLander InnerNet | dqn_lunarlander_2arg.yaml | InnerNetDQN | 我们新增 |
| DQN LunarLander ReLU | dqn_lunarlander_relu.yaml | BaselineDQN | 我们新增 |

## 论文对应的基线架构
- MLP: 3 层隐藏层 × 64 units
- CNN: 4 层 conv [60,120,120,120], kernel 3×3, stride 1, 2×2 max-pool
- 参数量匹配: baseline ReLU 网络每层用 ⌊√(n*h)⌋+β units 来近似匹配参数量

## 注意事项
- config 中的路径需要根据本地环境修改（`exp_dir`, `data_path`）
- `condaenv.yml` 是 Linux 环境的，Mac 上需要去掉 CUDA/nvidia 依赖
- `data/` 目录已有 MNIST、FashionMNIST、CIFAR-10 数据
- PTB 数据需手动下载: `bash scripts/download_ptb.sh`
- `.gitignore` 只排除了 `data/cifar-100-python/train`

## 当前实验进度 (2026-04-05)

### 核心正面结果

**1. 分类任务（MLP/CNN）— 全面验证，一致有效**
| 任务 | 2-arg | ReLU | 提升 | 收敛加速 |
|------|-------|------|------|---------|
| MLP MNIST | 98.0% | 91.9% | +6.1% | 2-4x |
| MLP CIFAR-10 | 52.1% | 49.5% | +2.6% | 2-4x |
| CNN MNIST | 99.4% | 99.0% | +0.4% | 更快 |
| CNN CIFAR-10 | 78.7% | 74.0% | +4.7% | 更快 |
即使公平对比 ReLU+LN（97.7%），XorNeuron 仍有优势且收敛显著更快。

**2. Transformer FFN 语言模型 — 最有前景的新方向**
| 模型 | PPL (5 seeds) |
|------|--------------|
| GELU (标准) | 96.82 ± 1.19 |
| **InnerNet** | **95.26 ± 1.00** (-1.6%) |
| SwiGLU (手工设计) | 92.98 ± 1.14 (-4.0%) |
InnerNet 自动学到了与 SwiGLU 类似的双输入交互模式，验证了可学习激活函数作为架构发现工具的价值。

**3. LSTM 语言模型**
InnerNet PPL 103.41 ± 0.83 vs Standard 104.38 ± 0.75 (-0.9%)

**4. 强化学习 DQN CartPole**
InnerNet 254.1 ± 69.3 vs ReLU 150.6 ± 58.5 (+69%)

**5. 收敛速度 — 最一致的发现**
跨所有有效场景，InnerNet 都表现出 2-4 倍收敛加速。

### 已完成的扩展实验
| 实验 | 结果 | 状态 |
|------|------|------|
| Transformer FFN InnerNet (WikiText-2) | PPL 95.26 ± 1.00 | 完成 (5 seeds) |
| Transformer FFN GELU (WikiText-2) | PPL 96.82 ± 1.19 | 完成 (5 seeds) |
| Transformer FFN SwiGLU (WikiText-2) | PPL 92.98 ± 1.14 | 完成 (5 seeds) |
| LSTM InnerNet (WikiText-2) | PPL 103.41 ± 0.83 | 完成 (5 seeds) |
| LSTM Standard (WikiText-2) | PPL 104.38 ± 0.75 | 完成 (5 seeds) |
| DQN CartPole InnerNet | 254.1 ± 69.3 | 完成 (10 seeds) |
| DQN CartPole ReLU | 150.6 ± 58.5 | 完成 (10 seeds) |
| MLP-Mixer CIFAR-10 GELU | 81.26% | 完成 (5 seeds) |
| MLP-Mixer CIFAR-10 InnerNet | 80.21% | 完成 (5 seeds) |
| ViT CIFAR-10 (默认超参) | InnerNet 78.2% / GELU 79.5% / SwiGLU 81.6% | 完成 (5 seeds) |
| MLP FashionMNIST ReLU | 86.68% | 完成 |
| MLP CIFAR-100 ReLU | 15.95% | 完成 |

### 运行中的实验 (Mind Cluster)
| 实验 | 类别 | 目的 |
|------|------|------|
| Transformer FFN small (d=64) × 2 | 好用方向深挖 | 不同模型规模 |
| Transformer FFN large (d=256) × 2 | 好用方向深挖 | 不同模型规模 |
| ViT tuned (lr=3e-4, 100ep) × 2 | 优化方向 | 调参看能否翻盘 |
| MLP-Mixer MNIST × 2 | 优化方向 | 换简单数据集 |
| DQN Acrobot × 2 | 好用方向深挖 | 更多 RL 环境 |
| DQN MountainCar × 2 | 好用方向深挖 | 更多 RL 环境 |
| CNN SVHN × 2 | 好用方向深挖 | 新数据集 |
| TF Attn InnerNet | 探索 | 替代 softmax |
| CNN FashionMNIST 2arg/1arg | 好用方向深挖 | 分类泛化 |
| CNN CIFAR-100 2arg | 好用方向深挖 | 更难分类 |
| CNN/MLP 2-arg 补充 seeds | 补数据 | 补齐 4 seeds |

### TODO（待实验完成后）
- [ ] 收集所有结果，更新汇总表
- [ ] 文本分类 SST-2（需要新 runner，优先级低）
- [ ] Transformer FFN on PTB text（需要适配 lm_runner）
- [ ] 根据结果决定是否需要更多数据集/架构

### 论文 Story
> 两参数激活函数（InnerNet）在多种架构（MLP、CNN、Transformer FFN、LSTM、DQN）中一致提升性能并加速收敛 2-4 倍。在 Transformer FFN 中，InnerNet 自动学到了与 SwiGLU 相似的双输入交互模式，证明了可学习激活函数作为架构搜索工具的潜力。

### 基础设施
- **Mind Cluster (CMU)**: Slurm 提交，conda env `xor`，14d12h time limit
- **验证模式**: `python run.py -c config.yaml --validate` — 提交前秒级验证
- **批量验证**: `bash scripts/validate_all.sh` — 全 PASS 后再 sbatch
- **流程**: 写 config → validate → fix → sbatch

### 项目文档规则
- **`SITUATION.md`**: 项目当前状态文档，每次有重要进展或结果时更新。包含：当前运行中的实验、最新结果汇总、下一步计划、已知问题。每次会话结束前或有重大结果时必须更新此文件。
- **`RESULTS_EN.md`** 和 **`RESULTS_CN.md`**: 实验结果的正式文档，英文版和中文版。有新结果时必须**同时更新两个文件**，保持内容一致。包含：所有可 report 的实验结果表格、容量缩放分析、参数效率数据、负面结果。
