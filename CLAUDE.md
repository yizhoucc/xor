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

## 当前实验进度 (2026-03-08)

### 论文复现结果（1 seed，论文用 4 seeds）
| 实验 | 我们的结果 | 论文参考 (Figure 4d 目测) | 匹配？ |
|------|-----------|-------------------------|--------|
| MLP MNIST 2-arg | 97.99% | ~98% | 匹配 |
| MLP MNIST 1-arg | 98.35% | ~97.5% | 略高 |
| MLP MNIST ReLU | 91.27% | ~97% | 差距大（见已知问题） |
| MLP CIFAR 2-arg | 52.14% | ~52-53% | 匹配 |
| MLP CIFAR 1-arg | 54.43% | ~50% | 略高 |
| MLP CIFAR ReLU | 49.31% | ~48-49% | 匹配 |
| CNN MNIST 2-arg | 99.40% | ~99% | 匹配 |
| CNN MNIST 1-arg | 99.37% | ~98.8% | 略高 |
| CNN MNIST ReLU | 98.99% | ~98.5% | 匹配 |
| CNN CIFAR 2-arg | 78.68% | ~72-73% | 趋势一致，数值偏高 |
| CNN CIFAR 1-arg | 80.24% | ~70% | 趋势一致，数值偏高 |
| CNN CIFAR ReLU | 73.98% | ~68-69% | 趋势一致，数值偏高 |

### 扩展实验（超越论文）
| 实验 | 结果 | 状态 |
|------|------|------|
| DQN CartPole InnerNet | 254.1 ± 69.3 avg reward | 完成 (10 seeds) |
| DQN CartPole ReLU | 150.6 ± 58.5 avg reward | 完成 (10 seeds) |
| DQN LunarLander InnerNet | -38.9 avg reward | 完成（InnerNet 输） |
| DQN LunarLander ReLU | 152.8 avg reward | 完成 |
| LSTM WikiText-2 InnerNet | PPL 103.41 ± 0.83 | 完成 (5 seeds) |
| LSTM WikiText-2 Standard | PPL 104.38 ± 0.75 | 完成 (5 seeds) |
| Transformer WikiText-2 InnerNet | PPL 95.26 ± 1.00 | 完成 (5 seeds) |
| Transformer WikiText-2 GELU | PPL 96.82 ± 1.19 | 完成 (5 seeds) |
| Transformer WikiText-2 SwiGLU | PPL 92.98 ± 1.14 | 完成 (5 seeds) |
| RNN PTB tanh | PPL 140.13 | 完成 |
| RNN PTB 2-arg | Epoch 5, Val PPL ~240 | 中止（收敛太慢） |
| RNN PTB 1-arg | Epoch 16, Val PPL ~209 | 中止（收敛太慢） |

### 公平对比基线（ReLU + LayerNorm）
原始 Baseline 无 LayerNorm，而 XorNeuron 在 InnerNet 前有 LayerNorm，导致比较不公平。
已新增 4 个 ReLU+LayerNorm 配置用于公平对比：
| 实验 | 配置文件 | 模型 | 备注 |
|------|----------|------|------|
| MLP MNIST ReLU+LN | mlp_mnist_relu_ln.yaml | BaselineMLP | 公平对比 |
| MLP CIFAR ReLU+LN | mlp_cifar_relu_ln.yaml | BaselineMLP | 公平对比 |
| CNN MNIST ReLU+LN | cnn_mnist_relu_ln.yaml | BaselineCNN | 公平对比 |
| CNN CIFAR ReLU+LN | cnn_cifar_relu_ln.yaml | BaselineCNN | 公平对比 |

### 已知问题
1. **Baseline 无 LayerNorm（已修复）**: XorNeuron 模型在 InnerNet 前有 LayerNorm，原 Baseline 完全没有。
   已给 BaselineMLP/BaselineCNN 添加 `use_layernorm` 选项，新增 4 个 `*_relu_ln.yaml` 配置。
   MLP ReLU baseline 在 MNIST 只有 91.27%（正常应 97%+）主要是因为缺少 LN。
   `run_multiseed.sh all` 已包含这 4 个 LN baseline。
2. **1-arg ≥ 2-arg**: 单 seed 下 MLP 的 1-arg 持续优于 2-arg，与论文预期不符。需多 seed 验证。
3. **CNN 数值偏高**: CNN CIFAR 比论文高 5-10%，可能与实现细节差异有关（第4层 kernel=1x1、参数匹配近似）。

### TODO
1. **跑 LayerNorm 基线** — 验证 ReLU+LN 是否接近论文 ~97%（MNIST）
2. **多 seed 实验** — `run.py` 已支持 `--seed` 参数，脚本: `scripts/run_multiseed.sh`（含 LN baselines）
3. **RNN PTB** — 收敛极慢（每 epoch ~30 min），已中止。低优先级。
4. **汇总报告** — 用 `scripts/aggregate_results.py` 聚合多 seed 结果

### 多 seed 运行方式
```bash
# 单个实验 4 seeds
nohup bash scripts/run_multiseed.sh config/experiments/mlp_mnist_2arg.yaml > /dev/null 2>&1 &

# 全部 16 个实验（12 论文 + 4 LN baseline）× 4 seeds
nohup bash scripts/run_multiseed.sh all > /dev/null 2>&1 &

# 查看聚合结果
python scripts/aggregate_results.py
```

### 关键发现
- **InnerNet 在 Transformer FFN 最有前景**: GLU 风格双投影，PPL 95.26 vs GELU 96.82 (-1.6%)
- **SwiGLU > InnerNet > GELU**: PPL 92.98 vs 95.26 vs 96.82。固定乘法门控（SwiGLU）比学到的 InnerNet 更优，
  说明结构化交互在 Transformer FFN 场景下比通用学到的交互更有效。
- **语义配对很重要**: InnerNet 的两个输入需要语义不同（value vs gate）
- **Xaq 反馈**: 注意 InnerNet 与已有乘法交互（attention, LSTM gating）的关系。我们 Transformer 只替换了 FFN（无乘法交互的地方），attention 完全没动
