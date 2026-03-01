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
| 实验 | 配置文件 | 模型 | 论文表格 |
|------|----------|------|----------|
| MLP MNIST 2-arg | mlp_mnist_2arg.yaml | XorNeuronMLP | Table 1 |
| MLP MNIST 1-arg | mlp_mnist_1arg.yaml | XorNeuronMLP | Table 1 |
| MLP MNIST ReLU | mlp_mnist_relu.yaml | BaselineMLP | Table 1 |
| MLP CIFAR 2-arg | mlp_cifar_2arg.yaml | XorNeuronMLP | Table 1 |
| MLP CIFAR 1-arg | mlp_cifar_1arg.yaml | XorNeuronMLP | Table 1 |
| MLP CIFAR ReLU | mlp_cifar_relu.yaml | BaselineMLP | Table 1 |
| CNN MNIST 2-arg | cnn_mnist_2arg.yaml | XorNeuronConv | Table 2 |
| CNN MNIST 1-arg | cnn_mnist_1arg.yaml | XorNeuronConv | Table 2 |
| CNN MNIST ReLU | cnn_mnist_relu.yaml | BaselineCNN | Table 2 |
| CNN CIFAR 2-arg | cnn_cifar_2arg.yaml | XorNeuronConv | Table 2 |
| CNN CIFAR 1-arg | cnn_cifar_1arg.yaml | XorNeuronConv | Table 2 |
| CNN CIFAR ReLU | cnn_cifar_relu.yaml | BaselineCNN | Table 2 |
| RNN PTB 2-arg | rnn_ptb_2arg.yaml | ComplexNeuronRNN | Table 3 |
| RNN PTB 1-arg | rnn_ptb_1arg.yaml | ComplexNeuronRNN | Table 3 |
| RNN PTB tanh | rnn_ptb_tanh.yaml | BaselineRNN | Table 3 |

## 论文对应的基线架构
- MLP: 3 层隐藏层 × 64 units
- CNN: 4 层 conv [60,120,120,120], kernel 3×3, stride 1, 2×2 max-pool
- 参数量匹配: baseline ReLU 网络每层用 ⌊√(n*h)⌋+β units 来近似匹配参数量

## 注意事项
- config 中的路径需要根据本地环境修改（`exp_dir`, `data_path`）
- `condaenv.yml` 是 Linux 环境的，Mac 上需要去掉 CUDA/nvidia 依赖
- `data/` 目录已有 MNIST、FashionMNIST、CIFAR-10 数据
- `.gitignore` 只排除了 `data/cifar-100-python/train`
