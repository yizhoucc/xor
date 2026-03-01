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
- 层: `model/denselayer.py`, `model/conv2dlayer.py`, `model/rnncell.py`
- Runner: `runner/inference_runner.py`（很大，包含 pretrain/phase1/phase2/test 逻辑）
- 数据: `dataset/innernet_data.py`（101×101 网格，高斯核 σ=1/3）
- 配置: `config/*.yaml`
- 入口: `scripts/run_exp_local.py`（本地）, `scripts/run_exp.py`（DataJoint 集群）
- 实验输出: `exp/`
- Notebooks: `notebooks/`
- 实验结果图: `results/`
- 论文: `docs/`

## 运行
```bash
# 本地训练（完整流程：pretrain → phase1 → phase2 → test）
python scripts/run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml

# 仅测试
python scripts/run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml -t
```

## 论文对应的基线架构
- MLP: 3 层隐藏层 × 64 units
- CNN: 4 层 conv [60,120,120,120], kernel 3×3, stride 1, 2×2 max-pool
- 参数量匹配: baseline ReLU 网络每层用 ⌊√(n*h)⌋+β units 来近似匹配参数量

## 注意事项
- config 中的路径需要根据本地环境修改（`exp_dir`, `data_path`）
- `condaenv.yml` 是 Linux 环境的，Mac 上需要去掉 CUDA/nvidia 依赖
- `data/` 目录已有 MNIST、FashionMNIST、CIFAR-10 数据
- `.gitignore` 只排除了 `data/cifar-100-python/train`
