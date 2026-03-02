# XOR Neuron

复现论文 **"Two-argument activation functions learn soft XOR operations like cortical neurons"** (Yoon, Orhan, Kim, Pitkow, 2021, arXiv:2110.06871v2) 的实验代码。

## 论文核心思想

生物神经元远比人工神经网络中的单元复杂，具有不同的功能区（如基底树突和顶端树突）之间的非线性交互。本文提出用一个小型"内部网络"（InnerNet）替代传统标量激活函数（如 ReLU），使每个神经元成为一个多输入、单输出的非线性单元。

关键发现：
- 学到的二参数激活函数可靠地收敛为**软 XOR（soft XOR）**模式——类似二次函数 f(x1,x2) = c1*x1^2 + c2*x2^2 + c3*x1*x2 + ...
- 这与近期发现的人类皮层神经元树突能计算 XOR 运算（Gidon et al., 2020）一致
- 使用这些学到的激活函数的网络**学习更快、泛化更好、对扰动更鲁棒**

## 架构

```
外部网络 (Outer Net): 标准 MLP/CNN/RNN
    ↓
每个神经元接收 n 个加权输入（n=2，类比基底/顶端树突）
    ↓
内部网络 (InnerNet): 小型 MLP (2层隐藏层, 64 units, ReLU)
    输入: n 个标量 → 输出: 1 个标量
    所有神经元共享同一个 InnerNet（类似固定激活函数）
```

## 训练流程（三个 Session）

1. **Session I - Pretrain InnerNet**：生成随机平滑激活函数（均匀随机 → 高斯模糊 σ=3），用监督学习训练 InnerNet 拟合它
2. **Session II - 联合训练**：将预训练的 InnerNet 嵌入外部网络，同时训练两者做图像分类。激活函数在 1-5 epoch 内迅速成熟为稳定的空间模式
3. **Session III - 固定 InnerNet 重训外部网络**：冻结学到的激活函数，重新初始化并训练外部网络。验证学到的激活函数作为通用非线性的有效性

## 论文实验

| 实验 | 外部网络 | 数据集 | 关键结果 |
|------|----------|--------|----------|
| 分类性能 | MLP (3层×64), CNN (4层 conv) | MNIST, CIFAR-10 | 2-arg 比 ReLU 学得更快，渐近性能更好 |
| 鲁棒性-自然扰动 | CNN | CIFAR-10-C | mCE=91.3% (vs ReLU baseline 100%) |
| 鲁棒性-对抗攻击 | MLP, CNN | MNIST, CIFAR-10 | AutoAttack 下均优于 ReLU 和 1-arg |
| 谱分析 | MLP, CNN | MNIST, CIFAR-10 | 学到的函数在 l=2 (四极矩) 有更多能量 |

## 模型变体

| 模型 | 说明 | 对应论文 |
|------|------|----------|
| `ComplexNeuronMLP` | 多 cell type 的 MLP，每个 type 独立 InnerNet | 论文扩展：多细胞类型 |
| `ComplexNeuronConv` | 多 cell type 的 CNN | 同上 |
| `ComplexNeuronRNN` | 多 cell type 的 RNN | 同上 |
| `XorNeuronMLP` | 单一共享 InnerNet 的 MLP | 论文原始架构 |
| `XorNeuronMLP_v2` | v2 改进版 | 自定义改进 |
| `XorNeuronMLP_v3` | 用 Grouped Conv1d 加速 InnerNet | 自定义优化 |
| `XorNeuronConv` | 单一共享 InnerNet 的 CNN | 论文原始架构 |

## 项目结构

```
xor/
├── run.py                          # 统一实验入口（推荐）
├── model/
│   ├── xorneuron.py                # 核心模型：InnerNet, ComplexNeuron*, XorNeuron*
│   ├── baseline.py                 # ReLU/tanh baseline 模型
│   ├── denselayer.py               # 支持复杂神经元的全连接层 (多 cell type)
│   ├── conv2dlayer.py              # 支持复杂神经元的卷积层
│   └── rnncell.py                  # 支持复杂神经元的 RNN cell
├── runner/
│   ├── experiment_runner.py        # 标准化实验 Runner (pretrain → phase1 → phase2 → test)
│   └── inference_runner.py         # 旧版 Runner
├── dataset/
│   └── innernet_data.py            # InnerNet 预训练数据（1D/2D 高斯滤波随机函数）
├── config/
│   └── experiments/                # 论文复现实验配置（15 个）
│       ├── mlp_mnist_{2arg,1arg,relu}.yaml
│       ├── mlp_cifar_{2arg,1arg,relu}.yaml
│       ├── cnn_mnist_{2arg,1arg,relu}.yaml
│       ├── cnn_cifar_{2arg,1arg,relu}.yaml
│       └── rnn_ptb_{2arg,1arg,tanh}.yaml
├── utils/                          # 工具函数
├── data/                           # 数据集 (MNIST, CIFAR-10, PTB)
├── exp/                            # 实验输出 (checkpoint, 统计, 标记文件)
├── docs/
│   ├── 2110.06871v2.pdf            # 论文原文
│   └── paper_results.md            # 论文结果摘要与复现对比
├── scripts/                        # 旧版入口脚本
├── notebooks/                      # 可视化 notebook
└── results/                        # 实验结果图
```

## 运行

### 环境

```bash
# 方式一: venv (推荐)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 方式二: conda
conda env create -f condaenv.yml
conda activate xor
```

> GPU 用户安装 PyTorch 时按官方指引选择对应 CUDA 版本: https://pytorch.org/get-started/locally/

### 复现论文实验（推荐）

使用 `run.py` 统一入口，自动执行完整流程：pretrain → phase1 → phase2 → test。
支持 config hash 断点续传（重复运行同一 config 会自动跳过已完成实验）。

```bash
# ============ MLP 实验 ============
# MLP + MNIST
python run.py -c config/experiments/mlp_mnist_2arg.yaml    # 2-arg learned activation
python run.py -c config/experiments/mlp_mnist_1arg.yaml    # 1-arg learned activation
python run.py -c config/experiments/mlp_mnist_relu.yaml    # ReLU baseline

# MLP + CIFAR-10
python run.py -c config/experiments/mlp_cifar_2arg.yaml
python run.py -c config/experiments/mlp_cifar_1arg.yaml
python run.py -c config/experiments/mlp_cifar_relu.yaml

# ============ CNN 实验 ============
# CNN + MNIST
python run.py -c config/experiments/cnn_mnist_2arg.yaml
python run.py -c config/experiments/cnn_mnist_1arg.yaml
python run.py -c config/experiments/cnn_mnist_relu.yaml

# CNN + CIFAR-10
python run.py -c config/experiments/cnn_cifar_2arg.yaml
python run.py -c config/experiments/cnn_cifar_1arg.yaml
python run.py -c config/experiments/cnn_cifar_relu.yaml

# ============ RNN 实验 ============
# RNN + PTB
python run.py -c config/experiments/rnn_ptb_2arg.yaml
python run.py -c config/experiments/rnn_ptb_1arg.yaml
python run.py -c config/experiments/rnn_ptb_tanh.yaml
```

### 批量运行所有实验

```bash
for cfg in config/experiments/*.yaml; do python run.py -c "$cfg"; done
```

### 仅测试（跳过训练）

```bash
python run.py -c config/experiments/mlp_mnist_2arg.yaml -t
```

### 从断点恢复

```bash
# 手动指定实验目录恢复
python run.py -c config/experiments/mlp_mnist_2arg.yaml --resume exp/mlp_mnist_2arg_20260301_145935_59238531
```

### 旧版入口（仍可用）

```bash
python scripts/run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml
```

### 实验输出

每个实验在 `exp/` 下生成独立目录，包含：
- `config.yaml` — 实验配置副本
- `config_hash.txt` — 配置 hash（用于断点续传）
- `model_snapshot_best_*.pth` — 各阶段最优模型
- `train_stats_phase{1,2}.p` — 训练统计
- `test_results.p` — 测试结果
- `PRETRAIN_DONE`, `PHASE1_DONE`, `PHASE2_DONE`, `TEST_DONE`, `COMPLETED` — 阶段标记

## 关键配置参数

- `num_cell_types`: InnerNet 种类数（ComplexNeuron* 模型使用）
- `arg_in_dim`: InnerNet 输入维度 n（arity，论文主要用 n=2）
- `in_hidden_dim`: InnerNet 隐藏层维度（论文用 64）
- `out_hidden_dim`: 外部网络各层维度
- `dropout`: 0.5（放在 InnerNet 之后）

## 参考

```
@article{yoon2021two,
  title={Two-argument activation functions learn soft XOR operations like cortical neurons},
  author={Yoon, Kijung and Orhan, Emin and Kim, Juhyun and Pitkow, Xaq},
  journal={arXiv preprint arXiv:2110.06871},
  year={2021}
}
```
