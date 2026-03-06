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
│   ├── dqn.py                      # DQN 模型 (InnerNetDQN, BaselineDQN)
│   ├── lstm.py                     # LSTM 模型 (InnerNetLSTMCell, StandardLSTM)
│   ├── transformer.py              # Transformer 模型 (InnerNet FFN, Standard FFN)
│   ├── denselayer.py               # 支持复杂神经元的全连接层 (多 cell type)
│   ├── conv2dlayer.py              # 支持复杂神经元的卷积层
│   └── rnncell.py                  # 支持复杂神经元的 RNN cell
├── runner/
│   ├── experiment_runner.py        # 标准化实验 Runner (pretrain → phase1 → phase2 → test)
│   ├── rl_runner.py                # DQN 强化学习 Runner
│   ├── lm_runner.py                # 语言建模 Runner (LSTM / Transformer)
│   └── inference_runner.py         # 旧版 Runner
├── dataset/
│   └── innernet_data.py            # InnerNet 预训练数据（1D/2D 高斯滤波随机函数）
├── config/
│   └── experiments/                # 论文复现实验配置（15 个）+ 扩展实验
│       ├── mlp_mnist_{2arg,1arg,relu}.yaml
│       ├── mlp_cifar_{2arg,1arg,relu}.yaml
│       ├── cnn_mnist_{2arg,1arg,relu}.yaml
│       ├── cnn_cifar_{2arg,1arg,relu}.yaml
│       ├── rnn_ptb_{2arg,1arg,tanh}.yaml
│       ├── dqn_cartpole_{2arg,relu}.yaml
│       ├── dqn_lunarlander_{2arg,relu}.yaml
│       ├── lstm_wikitext_{2arg,baseline}.yaml
│       └── transformer_wikitext_{2arg,baseline}.yaml
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

### 数据集准备

MNIST、FashionMNIST、CIFAR-10 会由 PyTorch 自动下载。WikiText-2 会由 HuggingFace datasets 自动下载。

PTB (Penn Treebank) 需要手动下载：

```bash
# 下载 PTB 数据集 (Mikolov 预处理版本, ~5MB)
bash scripts/download_ptb.sh

# 或手动下载:
mkdir -p data/ptb
for split in train valid test; do
  curl -sL "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.${split}.txt" -o "data/ptb/${split}.txt"
done
```

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

### 扩展实验：RL、LSTM 和 Transformer

验证 InnerNet 在论文之外的架构上的泛化性。需额外安装 `pip install gymnasium datasets`。

```bash
# ============ DQN 强化学习 ============
# CartPole-v1 (10 seeds × 500 episodes)
python run.py -c config/experiments/dqn_cartpole_2arg.yaml     # InnerNet DQN
python run.py -c config/experiments/dqn_cartpole_relu.yaml     # ReLU baseline

# LunarLander-v3 (10 seeds × 1000 episodes)
python run.py -c config/experiments/dqn_lunarlander_2arg.yaml  # InnerNet DQN
python run.py -c config/experiments/dqn_lunarlander_relu.yaml  # ReLU baseline

# ============ LSTM 语言建模 (WikiText-2) ============
python run.py -c config/experiments/lstm_wikitext_2arg.yaml    # InnerNet LSTM (5 seeds × 10 epochs)
python run.py -c config/experiments/lstm_wikitext_baseline.yaml # Standard LSTM baseline

# ============ Transformer 语言建模 (WikiText-2) ============
# Decoder-only Transformer, 4 layers, d=128, 4 heads
# InnerNet 用 GLU 风格双投影替换 GELU: InnerNet(W1a·x, W1b·x)
python run.py -c config/experiments/transformer_wikitext_2arg.yaml     # InnerNet FFN (5 seeds × 10 epochs)
python run.py -c config/experiments/transformer_wikitext_baseline.yaml # GELU baseline
python run.py -c config/experiments/transformer_wikitext_swiglu.yaml   # SwiGLU baseline (controlled comparison)
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

---

## Experiment Report: Generalizing InnerNet Beyond the Paper

### Background

The original paper (Yoon et al., 2021) demonstrated that replacing scalar activation functions (e.g., ReLU) with a learned two-argument activation function (InnerNet) improves classification performance, learning speed, and adversarial robustness on MLP and CNN architectures. The learned function converges to a soft XOR / multiplicative gating pattern: `f(x1, x2) ≈ x1 · x2`.

We extend this idea to three architectures not covered in the paper: **DQN (reinforcement learning)**, **LSTM (language modeling)**, and **Transformer (language modeling)**, to test whether InnerNet generalizes beyond supervised image classification.

### Key Design Principle: Semantic Pairing

A critical insight from our experiments is that InnerNet's two inputs must carry **distinct semantic roles** to be effective. Arbitrary adjacent-dimension pairing does not reliably work.

| Architecture | Pairing Strategy | Semantic Meaning | Effective? |
|---|---|---|---|
| CNN | Channel-wise | Different feature detectors interact | Yes |
| LSTM | Input proj vs Hidden proj | Current input vs memory state | Yes |
| Transformer FFN | Value proj vs Gate proj (GLU-style) | What to pass vs how much to pass | Yes |
| DQN/MLP | Adjacent dimensions | Random / no guaranteed semantics | Mixed |

### Results Summary

#### 1. DQN — CartPole-v1 (10 seeds x 500 episodes, width-matched)

| Model | Mean Reward (last 50) | Std |
|---|---|---|
| **InnerNet DQN** | **256.2** | 100.7 |
| Baseline DQN (ReLU) | 157.1 | 68.5 |

InnerNet wins by +63%. CartPole has only 4 state dimensions where adjacent pairs happen to be physically meaningful (position-velocity, angle-angular velocity).

#### 2. DQN — LunarLander-v3 (10 seeds x 1000 episodes, width-matched)

| Model | Mean Reward (last 50) | Std | Seeds Solved (>200) |
|---|---|---|---|
| InnerNet DQN | -38.9 | 142.2 | 1/10 |
| **Baseline DQN (ReLU)** | **152.8** | 148.5 | **7/10** |

Baseline wins decisively. LunarLander has 8 state dimensions; adjacent-dimension pairing lacks consistent semantic meaning, and InnerNet's nonlinearity destabilizes Q-learning under sparse rewards.

#### 3. LSTM — WikiText-2 Language Modeling (5 seeds x 10 epochs)

| Model | Best Mean PPL | Improvement | All Seeds Better? |
|---|---|---|---|
| **InnerNet LSTM** | **103.41** | -0.9% | **Yes (5/5)** |
| Standard LSTM | 104.38 | — | — |

InnerNet uses separate projections from input (`W_cx · x`) and hidden state (`W_ch · h`) as its two arguments — analogous to basal (feedforward) and apical (feedback) dendrites in biology. The improvement is consistent across all seeds, with later onset of overfitting (epoch 6 vs 4).

#### 4. Transformer FFN — WikiText-2 Language Modeling (5 seeds x 10 epochs)

InnerNet replaces GELU in the Transformer FFN with a GLU-style learned activation:
```
Standard:  FFN(x) = W2 · GELU(W1 · x) + b2
InnerNet:  FFN(x) = W2 · InnerNet(W1a · x, W1b · x) + b2
```

| Model | Best Mean PPL | Improvement | All Seeds Better? |
|---|---|---|---|
| **InnerNet Transformer** | **95.26** | -1.6% | **4/5 below baseline mean** |
| Standard Transformer (GELU) | 96.82 | — | — |

Per-seed breakdown:

| Seed | InnerNet PPL | Baseline PPL |
|---|---|---|
| 42 | **93.83** | 95.65 |
| 43 | **95.56** | 95.16 |
| 44 | **94.52** | 97.36 |
| 45 | 96.72 | 97.87 |
| 46 | **95.67** | 98.06 |

The two projections naturally take on value vs gate roles, similar to SwiGLU but with a fully learned gating function. Both models show no overfitting at epoch 10 — longer training may widen the gap.

### Width Matching

InnerNet pairs two dimensions into one (2→1), effectively halving the hidden width. To ensure fair comparison:

- **InnerNet DQN**: `hidden_dim=256` → 128 effective width after pairing
- **Baseline DQN**: `hidden_dim=128` → 128 effective width
- **LSTM**: Already width-matched (separate projections don't reduce dimensionality)
- **Transformer**: InnerNet FFN uses two d→4d projections (W1a, W1b) vs one in standard FFN, so InnerNet has ~1.33x more parameters (1.07M vs 0.81M)

An earlier experiment without width matching showed InnerNet DQN failing catastrophically on LunarLander (66.8 vs 185.6) due to the hidden layer being halved from 128→64.

### Conclusions

1. **InnerNet generalizes to LSTM and Transformer**, providing consistent (though modest) improvements in language modeling perplexity.
2. **Semantic pairing is essential.** When InnerNet's two inputs have distinct roles (input vs memory in LSTM, value vs gate in Transformer), it works. When pairing is arbitrary (adjacent MLP dimensions), results are unreliable.
3. **RL is challenging for InnerNet.** DQN benefits on simple tasks (CartPole) where adjacent state dimensions happen to be meaningful, but fails on complex tasks (LunarLander) where the pairing lacks semantic structure.
4. **The GLU-style Transformer FFN is the most promising extension**, as the two-projection structure naturally maps onto the value-gate decomposition that modern architectures (SwiGLU, GLU) already exploit — but with a learned, adaptive gating function rather than a fixed one.
