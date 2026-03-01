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
├── model/
│   ├── xorneuron.py        # 核心模型：InnerNet, ComplexNeuron*, XorNeuron*
│   ├── denselayer.py        # 支持复杂神经元的全连接层 (多 cell type)
│   ├── conv2dlayer.py       # 支持复杂神经元的卷积层
│   └── rnncell.py           # 支持复杂神经元的 RNN cell
├── runner/
│   └── inference_runner.py  # 训练/推理 Runner (pretrain, phase1, phase2, test)
├── dataset/
│   └── innernet_data.py     # InnerNet 预训练数据：101×101 网格，高斯滤波随机函数
├── config/                  # YAML 配置文件
├── utils/                   # 工具函数 (arg_helper, train_helper, logger 等)
├── data/                    # 数据集 (MNIST, CIFAR-10, FashionMNIST)
├── exp/                     # 实验结果和模型 checkpoint
├── run_exp_local.py         # 本地运行入口
├── run_exp.py               # DataJoint 集群运行入口
├── 2110.06871v2.pdf         # 论文原文
└── *.ipynb                  # 实验 notebook 和可视化
```

## 运行

### 环境

```bash
conda env create -f condaenv.yml
conda activate xor
```

### 本地训练

```bash
python run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml
```

### 测试

```bash
python run_exp_local.py -c config/xor_neuron_mlp_mnist.yaml -t
```

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
