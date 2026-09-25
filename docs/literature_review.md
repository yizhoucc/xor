# InnerNet 相关工作综述

更新日期：2026-09-25

## 结论

现有文献支持把 InnerNet 定位为“宿主条件化的局部算子探针”。它在连续的二元函数空间中搜索与当前任务和网络状态相容的计算规则，训练后再把稳定结构写成简单公式并优化实现。

“先训练灵活函数，再提取公式”已有明确先例。KAN 使用可学习 spline 并进行 symbolification，Cranmer 等人对神经网络内部模块做 symbolic regression。自动发现激活或局部算子的工作还包括 Swish search、PANGAEA 和 EvoNorm。因此，本文不能声称首次提出 activation discovery 或 discover-then-distill。

本文可以守住四个区别：

1. 搜索对象是网络内部重复调用的共享二元局部算子。
2. 双宿主、跨初始化、冻结与联合训练的 40 条件实验直接识别了 host optimization basin 对最终算子的影响。
3. scratch、warm-start 和 frozen-host 实验把表示能力与优化难度分开。
4. 提炼实验同时测量公式拟合、替换后的任务性能和运行成本。

## 文献地图

### 1. 可学习激活函数

| 论文 | 类别 | 与本项目的关系 |
|---|---|---|
| [Yoon et al., 2022](https://doi.org/10.1109/ACCESS.2022.3178951) | 直接来源 | 提出共享二输入 InnerNet、软 XOR、MLP/CNN 实验和树突动机。二元激活与 XOR 发现不是本文首创。 |
| [He et al., 2015, PReLU](https://doi.org/10.1109/ICCV.2015.123) | 基线 | 只学习负半轴斜率，代表低自由度 trainable activation。 |
| [Molina et al., 2020, Padé Activation Units](https://openreview.net/forum?id=BJlBSkHtDS) | 直接先例 | 用低阶有理函数学习一元激活，兼顾函数自由度与闭式表达。 |
| [Bohra et al., 2020, Spline Activations](https://doi.org/10.1109/OJSP.2020.3039379) | 直接先例 | 用 B-spline 学习自由形态的一元激活，并以正则化控制复杂度。 |
| [Ma et al., 2021, ACON](https://doi.org/10.1109/CVPR46437.2021.00794) | 相邻方法 | 学习非线性与线性状态之间的切换，函数族由设计者预先限定。 |
| [Dai et al., 2020, Attention as Activation](https://arxiv.org/abs/2007.07729) | 相邻方法 | 把局部跨通道信息引入激活，接近 context-aware activation。 |
| [Liu et al., 2025, KAN](https://openreview.net/forum?id=Ozo7qJ5vZi) | 强先例 | 学习网络边上的一元 spline，并在训练后剪枝和符号化。与“先发现后提炼”高度重合。 |

### 2. 自动激活与算子搜索

| 论文 | 类别 | 与本项目的关系 |
|---|---|---|
| [Ramachandran et al., 2017, Searching for Activation Functions](https://arxiv.org/abs/1710.05941) | 直接先例 | 从预定义 primitive 组合中发现 Swish。两个分支来自同一标量输入。 |
| [Bingham and Miikkulainen, 2022, PANGAEA](https://doi.org/10.1016/j.neunet.2022.01.001) | 强先例 | 进化搜索函数图，再学习连续参数，并报告 architecture-specific activation。 |
| [Liu et al., 2020, EvoNorm](https://proceedings.neurips.cc/paper/2020/hash/9d4c03631b8b0c85ae08bf05eda37d0f-Abstract.html) | 强先例 | 从低级操作中搜索归一化与激活联合层，最终得到可直接部署的固定公式。 |
| [Liu et al., 2019, DARTS](https://openreview.net/forum?id=S1eYHoC5FX) | 相邻方法 | 连续松弛候选操作，最终离散化网络结构。搜索空间由预定义操作组成。 |
| [Real et al., 2020, AutoML-Zero](https://proceedings.mlr.press/v119/real20a.html) | 相邻方法 | 从基本数学 primitive 搜索完整学习算法，范围更广、成本更高。 |
| [Lin et al., 2014, Network In Network](https://arxiv.org/abs/1312.4400) | 相邻方法 | 用小型 MLP 替代局部线性滤波，但不研究共享二元标量函数及其表面。 |

### 3. 门控与乘法交互

| 论文 | 类别 | 与本项目的关系 |
|---|---|---|
| [Hochreiter and Schmidhuber, 1997, LSTM](https://doi.org/10.1162/neco.1997.9.8.1735) | 背景 | 用乘法门控制记忆写入、保留与读取。 |
| [Cho et al., 2014, GRU](https://aclanthology.org/D14-1179/) | 背景 | 用较紧凑的门控结构稳定循环状态。 |
| [Dauphin et al., 2017, GLU](https://proceedings.mlr.press/v70/dauphin17a.html) | 直接工程参照 | 两路线性投影通过逐元素门控组合。 |
| [Shazeer, 2020, SwiGLU](https://arxiv.org/abs/2002.05202) | 核心基线 | 系统比较 GLU 变体。恢复 SwiGLU 表面用于验证探针，不能视为发现新算子。 |
| [Jayakumar et al., 2020](https://openreview.net/forum?id=rylnK6VtDH) | 背景 | 将 gating、attention、hypernetwork 和动态卷积统一为乘法交互。 |

### 4. 神经科学与多区室计算

| 论文 | 类别 | 与本项目的关系 |
|---|---|---|
| [Gidon et al., 2020](https://doi.org/10.1126/science.aax6239) | 直接生物学依据 | 人类皮层树突事件可以区分线性不可分输入。正文应写 XOR-like，不写成严格布尔 XOR。 |
| [Poirazi et al., 2003](https://doi.org/10.1016/S0896-6273(03)00149-1) | 背景 | 多个非线性树突子单元使锥体神经元近似两层网络。 |
| [Beniaguev et al., 2021](https://doi.org/10.1016/j.neuron.2021.07.002) | 背景 | 复现主动树突神经元映射需要较深时序网络，支持单神经元计算复杂性。 |
| [Larkum et al., 1999](https://doi.org/10.1038/18686) | 背景 | 不同皮层层次输入通过树突事件发生非线性耦合。 |

### 5. 符号提取与结构迁移

| 论文 | 类别 | 与本项目的关系 |
|---|---|---|
| [Cranmer et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/c9f2f917078bd2db12f23c3b413d9cba-Abstract.html) | 强先例 | 训练神经模型后，对内部模块做 symbolic regression，再用闭式方程替换。 |
| [Sahoo et al., 2018](https://proceedings.mlr.press/v80/sahoo18a.html) | 相邻方法 | 从预定义数学 primitive 中学习稀疏方程，目标是恢复数据生成关系。 |
| [Wei et al., 2016, Network Morphism](https://arxiv.org/abs/1603.01670) | warm-start 背景 | 用保函数变换把成熟网络迁移到新结构，说明好的结构起点可以降低优化难度。 |

## 与强先例的边界

| 强先例 | 不能使用的表述 | 本文可使用的表述 |
|---|---|---|
| Yoon et al. | 首次提出二元激活、首次发现软 XOR | 系统研究二元局部单元在现代架构中的优化边界和结构辨识能力 |
| PANGAEA | 首次发现架构特异激活 | 通过宿主干预直接识别 optimization basin 对最终算子的作用 |
| Swish search、EvoNorm | 首次自动发现神经网络算子 | 无需预定义符号图的低维连续局部探针 |
| KAN | 首次先训练灵活函数再符号化 | 将该原则用于共享二元局部算子，并研究外围网络如何决定其形态 |
| Cranmer et al. | 首次从神经模块提取闭式表达式 | 提取网络中重复调用的局部算子，并测量替换后的任务性能与吞吐 |
| GLU、SwiGLU | 首次发现乘法或门控 | 用已知门控验证探针，再搜索任务和宿主特异的偏离结构 |

## 建议的论文表述

核心 claim：

> InnerNet 能在已经形成有效表示的宿主网络附近，搜索并恢复与该优化区域相容的高性能二元局部算子。训练后的表面可以生成候选公式，并交给独立训练、消融和系统优化验证。

部署 claim：

> 当前三阶多项式实验验证了可提炼性，但没有完成高效部署。现有 Poly3 是未融合实现，融合 kernel 尚未测量。本文报告的是精度与吞吐的真实中间结果，不把它外推为最终速度上限。

神经科学 claim：

> InnerNet 抽象了树突区室对多路输入进行非线性整合这一计算原则。它为生物学启发的结构假设提供可训练、可视化和可干预的实验对象，但不等同于完整的生物神经元模型。

## 检索与核对

书目信息通过论文原文、arXiv、Crossref、OpenAlex、PubMed、ACL Anthology、PMLR、NeurIPS Proceedings 和 OpenReview 核对。没有使用内部公司资料。任何检索都不能证明不存在遗漏先例，因此论文避免使用“首次”或“全局最优”等绝对表述。
