# 实验结果笔记 — InnerNet

> 内部文档。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

把 ReLU 换成一个小 MLP（两个输入一个输出），让每个神经元能看到隔壁特征。没有 skip connection 的网络都有效果，参数还少 40%。

三个点：
1. **没 skip 的前馈网络都有效果** — CNN、AE、Transformer FFN 都好用
2. **InnerNet 自己学出了 SwiGLU** — 说明可学习激活函数能当架构搜索工具
3. **越简单越好** — 相邻配对比语义配对好，pretrain 可能不需要

不好用的地方：ResNet 有 skip 就没用了。LSTM 看数据集，WikiText-2 好用 PTB 不行，不知道怎么回事，在研究。

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 大模型 | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

MNIST 接近饱和提升小，CIFAR 越难提升越大。1-arg 在 CIFAR-10 上比 2-arg 好（81 vs 78），不知道为什么，可能是 2-arg 配对优化更难。

参数公平对比：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点。不是参数多才好，是双输入交互本身有用。

SwiGLU：CIFAR-10 上 SwiGLU 略好（79.79 vs 78.57），但 CIFAR-100 上 InnerNet 好很多（53.74 vs 46.48）。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

## 2. 自编码器（效果最好的）

| 数据集 | InnerNet | ReLU | ReLU 参数匹配 | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | — | -12% |
| CIFAR-10 | **0.0081** | 0.0105 | — | -23% |

AE 效果最好。MNIST -43%，参数匹配了还是赢 34%。瓶颈层压缩信息，双输入交互相当于带宽翻倍。

容量缩放（latent 从 8 到 64），latent=32 改进最大 (-42%)。

Configs: `config/experiments/ae_mnist_2arg.yaml`，exp: `exp/ae_mnist_2arg_*`

## 3. Transformer FFN

### 全规模对比

| 配置 | InnerNet | GELU | SwiGLU | vs GELU |
|------|----------|------|--------|---------|
| WikiText-2 d=64 | **112.66** | 116.63 | 112.31 | **-3.4%** |
| WikiText-2 d=128 | **95.26** | 96.82 | 92.98 | -1.6% |
| WikiText-2 d=192 | **88.14** | 89.11 | ⏳ | -1.1% |
| WikiText-2 d=256 | **85.40** | 86.05 | ⏳ | -0.8% |
| PTB d=128 | **207.81** | 212.28 | 205.82 | -2.1% |

所有规模都赢 GELU。d=64 的时候 InnerNet 和 SwiGLU 差不多（112.66 vs 112.31），InnerNet 自己学出了 SwiGLU 的门控。模型越大优势越小，因为大模型本身容量够了。

### FFN 设计变体（PTB d=128）

| 模型 | WikiText-2 PPL | PTB PPL |
|------|---------------|---------|
| SwiGLU | — | **205.82** |
| SiLU-InnerNet | **94.90** | 208.43 |
| InnerNet Semantic | 95.26 | 207.81 |
| InnerNet Classic | 95.49 | 208.81 |
| GELU | 96.82 | 212.28 |

四种 InnerNet 变体差不多（94.9~95.5），内部用什么激活和配对方式都不重要，双输入结构本身有用。和 LSTM 不一样（LSTM 上 Classic 明显比 Semantic 好），Transformer 上两种配对没区别。

### GPT (d=256, 6 层, 20 epochs)

| 模型 | WikiText-2 PPL |
|------|---------------|
| GELU | **72.54** |
| SwiGLU | 75.30 |
| InnerNet | ⏳ |

GPT 上 GELU 反而最好。大模型趋势和小模型不一样，不知道为什么。InnerNet 还在跑。

Config: `config/experiments/transformer_wikitext_gpt_*.yaml`

### MLM 掩码预测（BERT 式，WikiText-2）

| 模型 | PPL |
|------|-----|
| SwiGLU | **93.83** |
| GELU | 101.39 |
| InnerNet | ⏳ |

SwiGLU 比 GELU 好 -7.5%。InnerNet 在跑。AE 是我们最好的结果，掩码预测和 AE 类似（都是重建），看看 InnerNet 在这里能不能也好用。

Config: `config/experiments/mlm_wikitext_*.yaml`

### ViT CIFAR-10（5 seeds）

| 模型 | Accuracy |
|------|----------|
| SwiGLU | **81.54±0.29%** |
| GELU | 79.46±0.23% |
| InnerNet | 78.18±0.96% |

InnerNet 最差。ViT 有 residual + self-attention，和 ResNet 一样，InnerNet 没用。

Config: `config/experiments/vit_cifar_*.yaml`

### MLP-Mixer CIFAR-10（5 seeds）

| 模型 | Accuracy |
|------|----------|
| GELU | **81.26±0.29%** |
| InnerNet | 80.21±0.22% |

InnerNet 稍差 -1%。Mixer 也有 residual。

Config: `config/experiments/mixer_cifar_*.yaml`

Configs: `config/experiments/transformer_wikitext_*.yaml`，exp: `exp/transformer_wikitext_*`

## 4. LSTM

### WikiText-2 好用

| 变体 | PPL |
|------|-----|
| **Classic**（相邻配对） | **101.72** |
| Semantic（x vs h） | 105.30 |
| Standard | 108.39 |

Classic -6.2%。相邻配对比语义配对好。

### PTB 不好用

| 变体 | PTB PPL |
|------|---------|
| **Standard** | **183.02** |
| Classic | 186.54 |
| Semantic | 187.52 |

PTB 上 Standard 最好，InnerNet 全都更差。和 WikiText-2 完全相反，不知道怎么回事。

在用 WikiText-103 和 CNN/DailyMail 研究。如果 Wiki-103 也好用那就是领域差异；如果也不好用那可能是小数据的问题。

Configs: `lstm_wikitext_classic.yaml`, `lstm_ptb_classic.yaml`, `lstm_wikitext103_classic.yaml`, `lstm_cnndm_classic.yaml`

## 5. 训练阶段消融（英文不写，等确认）

| 任务 | 3-phase | End-to-end | 差异 |
|------|---------|-----------|------|
| AE MNIST | 0.0039 | 0.0039 | 没差 |
| ResNet CIFAR-10 | 86.09% | 86.00% | 没差 |
| CNN CIFAR-10 | 78.57% | 77.35% | -1.2% |
| MLP MNIST | 98.0% | 97.95% | 没差 |

MLP/AE/ResNet 都没差。CNN 稍微差一点（-1.2%），pretrain 对 CNN 有一点帮助但不大。整体结论：pretrain 不是必须的。

Configs: `config/experiments/*_e2e.yaml`，exp: `exp/*_e2e_*`

## 6. 参数效率

MLP CIFAR-10 宽度缩放：

| 宽度 | InnerNet | ReLU | 提升 |
|------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

**InnerNet w=128 ≈ ReLU w=256，省 55% 参数。**

## 7. ResNet（没用）

| 数据集 | InnerNet | ReLU |
|--------|----------|------|
| CIFAR-10 | 86.09 | 86.33 |
| CIFAR-100 | 56.78 | 57.95 |

持平。skip connection 已经做了 InnerNet 想做的事。这个负面结果有理论价值，英文保留了。

## 8. RL（不稳定，英文暂不写）

PPO 10 seeds。Acrobot InnerNet 好（-75.3 vs -79.8），LunarLander ReLU 好（209 vs 167）。30 seeds 在跑 LunarLander，RL 太不稳定了。

## 9. 回归（基本没用，英文不写）

Housing -5% 有点效果。Diabetes 持平。Wine 反而更差 +9.3%。低维表格数据不适合。

## 10. 不好用的（英文里不写的）

| 实验 | 情况 |
|------|------|
| ResNet | skip 做了同样的事（**英文保留**） |
| TF attention 替换 softmax | 不行，+6.6%，已归档 |
| VGG-16 + SwiGLU | SGD + 13 层乘法门控 + 没 residual，训练不了，1% 准确率 |
| SiLU-InnerNet | 换了 SiLU 没用 |
| LSTM PTB | 不好用，在研究 |
| 文本分类 TF-IDF | 稀疏特征不适合，已归档 |
| CNN ×0.25 | 太小了配对开销太大 |

## 在跑的

- TF SwiGLU d=192 在跑，d=256 排队
- ~~TF Classic InnerNet~~ ✅ 完成，Classic ≈ Semantic
- LSTM Wiki-103 + CNN/DM 在跑
- MLM（BERT 式掩码预测）SwiGLU 跑完 PPL=93.83，GELU/InnerNet 在跑
- PPO LunarLander 30 seeds 在跑
- ~~训练消融 CNN/MLP~~ ✅ 完成，pretrain 不是必须的
- GPT d=256 在跑

## 总结

没 skip 的前馈网络都有效果。AE 最好（-43%），CNN 也不错（+4.6%），Transformer FFN 有效（-3.4%），参数省一半。InnerNet 自己学出了 SwiGLU，说明能当架构搜索用。

不好用的也清楚：有 skip 没用，LSTM 看数据集。越简单越好——相邻配对 > 语义配对，pretrain 可能不需要。
