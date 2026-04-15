# 实验结果笔记 — InnerNet

> 内部文档。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

把 ReLU 换成一个小 MLP（两个输入一个输出），让每个神经元能看到隔壁特征。没有 skip connection 的网络有效果，参数还少 40%。

确认有效的：CNN、AE、Transformer FFN（部分出来了）、LSTM WikiText-2、PPO。
不好用的：ResNet（skip connection）、LSTM PTB。

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

参数公平对比：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

## 2. 自编码器

| 数据集 | InnerNet | ReLU | ReLU 参数匹配 | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | — | -12% |
| CIFAR-10 | **0.0081** | 0.0105 | — | -23% |

效果最好的。容量缩放 latent=32 改进最大 (-42%)。

Configs: `config/experiments/ae_mnist_2arg.yaml`，exp: `exp/ae_mnist_2arg_*`

## 3. Transformer FFN

| 配置 | GELU | SwiGLU | InnerNet | vs GELU |
|------|------|--------|----------|---------|
| Wiki d=64 | 116.63 | 112.31 | **112.83** | **-3.3%** |
| Wiki d=128 | 96.82 | 92.98 | **95.23** | **-1.6%** |
| Wiki d=192 | 89.11 | 85.43 | ⏳ | — |
| Wiki d=256 | 86.05 | ⏳ | ⏳ | — |
| PTB d=128 | 212.28 | 205.82 | **207.91** | **-2.1%** |

InnerNet 一直赢 GELU。InnerNet 一直赢 GELU。模型越大优势越小（-3.3% → -1.6% → -0.8%）。d=64 时 InnerNet 和 SwiGLU 差不多。SwiGLU 一直比 InnerNet 好，尤其是大模型。

SwiGLU 是 InnerNet 的子集（InnerNet 理论上能学成 SwiGLU 一样），但实际打不过。在做 warm-start 实验（U21）：先训好 SwiGLU → InnerNet 拟合 SwiGLU → 替换 → 继续训练，看能不能超过 SwiGLU。如果能 → 之前是优化问题；如果不能 → SwiGLU 可能就是最优的。

Config: `scripts/swiglu_warmstart.py`，exp: `exp/warmstart_d128`

### MLM 掩码预测（BERT 式）

| 模型 | PPL |
|------|-----|
| SwiGLU | **93.83** |
| GELU | 101.39 |
| InnerNet | 124.82 |

InnerNet 比 GELU 差。掩码预测上 InnerNet 没用。

### GPT (d=256)

| 模型 | PPL |
|------|-----|
| GELU | **72.54** |
| SwiGLU | 75.30 |
| InnerNet | ⏳ |

Configs: `config/experiments/transformer_wikitext_*.yaml`

## 4. LSTM

### WikiText-2 好用

| 变体 | PPL |
|------|-----|
| **Classic** | **101.72** |
| Semantic | 105.30 |
| Standard | 108.39 |

### PTB 不好用

| 变体 | PPL |
|------|-----|
| **Standard** | **183.02** |
| Classic | 186.54 |
| Semantic | 187.52 |

### WikiText-103（部分）

| 变体 | Best PPL |
|------|---------|
| Standard | **63.82** |
| Classic | ⏳ |
| 2arg | ⏳ |

### CNN/DailyMail
⏳ 在跑

WikiText-2 上好用，PTB 上不好用。在用更多数据集研究。

Configs: `lstm_wikitext_classic.yaml`, `lstm_ptb_classic.yaml`, `lstm_wikitext103_*.yaml`, `lstm_cnndm_*.yaml`

## 5. 训练阶段消融

| 任务 | 3-phase | End-to-end | 差异 |
|------|---------|-----------|------|
| AE MNIST | 0.0039 | 0.0039 | 没差 |
| ResNet CIFAR-10 | 86.09% | 86.00% | 没差 |
| CNN CIFAR-10 | 78.57% | 77.35% | -1.2% |
| MLP MNIST | 98.0% | 97.95% | 没差 |

pretrain 不是必须的。以后默认 end-to-end。

Configs: `config/experiments/*_e2e.yaml`

## 6. 参数效率

InnerNet w=128 (415K) ≈ ReLU w=256 (921K) → **省 55% 参数**。

## 7. ResNet

全换没用（CIFAR-10 86.10% vs ReLU 86.33%，CIFAR-100+aug 73.00% vs 73.51%）。

**Internal-only（只换 block 内部）有效果**：

| 数据集 | Internal InnerNet | ReLU | 全换 |
|--------|------------------|------|------|
| CIFAR-10 | **87.7%** (2/5 done) | 86.33% | 86.10% |
| CIFAR-100+aug | **74.97%** (5 seeds) | 73.51% | 73.00% |

只换没 skip 保护的位置就有效果（+1.4~1.5%），全换反而没用。CIFAR-10 还有 3 seed OOM 重提交了。

Configs: `config/experiments/resnet_cifar_internal_2arg.yaml`

## 8. PPO RL

| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| Acrobot | **-75.3** | -79.8 | -81.7 |
| LunarLander 30s | **187.6** | 158.8 | -249.7 |

30 seeds 后 InnerNet 赢了 LunarLander。

## 9. 不好用的

| 实验 | 情况 |
|------|------|
| ResNet 全换 | skip connection 下没用（但 internal-only 有效果） |
| MLM InnerNet | 比 GELU 差（124.82 vs 101.39） |
| LSTM PTB | Standard 赢 |
| CNN ×0.25 | 太小了 |

## 在跑的

- TF d=256 在跑
- ResNet C10 internal 3 seed OOM 重提交了
- GPT InnerNet 在跑
- LSTM Wiki-103/CNN-DM 拆成单 seed 并行跑了
- U21 SwiGLU warm-start InnerNet (d=128) 在跑

## 总结

有效的：**CNN (+0.4~4.6%)，AE (-43%)，TF FFN d=64 (-3.3%) d=128 (-1.6%) PTB (-2.1%)，LSTM WikiText-2 (-6.2%)，ResNet internal-only (+1.5%)，参数省 55%，PPO LunarLander (+18%)**。

没用的：ResNet 全换（持平），MLM InnerNet（差），LSTM PTB（差）。

关键发现：InnerNet 放在没 skip 保护的位置有效果。ResNet 全换没用但 internal-only 赢 +1.5%。
