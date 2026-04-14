# 实验结果笔记 — InnerNet

> 内部文档。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

把 ReLU 换成一个小 MLP（两个输入一个输出），让每个神经元能看到隔壁特征。没有 skip connection 的网络有效果，参数还少 40%。

确认有效的：CNN、AE。LSTM 看数据集。Transformer 发现了 param sharing bug，在重跑。

---

## 1. CNN 图像分类（5 seeds）✅ 代码正确

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

参数公平对比：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

## 2. 自编码器 ✅ 代码正确

| 数据集 | InnerNet | ReLU | ReLU 参数匹配 | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | — | -12% |
| CIFAR-10 | **0.0081** | 0.0105 | — | -23% |

效果最好的。容量缩放 latent=32 改进最大 (-42%)。

Configs: `config/experiments/ae_mnist_2arg.yaml`，exp: `exp/ae_mnist_2arg_*`

## 3. Transformer FFN ⚠️ InnerNet 结果在重跑

| 配置 | GELU | SwiGLU | InnerNet (修复版) |
|------|------|--------|-----------------|
| Wiki d=64 | 116.63 | 112.31 | ⏳ |
| Wiki d=128 | 96.82 | 92.98 | ⏳ |
| Wiki d=192 | 89.11 | 85.43 | ⏳ |
| Wiki d=256 | 86.05 | ⏳ | ⏳ |
| PTB d=128 | 212.28 | 205.82 | ⏳ |
| Classic Wiki | — | — | ⏳ |
| Classic PTB | — | — | ⏳ |

之前的 InnerNet 结果有 param sharing bug（每层各一个 InnerNet，没有共享）。已修复，在重跑。GELU 和 SwiGLU 不受影响。

### GPT (d=256)

| 模型 | PPL |
|------|-----|
| GELU | **72.54** |
| SwiGLU | 75.30 |
| InnerNet | ⏳ 重跑 |

### MLM 掩码预测（BERT 式）

| 模型 | PPL |
|------|-----|
| SwiGLU | **93.83** |
| GELU | 101.39 |
| InnerNet | ⏳ 重跑 |

### ViT / MLP-Mixer CIFAR-10

ViT: SwiGLU 81.54% > GELU 79.46%。InnerNet 需要重跑。
Mixer: GELU 81.26%。InnerNet 需要重跑。

Configs: `config/experiments/transformer_wikitext_*.yaml`

## 4. LSTM ✅ 代码正确

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

不知道怎么回事，WikiText-2 上 InnerNet 好用，PTB 上不好用。在用更多数据集研究。

Configs: `lstm_wikitext_classic.yaml`, `lstm_ptb_classic.yaml`, `lstm_wikitext103_*.yaml`, `lstm_cnndm_*.yaml`

## 5. 训练阶段消融 ✅ 代码正确

| 任务 | 3-phase | End-to-end | 差异 |
|------|---------|-----------|------|
| AE MNIST | 0.0039 | 0.0039 | 没差 |
| ResNet CIFAR-10 | 86.09% | 86.00% | 没差 |
| CNN CIFAR-10 | 78.57% | 77.35% | -1.2% |
| MLP MNIST | 98.0% | 97.95% | 没差 |

pretrain 不是必须的。CNN 稍差一点。

Configs: `config/experiments/*_e2e.yaml`

## 6. 参数效率 ✅ 代码正确

InnerNet w=128 (415K) ≈ ReLU w=256 (921K) → **省 55% 参数**。优势随模型增大递增。

## 7. ResNet ⚠️ InnerNet 在重跑

ReLU baseline 不受影响：CIFAR-10 86.33, CIFAR-100 57.95, CIFAR-100+aug 73.51。

修复版初步结果：CIFAR-10 ~86.2%（4/5 done，持平），CIFAR-100+aug 72.72%（1/5 done，还是差）。

internal-only（只换 block 内部 ReLU）也在重跑。

## 8. PPO RL ✅ 代码正确

| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| Acrobot | **-75.3** | -79.8 | -81.7 |
| LunarLander 30s | **187.6** | 158.8 | -249.7 |

30 seeds 后 InnerNet 赢了 LunarLander。

## 9. 不好用的

| 实验 | 情况 |
|------|------|
| ResNet | skip 做了同样的事，修复版初步确认还是没用 |
| LSTM PTB | Standard 赢 |
| VGG SwiGLU | 训练不了 |
| CNN ×0.25 | 太小了 |

## 在跑的

- ⚠️ **U20 param sharing 修复版**：TF 6 jobs（在跑，快出结果），ResNet full+internal 20 jobs（部分在跑），MLM 1 job，GPT 1 job
- LSTM Wiki-103 classic/2arg 在跑
- LSTM CNN/DM 在跑

## 总结

确认有效的：**CNN (+0.4~4.6%)，AE (-43%)，LSTM WikiText-2 (-6.2%)，参数省 55%，PPO LunarLander (+18%)**。

等结果的：Transformer FFN（最关键，param sharing 修复版在跑），ResNet internal-only。

LSTM 看数据集。pretrain 不是必须的。
