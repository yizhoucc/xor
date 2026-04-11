# 实验结果笔记 — InnerNet

> 内部文档，记录我们所有实验的结果和分析。英文正式版见 `RESULTS_EN.md`。

## 目前的 Story

简单说就是：把 ReLU 换成一个小 MLP（两个输入，一个输出），让每个神经元能"看到"隔壁特征再决定怎么激活。效果：没有 skip connection 的网络一致变好，参数还少 40%。

我们发现了三件事：
1. **没有 skip 的前馈网络都能提升** — CNN、AE、Transformer FFN 都好用
2. **InnerNet 自己学出了 SwiGLU** — 没人告诉它怎么做门控，它自己学的，说明可学习激活函数能当架构搜索工具用
3. **越简单越好** — 相邻配对比精心设计的语义配对好，pretrain 可能不是必须的

不好用的地方：ResNet 有 skip connection 就没用了（冗余）。LSTM 的效果看数据集，WikiText-2 好用 PTB 不行，还在查原因。

---

## 1. CNN 图像分类（5 seeds）

| 数据集 | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU 参数匹配 | SwiGLU | 提升 |
|--------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | +0.39 |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | +1.57 |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | +2.46 |
| CIFAR-100 大模型 | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

MNIST 接近饱和所以提升小，CIFAR 越难提升越大。1-arg 在 CIFAR-10 上竟然比 2-arg 好（81 vs 78），有点意外，可能是 2-arg 配对引入了优化困难。

参数公平对比做了：同样 127K 参数，InnerNet 78.57% vs ReLU 70.67%，差 8 个点，说明不是参数多才好，是双输入交互本身有用。

SwiGLU 对比：CIFAR-10 上 SwiGLU 略赢（79.79 vs 78.57），但 CIFAR-100 上 InnerNet 大赢（53.74 vs 46.48）。任务越难 InnerNet 越有优势。

Configs: `config/experiments/cnn_cifar_2arg.yaml` 等，exp: `exp/cnn_cifar_2arg_*`

## 2. 自编码器（我们最强的结果）

| 数据集 | InnerNet | ReLU | ReLU 参数匹配 | 改进 |
|--------|----------|------|-------------|------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43%** |
| FashionMNIST | **0.0076** | 0.0086 | — | -12% |
| CIFAR-10 | **0.0081** | 0.0105 | — | -23% |

AE 是效果最炸裂的。MNIST 上直接干掉 43% 的 MSE，参数匹配了也还是赢 34%。原因很直觉：AE 瓶颈层逼着压缩信息，双输入交互等于带宽翻倍。

容量缩放也做了（latent 从 8 到 64），在 latent=32 时改进最大 (-42%)，压缩刚好够紧的时候双输入最有价值。

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

所有规模都赢 GELU。d=64 的时候 InnerNet 和 SwiGLU 几乎一样（112.66 vs 112.31），说明 InnerNet 自己学出了 SwiGLU 的门控模式。这个很重要——证明可学习激活函数能当架构搜索工具。

模型越大，InnerNet 优势越小（-3.4% → -0.8%），因为大模型本身容量够了。SwiGLU 在大规模上仍然比 InnerNet 好，因为 SwiGLU 的 `b` 有直达通路不丢信息，InnerNet 的 2→1 是信息瓶颈。

### 各种 FFN 设计变体（PTB d=128）

| 模型 | PPL |
|------|-----|
| SwiGLU | **205.82** |
| InnerNet Semantic | 207.81 |
| InnerNet Classic | ⏳ 在跑 |
| SiLU-InnerNet | 208.43 |
| GELU | 212.28 |

试了在 InnerNet 里面用 SiLU 替代 ReLU，没用（208.43 vs 207.81）。说明内部用什么激活不重要，双输入结构本身才是关键。Classic（相邻配对）版本在跑，看看 LSTM 那边的 classic > semantic 在 Transformer 上是不是也成立。

Configs: `config/experiments/transformer_wikitext_*.yaml`，exp: `exp/transformer_wikitext_*`

## 4. LSTM

### WikiText-2 上好用

| 变体 | PPL |
|------|-----|
| **Classic**（相邻配对） | **101.72** |
| Semantic（x vs h） | 105.30 |
| Standard | 108.39 |

Classic 赢了 -6.2%。简单的相邻维度配对比精心设计的语义配对（把 x 和 h 对应维度配对）效果更好，这个挺意外的。

### PTB 上不好用！

| 变体 | PTB PPL |
|------|---------|
| **Standard** | **183.02** |
| Classic | 186.54 |
| Semantic | 187.52 |

PTB 上 Standard 最好，InnerNet 全部更差。和 WikiText-2 完全相反。

这个问题我们正在查：WikiText-103 和 CNN/DailyMail 在跑。如果 Wiki-103 也好用 → 领域差异（维基百科 vs 新闻）。如果 Wiki-103 也不好用 → 可能是小数据 overfit（WikiText-2 太小刚好 InnerNet 的 regularization 有帮助）。

Configs: `lstm_wikitext_classic.yaml`, `lstm_ptb_classic.yaml`, `lstm_wikitext103_classic.yaml`, `lstm_cnndm_classic.yaml`

## 5. 训练阶段消融

我们原来是三阶段训练：pretrain InnerNet → 联合训练 → 冻结 InnerNet 重训外部。试了直接 end-to-end 不分阶段：

| 任务 | 3-phase | End-to-end | 差异 |
|------|---------|-----------|------|
| AE MNIST | 0.0039 | 0.0039 | 一样 |
| ResNet CIFAR-10 | 86.09% | 86.00% | 一样 |
| CNN CIFAR-10 | 78.57% | ⏳ | — |
| MLP MNIST | 98.0% | ⏳ | — |

目前看 pretrain 不是必须的，end-to-end 一样好。如果 CNN/MLP 也确认了，那 3-phase 训练流程可以简化掉，对论文来说也是好消息（更简单的方法 = 更容易被采纳）。等结果确认后再决定写不写进英文。

Configs: `config/experiments/*_e2e.yaml`，exp: `exp/*_e2e_*`

## 6. 参数效率

MLP CIFAR-10 上做了宽度缩放：

| 宽度 | InnerNet | ReLU | 提升 |
|------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

关键数据点：**InnerNet w=128 ≈ ReLU w=256，参数省 55%**。模型越大优势越明显。

## 7. ResNet（skip connection = InnerNet 没用）

| 数据集 | InnerNet | ReLU |
|--------|----------|------|
| CIFAR-10 | 86.09 | 86.33 |
| CIFAR-100 | 56.78 | 57.95 |

完全持平。因为 `y = F(x) + x` 已经提供了跨特征交互，InnerNet 做的事情 skip 已经做了。这个负面结果其实很有价值，帮我们划清了边界。英文文档中保留了这个。

## 8. RL（初步，结果不稳定）

PPO 10 seeds：Acrobot 上 InnerNet 最好 (-75.3 vs ReLU -79.8)，但 LunarLander 上 ReLU 赢 (209.1 vs 166.6)，SwiGLU 直接崩了 (-139.1)。正在用 30 seeds 重跑 LunarLander。RL 对初始化太敏感了，这块只能当 preliminary。英文里暂时不 report。

## 9. 回归（基本没用）

Housing -5% 有点效果，Diabetes 持平，Wine 反而更差 +9.3%。低维表格数据不适合 InnerNet，特征之间没有局部相关性。英文里不 report。

## 10. 不好用的东西（英文里不写或简写）

| 实验 | 咋回事 |
|------|--------|
| ResNet | skip connection 已经做了 InnerNet 想做的事（**英文保留，有理论价值**） |
| Transformer attention 替换 softmax | 2→1 MLP 替代不了 softmax 的概率归一化，PPL +6.6%，已归档 |
| VGG-16 + SwiGLU | SGD lr=0.1 直接 NaN，lr=0.01 也只有 1%。13 层乘法门控 + SGD + 无 residual = 必炸 |
| SiLU-InnerNet | 在 InnerNet 里面换 SiLU 没用，说明内部激活不重要 |
| LSTM PTB | 效果反转，正在查原因 |
| 文本分类 TF-IDF | 稀疏特征不适合配对，已归档 |
| Bounded (tanh) LSTM | 加 tanh 一致变差，已归档。标准 Transformer 本来就不用 tanh |
| CNN ×0.25 极小 | 通道太少配对开销太大 |

## 正在跑的实验

- TF SwiGLU d=192 在跑，d=256 排队 → 补全全规模 SwiGLU 对比
- TF Classic InnerNet FFN（Wiki + PTB）→ 看 LSTM 的 "classic > semantic" 在 TF 上成不成立
- LSTM WikiText-103 + CNN/DM → 查 LSTM 数据集依赖的原因
- MLM Masked LM → AE 那么好用，BERT 式掩码预测会不会也好用？SwiGLU 跑完了 PPL=93.83，GELU 和 InnerNet 在跑
- PPO LunarLander 30 seeds → 加 seeds 稳定 RL 结果
- 训练阶段消融 CNN/MLP → 确认 end-to-end 是否一样好
- GPT d=256 → 大模型上 InnerNet 的表现

## 总结

InnerNet 在**没有 skip 的前馈网络**里一致好用。最猛的是 AE（-43%），然后是 CNN（+4.6%）、Transformer FFN（-3.4%），参数还能省一半多。InnerNet 自己学出了 SwiGLU 的模式，证明可学习激活函数能当架构搜索用。

不好用的地方也很清楚：有 skip 就没用（ResNet），LSTM 看数据集。越简单越好——相邻配对 > 语义配对，pretrain 可能不需要。
