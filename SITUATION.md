# 项目状态 — 2026-04-06

## 集群运行中

| 类别 | Jobs | 状态 |
|------|------|------|
| 容量缩放 (MLP CIFAR-10, w=32/64/128/256/512 × 2arg/relu × 5 seeds) | 50 | running/pending |
| 回归 Housing (2arg/relu × 5 seeds) | 10 | running/pending |
| AutoAttack eval (CNN CIFAR 2arg + 1arg) | 2 | pending |
| CIFAR-10-C eval (CNN CIFAR 2arg + 1arg) | 2 | pending |
| CNN CIFAR ReLU 重训 (给 robustness eval 用) | 1 | pending |
| Transformer large InnerNet (d=256) 续跑 | 1 | running |
| **已写好代码待提交** | | |
| ResNet CIFAR-10 (2arg/relu × 5 seeds) | 10 | 待提交 |
| Dense Embedding 文本 (SST-2/AGNews × 2arg/relu × 5 seeds) | 20 | 待提交 |
| Autoencoder MNIST (2arg/relu × 5 seeds) | 10 | 待提交 |

## 已完成实验结果汇总

### 核心结论
InnerNet 在**容量受限的前馈网络**中最有价值。baseline 越弱，增益越大。在已有充分特征交互（self-attention）或输入结构不适合 pairwise 配对的场景下无效。

### MLP 图像分类 (5 seeds)
| 任务 | 2-arg | ReLU | 提升 |
|------|-------|------|------|
| MNIST | 97.97 ± 0.11 | 91.95 ± 3.03 | **+6.03** |
| CIFAR-10 | 51.86 ± 0.19 | 49.53 ± 0.22 | **+2.33** |
| FashionMNIST | 88.53 | 86.68 | **+1.85** |
| CIFAR-100 | 18.20 | 15.95 | **+2.25** |

### CNN 图像分类
| 任务 | 2-arg | ReLU | 提升 |
|------|-------|------|------|
| MNIST | 99.41 ± 0.04 | 99.02 ± 0.03 | **+0.39** |
| CIFAR-10 | 78.29 ± 0.54 | 73.99 ± 0.49 | **+4.30** |
| FashionMNIST | 90.87 | 89.46 | **+1.41** |
| CIFAR-100 | 34.65 | 29.70 | **+4.95** |
| SVHN | 95.01 | 92.63 | **+2.38** |

### 非图像分类 (5 seeds)
| 任务 | 2-arg | ReLU | 差异 |
|------|-------|------|------|
| Adult (表格) | 84.72 ± 0.07 | 84.69 ± 0.07 | +0.03 (持平) |
| Wine (表格) | **63.44 ± 0.52** | 60.56 ± 1.60 | **+2.88** |
| SST-2 (TF-IDF 文本) | 80.00 ± 1.24 | **81.31 ± 0.35** | -1.31 |
| AG News (TF-IDF 文本) | 90.86 ± 0.18 | **91.08 ± 0.05** | -0.22 |
| ECG200 (时间序列) | 80.80 ± 5.78 | 82.20 ± 3.76 | -1.40 |
| Speech Commands (音频) | **16.44 ± 0.52** | 15.52 ± 0.18 | +0.92 (都很低) |

### Transformer 语言模型 (PPL↓)
| 配置 | InnerNet | Baseline | 差异 |
|------|----------|----------|------|
| FFN d=128 | **95.26 ± 1.00** | GELU 96.82 ± 1.19 | **-1.6%** |
| FFN d=64 | **112.66 ± 0.66** | 116.63 ± 0.84 | **-3.4%** |
| FFN d=256 | ⏳ running | 86.05 ± 0.97 | - |
| SwiGLU d=128 | - | **92.98 ± 1.14** | 手工最优 |
| Attn InnerNet | 103.21 ± 1.40 | 96.82 | 失败 |

### LSTM (PPL↓)
InnerNet 105.30 ± 0.31 vs Standard 108.39 ± 0.75 → **-2.9%**

### ViT / MLP-Mixer (InnerNet 略逊)
| 模型 | InnerNet | GELU | SwiGLU |
|------|----------|------|--------|
| ViT CIFAR | 78.18 | 79.46 | **81.54** |
| ViT tuned | 80.30 | **81.06** | - |
| Mixer CIFAR | 80.21 | **81.26** | - |
| Mixer MNIST | **99.10** | 99.01 | - |

### DQN 强化学习
| 环境 | InnerNet | ReLU | SwiGLU |
|------|----------|------|--------|
| CartPole | **228.3** | 149.7 | 107.3 |
| Acrobot | -167.2 | -191.7 | **-127.3** |
| MountainCar | -155.4 | -169.6 | **-153.7** |
| LunarLander | -28.1 | **153.3** | -175.6 |

## 已知问题
- `cnn_ci_2arg_s42`: TITAN X OOM (CUDA Graphs 显存不足)
- Speech Commands 准确率极低 (~16%)，MLP 可能不适合此任务
- LunarLander: InnerNet 和 SwiGLU 都差于 ReLU，pairwise pairing 可能打乱状态结构

## 下一步
1. 等容量缩放和 housing 实验完成 → 验证"小模型增益大"假说
2. 等 robustness eval 完成 → 补齐论文 Table 1 和 Figure 9
3. 排队跑完后提交 ResNet、Dense Embedding、Autoencoder 实验
4. tf_2arg_large 续跑完成后更新 Transformer 规模结果
