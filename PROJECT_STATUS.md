# 项目状态 — 2026-04-08

## 集群运行中

| 实验 | Jobs | 状态 |
|------|------|------|
| C1: 补齐 5 seeds (CNN FMNIST/SVHN, AE FMNIST/CIFAR, STL-10) | 34 | pending |
| C2: CNN 参数公平对比 (ReLU ~127K params) | 5 | pending |
| C3: AE 参数匹配 (ReLU ~660K params) | 5 | pending |
| C4: 1-arg 系统对比 (SVHN + FMNIST seeds) | 9 | pending |
| C5: ReLU+LN ablation (FMNIST/SVHN/MNIST/CIFAR) | 18 | pending |
| GPT Transformer 2arg (d=256) | 1 | running |
| Transformer d=192 InnerNet | 1 | running |
| GPT Transformer gelu/swiglu | 2 | running |
| tf_2arg_large (d=256 续跑) | 1 | running |
| PTB Transformer (2arg + gelu) | 2 | pending |
| CNN scale s2 2arg | 1 | running |

---

## TODO — 按优先级

### 🔴 Critical（blocking paper submission）

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| C1 | 补齐 5 seeds | ⏳ 已提交 | FashionMNIST/SVHN/STL-10/AE/CIFAR-100 big |
| C2 | CNN 参数公平对比 | ⏳ 已提交 | ReLU [36,72,72,72] ~127K params vs InnerNet 127K |
| C3 | AE 参数匹配 | ⏳ 已提交 | BaselineAE [384,96] ~660K params vs InnerNetAE 660K |
| C4 | 1-arg 系统对比 | ⏳ 已提交 | SVHN 1-arg 新增，FMNIST 1-arg 补 seeds |
| C5 | ReLU+LN ablation | ⏳ 已提交 | 4 个数据集 × 5 seeds |
| M1 | 训练曲线 | ⏳ 脚本运行中 | 从 log 提取 epoch vs acc 曲线 |

### 🟡 Major

| # | 项目 | 状态 |
|---|------|------|
| M2 | XOR 可视化（学到的 2D 激活函数表面） | TODO |
| M3 | CNN 小 scale 反转解释 | TODO（需要在论文中讨论） |
| M4 | 回归 inconsistency 解释 | TODO |
| M5 | RL inconsistency（降级 preliminary） | TODO |
| M6 | PReLU/Swish baseline 对比 | TODO |
| M7 | CIFAR-100 合理 baseline: ResNet-18+aug (目标76%+) | ⏳ 已提交 |
| M8 | 备选: VGG-16+BN on CIFAR-100 (如 ResNet 不够) | TODO |
| M9 | 备选: WideResNet-28-10 on CIFAR-100 (如需更高 baseline) | TODO |

### 🟢 Minor

| # | 项目 | 状态 |
|---|------|------|
| m1 | ResNet baseline 提升（data augmentation, 200ep） | TODO |
| m2 | 论文原始数字复现对比表 | TODO |
| m3 | 更多 LM dataset | ⏳ PTB 在跑 |
| m4 | 计算开销分析（FLOPs + wall-clock） | TODO |
| m5 | 显著性检验 p-value | TODO |
| 24 | 参数效率分析出图 | 数据已有 |

---

## 当前确认的可 Report 结果

### InnerNet 有效（baseline 合理）
- CNN 图像分类: MNIST/CIFAR-10/FashionMNIST/SVHN/CIFAR-100 (+0.4~4.3%)
- Autoencoder: MNIST/FashionMNIST/CIFAR-10 (-12~43% MSE)
- Transformer FFN: WikiText-2 d=64/128 (-1.6~3.4% PPL)
- LSTM: WikiText-2 (-2.9% PPL)
- 回归: Housing (-5% MSE)
- Big MLP MNIST (+0.46%)
- PPO Acrobot (+5.6%)

### InnerNet 无效
- ResNet (skip connection 消除优势)
- 文本分类 (TF-IDF/embedding 持平)
- Transformer attention 替换 (更差)
- Wine 回归 (更差)
- CNN 极小 scale (更差)

### 参数效率
- CNN: InnerNet 用 60% 参数就赢
- MLP 容量缩放: InnerNet w=128 ≈ ReLU w=256 (节省 55%)
- AE: InnerNet 全 latent dim 都更好

---

## 不再追踪

| 实验 | 原因 |
|------|------|
| MLP CIFAR-100 (16%) | baseline 太低 |
| Speech Commands MLP (16%) | MLP 不适合音频 |
| ECG200 (82%, std=5.8) | 数据集太小 |
| PPO MountainCar (-200) | 全部失败 |
| Cora 图分类 / S4/Mamba | 改动太大 |
