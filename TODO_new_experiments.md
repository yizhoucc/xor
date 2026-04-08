# 实验 TODO

## 🔴 Critical（blocking paper submission）

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| C1 | 补齐 5 seeds | 🔜 | FashionMNIST CNN/SVHN CNN/STL-10/AE FMNIST/AE CIFAR-10/CIFAR-100 big |
| C2 | CNN 参数公平对比 | 🔜 | 创建 ~127K params 的 ReLU CNN，与 InnerNet 127K 直接对比 |
| C3 | AE 参数匹配 | 🔜 | 创建 ~660K params 的 BaselineAE，消除 params 差异 |
| C4 | 1-arg 系统对比 | 🔜 | 在所有 CNN 实验加 1-arg，展示 2-arg > 1-arg > ReLU |
| C5 | ReLU+LN ablation | 🔜 | 加 ReLU+LN 列，证明改进不只来自 LayerNorm |
| M1 | 训练曲线 | 🔜 | 从 train_stats 提取 epoch vs acc 曲线 |

## 🟡 Major

| # | 项目 | 状态 |
|---|------|------|
| M2 | XOR 可视化（学到的 2D 激活函数表面） | TODO |
| M3 | CNN 小 scale 反转解释 | TODO |
| M4 | 回归 inconsistency 解释 | TODO |
| M5 | RL inconsistency（降级 preliminary） | TODO |
| M6 | PReLU/Swish baseline 对比 | TODO |

## 🟢 Minor

| # | 项目 | 状态 |
|---|------|------|
| m1 | ResNet baseline 提升（data augmentation） | TODO |
| m2 | 论文原始数字复现对比表 | TODO |
| m3 | 更多 LM dataset（PTB 在跑） | ⏳ |
| m4 | 计算开销分析（FLOPs + wall-clock） | TODO |
| m5 | 显著性检验 p-value | TODO |

## 之前的 TODO（保留追踪）

| # | 实验 | 状态 |
|---|------|------|
| 24 | 参数效率分析（容量缩放曲线） | 数据已有，需出图 |
| — | GPT Transformer (d=256) | ⏳ running |
| — | Transformer d=192 InnerNet | ⏳ running |
| — | PTB Transformer | ⏳ running |
| — | CNN scale s2 2arg | ⏳ running |
