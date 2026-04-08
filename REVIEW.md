# 批判性审稿意见 — 漏洞、问题与 TODO

作为 paper reviewer / grant reviewer 的角度审查当前结果。

---

## 🔴 Critical Issues（必须解决）

### C1. 统计不足：多个实验只有 1–3 seeds
| 实验 | 当前 seeds | 问题 |
|------|-----------|------|
| CNN FashionMNIST | 1 | 无法判断显著性 |
| CNN SVHN | 1 | 同上 |
| CNN CIFAR-100 big 2arg | 2 | 不够 |
| AE FashionMNIST | 3 | 偏少 |
| AE CIFAR-10 | 3 | 偏少 |
| STL-10 | 2 | 不够 |

**TODO**: 所有可 report 的实验必须补齐 5 seeds。

### C2. 参数不公平：CNN InnerNet 用 60% 参数就赢了——但对比不公平
当前声称 "InnerNet 用更少参数更好"。但 reviewer 会问：**如果 ReLU 也用 60% 的参数呢？** 也就是说，需要在**相同参数量**下对比：
- InnerNet [60,120,120,120] (127K params)
- ReLU [60,120,120,120] (212K params) ← 当前对比
- ReLU [??,??,??,??] (~127K params) ← 缺失！

CNN 容量缩放实验部分覆盖了这个问题（s0.5 ReLU 67.83% vs s1 InnerNet 78.74%），但需要更精确的参数匹配对比。

**TODO**: 创建一个参数量与 InnerNet 完全匹配的 ReLU CNN config（约 127K params），直接对比。

### C3. Autoencoder 参数不匹配
InnerNetAE 有 659K params，BaselineAE 只有 440K（1.5×）。-43% MSE 的改进可能部分来自更多参数，而不是更好的激活函数。

**TODO**: 创建一个 ~660K params 的 BaselineAE（加宽隐藏层），公平对比。

### C4. 缺少 1-arg 系统对比
论文的 key claim 是 **2-arg** 比 1-arg 好（pairwise interaction matters）。我们有 1-arg 数据但没有系统报告。

**TODO**: 在所有 CNN 实验中加入 1-arg 结果，展示 2-arg > 1-arg > ReLU 的层级关系。

### C5. 缺少 LayerNorm ablation
InnerNet pipeline 包含 LayerNorm → InnerNet → Dropout。ReLU baseline 没有 LayerNorm。改进可能来自 LayerNorm 而不是 InnerNet。

我们有 ReLU+LN 的实验但没有系统报告。

**TODO**: 在结果表中加入 ReLU+LN 列，证明 InnerNet 的改进不仅来自 LayerNorm。

---

## 🟡 Major Concerns（应该解决）

### M1. 没有训练曲线 / 收敛速度分析
论文强调 "InnerNet 收敛快 2-4 倍"，但我们只报告最终准确率，没有展示训练过程中的收敛速度对比。

**TODO**: 从 train_stats 文件中提取 epoch vs accuracy 曲线，对比收敛速度。至少对 CNN CIFAR-10 和 Transformer 画出来。

### M2. 没有可视化学到的激活函数
论文核心 claim 是 InnerNet "learns soft XOR"。我们应该可视化学到的 2D 激活函数表面，验证它确实是 XOR-like。

**TODO**: 加载训练好的 InnerNet 权重，画 101×101 网格上的 2D 响应面。对比不同任务学到的函数是否都是 XOR。

### M3. CNN 容量缩放在小 scale 反转
s0.25 InnerNet 49.61% < ReLU 53.07%，s0.5 InnerNet 62.43% < ReLU 67.83%。这说明 InnerNet 在小模型上**更差**，与 "参数效率" 的 narrative 矛盾。

原因是 CNN 的 InnerNet 会把 channel 数减半（pairwise），小模型的有效 channel 太少了。但 reviewer 会用这个攻击我们的参数效率论点。

**TODO**: 在论文中需要解释这个现象。同时考虑用等效 channel 数对比（InnerNet 用 2× channels 输入但 1× 有效输出）。

### M4. 回归结果 inconsistent
3 个回归数据集：1 好 (Housing)、1 中性 (Diabetes)、1 差 (Wine)。Reviewer 会质疑 "why cherry-pick Housing?"

**TODO**: 分析 Wine 为什么差（可能是数据集太小 1599 samples，或者 feature 维度 11 太低）。如果确实不适合就在论文中解释 InnerNet 的适用条件。

### M5. RL 结果不一致
DQN 和 PPO 在同一环境上结果不同（LunarLander: DQN InnerNet 差 → PPO InnerNet 好 → PPO bigger 又差了）。这说明结果对超参数敏感，不可靠。

**TODO**: 如果要 report RL，需要更多环境和更一致的结论。否则降级为 "preliminary" 或不 report。

### M6. 缺少其他可学习激活函数的对比
我们对比了 ReLU、GELU、SwiGLU。但有很多其他可学习/自适应激活函数：
- PReLU (learnable slope)
- Swish / SiLU
- Maxout
- KAN (Kolmogorov-Arnold Networks)

Reviewer 会问：InnerNet 比这些都好吗？

**TODO**: 至少加 PReLU 和 Swish 作为 baseline 在 CNN CIFAR-10 上对比。

---

## 🟢 Minor Issues（建议解决）

### m1. ResNet baseline 偏低
ResNet-18 CIFAR-10 应该 ~93%，我们只有 86%。可能是训练不够（150 epochs）或者超参数需要调。

**TODO**: 增加 epoch 到 200，添加 data augmentation（RandomCrop, HorizontalFlip），看看能否达到 90%+。

### m2. 论文 Figure 4d 复现对比
我们有数据但没有与论文原始数字逐项对比。应该做一个表：paper 报告的数 vs 我们复现的数。

**TODO**: 制作 paper vs reproduction 对比表。

### m3. Transformer 只有 WikiText-2
所有 Transformer 实验都在 WikiText-2 上。PTB 在跑但还没结果。需要至少 2 个 LM dataset。

### m4. 没有 computational cost 分析
InnerNet 增加了推理时间和训练时间。reviewer 会问 "额外的计算开销是多少？"

**TODO**: 测量 InnerNet vs ReLU 的 FLOPs 和 wall-clock training time。

### m5. 缺少 statistical significance tests
只报告了 mean±std，没有 p-value。

**TODO**: 对关键结果跑 paired t-test 或 Wilcoxon test，报告 p-value。

---

## 优先级排序

### 最高优先级（blocking for paper submission）
1. **C1** 补齐 5 seeds
2. **C2** 参数公平对比（CNN）
3. **C3** AE 参数匹配
4. **C5** ReLU+LN ablation
5. **M1** 训练曲线

### 高优先级（strengthens the paper significantly）
6. **C4** 1-arg 系统对比
7. **M2** XOR 可视化
8. **M6** PReLU/Swish baseline
9. **m5** 显著性检验

### 中优先级（nice to have）
10. **M3** CNN 小 scale 反转解释
11. **M4** 回归 inconsistency 解释
12. **m1** ResNet baseline 提升
13. **m2** 论文复现对比表
14. **m4** 计算开销分析

### 低优先级（可以不做）
15. **M5** RL inconsistency（降级为 preliminary）
16. **m3** 更多 LM dataset（PTB 在跑）
