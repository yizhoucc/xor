# InnerNet 性能优化讨论

## 目前的问题

InnerNet 比 SwiGLU 慢很多。CPU benchmark（B=128, S=64, D=512）：

| 激活函数 | 时间 (ms) | vs ReLU | 参数 |
|---------|----------|---------|------|
| ReLU | 0.41 | 1× | 0 |
| GELU | 2.56 | 6× | 0 |
| SiLU | 2.30 | 6× | 0 |
| a×b | 1.17 | 3× | 0 |
| SwiGLU | 4.19 | 10× | 0 |
| **InnerNet h=8** | **27.35** | **67×** | 33 |
| **InnerNet h=32** | **65.94** | **161×** | 129 |

InnerNet h=32 比 SwiGLU 慢 **16 倍**。主要瓶颈不是参数量（才 129 个），是每个 token 的每个 feature 都要过一个小 MLP。

## 瓶颈分析

输入 (B=128, S=64, D=512)：

1. **reshape**: 4M 个元素 → 4M 个 (a,b) pair
2. **Linear(2, 32)**: 4M × 小矩阵乘法。这不是一个大矩阵乘法，是 4M 个独立的小向量乘法。GPU/CPU 对大矩阵乘法优化得好（cuBLAS），对很多很小的乘法效率低
3. **ReLU**: 4M 个元素
4. **Linear(32, 1)**: 又是 4M 个小乘法
5. **reshape back**: 恢复原形状

根本问题：**InnerNet 把一个 element-wise 操作变成了 4M 个独立的小矩阵乘法**。SwiGLU 的 `SiLU(a) × b` 是纯 element-wise，硬件友好。

## 可能的优化方向

### 1. 用公式替代 MLP（推理时）

训练用 InnerNet，推理时用拟合出来的多项式：
```
f(a,b) ≈ 0.12·a·b + 0.11 - 0.06·b + 0.03·a²·b
```
多项式是 element-wise 操作，和 SwiGLU 一样快。训练时学，推理时固化。

优点：推理零开销
缺点：拟合有误差（MSE=0.003），每个模型/每层需要单独拟合

### 2. 减小 InnerNet（h=8）

h=32 → h=8：快 2.4 倍（27ms vs 66ms），只有 33 个参数。
效果需要实验验证（U35 TODO）。

### 3. CUDA kernel 优化

写自定义 CUDA kernel 把 reshape + Linear + ReLU + Linear + reshape 融合成一个 kernel。避免 4M 次 kernel launch 的开销。

估计能快 5-10×，但需要写 CUDA 代码。

### 4. 向量化 InnerNet

把 InnerNet 改成对整个 feature vector 操作而不是每个 pair 独立：

目前：每个 (a_i, b_i) 独立过 MLP → 4M 次小计算
改成：整个 a, b 向量一起操作 → 用矩阵乘法

```python
# 目前（慢）
pairs = stack([a, b], dim=-1)  # [B,S,D,2]
out = MLP(pairs.reshape(-1, 2))  # 4M 个小计算

# 改成（快）
# 把 InnerNet 的第一层看成是对每个 pair 做同样的 W@[a_i,b_i]+b
# 等价于：W[0]*a + W[1]*b + bias，对整个 tensor 操作
W1 = inner.net[0].weight  # [32, 2]
b1 = inner.net[0].bias    # [32]
W2 = inner.net[2].weight  # [1, 32]
b2 = inner.net[2].bias    # [1]
h = F.relu(a.unsqueeze(-1) * W1[:, 0] + b.unsqueeze(-1) * W1[:, 1] + b1)  # [B,S,D,32]
out = (h @ W2.T).squeeze(-1) + b2  # [B,S,D]
```

这样把 4M 个独立小计算变成了正常的 tensor 操作。应该快很多。

### 5. 直接优化：展开 InnerNet

InnerNet 是 `Linear(2,H) → ReLU → Linear(H,1)`。展开：

```
f(a,b) = W2 @ ReLU(W1[0]*a + W1[1]*b + b1) + b2
```

其中 W1[0], W1[1] 是 H 维向量。这等价于 H 个 "neuron"：

```
f(a,b) = sum_j w2_j * max(0, w1_j0 * a + w1_j1 * b + b1_j) + b2
```

这是 H 个 ReLU 的加权和。每个 ReLU 定义了 2D 空间中的一条线，整个函数是分段线性的。

可以预计算每个 ReLU 的系数，用向量化操作一次性算完。

## 推荐方案

**训练阶段**：用 #4（向量化），预计能快 5-10×。改动小。

**推理阶段**：用 #1（多项式替代），零开销。

**论文写**：InnerNet 训练开销可接受（比 SwiGLU 慢几倍但不是数量级），推理可以用拟合公式消除开销。
