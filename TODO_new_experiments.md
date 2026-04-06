# 实验 TODO

## 已完成 ✅

| # | 实验 | 结果 |
|---|------|------|
| 1 | SwiGLU DQN (4 环境) | CP: InnerNet最佳, Acro/MC: SwiGLU最佳, LL: ReLU最佳 |
| 2 | SST-2 文本分类 (5 seeds) | 2arg 80.00 vs ReLU 81.31 — ReLU更好 |
| 3 | AG News 文本分类 (5 seeds) | 2arg 90.86 vs ReLU 91.08 — 持平 |
| 4 | UCI Adult 表格 (5 seeds) | 2arg 84.72 vs ReLU 84.69 — 持平 |
| 5 | UCI Wine 表格 (5 seeds) | 2arg 63.44 vs ReLU 60.56 — **+2.88%** |
| 6 | Speech Commands 音频 (5 seeds) | 2arg 16.44 vs ReLU 15.52 — 都很低 |
| 7 | ECG200 时间序列 (5 seeds) | 2arg 80.80 vs ReLU 82.20 — 持平 |
| 8 | Transformer 规模 small/large | small: InnerNet 112.66 vs 116.63 (**-3.4%**) |
| 9 | 补充 seeds (mlp/cnn 2arg) | 全部完成 |

## 进行中 ⏳

| # | 实验 | 状态 |
|---|------|------|
| 10 | Transformer large InnerNet (d=256) | 集群运行中 |

---

## 新 TODO

### 高优先级 — 论文复现缺失

#### 11. 对抗鲁棒性 — AutoAttack (论文 Table 1)
> 论文核心结论之一：InnerNet 提升对抗鲁棒性。
> 方法：在已训练好的 CNN CIFAR-10 模型上跑 AutoAttack (L∞, ε=8/255)

- [ ] 安装 autoattack 库
- [ ] 写 eval 脚本：加载 best model → AutoAttack 评估
- [ ] 在 2arg / 1arg / ReLU / ReLU+LN 模型上分别评估
- [ ] 提交 cluster

#### 12. 数据损坏鲁棒性 — CIFAR-10-C (论文 Figure 9)
> 论文另一核心结论：InnerNet 对 common corruptions 更鲁棒。
> 方法：在 CIFAR-10-C 15 种损坏 × 5 级别上评估已训练模型

- [ ] 下载 CIFAR-10-C 数据集
- [ ] 写 eval 脚本：遍历所有损坏类型和级别
- [ ] 对比 2arg / ReLU 模型
- [ ] 提交 cluster

### 高优先级 — 验证核心假说

#### 13. 容量缩放实验 — 验证"容量受限假说"
> 我们发现 InnerNet 在小模型增益大、大模型增益小。
> 系统验证：固定 MLP CIFAR-10，变化 hidden_dim=[32, 64, 128, 256, 512]

- [ ] 创建 5 个不同宽度的 2arg config
- [ ] 创建对应 5 个 ReLU config
- [ ] 提交 cluster
- [ ] 画 accuracy vs width 曲线

#### 14. ResNet — skip connection 是否消除 InnerNet 优势
> 假设：skip connection 提供了另一种特征交互通道，可能让 InnerNet 冗余。
> 实验：ResNet-18 on CIFAR-10, InnerNet vs ReLU

- [ ] 实现 InnerNetResNet (在 conv 后替换 ReLU)
- [ ] Config: resnet_cifar_2arg.yaml, resnet_cifar_relu.yaml
- [ ] 提交 cluster

### 中优先级 — 扩展场景

#### 15. 回归任务 — California Housing
> 我们只做了分类，回归是全新维度。
> 架构：MLP，输出层 1 unit，MSE loss

- [ ] 数据加载 (sklearn.datasets)
- [ ] 支持回归 loss (MSE) in experiment_runner
- [ ] Config: mlp_housing_2arg.yaml, mlp_housing_relu.yaml
- [ ] 提交 cluster

#### 16. Dense Embedding 文本分类
> TF-IDF 稀疏特征上 InnerNet 无效，换 dense embedding 试试。
> 方法：用预训练 sentence-transformer 提取 384-dim embedding → MLP

- [ ] 用 sentence-transformers 提取 SST-2 / AG News embedding
- [ ] 新 config: mlp_sst2_emb_2arg.yaml 等
- [ ] 提交 cluster

#### 17. Autoencoder — 重建任务
> 全新任务类型：无监督重建。InnerNet 是否有助于学习更好的表征？
> 架构：MLP Autoencoder on MNIST, MSE reconstruction loss

- [ ] 实现 InnerNetAutoencoder
- [ ] Config + runner
- [ ] 提交 cluster

### 低优先级

#### 18. GNN — Cora 节点分类
> 图结构数据，需要 PyG/DGL，改动大

- [ ] 评估可行性

#### 19. S4/Mamba — 现代 SSM
> 最新序列建模架构，复杂度高

- [ ] 评估可行性

---

## 优先级排序
1. **AutoAttack 鲁棒性** — 论文有，我们没复现，必做
2. **CIFAR-10-C 鲁棒性** — 同上
3. **容量缩放实验** — 直接验证我们的核心发现
4. **ResNet** — 验证 skip connection 假说
5. **回归 (Housing)** — 简单，扩展任务类型
6. **Dense Embedding 文本** — 验证"稀疏特征无效"假说
7. **Autoencoder** — 新任务类型
8. **GNN / S4** — 复杂，优先级低
