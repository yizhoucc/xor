# 实验 TODO

## 已完成 ✅

| # | 实验 | 结果 |
|---|------|------|
| 1 | SwiGLU DQN (4 环境) | CP: InnerNet最佳, Acro/MC: SwiGLU最佳, LL: ReLU最佳 |
| 2 | SST-2 文本分类 (5 seeds) | 2arg 80.00 vs ReLU 81.31 — ReLU更好 |
| 3 | AG News 文本分类 (5 seeds) | 2arg 90.86 vs ReLU 91.08 — 持平 |
| 4 | UCI Adult 表格 (5 seeds) | 2arg 84.72 vs ReLU 84.69 — 持平 |
| 5 | UCI Wine 表格 (5 seeds) | 2arg 63.44 vs ReLU 60.56 — **+2.88%** |
| 6 | Speech Commands 音频 (5 seeds) | 2arg 16.44 vs ReLU 15.52 — 都很低,不report |
| 7 | ECG200 时间序列 (5 seeds) | 2arg 80.80 vs ReLU 82.20 — 持平,样本太少 |
| 8 | Transformer 规模 small/large | small: InnerNet 112.66 vs 116.63 (**-3.4%**) |
| 9 | 补充 seeds (mlp/cnn 2arg) | 全部完成 |

## 集群运行中 ⏳

| # | 实验 | Jobs | 状态 |
|---|------|------|------|
| 10 | Transformer large InnerNet (d=256) | 1 | running |
| 11 | 容量缩放 (MLP CIFAR-10 w=32/64/128/256/512 × 2arg/relu × 5seeds) | 50 | running/pending |
| 12 | 回归 Housing (2arg/relu × 5seeds) | 10 | running/pending |
| 13 | AutoAttack eval (CNN CIFAR 2arg + 1arg) | 2 | pending |
| 14 | CIFAR-10-C eval (CNN CIFAR 2arg + 1arg) | 2 | pending |
| 15 | CNN CIFAR ReLU 重训 (给 robustness eval 用) | 1 | pending |

## 代码已写好，待用户确认后提交 🔜

| # | 实验 | Configs | 预期baseline | 目的 |
|---|------|---------|-------------|------|
| 16 | ResNet-18 CIFAR-10 (2arg/relu × 5seeds) | `resnet_cifar_*.yaml` | ~93% | skip connection 是否让 InnerNet 冗余 |
| 17 | ResNet-18 CIFAR-100 (2arg/relu × 5seeds) | `resnet_cifar100_*.yaml` | ~75% | 替代 30% 的小 CNN |
| 18 | 大 CNN CIFAR-100 (2arg/relu × 5seeds) | `cnn_cifar100_big_*.yaml` | ~55-65% | 更实际的 CNN baseline |
| 19 | Dense Embedding SST-2 (2arg/relu × 5seeds) | `mlp_sst2_emb_*.yaml` | ~85% | 替代 TF-IDF 稀疏特征 |
| 20 | Dense Embedding AG News (2arg/relu × 5seeds) | `mlp_agnews_emb_*.yaml` | ~90% | 同上 |
| 21 | Autoencoder MNIST (2arg/relu × 5seeds) | `ae_mnist_*.yaml` | MSE | 无监督重建任务 |

提交命令（等用户确认）：
```bash
# ResNet CIFAR-10/100
for cfg in config/experiments/resnet_cifar_*.yaml config/experiments/resnet_cifar100_*.yaml; do
  for seed in 1234 42 43 44 45; do sbatch scripts/slurm_run.sh "$cfg" "$seed"; done
done

# 大 CNN CIFAR-100
for cfg in config/experiments/cnn_cifar100_big_*.yaml; do
  for seed in 1234 42 43 44 45; do sbatch scripts/slurm_run.sh "$cfg" "$seed"; done
done

# Dense Embedding 文本 (需先安装 sentence-transformers)
for cfg in config/experiments/mlp_sst2_emb_*.yaml config/experiments/mlp_agnews_emb_*.yaml; do
  for seed in 1234 42 43 44 45; do sbatch scripts/slurm_run.sh "$cfg" "$seed"; done
done

# Autoencoder MNIST
for cfg in config/experiments/ae_mnist_*.yaml; do
  for seed in 1234 42 43 44 45; do sbatch scripts/slurm_run.sh "$cfg" "$seed"; done
done
```

## 不再追踪（baseline 太低或不适合）

| 实验 | 原因 |
|------|------|
| MLP CIFAR-100 (18.20%) | MLP 对 100 类图像太弱 |
| Speech Commands MLP (16.44%) | MLP 不适合音频 |
| ECG200 (80.80%) | 数据集太小(200样本)，方差太大 |
| Cora 图分类 | 需要 GNN，改动太大 |
| S4/Mamba | 复杂度高，优先级低 |
