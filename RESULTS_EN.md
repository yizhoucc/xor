# Experiment Results — InnerNet (Learnable 2-Argument Activation Function)

> Summary of all experiments for "Two-argument activation functions learn soft XOR operations like cortical neurons" (Yoon et al., 2021).

## Paper Story — Learnable Activations as Differentiable Architecture Search

InnerNet replaces scalar activations (ReLU) with a small learned MLP taking two inputs: `f(a, b) → output`, so each neuron computes a nonlinear interaction between two learned linear projections (analogous to the soft-XOR interactions of cortical neurons).

We position InnerNet not as a drop-in *better activation function*, but as a **differentiable probe for architectural primitives**: replace fixed activations with InnerNet, train, visualize the learned 2D function, and quantify it with simple closed-form operators. The central finding, with a supporting functional result:

- **Learned operators are basin-dependent** — within a SwiGLU-warm-started Transformer, random, identity, and multiplication InnerNet initializations converge to a scaled-SwiGLU surface (SwiGLU R² ≥ 0.98). In contrast, four completed jointly trained Bilinear-host runs converge to pure multiplication (mult R² ≥ 0.998) while reaching similarly strong PPL. InnerNet can recover and refine a useful bivariate operator, but the operator is not a network-independent SwiGLU attractor.
- **Functional recurrent gating** — a recurrent network with an additive memory channel and two InnerNets solves Sequential MNIST (~98% vs 11% for a plain RNN); we report this as a functional result, since the individual InnerNet surfaces are not identifiable as canonical LSTM/GRU gates.

This makes the large-scale behavior a boundary, not a contradiction: from-scratch InnerNet underperforms hand-designed SwiGLU because optimization and the surrounding network determine which functional basin is reached, not because InnerNet lacks the required capacity (frozen-InnerNet capacity matches SwiGLU).

**Supporting findings:**

1. **Capacity ceiling ≥ SwiGLU** — warm-start across 11 tasks: InnerNet wins or ties 10/11. Verified by ivs_d128 (5 seeds): frozen InnerNet 77.38±0.51 = SwiGLU 77.38±0.54. The from-scratch gap is an optimization barrier, not a capacity limit.
2. **Position determines effect** — InnerNet helps at positions without skip-connection bypass (CNN +0.4–4.6%, AE −43% MSE, Transformer FFN −0.8–3.3% PPL across 4 scales, ResNet internal-only +1.5%), and is redundant where a skip connection already provides a bypass.
3. **Scale boundary** — InnerNet remains below same-width GELU across the controlled d=64–256 sweep, but the gains are not monotonic (3.3%, 1.6%, 0.8%, 1.7%) and reverse in the larger GPT-style experiment. We therefore report scale as an empirical boundary, not as a fitted scaling law.
4. **Simplicity wins** — simple adjacent pairing > deliberate semantic pairing; no pretraining needed (end-to-end ≈ 3-phase).

**Relation to prior work:** Yoon et al. (IEEE Access 2022) introduced two-argument activations on MLP/CNN classification (MNIST/CIFAR), reporting modest accuracy gains and improved robustness. We extend this to 10+ architectures (Transformer, LSTM, recurrent models, autoencoders, ResNet/VGG/WRN, ViT, MLP-Mixer, RL, masked/causal LM) and study when learned two-argument interactions expose useful architectural structure.

---

## 1. CNN Image Classification (5 seeds)

| Dataset | 2-arg | 1-arg | ReLU | ReLU+LN | ReLU matched | SwiGLU | Gain |
|---------|-------|-------|------|---------|-------------|--------|------|
| MNIST | 99.41±0.04 | 99.42±0.06 | 99.02±0.03 | 99.18±0.02 | — | — | **+0.39** |
| CIFAR-10 | 78.57±0.74 | 81.02±1.02 | 73.99±0.49 | 75.14±0.34 | 70.67±0.43 | 79.79±0.54 | **+4.58** |
| FashionMNIST | 90.91±0.29 | ⏳ | 89.34±0.13 | 89.34±0.16 | — | — | **+1.57** |
| SVHN | ⏳ | 95.16±0.23 | 92.55±0.19 | 92.82±0.09 | — | — | **+2.46** |
| CIFAR-100 big | 53.74±0.88 | — | 50.00±0.83 | — | — | 46.48±0.50 | **+3.74** |

**Parameter fairness (CIFAR-10)**: InnerNet (127K params, 78.57%) > ReLU matched (127K params, 70.67%). Same params, **+7.9% accuracy**.

Configs: `config/experiments/cnn_cifar_2arg.yaml`, exp: `exp/cnn_cifar_2arg_*`

Reported ± values are the population standard deviation (ddof=0) over seeds. For the headline CIFAR-10 2-arg result (n=5, seeds 42–46: 78.68, 79.69, 78.94, 77.93, 77.62) the sample standard deviation (ddof=1) is 0.82.

Multi-seed training curves are provided in `results/figures/fig_training_curves.pdf` (MLP/CNN × MNIST/CIFAR-10; mean ± sample SD). The MLP-MNIST ReLU curve uses the parameter-matched width-112 configuration and excludes the older same-named width-64 run.

## 2. Autoencoder Reconstruction (MSE↓, 3–5 seeds)

| Dataset | InnerNet | ReLU | ReLU matched | Improvement |
|---------|----------|------|-------------|-------------|
| MNIST | **0.0039** | 0.0068 | 0.0059 | **-43% vs ReLU, -34% vs matched** |
| FashionMNIST | **0.0076** | 0.0086 | — | **-12%** |
| CIFAR-10 | **0.0081** | 0.0105 | — | **-23%** |

### AE Capacity Scaling (latent dimension)

| Latent Dim | InnerNet | ReLU | Improvement |
|-----------|----------|------|-------------|
| 8 | 0.0141 | 0.0183 | -23% |
| 16 | 0.0075 | 0.0114 | -34% |
| 32 | 0.0039 | 0.0067 | -42% |
| 64 | 0.0026 | 0.0042 | -39% |

Configs: `config/experiments/ae_mnist_2arg.yaml`, exp: `exp/ae_mnist_2arg_*`

## 3. Transformer Language Models (PPL↓, 5 seeds)

| Config | GELU | SwiGLU | InnerNet | vs GELU |
|--------|------|--------|----------|---------|
| WikiText-2 d=64 | 116.63±0.84 | **112.31±0.49** | **112.83±0.94** | **-3.3%** |
| WikiText-2 d=128 | 96.82±1.19 | **92.98±1.14** | **95.26±1.00** | **-1.6%** |
| WikiText-2 d=192 | 89.11±0.92 | **85.43±0.41** | **88.42±0.94** | **-0.8%** |
| WikiText-2 d=256 | 86.05±0.97 | **81.56±0.94** | **84.62±1.47** | **-1.7%** |
| PTB d=128 | 212.28±0.88 | **205.82±0.98** | **207.91** | **-2.1%** |

InnerNet reaches lower mean PPL than GELU at every scale (-0.8% to -3.3%), and at d=64 its PPL matches the hand-designed SwiGLU model (112.83 vs 112.31). Performance parity alone does not identify the learned operator. Paired tests against GELU give p=0.00022, 0.05095, 0.361, and 0.173 at d=64, 128, 192, and 256, respectively; thus the direction is consistent, while inferential support is strongest at the smallest scale. These are same-width comparisons, not parameter-matched ones.

The post-sharing scaling figure is generated directly from canonical audit artifacts (`results/figures/fig_scaling_law.pdf`). It replaces the archived hard-coded pre-fix plot and shows a positive but non-monotonic benefit (3.3%, 1.6%, 0.8%, 1.7%).

**Scale boundary**: The same-width advantage remains positive but non-monotonic through d=256 in the standard Transformer sweep. It reverses in the larger GPT-style experiment, while warm-start experiments show that the learned function remains expressive enough in the controlled setting.

### InnerNet vs SwiGLU: Fair Comparison (5 seeds)

When InnerNet is initialized from a trained SwiGLU model (10 epochs) and both continue training for 10 more epochs (total 20 each):

| Model | Best PPL (20ep total) |
|-------|----------------------|
| InnerNet (from SwiGLU init) | **76.85 ± 0.63** |
| SwiGLU (continued) | 77.04 ± 0.79 |

InnerNet wins in 4/5 seeds (-0.19 PPL). This pattern holds across multiple configurations:

| Config | SwiGLU | Shared (97p) | Δ | Non-shared (388p) | Δ |
|--------|--------|-------------|-----|------------------|-----|
| **CNN CIFAR-10** | 83.33% | **85.79%** | **+2.46%** | — | — |
| **MLM WikiText-2** | 54.92 | **38.74** | **-16.18** | ⏳ | — |
| PTB d=128 | 162.22 | **161.18** | -1.04 | **162.49** | **-2.11**† |
| TF d=128 | 77.04 | **76.85** | -0.19 | ⏳ | — |
| TF d=64 | 92.08 | **91.93** | -0.15 | — | — |
| MLP-Mixer | 81.13% | **81.25%** | +0.12% | — | — |
| ViT | 77.54% | 77.59% | Tied | — | — |
| TF d=256 | — | — | Tied | — | — |
| AE MNIST | 0.01206 | 0.01211 | Tied | — | — |
| GPT d=256 | ~73.1 | ~73.2 | Tied | — | — |
| **LSTM** | **104.71** | 105.79 | **+1.08** | — | — |

**Shared: 10/11 wins or ties (LSTM only loss). Non-shared consistently larger margins.**

| Task | Shared Δ | Non-shared Δ |
|------|---------|-------------|
| PTB | -1.04 | **-2.11** |
| MLM | -16.18 | **-3.28** (vs SwiGLU 18.91) |
| CNN | +2.46% | **+3.12%** |

Each layer learns a distinct activation function when non-shared (fig11). Parameter overhead negligible (291 extra).

†Non-shared PTB is a paired warm-start comparison against its own separately-trained SwiGLU baseline (InnerNet 162.49 vs SwiGLU 164.59, seeds 42–46, InnerNet wins 5/5, paired-t p=0.037, Cohen's dz=-1.38). It is not comparable to the 162.22 SwiGLU value in the shared-experiment column above, which comes from a different run. Source: `exp/warmstart_nonshared/results.p`.

### Capacity Verification (ivs_d128, 5 seeds)

Direct proof that InnerNet capacity ≥ SwiGLU: SwiGLU trained to convergence → replace activations with InnerNet → continue training. Final result: SwiGLU **77.38 ± 0.54** vs Frozen InnerNet **77.38 ± 0.51**. Exact match across 5 seeds. The from-scratch gap is purely optimization, not expressiveness.

The corresponding epoch-wise trajectories are available in `results/figures/fig_warmstart_curves.{png,pdf}`. The left panel shows the five-seed fork from a shared SwiGLU pretraining trajectory into continued SwiGLU and trainable-InnerNet branches; the right panel shows the frozen-outer-network capacity test. Curves report mean ± sample SD from `exp/ivs_d128_v2/results.p`.

### Initialization Does Not Matter (3 seeds, completed)

Free-init experiment (Wiki d=128, 3 seeds): 4 InnerNet initializations all converge to the same endpoint:

| Init | Seed 42 | Seed 43 | Seed 44 |
|------|---------|---------|---------|
| swiglu_fitted | 71.98 | 71.98 | 72.60 |
| multiply | 71.91 | 71.91 | 72.54 |
| random | 71.72 | 71.72 | 72.23 |
| identity | 71.99 | 71.99 | 72.33 |
| **SwiGLU** | **77.29** | **77.29** | **77.24** |

All 4 initializations converge to ~71.7–72.6 (vs SwiGLU ~77.3). The learned function is determined by the task and network weights, not the InnerNet starting point — and distilling these same checkpoints shows they converge not just to the same *performance* but to the same *SwiGLU surface* (§Distillation: SwiGLU R² ≥ 0.98 for random/identity/multiply init). This is the init-independence behind the discovery claim.

### Multiply Initialization — Multi-task (5 seeds, completed)

Multiply-initialized InnerNet (f(a,b)=a·b) across 4 tasks:

| Task | SwiGLU | MultInit | Δ |
|------|--------|----------|-----|
| Wiki d=64 | 92.01±0.34 | 92.25±0.38 | Tied |
| Wiki d=128 | 77.45±0.29 | **77.21±0.34** | -0.24 |
| PTB d=128 | 164.19±1.12 | **163.10±2.14** | -1.08 |
| **MLM** | **19.09±0.27** | **15.93±0.21** | **-3.16 (-16.6%)** |

MLM shows the largest gain. Simple multiplicative interaction f(a,b)=a·b provides a strong initialization, particularly for tasks where feature interaction dominates.

### Distillation: Initialization Robustness Within a Host Basin

We distill each trained InnerNet FFN surface f(a, b) into closed-form operators by least-squares on a [-5, 5]² grid (`scripts/distill_innernet.py`), reporting R² per family. The decisive test varies the **InnerNet initialization** (a capable network is held fixed) and asks what surface it converges to:

| InnerNet init | pure mult `a·b` R² | **SwiGLU `silu(a)·b` R²** | outcome |
|---|:---:|:---:|---|
| random | 0.63 | **0.988** | reaches optimum |
| identity | 0.61 | **0.990** | reaches optimum |
| multiplication (starts at `a·b`) | 0.57 | **0.991** | reaches optimum |
| SwiGLU (control) | 0.58 | **0.984** | reaches optimum |
| random / multiply, **from-scratch** (fail) | **0.72–0.89** | 0.27–0.46 | stalls at `a·b` |

Within the SwiGLU-warm-started host, all tested InnerNet initializations converge to the scaled-SwiGLU surface. The multiplication-initialized branch moves from `a·b` to `silu(a)·b`, showing that the final surface is not merely retained from the InnerNet initialization. This conclusion is conditional on the host basin.

A causal host control gives the complementary result. Starting from a trained Bilinear-GLU host (`a·b`), four completed joint-training seeds improve from host PPL **79.60±0.31** to **73.51±0.37** (random init) and **73.72±0.35** (multiply init), yet their final surfaces remain pure multiplication: mult R² **0.9982±0.0006** / **0.9989±0.0005**, versus SwiGLU R² 0.528/0.549. Available frozen-host controls also recover `a·b` (mult R²≈0.999). The PPL gain cannot be assigned solely to the activation because no Bilinear-continued branch was run; the surface result nevertheless rejects a universal, network-independent SwiGLU attractor.

Single-checkpoint fits corroborate the operator across settings (Transformer FFN d=128 SwiGLU R²=0.94 / `0.24·silu(a)·b`; CNN CIFAR-10 0.91 / `0.35·silu(a)·b`; SwiGLU-fit control 0.99, validating the fitting procedure).

Config: `scripts/distill_crossinit.py`, `scripts/warmstart_causal.py`, and `scripts/distill_innernet.py`. The causal matrix is incomplete because several jobs timed out or landed on an unsupported RTX Pro 6000 node, but all four completed joint Bilinear seeds agree on the operator family.

---

Config: `scripts/innernet_vs_swiglu.py`, `warmstart_cnn.py`, `warmstart_ae.py`

## 4. LSTM (WikiText-2, PPL↓, 5 seeds)

| Variant | PPL |
|---------|-----|
| **Classic InnerNet** (adjacent pair) | **101.72±0.99** |
| Semantic InnerNet (x vs h pair) | 105.30±0.31 |
| Standard LSTM | 108.39±0.75 |

Classic InnerNet achieves **-6.2% PPL** vs standard LSTM. Adjacent-dimension pairing outperforms deliberate semantic pairing.

Config: `config/experiments/lstm_wikitext_classic.yaml`, exp: `exp/lstm_wikitext_classic_*`

## 4b. Sequential MNIST — Functional Gating with Additive Memory

**Hypothesis**: LSTM outperforms RNN due to learned gates. InnerNet's dual-input activation naturally supports gating patterns (σ(a)·b). Can RNN+InnerNet approach LSTM on a task requiring long-term memory?

Sequential MNIST reads each 28×28 image pixel-by-pixel (784 timesteps), then classifies. Standard RNN fails entirely (~11%) while gated architectures succeed.

| Model | Params | Best Acc | Seeds |
|-------|--------|----------|-------|
| SeqRNN (tanh) | 18K | 11.36% ± 0.02% | 5/5 |
| SeqLSTM | 68K | 77.03% ± 15.66% | 5/5 |
| SeqGRU | 52K | **98.72% ± 0.14%** | 5/5 |
| SeqInnerNetRNN (Plan A) | 19K | 11.04% ± 0.62% | 5/5 |
| **SeqGatedRNN (Plan B)** | 37K | **~98.4%** (98.42 / 98.15 / 98.51) | 3/5 successful (2 NaN) |
| **SeqMinGatedRNN (no `W_c`)** | 21K | **98.44% ± 0.22%** | 8/9 trained (1 NaN; 1 additional infrastructure OOM) |

**Plan A** (single InnerNet replacing tanh) fails — identical to vanilla RNN. Without an additive memory channel, InnerNet cannot learn to prevent information decay over 784 steps.

**Plan B** establishes functional sufficiency: an additive memory path plus learned bivariate interactions can solve the long-sequence task. Removing `W_c` reduces the model to 21K parameters (69% fewer than LSTM) while retaining 98.44% accuracy and increasing observed training success to 8/9 valid runs. One valid run diverged to NaN; one additional submitted job failed before training because its Slurm node exposed no GPU and exhausted host memory. Cross-seed surface fits remain non-identifiable: constrained inner_net1/inner_net2 gate R² values are 0.245±0.223 and 0.447±0.263, essentially unchanged from the unconstrained model.

Config: `config/experiments/seq_mnist_min_gated.yaml`; raw runs: `exp/seq_mnist_min_gated_20260726_*`; surface analysis: `results/figures/gate_crossseed_seqmin_constrained.json`.

## 5. Parameter Efficiency — MLP CIFAR-10 (5 seeds)

| Width | InnerNet (Params) | ReLU (Params) | Gain |
|-------|----------|------|------|
| 32 | 38.34% (104K) | 37.47% (101K) | +0.87 |
| 64 | 47.66% (206K) | 45.77% (206K) | +1.89 |
| 128 | 52.05% (415K) | 49.87% (428K) | +2.17 |
| 256 | 54.82% (858K) | 51.99% (921K) | +2.83 |
| 512 | 55.63% (1.84M) | 52.63% (2.10M) | +3.00 |

**InnerNet w=128 (415K) ≥ ReLU w=256 (921K) → 55% parameter savings.**

## 6. Big MLP MNIST (3×256, dropout=0.3, 5 seeds)

| Model | Accuracy |
|-------|----------|
| InnerNet | **98.39±0.04** |
| ReLU | 97.93±0.07 |
| Improvement | **+0.46%** |

## 7. ResNet — Position Matters

| Setup | CIFAR-10 | CIFAR-100+aug |
|-------|----------|---------------|
| ReLU baseline | 86.33% | 73.51% |
| InnerNet (all positions) | 86.10% (neutral) | 73.00% (neutral) |
| **InnerNet (internal-only)** | **87.7%** (2/5 done) | **74.97%** (+1.5%) |

Replacing ALL activations (including post-skip) shows no benefit. But replacing ONLY the internal activation (between conv1 and conv2, no skip protection) improves performance. This mirrors the Transformer FFN finding — InnerNet helps at positions without residual bypass.

Config: `config/experiments/resnet_cifar_internal_2arg.yaml`, `resnet_cifar100_aug_internal_2arg.yaml`

## 8. PPO Reinforcement Learning (30 seeds)

| Environment | InnerNet | ReLU | SwiGLU |
|-------------|----------|------|--------|
| CartPole | 499.9 | 500.0 | 500.0 |
| Acrobot | **-75.3** | -79.8 | -81.7 |
| LunarLander | **187.6** | 158.8 | -249.7 |

InnerNet wins LunarLander (+18% vs ReLU) with 30 seeds. SwiGLU fails entirely on LunarLander (-249.7).

## 9. Where InnerNet Does Not Help

| Experiment | Result | Interpretation |
|------------|--------|---------------|
| ResNet (all positions) | Neutral | Post-skip activation redundant with residual path |
| CNN at very small scale (×0.25) | Worse (-5%) | Channel-pairing overhead dominates when model is tiny |

## Reproducibility and Statistical Provenance

All structured local experiment artifacts are indexed by a canonical manifest (`scripts/build_result_manifest.py`) and aggregated by scientific configuration, condition, and run outcome rather than directory name (`scripts/summarize_result_manifest.py`). The audit includes unified-runner artifacts and standalone deploy, Sequential-MNIST, and warm-start results, including epoch-wise `ivs_d128_v2` branches: 1,121 metric rows from 472 experiment directories, with 388 raw-verified and 84 incomplete experiments. It produces 208 reportable groups with raw seed values, both sample and population standard deviations, and no unresolved within-configuration seed conflicts. Failed/NaN runs remain explicit rather than being silently removed; for example, the constrained Sequential-MNIST model has an eight-seed successful group at 98.435% (sample SD 0.236%; population SD 0.220%) and a separately retained NaN run. A single reused experiment name (`mlp_mnist_relu`) was detected and separated into its unmatched 64-width and parameter-matched 112-width configurations.

Fifteen headline comparisons are registered in `config/audit/core_comparisons.yaml` and regenerated as `results/audit/core_comparisons.csv`. For InnerNet versus GELU at d=64/128/192/256, paired-t p-values are 0.00022/0.05095/0.361/0.173; SwiGLU versus InnerNet at d=128 gives p=0.00917. CNN, autoencoder, and large-MLP headline comparisons have paired-t p<0.014. Because a two-sided Wilcoxon test with n=5 has coarse resolution (typically a minimum p=0.0625), formal reporting includes raw seeds, bootstrap confidence intervals, parametric and non-parametric tests, and Cohen's dz rather than relying on a single threshold.

The registered headline cells in RESULTS_CN/EN are also checked directly against the canonical summary; all 40 currently registered cells match. Audit artifacts: `results/audit/grouped_metric_summary.csv`, `metric_conflicts.csv`, `experiment_variant_collisions.csv`, `core_comparisons.csv`, and `document_consistency.csv`.

## Distilled-Operator Deployment

On CIFAR-10 (5 seeds), the learned InnerNet reaches **84.95±0.57%**, while its fixed polynomial approximation reaches **81.32±0.32%**, compared with 79.93±0.17% for ReLU and 79.97±0.37% for SwiGLU (sample SD). The distilled operator is 2.68× faster than InnerNet, but loses 3.63 accuracy points (paired-t p=0.00015); it remains 1.39/1.35 points above ReLU/SwiGLU (p=0.0010/0.0046). It is still 2.46× slower than SwiGLU because the unfused polynomial uses several elementwise operations. InnerNet adds only 129 parameters over the fixed gated models, confirming that runtime—not parameter count—is the practical cost.

The corresponding Transformer deployment timed out before the distilled condition began (InnerNet 4/5 seeds; distilled 0/5). Completed branches show approximately 99.6k tokens/s for InnerNet versus 600.1k for SwiGLU, a 6.03× throughput gap on the same GPU. We therefore treat deployment as a bounded auxiliary result: closed-form distillation recovers part of the speed and retains some accuracy benefit in CNNs, but the current approximation is not lossless and the FFN deployment is incomplete. Source: `results/audit/deploy_analysis.json`.

## Summary

InnerNet provides consistent benefits in **feedforward networks without built-in feature interaction mechanisms**:

- **Autoencoders**: -23% to -43% MSE
- **CNNs**: +0.4–4.6% accuracy with 40% fewer parameters, consistent across 5 datasets
- **Transformer FFN**: -0.8–3.3% PPL across 4 model sizes; slightly exceeds SwiGLU with warm-start (76.85 vs 77.04)
- **LSTM**: -6.2% PPL (WikiText-2, classic pairing)
- **ResNet internal-only**: +1.5% on CIFAR-100 — position matters
- **Parameter efficiency**: 55% parameter savings (InnerNet w=128 ≈ ReLU w=256)
- **Multiply-init MLM**: -16.6% PPL vs SwiGLU (5 seeds)
- **Functional recurrent interaction**: the 21K constrained model reaches 98.44±0.22% on 8/9 trained Sequential MNIST runs; individual gate surfaces remain non-identifiable

**Capacity and optimization**: InnerNet wins or ties SwiGLU in **10/11** warm-start tasks. The ivs_d128 experiment directly verifies capacity ≥ SwiGLU (Frozen InnerNet 77.38±0.51 = SwiGLU 77.38±0.54, 5 seeds). Within a SwiGLU host, four different InnerNet initializations reach the same performance basin; the Bilinear-host control shows that the learned operator family remains host-dependent. The from-scratch gap is an optimization issue: InnerNet's per-epoch compute is higher, and the advantage decreases with model size.

**Boundaries**: InnerNet is redundant at positions protected by skip connections, but effective at unprotected internal positions even in residual networks.

**Simplicity principle**: Adjacent pairing > semantic pairing — the dual-input interaction itself drives the improvement.
