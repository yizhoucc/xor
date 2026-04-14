"""Visualize InnerNet pretrain targets."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 150

out_dir = 'results/figures'

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

nb = 101
x = np.linspace(-5, 5, nb)
y = np.linspace(-5, 5, nb)
xv, yv = np.meshgrid(x, y)
xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T

# Panel A: Gaussian-blurred random function (CNN/MLP pretrain target)
npr = np.random.RandomState(seed=1234)
mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
gaussian_kernel /= gaussian_kernel.sum()
init_unif = npr.uniform(-1, 1, size=(nb, nb))
target_random = convolve2d(init_unif, gaussian_kernel, mode='same')

im0 = axes[0].contourf(xv, yv, target_random, levels=30, cmap='RdBu_r')
axes[0].set_title('(a) Pretrain target: smoothed random\n(CNN/MLP, seed=1234)', fontsize=11)
axes[0].set_xlabel('a')
axes[0].set_ylabel('b')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Panel B: Simple Gaussian (Transformer/LSTM pretrain target)
a = np.linspace(-3, 3, 200)
b = np.linspace(-3, 3, 200)
A, B = np.meshgrid(a, b)
target_gaussian = np.exp(-(A**2 + B**2))

im1 = axes[1].contourf(A, B, target_gaussian, levels=30, cmap='RdBu_r')
axes[1].set_title('(b) Pretrain target: Gaussian\n(Transformer/LSTM)', fontsize=11)
axes[1].set_xlabel('a')
axes[1].set_ylabel('b')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Panel C: SwiGLU for comparison
def silu(x):
    return x / (1 + np.exp(-x))

Z_swiglu = silu(A) * B
Z_swiglu = np.clip(Z_swiglu, -5, 5)
im2 = axes[2].contourf(A, B, Z_swiglu, levels=30, cmap='RdBu_r')
axes[2].set_title('(c) SwiGLU: SiLU(a) × b\n(what InnerNet might converge to)', fontsize=11)
axes[2].set_xlabel('a')
axes[2].set_ylabel('b')
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig(f'{out_dir}/fig5_pretrain_targets.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig5_pretrain_targets.png', bbox_inches='tight')
print('Saved fig5_pretrain_targets')
