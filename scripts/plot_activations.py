"""Generate publication-quality activation function comparison figures."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['figure.dpi'] = 150

out_dir = 'results/figures'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x):
    return x * sigmoid(x)


def swiglu_surface(a, b):
    return silu(a) * b


# ── Figure 1: 1D activation functions ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x = np.linspace(-4, 4, 500)

# Panel A: Classic (saturating)
ax = axes[0]
ax.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
ax.plot(x, np.tanh(x), label='Tanh', linewidth=2)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_title('(a) Saturating', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend(frameon=False)
ax.set_ylim(-1.5, 1.5)

# Panel B: ReLU family
ax = axes[1]
ax.plot(x, relu(x), label='ReLU', linewidth=2)
ax.plot(x, np.where(x > 0, x, 0.1 * x), label='Leaky ReLU', linewidth=2, linestyle='--')
ax.plot(x, np.where(x > 0, x, 0.3 * (np.exp(x) - 1)), label='ELU', linewidth=2, linestyle='-.')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_title('(b) Piecewise Linear', fontsize=13)
ax.set_xlabel('x')
ax.legend(frameon=False)
ax.set_ylim(-1.5, 4)

# Panel C: Smooth (modern)
ax = axes[2]
ax.plot(x, gelu(x), label='GELU', linewidth=2)
ax.plot(x, silu(x), label='SiLU/Swish', linewidth=2, linestyle='--')
ax.plot(x, relu(x), label='ReLU', linewidth=2, alpha=0.4, color='gray')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_title('(c) Smooth', fontsize=13)
ax.set_xlabel('x')
ax.legend(frameon=False)
ax.set_ylim(-1.5, 4)

plt.tight_layout()
plt.savefig(f'{out_dir}/fig1_activation_functions_1d.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig1_activation_functions_1d.png', bbox_inches='tight')
print('Saved fig1_activation_functions_1d')


# ── Figure 2: 2D gating functions (SwiGLU vs InnerNet concept) ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

a = np.linspace(-3, 3, 200)
b = np.linspace(-3, 3, 200)
A, B = np.meshgrid(a, b)

# Panel A: ReLU (1D, applied independently)
Z_relu = relu(A)  # only depends on first input
im0 = axes[0].contourf(A, B, Z_relu, levels=30, cmap='RdBu_r')
axes[0].set_title('(a) ReLU: f(a) = max(0, a)', fontsize=12)
axes[0].set_xlabel('a')
axes[0].set_ylabel('b')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Panel B: SwiGLU
Z_swiglu = swiglu_surface(A, B)
Z_swiglu = np.clip(Z_swiglu, -5, 5)
im1 = axes[1].contourf(A, B, Z_swiglu, levels=30, cmap='RdBu_r')
axes[1].set_title('(b) SwiGLU: SiLU(a) × b', fontsize=12)
axes[1].set_xlabel('a')
axes[1].set_ylabel('b')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Panel C: Soft XOR (what InnerNet might learn)
# XOR-like: high when signs differ, low when same
Z_xor = np.tanh(A) * (1 - np.tanh(B)) + np.tanh(B) * (1 - np.tanh(A))
# Alternative: just show a*b (multiplicative interaction)
# or a learned-looking function
Z_innernet = np.sign(A * B) * np.sqrt(np.abs(A * B)) * 0.8 + 0.3 * (A - B)
Z_innernet = np.clip(Z_innernet, -4, 4)
im2 = axes[2].contourf(A, B, Z_innernet, levels=30, cmap='RdBu_r')
axes[2].set_title('(c) InnerNet: MLP(a, b) → learned', fontsize=12)
axes[2].set_xlabel('a')
axes[2].set_ylabel('b')
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig(f'{out_dir}/fig2_2d_activation_surfaces.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig2_2d_activation_surfaces.png', bbox_inches='tight')
print('Saved fig2_2d_activation_surfaces')


# ── Figure 3: Evolution timeline ──
fig, ax = plt.subplots(figsize=(14, 3.5))

events = [
    (1990, 'Sigmoid\nTanh', 'saturating'),
    (2011, 'ReLU', 'piecewise'),
    (2015, 'PReLU', 'piecewise'),
    (2016, 'ELU', 'piecewise'),
    (2017, 'GELU\nSiLU/Swish', 'smooth'),
    (2020, 'SwiGLU\nGeGLU', 'gated'),
    (2021, 'InnerNet\n(ours)', 'learned'),
]

colors = {
    'saturating': '#e74c3c',
    'piecewise': '#3498db',
    'smooth': '#2ecc71',
    'gated': '#f39c12',
    'learned': '#9b59b6',
}

ax.set_xlim(1988, 2024)
ax.set_ylim(-0.5, 1.5)
ax.axhline(y=0, color='black', linewidth=1.5)

for year, name, cat in events:
    ax.plot(year, 0, 'o', color=colors[cat], markersize=12, zorder=5)
    ax.annotate(name, (year, 0.15), ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=colors[cat])

# Legend
for cat, color in colors.items():
    label = {'saturating': 'Saturating', 'piecewise': 'Piecewise Linear',
             'smooth': 'Smooth', 'gated': 'Gated (2-input)', 'learned': 'Learned (ours)'}[cat]
    ax.plot([], [], 'o', color=color, label=label, markersize=8)

ax.legend(loc='upper left', frameon=False, ncol=5, fontsize=9)
ax.set_xlabel('Year', fontsize=12)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{out_dir}/fig3_activation_timeline.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig3_activation_timeline.png', bbox_inches='tight')
print('Saved fig3_activation_timeline')


# ── Figure 4: Architecture diagram — where InnerNet replaces ReLU ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A: Standard network
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_title('(a) Standard: f(x) = ReLU(Wx + b)', fontsize=12)

# Draw neurons
for i, y in enumerate([1, 3, 5]):
    ax.add_patch(plt.Circle((2, y), 0.4, fill=False, linewidth=1.5))
    ax.text(2, y, f'x{i+1}', ha='center', va='center', fontsize=9)

for i, y in enumerate([1.5, 3, 4.5]):
    ax.add_patch(plt.Circle((5, y), 0.4, fill=True, facecolor='#3498db', alpha=0.3, linewidth=1.5))
    ax.text(5, y, 'Σ', ha='center', va='center', fontsize=9)

for i, y in enumerate([1.5, 3, 4.5]):
    ax.add_patch(plt.Rectangle((6.5, y - 0.3), 1.2, 0.6, fill=True, facecolor='#e74c3c', alpha=0.3, linewidth=1.5))
    ax.text(7.1, y, 'ReLU', ha='center', va='center', fontsize=8)

# Arrows
for y_from in [1, 3, 5]:
    for y_to in [1.5, 3, 4.5]:
        ax.annotate('', xy=(4.6, y_to), xytext=(2.4, y_from),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

for y in [1.5, 3, 4.5]:
    ax.annotate('', xy=(6.5, y), xytext=(5.4, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax.axis('off')

# Panel B: InnerNet — pairs of neurons
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_title('(b) InnerNet: f(a,b) = MLP(a, b)', fontsize=12)

for i, y in enumerate([1, 3, 5]):
    ax.add_patch(plt.Circle((2, y), 0.4, fill=False, linewidth=1.5))
    ax.text(2, y, f'x{i+1}', ha='center', va='center', fontsize=9)

# Paired neurons
for i, (y1, y2) in enumerate([(1.5, 3), (4.5, 4.5)]):
    ax.add_patch(plt.Circle((5, y1), 0.4, fill=True, facecolor='#3498db', alpha=0.3, linewidth=1.5))
    ax.text(5, y1, 'a', ha='center', va='center', fontsize=9)
    if i == 0:
        ax.add_patch(plt.Circle((5, y2), 0.4, fill=True, facecolor='#3498db', alpha=0.3, linewidth=1.5))
        ax.text(5, y2, 'b', ha='center', va='center', fontsize=9)

# InnerNet box (shared)
ax.add_patch(plt.Rectangle((6.5, 1.2), 1.5, 2.1), )
ax.patches[-1].set_facecolor('#9b59b6')
ax.patches[-1].set_alpha(0.3)
ax.patches[-1].set_linewidth(1.5)
ax.text(7.25, 2.25, 'InnerNet\nMLP(a,b)', ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows from inputs
for y_from in [1, 3, 5]:
    for y_to in [1.5, 3, 4.5]:
        ax.annotate('', xy=(4.6, y_to), xytext=(2.4, y_from),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# Paired arrows to InnerNet
ax.annotate('', xy=(6.5, 2.0), xytext=(5.4, 1.5),
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))
ax.annotate('', xy=(6.5, 2.5), xytext=(5.4, 3.0),
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

# "shared" annotation
ax.annotate('shared\nweights', xy=(7.25, 1.0), fontsize=8, ha='center',
            color='#9b59b6', style='italic')

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{out_dir}/fig4_innernet_architecture.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/fig4_innernet_architecture.png', bbox_inches='tight')
print('Saved fig4_innernet_architecture')

print('\nAll figures saved to results/figures/')
