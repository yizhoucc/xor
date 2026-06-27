"""Scaling-law figure (PROJECT_STATUS P3 / U30): InnerNet advantage shrinks with size.

Transformer FFN on WikiText-2, InnerNet vs GELU baseline (val PPL, lower better).
Numbers from the d=64/128/192/256 sweep (see RESULTS_*.md, exp/ivs_d* and
transformer_wikitext_* runs). Shows the InnerNet-over-baseline improvement
narrowing monotonically as d_model grows — larger models have enough capacity
to learn equivalent interactions through their own weights.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# d_model : (InnerNet PPL, GELU PPL)
DATA = {
    64:  (112.66, 116.63),
    128: (95.26, 96.82),
    192: (88.14, 89.11),
    256: (85.40, 86.05),
}

ds = sorted(DATA)
inner = [DATA[d][0] for d in ds]
base = [DATA[d][1] for d in ds]
pct = [100 * (b - i) / b for i, b in zip(inner, base)]  # % improvement

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(ds, base, 'o-', label='GELU', color='#888', lw=2)
ax1.plot(ds, inner, 's-', label='InnerNet', color='#c0392b', lw=2)
ax1.set_xlabel('d_model'); ax1.set_ylabel('Val PPL (lower better)')
ax1.set_title('Transformer FFN on WikiText-2'); ax1.set_xticks(ds)
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(ds, pct, 'D-', color='#2980b9', lw=2)
for d, p in zip(ds, pct):
    ax2.annotate(f'{p:.1f}%', (d, p), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)
ax2.axhline(0, color='k', lw=0.8)
ax2.set_xlabel('d_model'); ax2.set_ylabel('InnerNet improvement over GELU (%)')
ax2.set_title('Advantage shrinks with model size'); ax2.set_xticks(ds)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = 'results/figures/fig_scaling_law'
os.makedirs('results/figures', exist_ok=True)
fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print(f"Wrote {out}.png/.pdf")
print("d_model:", ds)
print("InnerNet improvement %:", [round(p, 2) for p in pct])
