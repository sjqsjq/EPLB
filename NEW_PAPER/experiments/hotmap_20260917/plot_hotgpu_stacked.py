"""
Fig 15c: per-domain hot-GPU stacked-bar distribution under identity placement.
Companion to fig15b (entropy bars) --- makes the "shape" behind entropy visible.
"""
import json, os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(f"{HERE}/hotgpu_distributions.json"))
ORDER = ["MMLU","GSM8K","ARC","ARC-E","CSQA","OBQA","prover","HumanEval","CMMLU"]
data = {k: data[k] for k in ORDER if k in data}

props = np.array([data[k]["prop_per_gpu"] for k in data.keys()])  # (D, 8)
Hs    = np.array([data[k]["entropy_bits"] for k in data.keys()])
labels = list(data.keys())
D = len(labels)

fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor='white')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

cmap = plt.get_cmap('tab10')
gpu_colors = [cmap(i) for i in range(8)]
x = np.arange(D); width = 0.68
bottom = np.zeros(D)
for g in range(8):
    ax.bar(x, props[:,g], width, bottom=bottom, color=gpu_colors[g],
           edgecolor='white', linewidth=0.4, label=f'GPU{g}')
    # annotate the modal GPU on each bar
    for i in range(D):
        if props[i,g] >= 0.15:
            ax.text(i, bottom[i] + props[i,g]/2, f'GPU{g}\n{props[i,g]*100:.0f}%',
                    ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    bottom += props[:,g]

# entropy label on top of each bar
for i, H in enumerate(Hs):
    ax.text(i, 1.02, f'H={H:.2f}', ha='center', va='bottom',
            fontsize=8.6, color='#c0392b' if H < 1.5 else '#1f3d7a', fontweight='bold')

# horizontal reference: uniform (H = 3 bits)
ax.axhline(1.0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9.5)
ax.set_ylim(0, 1.13); ax.set_ylabel('per-forward hot-GPU share (identity placement)', fontsize=10.5)
ax.set_title('Fig 15c: Hot-GPU distribution under identity placement (per-domain)\n'
             'prover pinned to GPU5 (H=0.00); other domains modal on 1–2 GPUs (H=1.5–2.4)',
             fontsize=10.5, loc='left', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', bbox_to_anchor=(1.16, 1.0), fontsize=8.4, frameon=False, title='GPU', title_fontsize=9)
ax.grid(axis='y', alpha=0.18)
plt.tight_layout()

out = "/workspace/EPLB/NEW_PAPER/figures/fig15c_identity_hotgpu_distribution.png"
fig.savefig(out, dpi=190, bbox_inches='tight')
fig.savefig(f"{HERE}/fig15c_identity_hotgpu_distribution.png", dpi=190, bbox_inches='tight')
print("wrote:", out)
