#!/usr/bin/env python3
"""Publication figure: dense DeepGEMM FP8 staircase (H20, Qwen3-235B w13).
Raw 232 pts (light) + median-smoothed trend (bold). Pre-256 noise spikes
removed so the flat floor reads as cleanly as the post-256 plateaus."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'deepgemm_flat_dense.json')))
r = d['results']
M  = np.array([x['M']  for x in r], dtype=float)
us = np.array([x['us'] for x in r], dtype=float)

# median smoothing — kernel length must be odd. Dense up to 256 (<=1 tok step),
# sparse after. Use a small odd window that flattens spikes but keeps steps.
sm = medfilt(us, kernel_size=5)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.linewidth': 0.9, 'grid.linewidth': 0.5,
    'axes.edgecolor': '#444444', 'axes.labelcolor': '#222222',
    'xtick.color': '#444444', 'ytick.color': '#444444',
})
NAVY = '#1f3d7a'; RED = '#c0392b'; GREEN = '#2e7d32'; GOLD = '#b8860b'
ROSE = '#fbe9e7'; BLUEBG = '#e8eef7'

fig = (plt.figure(figsize=(9.0, 6.6), facecolor='white'))
gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.34,
                       left=0.105, right=0.965, top=0.93, bottom=0.105)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for ax in (ax1, ax2):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# ---------------- top: full staircase ----------------
ax1.scatter(M, us, s=9, color=NAVY, alpha=0.30, zorder=2, label='raw (232 pts)')
ax1.plot(M, sm, '-', color=NAVY, lw=2.1, zorder=3, label='median-smoothed')

# flat floor shade
ax1.axvspan(0.9, 256, color=BLUEBG, alpha=0.55, zorder=0)
ax1.text(1.15, 96, 'FLAT FLOOR  ≈ 31 µs\n(1 wave, spare capacity:\nadding tokens is free)',
         fontsize=8.6, color=NAVY, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NAVY, alpha=0.85))

# jumps
jumps = [
    (65,  'BM 64→128\nBN 48→80', GREEN),
    (129, 'BM 64→128',          GREEN),
    (257, 'BM 128→256\n(256-tile pad)', RED),
    (513, 'BN 80→128\n+1 tile', RED),
    (769, 'wave 1→2\n+1 tile',  RED),
]
for m0, label, col in jumps:
    i = np.argmin(np.abs(M - m0))
    ax1.axvline(m0, color=col, ls='--', lw=0.8, alpha=0.65, zorder=1)
    ax1.annotate(label, xy=(m0, sm[i]), xytext=(7, 5),
                 textcoords='offset points', fontsize=7.6, color=col,
                 fontweight='bold')

# plateau labels
for xm, txt, yy in [(380,'≈52 µs',55),(640,'≈81 µs',86),(900,'≈100 µs',106)]:
    ax1.annotate(txt, xy=(xm, yy), fontsize=8, color='#555555', ha='center',
                 style='italic')

ax1.set_xscale('log')
ax1.set_xlim(0.9, 1100)
ax1.set_xticks([1,2,4,8,16,32,64,128,256,512,1024])
ax1.set_xticklabels(['1','','','','16','','64','128','256','512','1024'])
ax1.set_ylim(25, 112)
ax1.set_ylabel('GEMM time per call  (µs)', fontsize=11)
ax1.set_title('DeepGEMM FP8 kernel — per-expert GEMM time vs tokens M  (H20, K=4096 N=3072)',
              fontsize=11.5, loc='left', fontweight='bold', pad=8)
ax1.text(0.012, 0.96, 'Qwen3-235B  w13 expert · pure GPU-exec timing (CUDA event, cast excluded)',
         transform=ax1.transAxes, fontsize=8.3, color='#666666', va='top')
ax1.grid(True, which='both', alpha=0.16)
ax1.legend(loc='lower right', frameon=True, framealpha=0.9,
           fontsize=8.5, edgecolor='#cccccc')

# ---------------- bottom: zoom flat 1..64 ----------------
m = M <= 64
Mf, uf, sf = M[m], us[m], sm[m]
ax2.scatter(Mf, uf, s=11, color=NAVY, alpha=0.35, zorder=2)
ax2.plot(Mf, sf, '-', color=NAVY, lw=2.0, zorder=3)
floor = float(np.median(uf))
ax2.axhline(floor, color=RED, ls='--', lw=1.0, alpha=0.8,
           label=f'median floor = {floor:.1f} µs')
ax2.set_xlim(0, 65)
ax2.set_xticks([1,8,16,24,32,40,48,56,64])
ax2.set_ylim(29, 42)
ax2.set_xlabel('M  (tokens per expert)', fontsize=11)
ax2.set_ylabel('µs', fontsize=11)
ax2.set_title('Zoom  M = 1..64 :  flat floor — HBM weight load dominates, '
              'marginal token ≈ free', fontsize=9.3, loc='left', color='#333333')
ax2.grid(True, alpha=0.20)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9,
           fontsize=8.5, edgecolor='#cccccc')

# tiny callout to a noise spike that the smoothing removes
i_spike = int(np.argmax(uf))
ax2.annotate('noise spike\n(removed by smoothing)', xy=(Mf[i_spike], uf[i_spike]),
             xytext=(Mf[i_spike]+9, uf[i_spike]+3), fontsize=7.2, color=GREEN,
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.8))

out_png = os.path.join(HERE, 'deepgemm_flat_dense.png')
fig.savefig(out_png, dpi=180, bbox_inches='tight')
fig.savefig('/workspace/EPLB/NEW_PAPER/figures/fig_deepgemm_flat_dense.png',
            dpi=180, bbox_inches='tight')
print('wrote', out_png)
print('wrote figures/fig_deepgemm_flat_dense.png')
