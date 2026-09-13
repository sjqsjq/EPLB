#!/usr/bin/env python3
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'deepgemm_flat_dense.json')))
r = d['results']
M  = np.array([x['M']  for x in r])
us = np.array([x['us'] for x in r])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.4),
                                gridspec_kw={'height_ratios': [2.1, 1]})

# ---- top: full staircase 1..1024 ----
ax1.plot(M, us, '-', color='#1f4e79', lw=1.0, alpha=0.55, zorder=1)
ax1.scatter(M, us, s=7, color='#1f4e79', zorder=2)

# annotate the block switches (M, block change, color)
jumps = [
    (65,  'BN 48→80',  '#2e7d32'),
    (129, 'BM 64→128', '#2e7d32'),
    (257, 'BM 128→256\n(256-tile pad)', '#c0392b'),
    (513, 'BN 80→128', '#c0392b'),
    (769, 'waves 1→2', '#c0392b'),
]
for m, label, col in jumps:
    v = us[np.argmin(np.abs(M - m))]
    ax1.axvline(m, color=col, ls='--', lw=0.7, alpha=0.6, zorder=0)
    # place text near the step
    ax1.annotate(label, xy=(m, v), xytext=(8, 6), textcoords='offset points',
                 fontsize=8, color=col, fontweight='bold')

ax1.set_xscale('log')
ax1.set_xlim(0.9, 1100)
ax1.set_xticks([1,2,4,8,16,32,64,128,256,512,1024])
ax1.set_xticklabels(['1','','','','16','','64','128','256','512','1024'])
ax1.set_ylabel('GEMM time per call (µs)')
ax1.set_title('DeepGEMM FP8 kernel staircase (H20, K=4096 N=3072, Qwen3-235B w13 expert)\n'
              'pure GPU-execution time (CUDA events, cast excluded)', fontsize=10.5, loc='left')
ax1.grid(True, which='both', alpha=0.18)

# shade the flat regime
ax1.axvspan(0.9, 64, color='#fff3cd', alpha=0.5, zorder=0)
ax1.text(2.2, 95, 'FLAT floor\n~33 µs', fontsize=8.5, color='#8a6d00',
         fontweight='bold', va='top')

# ---- bottom: zoom flat region 1..64 ----
mask = M <= 64
Mf, uf = M[mask], us[mask]
ax2.plot(Mf, uf, '-o', color='#1f4e79', lw=1.0, ms=4, alpha=0.8)
ax2.axhline(np.median(uf), color='#c0392b', ls='--', lw=0.9, alpha=0.7,
            label=f'median = {np.median(uf):.1f} µs')
ax2.set_xlim(0, 65)
ax2.set_xticks([1,8,16,24,32,40,48,56,64])
ax2.set_ylim(min(uf)-2, max(uf)+4)
ax2.set_xlabel('M  (tokens per expert)')
ax2.set_ylabel('µs')
ax2.set_title('Zoom: M=1..64 flat region — adding tokens is free '
              '(HBM weight load dominates)', fontsize=9.5, loc='left')
ax2.grid(True, alpha=0.22)
ax2.legend(loc='upper right', frameon=False, fontsize=8.5)

plt.tight_layout()
out_png = os.path.join(HERE, 'deepgemm_flat_dense.png')
fig.savefig(out_png, dpi=170, bbox_inches='tight')
# also copy to paper figures
fig_dir = '/workspace/EPLB/NEW_PAPER/figures'
fig.savefig(os.path.join(fig_dir, 'fig_deepgemm_flat_dense.png'), dpi=170, bbox_inches='tight')
print('wrote', out_png)
print('wrote', os.path.join(fig_dir, 'fig_deepgemm_flat_dense.png'))
