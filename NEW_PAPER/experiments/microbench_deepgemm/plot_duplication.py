#!/usr/bin/env python3
"""Duplication negative-benefit figure (decode): GEMM gain from replicating a hot
expert to K cards. Derived from median-smoothed kernel curve (honest de-noised),
confirmed by raw bench. Decode (M_hot<=256): gain ~= 0 (flat floor). Prefill: gain>0."""
import json, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'deepgemm_flat_dense.json')))
r = sorted(d['results'], key=lambda x: x['M'])
Mraw = np.array([x['M'] for x in r], dtype=float)
Uraw = np.array([x['us'] for x in r], dtype=float)
Usm = medfilt(Uraw, 5)

def us_at(m):
    """smoothed GEMM time at token count m (interp on the dense curve)."""
    if m <= Mraw[0]: return float(Usm[0])
    if m >= Mraw[-1]: return float(Usm[-1])
    i = np.searchsorted(Mraw, m)
    # take the nearest sampled point at or above m (tile padding => ceil behavior)
    return float(Usm[i])

M_hots = list(range(8, 1025, 8))   # dense sweep of hot-expert load
Ks = [2, 4, 8]
floor = float(np.median(Usm[Mraw <= 64]))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.linewidth': 0.9, 'grid.linewidth': 0.5,
    'axes.edgecolor': '#444', 'xtick.color': '#444', 'ytick.color': '#444',
})
NAVY='#1f3d7a'; RED='#c0392b'; GREEN='#2e7d32'; ORANGE='#e67e22'; PURPLE='#6a4c93'
BLUEBG='#e8eef7'; ROSEBG='#fdecea'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.8), facecolor='white',
        gridspec_kw={'height_ratios':[1.15,1], 'hspace':0.34})
for ax in (ax1, ax2):
    for s in ('top','right'): ax.spines[s].set_visible(False)

# ---- top: straggler GEMM time vs M_hot, per K ----
ax1.axvspan(8, 256, color=ROSEBG, alpha=0.6, zorder=0)
ax1.axvspan(256, 1025, color=BLUEBG, alpha=0.55, zorder=0)
ax1.axhline(floor, color='#999', ls=':', lw=0.8)
for k, col in zip([1,2,4,8], [NAVY, GREEN, ORANGE, PURPLE]):
    ts = [us_at(math.ceil(m/k)) for m in M_hots]
    ax1.plot(M_hots, ts, '-', color=col, lw=1.9, label=f'K={k} replicas')
ax1.text(20, 105, 'DECODE\nM_hot <= 256\nall K collapse\nonto flat floor', fontsize=8,
         color=RED, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.9))
ax1.text(560, 105, 'PREFILL\nM_hot > 256\nmore replicas\n=> lower time', fontsize=8,
         color=NAVY, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NAVY, alpha=0.9))
ax1.set_xlim(8, 1025); ax1.set_ylim(25, 115)
ax1.set_xscale('log')
ax1.set_xticks([8,16,32,64,128,256,512,1024])
ax1.set_xticklabels(['8','16','32','64','128','256','512','1024'])
ax1.set_xlabel('hot-expert load  M_hot  (tokens/expert/forward)')
ax1.set_ylabel('straggler GEMM time  (µs)')
ax1.set_title('Replicating a hot expert to K cards: GEMM straggler time  (H20, Qwen3-235B w13)',
              fontsize=11.3, loc='left', fontweight='bold')
ax1.grid(True, which='both', alpha=0.16)
ax1.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=8.7, edgecolor='#ccc')

# ---- bottom: GEMM gain vs M_hot ----
ax2.axvspan(8, 256, color=ROSEBG, alpha=0.6, zorder=0)
ax2.axvspan(256, 1025, color=BLUEBG, alpha=0.55, zorder=0)
ax2.axhline(0, color='#555', lw=1.0)
for k, col in zip(Ks, [GREEN, ORANGE, PURPLE]):
    base = [us_at(m) for m in M_hots]
    rep  = [us_at(math.ceil(m/k)) for m in M_hots]
    gain = [b - r for b, r in zip(base, rep)]
    ax2.plot(M_hots, gain, '-', color=col, lw=1.9, label=f'K={k}')
# shade where gain <= ~noise (i.e. no real benefit)
ax2.text(20, 55, 'decode: GEMM gain ≈ 0\n(replicas all land in flat floor\n=> duplicating is pure cost)',
         fontsize=8.3, color=RED, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.9))
ax2.text(540, 8, 'prefill: real GEMM gain\n(crosses tile plateaus)', fontsize=8.3,
         color=NAVY, fontweight='bold', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NAVY, alpha=0.9))
ax2.set_xlim(8, 1025)
ax2.set_xscale('log')
ax2.set_xticks([8,16,32,64,128,256,512,1024])
ax2.set_xticklabels(['8','16','32','64','128','256','512','1024'])
ax2.set_xlabel('hot-expert load  M_hot  (tokens/expert/forward)')
ax2.set_ylabel('GEMM time gain  (µs)')
ax2.set_title('Net GEMM benefit of duplication  =  T(M_hot) − T(ceil(M_hot/K))',
              fontsize=10.6, loc='left', fontweight='bold')
ax2.grid(True, which='both', alpha=0.16)
ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.7, edgecolor='#ccc')

out_png = os.path.join(HERE, 'deepgemm_duplication.png')
fig.savefig(out_png, dpi=180, bbox_inches='tight')
fig.savefig('/workspace/EPLB/NEW_PAPER/figures/fig_duplication_negative_benefit.png', dpi=180, bbox_inches='tight')
print('wrote', out_png)
print('wrote figures/fig_duplication_negative_benefit.png')
