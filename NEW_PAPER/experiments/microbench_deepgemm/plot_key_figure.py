#!/usr/bin/env python3
"""THE key paper figure: DeepGEMM FP8 T(M) staircase = operator-level root cause
of the dead zone. Clean 0-256 (high-rep) + 256-1024. Decode hot-expert marker
(M~20, measured) sits in the flat floor; K=2 sharding of decode (20->10) stays
on floor (0 gain); prefill (600->300) crosses plateau (gain). Crossover at 256."""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
HERE=os.path.dirname(os.path.abspath(__file__))
c=json.load(open(f'{HERE}/deepgemm_flat_0_256_clean.json'))['results']
o=json.load(open(f'{HERE}/deepgemm_flat_dense.json'))['results']
r=sorted({x['M']:x for x in c+[y for y in o if y['M']>=256]}.values(),key=lambda z:z['M'])
M=np.array([x['M'] for x in r],dtype=float); U=np.array([x['us'] for x in r])
Us=medfilt(U,5)
def T(m):
    if m<=M[0]: return float(Us[0])
    if m>=M[-1]: return float(Us[-1])
    return float(Us[np.searchsorted(M,m)])
FLOOR=float(np.median(Us[M<=64]))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.linewidth':0.9,
    'grid.linewidth':0.5,'axes.edgecolor':'#444','xtick.color':'#444','ytick.color':'#444'})
NAVY='#1f3d7a';RED='#c0392b';GREEN='#2e7d32';GOLD='#b8860b';ROSE='#fdecea';BLUE='#e8eef7'
fig,ax=plt.subplots(figsize=(9.6,5.6),facecolor='white')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.fill_between([0,256],[25,25],[116,116],color=ROSE,alpha=0.6,zorder=0)
ax.fill_between([256,1024],[25,25],[116,116],color=BLUE,alpha=0.5,zorder=0)
ax.scatter(M,U,s=9,color=NAVY,alpha=0.22,zorder=2)
ax.plot(M,Us,'-',color=NAVY,lw=2.3,zorder=3)
ax.axhline(FLOOR,color=GOLD,ls=':',lw=1.2)
ax.text(6,FLOOR+1.6,f'flat floor ≈{FLOOR:.0f}µs',color=GOLD,fontsize=9,fontweight='bold')

# decode hot-expert marker (measured 满载 ~13-26)
ax.axvspan(13,26,color=RED,alpha=0.18,zorder=1)
ax.annotate('measured decode\nhot expert M≈13–26',xy=(20,T(20)),
    xytext=(55,T(20)+34),fontsize=8.6,color=RED,fontweight='bold',
    arrowprops=dict(arrowstyle='->',color=RED,lw=1.0))
# K=2 decode sharding: 20 -> 10, both floor
ax.plot([20],[T(20)],'o',color=RED,ms=9,zorder=5)
ax.annotate('',xy=(10,T(10)),xytext=(20,T(20)),
    arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.5,connectionstyle='arc3,rad=-0.35'))
ax.plot([10],[T(10)],'o',color=GREEN,ms=8,zorder=5)
ax.text(80,52,'K=2 uniform: 20→10/10\nboth on flat floor\n=> GEMM gain = 0\n(duplicate or swap: '
        'no benefit in decode)',fontsize=8.2,color=RED,fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35',fc='white',ec=RED,alpha=0.92))

# prefill example 600 -> 300
ax.plot([600],[T(600)],'o',color=RED,ms=9,zorder=5)
ax.annotate('M_hot=600 (prefill)',xy=(600,T(600)),xytext=(470,T(600)+13),
    fontsize=8.6,color=RED,fontweight='bold',ha='right',
    arrowprops=dict(arrowstyle='->',color=RED,lw=1.0))
ax.annotate('',xy=(300,T(300)),xytext=(600,T(600)),
    arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.5,connectionstyle='arc3,rad=0.35'))
ax.plot([300],[T(300)],'o',color=GREEN,ms=8,zorder=5)
ax.text(340,72,'K=2: 600→300\ncrosses plateau\n=> gain ≈29µs',fontsize=8.2,
    color=NAVY,fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35',fc='white',ec=NAVY,alpha=0.92))

for m0,lab in [(257,'+1 tile'),(513,'+1 tile'),(769,'wave 1→2')]:
    ax.axvline(m0,color=RED,ls='--',lw=0.8,alpha=0.55)
    ax.annotate(lab,xy=(m0,T(m0)),xytext=(6,4),textcoords='offset points',
        fontsize=7.2,color=RED)
for xm,txt in [(380,'≈52µs'),(640,'≈81µs'),(900,'≈100µs')]:
    ax.annotate(txt,xy=(xm,T(xm)),xytext=(0,9),textcoords='offset points',
        fontsize=7.6,color='#555',ha='center',style='italic')
ax.axvline(256,color='#555',ls='--',lw=1.1)
ax.text(258,112,'crossover\nM=256',fontsize=7.6,color='#555',va='top')
ax.text(6,111,'DECODE  M≤256  (flat floor = dead zone)',fontsize=8.8,color=RED,fontweight='bold',va='top')
ax.text(560,30,'PREFILL  M>256',fontsize=8.8,color=NAVY,fontweight='bold',va='bottom')
ax.set_xlim(0,1024); ax.set_ylim(25,116)
ax.set_xticks([0,64,128,192,256,384,512,640,768,896,1024])
ax.set_xlabel('tokens per expert  M  (per forward, per EP-rank)',fontsize=11)
ax.set_ylabel('FP8 GEMM time  T(M)  (µs)',fontsize=11)
ax.set_title('Operator-level root cause of the dead zone: DeepGEMM FP8 tile-padding '
             'flat floor (0–256) + 256-tile staircase\n'
             'H20 · Qwen3-235B w13 expert (K=4096, N=3072) · CUDA-event pure GPU time',
             fontsize=10.2,loc='left',fontweight='bold')
ax.grid(True,alpha=0.18)
out=f'{HERE}/deepgemm_key.png'
fig.savefig(out,dpi=190,bbox_inches='tight')
fig.savefig('/workspace/EPLB/NEW_PAPER/figures/fig_deepgemm_staircase.png',dpi=190,bbox_inches='tight')
print('wrote',out)
print('wrote figures/fig_deepgemm_staircase.png  (key paper figure)')
