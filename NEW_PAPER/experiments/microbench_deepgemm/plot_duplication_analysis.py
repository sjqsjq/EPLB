#!/usr/bin/env python3
"""THE duplication analysis figure: linear-x T(M) staircase + on-curve K=2 sharding.
For a FIXED hot-expert load M_hot, uniform sharding to K=2 -> each M_hot/2.
Benefit = T(M_hot) - T(M_hot/2). Decode M_hot (<=256): shards stay in flat floor
=> gain 0. Prefill M_hot (>256): crosses plateau boundary => gain > 0.
Selective sharding can't beat uniform (straggler-bound), and below 256 no split helps."""
import json, os, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

HERE=os.path.dirname(os.path.abspath(__file__))
c=json.load(open(os.path.join(HERE,'deepgemm_flat_0_256_clean.json')))['results']
o=json.load(open(os.path.join(HERE,'deepgemm_flat_dense.json')))['results']
r=sorted({x['M']:x for x in c+[y for y in o if y['M']>=256]}.values(),key=lambda z:z['M'])
M=np.array([x['M'] for x in r],dtype=float); U=np.array([x['us'] for x in r])
Us=medfilt(U,5)
def T(m):
    if m<=M[0]: return float(Us[0])
    if m>=M[-1]: return float(Us[-1])
    i=np.searchsorted(M,m); return float(Us[i])
FLOOR=float(np.median(Us[M<=64]))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':0.9,
    'grid.linewidth':0.5,'axes.edgecolor':'#444','xtick.color':'#444','ytick.color':'#444'})
NAVY='#1f3d7a';RED='#c0392b';GREEN='#2e7d32';ORANGE='#e67e22';PURPLE='#6a4c93'
ROSE='#fdecea';BLUE='#e8eef7';GOLD='#b8860b'

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.4),facecolor='white',
        gridspec_kw={'width_ratios':[1.5,1],'wspace':0.22})
for ax in (ax1,ax2):
    for s in ('top','right'): ax.spines[s].set_visible(False)

# ===== Panel A: T(M) linear, on-curve K=2 sharding for 2 fixed M_hot =====
ax1.fill_between([0,256],[25,25],[115,115],color=ROSE,alpha=0.65,zorder=0)
ax1.fill_between([256,1024],[25,25],[115,115],color=BLUE,alpha=0.55,zorder=0)
ax1.scatter(M,U,s=8,color=NAVY,alpha=0.28,zorder=2)
ax1.plot(M,Us,'-',color=NAVY,lw=2.1,zorder=3)
ax1.axhline(FLOOR,color=GOLD,ls=':',lw=1.1)
ax1.text(8,FLOOR+1.5,f'flat floor {FLOOR:.0f}µs',color=GOLD,fontsize=8,fontweight='bold')

# --- decode example: M_hot = 64 (typical decode hot expert) ---
mh_d=20
ax1.plot([mh_d],[T(mh_d)],'o',color=RED,ms=9,zorder=5)
ax1.annotate('M_hot=20\n(measured decode hot\nexpert, peak batch 13)',xy=(mh_d,T(mh_d)),
    xytext=(mh_d+18,T(mh_d)+30),fontsize=8,color=RED,fontweight='bold',
    arrowprops=dict(arrowstyle='->',color=RED,lw=0.9))
# K=2 uniform -> 32
ax1.annotate('',xy=(10,T(10)),xytext=(mh_d,T(mh_d)),
    arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.4,connectionstyle='arc3,rad=-0.3'))
ax1.plot([10],[T(10)],'o',color=GREEN,ms=8,zorder=5)
ax1.text(11,T(10)+2,'10',color=GREEN,fontsize=7.5,fontweight='bold')
ax1.text(40,40,'K=2 uniform: both shards\nland on flat floor\n=> gain = 0',
    fontsize=7.8,color=RED,fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=RED,alpha=0.9))

# --- prefill example: M_hot = 600 ---
mh_p=600
ax1.plot([mh_p],[T(mh_p)],'o',color=RED,ms=9,zorder=5)
ax1.annotate('M_hot=600\n(prefill)',xy=(mh_p,T(mh_p)),
    xytext=(mh_p-40,T(mh_p)+14),fontsize=8,color=RED,fontweight='bold',ha='right',
    arrowprops=dict(arrowstyle='->',color=RED,lw=0.9))
ax1.annotate('',xy=(300,T(300)),xytext=(mh_p,T(mh_p)),
    arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.4,connectionstyle='arc3,rad=0.3'))
ax1.plot([300],[T(300)],'o',color=GREEN,ms=8,zorder=5)
ax1.text(305,T(300)+2,'300',color=GREEN,fontsize=7.5,fontweight='bold')
ax1.text(360,70,'K=2: crosses plateau\n=> gain = %.0fµs'%(T(mh_p)-T(300)),
    fontsize=7.8,color=NAVY,fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=NAVY,alpha=0.9))

ax1.axvline(256,color='#555',ls='--',lw=1.0)
ax1.text(258,108,'crossover\nM_hot=256',fontsize=7.5,color='#555',va='top')
ax1.text(20,110,'DECODE\nM_hot<=256',fontsize=8,color=RED,fontweight='bold',va='top')
ax1.text(600,30,'PREFILL  M_hot>256',fontsize=8,color=NAVY,fontweight='bold',va='bottom')
ax1.set_xlim(0,1024); ax1.set_ylim(25,115)
ax1.set_xticks([0,128,256,384,512,640,768,896,1024])
ax1.set_xlabel('tokens per hot expert  M_hot')
ax1.set_ylabel('GEMM time  T(M)  (µs)')
ax1.set_title('(A)  On the T(M) curve: does K=2 uniform sharding help?',
              fontsize=10.5,loc='left',fontweight='bold')
ax1.grid(True,alpha=0.18)

# ===== Panel B: gain vs M_hot, K=2, with crossover band =====
Mh=list(range(4,1025,4))
g=[T(m)-T(math.ceil(m/2)) for m in Mh]
ax2.fill_between([0,256],[-5,-5],[75,75],color=ROSE,alpha=0.6,zorder=0)
ax2.fill_between([256,1024],[-5,-5],[75,75],color=BLUE,alpha=0.5,zorder=0)
ax2.plot(Mh,g,'-',color=PURPLE,lw=2.2)
ax2.axhline(0,color='#555',lw=1.0)
ax2.axvline(256,color='#555',ls='--',lw=1.0)
ax2.text(20,60,'DECODE  M_hot<=256\nGEMM gain = 0\n(any split stays in\nflat floor; '
         'selective cannot help\n— straggler-bound)',
         fontsize=7.8,color=RED,fontweight='bold',va='top',
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=RED,alpha=0.92))
ax2.text(560,4,'PREFILL  M_hot>256\nreal gain\n(crosses plateau boundary)\n'
         'uniform split is optimal\n(selective <= uniform)',
         fontsize=7.8,color=NAVY,fontweight='bold',va='bottom',
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=NAVY,alpha=0.92))
ax2.set_xlim(0,1024); ax2.set_ylim(-5,70)
ax2.set_xticks([0,128,256,384,512,640,768,896,1024])
ax2.set_xlabel('hot-expert load  M_hot')
ax2.set_ylabel('GEMM gain  =  T(M_hot) − T(M_hot/2)  (µs)')
ax2.set_title('(B)  Duplication payoff: crossover at M_hot = 256',
              fontsize=10.5,loc='left',fontweight='bold')
ax2.grid(True,alpha=0.18)

fig.suptitle('Duplicating a hot expert helps ONLY when M_hot > 256 (above flat floor)  —  '
             'decode hot experts sit in the dead zone',
             fontsize=11,fontweight='bold',y=1.01)
fig.tight_layout()
out=os.path.join(HERE,'duplication_analysis.png')
fig.savefig(out,dpi=180,bbox_inches='tight')
fig.savefig('/workspace/EPLB/NEW_PAPER/figures/fig_duplication_analysis.png',dpi=180,bbox_inches='tight')
print('wrote',out)
print('wrote figures/fig_duplication_analysis.png')
