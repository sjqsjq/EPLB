#!/usr/bin/env python3
"""Cost-benefit of duplicating a hot expert in decode — ALL live-measured.
Panel A: GEMM gain/forward (duplication bench) — decode ~0 (dead zone), prefill >0.
Panel B: P2P event cost — swap (fixed, PB-OEPLB) vs duplication (scales K, EPLB) + memory."""
import json, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

HERE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(HERE,'deepgemm_flat_dense.json')))
r=sorted(d['results'],key=lambda x:x['M'])
Mraw=np.array([x['M'] for x in r],dtype=float); Uraw=np.array([x['us'] for x in r]); Usm=medfilt(Uraw,5)
def us_at(m):
    if m<=Mraw[0]: return float(Usm[0])
    if m>=Mraw[-1]: return float(Usm[-1])
    return float(Usm[np.searchsorted(Mraw,m)])
p=json.load(open(os.path.join(HERE,'p2p_cost.json')))
swap_us=p['swap_us_bidir']; dup={int(k):v for k,v in p['dup_broadcast_us'].items()}; EMB=p['expert_MB']

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':0.9,
    'grid.linewidth':0.5,'axes.edgecolor':'#444','xtick.color':'#444','ytick.color':'#444'})
NAVY='#1f3d7a';RED='#c0392b';GREEN='#2e7d32';ORANGE='#e67e22';PURPLE='#6a4c93'
ROSE='#fdecea';BLUE='#e8eef7'
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,5.2),facecolor='white')
for ax in (ax1,ax2):
    for s in ('top','right'): ax.spines[s].set_visible(False)

# ---- Panel A: GEMM gain vs M_hot ----
ax1.axvspan(8,256,color=ROSE,alpha=0.7,zorder=0)
ax1.axvspan(256,1025,color=BLUE,alpha=0.6,zorder=0)
M_hots=list(range(8,1025,8))
for k,col in zip([2,4,8],[GREEN,ORANGE,PURPLE]):
    gain=[us_at(m)-us_at(math.ceil(m/k)) for m in M_hots]
    ax1.plot(M_hots,gain,'-',color=col,lw=1.9,label=f'K={k} replicas')
ax1.axhline(0,color='#555',lw=1.0)
ax1.text(18,62,'DECODE  M_hot <= 256\nGEMM gain ≈ 0\n(flat floor / dead zone)',
         fontsize=8.3,color=RED,fontweight='bold',va='top',
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=RED,alpha=0.92))
ax1.text(560,8,'PREFILL  M_hot > 256\nreal GEMM gain\n(crosses tile plateaus)',
         fontsize=8.3,color=NAVY,fontweight='bold',va='bottom',
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=NAVY,alpha=0.92))
ax1.set_xscale('log'); ax1.set_xlim(8,1025)
ax1.set_xticks([8,16,32,64,128,256,512,1024])
ax1.set_xticklabels(['8','16','32','64','128','256','512','1024'])
ax1.set_xlabel('hot-expert load  M_hot  (tokens/expert/forward)')
ax1.set_ylabel('GEMM time gain  (µs / forward)')
ax1.set_title('(A)  Benefit: GEMM gain from duplicating  =  T(M_hot) − T(M_hot/K)',
              fontsize=10.3,loc='left',fontweight='bold')
ax1.grid(True,which='both',alpha=0.16)
ax1.legend(loc='upper left',frameon=True,framealpha=0.92,fontsize=8.5,edgecolor='#ccc')

# ---- Panel B: P2P event cost, swap vs duplication + memory ----
Ks=sorted(dup.keys())
dup_v=[dup[k] for k in Ks]
x=np.arange(len(Ks))
bars=ax2.bar(x,dup_v,width=0.55,color=ORANGE,alpha=0.85,label='duplication (EPLB)')
for xi,k,v in zip(x,Ks,dup_v):
    ax2.text(xi,v+6,f'{v:.0f}µs',ha='center',fontsize=7.6,color=ORANGE,fontweight='bold')
    ax2.text(xi,v+30,f'+{k}×{EMB:.0f}MB\nmemory',ha='center',fontsize=6.8,color=RED)
ax2.axhline(swap_us,color=NAVY,ls='--',lw=2.2,
            label=f'swap (PB-OEPLB): {swap_us:.0f}µs, +0 memory')
ax2.text(len(Ks)-0.5,swap_us+8,f'swap {swap_us:.0f}µs\nfixed',color=NAVY,fontsize=8,
         ha='right',fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels([f'K={k}' for k in Ks])
ax2.set_ylim(0,300)
ax2.set_xlabel('replica count K  (broadcast expert to K cards)')
ax2.set_ylabel('P2P move cost  (µs / rebalance event)')
ax2.set_title('(B)  Cost: duplicating broadcast (scales with K) vs swap (fixed)',
              fontsize=10.3,loc='left',fontweight='bold')
ax2.grid(True,axis='y',alpha=0.18)
ax2.legend(loc='upper left',frameon=True,framealpha=0.92,fontsize=8.5,edgecolor='#ccc')

fig.suptitle('Duplicating a hot expert in DECODE is net-negative: benefit ≈ 0 (dead zone) '
             'while cost scales with K  — 8×H20, Qwen3-235B, all live-measured',
             fontsize=10.6,fontweight='bold',y=1.02)
fig.tight_layout()
out=os.path.join(HERE,'duplication_cost_benefit.png')
fig.savefig(out,dpi=180,bbox_inches='tight')
fig.savefig('/workspace/EPLB/NEW_PAPER/figures/fig_duplication_cost_benefit.png',dpi=180,bbox_inches='tight')
print('wrote',out)
print('wrote figures/fig_duplication_cost_benefit.png')
