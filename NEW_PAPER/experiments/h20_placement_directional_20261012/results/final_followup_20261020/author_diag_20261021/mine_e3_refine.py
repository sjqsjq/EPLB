#!/usr/bin/env python3
import json, glob, os, re, calendar, time, statistics as st
G = '/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020/phase4_core/boots'
REF = 6.7419
win_re = re.compile(r'^\[([\d\-]+ [\d:]+) DP\d+.*window#(\d+): issued (\d+) swap')
sp_re  = re.compile(r'S=(\d+)[/ ]H=(\d+)[/ ]W=(\d+)')

def pct(v,q):
    v=sorted(v); i=(len(v)-1)*q/100; lo=int(i); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(i-lo)

buckets = {'0-3s':[], '3-8s':[], '8-20s':[], '>20s':[]}
over_in = {k:0 for k in buckets}; n_in = {k:0 for k in buckets}
smax_per_boot = {}
for b in sorted(glob.glob(os.path.join(G,'s_E3_P_b*'))):
    lab=os.path.basename(b)
    rows=[json.loads(l) for l in open(os.path.join(b,lab+'_perreq.jsonl'))]
    wins={}
    smax=1; slines=0
    for line in open(os.path.join(b,'server.log'),errors='ignore'):
        m=win_re.match(line)
        if m:
            ts=calendar.timegm(time.strptime(m.group(1),'%Y-%m-%d %H:%M:%S'))
            wins.setdefault(int(m.group(2)),[]).append(ts)
        s=sp_re.search(line)
        if s:
            slines+=1; smax=max(smax,int(s.group(1)))
    smax_per_boot[lab]=(smax,slines)
    wt=sorted(st.median(v) for v in wins.values())
    for r in rows:
        mid=r['t_send_wall_s']+r['ttft_s']/2
        d=min(abs(mid-w) for w in wt) if wt else 1e9
        k='0-3s' if d<=3 else '3-8s' if d<=8 else '8-20s' if d<=20 else '>20s'
        buckets[k].append(r['ttft_s']); n_in[k]+=1
        if r['ttft_s']>REF: over_in[k]+=1

print('P arm: TTFT by distance to nearest swap-window issuance')
print('| dist | n | p50 | p90 | p95 | p99 | over_ref% |')
print('|---|---|---|---|---|---|---|')
for k in ['0-3s','3-8s','8-20s','>20s']:
    v=buckets[k]
    if not v: continue
    print(f'| {k} | {len(v)} | {pct(v,50):.2f} | {pct(v,90):.2f} | {pct(v,95):.2f} | {pct(v,99):.2f} | {100*over_in[k]/n_in[k]:.1f} |')
print()
print('per-boot S_max (SP state lines seen):')
for lab,(smax,sl) in smax_per_boot.items():
    p95=pct([json.loads(l)['ttft_s'] for l in open(os.path.join(G,lab,lab+'_perreq.jsonl'))],95)
    print(f'  {lab}: S_max={smax} sp_lines={sl} ttft_p95={p95:.2f}')
# ramp vs steady for both arms
print()
for arm in 'PI':
    ramp=[];steady=[]
    for b in sorted(glob.glob(os.path.join(G,f's_E3_{arm}_b*'))):
        lab=os.path.basename(b)
        rows=[json.loads(l) for l in open(os.path.join(b,lab+'_perreq.jsonl'))]
        t0=min(r['t_send_wall_s'] for r in rows)
        dur=max(r['t_last_wall_s'] for r in rows)-t0
        r40=sum(1 for r in rows if r['t_last_wall_s']-t0<=40)
        rs=sum(1 for r in rows if r['t_last_wall_s']-t0>40)
        ramp.append(r40/40); steady.append(rs/(dur-40) if dur>40 else 0)
    print(f'{arm}: ramp(first40s)={st.mean(ramp):.1f} req/s  steady={st.mean(steady):.1f} req/s')
