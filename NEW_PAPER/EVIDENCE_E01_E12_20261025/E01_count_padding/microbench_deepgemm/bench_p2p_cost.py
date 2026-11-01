#!/usr/bin/env python3
"""Live P2P cost (8xH20): swap (1-to-1 bidirectional, PB-OEPLB) vs duplication
(broadcast to K peers, EPLB). Expert weight = w13+w2 FP8 ~18.9MB/layer.
Uses dedicated streams + correct sync. All measured live."""
import os, json, statistics, torch
torch.cuda.set_device(0)
N_GPU = torch.cuda.device_count()
W13=(3072,4096); W2=(4096,1536)
EB=(W13[0]*W13[1]+W2[0]*W2[1])
print(f"[p2p] {N_GPU}xH20  expert={EB/1e6:.2f}MB FP8")
REPS,ROUNDS,WARM=2000,5,100

def buf(d): return torch.empty(W13, device=f'cuda:{d}', dtype=torch.float8_e4m3fn)

def time_fn(fn, sync_devs=(0,)):
    for _ in range(WARM): fn()
    for d in sync_devs: torch.cuda.synchronize(d)
    ts=[]
    for _ in range(ROUNDS):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(0); s.record()
        for _ in range(REPS): fn()
        for d in sync_devs: torch.cuda.synchronize(d)
        e.record(); torch.cuda.synchronize(0)
        ts.append(s.elapsed_time(e)*1e3/REPS)
    return statistics.median(ts)

# one-way single copy (baseline)
src=buf(0); dst=buf(1)
one = time_fn(lambda: dst.copy_(src, non_blocking=True))
print(f"[one-way 0->1] {one:.2f} us  ({EB/one/1e3:.0f} GB/s)")

# swap: 0<->1 bidirectional on dedicated streams (overlap)
s0=torch.cuda.Stream(0); s1=torch.cuda.Stream(1)
sa=buf(0); da=buf(1); sb=buf(1); db=buf(0)
def swap():
    with torch.cuda.stream(s0): da.copy_(sa, non_blocking=True)
    with torch.cuda.stream(s1): db.copy_(sb, non_blocking=True)
swp = time_fn(swap, sync_devs=(0,1))
print(f"[swap 0<->1 bidir] {swp:.2f} us  ({2*EB/swp/1e3:.0f} GB/s eff)")

# duplication: 0 -> K peers (broadcast from one source, bandwidth-bottlenecked)
dup={}
for K in range(1,N_GPU):
    dsts=[buf(i) for i in range(1,K+1)]; src=buf(0)
    def dup_k(dsts=dsts,src=src):
        st=torch.cuda.Stream(0)
        with torch.cuda.stream(st):
            for d in dsts: d.copy_(src, non_blocking=True)
    us=time_fn(dup_k, sync_devs=tuple(range(K+1)))
    dup[K]=us
    print(f"[dup broadcast 0->{K}] {us:.2f} us  ({K*EB/us/1e3:.0f} GB/s agg)")

out={'desc':'Live P2P expert-weight movement, 8xH20, FP8 expert w13+w2/layer',
     'expert_MB':round(EB/1e6,2),
     'oneway_us_0to1':round(one,2),
     'swap_us_bidir':round(swp,2),
     'dup_broadcast_us':{str(k):round(v,2) for k,v in dup.items()}}
op=os.path.join(os.path.dirname(os.path.abspath(__file__)),'p2p_cost.json')
json.dump(out,open(op,'w'),indent=2)
print(f"[p2p] wrote {op}")
