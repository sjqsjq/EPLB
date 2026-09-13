import os, json, statistics, torch
import deep_gemm as dg
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
K,N=4096,3072; REPS=1500; ROUNDS=10; WARM=50
torch.cuda.set_device(0)
b=torch.randn((N,K),device='cuda',dtype=torch.bfloat16)
bfp,sfb=per_block_cast_to_fp8(b,use_ue8m0=False); _=bfp.float()
def t(m):
    a=torch.randn((m,K),device='cuda',dtype=torch.bfloat16)
    afp,sfa=per_token_cast_to_fp8(a,use_ue8m0=False)
    d=torch.empty((m,N),device='cuda',dtype=torch.bfloat16)
    fn=lambda: dg.fp8_gemm_nt((afp,sfa),(bfp,sfb),d,None,recipe=(1,128,128),disable_ue8m0_cast=True)
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(ROUNDS):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS): fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1e3/REPS)
    return statistics.median(ts)
Ms=list(range(1,257,2))  # every 2 tokens, 128 pts, high-rep
res=[]
for i,m in enumerate(Ms):
    v=t(m); res.append({'M':m,'us':round(v,2)})
    if i%16==0: print(f"[{i}/{len(Ms)}] M={m} {v:.2f}",flush=True)
out={'kernel':'deep_gemm.fp8_gemm_nt','K':K,'N':N,'recipe':[1,128,128],'use_ue8m0':False,
     'timing':'CUDA event, 1500 reps x10 rounds median (high-rep clean)','desc':'clean 0-256 flat, high-rep','results':res}
json.dump(out,open('/workspace/EPLB/NEW_PAPER/experiments/microbench_deepgemm/deepgemm_flat_0_256_clean.json','w'),indent=2)
print("wrote clean json", len(res), "pts")
