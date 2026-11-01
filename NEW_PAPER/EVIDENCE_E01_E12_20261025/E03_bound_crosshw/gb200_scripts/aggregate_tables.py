import json, glob, statistics as st, os
RES="/workspace/EPLB/OEPLB/benchmarks/results"
methods=[("identity","identity"),("datafore","DataForest-Remap"),("eplb_static","EPLB静态(16冗余)"),
         ("eplb_dyn","EPLB动态(16冗余)"),("oeplb","PB-OEPLB"),("moetuner","MoETuner(ILP1)")]
def med(pat):
    v=[]
    for f in sorted(glob.glob(pat)):
        try: v.append(json.load(open(f))["tps"])
        except: pass
    return (st.median(v) if v else None), v
print("="*66)
print("表1 同域 (prover_256tok, N=256 conc=256 O=1, 5次中位)  [EP=4/GB200]")
print("="*66)
t1={}
print(f"{'方法':<20}{'req/s':>9}{'vs identity':>13}   各次")
base=None
for tag,name in methods:
    m,v=med(f"{RES}/_0914_t1_{tag}_r*.json")
    t1[tag]=m
    if tag=="identity": base=m
    if m is None: print(f"{name:<20}{'—':>9}"); continue
    g="" if base is None or tag=="identity" else f"{(m/base-1)*100:+.1f}%"
    print(f"{name:<20}{m:>9.1f}{g:>13}   {[round(x,1) for x in v]}")
print()
print("="*66)
print("表2 跨域 (freq6: 6段book↔prover, N=1800 conc=32 O=10, 2次中位)  [EP=4/GB200]")
print("="*66)
t2={}
print(f"{'方法':<20}{'req/s':>9}{'vs identity':>13}   各次")
base=None
for tag,name in methods:
    m,v=med(f"{RES}/_freq6_t2_{tag}_r*.json")
    t2[tag]=m
    if tag=="identity": base=m
    if m is None: print(f"{name:<20}{'—':>9}"); continue
    g="" if base is None or tag=="identity" else f"{(m/base-1)*100:+.1f}%"
    print(f"{name:<20}{m:>9.2f}{g:>13}   {[round(x,2) for x in v]}")
json.dump({"t1":t1,"t2":t2}, open("/workspace/logs/tables_agg.json","w"), indent=2)
