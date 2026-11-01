#!/usr/bin/env python3
"""E-P offline frozen-layout-quality comparison: PB-OEPLB heuristic
first-improvement planner vs P11 best_improvement_rank_pair planner.

Deterministic, CPU-only, NO timing measurements (planning cost is quantified
online from PROF planbuild logs). Inputs: the retained REAL calibration
routing counts (counts_calibration.npz, v3 setup lineage) and identity initial
placement (trivial map, as every service start). Both planners receive
identical budgets and acceptance margin. Refuses to overwrite.
"""
import importlib.util, json, hashlib, os, sys, datetime
import torch

EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUT=EXP+"/results/ep_v1/planner_offline_quality.json"
NPZ=EXP+"/results/fullservice_v3/setup/counts_calibration.npz"
REBAL="/workspace/EPLB/OEPLB/src/rebalancer.py"
NR, NL = 8, 16
THRESHOLD = 1.02
MAX_OPS_PER_BUILD = 300
MAX_BUILDS_ITERATED = 10
MIN_SWAP_OPS = 8

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()

spec=importlib.util.spec_from_file_location("rebalancer",REBAL)
rb=importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

def apply_plan(counts,p2l,plan):
    counts=counts.clone(); p2l=p2l.clone()
    for op in plan:
        l,a,b=op.layer_id,op.phys_slot_a,op.phys_slot_b
        ca,cb=counts[l][a].clone(),counts[l][b].clone()
        counts[l][a],counts[l][b]=cb,ca
        pa,pb=p2l[l][a].clone(),p2l[l][b].clone()
        p2l[l][a],p2l[l][b]=pb,pa
    return counts,p2l

def build_plan(fn,counts,p2l):
    return fn(logical_count=counts,physical_to_logical_map=p2l,num_ranks=NR,num_local=NL,
              threshold_ratio=THRESHOLD,max_swaps_per_layer=64,
              max_total_swap_layers=94,max_total_ops=MAX_OPS_PER_BUILD)

def layer_metrics(counts,p2l):
    """Per-layer: raw rank loads, padded per-rank workload (128-row blocks per
    logical expert, 0 blocks if n_e==0), W_max, r_raw, LB tokens."""
    L=counts.shape[0]
    rows={"W_max":[], "r_raw":[], "W_sum":[], "LB_tokens":[], "max_we":[]}
    for l in range(L):
        c=counts[l].tolist()
        raw=[sum(c[r*NL:(r+1)*NL]) for r in range(NR)]
        avg=max(sum(raw)/NR,1.0)
        w=[128*((n+127)//128) if n>0 else 0 for n in c]  # per logical expert padded rows
        wg=[sum(w[r*NL:(r+1)*NL]) for r in range(NR)]
        rows["W_max"].append(max(wg)); rows["r_raw"].append(max(raw)/avg)
        rows["W_sum"].append(sum(wg)); rows["max_we"].append(max(w))
        rows["LB_tokens"].append(max(sum(w)/NR, max(w)))
    return rows

def mean(x): return sum(x)/len(x)

def run_planner(name,fn,counts0,p2l0,mode):
    counts,p2l=counts0.clone(),p2l0.clone()
    builds=[]; total_ops=0
    for b in range(1 if mode=="single_build" else MAX_BUILDS_ITERATED):
        plan=build_plan(fn,counts,p2l)
        # determinism check on first build
        if b==0:
            plan2=build_plan(fn,counts0.clone(),p2l0.clone())
            assert list(map(tuple,plan))==list(map(tuple,plan2)), name+": nondeterministic"
        if not plan:
            builds.append({"build":b,"ops":0,"status":"empty"}); break
        counts,p2l=apply_plan(counts,p2l,plan)
        total_ops+=len(plan)
        builds.append({"build":b,"ops":len(plan),"status":"applied"})
        if mode=="single_build": break
        if len(plan)<MIN_SWAP_OPS:
            builds.append({"build":b+1,"ops":0,"status":"stopped_below_min_swap_ops"}); break
    m=layer_metrics(counts,p2l)
    return {
        "planner":name,"mode":mode,"builds":builds,"total_ops":total_ops,
        "aggregate":{
            "mean_layer_W_max":mean(m["W_max"]),
            "sum_layer_W_max":sum(m["W_max"]),
            "mean_layer_r_raw":mean(m["r_raw"]),
            "max_layer_r_raw":max(m["r_raw"]),
            "mean_layer_W_sum":mean(m["W_sum"]),
            "mean_unsplittability_price_tokens":mean([wm-lb for wm,lb in zip(m["W_max"],m["LB_tokens"])]),
            "max_unsplittability_price_tokens":max([wm-lb for wm,lb in zip(m["W_max"],m["LB_tokens"])]),
        },
        "per_layer":{"W_max":m["W_max"],"r_raw":m["r_raw"],"W_sum":m["W_sum"],"LB_tokens":m["LB_tokens"]},
    }

def main():
    if os.path.exists(OUT): sys.exit("REFUSE overwrite "+OUT)
    d=dict(torch.load(NPZ)) if False else None
    import numpy as np
    z=np.load(NPZ)
    P=torch.from_numpy(z["P"].astype("int64"))
    assert P.shape==(94,128), P.shape
    counts0=P.clone()
    p2l0=torch.arange(128,dtype=torch.int64).repeat(94,1)  # identity/trivial map
    planners=[("heuristic_first_improvement_PB_OEPLB",rb.try_build_swap_plan),
              ("best_improvement_rank_pair_P11",rb.try_build_swap_plan_best_improvement)]
    results=[]
    initial=layer_metrics(counts0,p2l0)
    init_agg={"mean_layer_W_max":mean(initial["W_max"]),"mean_layer_r_raw":mean(initial["r_raw"]),
              "max_layer_r_raw":max(initial["r_raw"]),"mean_unsplittability_price_tokens":
              mean([wm-lb for wm,lb in zip(initial["W_max"],initial["LB_tokens"])])}
    for name,fn in planners:
        for mode in ("single_build","iterated"):
            r=run_planner(name,fn,counts0,p2l0,mode)
            results.append(r)
            print(f"{name} [{mode}]: ops={r['total_ops']} meanWmax={r['aggregate']['mean_layer_W_max']:.1f} "
                  f"mean_r={r['aggregate']['mean_layer_r_raw']:.4f} maxprice={r['aggregate']['max_unsplittability_price_tokens']:.0f}")
    out={
        "artifact_id":"ep-v1-planner-offline-quality",
        "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "inputs":{"counts_npz":{"path":NPZ,"sha256":sha(NPZ),"note":"REAL calibration-request routing counts (v3 setup lineage); LOGICAL expert counts; identity p2l => physical==logical indexing"},
                  "rebalancer":{"path":REBAL,"sha256":sha(REBAL)}},
        "protocol":{"num_ranks":NR,"num_local":NL,"threshold_ratio":THRESHOLD,
                    "max_ops_per_build":MAX_OPS_PER_BUILD,"max_builds_iterated":MAX_BUILDS_ITERATED,
                    "min_swap_ops":MIN_SWAP_OPS,"initial_placement":"identity (trivial map)",
                    "padded_workload_rule":"w_e = 128*ceil(n_e/128), 0 if n_e==0; W_g = sum over slots on rank g",
                    "no_timing":"offline comparison is deterministic workload arithmetic only"},
        "initial_state_aggregate":init_agg,
        "results":results,
        "interpretation_rules":[
            "Layout quality is judged on final padded W_max and r_raw per layer (aggregated), NOT on timings.",
            "A planner is 'better layout' only if it is better on mean_layer_W_max AND mean_layer_r_raw in BOTH budget modes.",
            "Online net effect (campaign A11-P11 contrast) separates layout quality from planning/migration cost exposure; planbuild ms and issued-op counts come from the online logs.",
            "Equal offline quality + equal online time => rules behaviourally equivalent on this workload; report as such.",
            "Deterministic single runs: no CIs offline; differences are exact for these inputs."],
    }
    tmp=OUT+".tmp"
    json.dump(out,open(tmp,"w"),indent=1)
    os.rename(tmp,OUT)
    print("WROTE",OUT); print("sha256:",sha(OUT))

if __name__=="__main__":
    main()
