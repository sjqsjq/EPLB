#!/usr/bin/env python3
"""Log-based checks (existing campaign logs only, read-only):
1) W deviation: reconstruct effective sync_window trajectory per start (ADW/WINDOW events),
   time-weighted fraction at each W; cross-check against tok_global of issued windows
   (tok_global = fwds*8192*94layers*8topk -> W_obs = tok_global/603979776... exactly /8192/94/8).
2) Reset classification for D-arms: boundary-locked vs mid-segment, large vs threshold-level delta.
   Descriptive classification (bimodal separation at 0.10); ground truth of 'false' fires is NOT
   observable from logs - wording kept as 'threshold-level, noise-consistent'.
3) Cumulative per-start controller-side accounting: last PROF entry (cumulative record/allreduce/
   planbuild/finalize), migration walls (n, sum, max - reported alongside, never subtracted from
   or added to service time), issued swaps.
"""
import json, re, statistics, collections
from datetime import datetime, timezone

EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUT=f"{EXP}/results/ea_v1/workload_variation"
tl=json.load(open(f"{OUT}/timeline_events.json"))
TOKPER=(8192*94*8)   # assignments per forward, all layers
bounds=tl["boundaries_cum_token_frac"]
ARMS=["A00","A10","A01","A11","P11"]

def parse_ts(s): return datetime.fromisoformat(s).timestamp()

res={"w_trajectories":{},"reset_classification":{},"cumulative_costs":{},"notes":[
 "W reconstruction: event-based from DP0 ADW 'sw to N' and WINDOW 'X -> Y' lines; time-weighted over bench [t0,t1]",
 "tok_global cross-check: W_obs=tok_global/(8192*94*8) at issued windows (full windows only)",
 "reset classes: A=boundary-zone(|frac-b|<=0.03) & delta>=0.10; B=boundary-zone & delta<0.10; C=mid-segment & delta>=0.10; D=mid-segment & delta<0.10 (threshold-level, noise-consistent; 'false' not provable from logs)",
 "cumulative costs are controller-side bookkeeping/migration wall times, reported ALONGSIDE service time; never subtracted from measured contrasts (frozen rule)"]}

wfrac_agg=collections.defaultdict(lambda: collections.defaultdict(list))
mismatch_tot=0; check_tot=0
for arm in ARMS:
    for r in range(1,6):
        run=f"eacR{r}_{arm}"; s=tl["starts"][run]
        t0=parse_ts(s["t0_utc"]); t1=t0+s["dur_s"]
        # W trajectory
        ev=[]
        for e in s["events"]["adw"]:
            m=re.search(r"sw to (\d+)",e.get("detail") or "")
            if m and e["kind"] in("shrink","grow"): ev.append((e["ts"],int(m.group(1))))
        for e in s["events"]["window"]:
            if e.get("to"): ev.append((e["ts"],int(e["to"])))
        ev=sorted(set(ev))
        W=16; cur=t0; seg={16:0.0}
        for ts,w in ev:
            ts=min(max(ts,t0),t1)
            seg[W]=seg.get(W,0.0)+(ts-cur); W=w; cur=ts
        seg[W]=seg.get(W,0.0)+(t1-cur)
        tot=sum(seg.values()); frac={k:round(v/tot,3) for k,v in seg.items() if v>0}
        # tok cross-check
        mm=0; ck=0
        for e in s["events"]["issued"]:
            if e["tok_global"]%TOKPER==0 and e["tok_global"]>0:
                wobs=e["tok_global"]//TOKPER; ck+=1
                # find reconstructed W at that ts
                Wr=16
                for ts,w in ev:
                    if ts<=e["ts"]: Wr=w
                    else: break
                if wobs!=Wr: mm+=1
        mismatch_tot+=mm; check_tot+=ck
        res["w_trajectories"][run]={"w_time_frac":frac,"n_changes":len(ev),
            "tok_crosscheck":{"windows_checked":ck,"mismatches":mm}}
        for w,f in frac.items(): wfrac_agg[arm][w].append(f)
    # reset classification
    cls=collections.Counter(); deltas={"A":[],"D":[],"C":[]}
    boundaries_hit=collections.Counter()
    for r in range(1,6):
        run=f"eacR{r}_{arm}"; s=tl["starts"][run]
        for e in s["events"]["reset"]:
            d=e["current"]-e["prev"]; z=e["zone"]
            b=z.startswith("boundary")
            k=("A" if b and d>=0.10 else "B" if b else "C" if d>=0.10 else "D")
            cls[k]+=1
            if k in deltas: deltas[k].append(round(d,3))
            if b and d>=0.10: boundaries_hit[z]+=1
    tot=sum(cls.values())
    if tot:
        res["reset_classification"][arm]={"total":tot,"classes":dict(cls),
            "class_fractions":{k:round(v/tot,3) for k,v in cls.items()},
            "distinct_boundary_zones_with_large_delta":dict(boundaries_hit),
            "delta_samples":{"A":sorted(deltas["A"])[:12],"C":sorted(deltas["C"])[:12],"D":sorted(deltas["D"])[:12]}}

# cumulative costs
diag_dir=f"{EXP}/results/ea_v1/fivearm_campaign/diag"
state=json.load(open(f"{EXP}/results/ea_v1/fivearm_campaign/campaign_state.json"))
bench_by_run={e["run"]:e["bench"] for e in state if e.get("phase")=="campaign" and "run" in e}
for arm in ARMS:
    rows=[]
    for r in range(1,6):
        run=f"eacR{r}_{arm}"
        d=json.load(open(f"{diag_dir}/{run}_diag.json"))
        prof=d["prof_dp0"][-1] if d["prof_dp0"] else None
        walls=d["wall_summary"]
        iss=sum((e[1] if isinstance(e,list) else e["swaps"]) for e in d["issued_windows_dp0"])
        rows.append({"run":run,"service_s":bench_by_run[run]["total_time_s"],
            "prof_cum_s":{"record":round(prof["record_ms"]/1000,2),"allreduce":round(prof["allreduce_ms"]/1000,2),
                          "planbuild":round(prof["planbuild_ms"]/1000,3),"finalize":round(prof["finalize_ms"]/1000,2),
                          "total":round(prof["total_ms"]/1000,2)} if prof else None,
            "walls":{"n":walls["n_events"],"sum_ms":round(walls["sum_wall_ms"],0),"max_ms":round(walls["max_wall_ms"],0)},
            "issued_swaps_dp0":iss,"swap_ops":d["swap_stats"]["n_ops"],
            "reversals":d["swap_stats"]["n_reversals"],"repeats":d["swap_stats"]["n_repeats"]})
    def agg(f): 
        v=[f(x) for x in rows]; return {"mean":round(statistics.mean(v),2),"min":round(min(v),2),"max":round(max(v),2)}
    res["cumulative_costs"][arm]={"per_start":rows,"summary":{
        "service_s":agg(lambda x:x["service_s"]),
        "prof_total_cum_s":agg(lambda x:x["prof_cum_s"]["total"]),
        "prof_record_cum_s":agg(lambda x:x["prof_cum_s"]["record"]),
        "prof_planbuild_cum_s":agg(lambda x:x["prof_cum_s"]["planbuild"]),
        "wall_sum_s":agg(lambda x:x["walls"]["sum_ms"]/1000),
        "wall_max_s":agg(lambda x:x["walls"]["max_ms"]/1000),
        "wall_n":agg(lambda x:x["walls"]["n"]),
        "issued_swaps_dp0":agg(lambda x:x["issued_swaps_dp0"]),
        "swap_ops":agg(lambda x:x["swap_ops"])}}
res["tok_crosscheck_total"]={"windows_checked":check_tot,"mismatches":mismatch_tot}
res["w_time_frac_by_arm"]={a:{str(w):{"mean":round(statistics.mean(v),3),"min":round(min(v),3),"max":round(max(v),3)}
    for w,v in d.items()} for a,d in wfrac_agg.items()}
json.dump(res,open(f"{OUT}/log_check_w_reset_cost.json","w"),indent=1)

print("=== W time-fraction by arm (mean over 5 starts) ===")
for a in ARMS: print(a, {w:round(statistics.mean(v),3) for w,v in wfrac_agg[a].items()}, "changes/start:", round(statistics.mean([res['w_trajectories'][f'eacR{r}_{a}']['n_changes'] for r in range(1,6)]),1))
print("tok cross-check:",res["tok_crosscheck_total"])
print("\n=== reset classification ===")
for a in ["A01","A11","P11"]: print(a,json.dumps(res["reset_classification"][a]))
print("\n=== cumulative costs (mean over starts) ===")
for a in ARMS:
    s=res["cumulative_costs"][a]["summary"]
    print(f"{a}: service {s['service_s']['mean']}s | prof_cum total {s['prof_total_cum_s']['mean']}s (record {s['prof_record_cum_s']['mean']}, planbuild {s['prof_planbuild_cum_s']['mean']}) | walls n{s['wall_n']['mean']} sum {s['wall_sum_s']['mean']}s max {s['wall_max_s']['mean']}s | swaps issued_dp0 {s['issued_swaps_dp0']['mean']} ops {s['swap_ops']['mean']}")
