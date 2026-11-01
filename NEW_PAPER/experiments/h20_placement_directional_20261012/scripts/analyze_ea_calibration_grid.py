#!/usr/bin/env python3
"""E-A calibration-grid analysis: C5-dynamic smoke signature verification +
fixed-parameter selection per the frozen protocol. Writes:
  results/ea_v1/grid_selection.json         (analysis; refuses overwrite)
  configs/followup_v1/ea_v1_fixed_params.json (frozen selection; refuses overwrite)
Selection uses ONLY calibration-request timings from valid grid starts.
Failed starts are retained and retired, never retried. No best-start selection.
"""
import json, hashlib, os, sys, datetime
EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF=EXP+"/configs/followup_v1"
OUT_A=EXP+"/results/ea_v1/grid_selection.json"
OUT_B=CONF+"/ea_v1_fixed_params.json"
STATE=EXP+"/results/ea_v1/calibration_grid/campaign_state.json"
PROT=CONF+"/ea_v1_calibration_grid_protocol.json"
LOCK=CONF+"/ea_v1_code_lock.json"
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()
for o in (OUT_A,OUT_B):
    if os.path.exists(o): sys.exit("REFUSE overwrite "+o)

prot=json.load(open(PROT)); lock=json.load(open(LOCK))
# code must still match the lock (no mid-campaign deployed changes)
assert sha(lock["controller_deployed"]["path"])==lock["controller_deployed"]["sha256"], "deployed controller changed after lock!"
state=json.load(open(STATE))
runs=[e for e in state if e.get("run")]

# ---------- C5-dynamic smoke verification ----------
smoke_expect={
 "easmoke_A10_s1":{"arg_adw_true":">=1","arg_add_true":"==0","reset":"==0","warmstart":"==0","calib_tier":"==0","refine":"==0","adw_decay0_window_fixed":"==0","note":"W mechanisms may fire (workload-dependent)"},
 "easmoke_A01_s1":{"arg_adw_true":"==0","arg_add_true":">=1","adw_shrink":"==0","adw_grow":"==0","window_shift":"==0","window_stable":"==0","warmstart":"==0","calib_tier":"==0","refine":"==0","note":"decay0_window_fixed and reset may fire (workload-dependent); window frozen"},
 "easmoke_A11_s1":{"arg_adw_true":">=1","arg_add_true":">=1","warmstart":"==0","calib_tier":"==0","refine":"==0","note":"v3-style signatures"},
}
def check(counts,exp):
    fails=[]
    for k,v in exp.items():
        if k=="note": continue
        got=counts.get(k)
        if v=="==0" and got!=0: fails.append(f"{k}={got} expected 0")
        if v==">=1" and not (got and got>=1): fails.append(f"{k}={got} expected >=1")
    return fails
smoke_results={}
for e in runs:
    if e.get("phase")=="smoke":
        m=e.get("mechanisms",{})
        fails=check(m,smoke_expect[e["run"]]) if e["run"] in smoke_expect else ["unknown smoke"]
        smoke_results[e["run"]]={"status":e.get("status"),"mechanisms":m,"expectation_failures":fails,
                                 "bench":e.get("bench",{}),"passed":not fails and e.get("status")=="completed"}
smoke_all_pass=all(v["passed"] for v in smoke_results.values()) and len(smoke_results)==3

# ---------- grid combo scores ----------
OK_REQUESTS=8037
combos={}
grid_rows=[]
for e in runs:
    if e.get("phase")!="grid": continue
    b=e.get("bench",{})
    valid=(e.get("status")=="completed" and e.get("bench_rc")==0
           and b.get("errors")==0 and b.get("ok")==OK_REQUESTS)
    m=e.get("mechanisms",{})
    frozen_ok=(m.get("adw_shrink",1)==0 and m.get("adw_grow",1)==0
               and m.get("window_shift",1)==0 and m.get("window_stable",1)==0
               and m.get("reset",1)==0 and m.get("warmstart",1)==0
               and m.get("calib_tier",1)==0 and m.get("refine",1)==0)
    row={"run":e["run"],"W":e["W"],"alpha":e["alpha"],"status":e.get("status"),
         "bench_rc":e.get("bench_rc"),"valid":valid,"fixed_arm_signature_ok":frozen_ok,
         "total_time_s":b.get("total_time_s"),"tps":b.get("tps"),"mechanisms":m}
    grid_rows.append(row)
    if valid:
        combos.setdefault((e["W"],e["alpha"]),[]).append(b["total_time_s"])
assert all(r["fixed_arm_signature_ok"] for r in grid_rows if r["status"]=="completed"), \
    "a fixed-combo start showed window/history mutations -- C5 DYNAMIC FAIL"
scored=[]
for (W,A),ts in sorted(combos.items()):
    scored.append({"W":W,"alpha":A,"n_valid_starts":len(ts),"times_s":ts,
                   "mean_time_s":sum(ts)/len(ts),"flag_n1":len(ts)<2})
ineligible=sorted({(r["W"],r["alpha"]) for r in grid_rows} - set(combos.keys()))
assert scored, "no valid combos"
best=sorted(scored,key=lambda c:(c["mean_time_s"],c["W"],c["alpha"],0 if (c["W"],c["alpha"])!=(16,0.5) else 1))[0]
W_STAR,A_STAR=best["W"],best["alpha"]

analysis={
 "artifact_id":"ea-v1-grid-selection",
 "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "protocol_sha256":sha(PROT),
 "c5_dynamic_smoke_verification":{"all_pass":smoke_all_pass,"per_smoke":smoke_results},
 "grid_rows":grid_rows,
 "combo_scores":scored,
 "ineligible_combos":[{"W":w,"alpha":a} for w,a in ineligible],
 "selection":{"W_star":W_STAR,"alpha_star":A_STAR,
   "rule":"argmin mean total_time_s over valid starts; tie-break smaller W, smaller alpha, (16,0.5) lineage",
   "n_valid_starts":best["n_valid_starts"],"mean_time_s":best["mean_time_s"]},
 "smoke_all_pass":smoke_all_pass,
}
if not smoke_all_pass:
    analysis["WARNING"]="smoke signature failures present -- DO NOT proceed to campaign freeze without resolution"
tmp=OUT_A+".tmp"; json.dump(analysis,open(tmp,"w"),indent=1); os.rename(tmp,OUT_A)
fixed={
 "artifact_id":"ea-v1-fixed-params",
 "status":"frozen_before_evaluation",
 "frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "W_star":W_STAR,"alpha_star":A_STAR,
 "selection_basis":{"protocol":sha(PROT),"analysis":sha(OUT_A),
   "data":"calibration requests only (8037 per start, concurrency 256); grid W x alpha 2 starts/combo",
   "mean_time_s":best["mean_time_s"],"n_valid_starts":best["n_valid_starts"],
   "all_combo_scores":scored},
 "arm_definitions":prot["arm_parameter_mapping_after_selection"],
 "never_revised_after":"any change requires a NEW frozen protocol and new campaign; this file is write-once",
}
tmp=OUT_B+".tmp"; json.dump(fixed,open(tmp,"w"),indent=1); os.rename(tmp,OUT_B)
print("smoke_all_pass:",smoke_all_pass)
for k,v in smoke_results.items(): print(" ",k,"passed=",v["passed"],"fails=",v["expectation_failures"])
print("valid grid starts:",sum(1 for r in grid_rows if r["valid"]),"/",len(grid_rows))
print("combos scored:",len(scored),"ineligible:",ineligible)
for c in scored: print(f"  W={c['W']:>2} alpha={c['alpha']} n={c['n_valid_starts']} mean={c['mean_time_s']:.2f}s {'<-- SELECTED' if (c['W'],c['alpha'])==(W_STAR,A_STAR) else ''}")
print("W*=",W_STAR,"alpha*=",A_STAR)
print("WROTE",OUT_A,sha(OUT_A)); print("WROTE",OUT_B,sha(OUT_B))
