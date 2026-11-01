#!/usr/bin/env python3
"""Five-arm campaign analysis (A00/A10/A01/A11/P11), executes the frozen
statistics plan in ea_v1_campaign_protocol.json:
 - replicate unit = independent formal start (5/arm); rounds = paired blocks
 - primary contrasts (pre-fixed): A11-A00 (net), A11-P11 (planner rule)
 - secondary: A11-A10 (D increment), A11-A01 (W increment), WxD interaction
 - paired per-round diffs -> mean +/- t(0.975, df) 95% CI
 - negative effects preserved exactly; CI crossing zero => "compatible with
   zero", NOT proof of no effect, NOT equivalence
 - migration wall reported ALONGSIDE per start; never subtracted; not
   automatically a critical-path addition
 - no best-start selection, no imputation, no significance-driven extra runs
Writes results/ea_v1/fivearm_campaign_analysis.json (+ summary.md). Refuses overwrite.
"""
import json, hashlib, os, sys, datetime, math

EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTJ=os.path.join(EXP,"results/ea_v1/fivearm_campaign_analysis.json")
OUTM=os.path.join(EXP,"results/ea_v1/fivearm_campaign_summary.md")
STATE=os.path.join(EXP,"results/ea_v1/fivearm_campaign/campaign_state.json")
PROT=os.path.join(EXP,"configs/followup_v1/ea_v1_campaign_protocol.json")
for p in (OUTJ,OUTM):
    if os.path.exists(p): sys.exit("REFUSE to overwrite "+p)
prot=json.load(open(PROT)); state=json.load(open(STATE))
OK_REQUESTS=prot["workload"]["n_requests"]
ARMS=["A00","A10","A01","A11","P11"]
T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

runs=[e for e in state if e.get("phase")=="campaign"]
by_label={e["run"]:e for e in runs}
# schedule coverage check
sched=prot["schedule"]
missing=[s["label"] for s in sched if s["label"] not in by_label]
# validity
def valid(e):
    b=e.get("bench",{})
    return (e.get("status")=="completed" and e.get("boot_rc")==0
            and e.get("config_verify")=="CONFIG_OK" and e.get("bench_rc")==0
            and b.get("errors")==0 and b.get("ok")==OK_REQUESTS
            and e.get("mechanisms",{}).get("tracebacks",1)==0)
rows={}
for s in sched:
    e=by_label.get(s["label"])
    if e is None: continue
    v=valid(e); b=e.get("bench",{})
    rows[(s["arm"],s["round"])]={"label":s["label"],"valid":v,"status":e.get("status"),
        "total_time_s":b.get("total_time_s"),"tps":b.get("tps"),
        "ok":b.get("ok"),"errors":b.get("errors"),"lat_mean_ms":b.get("lat_mean_ms"),
        "diag_file":e.get("diag_file"),"diag_sha256":e.get("diag_sha256"),
        "mechanisms":e.get("mechanisms",{})}
# per-arm summaries + diag-derived descriptors
arm_summary={}; diag_detail={}
for a in ARMS:
    times=[rows[(a,r)]["total_time_s"] for r in range(1,6) if (a,r) in rows and rows[(a,r)]["valid"] and rows[(a,r)]["total_time_s"] is not None]
    n_all=len([r for r in range(1,6) if (a,r) in rows])
    arm_summary[a]={"n_starts_recorded":n_all,"n_valid":len(times),
        "times_s":sorted(times),
        "mean_time_s":(sum(times)/len(times)) if times else None,
        "sd_time_s":(math.sqrt(sum((x-sum(times)/len(times))**2 for x in times)/(len(times)-1))) if len(times)>1 else None,
        "min_time_s":min(times) if times else None,"max_time_s":max(times) if times else None}
    # aggregate mechanisms + diag/wall from per-start diag files
    mech_tot={}; planb=[]; walls=[]; reversals=0; repeats=0; ops=0; ratios_after=[]; maxratios_after=[]; finals=[]; resets=0; win_events=0; issued_ops=0
    for r in range(1,6):
        if (a,r) not in rows or not rows[(a,r)]["valid"]: continue
        e=by_label[rows[(a,r)]["label"]]
        for k,vv in e.get("mechanisms",{}).items():
            if isinstance(vv,int): mech_tot[k]=mech_tot.get(k,0)+vv
        dpath=os.path.join(EXP,"results/ea_v1/fivearm_campaign",rows[(a,r)]["diag_file"] or "")
        if rows[(a,r)]["diag_file"] and os.path.exists(dpath):
            got=sha(dpath)
            if got!=rows[(a,r)]["diag_sha256"]:
                diag_detail.setdefault("sha_mismatch",[]).append(dpath); continue
            d=json.load(open(dpath))
            pb=[p["planbuild_ms"] for p in d.get("prof_dp0",[])]
            planb+= pb
            walls.append(d.get("wall_summary",{}).get("max_wall_ms",0.0))
            ss=d.get("swap_stats",{}); reversals+=ss.get("n_reversals",0); repeats+=ss.get("n_repeats",0); ops+=ss.get("n_ops",0)
            start_maxr=[dg["max_ratio_after"] for dg in d.get("diag_dp0",[])]
            for dg in d.get("diag_dp0",[]):
                ratios_after.append(dg["avg_ratio_after"]); maxratios_after.append(dg["max_ratio_after"])
            if start_maxr: finals.append(start_maxr[-1])
            resets+=len(d.get("reset_events_dp0",[]))
            win_events+=len(d.get("window_events_dp0",[]))
            issued_ops+=sum(w[1] for w in d.get("issued_windows_dp0",[]))
    diag_detail[a]={"mech_totals":mech_tot,
        "planbuild_ms":{"n":len(planb),"mean":(sum(planb)/len(planb)) if planb else None,"max":max(planb) if planb else None},
        "migration_wall_ms_per_start_max":walls,
        "migration_wall_note":"reported alongside; NEVER subtracted from any contrast; not automatically a critical-path addition",
        "swap_ops_total":ops,"swap_reversals_total":reversals,"swap_repeats_total":repeats,
        "issued_ops_dp0_total":issued_ops,
        "layout_quality_proxy":{"mean_avg_ratio_after":(sum(ratios_after)/len(ratios_after)) if ratios_after else None,
            "mean_max_ratio_after":(sum(maxratios_after)/len(maxratios_after)) if maxratios_after else None,
            "final_max_ratio_after_mean_per_start":(sum(finals)/len(finals)) if finals else None},
        "reset_events_total":resets,"window_events_dp0_total":win_events}

def paired(a1,a2):
    ds=[];used=[]
    for r in range(1,6):
        k1,k2=(a1,r),(a2,r)
        if k1 in rows and k2 in rows and rows[k1]["valid"] and rows[k2]["valid"] \
           and rows[k1]["total_time_s"] is not None and rows[k2]["total_time_s"] is not None:
            ds.append(rows[k1]["total_time_s"]-rows[k2]["total_time_s"]); used.append(r)
    if not ds: return {"n_paired_rounds":0,"note":"no valid paired rounds"}
    m=sum(ds)/len(ds)
    sd=math.sqrt(sum((x-m)**2 for x in ds)/(len(ds)-1)) if len(ds)>1 else float("nan")
    t=T975.get(len(ds)-1)
    ci=[m-t*sd/math.sqrt(len(ds)),m+t*sd/math.sqrt(len(ds))] if t and len(ds)>1 and not math.isnan(sd) else None
    return {"n_paired_rounds":len(ds),"rounds_used":used,"diffs_s":[round(x,3) for x in ds],
            "mean_diff_s":m,"sd_diff_s":sd,"ci95_s":ci,
            "excludes_zero":(ci is not None and (ci[0]>0 or ci[1]<0)),
            "sign":"faster" if m<0 else ("slower" if m>0 else "equal"),
            "underpowered_indeterminate":len(ds)<3}
contrasts={
 "primary_A11_minus_A00":paired("A11","A00"),
 "primary_A11_minus_P11":paired("A11","P11"),
 "secondary_A11_minus_A10":paired("A11","A10"),
 "secondary_A11_minus_A01":paired("A11","A01"),
}
# WxD interaction per round: (A11-A10)-(A01-A00)
ids=[]
for r in range(1,6):
    ks=[(a,r) for a in ARMS[:4]]
    if all(k in rows and rows[k]["valid"] and rows[k]["total_time_s"] is not None for k in ks):
        ids.append((rows[("A11",r)]["total_time_s"]-rows[("A10",r)]["total_time_s"])
                   -(rows[("A01",r)]["total_time_s"]-rows[("A00",r)]["total_time_s"]))
if ids:
    m=sum(ids)/len(ids)
    sd=math.sqrt(sum((x-m)**2 for x in ids)/(len(ids)-1)) if len(ids)>1 else float("nan")
    t=T975.get(len(ids)-1)
    ci=[m-t*sd/math.sqrt(len(ids)),m+t*sd/math.sqrt(len(ids))] if t and len(ids)>1 else None
    contrasts["interaction_WxD"]={"n_paired_rounds":len(ids),"values_s":[round(x,3) for x in ids],
        "mean_s":m,"ci95_s":ci,"excludes_zero":(ci is not None and (ci[0]>0 or ci[1]<0)),
        "underpowered_indeterminate":len(ids)<3}
else:
    contrasts["interaction_WxD"]={"n_paired_rounds":0,"note":"no complete valid rounds"}

out={"artifact":"fivearm_campaign_analysis",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "protocol":{"path":PROT,"sha256":sha(PROT)},
 "state":{"path":STATE,"sha256":sha(STATE)},
 "missing_starts":missing,
 "start_rows":{f"{a}_r{r}":rows[(a,r)] for a in ARMS for r in range(1,6) if (a,r) in rows},
 "arm_summary":arm_summary,
 "contrasts":contrasts,
 "descriptors":diag_detail,
 "interpretation_rules":[
  "negative mean_diff_s preserved exactly (A11 slower than comparator where observed)",
  "CI crossing zero = compatible with zero; NOT proof of no effect; NOT equivalence",
  "7963 requests are NOT independent replicates; replicate unit = independent start; rounds = paired blocks",
  "migration wall reported alongside only; complete service time already contains real exposed overhead; never subtract twice",
  "budget frozen: no significance-driven additional runs; uncertainty reported as uncertainty",
  "A11-P11 answers net system effect; layout-quality proxies answer 'better layout found' vs 'less planning/migration time' (ep_v1 distinction_rule)"]}
json.dump(out,open(OUTJ,"w"),indent=1,ensure_ascii=False,allow_nan=True)
h=sha(OUTJ)
# markdown summary
L=[]
L.append("# Five-arm campaign summary (A00/A10/A01/A11/P11)")
L.append(f"\nanalysis sha256: {h}\n")
L.append("| arm | n_valid | mean total_time_s | sd | min | max |")
L.append("|---|---|---|---|---|---|")
for a in ARMS:
    s=arm_summary[a]
    f=lambda x:("%.2f"%x) if isinstance(x,float) else str(x)
    L.append(f"| {a} | {s['n_valid']}/5 | {f(s['mean_time_s'])} | {f(s['sd_time_s'])} | {f(s['min_time_s'])} | {f(s['max_time_s'])} |")
L.append("\n## Contrasts (paired by round; negative = first arm faster)\n")
L.append("| contrast | n | mean diff (s) | 95% CI | excludes zero |")
L.append("|---|---|---|---|---|")
for k,v in contrasts.items():
    if v.get("n_paired_rounds",0)==0: L.append(f"| {k} | 0 | - | - | - |"); continue
    ci=v.get("ci95_s")
    cis=("[%.2f, %.2f]"%(ci[0],ci[1])) if ci else "n/a"
    L.append(f"| {k} | {v['n_paired_rounds']} | {v.get('mean_diff_s',v.get('mean_s')):.2f} | {cis} | {v.get('excludes_zero')} |")
L.append("\nMigration wall (alongside, never subtracted): "+json.dumps({a:diag_detail[a]["migration_wall_ms_per_start_max"] for a in ARMS if a in diag_detail and isinstance(diag_detail.get(a),dict)}))
L.append("\nMissing starts: "+(json.dumps(missing) if missing else "none"))
open(OUTM,"w").write("\n".join(L)+"\n")
print("WROTE",OUTJ);print("sha256",h);print("WROTE",OUTM)
