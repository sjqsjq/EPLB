#!/usr/bin/env python3
"""Deliverable-D analysis of the ADW v2 behavior validation run (W0, frozen-candidate knobs).
Inputs : calib_step5/v2behavior_rank0.jsonl, v2behavior.json
Rules  : pre-registered calibration_plan_v2.md sec.4 + fit_rules_v2 (boundaries 2075/4052/6068 of 8037 rows).
Boundary->window mapping uses cumulative n_fresh fraction (n_fresh is proportional to executed steps
at saturation; same dataset+protocol as the fit run that established the row boundaries).
Outputs: calib_step5/behavior_analysis.json + printed report. No knob is touched here.
"""
import json, collections, os
OUT = os.path.dirname(os.path.abspath(__file__)) + "/../results/adw_v2/calib_step5"
rows = [json.loads(l) for l in open(OUT + "/v2behavior_rank0.jsonl")]
bench = json.load(open(OUT + "/v2behavior.json"))
N_ROWS, BOUNDS = 8037, [2075, 4052, 6068]
Z_LO, Z_HI, Z_URG, TH_LOW, TH_ACT = 3.737, 19.864, 115.619, 988725, 1065054
total_nf = sum(r["n_fresh_assignments"] for r in rows)

# 1) boundary -> window mapping + detection latency
cum, bnd_win = 0, []
for b in BOUNDS:
    tgt = b / N_ROWS * total_nf
    c = 0
    for i, r in enumerate(rows):
        c += r["n_fresh_assignments"]
        if c >= tgt:
            bnd_win.append(i); break

detections = []
for k, bw in enumerate(bnd_win):
    lo, hi = max(0, bw - 3), min(len(rows), bw + 8)
    zwin = [(i, rows[i]["z"], rows[i]["event"], rows[i]["W_before"], rows[i]["W_after"], rows[i]["h_w_assignments"])
            for i in range(lo, hi)]
    shrink = next((i for i, z, ev, wb, wa, h in zwin
                   if i >= bw and ev in ("urgent_shrink", "urgent_shrink_capped", "confirmed_shrink")), None)
    zmax = max((z for i, z, *_ in zwin if i >= bw - 1 and z is not None), default=None)
    detections.append({"boundary_row": BOUNDS[k], "boundary_window": bw,
                       "first_shrink_window": shrink,
                       "latency_windows": (shrink - bw) if shrink is not None else None,
                       "z_max_near": zmax, "windows": zwin})

# 2) dwell distributions
steps_by_W = collections.Counter(); win_by_W = collections.Counter()
for r in rows:
    win_by_W[r["W_after"]] += 1; steps_by_W[r["W_after"]] += r["steps_actual"]
tot_steps = sum(steps_by_W.values()); T = bench["total_time_s"]

# 3) event census
ev = collections.Counter(r["event"] for r in rows)

# 4) costs
c_all = [r["c_window_ms"] for r in rows]
c_rest = c_all[1:]
def stats(x): 
    s = sorted(x); n = len(s)
    return {"n": n, "median": s[n//2], "mean": sum(s)/n, "max": s[-1], "p90": s[int(n*0.9)]}

# 5) pathology checks
desync_any = any(r["desync"] for r in rows)
uncertain_any = any(r["event"].startswith(("uncertain", "forced_shrink")) for r in rows)
traj = [r["W_after"] for r in rows]
# floor ratchet: after each shrink-to-8 episode, does W return >=16 within 20 windows?
shrinks = [i for i, r in enumerate(rows) if r["W_after"] < r["W_before"]]
recovers = []
for i in shrinks:
    nxt = next((j for j in range(i+1, min(i+21, len(rows))) if rows[j]["W_after"] > rows[i]["W_after"]), None)
    recovers.append({"shrink_at": i, "to_W": rows[i]["W_after"], "recovered_at": nxt,
                     "recovered_to": rows[nxt]["W_after"] if nxt is not None else None})
# ping-pong: grow within 2 windows after a shrink (dwell=2 should prevent direction flip-flop < 2...
# dwell counts windows since last W change; grow_blocked_dwell events prove the guard fired)
pingpong = [i for i in range(1, len(rows))
            if rows[i]["W_after"] > rows[i]["W_before"] and rows[i-1]["W_after"] < rows[i-1]["W_before"]]
# grow windows must have h < TH_LOW; shrink windows must have h >= TH_ACT (or urgent w/ h>=TH_ACT)
grow_ok = all(r["h_w_assignments"] < TH_LOW for r in rows if r["event"] == "confirmed_grow")
shrink_ok = all(r["h_w_assignments"] >= TH_ACT for r in rows
                if r["event"] in ("confirmed_shrink", "urgent_shrink", "urgent_shrink_capped"))
# switch#3 silence
b3 = bnd_win[2]
z3 = [rows[i]["z"] for i in range(max(0,b3-3), min(len(rows), b3+6)) if rows[i]["z"] is not None]

res = {
 "run": {"ok": bench["ok"], "errors": bench["errors"], "total_time_s": T,
         "baselines_s": {"fit_run_W16_locked": 363.5, "grid_fixed_W16": 362.37},
         "delta_vs_fit_pct": (T-363.5)/363.5*100, "delta_vs_grid_pct": (T-362.37)/362.37*100},
 "windows": len(rows), "trajectory": traj,
 "boundaries": {"rows": BOUNDS, "windows": bnd_win,
                "mapping": "cumulative n_fresh fraction x 8037 rows (n_fresh prop. to steps at saturation)"},
 "detections": [{k: v for k, v in d.items() if k != "windows"} for d in detections],
 "dwell": {"windows": dict(win_by_W), "steps": dict(steps_by_W),
           "time_est_s": {w: round(s/tot_steps*T, 1) for w, s in steps_by_W.items()},
           "note": "time_est assumes uniform step duration under saturation; n_fresh prop. steps"},
 "events": dict(ev),
 "costs": {"c_window_ms_all": stats(c_all), "c_window_ms_excl_first": stats(c_rest),
           "prof_allreduce_cum_final_ms": rows[-1]["prof_allreduce_cum_ms"],
           "note": "allreduce_cum is the legacy 94-layer ratio loop, kept in BOTH arms (cost symmetry)"},
 "pathology_checks": {
   "desync_any": desync_any, "uncertain_events_any": uncertain_any,
   "shrink_episodes": recovers, "pingpong_grow_right_after_shrink": pingpong,
   "grow_only_below_theta_H_low": grow_ok, "shrink_only_above_theta_H_act": shrink_ok,
   "switch3_z_max": max(z3) if z3 else None, "switch3_z_hi": Z_HI,
   "switch3_silent": (max(z3) if z3 else 0) < Z_HI},
 "detection_detail_windows": {str(d["boundary_row"]): d["windows"] for d in detections},
}
json.dump(res, open(OUT + "/behavior_analysis.json", "w"), indent=1)

print("== run ==", json.dumps(res["run"]))
print("== boundaries ==", bnd_win, "(windows)")
for d in res["detections"]: print("  detect:", d)
print("== dwell ==", json.dumps(res["dwell"], indent=1))
print("== events ==", json.dumps(res["events"]))
print("== costs ==", json.dumps(res["costs"]["c_window_ms_excl_first"]), "allmax", res["costs"]["c_window_ms_all"]["max"], "allreduce_cum", res["costs"]["prof_allreduce_cum_final_ms"])
print("== pathology ==", json.dumps({k:v for k,v in res["pathology_checks"].items() if k!='shrink_episodes'}, indent=1))
print("== shrink episodes ==")
for e in recovers: print("  ", e)
