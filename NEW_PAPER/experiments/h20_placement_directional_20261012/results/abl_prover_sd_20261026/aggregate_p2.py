#!/usr/bin/env python3
"""abl_prover_sd_20261026 phase-2 aggregation — implements PHASE2_PREREGISTRATION.md §5 exactly.
Primary: per-boot whole-stream tps; pairs {P vs F(1,16,16), P vs F(4,16,64)}; two-sided Welch; Holm alpha=0.05.
Analysis set: boots with completeness_pass true; excluded boots registered with reason; NO retry logic here.
Mechanism facts reported per arm separately from performance. h_rate is a low-headroom-band signal — never a speedup headroom."""
import json, math, os, sys, glob
from collections import Counter

A = os.path.dirname(os.path.abspath(__file__))
BOOTS = os.path.join(A, "boots_p2")
ARMS = [("P", "apsd2_P_b"), ("F_1_16_16", "apsd2_F_1_16_16_b"), ("F_4_16_64", "apsd2_F_4_16_64_b")]
PAIRS = [("P", "F_1_16_16"), ("P", "F_4_16_64")]

def welch(x, y):
    nx, ny = len(x), len(y)
    mx, my = sum(x)/nx, sum(y)/ny
    vx = sum((v-mx)**2 for v in x)/(nx-1) if nx > 1 else 0.0
    vy = sum((v-my)**2 for v in y)/(ny-1) if ny > 1 else 0.0
    se2 = vx/nx + vy/ny
    if se2 <= 0: return {"t": None, "df": None, "p": None, "note": "zero_variance"}
    t = (mx-my)/math.sqrt(se2)
    df = se2**2/((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    p = None
    try:
        from scipy import stats
        p = float(2*stats.t.sf(abs(t), df))
    except Exception:
        p = "MISSING_SCIPY"
    return {"t": t, "df": df, "p": p, "mean_x": mx, "mean_y": my, "sd_x": math.sqrt(vx), "sd_y": math.sqrt(vy), "n_x": nx, "n_y": ny}

def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0]*len(pvals); prev = 0.0
    for rank, i in enumerate(idx):
        a = min(1.0, (len(pvals)-rank)*pvals[i]); prev = max(prev, a); adj[i] = prev
    return adj

boots = {}
for arm, pref in ARMS:
    for d in sorted(glob.glob(os.path.join(BOOTS, pref + "*"))):
        lab = os.path.basename(d)
        rec = {"arm": arm, "label": lab, "dir": d}
        mp = os.path.join(d, "MECH_CHECK.json")
        if not os.path.exists(mp):
            rec["status"] = "EXCLUDED_no_mech_check"; boots[lab] = rec; continue
        m = json.load(open(mp)); f = m.get("mechanism_facts", {})
        rec["completeness_pass"] = m.get("completeness_pass")
        rec["bench"] = m.get("bench_record_no_judgment", {})
        if not rec["completeness_pass"]:
            fails = [k for k, v in (m.get("completeness") or {}).items() if v is not True]
            rec["status"] = "EXCLUDED_incomplete:" + ",".join(fails[:4]); boots[lab] = rec; continue
        rec["status"] = "USABLE"
        rec["tps"] = rec["bench"].get("tps")
        rec["mech"] = {
            "rows": f.get("n_rows"), "sp_mode": f.get("sp_mode"),
            "forwards": f.get("true_forwards_total"), "cycles": f.get("detection_cycles_committed"),
            "transitions": [(t.get("window_id"), t.get("before"), t.get("after")) for t in f.get("state_transitions", [])],
            "n_transitions": len(f.get("state_transitions", [])),
            "swaps": (f.get("migration") or {}).get("swaps_started_sum"),
            "issued": (f.get("migration") or {}).get("serverlog_issued_windows"),
            "planhash8": (f.get("migration") or {}).get("planhash_8rank_identical"),
            "det_map_final": (f.get("planning") or {}).get("det_map_version_final"),
            "s_change_rows": (f.get("planning") or {}).get("s_change_logical_rows"),
            "valid_rows_sum": (f.get("recording") or {}).get("valid_rows_sum"),
            "stable_s_max": f.get("stable_s_max"),
            "z_range": f.get("z_range"), "residency_final": f.get("residency_final"),
            "shw_rle": [(r.get("shw"), r.get("rows")) for r in f.get("shw_trajectory_rle", [])],
        }
        boots[lab] = rec

tps = {arm: sorted([r["tps"] for r in boots.values() if r["arm"] == arm and r["status"] == "USABLE" and isinstance(r.get("tps"), (int, float))]) for arm, _ in ARMS}
desc = {arm: {"n": len(v), "median": (v[len(v)//2] if len(v) % 2 else (v[len(v)//2-1]+v[len(v)//2])/2) if v else None,
              "min": v[0] if v else None, "max": v[-1] if v else None,
              "mean": sum(v)/len(v) if v else None} for arm, v in tps.items()}
tests = {}
for a, b in PAIRS:
    if len(tps[a]) >= 2 and len(tps[b]) >= 2:
        tests[f"{a}_vs_{b}"] = welch(tps[a], tps[b])
    else:
        tests[f"{a}_vs_{b}"] = {"p": None, "note": "insufficient_n"}
pv = [tests[f"{a}_vs_{b}"].get("p") for a, b in PAIRS]
if all(isinstance(p, float) for p in pv):
    adj = holm(pv)
    for (a, b), p, pa in zip(PAIRS, pv, adj):
        tests[f"{a}_vs_{b}"]["p_holm"] = pa
        tests[f"{a}_vs_{b}"]["significant_alpha05"] = pa < 0.05

mech_sum = {}
for arm, _ in ARMS:
    rs = [r for r in boots.values() if r["arm"] == arm and r["status"] == "USABLE"]
    mech_sum[arm] = {
        "boots_usable": len(rs),
        "transitions_total": sum(r["mech"]["n_transitions"] for r in rs),
        "boots_with_transition": sum(1 for r in rs if r["mech"]["n_transitions"] > 0),
        "transition_kinds": dict(Counter(str(x[1]) + "->" + str(x[2]) for r in rs for x in r["mech"]["transitions"])),
        "swaps_total": sum(r["mech"]["swaps"] or 0 for r in rs),
        "det_map_finals": sorted(set(r["mech"]["det_map_final"] for r in rs)),
        "stable_s_max_range": [min((r["mech"]["stable_s_max"] for r in rs), default=None), max((r["mech"]["stable_s_max"] for r in rs), default=None)],
        "forwards_set": sorted(set(r["mech"]["forwards"] for r in rs)),
        "planhash8_all_identical": all(r["mech"]["planhash8"] for r in rs),
    }

out = {"generated_utc": __import__("datetime").datetime.utcnow().strftime("%FT%TZ"),
       "prereg": "PHASE2_PREREGISTRATION.md sha ccbb50093ea1535c680cd5ebac61ee73ad40938db6d59610756dd939fa0e7327",
       "primary_metric": "per-boot whole-stream tps",
       "descriptives_tps": desc, "welch": tests,
       "holm": "two pre-declared pairs, alpha=0.05",
       "mechanism_summary": mech_sum,
       "excluded_boots": {lab: r["status"] for lab, r in boots.items() if r["status"] != "USABLE"},
       "per_boot": {lab: {"arm": r["arm"], "status": r["status"], "tps": r.get("tps"), "mech": r.get("mech")} for lab, r in sorted(boots.items())}}
json.dump(out, open(os.path.join(A, "phase2_summary.json"), "w"), indent=1)

L = ["# RESULTS_PHASE2 — prover同域调度消融 正式阶段", "",
     "预注册: PHASE2_PREREGISTRATION.md (§5 分析规则先于数据冻结)。主指标 = 每 boot 完整流 tps。",
     "解释护栏: h_rate 为低余量带信号, 不表示还可以加速多少; 机制事实与性能分开陈述。", "",
     "## 每臂 tps 描述统计 (USABLE boots)", ""]
for arm, _ in ARMS:
    d = desc[arm]; r2 = lambda v: round(v, 2) if isinstance(v, (int, float)) else v; L.append(f"- {arm}: n={d['n']} median={r2(d['median'])} mean={r2(d['mean'])} min={r2(d['min'])} max={r2(d['max'])}")
L += ["", "## 预注册对比 (Welch 双侧 + Holm, alpha=0.05)", ""]
for k, v in tests.items():
    L.append(f"- {k}: " + json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}, ensure_ascii=False))
L += ["", "## 机制终点汇总 (每臂)", ""]
for arm, s in mech_sum.items():
    L.append(f"- {arm}: " + json.dumps(s, ensure_ascii=False))
L += ["", "## 排除 boot 登记", ""]
L.append(json.dumps(out["excluded_boots"], ensure_ascii=False) if out["excluded_boots"] else "- 无 (全部 USABLE)")
L += ["", "## 每 boot 明细见 phase2_summary.json (per_boot)", ""]
open(os.path.join(A, "RESULTS_PHASE2.md"), "w").write("\n".join(L) + "\n")
print("AGG_OK usable:", {arm: desc[arm]["n"] for arm, _ in ARMS}, "excluded:", len(out["excluded_boots"]))
