#!/usr/bin/env python3
"""bsk exploratory descriptive analysis (single-domain pure book, n=2000, 12 boots).
Written from PROTOCOL_bsk.md sec.6 endpoints BEFORE results inspection (sec.7 rule).
FACTS ONLY; R1-R4 interpretation applied by the analyst after this script runs.
Note: bias-gate skip log lines are emitted only when window_count % 8 == 1
(controller.py), so the logged skip count is a LOWER BOUND on actual skips."""
import json, os, re
from collections import Counter

BSK = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/book_single_knobs_20261027"
OUT = BSK + "/analysis"
N = 2000; HALF = 1000; BKT = 250
LABELS = ["P0_r1", "P0B_r1", "P0S_r1", "P0F_r1", "ED_r1", "DF_r1",
          "P0B_r2", "P0S_r2", "P0F_r2", "ED_r2", "DF_r2", "P0_r2"]
ARMS = ["P0", "P0B", "P0S", "P0F", "ED", "DF"]
import datetime
tpat = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
spat = re.compile(r"swap\(s\) done \(([0-9.]+)ms\)")
R = {}
for lab in LABELS:
    boot = BSK + "/boots/" + lab
    rec = {"label": lab}
    sj = boot + "/" + lab + ".json"
    if os.path.exists(sj):
        s = json.load(open(sj))
        rec["tps"] = s.get("tps"); rec["ok"] = s.get("ok"); rec["err"] = s.get("errors")
        rec["lat_mean_ms"] = s.get("lat_mean_ms"); rec["ttft_mean_ms"] = s.get("ttft_mean_ms")
    pf = boot + "/" + lab + "_perreq.jsonl"
    if os.path.exists(pf):
        rows = [json.loads(l) for l in open(pf)]
        def ph(lo, hi):
            sub = [r for r in rows if lo <= r["idx"] < hi]
            if not sub: return None
            t0 = min(r["t_send_wall_s"] for r in sub); t1 = max(r["t_last_wall_s"] for r in sub)
            return round(len(sub) / (t1 - t0), 3)
        rec["half1"] = ph(0, HALF); rec["half2"] = ph(HALF, N)
        bk = []; b = 0
        while b < N:
            bk.append(ph(b, min(b + BKT, N))); b += BKT
        rec["buckets250"] = bk
    lg = boot + "/server.log"
    if os.path.exists(lg):
        cum = 0.0; nsw = 0; first = None; last = None
        n_exh = 0; n_skip = 0
        for line in open(lg, errors="replace"):
            m = spat.search(line)
            if m: cum += float(m.group(1)); nsw += 1
            if "swap budget exhausted" in line: n_exh += 1
            if "noise/margin too high" in line: n_skip += 1
            t = tpat.search(line)
            if t:
                if first is None: first = t.group(1)
                last = t.group(1)
        rec["swap_events"] = nsw
        rec["cum_swap_s"] = round(cum / 1000.0, 1)
        if first and last:
            t0 = datetime.datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
            span = (t1 - t0).total_seconds()
            rec["log_span_s"] = span
            rec["spent_frac"] = round((cum / 1000.0) / span, 4) if span > 0 else None
        rec["budget_exhausted_lines"] = n_exh
        rec["bias_skip_log_lines_lower_bound"] = n_skip
    vf = boot + "/v2p1_rank0.jsonl"
    if os.path.exists(vf):
        vrows = [json.loads(l) for l in open(vf) if l.strip()]
        per = {}
        for r in vrows:
            w = r.get("window_id")
            if w is not None: per.setdefault(w, r)
        ws = sorted(per)
        sw_w = [w for w in ws if (per[w].get("sp_swaps_started") or 0) > 0]
        ntail = max(1, len(ws) // 5)
        zs_tail = [per[w].get("z") for w in ws[-ntail:] if isinstance(per[w].get("z"), float)]
        rec["v2p1"] = {"rows": len(vrows), "windows": len(ws), "swap_windows": len(sw_w),
                       "swap_window_ids": sw_w,
                       "z_range": [round(min(per[w]["z"] for w in ws if isinstance(per[w].get("z"), float)), 2),
                                   round(max(per[w]["z"] for w in ws if isinstance(per[w].get("z"), float)), 2)] if ws else None,
                       "z_tail20_mean": round(sum(zs_tail) / len(zs_tail), 3) if zs_tail else None,
                       "events": dict(Counter(str(per[w].get("event")) for w in ws))}
        sep = boot + "/sp_run_end.json"
        if os.path.exists(sep):
            se = json.load(open(sep))
            rec["v2p1"]["swaps_total"] = se.get("swaps_started_total")
            rec["v2p1"]["true_steps"] = se.get("true_steps_total")
            rec["v2p1"]["state_final"] = se.get("state_final")
    R[lab] = rec

with open(OUT + "/descriptive_bsk.json", "w") as f:
    json.dump(R, f, indent=1)

L = []
L.append("# bsk single-domain pure-book descriptive analysis (EXPLORATORY, n=2 rounds, NON-INFERENTIAL)")
L.append("")
L.append("## 1. per-boot endpoints")
L.append("| boot | tps | half1 | half2 | lat_ms | ttft_ms | swap_ev | cum_sw_s | spent_frac | budget_exh | bias_skip_lb | wins | sw_wins | z_tail20 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for lab in LABELS:
    r = R.get(lab, {})
    v = r.get("v2p1") or {}
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        lab, r.get("tps"), r.get("half1"), r.get("half2"),
        round(r["lat_mean_ms"], 1) if r.get("lat_mean_ms") else None,
        round(r["ttft_mean_ms"], 1) if r.get("ttft_mean_ms") else None,
        r.get("swap_events"), r.get("cum_swap_s"), r.get("spent_frac"),
        r.get("budget_exhausted_lines"), r.get("bias_skip_log_lines_lower_bound"),
        v.get("windows"), v.get("swap_windows"), v.get("z_tail20_mean")))
L.append("")
L.append("## 2. arm round-means (2 reps, descriptive)")
L.append("| arm | tps mean | half1 mean | half2 mean | swap_ev mean | spent_frac mean | budget_exh (r1/r2) | bias_skip_lb (r1/r2) |")
L.append("|---|---|---|---|---|---|---|---|")
def m2(a, k, sub=None):
    vals = []
    for rd in (1, 2):
        r = R.get("%s_r%d" % (a, rd), {})
        src = r.get(sub) if sub else r
        if isinstance(src, dict) and src.get(k) is not None: vals.append(src[k])
        elif not sub and r.get(k) is not None: vals.append(r[k])
    return round(sum(vals) / len(vals), 3) if vals else None
for a in ARMS:
    r1 = R.get(a + "_r1", {}); r2 = R.get(a + "_r2", {})
    L.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s / %s |" % (
        a, m2(a, "tps"), m2(a, "half1"), m2(a, "half2"), m2(a, "swap_events"), m2(a, "spent_frac"),
        r1.get("budget_exhausted_lines"), r2.get("budget_exhausted_lines"),
        r1.get("bias_skip_log_lines_lower_bound"), r2.get("bias_skip_log_lines_lower_bound")))
L.append("")
L.append("## 3. bucket trajectories (250 reqs)")
for lab in LABELS:
    r = R.get(lab, {})
    if r.get("buckets250"): L.append("- %s: %s" % (lab, r["buckets250"]))
L.append("")
L.append("## 4. v2p1 events per arm (r1)")
for a in ARMS:
    v = (R.get(a + "_r1") or {}).get("v2p1") or {}
    if v: L.append("- %s_r1: %s" % (a, json.dumps(v.get("events"))))
with open(OUT + "/descriptive_bsk.md", "w") as f:
    f.write("\n".join(L) + "\n")
print("WROTE descriptive_bsk.json/.md")
