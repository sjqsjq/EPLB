# timeline_xd.py -- cross_domain_e2e_20261027 sec.5.5 timeline material extraction
# Domain boundaries at request idx 2048/4096/6144 (windows 8/16/24 under 256 req/window).
# Reads per-boot artifacts read-only; writes analysis/timeline_5_5.json + .md
import json, os, re, calendar, datetime
X = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027"
ARMS = ["I", "ES", "ED", "MT", "DF", "P0"]
ROUNDS = [1, 2, 3, 4]
BIDX = [2048, 4096, 6144]
PHASES = ["sharegpt", "prover", "book", "code"]

def pct(v, p):
    if not v: return None
    s = sorted(v); k = int(round(p / 100.0 * (len(s) - 1)))
    return round(s[k], 4)

def logts_to_epoch(s):
    return float(calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))

out = {"campaign": "cross_domain_e2e_20261027",
       "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
       "boundary_request_idx": BIDX,
       "window_map": "32 windows x 256 req; boundary windows 8/16/24; v2p1 has no wall clock (ts_min/max are step counters)",
       "ed_correction": "rebalance start rows = events x 8 DP ranks; true events = 2 per boot (period 256 steps of 512 total)",
       "boots": {}}

for r in ROUNDS:
    for arm in ARMS:
        label = arm + "_r" + str(r)
        d = os.path.join(X, "boots", label)
        b = json.load(open(os.path.join(d, label + ".json")))
        ws = b["wall_start_s"]
        recs = [json.loads(l) for l in open(os.path.join(d, label + "_perreq.jsonl"))]
        recs.sort(key=lambda x: x["idx"])
        boot = {"bench": {k: b.get(k) for k in ["ok", "errors", "total_time_s", "tps", "wall_start_s", "wall_end_s"]}}
        # per-phase stats
        ph = {}
        for p in PHASES:
            sel = [x for x in recs if x["id"].startswith("e3_" + p + "_")]
            ttft = [x["ttft_s"] for x in sel]; el = [x["elapsed_s"] for x in sel]
            sends = [x["t_send_wall_s"] for x in sel]; ends = [x["t_last_wall_s"] for x in sel]
            ph[p] = {"n": len(sel), "ttft_mean": round(sum(ttft) / len(ttft), 4), "ttft_p50": pct(ttft, 50),
                     "ttft_p95": pct(ttft, 95), "elapsed_mean": round(sum(el) / len(el), 4), "elapsed_p95": pct(el, 95),
                     "send_off_first": round(min(sends) - ws, 3), "end_off_last": round(max(ends) - ws, 3),
                     "phase_dur_s": round(max(ends) - min(sends), 3),
                     "req_per_s": round(len(sel) / (max(ends) - min(sends)), 3)}
        boot["phases"] = ph
        # boundary crossing offsets
        bnd = []
        for bi in BIDX:
            cur = next(x for x in recs if x["idx"] == bi)
            prev = next(x for x in recs if x["idx"] == bi - 1)
            bnd.append({"idx": bi, "send_off_s": round(cur["t_send_wall_s"] - ws, 3),
                        "prev_end_off_s": round(prev["t_last_wall_s"] - ws, 3)})
        boot["boundaries"] = bnd
        if arm == "P0":
            v = [json.loads(l) for l in open(os.path.join(d, "v2p1_rank0.jsonl"))]
            keep = ["window_id", "event", "committed", "z", "h_rate", "s_change_logical", "sp_state_before",
                    "sp_state_after", "sp_transition", "sp_swaps_started", "sp_plan_opportunity", "alert_k", "stable_s"]
            boot["windows"] = [dict([(k, x.get(k)) for k in keep] +
                                [("boundary_win", x.get("window_id") in (8, 16, 24)),
                                 ("post_boundary", x.get("window_id") in (9, 17, 25))]) for x in v]
            boot["sp_run_end"] = json.load(open(os.path.join(d, "sp_run_end.json")))
            m = json.load(open(os.path.join(d, "MECH_CHECK.json")))
            mf = m.get("mechanism_facts") or {}
            boot["migration"] = mf.get("migration")
            boot["v2p1_gate"] = mf.get("v2p1_gate")
            boot["state_transitions"] = mf.get("state_transitions")
        if arm == "ED":
            txt = open(os.path.join(d, "server.log"), errors="replace").read()
            starts = sorted(set(re.findall(r"\[([\d\- :]+) DP\d+ TP\d+ EP\d+\] \[EPLBManager\] rebalance start", txt)))
            evs = []
            for s in starts:
                off = logts_to_epoch(s) - ws
                near = min(range(3), key=lambda i: abs(off - bnd[i]["send_off_s"]))
                evs.append({"utc": s, "off_from_bench_start_s": round(off, 1),
                            "nearest_boundary_idx": BIDX[near],
                            "delta_to_boundary_s": round(off - bnd[near]["send_off_s"], 1),
                            "phase_at_event": next((p for p in PHASES if ph[p]["send_off_first"] <= off <= ph[p]["end_off_last"]), "outside")})
            boot["rebalance_events"] = evs
            boot["eplb_row_census"] = {k: len(re.findall(k, txt)) for k in
                [r"\[EPLBManager\] rebalance start", r"\[EPLBManager\]", r"eplb", r"EPLB"]}
        if arm in ("ES", "MT", "DF", "I"):
            m = json.load(open(os.path.join(d, "MECH_CHECK.json")))
            bf = m.get("baseline_facts") or {}
            boot["baseline_facts"] = {k: bf.get(k) for k in ["eplb_banner", "eplb_rebalance_starts", "injection_evidence", "arm", "policy"]}
            txt = open(os.path.join(d, "server.log"), errors="replace").read()
            starts = sorted(set(re.findall(r"\[([\d\- :]+) DP\d+ TP\d+ EP\d+\] \[EPLBManager\] rebalance start", txt)))
            boot["runtime_rebalance_rows"] = len(re.findall(r"\[EPLBManager\] rebalance start", txt))
            boot["runtime_rebalance_events_unique_ts"] = [{"utc": s, "off_from_bench_start_s": round(logts_to_epoch(s) - ws, 1)} for s in starts]
        out["boots"][label] = boot

json.dump(out, open(os.path.join(X, "analysis", "timeline_5_5.json"), "w"), indent=1, ensure_ascii=False)

# ---------- markdown digest ----------
md = []
md.append("# cross_domain_e2e_20261027 -- sec.5.5 timeline material (auto-generated)")
md.append("")
md.append("Stream: e3_multidomain_switch.jsonl (sha d55fe2f7...), 4 phases x 2048 req, boundaries idx 2048/4096/6144.")
md.append("Windows: 32 x 256 req -> boundary windows 8/16/24. v2p1 rows carry no wall clock; mapping is count-based.")
md.append("")
md.append("## T1 boundary crossing offsets (s from bench start, mean over 4 rounds)")
md.append("| arm | b1 idx2048 send | b2 idx4096 send | b3 idx6144 send | wall mean s |")
md.append("|---|---|---|---|---|")
for arm in ARMS:
    bs = [out["boots"][arm + "_r" + str(r)]["boundaries"] for r in ROUNDS]
    wm = sum(out["boots"][arm + "_r" + str(r)]["bench"]["total_time_s"] for r in ROUNDS) / 4.0
    md.append("| %s | %.1f | %.1f | %.1f | %.1f |" % (arm,
        sum(x[0]["send_off_s"] for x in bs) / 4, sum(x[1]["send_off_s"] for x in bs) / 4,
        sum(x[2]["send_off_s"] for x in bs) / 4, wm))
md.append("")
md.append("## T2 per-phase ttft p95 (s), mean over rounds -- load migration near boundaries")
md.append("| arm | sharegpt | prover | book | code |")
md.append("|---|---|---|---|---|")
for arm in ARMS:
    vals = []
    for p in PHASES:
        vals.append(sum(out["boots"][arm + "_r" + str(r)]["phases"][p]["ttft_p95"] for r in ROUNDS) / 4)
    md.append("| %s | %.2f | %.2f | %.2f | %.2f |" % (arm, vals[0], vals[1], vals[2], vals[3]))
md.append("")
md.append("## T3 per-phase req/s (mean over rounds)")
md.append("| arm | sharegpt | prover | book | code |")
md.append("|---|---|---|---|---|")
for arm in ARMS:
    vals = [sum(out["boots"][arm + "_r" + str(r)]["phases"][p]["req_per_s"] for r in ROUNDS) / 4 for p in PHASES]
    md.append("| %s | %.1f | %.1f | %.1f | %.1f |" % (arm, vals[0], vals[1], vals[2], vals[3]))
md.append("")
md.append("## T4 P0 controller activity per boot")
md.append("| boot | windows | swaps_started | transitions | issued_windows | boundary-adjacent (9/17/25) events |")
md.append("|---|---|---|---|---|---|")
for r in ROUNDS:
    b = out["boots"]["P0_r" + str(r)]
    sp = b["sp_run_end"]; mg = b.get("migration") or {}
    adj = [w for w in b["windows"] if w["post_boundary"] and (w["event"] not in ("hold", "init") or (w["sp_swaps_started"] or 0) > 0)]
    md.append("| P0_r%d | %d | %d | %d | %s | %s |" % (r, sp["window_count"], sp["swaps_started_total"],
        len(sp["transitions"] or []), str(mg.get("serverlog_issued_windows")), str([w["window_id"] for w in adj])))
md.append("")
md.append("## T5 ED true rebalance events (correction: 16 log rows = 2 events x 8 DP ranks)")
md.append("| boot | ev1 off_s | ev1 nearest bnd (delta_s) | ev2 off_s | ev2 nearest bnd (delta_s) |")
md.append("|---|---|---|---|---|")
for r in ROUNDS:
    ev = out["boots"]["ED_r" + str(r)]["rebalance_events"]
    md.append("| ED_r%d | %.1f | idx%d (%+.1f) | %.1f | idx%d (%+.1f) |" % (r,
        ev[0]["off_from_bench_start_s"], ev[0]["nearest_boundary_idx"], ev[0]["delta_to_boundary_s"],
        ev[1]["off_from_bench_start_s"], ev[1]["nearest_boundary_idx"], ev[1]["delta_to_boundary_s"]))
md.append("")
md.append("## T6 static arms layout-freeze evidence")
md.append("| boot | banner | injection_evidence (trunc) | runtime rebalance rows |")
md.append("|---|---|---|---|")
for arm in ["ES", "MT", "DF", "I"]:
    for r in ROUNDS:
        b = out["boots"][arm + "_r" + str(r)]
        bf = b.get("baseline_facts") or {}
        md.append("| %s_r%d | %s | %s | %s |" % (arm, r, str(bf.get("eplb_banner"))[:60],
            str(bf.get("injection_evidence"))[:80], str(b.get("runtime_rebalance_rows"))))
md.append("")
open(os.path.join(X, "analysis", "timeline_5_5.md"), "w").write("\n".join(md) + "\n")
print("TIMELINE_OK boots=" + str(len(out["boots"])))
