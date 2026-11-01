import json, os, re, sys
ART = sys.argv[1]; LABEL = sys.argv[2]; NREQ = int(sys.argv[3])
NO_BENCH = "--no-bench" in sys.argv
arm = LABEL.replace("T1_", "").replace("SMOKE_", "")
logp = os.path.join(ART, "server.log")
log = open(logp, errors="replace").read() if os.path.isfile(logp) else ""
facts = {"label": LABEL, "arm": arm, "nreq_expected": NREQ, "no_bench": NO_BENCH}
facts["ready_to_roll"] = "ready to roll" in log
m = re.search(r"deepep_mode=.(\w+)", log); facts["deepep_mode"] = m.group(1) if m else None
m = re.search(r"disable_cuda_graph=(\w+)", log); facts["disable_cuda_graph"] = m.group(1) if m else None
m = re.search(r"init_expert_location=.([^,]*).,", log); facts["init_expert_location"] = m.group(1) if m else None
m = re.search(r"pb_oeplb_max_total_ops=(\d+)", log); facts["pb_max_total_ops"] = int(m.group(1)) if m else None
m = re.search(r"ep_num_redundant_experts=(\d+)", log); facts["redundancy"] = int(m.group(1)) if m else None
iss = re.findall(r"DP(\d+) TP(\d+) EP(\d+)\] \[PB-OEPLB\] window#(\d+): issued (\d+) swap", log)
per_win = {}
for dp, tp, ep, w, n in iss:
    per_win.setdefault(int(w), {})[int(dp)] = int(n)
dedup_ok = all(len(set(v.values())) == 1 for v in per_win.values())
facts["issued_lines_total"] = len(iss)
facts["issued_windows"] = len(per_win)
facts["issued_ops_dedup_sum"] = sum(list(v.values())[0] for v in per_win.values())
facts["dedup_cross_rank_identical"] = dedup_ok
facts["issued_lines_expect_windows_x8"] = (len(iss) == 8 * len(per_win)) if per_win else None
v2p = os.path.join(ART, "v2p1_rank0.jsonl")
facts["v2p1_present"] = os.path.isfile(v2p)
if facts["v2p1_present"]:
    vl = [json.loads(x) for x in open(v2p) if x.strip()]
    facts["v2p1_lines"] = len(vl)
    facts["v2p1_committed_sum"] = sum(int(x.get("committed", 0)) for x in vl)
checks = {"ready": bool(facts["ready_to_roll"]), "deepep_normal": facts["deepep_mode"] == "normal", "graph_off": facts["disable_cuda_graph"] == "True", "trivial_init": (facts["init_expert_location"] or "").find("trivial") >= 0, "redundancy_zero": facts["redundancy"] == 0}
if arm == "ORIG":
    checks["no_plans_issued"] = facts["issued_lines_total"] == 0
else:
    checks["dedup_ok"] = bool(dedup_ok)
    if facts["pb_max_total_ops"] is not None:
        exp = {"GAP32": 32, "GAP128": 128, "GAP300": 300, "MD300": 300, "BI300": 300}.get(arm)
        checks["budget_matches_arm"] = facts["pb_max_total_ops"] == exp
if not NO_BENCH:
    reps = {}
    allok = True
    for r in (1, 2):
        p = os.path.join(ART, "%s_rep%d.json" % (LABEL, r))
        if not os.path.isfile(p):
            reps[r] = None; allok = False; continue
        b = json.load(open(p))
        reps[r] = {"ok": b.get("ok"), "num_requests": b.get("num_requests"), "total_time_s": b.get("total_time_s"), "tps": b.get("tps"), "avg_output_tokens": b.get("avg_output_tokens"), "ttft_mean_ms": b.get("ttft_mean_ms"), "tpot_mean_ms": b.get("tpot_mean_ms"), "lat_mean_ms": b.get("lat_mean_ms")}
        if b.get("ok") != NREQ: allok = False
    facts["reps"] = reps
    checks["bench_both_reps_ok"] = allok
out = {"label": LABEL, "arm": arm, "facts": facts, "checks": checks, "all_ok": all(bool(v) for v in checks.values())}
json.dump(out, open(os.path.join(ART, "MECH_CHECK.json"), "w"), indent=1)
print("MECH_T1", LABEL, "all_ok=%s" % out["all_ok"], json.dumps(checks))
sys.exit(0 if out["all_ok"] or NO_BENCH and checks["ready"] else (0 if out["all_ok"] else 1))
