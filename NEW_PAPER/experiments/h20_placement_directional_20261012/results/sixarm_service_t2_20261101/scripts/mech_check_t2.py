import json, os, re, sys
ART = sys.argv[1]; LABEL = sys.argv[2]; NREQ = int(sys.argv[3])
NO_BENCH = "--no-bench" in sys.argv
tok = LABEL.split("_")
scen = tok[0]; arm = tok[1]
logp = os.path.join(ART, "server.log")
log = open(logp, errors="replace").read() if os.path.isfile(logp) else ""
facts = {"label": LABEL, "scenario": scen, "arm": arm, "nreq_expected": NREQ, "no_bench": NO_BENCH}
facts["ready_to_roll"] = "ready to roll" in log
m = re.search(r"deepep_mode=.(\w+)", log); facts["deepep_mode"] = m.group(1) if m else None
m = re.search(r"disable_cuda_graph=(\w+)", log); facts["disable_cuda_graph"] = m.group(1) if m else None
m = re.search(r"ep_num_redundant_experts=(\d+)", log); facts["redundancy"] = int(m.group(1)) if m else None
m = re.search(r"init_expert_location=.([^,]*).,", log); facts["init_expert_location"] = (m.group(1) if m else None)
m = re.search(r"eplb=(\w+)", log); facts["eplb"] = m.group(1) if m else None
m = re.search(r"eplb_algorithm=.([\w]+)", log); facts["eplb_algorithm"] = m.group(1) if m else None
m = re.search(r"eplb_rebalance_num_iterations=(\d+)", log); facts["eplb_iters"] = int(m.group(1)) if m else None
exp_red = 16 if arm in ("ES", "ED") else 0
exp_loc = {"ES": "datafore_e3_placement.json", "DF": "datafore_e3_remap_placement.json", "MT": "placement_e3_fair.json"}
checks = {"ready": bool(facts["ready_to_roll"]), "deepep_normal": facts["deepep_mode"] == "normal", "graph_off": facts["disable_cuda_graph"] == "True", "redundancy_expected": facts["redundancy"] == exp_red}
loc = facts["init_expert_location"] or ""
if arm in exp_loc:
    checks["init_location_expected"] = loc.find(exp_loc[arm]) >= 0
else:
    checks["init_location_trivial"] = loc.find("trivial") >= 0
if arm == "ED":
    checks["eplb_enabled"] = facts["eplb"] == "True"
    if facts["eplb_iters"] is not None:
        checks["eplb_iters_256"] = facts["eplb_iters"] == 256
if arm == "P0":
    facts["v2p1_present"] = os.path.isfile(os.path.join(ART, "v2p1_rank0.jsonl"))
if not NO_BENCH:
    p = os.path.join(ART, LABEL + ".json")
    if os.path.isfile(p):
        b = json.load(open(p))
        facts["bench"] = {"ok": b.get("ok"), "num_requests": b.get("num_requests"), "total_time_s": b.get("total_time_s"), "tps": b.get("tps"), "avg_output_tokens": b.get("avg_output_tokens"), "ttft_mean_ms": b.get("ttft_mean_ms"), "ttft_p99_ms": b.get("ttft_p99_ms"), "tpot_mean_ms": b.get("tpot_mean_ms"), "lat_mean_ms": b.get("lat_mean_ms"), "lat_p99_ms": b.get("lat_p99_ms")}
        checks["bench_ok_eq_nreq"] = b.get("ok") == NREQ
        aot = b.get("avg_output_tokens")
        checks["actual_o32"] = (aot is not None and aot >= 31)
    else:
        facts["bench"] = None
        checks["bench_ok_eq_nreq"] = False
out = {"label": LABEL, "scenario": scen, "arm": arm, "facts": facts, "checks": checks, "all_ok": all(bool(v) for v in checks.values())}
json.dump(out, open(os.path.join(ART, "MECH_CHECK.json"), "w"), indent=1)
print("MECH_T2", LABEL, "all_ok=%s" % out["all_ok"], json.dumps(checks))
sys.exit(0 if out["all_ok"] else 1)
