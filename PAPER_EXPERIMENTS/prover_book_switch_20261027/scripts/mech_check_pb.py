#!/usr/bin/env python3
"""prover_book_switch_20261027 formal check.

FACTS ONLY (PROTOCOL_pc.md sec.6): no performance judgment, no comparison
verdicts. rc=0 iff infrastructural completeness checks pass. Mechanism facts
(window count, transitions, swaps, recording volume, ED rebalance starts)
NEVER affect rc - anti-tuning clause: no gate may create pressure toward
triggering. ED event counts are RECORDED, NOT GATED: with O=1 batch-level
steps, events are not request-count predictable (lesson: e3 campaign 16 log
rows = 2 events x 8 DP ranks). Cloned from mech_check_xd.py with deltas:
stream sha, launcher sha, purpose pb_switch, placement locks (prover-frozen
paper lineage) from code_lock/placement_shas.env cross-verified vs boot_meta.

Usage: mech_check_pb.py ART_DIR LABEL NREQ [--no-bench]
Writes: ART_DIR/MECH_CHECK.json
"""
import json, os, re, sys
from collections import Counter

V2_ASYNC = "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f"
V2_CTRL  = "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697"
DS_SHA   = "2da3467263d4f52889ea2931fa84d9fb244ed3dfabe136fddb5c1b6feb7bf904"
SP_STATE = "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466"
LAUNCHER = "cfbdaee752283935047b9d42bbcc36e74da327400ca0fcd3e53533965ced0c2c"
X_DIR    = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_book_switch_20261027"

art, label, nreq = sys.argv[1], sys.argv[2], int(sys.argv[3])
no_bench = "--no-bench" in sys.argv
res = {"label": label, "nreq": nreq, "purpose": "pb_switch",
       "facts_only": True, "completeness": {}, "mechanism_facts": {}}

def addc(k, ok, detail=""):
    res["completeness"][k] = {"ok": bool(ok), "detail": str(detail)}

# --- placement lock file (single source of truth, frozen at campaign open) ---
lock = {}
try:
    lp = os.path.join(X_DIR, "code_lock", "placement_shas.env")
    for line in open(lp):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            lock[k] = v
except Exception:
    pass

# --- boot_meta locks ---
bm = {}
try:
    bm = json.load(open(os.path.join(art, "boot_meta.json")))
    addc("code_v2_async", bm.get("code_async_swapper") == V2_ASYNC, str(bm.get("code_async_swapper"))[:16])
    addc("code_v2_ctrl", bm.get("code_controller") == V2_CTRL, str(bm.get("code_controller"))[:16])
    addc("ds_sha_frozen", bm.get("ds_sha256") == DS_SHA, str(bm.get("ds_sha256"))[:16])
    addc("code_sp_state_rg", bm.get("code_sp_state") == SP_STATE, str(bm.get("code_sp_state"))[:16])
    addc("launcher_pc", bm.get("launch_arm_pc_sha") == LAUNCHER, str(bm.get("launch_arm_pc_sha"))[:16])
    addc("placement_es_lock", bool(lock.get("PL_ES")) and bm.get("placement_es") == lock.get("PL_ES"), str(bm.get("placement_es"))[:16])
    addc("placement_df_lock", bool(lock.get("PL_DF")) and bm.get("placement_df") == lock.get("PL_DF"), str(bm.get("placement_df"))[:16])
    addc("placement_mt_lock", bool(lock.get("PL_MT")) and bm.get("placement_mt") == lock.get("PL_MT"), str(bm.get("placement_mt"))[:16])
except Exception as e:
    addc("boot_meta_readable", False, e)

# --- server.log infra health + issued/planhash facts ---
issued, ph, ph_ok = [], {}, False
try:
    log = open(os.path.join(art, "server.log"), errors="replace").read()
    addc("no_watchdog", not re.search(r"Watchdog caught", log))
    addc("no_scheduler_exc", not re.search(r"Scheduler hit an exception", log))
    addc("no_rank_desync", not re.search(r"RANK DESYNC", log))
    issued = sorted(set(int(m) for m in re.findall(r"\[PB-OEPLB\] window#(\d+): issued", log)))
    for m in re.finditer(r"\[PB-OEPLB-PLANHASH\] window#(\d+) rank=(\d+) n_ops=(\d+) plan_hash=(\w+)", log):
        ph.setdefault(int(m.group(1)), {})[int(m.group(2))] = (int(m.group(3)), m.group(4))
    ph_ok = bool(ph) and all(len({h for _, h in p.values()}) == 1 and len(p) == 8 for p in ph.values())
    reb = len(re.findall(r"\[EPLBManager\] rebalance start", log))
    banner = bool(re.search(r"\[EPLBManager\] system started", log))
    inj = [l.strip()[:160] for l in log.splitlines() if re.search(r"expert_location|init_by_eplb|init_by_mapping|EPLBManager", l)][:3]
    res["baseline_facts"] = {"arm": bm.get("arm"), "policy": bm.get("policy"), "eplb_rebalance_starts": reb, "eplb_banner": banner, "injection_evidence": inj}
except Exception as e:
    addc("server_log_readable", False, e)

# --- bench completeness (recorded, not judged) ---
bj = {}
if not no_bench:
    try:
        bj = json.load(open(os.path.join(art, label + ".json")))
        ok_n, err_n = bj.get("ok", -1), bj.get("errors", -1)
        addc("bench_complete", ok_n + err_n == nreq, "ok=%s err=%s" % (ok_n, err_n))
        prq = os.path.join(art, label + "_perreq.jsonl")
        addc("perreq_lines", os.path.exists(prq) and sum(1 for _ in open(prq)) == nreq, prq)
    except Exception as e:
        addc("bench_complete", False, e)
if bj:
    res["bench_record_no_judgment"] = {k: bj.get(k) for k in
        ["ok", "errors", "total_time_s", "tps", "total_comp_tokens", "total_prompt_tokens",
         "ttft_mean_ms", "lat_mean_ms", "wall_start_s", "wall_end_s", "client_version"]}

# --- v2p1 mechanism facts (never gated) ---
mf = res["mechanism_facts"]
vp = os.path.join(art, "v2p1_rank0.jsonl")
if os.path.exists(vp):
    rows = [json.loads(l) for l in open(vp) if l.strip()]
    mf["n_rows"] = len(rows)
    if rows:
        mf["sp_mode"] = rows[0].get("sp_mode")
        mf["protocol_rev"] = rows[-1].get("protocol_rev")
        mf["spec_rev"] = rows[-1].get("spec_rev")
        mf["true_forwards_total"] = max((r.get("true_steps_total") or 0) for r in rows)
        mf["true_steps_monotonic"] = all((rows[i].get("true_steps_total") or 0) <= (rows[i+1].get("true_steps_total") or 0) for i in range(len(rows)-1))
        mf["detection_cycles_committed"] = sum(1 for r in rows if r.get("committed"))
        mf["event_counts"] = dict(Counter(str(r.get("event")) for r in rows))
        mf["rg_annotation_rows"] = sum(1 for r in rows if any("[AERO-RG]" in str(x) for x in (r.get("sp_reasons") or [])))
        _sre = {}
        _srep = os.path.join(art, "sp_run_end.json")
        if os.path.exists(_srep):
            try:
                _sre = json.load(open(_srep))
            except Exception as _e:
                _sre = {"error": str(_e)}
        mf["sp_run_end_transitions"] = _sre.get("transitions")
        mf["rg_policy_tagged_transitions"] = sum(1 for t in (_sre.get("transitions") or []) if any(str(x).startswith("[AERO-RG] policy") for x in (t.get("reasons") or [])))
        zs = [r["z"] for r in rows if r.get("z") is not None]
        hs = [r["h_rate"] for r in rows if r.get("h_rate") is not None]
        mf["z_range"] = [min(zs), max(zs)] if zs else None
        mf["h_rate_range"] = [min(hs), max(hs)] if hs else None
        mf["alert_k_max"] = max((r.get("alert_k") or 0) for r in rows)
        mf["stable_s_max"] = max((r.get("stable_s") or 0) for r in rows)
        traj, rle = [], None
        for r in rows:
            k = [r.get("sp_s"), r.get("sp_h"), r.get("sp_w")]
            if rle and rle[0] == k:
                rle[2] += 1
            else:
                rle = [k, r.get("window_id"), 1]
                traj.append(rle)
        mf["shw_trajectory_rle"] = [{"shw": t[0], "from_window": t[1], "rows": t[2]} for t in traj]
        mf["state_transitions"] = [{"window_id": r.get("window_id"), "event": r.get("event"),
                                    "before": r.get("sp_state_before"), "after": r.get("sp_state_after"),
                                    "reasons": r.get("sp_reasons")} for r in rows if r.get("sp_transition")]
        mf["residency_final"] = rows[-1].get("sp_residency")
        mf["recording"] = {
            "recorded_steps_sum": sum(r.get("sp_recorded_steps") or 0 for r in rows),
            "sample_opportunities_sum": sum(r.get("sp_sample_opportunities") or 0 for r in rows),
            "valid_rows_sum": sum(r.get("sp_valid_rows") or 0 for r in rows),
            "valid_rows_max_cycle": max((r.get("sp_valid_rows") or 0) for r in rows)}
        mf["planning"] = {
            "plan_opportunity_rows": sum(1 for r in rows if r.get("sp_plan_opportunity")),
            "s_change_logical_rows": sum(1 for r in rows if r.get("s_change_logical")),
            "det_map_version_final": rows[-1].get("det_map_version"),
            "det_map_version_bumps": sum(1 for i in range(1, len(rows))
                if (rows[i].get("det_map_version") or 0) > (rows[i-1].get("det_map_version") or 0))}
        mf["migration"] = {
            "swaps_started_sum": sum(r.get("sp_swaps_started") or 0 for r in rows),
            "swap_pending_rows": sum(1 for r in rows if r.get("sp_swap_pending")),
            "serverlog_issued_windows": issued,
            "planhash_windows": len(ph),
            "planhash_8rank_identical": ph_ok}
        mf["integrity"] = {
            "desync_rows": sum(1 for r in rows if r.get("desync")),
            "h_hash_identical_all_rows": all(r.get("h_min") == r.get("h_max") for r in rows),
            "lockstep_ok_all_rows": all(r.get("lockstep_ok") is not False for r in rows)}
        mf["exercise_summary"] = {
            "detection_cycles_present": len(rows) > 0,
            "recording_present": mf["recording"]["recorded_steps_sum"] > 0,
            "planning_present": mf["planning"]["plan_opportunity_rows"] > 0,
            "migration_present": (mf["migration"]["swaps_started_sum"] > 0) or bool(issued),
            "state_transition_present": len(mf["state_transitions"]) > 0,
            "shw_constant": len(traj) == 1}
else:
    mf["error"] = "v2p1_rank0.jsonl missing"
    if str(bm.get("arm")) == "P":
        addc("v2p1_present", False, vp)
    else:
        mf["v2p1_not_applicable"] = True

res["completeness_pass"] = all(c["ok"] for c in res["completeness"].values())
with open(os.path.join(art, "MECH_CHECK.json"), "w") as f:
    json.dump(res, f, indent=1)
print(("COMPLETE " if res["completeness_pass"] else "INCOMPLETE ") + label +
      " facts=" + json.dumps(mf.get("exercise_summary", mf.get("error", {}))))
sys.exit(0 if res["completeness_pass"] else 1)
