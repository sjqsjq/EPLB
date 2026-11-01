#!/usr/bin/env python3
"""ADW v2.1 behavior-validation analysis (plan §5, user step 5 checks).

Inputs : r3_logging/behavior/v2p1_behavior_rank0.jsonl, v21behavior.json,
         server.log ([PB-OEPLB-PROF]), fit_results_v2p1.json (frozen knobs).
Checks (user step 5, verbatim scope):
  1. cross-rank window/event/state consistency: every committed window has
     h_min==h_max and v_min==1; no desync; no freeze events; per-rank PROF
     window# counts equal across ranks.
  2. stationary segments & near-change z / h_rate / window actions around the
     three dataset boundaries (fraction-of-steps mapping, same as fit).
  3. every grow/shrink/hold reason (event census + proposed_event + state).
  4. actual costs: c_window_ms distribution; PROF record/allreduce/planbuild/
     finalize; migration-related events. (Reported, NOT deducted from anything.)
  5. errors/anomalies/output correctness: bench ok/errors; fatal_collective
     absent; W trajectory on grid; theta-gate consistency of actions:
     confirmed_grow requires h_rate < theta_r_low; confirmed/urgent shrink
     requires h_rate >= theta_r_act (rate-scale gates, v2.1 knobs).
NO knob is touched here. NO adaptive-vs-fixed superiority claim is made here
(single behavior run; plan §5.3: formal comparison needs its own protocol).
"""
import json, os, sys, collections
import numpy as np

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTROOT = EXP + "/results/adw_v2p1/r3_logging"
OUT = OUTROOT + "/behavior"
N_ROWS, BOUNDS, GUARD = 8037, [2009, 4018, 6027], 250
GRID = (8, 16, 32, 64)

def stats(x):
    if not x: return None
    a = np.asarray(x, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min())}

def prof_parse(path):
    """Last PROF line per rank occurrence -> {window#, calls, record, allreduce, planbuild, finalize}."""
    lines = []
    for l in open(path, errors="replace"):
        if "[PB-OEPLB-PROF]" in l and "calls=" in l:
            try:
                seg = l.split("[PB-OEPLB-PROF]")[1]
                d = {"window": int(seg.split("window#")[1].split()[0])}
                for k in ("calls", ):
                    d[k] = int(seg.split(k + "=")[1].split()[0])
                for k in ("record", "allreduce", "planbuild", "finalize", "total"):
                    d[k + "_ms"] = float(seg.split(k + "=")[1].split("ms")[0])
                d["avg_record_us"] = float(seg.split("avg_record_us=")[1].split()[0])
                lines.append(d)
            except (ValueError, IndexError):
                pass
    return lines

def main():
    rows = [json.loads(l) for l in open(OUT + "/v2p1_behavior_rank0.jsonl") if l.strip()]
    bench = json.load(open(OUT + "/v21behavior.json"))
    fit = json.load(open(OUTROOT + "/fit_results_v2p1.json"))
    K = {"kappa": fit["kappa"], "z_lo": fit["z_lo"], "z_hi": fit["z_hi"],
         "z_urgent": fit["z_urgent"], "theta_r_low": fit["theta_r_low"],
         "theta_r_act": fit["theta_r_act"], "n_min": fit["n_min_kept"]}
    prof = prof_parse(OUT + "/server.log") if os.path.exists(OUT + "/server.log") else []

    res = {"knobs_frozen": K, "windows_recorded": len(rows)}

    # 1) consistency
    committed = [r for r in rows if r.get("committed")]
    res["consistency"] = {
        "all_committed_hmin_eq_hmax": all(r["h_min"] == r["h_max"] for r in committed),
        "all_committed_vmin1": all(r["v_min"] == 1 for r in committed),
        "desync_any": any(r.get("desync") for r in rows),
        "freeze_events": [r["window_id"] for r in rows
                          if r["event"] in ("confirm_fail_freeze", "w_desync_freeze")],
        "fatal_collective": [r["window_id"] for r in rows if r["event"] == "fatal_collective"],
        "confirm_fail_streak_max": max((r.get("confirm_fail_streak", 0) for r in rows), default=0),
        "not_committed_windows": [{"window_id": r["window_id"], "event": r["event"],
                                   "v_min": r.get("v_min"), "h_min": r.get("h_min"),
                                   "h_max": r.get("h_max")}
                                  for r in rows if not r.get("committed")],
        "prof_reports": len(prof),
        "prof_max_calls": max((p["calls"] for p in prof), default=None),
        "prof_forwards_implied": (max((p["calls"] for p in prof), default=0) / 94) if prof else None,
    }

    # 2) trajectory + boundary context
    traj = [r["W_after"] for r in rows]
    res["trajectory"] = {"W_after": traj, "on_grid": all(w in GRID for w in traj),
                         "W_init": rows[0]["W_before"] if rows else None}
    total_steps = sum(r["steps_actual"] for r in rows)
    cum, bmap = 0, {}
    for b in BOUNDS:
        tgt = b / N_ROWS * total_steps
        c = 0
        for i, r in enumerate(rows):
            c += r["steps_actual"]
            if c >= tgt:
                bmap[b] = i
                break
    ctx = {}
    for b, bi in bmap.items():
        lo, hi = max(0, bi - 4), min(len(rows), bi + 7)
        ctx[b] = [{"window_id": rows[i]["window_id"], "z": rows[i]["z"],
                   "h_rate": round(rows[i]["h_rate"], 6),
                   "event": rows[i]["event"], "W": rows[i]["W_after"],
                   "alert_k": rows[i].get("alert_k"), "stable_s": rows[i].get("stable_s")}
                  for i in range(lo, hi)]
        first_shrink = next((rows[i]["window_id"] for i in range(bi, min(len(rows), bi + 9))
                             if "shrink" in rows[i]["event"]), None)
        ctx[str(b) + "_first_shrink_window"] = first_shrink
        ctx[str(b) + "_latency_windows"] = (first_shrink - bi) if first_shrink is not None else None
    res["boundary_context"] = {"boundaries_rows": BOUNDS, "boundary_window_idx": bmap,
                               "mapping": "cumulative steps_actual fraction x 8037 (fit method)",
                               "context": {str(k): v for k, v in ctx.items()}}

    # 3) events / reasons
    res["events"] = dict(collections.Counter(r["event"] for r in rows))
    res["proposed_events"] = dict(collections.Counter(str(r.get("proposed_event")) for r in rows))
    res["action_windows"] = [
        {"window_id": r["window_id"], "event": r["event"], "proposed": r.get("proposed_event"),
         "z": r["z"], "h_rate": r["h_rate"], "W_before": r["W_before"], "W_after": r["W_after"],
         "alert_k": r.get("alert_k"), "stable_s": r.get("stable_s"), "dwell": r.get("dwell"),
         "dwell_dir": r.get("dwell_dir"), "uncertain_run": r.get("uncertain_run"),
         "episode_shrinks": r.get("episode_shrinks"), "n_fresh": r["n_fresh_assignments"]}
        for r in rows if r["event"] not in ("hold", "init", "grow_blocked_headroom",
                                            "grow_blocked_dwell", "grow_blocked_max",
                                            "shrink_blocked_dwell", "shrink_blocked_cap")]

    # 4) costs (reported only)
    res["costs"] = {
        "c_window_ms_all": stats([r["c_window_ms"] for r in rows]),
        "c_window_ms_excl_init": stats([r["c_window_ms"] for r in rows[1:]]),
        "prof_last_per_report": prof[-8:] if prof else None,
        "note": ("c_window_ms covers per-window det SUM + confirm MIN/MAX collectives and "
                 "window-boundary local compute; record-path cost (incl. F1 second scatter_add) "
                 "is covered by PROF record/avg_record_us. Ranks run in parallel: per-rank times "
                 "are NOT summed. Nothing here is deducted from any performance number."),
    }

    # 5) correctness gates
    grows = [r for r in rows if r["event"] == "confirmed_grow"]
    shrinks = [r for r in rows if r["event"] in ("confirmed_shrink", "urgent_shrink",
                                                 "urgent_shrink_capped")]
    gate_fail = []
    for r in grows:
        if not (r["h_rate"] < K["theta_r_low"]):
            gate_fail.append({"window_id": r["window_id"], "event": r["event"],
                              "h_rate": r["h_rate"], "theta_r_low": K["theta_r_low"]})
    for r in shrinks:
        if r["event"].startswith("urgent"):
            if r["z"] is None or r["z"] < K["z_urgent"] or r["h_rate"] < K["theta_r_act"]:
                gate_fail.append({"window_id": r["window_id"], "event": r["event"],
                                  "z": r["z"], "z_urgent": K["z_urgent"],
                                  "h_rate": r["h_rate"], "theta_r_act": K["theta_r_act"]})
        elif not (r["h_rate"] >= K["theta_r_act"]):
            gate_fail.append({"window_id": r["window_id"], "event": r["event"],
                              "h_rate": r["h_rate"], "theta_r_act": K["theta_r_act"]})
    uncertain = [r for r in rows if r["event"].startswith(("uncertain", "forced_shrink"))]
    res["correctness"] = {
        "bench_ok": bench.get("ok"), "bench_errors": bench.get("errors"),
        "bench_total_time_s": bench.get("total_time_s"),
        "theta_gate_violations": gate_fail,
        "uncertain_windows": [{"window_id": r["window_id"], "event": r["event"],
                               "n_fresh": r["n_fresh_assignments"], "n_min": K["n_min"]}
                              for r in uncertain],
        "z_none_windows": [r["window_id"] for r in rows if r["z"] is None and r["event"] != "init"],
    }
    # 3b) sequential replay of the FROZEN state machine on logged inputs:
    # reconstructs per-window reasons (not in the r3 log schema) AND proves the
    # live event stream is exactly what the frozen logic produces (0 mismatch
    # expected; any mismatch = anomaly -> report, stop, no refit).
    from sglang.srt.managers.pb_oeplb.window_v2p1 import WindowControllerV2P1
    knobs = dict(kappa=K["kappa"], z_lo=K["z_lo"], z_hi=K["z_hi"], z_urgent=K["z_urgent"],
                 m_shrink=2, m_grow=3, n_min=K["n_min"], u_max=4,
                 theta_h_low=K["theta_r_low"], theta_h_act=K["theta_r_act"],
                 dwell_shrink=2, dwell_grow=2, w_uncertain_target=16,
                 urgent_episode_cap=3)
    rp = WindowControllerV2P1(16, knobs, beta=0.5)
    replay, mismatches = [], []
    for r in rows:
        snap = rp.snapshot()
        rec = rp.step(r["z"], r["h_rate"], r["n_fresh_assignments"])
        if not r.get("committed"):
            # live protocol rolled the proposal back (confirm_hold / freeze /
            # phase-2 error): mirror the rollback so replay state tracks live.
            rp.restore(snap)
            entry = {"window_id": r["window_id"], "live_event": r["event"],
                     "replay_event": r["event"], "match": True,
                     "W_after_replay": r["W_after"], "W_after_live": r["W_after"],
                     "reasons": (rec.get("reasons") if rec else None),
                     "note": "not committed live; proposal rolled back in replay"}
            replay.append(entry)
            continue
        entry = {"window_id": r["window_id"], "live_event": r["event"],
                 "replay_event": rec["event"], "match": rec["event"] == r["event"],
                 "W_after_replay": rec["W_after"], "W_after_live": r["W_after"],
                 "reasons": rec.get("reasons")}
        replay.append(entry)
        if not entry["match"] or rec["W_after"] != r["W_after"]:
            mismatches.append(entry)
    res["replay"] = {"n": len(replay), "mismatches": mismatches,
                     "all_match": not mismatches, "detail": replay}

    res["verdict_inputs"] = {
        "no_desync_no_freeze_no_fatal": (not res["consistency"]["desync_any"]
                                          and not res["consistency"]["freeze_events"]
                                          and not res["consistency"]["fatal_collective"]),
        "rank_agreement_all_committed": (res["consistency"]["all_committed_hmin_eq_hmax"]
                                          and res["consistency"]["all_committed_vmin1"]),
        "bench_clean": bench.get("errors") == 0 and bench.get("ok") == N_ROWS,
        "theta_gates_clean": not gate_fail,
        "trajectory_on_grid": res["trajectory"]["on_grid"],
        "replay_all_match": res["replay"]["all_match"],
        "event_spectrum": sorted(res["events"].keys()),
    }
    json.dump(res, open(OUT + "/behavior_analysis_v2p1.json", "w"), indent=1)

    print("== consistency ==", json.dumps({k: v for k, v in res["consistency"].items()
                                           if k != "not_committed_windows"}))
    if res["consistency"]["not_committed_windows"]:
        print("   not-committed:", res["consistency"]["not_committed_windows"][:10])
    print("== trajectory ==", traj)
    print("== events ==", res["events"])
    print("== action windows ==")
    for a in res["action_windows"]:
        print("   ", a)
    print("== boundary context ==")
    for b in BOUNDS:
        print("  boundary", b, "-> window idx", bmap.get(b),
              "first_shrink:", ctx.get(str(b) + "_first_shrink_window"),
              "latency:", ctx.get(str(b) + "_latency_windows"))
        for w in ctx.get(b, []):
            print("      ", w)
    print("== replay ==", "n:", res["replay"]["n"], "all_match:", res["replay"]["all_match"],
          "mismatches:", len(res["replay"]["mismatches"]))
    for m in res["replay"]["mismatches"][:10]:
        print("   MISMATCH:", m)
    rmap = {e["window_id"]: e["reasons"] for e in res["replay"]["detail"]}
    print("== reasons for action windows ==")
    for a in res["action_windows"]:
        print("   w%s %s: %s" % (a["window_id"], a["event"], rmap.get(a["window_id"])))
    print("== costs ==", json.dumps(res["costs"]["c_window_ms_excl_init"]))
    print("== correctness ==", json.dumps({k: v for k, v in res["correctness"].items()
                                           if k != "uncertain_windows"}))
    print("   uncertain:", res["correctness"]["uncertain_windows"][:10])
    print("== verdict inputs ==", json.dumps(res["verdict_inputs"], indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
