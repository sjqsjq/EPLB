#!/usr/bin/env python3
"""M2->M3 stage gate (PRE_REG_calibration_m1m2m3.md §C): MECHANISM ONLY,
never endpoint values. Reads the three M2 gate-mode artifact dirs.

Checks per run (cal_m2_w1_norm, cal_m2_w1_abort [MECHANISM config],
cal_m2_w2_obs):
  H1 verdict VALID.
  H2 license at most once (<=1 trigger action; probe_run_end license_used).
  H3 trigger window == 12, or every window in [12, trigger) justifiably
     delayed (committed event in S3 family or alert_k > 0).
  H4 segment window counts == frozen budgets (norm/obs: REF 4, CAND 3, FB 4;
     abort run: CAND == force_abort_at + lag windows, FB 4), or -- when S3
     contamination fired -- truncated segments WITH contaminated flag and
     state ABORTED (S3/DONE semantics per PRE_REG).
  H5 normal path completes ABA -> W16: after EVAL every committed W_after is
     16 (W1), or 16/S3-shrink only (W2: any W_after != 16 after EVAL must
     carry an S3-family event).
  H6 abort path (abort run): probe_abort_risk committed, abort_reason ==
     force_abort_at, overshoot fields recorded (int >= 0), lag window flagged
     inside CAND, W back to 16 afterwards, eval classification UNKNOWN with
     abort flag. Norm runs must NOT abort.
  H7 protocol health on every window: lockstep_ok, h_min==h_max on commits,
     desync false, no fatal_collective/*_freeze, probe_mode == gate.
  H8 probe_state series monotone ARMED->REF->CAND->FB->DONE(/ABORTED);
     terminal state matches probe_run_end.json.
  H9 gating audit: no committed confirmed_grow; every committed W transition
     event in {probe_start, probe_fallback, probe_abort_risk} U S3 U {init};
     committed W_after never 64; W_after==32 only inside CAND windows;
     grow_gated_probe fired at least once (real growth pressure existed).
  H10 probe_run_end.json present and consistent with the jsonl.
Exit 0 = PASS (proceed to M3); 1 = FAIL (HALT + report); 2 = bad usage.
Writes $CAL/m2_gate.json. Ledger values are REPORTED (final analysis), not
gated here beyond eps_log sanity already applied at M1.
"""
import json
import os
import statistics
import sys

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CAL = os.path.join(EXP, "results/adw_v2p1/calibration_m1m2m3")
S3 = {"urgent_shrink", "confirmed_shrink", "forced_shrink_uncertain",
      "urgent_shrink_capped", "w_desync_freeze", "confirm_fail_freeze"}
PROBE_W_EVENTS = {"probe_start", "probe_fallback", "probe_abort_risk"}
M2_RUNS = [("cal_m2_w1_norm", "W1", "norm"),
           ("cal_m2_w1_abort", "W1", "abort"),
           ("cal_m2_w2_obs", "W2", "norm")]
STATE_ORDER = ["ARMED", "REF", "CAND", "FB", "DONE"]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "window_id" in r:
                rows.append(r)
    return rows


def main():
    report = {"gate": "M2", "runs": {}, "pass": None}
    ok = True
    for label, cond, kind in M2_RUNS:
        art = os.path.join(CAL, label)
        r = {"condition": cond, "kind": kind, "checks": {}}
        def chk(name, passed, detail=""):
            r["checks"][name] = {"pass": bool(passed), "detail": detail}
            return passed
        try:
            verdict = json.load(open(os.path.join(art, "verdict.json"))).get("verdict", "?")
        except Exception as e:
            verdict = f"MISSING({e})"
        ok &= chk("H1_valid", verdict == "VALID", verdict)
        try:
            rows = load_jsonl(os.path.join(art, "v2p1_rank0.jsonl"))
        except FileNotFoundError:
            ok &= chk("H0_jsonl", False, "missing")
            report["runs"][label] = r
            continue
        ok &= chk("H0_jsonl", len(rows) > 0, f"{len(rows)} windows")
        pre = None
        try:
            pre = json.load(open(os.path.join(art, "probe_run_end.json")))
        except Exception as e:
            pass
        ok &= chk("H10_run_end_present", pre is not None, "" if pre else "probe_run_end.json missing")
        by_w = {x["window_id"]: x for x in rows}

        # H2 license at most once
        trig_ws = [x["window_id"] for x in rows
                   if x.get("probe_action") in ("trigger", "trigger_gate_grow")]
        ok &= chk("H2_license_once", len(trig_ws) <= 1, str(trig_ws))
        if pre:
            ok &= chk("H2_license_flag", bool(pre["license_used"]) == (len(trig_ws) == 1),
                      f"flag={pre['license_used']} triggers={trig_ws}")

        # H3 trigger window
        trig_w = trig_ws[0] if trig_ws else None
        if trig_w is not None:
            if trig_w != 12:
                justification = []
                for w in range(12, trig_w):
                    x = by_w.get(w)
                    if x is None:
                        justification.append((w, "missing"))
                    elif x["event"] in S3 or (x.get("alert_k") or 0) > 0:
                        justification.append((w, x["event"], x.get("alert_k")))
                    else:
                        justification.append((w, "UNJUSTIFIED", x["event"], x.get("alert_k")))
                bad = [j for j in justification if "UNJUSTIFIED" in j or j[1] == "missing"]
                ok &= chk("H3_trigger_w12_or_delayed", not bad, str(justification))
                r["trigger_window"] = trig_w
                r["trigger_delay_justification"] = justification
            else:
                ok &= chk("H3_trigger_w12_or_delayed", True, "trigger at w12")
                r["trigger_window"] = 12
        else:
            ok &= chk("H3_trigger_w12_or_delayed", False, "no trigger fired at all")

        # H7 protocol health
        bad_lock = [x["window_id"] for x in rows if x.get("lockstep_ok") is not True]
        bad_h = [x["window_id"] for x in rows if x["committed"] and x.get("h_min") != x.get("h_max")]
        bad_des = [x["window_id"] for x in rows if x.get("desync")]
        bad_ev = [x["window_id"] for x in rows if x.get("event") in ("fatal_collective", "w_desync_freeze", "confirm_fail_freeze")]
        bad_mode = [x["window_id"] for x in rows if x.get("probe_mode") != "gate"]
        bad_perr = [x["window_id"] for x in rows if x.get("probe_error")]
        ok &= chk("H7_lockstep", not bad_lock, str(bad_lock[:5]))
        ok &= chk("H7_hash_unanimous", not bad_h, str(bad_h[:5]))
        ok &= chk("H7_no_desync", not bad_des, str(bad_des[:5]))
        ok &= chk("H7_no_fatal_freeze", not bad_ev, str(bad_ev[:5]))
        ok &= chk("H7_mode_gate", not bad_mode, str(bad_mode[:5]))
        ok &= chk("H7_no_probe_error", not bad_perr, str(bad_perr[:5]))

        # H8 state series monotone
        seen, order_ok, prev_idx = [], True, -1
        for x in rows:
            st = x.get("probe_state") or ""
            if st and (not seen or seen[-1] != st):
                seen.append(st)
            if st in STATE_ORDER:
                idx = STATE_ORDER.index(st)
                if idx < prev_idx:
                    order_ok = False
                prev_idx = max(prev_idx, idx)
            elif st == "ABORTED":
                pass
            elif st:
                order_ok = False
        r["state_series"] = seen
        ok &= chk("H8_state_monotone", order_ok, str(seen))
        if pre:
            ok &= chk("H8_terminal_matches", pre["state"] == (seen[-1] if seen else None),
                      f"run_end={pre['state']} jsonl_last={seen[-1] if seen else None}")

        # segment windows from jsonl probe_seg
        seg_ws = {"REF": [], "CAND": [], "FB": []}
        for x in rows:
            sg = x.get("probe_seg") or ""
            if sg in seg_ws:
                seg_ws[sg].append(x["window_id"])
        r["segment_windows"] = seg_ws
        contaminated = bool(pre and pre.get("contaminated"))
        aborted = bool(pre and pre.get("abort_reason"))
        if kind == "abort":
            # H4: CAND == force_abort_at(1) + lag windows (>=1); REF 4; FB 4
            ok &= chk("H4_budgets", len(seg_ws["REF"]) == 4 and len(seg_ws["CAND"]) >= 2
                      and len(seg_ws["FB"]) == 4,
                      f"REF={len(seg_ws['REF'])} CAND={len(seg_ws['CAND'])} FB={len(seg_ws['FB'])}")
        elif contaminated:
            ok &= chk("H4_budgets_or_contaminated", pre["state"] == "ABORTED",
                      f"truncated segments {({k: len(v) for k, v in seg_ws.items()})} + contaminated + ABORTED")
        else:
            ok &= chk("H4_budgets", len(seg_ws["REF"]) == 4 and len(seg_ws["CAND"]) == 3
                      and len(seg_ws["FB"]) == 4,
                      f"REF={len(seg_ws['REF'])} CAND={len(seg_ws['CAND'])} FB={len(seg_ws['FB'])}")

        # H6 abort semantics
        abort_ws = [x["window_id"] for x in rows if x.get("probe_action") == "probe_abort_risk" and x["committed"]]
        if kind == "abort":
            ok &= chk("H6_abort_fired", len(abort_ws) == 1, str(abort_ws))
            if pre:
                ev = pre.get("eval") or {}
                lag = [e for e in pre["segments"]["CAND"] if e.get("lag_window")]
                ok &= chk("H6_abort_reason", pre.get("abort_reason") == "force_abort_at", str(pre.get("abort_reason")))
                ok &= chk("H6_overshoot_recorded",
                          isinstance(pre.get("overshoot_us"), int) and pre["overshoot_us"] >= 0
                          and isinstance(pre.get("excess_cum_us"), int),
                          f"excess={pre.get('excess_cum_us')} overshoot={pre.get('overshoot_us')}")
                ok &= chk("H6_lag_flagged", len(lag) >= 1, str(lag))
                ok &= chk("H6_eval_unknown_abort_flag",
                          ev.get("classification") == "UNKNOWN"
                          and any(f.startswith("abort:") for f in ev.get("flags", [])),
                          f"cls={ev.get('classification')} flags={ev.get('flags')}")
                aw = abort_ws[0] if abort_ws else None
                after = [x for x in rows if aw and x["window_id"] > aw and x["committed"]]
                ok &= chk("H6_w16_after_abort", all(x["W_after"] == 16 for x in after),
                          str([(x['window_id'], x['W_after']) for x in after if x['W_after'] != 16][:5]))
        else:
            ok &= chk("H6_no_abort_norm", not aborted and not abort_ws,
                      f"reason={pre.get('abort_reason') if pre else None} ws={abort_ws}")

        # H5 ABA completes -> W16 after EVAL
        if pre and pre.get("eval_window") and not contaminated:
            ew = pre["eval_window"]
            after = [x for x in rows if x["window_id"] > ew and x["committed"]]
            bad = []
            for x in after:
                if x["W_after"] != 16 and x["event"] not in S3:
                    bad.append((x["window_id"], x["W_after"], x["event"]))
            ok &= chk("H5_w16_after_eval", not bad, str(bad[:5]))
            r["eval_window"] = ew
            r["eval_classification"] = (pre.get("eval") or {}).get("classification")
            r["delta_net_us_per_step"] = (pre.get("eval") or {}).get("delta_net_us_per_step")

        # H9 gating audit
        grow_committed = [x["window_id"] for x in rows if x["committed"] and x["event"] == "confirmed_grow"]
        ok &= chk("H9_no_committed_grow", not grow_committed, str(grow_committed[:5]))
        w64 = [x["window_id"] for x in rows if x["committed"] and x["W_after"] == 64]
        ok &= chk("H9_never_w64", not w64, str(w64[:5]))
        cand_set = set(seg_ws["CAND"]) | {max(seg_ws["CAND"]) + 1 if seg_ws["CAND"] else -1}
        # 2026-10-18 POST-FREEZE CORRECTION (PRE_REG §F #6): original implementation
        # flagged EVERY committed W_after==32 window whose event != probe_start, which
        # contradicts the frozen timeline (CAND segment runs n_cand windows AT W32 --
        # hold windows w17/w18 have W_before==32, seg==CAND, by design; unit-tested
        # pre-freeze). Corrected to the frozen intent, STRICTER than transition-only:
        # W_after==32 legal iff (a) entry via probe_start, or (b) hold inside the CAND
        # segment (W_before==32 and probe_seg=="CAND"). Any other W32 window fails.
        w32_bad = [(x["window_id"], x["W_before"], x["event"], x.get("probe_seg"))
                   for x in rows
                   if x["committed"] and x["W_after"] == 32
                   and x["event"] != "probe_start"
                   and not (x["W_before"] == 32 and x.get("probe_seg") == "CAND")]
        ok &= chk("H9_w32_only_probe_start", not w32_bad, str(w32_bad[:5]))
        trans_bad = []
        for x in rows:
            if x["committed"] and x["W_after"] != x["W_before"]:
                if x["event"] not in PROBE_W_EVENTS | S3 | {"init"}:
                    trans_bad.append((x["window_id"], x["W_before"], x["W_after"], x["event"]))
        ok &= chk("H9_transition_events", not trans_bad, str(trans_bad[:5]))
        gated = [x["window_id"] for x in rows if x["committed"] and x["event"] == "grow_gated_probe"]
        r["grow_gated_windows"] = gated
        ok &= chk("H9_gating_fired", len(gated) >= 1, str(gated[:8]))

        # ledger (reported, not gated here)
        r["ledger"] = {
            "median_c_window_ms": round(statistics.median([x["c_window_ms"] for x in rows]), 4),
            "median_prev_bcost_us": statistics.median([x.get("prev_bcost_us") or 0 for x in rows]),
            "final_prof_probe_cum_ms": rows[-1].get("prof_probe_cum_ms"),
            "avg_probe_us_per_window": round(1000 * (rows[-1].get("prof_probe_cum_ms") or 0) / len(rows), 2),
            "total_time_s": None,
        }
        try:
            bj = json.load(open(os.path.join(art, "bench.json")))
            r["ledger"]["total_time_s"] = bj.get("total_time_s")
            ok &= chk("H11_bench_mono", all(k in bj for k in ("mono_start_ns", "mono_end_ns")), "")
        except Exception:
            ok &= chk("H11_bench_mono", False, "bench.json missing")
        report["runs"][label] = r
    report["pass"] = bool(ok)
    out = os.path.join(CAL, "m2_gate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("M2_GATE:", "PASS" if ok else "FAIL", "->", out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
