"""Offline unit tests for probe_admission.ProbeAdmission (spec v1.1-calib).

No torch / no GPU / no I/O. Run: python3 test_probe_admission.py
Frozen-protocol references: PRE_REG_calibration_m1m2m3.md sections A/C/F.
"""
import json
import sys

from probe_admission import (
    ProbeAdmission, S3_EVENTS, EV_GATE_GROW, EV_PROBE_START,
    EV_PROBE_START_AAA, EV_PROBE_FALLBACK, EV_PROBE_FALLBACK_AAA,
    EV_PROBE_ABORT, EV_COMMIT_ADMIT)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def mkrec(event, w, w_after=None, alert_k=0):
    return {"event": event, "W_before": w,
            "W_after": w if w_after is None else w_after,
            "alert_k": alert_k, "reasons": []}


RATE_BY_STATE = {"ARMED": 100.0, "REF": 100.0, "CAND": 110.0,
                 "FB": 120.0, "DONE": 130.0, "ABORTED": 130.0}


def run(n_windows=30, mode="gate", eps_risk=2000000, force=0, commit=False,
        grow_at=(), s3_at=(), holds=(), alert_at=(), r=12,
        rate_map=None, eval_commit=False):
    """Deterministic harness: one window = propose -> (controller applies
    override) -> advance with reduced series. Returns (machine, trace)."""
    m = ProbeAdmission(mode=mode, r=r, n_ref=4, n_cand=3, n_fb=4,
                       w_base=16, w_cand=32, eps_risk_us=eps_risk,
                       force_abort_at=force, commit_enabled=commit)
    W = 16
    cum_true = 0
    trace = []
    for w in range(1, n_windows + 1):
        state_at_exec = m.state
        steps = W
        cum_true += steps
        rate = (rate_map or {}).get(w, RATE_BY_STATE.get(state_at_exec, 100.0))
        span_us = int(steps * rate)
        if w in s3_at:
            rec = mkrec("confirmed_shrink", W, w_after=8)
        elif w in grow_at:
            rec = mkrec("confirmed_grow", W, w_after=min(64, W * 2))
        else:
            rec = mkrec("midband", W, alert_k=1 if w in alert_at else 0)
        pact = m.propose(w, rec)
        committed = w not in holds
        w_applied = W
        if committed:
            if pact.get("override"):
                w_applied = int(pact["override"]["W_after"])
            else:
                w_applied = int(rec["W_after"])
        pinfo = m.advance(w, committed, steps=steps, span_us=span_us,
                          bcost_us=500, true_steps_total=cum_true,
                          p2l_version=0)
        if committed and pinfo and pinfo.get("directive"):
            d = pinfo["directive"]
            if "set_W" in d:
                W = int(d["set_W"])
            else:
                W = w_applied
        elif committed:
            W = w_applied
        trace.append({"w": w, "state": state_at_exec, "action": pact["action"],
                      "override": pact["override"], "committed": committed,
                      "W_next": W, "steps": steps, "span_us": span_us,
                      "state_after": m.state})
    return m, trace


def actions_after(trace, w0):
    return {t["w"]: t for t in trace if t["w"] >= w0}


# ---------------------------------------------------------------- scenarios
def test_gate_normal():
    print("[gate normal path]")
    m, tr = run(n_windows=30, mode="gate")
    t = actions_after(tr, 1)
    check("trigger at w12", m.trigger_window == 12 and m.license_used,
          str(m.trigger_window))
    check("w12 action=trigger", t[12]["action"] == "trigger", t[12]["action"])
    check("REF = w13..w16", [e["window_id"] for e in m.segments["REF"]] == [13, 14, 15, 16])
    check("probe_start override at w16 (W32)",
          t[16]["action"] == "probe_start" and t[16]["override"]["W_after"] == 32
          and t[16]["override"]["event"] == EV_PROBE_START)
    check("CAND = w17..w19 @32 steps",
          [e["window_id"] for e in m.segments["CAND"]] == [17, 18, 19]
          and all(e["steps"] == 32 for e in m.segments["CAND"]))
    check("probe_fallback override at w19 (W16)",
          t[19]["action"] == "probe_fallback" and t[19]["override"]["W_after"] == 16
          and t[19]["override"]["event"] == EV_PROBE_FALLBACK)
    check("FB = w20..w23 @16 steps",
          [e["window_id"] for e in m.segments["FB"]] == [20, 21, 22, 23]
          and all(e["steps"] == 16 for e in m.segments["FB"]))
    check("DONE at w23", m.state == "DONE" and m.eval_window == 23,
          "%s/%s" % (m.state, m.eval_window))
    check("segment step budgets 64/96/64",
          m.eval["n_steps"] == {"REF": 64, "CAND": 96, "FB": 64}, str(m.eval["n_steps"]))
    # hand math: rate_ref=100 rate_cand=110 rate_fb=120; mid_ref=224 mid_cand=304
    # mid_fb=384 -> g=0.125 pred=110 delta=0 -> UNKNOWN (drift fully corrected)
    ev = m.eval
    check("hand math rates", abs(ev["rate_ref_us_per_step"] - 100) < 1e-9
          and abs(ev["rate_cand_us_per_step"] - 110) < 1e-9
          and abs(ev["rate_fb_us_per_step"] - 120) < 1e-9)
    check("hand math midpoints", ev["midpoints_steps"] ==
          {"ref": 224.0, "cand": 304.0, "fb": 384.0}, str(ev["midpoints_steps"]))
    check("hand math g/pred/delta", abs(ev["g_us_per_step2"] - 0.125) < 1e-12
          and abs(ev["predicted_cand_us_per_step"] - 110) < 1e-9
          and abs(ev["delta_net_us_per_step"]) < 1e-9, str(ev))
    check("flat-drift classification UNKNOWN", ev["classification"] == "UNKNOWN")
    check("no flags", ev["flags"] == [], str(ev["flags"]))
    check("no causal claim / coverage blank",
          ev["causal_isolation_claimed"] is False and ev["coverage_guarantee"] is None)
    check("commit not eligible (calib COMMIT=0)", ev["commit_eligible"] is False)
    # W after DONE stays 16, growth gated whole-run
    check("post-DONE growth gated", all(
        t[w]["action"] in ("gate_grow", "") for w in range(24, 31)))


def test_growth_gating_armed():
    print("[growth gating in ARMED]")
    m, tr = run(n_windows=14, mode="gate", grow_at=(1, 2, 5))
    t = {x["w"]: x for x in tr}
    for w in (1, 2, 5):
        check("w%d grow gated" % w,
              t[w]["action"] == "gate_grow"
              and t[w]["override"]["event"] == EV_GATE_GROW
              and t[w]["override"]["W_after"] == 16)
    check("state stays ARMED until trigger", t[5]["state"] == "ARMED")
    check("W never left 16", all(x["W_next"] == 16 for x in tr if x["w"] < 13))
    check("trigger still fires at w12", m.trigger_window == 12)
    # trigger window itself with a grow rec -> combined action
    m2, tr2 = run(n_windows=13, mode="gate", grow_at=(12,))
    t2 = {x["w"]: x for x in tr2}
    check("w12 trigger+gate combined",
          t2[12]["action"] == "trigger_gate_grow"
          and t2[12]["override"]["event"] == EV_GATE_GROW
          and t2[12]["override"]["W_after"] == 16)
    check("license consumed on combined trigger", m2.trigger_window == 12)


def test_favorable_unfavorable():
    print("[classification favorable/unfavorable]")
    # cand rate 105 vs predicted 110 -> delta=-5 <= -eps_prac(=1.0) favorable
    rm = {17: None}
    m, _ = run(n_windows=24, mode="gate",
               rate_map={**{w: 105.0 for w in (17, 18, 19)}})
    ev = m.eval
    check("favorable delta=-5", abs(ev["delta_net_us_per_step"] + 5) < 1e-9
          and ev["classification"] == "favorable", str(ev["delta_net_us_per_step"]))
    m, _ = run(n_windows=24, mode="gate",
               rate_map={**{w: 115.0 for w in (17, 18, 19)}})
    ev = m.eval
    check("unfavorable delta=+5", abs(ev["delta_net_us_per_step"] - 5) < 1e-9
          and ev["classification"] == "unfavorable")
    m, _ = run(n_windows=24, mode="gate",
               rate_map={**{w: 110.5 for w in (17, 18, 19)}})
    check("within eps_prac -> UNKNOWN",
          m.eval["classification"] == "UNKNOWN"
          and abs(m.eval["delta_net_us_per_step"] - 0.5) < 1e-9)


def test_aaa():
    print("[aaa null arm]")
    m, tr = run(n_windows=30, mode="aaa")
    t = {x["w"]: x for x in tr}
    check("cand target 6 windows (96 steps @W16)", m.cand_window_target() == 6)
    check("probe_start_aaa at w16, W stays 16",
          t[16]["action"] == "probe_start_aaa"
          and t[16]["override"]["W_after"] == 16
          and t[16]["override"]["event"] == EV_PROBE_START_AAA)
    check("CAND = w17..w22 @16 steps",
          [e["window_id"] for e in m.segments["CAND"]] == [17, 18, 19, 20, 21, 22]
          and all(e["steps"] == 16 for e in m.segments["CAND"]))
    check("probe_fallback_aaa at w22",
          t[22]["action"] == "probe_fallback_aaa"
          and t[22]["override"]["event"] == EV_PROBE_FALLBACK_AAA)
    check("FB = w23..w26", [e["window_id"] for e in m.segments["FB"]] == [23, 24, 25, 26])
    check("DONE at w26", m.state == "DONE" and m.eval_window == 26)
    check("budgets match ABA (64/96/64)",
          m.eval["n_steps"] == {"REF": 64, "CAND": 96, "FB": 64}, str(m.eval["n_steps"]))
    check("W never != 16 in aaa", all(x["W_next"] == 16 for x in tr))
    check("no set_W directives in aaa", all(
        "set_W" not in (x["override"] or {}) for x in tr))
    # AAA with flat rates everywhere -> delta 0 -> no false positive
    m2, _ = run(n_windows=28, mode="aaa",
                rate_map={w: 100.0 for w in range(1, 29)})
    check("AAA flat: delta=0 UNKNOWN (no false positive)",
          abs(m2.eval["delta_net_us_per_step"]) < 1e-9
          and m2.eval["classification"] == "UNKNOWN")


def test_risk_abort():
    print("[risk abort + overshoot + one-window lag]")
    # eps_risk tiny; cand windows cost rate 5000 us/step
    m, tr = run(n_windows=26, mode="gate", eps_risk=1000,
                rate_map={17: 5000.0, 18: 5000.0})
    t = {x["w"]: x for x in tr}
    check("pending_abort set at w17 close", t[18]["action"] == "probe_abort_risk"
          and t[18]["override"]["W_after"] == 16
          and t[18]["override"]["event"] == EV_PROBE_ABORT)
    check("abort_reason eps_risk", m.abort_reason == "eps_risk", str(m.abort_reason))
    # excess after w17: 32 steps * (5000-100) = 156800; overshoot = 155800
    check("overshoot recorded", m.overshoot_us == 155800, str(m.overshoot_us))
    check("lag window w18 recorded in CAND with lag flag",
          len(m.segments["CAND"]) == 2
          and m.segments["CAND"][1]["window_id"] == 18
          and m.segments["CAND"][1].get("lag_window") is True)
    check("FB runs w19..w22 then DONE",
          [e["window_id"] for e in m.segments["FB"]] == [19, 20, 21, 22]
          and m.state == "DONE")
    check("eval UNKNOWN with abort flag",
          m.eval["classification"] == "UNKNOWN"
          and "abort:eps_risk" in m.eval["flags"], str(m.eval["flags"]))
    check("license consumed (no retry)", m.license_used and m.trigger_window == 12)
    check("no second probe after DONE", m.state == "DONE")


def test_force_abort():
    print("[force abort (M2 mechanism config)]")
    m, tr = run(n_windows=26, mode="gate", eps_risk=2000000, force=1)
    t = {x["w"]: x for x in tr}
    check("force abort after 1 cand window", t[18]["action"] == "probe_abort_risk")
    check("abort_reason force_abort_at", m.abort_reason == "force_abort_at",
          str(m.abort_reason))
    check("excess below eps -> overshoot 0", m.overshoot_us == 0)
    check("eval UNKNOWN abort flag", m.eval["classification"] == "UNKNOWN"
          and "abort:force_abort_at" in m.eval["flags"])


def test_s3_contamination():
    print("[S3 regime contamination mid-CAND]")
    m, tr = run(n_windows=26, mode="gate", s3_at=(18,))
    t = {x["w"]: x for x in tr}
    check("takeover_s3 at w18, detector rec passes through",
          t[18]["action"] == "takeover_s3" and t[18]["override"] is None)
    check("W follows detector shrink to 8", t[18]["W_next"] == 8)
    check("state ABORTED + contaminated", m.state == "ABORTED" and m.contaminated)
    check("segments truncated but retained",
          [e["window_id"] for e in m.segments["CAND"]] == [17, 18]
          and m.segments["FB"] == [])
    check("eval UNKNOWN contaminated_s3",
          m.eval["classification"] == "UNKNOWN"
          and "contaminated_s3" in m.eval["flags"])
    check("no retry after contamination", m.license_used)


def test_hold_anomaly_and_idempotent_retry():
    print("[hold anomaly + idempotent transition retry]")
    # hold at w14 (inside REF) -> anomaly flag; segment still completes
    m, tr = run(n_windows=24, mode="gate", holds=(14,))
    check("hold window recorded in REF",
          14 in [e["window_id"] for e in m.segments["REF"]])
    check("REF flagged hold_anomaly", "hold_anomaly" in m.seg_flags["REF"])
    check("eval UNKNOWN due to anomaly",
          m.eval["classification"] == "UNKNOWN"
          and any("hold_anomaly" in f for f in m.eval["flags"]))
    # hold exactly at the transition window w16 -> probe_start re-proposed at w17
    m2, tr2 = run(n_windows=26, mode="gate", holds=(16,))
    t2 = {x["w"]: x for x in tr2}
    check("w16 hold: no transition", t2[16]["state_after"] == "REF")
    check("w17 re-proposes probe_start", t2[17]["action"] == "probe_start"
          and t2[17]["override"]["W_after"] == 32)
    check("REF has 5 entries (>= budget, idempotent)",
          len(m2.segments["REF"]) == 5
          and [e["window_id"] for e in m2.segments["REF"]] == [13, 14, 15, 16, 17])
    check("CAND w18..w20 after delayed start",
          [e["window_id"] for e in m2.segments["CAND"]] == [18, 19, 20])
    check("eval UNKNOWN (hold flag)", m2.eval["classification"] == "UNKNOWN")


def test_delayed_trigger_and_zero_probe():
    print("[delayed trigger / zero-probe legal outcome]")
    m, tr = run(n_windows=30, mode="gate", s3_at=(12,), alert_at=(13,))
    check("no trigger at w12 (S3) nor w13 (alert_k>0)", m.trigger_window == 14,
          str(m.trigger_window))
    check("REF starts w15", [e["window_id"] for e in m.segments["REF"]][0] == 15)
    # zero-probe: every window >= R has alert_k>0 -> license never consumed
    m2, tr2 = run(n_windows=40, mode="gate", alert_at=tuple(range(12, 41)))
    check("zero probe legal: license unused, state ARMED",
          m2.license_used is False and m2.state == "ARMED"
          and m2.trigger_window is None and m2.eval is None)


def test_production_deferred_commit_offline():
    print("[production deferred commit (offline-only path)]")
    m, tr = run(n_windows=26, mode="gate", commit=True,
                rate_map={17: 105.0, 18: 105.0, 19: 105.0})
    t = {x["w"]: x for x in tr}
    check("eval favorable", m.eval["classification"] == "favorable")
    check("commit deferred exactly one window after EVAL (w23+1=w24)",
          t[24]["action"] == "commit_admit"
          and t[24]["override"]["event"] == EV_COMMIT_ADMIT
          and t[24]["override"]["W_after"] == 32, t[24]["action"])
    check("admission applied once", m.admission_applied is True)
    check("no further commit_admit", all(t[w]["action"] != "commit_admit"
                                         for w in range(25, 27)))


def test_determinism_and_purity():
    print("[determinism + propose purity]")
    m1, tr1 = run(n_windows=26, mode="gate", grow_at=(3, 9), holds=(15,))
    m2, tr2 = run(n_windows=26, mode="gate", grow_at=(3, 9), holds=(15,))
    check("identical runs -> identical summaries",
          json.dumps(m1.summary(), sort_keys=True, default=str)
          == json.dumps(m2.summary(), sort_keys=True, default=str))
    # propose purity: double propose without advance changes nothing
    m3 = ProbeAdmission(mode="gate")
    rec = mkrec("midband", 16)
    for w in range(1, 16):
        m3.propose(w, rec)
    a = json.dumps(m3.summary(), sort_keys=True, default=str)
    for w in range(1, 16):
        m3.propose(w, rec)
    b = json.dumps(m3.summary(), sort_keys=True, default=str)
    check("propose does not mutate segments/state", a == b)


def test_off_mode():
    print("[mode off = inert]")
    m, tr = run(n_windows=20, mode="off", grow_at=(1, 5))
    check("off: no actions, no segments, no eval",
          all(x["action"] == "" for x in tr) and m.state == "ARMED"
          and m.eval is None and not m.license_used
          and all(x["override"] is None for x in tr))


if __name__ == "__main__":
    test_gate_normal()
    test_growth_gating_armed()
    test_favorable_unfavorable()
    test_aaa()
    test_risk_abort()
    test_force_abort()
    test_s3_contamination()
    test_hold_anomaly_and_idempotent_retry()
    test_delayed_trigger_and_zero_probe()
    test_production_deferred_commit_offline()
    test_determinism_and_purity()
    test_off_mode()
    print()
    if FAILURES:
        print("FAILURES: %d -> %s" % (len(FAILURES), FAILURES))
        sys.exit(1)
    print("ALL TESTS PASSED")
