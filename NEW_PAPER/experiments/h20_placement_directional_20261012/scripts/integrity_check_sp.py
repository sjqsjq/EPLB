#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrity_check_sp.py -- per-start verdict gate for sp_coupling_v1 dev runs
(plan 2026-10-18 Sec.7/Sec.8; cloned from integrity_check.py, SP-adapted;
locked BEFORE the first smoke start; never compares arms; no endpoint values).

Usage:
  python3 integrity_check_sp.py --artifact-dir DIR --arm {A,F_dense,F_sparse}
      --condition {W1,W2,SMOKE} --label LABEL --expected-requests N

Verdicts (printed as "VERDICT:<X> t=<total_time_s>"):
  VALID           exit 0
  MECH_RETIRE     exit 2 -- fatal_collective / w_desync_freeze /
                    confirm_fail_freeze / desync==true in rank0 jsonl;
                    fixed-arm transition or non-constant S/H/W; any
                    (sp_s,sp_h,sp_w) off the frozen table; PD-9 gate
                    count violation (S1: opps==sp_fwd; S4: |opps-sp_fwd/4|>1);
                    sp_run_end vs jsonl H-cycle mismatch
  RESULT_RETIRE   exit 1 -- bench.json missing/unparsable, ok<N, errors>0,
                    metadata mismatch (label/dataset/concurrency)
  INTEGRITY_FAIL  exit 3 -- ADW21 ACTIVE banner <8; SP ACTIVE banner <8 or
                    wrong mode/S/H/W/seed; knob strings not unique or !=
                    frozen authority; meta.txt SP env mismatch; rank0 jsonl
                    lines < 90% expected; sp_run_end.json missing/unparsable

Priority: MECH_RETIRE > RESULT_RETIRE > INTEGRITY_FAIL. Writes verdict.json.

Expected-line basis (locked, same constants as integrity_check.py): measured
W0 dispatch 8037 rows -> 1056 engine steps; est_steps(N)=round(1056*N/8037);
lines are per H cycle: H_ref=16 (A, F_dense), 64 (F_sparse); gate=floor(0.9*).
"""
import argparse, json, os, re, sys

FATAL_EVENTS = {"fatal_collective", "w_desync_freeze", "confirm_fail_freeze"}
SP_TABLE_SET = {(1, 16, 16), (4, 16, 64), (1, 8, 8)}
STEPS_W0_8037 = 1056
NROWS = {"W1": 4000, "W2": 8000, "SMOKE": 300}
DSNAME = {"W1": "w1_stable_code_4000.jsonl",
          "W2": "w2_switch_code_chinese_8000.jsonl",
          "SMOKE": "smoke_w1_300.jsonl"}
A_KNOB_MARKS = ["'kappa': 3875.1768798828125", "'z_lo': 22.643858727592757",
                "'z_hi': 314.14010272527486", "'z_urgent': 942.4203081758246",
                "'n_min': 12320768", "'theta_h_low': 0.010095552687949321",
                "'theta_h_act': 0.011395643112507271"]
# frozen per-arm SP identity (MAPPING PD-11/PD-12; plan Sec.6)
ARM_SPEC = {
    "A":        {"mode": "adaptive", "S": 1, "H": 16, "W": 16,
                 "env": {"OEPLB_SP_MODE": "adaptive", "OEPLB_SP_SEED": "20261018",
                         "OEPLB_SP_QUIET_STREAK": "3", "OEPLB_SP_RECOVER_MIN_CYCLES": "2"}},
    # rev1 dev protocol (PROTOCOL_dev_r1_validation_20261018.md): A_v1 is the
    # original adaptive arm re-run under the rev gate (SP_REV=0, byte-frozen
    # v1 behavior); A_rev1 is the revised QUIET-exit routing (SP_REV=1).
    # Both are ADAPTIVE -- identical ledgers/bounds as "A"; the only env
    # difference is OEPLB_SP_REV.
    "A_v1":     {"mode": "adaptive", "S": 1, "H": 16, "W": 16,
                 "env": {"OEPLB_SP_MODE": "adaptive", "OEPLB_SP_SEED": "20261018",
                         "OEPLB_SP_QUIET_STREAK": "3", "OEPLB_SP_RECOVER_MIN_CYCLES": "2",
                         "OEPLB_SP_REV": "0"}},
    "A_rev1":   {"mode": "adaptive", "S": 1, "H": 16, "W": 16,
                 "env": {"OEPLB_SP_MODE": "adaptive", "OEPLB_SP_SEED": "20261018",
                         "OEPLB_SP_QUIET_STREAK": "3", "OEPLB_SP_RECOVER_MIN_CYCLES": "2",
                         "OEPLB_SP_REV": "1"}},
    "F_dense":  {"mode": "fixed", "S": 1, "H": 16, "W": 16,
                 "env": {"OEPLB_SP_MODE": "fixed", "OEPLB_SP_S": "1", "OEPLB_SP_W": "16"}},
    "F_sparse": {"mode": "fixed", "S": 4, "H": 64, "W": 64,
                 "env": {"OEPLB_SP_MODE": "fixed", "OEPLB_SP_S": "4", "OEPLB_SP_W": "64"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_SPEC))
    ap.add_argument("--condition", required=True, choices=sorted(NROWS))
    ap.add_argument("--label", required=True)
    ap.add_argument("--expected-requests", required=True, type=int)
    a = ap.parse_args()
    art = a.artifact_dir
    spec = ARM_SPEC[a.arm]
    det = {"arm": a.arm, "condition": a.condition, "label": a.label,
           "protocol": "sp_coupling_v1", "checks": {}}
    mech, res, integ = [], [], []

    # ---------- bench.json (result validity) ----------
    ttime = None
    try:
        b = json.load(open(os.path.join(art, "bench.json")))
        det["checks"]["bench"] = {k: b.get(k) for k in
            ("label", "dataset", "num_requests", "concurrency", "ok",
             "errors", "total_time_s")}
        if b.get("ok") != a.expected_requests:
            res.append("ok=%s != expected %d" % (b.get("ok"), a.expected_requests))
        if b.get("errors", 1) != 0:
            res.append("errors=%s != 0" % b.get("errors"))
        if b.get("num_requests") != a.expected_requests:
            res.append("num_requests=%s != %d" % (b.get("num_requests"), a.expected_requests))
        if b.get("concurrency") != 256:
            res.append("concurrency=%s != 256" % b.get("concurrency"))
        if b.get("label") != a.label:
            res.append("label=%s != %s" % (b.get("label"), a.label))
        if os.path.basename(str(b.get("dataset", ""))) != DSNAME[a.condition]:
            res.append("dataset=%s != %s" % (b.get("dataset"), DSNAME[a.condition]))
        t = b.get("total_time_s")
        if not (isinstance(t, (int, float)) and t > 0):
            res.append("total_time_s missing/nonpositive: %r" % (t,))
        else:
            ttime = t
    except Exception as e:
        res.append("bench.json missing/unparsable: %r" % (e,))
        det["checks"]["bench"] = {"error": repr(e)}

    # ---------- server.log banners ----------
    try:
        slog = open(os.path.join(art, "server.log"), encoding="utf-8",
                    errors="replace").read()
    except Exception as e:
        slog = ""
        integ.append("server.log missing: %r" % (e,))
    nban = slog.count("[PB-OEPLB-ADW21] limited-revision controller v2.1 ACTIVE")
    det["checks"]["adw21_banner_count"] = nban
    if nban < 8:
        integ.append("ADW21 ACTIVE banner count %d < 8 ranks" % nban)
    ks = re.findall(r"knobs=(\{.*?\}) \(theta_", slog)
    uniq = sorted(set(ks))
    det["checks"]["knob_strings_unique"] = len(uniq)
    if len(uniq) != 1:
        integ.append("knob strings not unique across ranks: %d variants" % len(uniq))
    else:
        miss = [m for m in A_KNOB_MARKS if m not in uniq[0]]
        det["checks"]["knob_authority"] = "A-frozen" if not miss else "MISMATCH %s" % miss
        if miss:
            integ.append("knob authority mismatch (planner knobs are shared "
                         "and frozen for ALL sp arms): %s" % miss)
    nsp = slog.count("[PB-OEPLB-SP] sampling/planning coupling ACTIVE")
    det["checks"]["sp_banner_count"] = nsp
    if nsp < 8:
        integ.append("SP ACTIVE banner count %d < 8 ranks" % nsp)
    spm = re.findall(r"coupling ACTIVE: mode=(\S+) S=(\d+) H=(\d+) W=(\d+) "
                     r"seed=(\d+) quiet_streak=(\d+) recover_min_cycles=(\d+)",
                     slog)
    spu = sorted(set(spm))
    det["checks"]["sp_banner_variants"] = spu
    if len(spu) != 1:
        integ.append("SP banner not identical across ranks: %d variants" % len(spu))
    else:
        mode, S, H, W, seed, qs, rmc = spu[0]
        exp = (spec["mode"], str(spec["S"]), str(spec["H"]), str(spec["W"]))
        if (mode, S, H, W) != exp:
            integ.append("SP identity %s != expected %s" % ((mode, S, H, W), exp))
        if (seed, qs, rmc) != ("20261018", "3", "2"):
            integ.append("SP streak params %s != (20261018, 3, 2)" % ((seed, qs, rmc),))

    # ---------- meta.txt SP env ----------
    try:
        meta = open(os.path.join(art, "meta.txt")).read()
        menv = dict(l.split("=", 1) for l in meta.splitlines()
                    if l.startswith("OEPLB_SP_") and "=" in l)
        det["checks"]["meta_sp_env"] = menv
        for k, v in spec["env"].items():
            if menv.get(k) != v:
                integ.append("meta.txt %s=%r != expected %r" % (k, menv.get(k), v))
        if menv.get("OEPLB_PROBE_MODE", "off") != "off":
            integ.append("probe not off in SP run: %r" % menv.get("OEPLB_PROBE_MODE"))
    except Exception as e:
        integ.append("meta.txt missing/unparsable: %r" % (e,))

    # ---------- rank0 jsonl (ALL sp arms) ----------
    rows = []
    try:
        for ln in open(os.path.join(art, "v2p1_rank0.jsonl"), encoding="utf-8"):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    except Exception as e:
        integ.append("rank0 jsonl missing/unparsable: %r" % (e,))
    det["checks"]["jsonl_lines"] = len(rows)
    est_steps = round(STEPS_W0_8037 * NROWS[a.condition] / 8037.0)
    h_ref = spec["H"] if a.arm == "F_sparse" else 16
    need = int(0.9 * (est_steps / h_ref))
    det["checks"]["jsonl_min_lines"] = need
    if rows and len(rows) < need:
        integ.append("jsonl lines %d < 90%% expected (%d)" % (len(rows), need))
    fixed = (spec["mode"] != "adaptive")   # rev1: mode-based, not arm-name
    for r in rows:
        wid = r.get("window_id")
        ev = r.get("event")
        if ev in FATAL_EVENTS:
            mech.append("fatal mechanism event %s at window %s" % (ev, wid))
        if r.get("desync") is True:
            mech.append("desync==true at window %s" % wid)
        shw = (r.get("sp_s"), r.get("sp_h"), r.get("sp_w"))
        if shw not in SP_TABLE_SET and not (fixed and shw == (
                spec["S"], spec["H"], spec["W"])):
            mech.append("S/H/W %s off frozen table at sp_h %s" % (shw, r.get("sp_h_id")))
        if r.get("sp_mode") != spec["mode"]:
            mech.append("sp_mode %r != %r at sp_h %s" % (r.get("sp_mode"), spec["mode"], r.get("sp_h_id")))
        if fixed and r.get("sp_transition"):
            mech.append("fixed arm transitioned at sp_h %s" % r.get("sp_h_id"))
        if r.get("sp_recorded_steps", 0) > r.get("sp_sample_opportunities", 0):
            mech.append("recorded>opportunities at sp_h %s" % r.get("sp_h_id"))
    mech = mech[:10]

    # ---------- sp_run_end.json (mechanism ledger) ----------
    try:
        se = json.load(open(os.path.join(art, "sp_run_end.json")))
        det["checks"]["sp_run_end"] = {k: se.get(k) for k in
            ("mode", "state_final", "S_final", "H_final", "W_final",
             "sp_fwd_total", "h_cycles", "window_count", "true_steps_total",
             "sample_opportunities", "recorded_steps", "swaps_started_total",
             "residency_committed_h", "p2l_version")}
        det["checks"]["sp_transitions"] = se.get("transitions")
        det["checks"]["sp_seals"] = se.get("seals")
        if se.get("mode") != spec["mode"]:
            integ.append("sp_run_end mode %r != %r" % (se.get("mode"), spec["mode"]))
        if fixed:
            if se.get("transitions"):
                mech.append("fixed arm sp_run_end shows %d transitions"
                            % len(se["transitions"]))
            if (se.get("S_final"), se.get("H_final"), se.get("W_final")) != (
                    spec["S"], spec["H"], spec["W"]):
                mech.append("fixed arm final S/H/W %s != %s" % (
                    (se.get("S_final"), se.get("H_final"), se.get("W_final")),
                    (spec["S"], spec["H"], spec["W"])))
        else:
            if (se.get("S_final"), se.get("H_final"), se.get("W_final")) not in SP_TABLE_SET:
                mech.append("adaptive final S/H/W %s off table" % (
                    (se.get("S_final"), se.get("H_final"), se.get("W_final")),))
            res_sum = sum(int(v) for v in (se.get("residency_committed_h") or {}).values())
            if res_sum != int(se.get("h_cycles", -1)):
                mech.append("residency sum %d != h_cycles %s" % (res_sum, se.get("h_cycles")))
        # PD-9 gate arithmetic (deterministic, data-independent).
        # CHECKER_FIX 2026-10-18 (seq2 MECH HALT audit): the exact S1
        # identity is mathematically inapplicable to the ADAPTIVE arm,
        # whose S varies by state by design (QUIET=S4). It is kept for
        # fixed arms only; for A the gate count is bounded from the
        # run's own committed ledgers (residency + QUIET episodes),
        # independent of any measured endpoint value:
        #   steps_q  = QUIET_cycles*16 ; steps_s1 = TRACK*16 + RECOVER*8
        #   lo = steps_s1 + steps_q//4 - 2*q_eps
        #   hi = (fwd - steps_q) + ceil(steps_q/4) + 2*q_eps + idle_slack
        # (2*q_eps: one block-edge slack per QUIET entry AND exit; the hi
        # side charges every idle forward at the S1 rate, the maximum.)
        fwd = int(se.get("sp_fwd_total", -1))
        opps = int(se.get("sample_opportunities", -2))
        if fixed:
            if spec["S"] == 1 and fwd >= 0 and opps != fwd:
                mech.append("S1 gate: opportunities %d != sp_fwd %d" % (opps, fwd))
            if spec["S"] == 4 and fwd >= 0 and abs(opps * 4 - fwd) > 4:
                mech.append("S4 gate: opportunities %d not ~= sp_fwd/4 (%d)" % (opps, fwd))
        else:
            res_h = {k: int(v) for k, v in
                     (se.get("residency_committed_h") or {}).items()}
            steps_q = res_h.get("QUIET", 0) * 16
            steps_s1 = res_h.get("TRACK", 0) * 16 + res_h.get("RECOVER", 0) * 8
            q_eps = sum(1 for t in (se.get("transitions") or [])
                        if t.get("to") == "QUIET")
            idle_n = max(0, fwd - (steps_q + steps_s1))
            lo = steps_s1 + steps_q // 4 - 2 * q_eps
            hi = (fwd - steps_q) + (steps_q + 3) // 4 + 2 * q_eps
            det["checks"]["pd9_adaptive_bound"] = [lo, hi, idle_n]
            if fwd >= 0 and not (lo <= opps <= hi):
                mech.append("adaptive PD-9 bound: opportunities %d not in "
                            "[%d,%d] (steps_s1=%d steps_q=%d q_eps=%d fwd=%d)"
                            % (opps, lo, hi, steps_s1, steps_q, q_eps, fwd))
        if int(se.get("recorded_steps", -1)) > opps:
            mech.append("recorded %s > opportunities %s" % (se.get("recorded_steps"), opps))
        # last-write consistency vs jsonl (summary refreshes every H boundary)
        if rows and abs(int(se.get("h_cycles", -1)) - len(rows)) > 1:
            mech.append("sp_run_end h_cycles %s vs jsonl lines %d diverge >1"
                        % (se.get("h_cycles"), len(rows)))
    except Exception as e:
        integ.append("sp_run_end.json missing/unparsable: %r" % (e,))

    # ---------- verdict ----------
    if mech:
        verdict, code = "MECH_RETIRE", 2
    elif res:
        verdict, code = "RESULT_RETIRE", 1
    elif integ:
        verdict, code = "INTEGRITY_FAIL", 3
    else:
        verdict, code = "VALID", 0
    det["verdict"] = verdict
    det["mech"] = mech; det["result"] = res; det["integrity"] = integ
    try:
        json.dump(det, open(os.path.join(art, "verdict.json"), "w"),
                  indent=1, ensure_ascii=False)
    except Exception:
        pass
    print("VERDICT:%s t=%s" % (verdict, ttime if ttime is not None else "-"))
    sys.exit(code)


if __name__ == "__main__":
    main()
