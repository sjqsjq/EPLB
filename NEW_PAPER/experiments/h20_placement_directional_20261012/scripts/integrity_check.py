#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrity_check.py — per-start verdict gate for the v2.1 formal evaluation
(protocol PROTOCOL_formal_eval_v2p1_DRAFT.md §8; locked before first start).

Usage:
  python3 integrity_check.py --artifact-dir DIR --arm {A,F,L,M} \
      --condition {W1,W2} --label LABEL --expected-requests N

Verdicts (printed as "VERDICT:<X> t=<total_time_s>"):
  VALID           exit 0 — bench valid + no mechanism failure + integrity gates pass
  MECH_RETIRE     exit 2 — fatal_collective / w_desync_freeze / confirm_fail_freeze
                           in rank0 jsonl, or desync==true, or (A) committed W off
                           grid {8,16,32,64}, or (M) any committed W_after != 16
  RESULT_RETIRE   exit 1 — bench.json missing/unparsable, ok < N, errors > 0,
                           or bench metadata mismatch (label/dataset/concurrency)
  INTEGRITY_FAIL  exit 3 — A/M: ADW21 ACTIVE banner count < 8, or knob strings not
                           unique, or knobs != locked authority, or rank0 jsonl
                           lines < 90% of expected;  F/L: any "PB-OEPLB-ADW"
                           banner present (wrong controller active)

Priority when several fail: MECH_RETIRE > RESULT_RETIRE > INTEGRITY_FAIL.
Never compares arms. Never prints cross-start statistics. Writes verdict.json
into the artifact dir (full check detail for the audit trail).

Expected-line basis (locked): measured W0 dispatch = 8037 rows -> 1056 engine
steps (r3 behavior + logging runs, sum steps_actual). est_steps(cond) =
round(1056 * nrows / 8037): W1 (4000) -> 526, W2 (8000) -> 1051.
Expected rank0 jsonl lines ~= est_steps / W_ref; gate = floor(0.9 * that) with
W_ref = 16 (M, fixed) or 64 (A, loosest grid value; A window count is
dynamic by design).
"""
import argparse, json, os, re, sys

FATAL_EVENTS = {"fatal_collective", "w_desync_freeze", "confirm_fail_freeze"}
GRID = {8, 16, 32, 64}
STEPS_W0_8037 = 1056
NROWS = {"W1": 4000, "W2": 8000}
DSNAME = {"W1": "w1_stable_code_4000.jsonl", "W2": "w2_switch_code_chinese_8000.jsonl"}
# authority knob fingerprints (LOCK_record_v2p1 §5 / FREEZE_record_v2p1)
A_KNOB_MARKS = ["'kappa': 3875.1768798828125", "'z_lo': 22.643858727592757",
                "'z_hi': 314.14010272527486", "'z_urgent': 942.4203081758246",
                "'n_min': 12320768", "'theta_h_low': 0.010095552687949321",
                "'theta_h_act': 0.011395643112507271"]
M_KNOB_MARKS = ["'kappa': 1.0", "'z_lo': 1000000000000.0", "'z_hi': 2000000000000.0",
                "'z_urgent': 3000000000000.0", "'n_min': 0",
                "'theta_h_low': 1e-12", "'theta_h_act': 2.0"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--arm", required=True, choices=["A", "F", "L", "M"])
    ap.add_argument("--condition", required=True, choices=["W1", "W2"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--expected-requests", required=True, type=int)
    a = ap.parse_args()
    art = a.artifact_dir
    det = {"arm": a.arm, "condition": a.condition, "label": a.label, "checks": {}}
    mech, res, integ = [], [], []

    # ---------- bench.json (result validity) ----------
    ttime = None
    bpath = os.path.join(art, "bench.json")
    try:
        b = json.load(open(bpath))
        det["checks"]["bench"] = {k: b.get(k) for k in
            ("label", "dataset", "num_requests", "concurrency", "ok", "errors", "total_time_s")}
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
    spath = os.path.join(art, "server.log")
    slog = ""
    try:
        slog = open(spath, encoding="utf-8", errors="replace").read()
    except Exception as e:
        integ.append("server.log missing: %r" % (e,))
    if a.arm in ("A", "M"):
        nban = slog.count("[PB-OEPLB-ADW21] limited-revision controller v2.1 ACTIVE")
        det["checks"]["banner_count"] = nban
        if nban < 8:
            integ.append("ADW21 ACTIVE banner count %d < 8 ranks" % nban)
        ks = re.findall(r"knobs=(\{.*?\}) \(theta_", slog)
        uniq = sorted(set(ks))
        det["checks"]["knob_strings_unique"] = len(uniq)
        if len(uniq) != 1:
            integ.append("knob strings not unique across ranks: %d variants" % len(uniq))
        elif a.arm == "A":
            miss = [m for m in A_KNOB_MARKS if m not in uniq[0]]
            det["checks"]["knob_authority"] = "A-frozen" if not miss else "MISMATCH %s" % miss
            if miss:
                integ.append("A knob authority mismatch: %s" % miss)
        else:
            miss = [m for m in M_KNOB_MARKS if m not in uniq[0]]
            det["checks"]["knob_authority"] = "M-inert" if not miss else "MISMATCH %s" % miss
            if miss:
                integ.append("M knob authority mismatch: %s" % miss)
    else:  # F / L: no ADW machinery may be active
        nadw = slog.count("PB-OEPLB-ADW")
        det["checks"]["adw_banner_count_must_be_0"] = nadw
        if nadw != 0:
            integ.append("F/L start shows %d PB-OEPLB-ADW banner lines (wrong controller)" % nadw)

    # ---------- rank0 jsonl (A/M only) ----------
    if a.arm in ("A", "M"):
        jpath = os.path.join(art, "v2p1_rank0.jsonl")
        rows = []
        try:
            for ln in open(jpath, encoding="utf-8"):
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        except Exception as e:
            integ.append("rank0 jsonl missing/unparsable: %r" % (e,))
        det["checks"]["jsonl_lines"] = len(rows)
        est_steps = round(STEPS_W0_8037 * NROWS[a.condition] / 8037.0)
        w_ref = 16 if a.arm == "M" else 64
        need = int(0.9 * (est_steps / w_ref))
        det["checks"]["jsonl_min_lines"] = need
        if rows and len(rows) < need:
            integ.append("jsonl lines %d < 90%% expected (%d)" % (len(rows), need))
        for r in rows:
            ev = r.get("event")
            if ev in FATAL_EVENTS:
                mech.append("fatal mechanism event %s at window %s" % (ev, r.get("window_id")))
            if r.get("desync") is True:
                mech.append("desync==true at window %s" % r.get("window_id"))
            if r.get("committed"):
                wa = r.get("W_after")
                if a.arm == "M" and wa != 16:
                    mech.append("M committed W_after=%r != 16 at window %s" % (wa, r.get("window_id")))
                if a.arm == "A" and wa not in GRID:
                    mech.append("A committed W_after=%r off grid at window %s" % (wa, r.get("window_id")))
        mech = mech[:10]  # cap detail, verdict unchanged

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
        json.dump(det, open(os.path.join(art, "verdict.json"), "w"), indent=1, ensure_ascii=False)
    except Exception:
        pass
    print("VERDICT:%s t=%s" % (verdict, ttime if ttime is not None else "-"))
    sys.exit(code)

if __name__ == "__main__":
    main()
