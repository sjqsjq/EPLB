#!/usr/bin/env python3
"""integrity_check_fp.py — final_plan_20261019 per-run verdict checker.

Verdict classes (identical semantics to the sealed rev1 driver):
  VALID           bench complete (ok==N, errors==0) AND arm mechanism markers
                  all present/correct.
  RESULT_RETIRE   bench ran but ok<N or errors>0 (retired, never retried,
                  consumes budget). NEVER triggered by endpoint VALUES.
  MECH_RETIRE     arm mechanism marker missing/contradicted (mechanism class
                  -> driver HALTs per governance).
  INTEGRITY_FAIL  bench.json/sp_run_end.json missing or unparsable, or
                  artifacts internally inconsistent.

Usage: integrity_check_fp.py --artifact-dir ART --arm ARM --label L
       --expected-requests N [--eplb-period P] [--expect-shw S,H,W]
Prints: VERDICT:<class> t=<total_time_s>
"""
import argparse, json, os, re, sys

def fail(cls, why, t="-"):
    print(f"CHECK: {cls}: {why}")
    print(f"VERDICT:{cls} t={t}")
    sys.exit(0 if cls == "VALID" else 1)

ap = argparse.ArgumentParser()
ap.add_argument("--artifact-dir", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--expected-requests", type=int, required=True)
ap.add_argument("--eplb-period", type=int, default=0)
ap.add_argument("--expect-shw", default="")
a = ap.parse_args()
ART = a.artifact_dir

# --- AMENDMENT 2026-10-19 (INCIDENT_E4F_hang.md; plan sec.9.9 / FREEZE sec.7) ---
# Server crash mid-bench (fatal markers in server.log) + bench.json missing/unparsable
# = METHOD-CRASH/timeout outcome => RESULT_RETIRE (completion-rate result; never
# retried, never deleted), NOT a measurement-pipeline INTEGRITY_FAIL. Evidence-gated:
# requires explicit fatal crash markers. Verdicts issued before this amendment are
# NOT re-scored. Endpoint values never influence this classification.
CRASH_MARKERS = ("Watchdog caught collective operation timeout",
                 "Fatal Python error: Aborted",
                 "CUDA out of memory",
                 "terminate called after throwing an instance of 'c10::DistBackendError'",
                 "Scheduler hit an exception")
def crash_evidence():
    try:
        sl = open(os.path.join(ART, "server.log"), errors="replace").read()
    except Exception:
        return []
    return [m for m in CRASH_MARKERS if m in sl]

bj = os.path.join(ART, "bench.json")
if not os.path.exists(bj):
    ev = crash_evidence()
    if ev:
        fail("RESULT_RETIRE", f"server crashed mid-bench, {len(ev)} fatal marker(s) e.g. '{ev[0]}'; bench.json missing; method-crash class per plan sec.9.9")
    fail("INTEGRITY_FAIL", "bench.json missing")
try:
    b = json.load(open(bj))
except Exception as e:
    ev = crash_evidence()
    if ev:
        fail("RESULT_RETIRE", f"server crashed mid-bench, {len(ev)} fatal marker(s); bench.json unparsable: {e}; method-crash class per plan sec.9.9")
    fail("INTEGRITY_FAIL", f"bench.json unparsable: {e}")
t = b.get("total_time_s", "-")
N = a.expected_requests
if b.get("label") != a.label:
    fail("INTEGRITY_FAIL", f"bench.json label {b.get('label')} != {a.label}", t)
if b.get("num_requests") != N:
    fail("INTEGRITY_FAIL", f"bench num_requests {b.get('num_requests')} != {N}", t)
if b.get("ok", 0) < N or b.get("errors", 0) > 0:
    fail("RESULT_RETIRE", f"ok={b.get('ok')} errors={b.get('errors')} (N={N})", t)
if b.get("client_version", "").startswith("bench_client_v2") is False:
    fail("INTEGRITY_FAIL", f"wrong client {b.get('client_version')}", t)
perreq = os.path.join(ART, f"{a.label}_perreq.jsonl")
if not os.path.exists(perreq):
    fail("INTEGRITY_FAIL", "per-request jsonl missing (v2 client required)", t)
nrows = sum(1 for _ in open(perreq))
if nrows != N:
    fail("INTEGRITY_FAIL", f"perreq rows {nrows} != {N}", t)

slog = os.path.join(ART, "server.log")
logs = ""
if os.path.exists(slog):
    logs = open(slog, errors="replace").read()

arm = a.arm
if arm == "I":
    if "[PB-OEPLB" in logs:
        fail("MECH_RETIRE", "I arm: PB-OEPLB log lines present (must be inert)", t)
    if "[EPLBManager]" in logs:
        fail("MECH_RETIRE", "I arm: EPLBManager active (must be inert)", t)
elif arm == "E":
    if a.eplb_period <= 0:
        fail("INTEGRITY_FAIL", "E arm requires --eplb-period", t)
    if f"[EPLBManager] system started, will rebalance per {a.eplb_period} iterations" not in logs:
        fail("MECH_RETIRE", f"E arm: EPLBManager start banner (period {a.eplb_period}) missing", t)
    if "[EPLBManager] rebalance start" not in logs:
        fail("MECH_RETIRE", "E arm: no rebalance actually triggered", t)
else:
    # PB-family arms: P, F, A1..A4
    if "[PB-OEPLB-ARMFLAGS]" not in logs:
        fail("MECH_RETIRE", "ARMFLAGS banner missing (patched build not running?)", t)
    se_p = os.path.join(ART, "sp_run_end.json")
    if not os.path.exists(se_p):
        fail("INTEGRITY_FAIL", "sp_run_end.json missing", t)
    try:
        se = json.load(open(se_p))
    except Exception as e:
        fail("INTEGRITY_FAIL", f"sp_run_end.json unparsable: {e}", t)
    if arm == "P":
        if se.get("mode") != "adaptive":
            fail("MECH_RETIRE", f"P arm: mode={se.get('mode')}", t)
        if not (se.get("window_count", 0) > 0):
            fail("MECH_RETIRE", "P arm: zero windows processed", t)
    elif arm == "F":
        if se.get("mode") != "fixed":
            fail("MECH_RETIRE", f"F arm: mode={se.get('mode')}", t)
        if a.expect_shw:
            S, H, W = [int(x) for x in a.expect_shw.split(",")]
            got = (se.get("S_final"), se.get("H_final"), se.get("W_final"))
            if got != (S, H, W):
                fail("MECH_RETIRE", f"F arm: S,H,W {got} != expected {(S,H,W)}", t)
    elif arm == "A1":
        if int(se.get("decode_record_batches", 0)) <= 0:
            fail("MECH_RETIRE", "A1: decode_record_batches == 0 (all-phase recording inactive)", t)
    elif arm == "A2":
        if "pair_rule=maxdelta" not in logs:
            fail("MECH_RETIRE", "A2: pair_rule=maxdelta not echoed", t)
    elif arm == "A3":
        if "force_s=1" not in logs:
            fail("MECH_RETIRE", "A3: force_s=1 not echoed", t)
        if se.get("mode") != "adaptive":
            fail("MECH_RETIRE", f"A3: mode={se.get('mode')} (must stay adaptive)", t)
    elif arm == "A4":
        if int(se.get("nocommit_planned_ops_total", 0)) <= 0:
            fail("MECH_RETIRE", "A4: nocommit_planned_ops_total == 0 (no plans built?)", t)
        if int(se.get("swaps_started_total", -1)) != 0:
            fail("MECH_RETIRE", f"A4: swaps_started_total={se.get('swaps_started_total')} != 0 (commit suppression failed)", t)
    else:
        fail("INTEGRITY_FAIL", f"unknown arm {arm}", t)

print(f"CHECK: VALID arm={arm} ok={b.get('ok')}/{N} t={t}")
print(f"VERDICT:VALID t={t}")
sys.exit(0)
