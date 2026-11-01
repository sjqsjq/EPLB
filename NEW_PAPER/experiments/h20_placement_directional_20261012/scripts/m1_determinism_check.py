#!/usr/bin/env python3
"""M1->M2 stage gate (PRE_REG_calibration_m1m2m3.md §C): MECHANISM ONLY.

Reads the four M1 (mode=off) artifact dirs and the sealed A-arm r1 references.
Gates (never endpoint values):
  G1 4/4 bench verdicts VALID (verdict.json).
  G2 decision sequence EXACTLY equals sealed r1 same condition:
     per-window (window_id, steps_actual, W_before, W_after, event, committed),
     Sigma steps, swap-initiation count (committed & W_after != W_before).
  G3 z descriptive (NOT bit-gated -- sealed rounds themselves jitter):
     max|Delta z| vs r1 + per-window margin to the nearest decision threshold;
     FAIL only if a z crosses a threshold CLASS relative to sealed r1.
  G4 ledger <= eps_log: median c_window_ms within +1.0 ms of sealed r1 median;
     final-PROF-block avg_record_us 8-rank median <= sealed <cond>_A__r1 final-block
     median * 1.10 (reference derived at runtime from sealed artifacts; 2026-10-18
     post-freeze pre-gate correction -- frozen constant 81.61 was a transcription
     error, see PRE_REG section F #5; tolerance unchanged);
     prof_probe_cum_ms == 0.0 (mode off must leave the probe ledger untouched).
  G5 protocol health: lockstep_ok true, h_min==h_max on commits, desync false,
     no fatal_collective / *_freeze events, probe_mode=="off" on every line.
  G6 bench.json carries the d1 monotonic alignment fields.
Exit 0 = PASS (safe to proceed to M2); 1 = FAIL (HALT + report); 2 = bad usage.
Writes $CAL/m1_gate.json.
"""
import json
import os
import statistics
import sys

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
FE = os.path.join(EXP, "results/adw_v2p1/formal_eval")
CAL = os.path.join(EXP, "results/adw_v2p1/calibration_m1m2m3")
Z_LO, Z_HI, Z_URG = 22.643858727592757, 314.14010272527486, 942.4203081758246
# 2026-10-18 POST-FREEZE PRE-GATE CORRECTION (PRE_REG section F #5): the previously
# frozen constant RECORD_US_REF=81.61 was a transcription error -- 81.61us comes from
# the sealed ADDENDUM R2 identity check (81.61 x 89770 calls = 7326ms), NOT from any
# sealed final-PROF avg_record_us. Sealed final-block 8-rank medians: W1_A r1/r2/r3 =
# 115.54/126.77/126.90; W2_A r1/r2/r3 = 114.67/113.76/123.97. The eps_log criterion
# is UNCHANGED (<= +10% vs sealed reference); the reference is now derived AT RUNTIME
# from sealed <cond>_A__r1 server.log final PROF block (8-rank median), which removes
# hard-coded transcription risk. Tolerance not relaxed; anchor fixed by sealed
# (pre-calibration) artifacts only.
RECORD_US_TOL = 1.10
CWIN_DELTA_CAP_MS = 1.0        # boundary added median cap (PRE_REG eps_log)

M1_RUNS = [("cal_m1_w1_a", "W1"), ("cal_m1_w1_b", "W1"),
           ("cal_m1_w2_a", "W2"), ("cal_m1_w2_b", "W2")]


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


def seq_tuple(rows):
    return [(r["window_id"], r["steps_actual"], r["W_before"], r["W_after"],
             r["event"], r["committed"]) for r in rows]


def swap_count(rows):
    return sum(1 for r in rows
               if r["committed"] and r["W_after"] != r["W_before"])


def z_class(z):
    if z is None:
        return "none"
    if z >= Z_URG:
        return "urgent"
    if z >= Z_HI:
        return "alert"
    if z >= Z_LO:
        return "midband"
    return "stable"


def final_prof_record_us(server_log):
    """8-rank stats over the LAST [PB-OEPLB-PROF] window block.

    Returns dict(median,min,max,n,window,vals) or None if unavailable.
    Symmetric: applied identically to sealed reference and calibration logs
    (2026-10-18 correction, PRE_REG section F #5).
    """
    vals = []
    lastwin = None
    try:
        with open(server_log, errors="replace") as f:
            for line in f:
                if "[PB-OEPLB-PROF]" in line and "avg_record_us=" in line:
                    try:
                        w = int(line.split("window#")[1].split()[0])
                        v = float(line.split("avg_record_us=")[1].split()[0])
                    except Exception:
                        continue
                    if lastwin is None or w == lastwin:
                        vals.append(v)
                        lastwin = w
                    else:
                        vals = [v]
                        lastwin = w
    except FileNotFoundError:
        return None
    if not vals:
        return None
    return {"median": statistics.median(vals), "min": min(vals),
            "max": max(vals), "n": len(vals), "window": lastwin,
            "vals": [round(v, 2) for v in vals]}


def main():
    report = {"gate": "M1", "runs": {}, "sealed_ref": {}, "pass": None}
    ok = True
    for cond in ("W1", "W2"):
        ref = load_jsonl(os.path.join(FE, f"{cond}_A__r1", "v2p1_rank0.jsonl"))
        ref_med = statistics.median([r["c_window_ms"] for r in ref])
        report["sealed_ref"][cond] = {
            "windows": len(ref), "sum_steps": sum(r["steps_actual"] for r in ref),
            "swaps": swap_count(ref), "median_c_window_ms": round(ref_med, 4),
            "record_us": final_prof_record_us(os.path.join(FE, f"{cond}_A__r1", "server.log")),
        }
    for label, cond in M1_RUNS:
        art = os.path.join(CAL, label)
        r = {"condition": cond, "checks": {}}
        def chk(name, passed, detail=""):
            r["checks"][name] = {"pass": bool(passed), "detail": detail}
            return passed
        # G1 verdict
        try:
            vj = json.load(open(os.path.join(art, "verdict.json")))
            verdict = vj.get("verdict", "?")
        except Exception as e:
            verdict = f"MISSING({e})"
        ok &= chk("G1_valid", verdict == "VALID", verdict)
        # load
        try:
            rows = load_jsonl(os.path.join(art, "v2p1_rank0.jsonl"))
        except FileNotFoundError:
            ok &= chk("G0_jsonl_present", False, "v2p1_rank0.jsonl missing")
            report["runs"][label] = r
            continue
        ok &= chk("G0_jsonl_present", len(rows) > 0, f"{len(rows)} windows")
        ref = load_jsonl(os.path.join(FE, f"{cond}_A__r1", "v2p1_rank0.jsonl"))
        # G2 decision sequence
        s_cal, s_ref = seq_tuple(rows), seq_tuple(ref)
        ok &= chk("G2_sequence_equal", s_cal == s_ref,
                  "" if s_cal == s_ref else
                  f"first mismatch: {next((a,b) for a,b in zip(s_cal,s_ref) if a!=b) if len(s_cal)==len(s_ref) else (len(s_cal), len(s_ref))}")
        ok &= chk("G2_sum_steps", sum(x[1] for x in s_cal) == sum(x[1] for x in s_ref),
                  f"{sum(x[1] for x in s_cal)} vs {sum(x[1] for x in s_ref)}")
        ok &= chk("G2_swap_count", swap_count(rows) == swap_count(ref),
                  f"{swap_count(rows)} vs {swap_count(ref)}")
        # G3 z descriptive + class-crossing gate
        if len(rows) == len(ref):
            dzs, margins, crossed = [], [], []
            for a, b in zip(rows, ref):
                za, zb = a.get("z"), b.get("z")
                if za is not None and zb is not None:
                    dzs.append(abs(za - zb))
                    m = min(abs(za - t) for t in (Z_LO, Z_HI, Z_URG))
                    margins.append(m)
                    if z_class(za) != z_class(zb):
                        crossed.append((a["window_id"], za, zb))
                elif (za is None) != (zb is None):
                    crossed.append((a["window_id"], za, zb))
            r["z_max_abs_delta"] = max(dzs) if dzs else None
            r["z_min_threshold_margin"] = min(margins) if margins else None
            ok &= chk("G3_z_class_no_crossing", not crossed, str(crossed[:5]))
        else:
            ok &= chk("G3_z_class_no_crossing", False, "length mismatch; G2 already failed")
        # G4 ledger
        med = statistics.median([r_["c_window_ms"] for r_ in rows])
        ref_med = statistics.median([r_["c_window_ms"] for r_ in ref])
        delta = med - ref_med
        r["median_c_window_ms"] = round(med, 4)
        r["c_window_delta_ms"] = round(delta, 4)
        ok &= chk("G4_cwin_delta", delta <= CWIN_DELTA_CAP_MS, f"{delta:+.4f} ms vs sealed r1")
        rec = final_prof_record_us(os.path.join(art, "server.log"))
        ref_rec = report["sealed_ref"][cond]["record_us"]
        r["record_us_final_block"] = rec
        if rec is None or ref_rec is None:
            r["record_us_cap"] = None
            ok &= chk("G4_record_us", False, f"missing PROF data cal={rec} sealed={ref_rec}")
        else:
            cap = ref_rec["median"] * RECORD_US_TOL
            r["record_us_cap"] = round(cap, 2)
            ok &= chk("G4_record_us", rec["median"] <= cap,
                      f"cal median {rec['median']:.2f} (range {rec['min']:.2f}-{rec['max']:.2f}, n={rec['n']}) "
                      f"<= sealed r1 median {ref_rec['median']:.2f} x {RECORD_US_TOL} = {cap:.2f}")
        probe_cum = [r_.get("prof_probe_cum_ms", None) for r_ in rows]
        pmax = max([p for p in probe_cum if p is not None], default=None)
        ok &= chk("G4_probe_ledger_zero", pmax == 0.0 or pmax is None, str(pmax))
        # G5 protocol health
        bad_lock = [r_["window_id"] for r_ in rows if r_.get("lockstep_ok") is not True]
        bad_h = [r_["window_id"] for r_ in rows
                 if r_["committed"] and r_.get("h_min") != r_.get("h_max")]
        bad_des = [r_["window_id"] for r_ in rows if r_.get("desync")]
        bad_ev = [r_["window_id"] for r_ in rows
                  if r_.get("event") in ("fatal_collective", "w_desync_freeze", "confirm_fail_freeze")]
        bad_mode = [r_["window_id"] for r_ in rows if r_.get("probe_mode") not in ("off", "none")]
        bad_pstate = [r_["window_id"] for r_ in rows if r_.get("probe_state") not in ("", None)]
        ok &= chk("G5_lockstep", not bad_lock, str(bad_lock[:5]))
        ok &= chk("G5_hash_unanimous", not bad_h, str(bad_h[:5]))
        ok &= chk("G5_no_desync", not bad_des, str(bad_des[:5]))
        ok &= chk("G5_no_fatal_freeze", not bad_ev, str(bad_ev[:5]))
        ok &= chk("G5_probe_off", not bad_mode and not bad_pstate,
                  f"mode={bad_mode[:3]} state={bad_pstate[:3]}")
        # G6 bench monotonic fields
        try:
            bj = json.load(open(os.path.join(art, "bench.json")))
            has_mono = all(k in bj for k in ("mono_start_ns", "mono_end_ns",
                                             "wall_start_s", "wall_end_s"))
            r["total_time_s"] = bj.get("total_time_s")
        except Exception as e:
            has_mono = False
            r["total_time_s"] = None
        ok &= chk("G6_bench_mono_fields", has_mono, "")
        report["runs"][label] = r
    report["pass"] = bool(ok)
    out = os.path.join(CAL, "m1_gate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("M1_GATE:", "PASS" if ok else "FAIL", "->", out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
