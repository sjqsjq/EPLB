#!/usr/bin/env python3
"""ADW v2.1 r3 calibration: window classification + one-shot blind fit + diagnostics + PT2(A/B).

Rules source (FROZEN, pre-registered): results/adw_v2p1/calibration_plan_v2p1.md (r3).
  §2   classification: boundaries #2009/#4018/#6027, ±250-request guard,
       window<->request alignment via cumulative steps_actual (x8192 tok/step)
       fraction of 8037 rows (same v2 method).
  §2.1 qualified window := committed==true and protocol_rev=="r3" and
       event not in EXCLUDED and z != null.
  §2.2 pool gates: stationary >=150, Z_c >=20; >30% below expectation -> STOP.
  §3   fit: kappa=median(Z_s); z_lo=p90(Z_s/kappa); z_hi=p99(Z_s/kappa);
       z_urgent=max(3*z_hi, p50(Z_c/kappa)); theta_r_low=p30(H_rate_s);
       theta_r_act=p60(H_rate_s).  (formula verbatim from calibration_plan_v2.md §3)
  §4   diagnostics ONLY (never knobs): G1-D, G2, G3, G5, D1, D2.
  PT2  plan §4 G1 acceptance, scoped per user instruction 2026-10-17:
       paired same-state scaling -> h_rate identity + headroom-gate boolean
       invariance. NO claim that replicated counts are new independent samples;
       NO requirement that recomputed z or UNCERTAIN judgments are invariant
       (n_min is an absolute-count gate; n_min side is reported separately).

Quantile method (documented per user step 4): numpy.percentile with
method="linear" (linear interpolation, numpy default); pool = concatenation
of per-window samples with EQUAL weight per window; effective n = pool length.

Outputs: fit_results_v2p1.json, window_inventory_v2p1.json (+ sha256 of the
canonical window list, for the FREEZE record), printed report.
Exits NONZERO on any pre-registered gate failure (STOP semantics: no knob is
relaxed, no run is appended, nothing is refit).
"""
import json, os, sys, math, hashlib, collections
from fractions import Fraction

import numpy as np

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTROOT = os.environ.get("V2P1_FIT_OUTROOT", EXP + "/results/adw_v2p1/r3_logging")
RUNS = [8, 16, 32, 64]                      # pre-registered order
N_ROWS = 8037
BOUNDS = [2009, 4018, 6027]                 # plan §2 (0-based row indices)
GUARD = 250                                 # ± requests
N_LAYERS = 94
EXCLUDED_EVENTS = {"init", "uncertain_carry", "uncertain_hold",
                   "forced_shrink_uncertain", "confirm_hold",
                   "confirm_fail_freeze", "w_desync_freeze", "fatal_collective"}
EXPECTED_STATIONARY = {8: (104, 107), 16: (51, 53), 32: (25, 26), 64: (12, 12)}
GATE_STATIONARY_POOL = 150
GATE_ZC_POOL = 20
SHORTFALL_FRAC = 0.30                       # rule 5: >30% below expectation -> STOP

def latest(path_glob_dir, prefix, suffix):
    cands = sorted(f for f in os.listdir(path_glob_dir)
                   if f.startswith(prefix) and f.endswith(suffix))
    return os.path.join(path_glob_dir, cands[-1]) if cands else None

def prof_forwards(server_log):
    """Max record calls across ranks / 94 layers = server-side forwards."""
    best = 0
    try:
        for line in open(server_log, errors="replace"):
            if "[PB-OEPLB-PROF]" in line and "calls=" in line:
                try:
                    c = int(line.split("calls=")[1].split()[0])
                    best = max(best, c)
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        return None
    return best / N_LAYERS if best else None

def load_run(W):
    d = os.path.join(OUTROOT, "W%d" % W)
    rows = [json.loads(l) for l in open(os.path.join(d, "v2p1_rank0.jsonl")) if l.strip()]
    bench_p = latest(d, "bench_a", ".json")
    bench = json.load(open(bench_p)) if bench_p else None
    slog = latest(d, "server_a", ".log")
    pf = prof_forwards(slog) if slog else None
    return d, rows, bench, pf

def classify(rows):
    """§2: window request-span via cumulative steps fraction; guard ±250."""
    total_steps = sum(r["steps_actual"] for r in rows)
    cum = 0
    spans = []
    for r in rows:
        r0 = cum / total_steps * N_ROWS
        cum += r["steps_actual"]
        r1 = cum / total_steps * N_ROWS
        spans.append((r0, r1))
    zones = []
    for (r0, r1) in spans:
        in_guard = any(not (r1 < b - GUARD or r0 > b + GUARD) for b in BOUNDS)
        zones.append("change" if in_guard else "stationary")
    return zones, total_steps, spans

def qualified(r):
    return (r.get("committed") is True and r.get("protocol_rev") == "r3"
            and r.get("event") not in EXCLUDED_EVENTS and r.get("z") is not None)

def excl_reason(r):
    if r.get("event") == "fatal_collective":
        return "fatal_collective"
    if r.get("committed") is not True:
        return "not_committed(event=%s)" % r.get("event")
    if r.get("protocol_rev") != "r3":
        return "protocol_rev=%s" % r.get("protocol_rev")
    if r.get("event") in EXCLUDED_EVENTS:
        return "excluded_event:%s" % r.get("event")
    if r.get("z") is None:
        return "z_null(event=%s)" % r.get("event")
    return None

def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q, method="linear"))

def main():
    stop_reasons = []
    report = {"runs": {}}
    Zs, Hs, Ns, Zc = [], [], [], []
    Zs_by_W, Hs_by_W = {}, {}
    inventory = {}
    all_rows_by_W = {}

    for W in RUNS:
        d, rows, bench, pf = load_run(W)
        all_rows_by_W[W] = rows
        zones, total_steps, spans = classify(rows)
        fatal = [r for r in rows if r.get("event") == "fatal_collective"]
        w_changed = [r for r in rows if r.get("W_before") != W or (r.get("W_after") not in (None, W))
                     and r.get("event") != "init"]
        # inert-knob integrity: W must stay fixed by construction
        w_not_fixed = [r["window_id"] for r in rows if r.get("W_after") != W]
        q_rows = [r for r in rows if qualified(r)]
        reasons = collections.Counter(excl_reason(r) for r in rows if not qualified(r))
        qs = [r for r in q_rows if zones[rows.index(r)] == "stationary"]
        qc = [r for r in q_rows if zones[rows.index(r)] == "change"]
        # z reconstruction consistency (§3.1): logged z at kappa_run=1.0
        zdiff = 0.0
        for r in q_rows:
            n_layer = max(r["n_fresh_assignments"] / float(N_LAYERS), 1.0)
            zr = r["s_change_logical"] / (1.0 / n_layer)
            zdiff = max(zdiff, abs(zr - r["z"]))
        rec = {
            "dir": d, "recorded_windows": len(rows),
            "window_id_range": [rows[0]["window_id"], rows[-1]["window_id"]] if rows else None,
            "recorded_steps_sum": total_steps,
            "server_forwards_prof": pf,
            "bench": {k: bench.get(k) for k in ("ok", "errors", "total_time_s", "completed")} if bench else None,
            "fatal_collective_events": len(fatal),
            "W_not_fixed_windows": w_not_fixed,
            "qualified": len(q_rows),
            "qualified_stationary": len(qs), "qualified_change": len(qc),
            "exclusion_reasons": dict(reasons),
            "z_reconstruction_max_absdiff": zdiff,
            "h_rate_log_consistency_max_absdiff": max(
                (abs(r["h_rate"] - r["h_w_assignments"] / float(r["n_fresh_assignments"]))
                 for r in q_rows if r.get("n_fresh_assignments")), default=0.0),
        }
        report["runs"][W] = rec
        inventory[W] = [{"window_id": r["window_id"], "zone": zones[rows.index(r)],
                         "qualified": qualified(r),
                         "req_span": [round(spans[rows.index(r)][0], 1), round(spans[rows.index(r)][1], 1)],
                         "z": r.get("z"), "h_rate": r.get("h_rate"),
                         "h_w": r.get("h_w_assignments"), "n_fresh": r.get("n_fresh_assignments"),
                         "event": r.get("event"), "p2l_version": r.get("p2l_version"),
                         "steps_actual": r.get("steps_actual")} for r in rows]
        if fatal:
            stop_reasons.append("W%d: fatal_collective present in log (%d)" % (W, len(fatal)))
        if w_not_fixed:
            stop_reasons.append("W%d: W not fixed by construction in windows %s (inert knobs breached)"
                                % (W, w_not_fixed[:10]))
        if bench and bench.get("errors"):
            stop_reasons.append("W%d: bench reported %s errors" % (W, bench.get("errors")))
        # §2.2 rule 5: shortfall vs expectation
        lo, hi = EXPECTED_STATIONARY[W]
        if len(qs) < (1 - SHORTFALL_FRAC) * lo:
            stop_reasons.append("W%d: qualified stationary %d is >30%% below expectation [%d,%d]"
                                % (W, len(qs), lo, hi))
        Zs += [r["z"] for r in qs]; Zc += [r["z"] for r in qc]
        Hs += [r["h_rate"] for r in qs]; Ns += [r["n_fresh_assignments"] for r in qs]
        Zs_by_W[W] = [r["z"] for r in qs]; Hs_by_W[W] = [r["h_rate"] for r in qs]

    report["pools"] = {"Z_s": len(Zs), "Z_c": len(Zc), "H_rate_s": len(Hs), "N_s": len(Ns)}
    if len(Zs) < GATE_STATIONARY_POOL:
        stop_reasons.append("stationary pool %d < gate %d" % (len(Zs), GATE_STATIONARY_POOL))
    if len(Zc) < GATE_ZC_POOL:
        stop_reasons.append("Z_c pool %d < gate %d" % (len(Zc), GATE_ZC_POOL))

    if stop_reasons:
        report["STOP"] = stop_reasons
        json.dump(report, open(os.path.join(OUTROOT, "fit_STOP_report.json"), "w"), indent=1)
        print("== STOP (pre-registered gates; no fit performed) ==")
        for s in stop_reasons:
            print("  -", s)
        print("details -> fit_STOP_report.json")
        return 2

    # ---- §3 one-shot blind fit (equal per-window weight, linear-interp percentiles) ----
    Zs_a = np.asarray(Zs, dtype=np.float64)
    kappa = float(np.median(Zs_a))
    Zsr = Zs_a / kappa
    z_lo = pct(Zsr, 90); z_hi = pct(Zsr, 99)
    Zc_a = np.asarray(Zc, dtype=np.float64) / kappa
    z_urgent = max(3.0 * z_hi, pct(Zc_a, 50))
    Hs_a = np.asarray(Hs, dtype=np.float64)
    theta_r_low = pct(Hs_a, 30); theta_r_act = pct(Hs_a, 60)
    if not (0.0 < theta_r_low < theta_r_act <= 2.0):
        print("STOP: fitted theta violates 0<theta_r_low<theta_r_act<=2:", theta_r_low, theta_r_act)
        return 2

    # ---- §4 diagnostics (report-only) ----
    def med(a): return float(np.median(a)) if len(a) else None
    g1d = {"h_rate_median_by_W": {W: med(Hs_by_W[W]) for W in RUNS},
           "h_rate_n_by_W": {W: len(Hs_by_W[W]) for W in RUNS}}
    ms = [g1d["h_rate_median_by_W"][W] for W in RUNS if g1d["h_rate_median_by_W"][W]]
    g1d["h_rate_max_min_ratio"] = (max(ms) / min(ms)) if ms and min(ms) > 0 else None
    hw_med = {W: med([r["h_w_assignments"] for r in all_rows_by_W[W]
                      if qualified(r)]) for W in RUNS}
    g1d["raw_h_w_median_by_W"] = hw_med
    hwm = [hw_med[W] for W in RUNS if hw_med[W]]
    g1d["raw_h_w_max_min_ratio"] = (max(hwm) / min(hwm)) if hwm and min(hwm) > 0 else None
    band = g1d["h_rate_max_min_ratio"]
    g1d["band"] = ("<=1.5 compatible" if band is not None and band <= 1.5 else
                   "1.5-3 report+offline segmentation check" if band is not None and band <= 3 else
                   ">3 STOP-before-freeze (F2 validity threat)")
    g2 = {"p50_Zc_rescaled": pct(Zc_a, 50), "z_hi": z_hi,
          "separation_holds": bool(pct(Zc_a, 50) > z_hi)}
    g3 = {"p90_over_p50_Zs": (pct(Zsr, 90) / pct(Zsr, 50)) if pct(Zsr, 50) > 0 else None,
          "heavy_tail_flag": bool(pct(Zsr, 50) > 0 and pct(Zsr, 90) / pct(Zsr, 50) > 10)}
    g5 = {"kappa_pooled": kappa,
          "kappa_by_W": {W: (med(Zs_by_W[W]) if len(Zs_by_W[W]) >= 3 else None) for W in RUNS}}
    g5["ratio_by_W"] = {W: (g5["kappa_by_W"][W] / kappa if g5["kappa_by_W"][W] else None) for W in RUNS}
    g5["all_in_half_to_double"] = all(
        v is None or 0.5 <= v <= 2.0 for v in g5["ratio_by_W"].values())
    g5["note_W64"] = "W64 single-run kappa is small-n (expected n~12): listed only"

    # swap windows (F1 context report): qualified window whose p2l_version differs
    # from the previous qualified window IN THE SAME RUN
    swap_ids, d1_rows = set(), []
    per_run_swap = {}
    for W in RUNS:
        qs_run = [r for r in all_rows_by_W[W] if qualified(r)]
        sw = []
        for i in range(1, len(qs_run)):
            if qs_run[i].get("p2l_version") != qs_run[i - 1].get("p2l_version"):
                sw.append(qs_run[i]["window_id"])
        per_run_swap[W] = sw
        zones_W, _, _ = classify(all_rows_by_W[W])
        zmap = {r["window_id"]: zones_W[all_rows_by_W[W].index(r)] for r in qs_run}
        for r in qs_run:
            if r["window_id"] in sw:
                swap_ids.add((W, r["window_id"]))
                d1_rows.append({"W": W, "window_id": r["window_id"], "zone": zmap[r["window_id"]],
                                "z": r["z"], "event": r["event"],
                                "z_next3": [x["z"] for x in qs_run[qs_run.index(r) + 1:qs_run.index(r) + 4]]})
    sw_stat = [x["z"] / kappa for x in d1_rows if x["zone"] == "stationary"]
    non_stat = []
    for W in RUNS:
        zones_W, _, _ = classify(all_rows_by_W[W])
        for i, r in enumerate(all_rows_by_W[W]):
            if qualified(r) and zones_W[i] == "stationary" and (W, r["window_id"]) not in swap_ids:
                non_stat.append(r["z"] / kappa)
    d1 = {"n_swap_stationary": len(sw_stat), "n_nonswap_stationary": len(non_stat),
          "R_swap": (med(sw_stat) / med(non_stat)) if len(sw_stat) >= 5 and len(non_stat) >= 5 and med(non_stat) > 0 else None,
          "median_z_swap_stat": med(sw_stat), "median_z_nonswap_stat": med(non_stat),
          "band": None, "swap_windows_by_run": per_run_swap, "swap_window_detail": d1_rows}
    if d1["R_swap"] is not None:
        d1["band"] = ("<=2 compatible (F1 active + real drift)" if d1["R_swap"] <= 2
                      else ">2 STOP-before-freeze: offline F1 path reconstruction required")
    else:
        d1["band"] = "merged n<5 on a side: point values only, no ratio"
    d2 = []
    for x in d1_rows:
        back = any(z is not None and z / kappa <= z_lo for z in x["z_next3"][:3])
        d2.append({"W": x["W"], "window_id": x["window_id"], "zone": x["zone"],
                   "z_back_to_lo_within3": back})
    n_sw = len(d2)
    d2_summary = {"n_swap_windows": n_sw,
                  "recovered_overall": (sum(x["z_back_to_lo_within3"] for x in d2) / n_sw) if n_sw else None,
                  "by_zone": {zn: {"n": sum(1 for x in d2 if x["zone"] == zn),
                                   "recovered": sum(x["z_back_to_lo_within3"] for x in d2 if x["zone"] == zn)}
                              for zn in ("stationary", "change")},
                  "note": "layered report; pseudo-switch-zone swaps naturally carry high z; NOT a fail criterion"}

    # ---- PT2 (A/B) pre-fit parts: paired same-state scaling identity ----
    M = float(np.median([r["n_fresh_assignments"] for r in all_rows_by_W[64] if qualified(r)]))
    pt2 = {"scale_target_M_median_W64_nfresh": M, "n_windows": 0,
           "exact_rational_identity": 0, "float64_bitwise_identical": 0,
           "float64_max_absdiff": 0.0, "float64_max_ulp": 0.0,
           "theta_boolean_invariance_sweep_failures": 0, "mismatches": []}
    thetas = sorted(set(Hs))
    sweep = [(thetas[i] + thetas[i + 1]) / 2 for i in range(len(thetas) - 1)] + [1e-12, 2.0]
    for W in RUNS:
        for r in all_rows_by_W[W]:
            if not qualified(r):
                continue
            zones_W, _, _ = classify(all_rows_by_W[W])
            i = all_rows_by_W[W].index(r)
            if zones_W[i] != "stationary":
                continue
            h, n = r["h_w_assignments"], r["n_fresh_assignments"]
            if not n:
                continue
            c = M / n                                  # float64 scale factor (plan PT2)
            c_ex = Fraction(M) / Fraction(n)           # exact rational same scale
            hr_log = r["h_rate"]
            hr_ex = Fraction(h) / Fraction(n)          # exact rational, unscaled (h is a float count)
            hr_ex_s = (c_ex * Fraction(h)) / (c_ex * Fraction(n))  # exact rational, scaled
            hs, ns = c * h, c * n
            hr_f64 = hs / ns
            pt2["n_windows"] += 1
            ok_ex = (hr_ex_s == hr_ex)
            pt2["exact_rational_identity"] += int(ok_ex)
            same_bits = (hr_f64 == hr_log)
            pt2["float64_bitwise_identical"] += int(same_bits)
            if not same_bits:
                diff = abs(hr_f64 - hr_log)
                ulp = diff / max(math.ulp(hr_log), 1e-300)
                pt2["float64_max_absdiff"] = max(pt2["float64_max_absdiff"], diff)
                pt2["float64_max_ulp"] = max(pt2["float64_max_ulp"], ulp)
                if len(pt2["mismatches"]) < 20:
                    pt2["mismatches"].append({"W": W, "window_id": r["window_id"],
                                              "hr_log": hr_log, "hr_scaled": hr_f64,
                                              "absdiff": diff, "ulp": ulp,
                                              "exact_ok": ok_ex})
            for th in sweep:
                if (hr_log < th) != (hr_f64 < th) or (hr_log >= th) != (hr_f64 >= th):
                    pt2["theta_boolean_invariance_sweep_failures"] += 1
    pt2["gate_exact_identity_pct"] = 100.0 * pt2["exact_rational_identity"] / max(pt2["n_windows"], 1)
    pt2["gate_pass"] = bool(pt2["exact_rational_identity"] == pt2["n_windows"]
                            and pt2["theta_boolean_invariance_sweep_failures"] == 0)
    pt2["scope_note"] = ("PT2 A/B only: h_rate identity + headroom-gate boolean invariance "
                         "(any theta). Per user scope: replicated counts are NOT new independent "
                         "samples; recomputed z / UNCERTAIN invariance is NOT claimed (n_min is an "
                         "absolute-count gate; the state-machine comparison PT2-C runs post-fit with "
                         "identical logged state + logged z and reports the n_min side separately).")
    if not pt2["gate_pass"]:
        print("== PT2 GATE FAIL: F2 implementation defect suspected -> STOP, do not freeze ==")
        print(json.dumps({k: v for k, v in pt2.items() if k != "mismatches"}, indent=1))
        json.dump({"pt2": pt2}, open(os.path.join(OUTROOT, "pt2_FAIL.json"), "w"), indent=1)
        return 3

    fit = {
        "rules": "calibration_plan_v2p1.md (r3) §2/§2.1/§2.2/§3; formula verbatim from calibration_plan_v2.md §3",
        "quantile_method": "numpy.percentile method=linear; equal weight per window; effective n = pool length",
        "pools": report["pools"],
        "kappa": kappa, "z_lo": z_lo, "z_hi": z_hi, "z_urgent": z_urgent,
        "theta_r_low": theta_r_low, "theta_r_act": theta_r_act,
        "n_min_kept": 12320768,
        "zs_rescaled_quantiles": {"p50": pct(Zsr, 50), "p75": pct(Zsr, 75), "p90": z_lo,
                                  "p99": z_hi, "max": float(Zsr.max())},
        "zc_rescaled_quantiles": {"p50": pct(Zc_a, 50), "max": float(Zc_a.max())},
        "h_rate_quantiles": {"p30": theta_r_low, "p50": pct(Hs_a, 50), "p60": theta_r_act,
                             "min": float(Hs_a.min()), "max": float(Hs_a.max())},
        "n_fresh_median_stationary": float(np.median(Ns)),
        "gates": {"stationary_pool": [len(Zs), GATE_STATIONARY_POOL, len(Zs) >= GATE_STATIONARY_POOL],
                  "zc_pool": [len(Zc), GATE_ZC_POOL, len(Zc) >= GATE_ZC_POOL]},
        "diagnostics": {"G1_D": g1d, "G2": g2, "G3": g3, "G5": g5, "D1": d1, "D2": d2_summary},
        "pt2": pt2,
        "per_run": {str(W): report["runs"][W] for W in RUNS},
    }
    inv_blob = json.dumps({W: [x["window_id"] for x in inventory[W] if x["qualified"]]
                           for W in RUNS}, sort_keys=True).encode()
    fit["window_list_sha256"] = hashlib.sha256(inv_blob).hexdigest()
    json.dump(fit, open(os.path.join(OUTROOT, "fit_results_v2p1.json"), "w"), indent=1)
    json.dump(inventory, open(os.path.join(OUTROOT, "window_inventory_v2p1.json"), "w"), indent=1)

    print("== per-run ==")
    for W in RUNS:
        r = report["runs"][W]
        print(" W%-3d windows=%d steps=%d forwards_prof=%s qualified=%d (stat=%d change=%d) excl=%s"
              % (W, r["recorded_windows"], r["recorded_steps_sum"], r["server_forwards_prof"],
                 r["qualified"], r["qualified_stationary"], r["qualified_change"],
                 r["exclusion_reasons"] or "{}"))
    print("== pools ==", report["pools"], "gates OK")
    print("== FIT (one-shot, blind) ==")
    print(" kappa=%.6f z_lo=%.6f z_hi=%.6f z_urgent=%.6f theta_r_low=%.6g theta_r_act=%.6g"
          % (kappa, z_lo, z_hi, z_urgent, theta_r_low, theta_r_act))
    print("== diagnostics ==", "G2:", g2["separation_holds"], "G3 heavy tail:", g3["heavy_tail_flag"],
          "G5 ok:", g5["all_in_half_to_double"], "G1-D band:", g1d["band"], "D1 band:", d1["band"])
    print("== PT2 A/B ==", "windows:", pt2["n_windows"],
          "exact identity:", pt2["exact_rational_identity"],
          "f64 bitwise:", pt2["float64_bitwise_identical"],
          "max_ulp:", pt2["float64_max_ulp"], "gate_pass:", pt2["gate_pass"])
    print("window_list_sha256:", fit["window_list_sha256"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
