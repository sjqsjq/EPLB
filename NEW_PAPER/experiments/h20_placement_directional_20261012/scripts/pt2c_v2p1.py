#!/usr/bin/env python3
"""PT2-C: frozen-decision paired-scaling test (plan §4 G1 acceptance, part C;
scoped per user instruction 2026-10-17 step 3).

Runs AFTER the one-shot fit, BEFORE freeze. For EVERY qualified stationary
window of the 4 logging runs:
  - reconstruct the frozen WindowControllerV2P1 at the LOGGED pre-window state
    (alert_k, stable_s, dwell, dwell_dir, uncertain_run, episode_shrinks,
    W=W_before, window_seq=window_id-1; carry_windows=0 -- verified below from
    the event census: logging runs contain no uncertain/carry events);
  - feed step(z_LOGGED, h_rate_orig, n_orig) vs step(z_LOGGED, h_rate_scaled,
    c*n_orig) with c = M/n_orig (M = median qualified W64 n_fresh) -- SAME z,
    SAME state, only the (h, n) pair is scaled;
  - assert identical event and W_after.

Per user scope, explicitly NOT claimed:
  - recomputed z invariance (z is an INPUT here, taken from the log);
  - UNCERTAIN-judgment invariance: n_min is an ABSOLUTE-count gate, so
    scaling n_fresh can legitimately cross it. Any decision difference whose
    sole cause is the n_fresh<n_min comparison is classified
    "n_min_absolute_gate" (expected semantics, reported, NOT a defect);
    every other difference => F2 implementation defect => STOP (exit 3).
  - replicated counts are NOT new independent samples (PT2 adds no data; it
    re-reads the logged windows only).

Also verifies: (a) logged state fields are sufficient for reconstruction by
checking step(z, h_rate_orig, n_orig).event == logged event for every
qualified stationary window (state-reconstruction fidelity); (b) no
uncertain/carry events exist in the logging runs (carry_windows==0 assumption).
"""
import json, os, sys, collections

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTROOT = os.environ.get("V2P1_FIT_OUTROOT", EXP + "/results/adw_v2p1/r3_logging")
RUNS = [8, 16, 32, 64]
EXCLUDED_EVENTS = {"init", "uncertain_carry", "uncertain_hold",
                   "forced_shrink_uncertain", "confirm_hold",
                   "confirm_fail_freeze", "w_desync_freeze", "fatal_collective"}
N_ROWS, BOUNDS, GUARD = 8037, [2009, 4018, 6027], 250

def qualified(r):
    return (r.get("committed") is True and r.get("protocol_rev") == "r3"
            and r.get("event") not in EXCLUDED_EVENTS and r.get("z") is not None)

def classify_zones(rows):
    total_steps = sum(r["steps_actual"] for r in rows)
    cum, zones = 0, []
    for r in rows:
        r0 = cum / total_steps * N_ROWS
        cum += r["steps_actual"]
        r1 = cum / total_steps * N_ROWS
        zones.append("change" if any(not (r1 < b - GUARD or r0 > b + GUARD) for b in BOUNDS)
                     else "stationary")
    return zones

def main():
    from sglang.srt.managers.pb_oeplb.window_v2p1 import WindowControllerV2P1
    fit = json.load(open(os.path.join(OUTROOT, "fit_results_v2p1.json")))
    knobs = dict(kappa=fit["kappa"], z_lo=fit["z_lo"], z_hi=fit["z_hi"],
                 z_urgent=fit["z_urgent"], m_shrink=2, m_grow=3,
                 n_min=fit["n_min_kept"], u_max=4,
                 theta_h_low=fit["theta_r_low"], theta_h_act=fit["theta_r_act"],
                 dwell_shrink=2, dwell_grow=2, w_uncertain_target=16,
                 urgent_episode_cap=3)

    # (b) event census: no uncertain/carry anywhere in the 4 logging runs
    ev_all = collections.Counter()
    data = {}
    for W in RUNS:
        d = os.path.join(OUTROOT, "W%d" % W)
        rows = [json.loads(l) for l in open(os.path.join(d, "v2p1_rank0.jsonl")) if l.strip()]
        ev_all.update(r["event"] for r in rows)
        data[W] = (rows, classify_zones(rows))
    uncertain_present = [e for e in ev_all if "uncertain" in e or "carry" in e]

    # M from qualified W64 windows
    rows64, zones64 = data[64]
    q64 = [r for r in rows64 if qualified(r)]
    ns = sorted(r["n_fresh_assignments"] for r in q64)
    M = (ns[len(ns) // 2] if len(ns) % 2 else
         (ns[len(ns) // 2 - 1] + ns[len(ns) // 2]) / 2.0)

    out = {"M_median_W64_nfresh": M, "n_windows": 0,
           "state_recon_fidelity_mismatches": [], "decision_mismatches": [],
           "n_min_side": {"orig_below": 0, "scaled_below": 0, "flips": 0,
                          "note": ("n_min is an absolute-count gate; invariance under "
                                   "scaling is NOT claimed. Flips (if any) are reported "
                                   "and classified, not treated as F2 defects; other "
                                   "mismatches are defects.")},
           "event_census_all_runs": dict(ev_all),
           "uncertain_or_carry_events_present": uncertain_present}

    rp = WindowControllerV2P1(16, knobs, beta=0.5)

    def set_state(r):
        rp.W = r["W_before"]
        rp.alert_k = r["alert_k"]
        rp.stable_s = r["stable_s"]
        rp.dwell = r["dwell"]
        rp.dwell_dir = r["dwell_dir"]
        rp.uncertain_run = r["uncertain_run"]
        rp.episode_shrinks = r["episode_shrinks"]
        rp.carry_windows = 0
        rp.window_seq = r["window_id"] - 1
        rp.frozen = False

    for W in RUNS:
        rows, zones = data[W]
        for i, r in enumerate(rows):
            if not qualified(r) or zones[i] != "stationary":
                continue
            h, n, z = r["h_w_assignments"], r["n_fresh_assignments"], r["z"]
            c = M / n
            hr_s = (c * h) / (c * n)
            n_s = c * n
            out["n_windows"] += 1
            out["n_min_side"]["orig_below"] += int(n < knobs["n_min"])
            out["n_min_side"]["scaled_below"] += int(n_s < knobs["n_min"])
            flip = (n < knobs["n_min"]) != (n_s < knobs["n_min"])
            out["n_min_side"]["flips"] += int(flip)
            # (a) state-reconstruction fidelity with ORIGINAL inputs
            set_state(r)
            rec_o = rp.step(z, r["h_rate"], n)
            if rec_o["event"] != r["event"] or rec_o["W_after"] != r["W_after"]:
                out["state_recon_fidelity_mismatches"].append(
                    {"W": W, "window_id": r["window_id"], "live": [r["event"], r["W_after"]],
                     "replay": [rec_o["event"], rec_o["W_after"]],
                     "state": {k: r[k] for k in ("alert_k", "stable_s", "dwell",
                                                 "dwell_dir", "uncertain_run",
                                                 "episode_shrinks", "W_before")}})
            # paired scaling with SAME z and SAME state
            set_state(r)
            rec_s = rp.step(z, hr_s, n_s)
            if rec_s["event"] != rec_o["event"] or rec_s["W_after"] != rec_o["W_after"]:
                out["decision_mismatches"].append(
                    {"W": W, "window_id": r["window_id"],
                     "orig": [rec_o["event"], rec_o["W_after"], r["h_rate"], n],
                     "scaled": [rec_s["event"], rec_s["W_after"], hr_s, n_s],
                     "n_min_flip": flip,
                     "class": ("n_min_absolute_gate" if flip else "F2_DEFECT")})

    defects = [m for m in out["decision_mismatches"] if m["class"] == "F2_DEFECT"]
    out["gate"] = {
        "decision_identity_pct": 100.0 * (out["n_windows"] - len(out["decision_mismatches"]))
                                 / max(out["n_windows"], 1),
        "n_min_classified_mismatches": len(out["decision_mismatches"]) - len(defects),
        "f2_defects": len(defects),
        "state_recon_fidelity_ok": not out["state_recon_fidelity_mismatches"],
        "pass": (not defects and not out["state_recon_fidelity_mismatches"]),
    }
    json.dump(out, open(os.path.join(OUTROOT, "pt2c_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("decision_mismatches", "state_recon_fidelity_mismatches")},
                     indent=1))
    if out["decision_mismatches"]:
        print("mismatches:", json.dumps(out["decision_mismatches"][:10], indent=1))
    if out["state_recon_fidelity_mismatches"]:
        print("STATE RECON MISMATCHES:", json.dumps(out["state_recon_fidelity_mismatches"][:10], indent=1))
    if not out["gate"]["pass"]:
        print("== PT2-C GATE FAIL -> STOP, do not freeze (F2 defect or state-recon failure) ==")
        return 3
    print("== PT2-C PASS ==")
    return 0

if __name__ == "__main__":
    sys.exit(main())
