#!/usr/bin/env python3
"""PT2-C v2: frozen-decision paired-scaling test + state-reconstruction
fidelity (plan §4 G1 acceptance, part C; scoped per user instruction
2026-10-17 step 3). Runs AFTER the one-shot fit, BEFORE freeze.

HARNESS FIX (v2) -- three v1 harness bugs found on first real-data run
(v1 output archived as pt2c_results_v1_harnessbug.json; v1 reported
0 F2 defects but 182/182 fidelity mismatches -- all artifacts of the
bugs below, none a controller defect; no knob, parameter or fit input
was touched by this fix):

  B1 (fidelity knob mismatch): v1 replayed the LOGGING-run events with
     the FROZEN knobs. The live events were produced under the INERT
     logging knobs (LOCK §5: kappa=1.0, z_lo/z_hi/z_urgent=1e12/2e12/
     3e12, theta_r_low=1e-12, theta_r_act=2.0, n_min=0). Reproduction
     across a knob change is impossible by construction. FIX: fidelity
     replays with the inert knobs.

  B2 (state-recon off-by-one): the controller builds the JSONL record
     AFTER ctrl.step() (controller.py phase 4; non-committed windows are
     restore(snap)-ed BEFORE logging), so the logged alert_k/stable_s/
     dwell/dwell_dir/uncertain_run/episode_shrinks are the EFFECTIVE
     END-OF-WINDOW state, i.e. the PRE-state of window i is the record
     of window i-1 -- not of window i (v1 used the same window). FIX:
     reconstruct pre-state from the previous row; additionally verify
     the replay's POST-state equals the current row's logged fields
     (full state chain check, stronger than v1).

  B3 (z basis): v1 fed z as logged (computed with kappa=1.0) into the
     frozen decision surface (z_hi=314.14, z_urgent=942.42), so every
     window landed in the alert/urgent band and the quiet/grow/midband
     paths -- the ones the behavior run will exercise -- were never
     probed. FIX: paired-identity arms use z_frozen recomputed via the
     DEPLOYED v2p1_signals.z_from_signals(s_change_logical, n_fresh,
     n_rows=94, kappa_frozen) -- the exact online formula. z_frozen is
     an INPUT held identical in both arms (the paired claim is about
     (h, n) scaling only; per user scope, recomputed-z invariance is
     NOT claimed and NOT tested).

Claims (unchanged from v1, per user scope):
  - For the SAME pre-state and SAME z input, scaling (h_w, n_fresh) ->
    (c*h_w, c*n_fresh), c = M/n (M = median qualified W64 n_fresh),
    leaves h_rate and every headroom-gated decision unchanged.
  - n_min is an ABSOLUTE-count gate: flips are classified
    "n_min_absolute_gate" (expected semantics), NOT F2 defects.
  - Replicated counts are NOT new independent samples.
Registered gate scope: qualified STATIONARY windows (Z_s pool). Qualified
CHANGE windows are additionally reported as SUPPLEMENTAL coverage (a
defect there also stops -- scope widens coverage, never leniency).
Fidelity runs on ALL qualified windows of all 4 runs.
"""
import json, os, sys, collections

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTROOT = os.environ.get("V2P1_FIT_OUTROOT", EXP + "/results/adw_v2p1/r3_logging")
RUNS = [8, 16, 32, 64]
EXCLUDED_EVENTS = {"init", "uncertain_carry", "uncertain_hold",
                   "forced_shrink_uncertain", "confirm_hold",
                   "confirm_fail_freeze", "w_desync_freeze", "fatal_collective"}
N_ROWS, BOUNDS, GUARD = 8037, [2009, 4018, 6027], 250
N_LAYERS = 94  # logical.shape[0]; controller passes it to z_from_signals
STATE_FIELDS = ("alert_k", "stable_s", "dwell", "dwell_dir",
                "uncertain_run", "episode_shrinks")

# INERT logging-run knobs -- verbatim from LOCK_record_r3_logging.md §5
# (env OEPLB_V21_*; theta_h_* carry RATE values per controller.py L1298).
INERT_KNOBS = dict(kappa=1.0, z_lo=1e12, z_hi=2e12, z_urgent=3e12,
                   m_shrink=2, m_grow=3, n_min=0, u_max=4,
                   theta_h_low=1e-12, theta_h_act=2.0,
                   dwell_shrink=2, dwell_grow=2, w_uncertain_target=16,
                   urgent_episode_cap=3)

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
    from sglang.srt.managers.pb_oeplb import v2p1_signals as sig
    fit = json.load(open(os.path.join(OUTROOT, "fit_results_v2p1.json")))
    frozen = dict(kappa=fit["kappa"], z_lo=fit["z_lo"], z_hi=fit["z_hi"],
                  z_urgent=fit["z_urgent"], m_shrink=2, m_grow=3,
                  n_min=fit["n_min_kept"], u_max=4,
                  theta_h_low=fit["theta_r_low"], theta_h_act=fit["theta_r_act"],
                  dwell_shrink=2, dwell_grow=2, w_uncertain_target=16,
                  urgent_episode_cap=3)

    ev_all = collections.Counter()
    data = {}
    for W in RUNS:
        d = os.path.join(OUTROOT, "W%d" % W)
        rows = [json.loads(l) for l in open(os.path.join(d, "v2p1_rank0.jsonl")) if l.strip()]
        # reconstruction-chain premise: every logged window committed
        bad = [r["window_id"] for r in rows if not (r.get("committed") is True
                                                    and r.get("prop_valid") == 1)]
        assert not bad, "run W%d has non-committed windows %s (chain premise)" % (W, bad)
        ev_all.update(r["event"] for r in rows)
        data[W] = (rows, classify_zones(rows))
    uncertain_present = [e for e in ev_all if "uncertain" in e or "carry" in e]

    rows64, _ = data[64]
    q64 = [r for r in rows64 if qualified(r)]
    ns = sorted(r["n_fresh_assignments"] for r in q64)
    M = (ns[len(ns) // 2] if len(ns) % 2 else
         (ns[len(ns) // 2 - 1] + ns[len(ns) // 2]) / 2.0)

    out = {"harness_version": "v2 (fixes B1 knob mismatch, B2 state off-by-one, "
                              "B3 z basis; v1 archived as pt2c_results_v1_harnessbug.json)",
           "M_median_W64_nfresh": M,
           "fidelity": {"n_windows": 0, "mismatches": [],
                        "note": ("inert-knob replay from previous-window state; "
                                 "checks live event + W_after + FULL post-state "
                                 "fields for every qualified window")},
           "paired": {"n_stationary": 0, "n_change_supplemental": 0,
                      "decision_mismatches": [],
                      "hr_bitwise_identical": 0, "hr_max_abs_delta": 0.0,
                      "n_min_side": {"orig_below": 0, "scaled_below": 0, "flips": 0},
                      "note": ("frozen knobs; z_frozen recomputed via deployed "
                               "z_from_signals (same value in BOTH arms); only "
                               "(h, n) scaled. n_min flips classified, not defects. "
                               "Stationary = registered gate; change = supplemental.")},
           "event_census_all_runs": dict(ev_all),
           "uncertain_or_carry_events_present": uncertain_present}

    rp_i = WindowControllerV2P1(16, INERT_KNOBS, beta=0.5)
    rp_f = WindowControllerV2P1(16, frozen, beta=0.5)

    def set_state(ctrl, r_prev, r_cur):
        # PRE-state of r_cur = EFFECTIVE end-of-window state of r_prev
        # (post-step if committed, post-restore if not; all committed here).
        ctrl.W = r_cur["W_before"]
        for f in STATE_FIELDS:
            setattr(ctrl, f, r_prev[f])
        ctrl.carry_windows = 0
        ctrl.window_seq = r_prev["window_id"]  # post-increment seq of prev step
        ctrl.frozen = False

    for W in RUNS:
        rows, zones = data[W]
        for i, r in enumerate(rows):
            if not qualified(r) or i == 0:
                continue  # window 1 is init (excluded); no prev for i==0 anyway
            prev = rows[i - 1]
            h, n, z_log = r["h_w_assignments"], r["n_fresh_assignments"], r["z"]
            zone = zones[i]

            # ---- FIDELITY: inert knobs, logged inputs, prev-window state ----
            set_state(rp_i, prev, r)
            rec = rp_i.step(z_log, r["h_rate"], n)
            out["fidelity"]["n_windows"] += 1
            post_bad = {f: (getattr(rp_i, f), r[f]) for f in STATE_FIELDS
                        if getattr(rp_i, f) != r[f]}
            if rec["event"] != r["event"] or rec["W_after"] != r["W_after"] or post_bad:
                out["fidelity"]["mismatches"].append(
                    {"W": W, "window_id": r["window_id"], "zone": zone,
                     "live": [r["event"], r["W_after"]],
                     "replay": [rec["event"], rec["W_after"]],
                     "post_state_mismatch_field_(replay,logged)": post_bad,
                     "pre_state_used": {f: prev[f] for f in STATE_FIELDS}})

            # ---- PAIRED IDENTITY: frozen knobs, frozen-basis z, (h,n) scaled ----
            z_f = sig.z_from_signals(r["s_change_logical"], n, N_LAYERS, frozen["kappa"])
            c = M / n
            hr_s = (c * h) / (c * n)
            n_s = c * n
            key = "n_stationary" if zone == "stationary" else "n_change_supplemental"
            out["paired"][key] += 1
            out["paired"]["hr_bitwise_identical"] += int(hr_s == r["h_rate"])
            out["paired"]["hr_max_abs_delta"] = max(
                out["paired"]["hr_max_abs_delta"], abs(hr_s - r["h_rate"]))
            ns_side = out["paired"]["n_min_side"]
            ns_side["orig_below"] += int(n < frozen["n_min"])
            ns_side["scaled_below"] += int(n_s < frozen["n_min"])
            flip = (n < frozen["n_min"]) != (n_s < frozen["n_min"])
            ns_side["flips"] += int(flip)

            set_state(rp_f, prev, r)
            rec_a = rp_f.step(z_f, r["h_rate"], n)
            set_state(rp_f, prev, r)
            rec_b = rp_f.step(z_f, hr_s, n_s)
            if rec_a["event"] != rec_b["event"] or rec_a["W_after"] != rec_b["W_after"]:
                out["paired"]["decision_mismatches"].append(
                    {"W": W, "window_id": r["window_id"], "zone": zone,
                     "orig": [rec_a["event"], rec_a["W_after"], r["h_rate"], n],
                     "scaled": [rec_b["event"], rec_b["W_after"], hr_s, n_s],
                     "n_min_flip": flip,
                     "class": ("n_min_absolute_gate" if flip else "F2_DEFECT")})

    defects = [m for m in out["paired"]["decision_mismatches"] if m["class"] == "F2_DEFECT"]
    n_pair = out["paired"]["n_stationary"] + out["paired"]["n_change_supplemental"]
    out["gate"] = {
        "decision_identity_pct": 100.0 * (n_pair - len(out["paired"]["decision_mismatches"]))
                                 / max(n_pair, 1),
        "n_min_classified_mismatches": len(out["paired"]["decision_mismatches"]) - len(defects),
        "f2_defects": len(defects),
        "state_recon_fidelity_ok": not out["fidelity"]["mismatches"],
        "pass": (not defects and not out["fidelity"]["mismatches"]),
    }
    json.dump(out, open(os.path.join(OUTROOT, "pt2c_results.json"), "w"), indent=1)
    summary = {k: v for k, v in out.items() if k != "event_census_all_runs"}
    summary["fidelity"] = {k: v for k, v in out["fidelity"].items() if k != "mismatches"}
    summary["fidelity"]["n_mismatches"] = len(out["fidelity"]["mismatches"])
    summary["paired"] = {k: v for k, v in out["paired"].items() if k != "decision_mismatches"}
    summary["paired"]["decision_mismatches"] = out["paired"]["decision_mismatches"]
    print(json.dumps(summary, indent=1))
    if out["fidelity"]["mismatches"]:
        print("FIDELITY MISMATCHES (first 10):",
              json.dumps(out["fidelity"]["mismatches"][:10], indent=1))
    if not out["gate"]["pass"]:
        print("== PT2-C v2 GATE FAIL -> STOP, do not freeze (F2 defect or state-recon failure) ==")
        return 3
    print("== PT2-C v2 PASS ==")
    return 0

if __name__ == "__main__":
    sys.exit(main())
