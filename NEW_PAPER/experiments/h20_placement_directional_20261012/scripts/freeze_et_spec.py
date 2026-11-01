#!/usr/bin/env python3
"""Freeze the E-T workload->time->slack spec with sha256 locks on every input.

Refuses to overwrite an existing spec. Locks: calibration index + 24 raw start
files, 6 calibration layout manifests, prediction model lock, nearscan analysis,
nearscan execution index, nearscan layout index + 12 manifests, and the E-T
analyzer script itself.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rec(p, purpose):
    p = pathlib.Path(p)
    if not p.exists():
        raise SystemExit(f"FATAL missing input: {p}")
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size, "purpose": purpose}


def main():
    out = ROOT / "configs/followup_v1/et_workload_time_spec.json"
    if out.exists():
        raise SystemExit(f"REFUSE to overwrite existing spec: {out}")

    cal_index_p = ROOT / "results/prediction_v1/calibration/index.json"
    cal_index = json.load(open(cal_index_p))
    model_lock_p = ROOT / "configs/followup_v1/prediction_model_lock.json"
    model_lock = json.load(open(model_lock_p))
    near_analysis_p = ROOT / "results/nearscan_v1/analysis.json"
    near_exec_index_p = ROOT / "results/nearscan_v1/execution/index.json"
    near_layout_index_p = ROOT / "inputs/nearscan_v1/layouts/index.json"
    near_layout_index = json.load(open(near_layout_index_p))
    analyzer_p = ROOT / "scripts/analyze_et_workload_time.py"

    locked = {
        "calibration_results_index": rec(cal_index_p, "calibration split hash index (24 starts)"),
        "prediction_model_lock": rec(model_lock_p, "locked prediction models + calibration manifest paths"),
        "nearscan_scan_analysis": rec(near_analysis_p, "nearscan frozen analysis: layout_times + cumulative contrasts (validation timing source)"),
        "nearscan_execution_index": rec(near_exec_index_p, "nearscan execution hash index (48 starts)"),
        "nearscan_layout_index": rec(near_layout_index_p, "nearscan layout manifest index (12 captures)"),
        "et_analyzer": rec(analyzer_p, "E-T analyzer implementation (sha-locked before run)"),
    }
    for r in cal_index["results"]:
        locked[f"calibration_raw_{r['start_id']}"] = rec(r["path"], f"calibration raw timing {r['start_id']} ({r['capture_id']})")
    for m in model_lock["calibration_layout_manifests"]:
        locked[f"calibration_manifest_{m['capture_id']}"] = rec(m["path"], f"calibration layout manifest {m['capture_id']}")
    for m in near_layout_index["manifests"]:
        locked[f"nearscan_manifest_{m['capture_id']}"] = rec(m["path"], f"nearscan layout manifest {m['capture_id']}")

    # sanity: all 12 nearscan captures carry the triple layouts (no timing involved)
    triple = ("identity", "bin-r-1.00-1.05", "base")
    for m in near_layout_index["manifests"]:
        man = json.load(open(m["path"]))
        ids = [l["layout_id"] for l in man["layouts"]]
        missing = [t for t in triple if t not in ids]
        if missing:
            raise SystemExit(f"FATAL {m['capture_id']} missing triple layouts {missing}")

    spec = {
        "spec_id": "h20-placement-et-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Validate the conversion workload -> execution time -> remaining slack on real DeepEP+DeepGEMM replay: how accurately the padded-workload lower bound and the linear-compute/fixed-communication time description reproduce measured full-path times and measured gains.",
        "supersedes": None,
        "no_new_performance_launches": True,
        "theory": {
            "workload_lower_bound": "For every layout pi: max_g W_g(pi) >= LB = max(sum_e w_e / 8, max_e w_e), where w_e = 0 if n_e == 0 else 128*ceil(n_e/128) is the padded per-expert workload (zero experts contribute zero padding). LB is layout-invariant: it is a property of the captured workload, an unsplittable-placement lower bound on per-GPU padded rows.",
            "primary_time_conversion": "T_hat(pi) = F_us + c_us_per_padded_token * W_max_pad(pi), W_max_pad = max_g sum_{e on g} w_e (padded rows of the heaviest GPU)",
            "diagnostic_time_conversion": "T_hat2 = F + c * W_max_pad + g * N_pad, N_pad = sum_e w_e (layout-invariant; tests whether cross-capture variance is token-count driven)",
            "assumptions_under_test": [
                "fixed communication: dispatch+combine time layout-invariant at fixed capture (token count), scaling with tokens across captures",
                "linear compute: grouped-expert path time linear in per-rank padded rows"
            ],
            "fallback_rule": "If the assumptions are violated: retain the strict workload lower bound (mathematically valid independently of the time model), rename time-model outputs 'balanced execution reference' (均衡执行参考), and DO NOT add experiments chasing a tight actual-time upper bound."
        },
        "data_roles": {
            "calibration": {
                "captures": sorted({r["capture_id"] for r in cal_index["results"]}),
                "n_capture_units": len({r["capture_id"] for r in cal_index["results"]}),
                "n_starts": len(cal_index["results"]),
                "structure": "each start: 10 layouts x 4 blocks x 20 timed forwards; primary metric completion_fenced_us_per_forward (minimal observation, full MoE layer path); diagnostic pass: same-device CUDA intervals around real runner.run with per-component arrays (dispatch_us, expert_gather_us, combine_us, combine_handoff_us, layer_us)",
                "usage": "fit F and c (primary), F/c/g (diagnostic); component structure analysis; in-sample LB sanity check. NEVER used for validation verdicts."
            },
            "validation_primary": {
                "capture_ids": [
                    "pred-v1-evaluation-01-layer-024",
                    "pred-v1-evaluation-01-layer-028",
                    "pred-v1-evaluation-01-layer-044",
                    "pred-v1-evaluation-01-layer-071"
                ],
                "designation_rule": "evaluation-01 trigger x the four locked layers (24,28,44,71), designated by batch order and layer set BEFORE any computation; no timing-based selection. Matches the frozen budget: at most 4 captures / 12 layout conditions (4 x triple = 12).",
                "n_conditions": 12,
                "structure": "each capture: 4 replay starts x 5 blocks x 50 timed forwards per condition per block (retained nearscan_v1 data)"
            },
            "validation_secondary": {
                "capture_ids": sorted(m["capture_id"] for m in near_layout_index["manifests"]
                                      if not m["capture_id"].startswith("pred-v1-evaluation-01")),
                "designation_rule": "evaluation-02/03 triggers x same layers; pre-registered robustness set from the same retained study; does NOT drive primary verdicts.",
                "n_conditions": 24
            }
        },
        "triple_layout_rule": {
            "identity": "identity anchor layout",
            "near_balanced": "bin-r-1.00-1.05 = FIRST greedy-path prefix whose raw r enters (1.00, 1.05], i.e. the first prefix with r <= 1.05 under the frozen nearscan first-entry rule (greedy path r is monotone non-increasing). If empty for some capture: use base anchor and flag (verified NON-empty for all 12 captures at freeze time).",
            "greedy_base": "base anchor = end of the frozen low-r lexicographic greedy path (greedy base)",
            "selection_uses_performance_timing": False
        },
        "fit_protocol": {
            "points": "60 capture x layout means (6 calibration capture units x 10 layouts); each point = mean over 4 starts x 4 blocks of completion_fenced_us_per_forward",
            "estimator": "ordinary least squares with intercept",
            "lock_order": "F and c are locked by the calibration fit BEFORE any validation quantity is computed; no refit on validation data; the analyzer enforces this by construction (single pass, spec-verified inputs)."
        },
        "measurements": {
            "absolute": "per validation condition: T_hat vs T_measured signed error (us), abs error, relative error; pooled MAE/bias/max; per-role signed bias with CI across captures",
            "gains": "q_hat = c * (W_id - W_cand) vs observed identity-cumulative contrast (start-level effects, paired blocks); per-capture error mean + CI; absolute us units (not correlation)",
            "slack": "per capture x triple layout: slack = T_measured - (F + c*LB_tokens), CI from 4 start-level means; LB share of T_identity; unsplittability price = W_max(base) - LB in tokens and us",
            "components": "calibration diagnostic pass: expert_gather_us vs rank_padded_rows pooled OLS (slope c_e, intercept, R2, resid sd); comm layout-invariance per capture (across-layout variance vs within-layout repeat variance); comm vs N_pad across captures (descriptive, n=6)",
            "reporting_rule": "absolute times AND gain differences are both reported; correlation coefficients alone are never sufficient"
        },
        "statistics": {
            "unit": "start-level means; blocks paired within start (contrast rows inherited from frozen nearscan analysis)",
            "per_capture_ci": "mean +/- t95(df=3)=3.1824463052837078 * sd/sqrt(4) over start-level values",
            "primary_role_bias_ci": "mean +/- t95(df=3) * sd/sqrt(4) across the 4 primary captures",
            "secondary_role_bias_ci": "t95(df=7)=2.3646242515927849 across 8 captures",
            "parameter_uncertainty": "validation CIs reflect start/capture variance only; calibration parameter uncertainty is reported via in-sample residual sd alongside",
            "negatives": "preserved exactly",
            "zero_crossing": "interval crossing zero = 'compatible with zero', not proof of no effect, not equivalence"
        },
        "decision_rules": {
            "role_verdict": "for each primary role: signed-bias CI containing zero -> compatible_with_fixed_comm_linear_compute; CI excluding zero -> violated_bias_excludes_zero (magnitude and direction reported)",
            "naming": "all primary roles compatible -> time_model_usable_in_validated_scope; any violated -> balanced_execution_reference (fallback rule applies)",
            "exit_condition": "obtain explicit errors and applicability scope; report uncertainty as-is; do NOT add experiments to chase significance or a tighter upper bound"
        },
        "prohibitions": [
            "no_new_performance_launches_for_et",
            "no_refit_on_validation_data",
            "no_timing_based_selection",
            "overwrite_existing_raw_result",
            "best_start_or_repeat_selection",
            "retry_failed_start_same_campaign",
            "subtract_overhead_from_measured_contrast",
            "artificial_delay_floor",
            "differential_delay",
            "sleep_or_spin",
            "mid_layer_collective_or_gate",
            "cross_device_timestamp_subtraction",
            "push_without_separate_instruction",
            "evaluation_requests_in_setup_or_map_generation"
        ],
        "locked_inputs": locked,
        "calibration_layout_manifests": model_lock["calibration_layout_manifests"],
        "schema_version": 1,
    }

    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(spec, f, indent=2)
    tmp.rename(out)
    print("FROZE", out)
    print("spec sha256:", sha(out))
    print("locked inputs:", len(locked))


if __name__ == "__main__":
    main()
