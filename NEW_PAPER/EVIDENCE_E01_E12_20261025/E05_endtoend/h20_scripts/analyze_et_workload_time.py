#!/usr/bin/env python3
"""E-T: workload lower bound -> execution time -> slack conversion validation.

RETAINED-DATA ANALYSIS ONLY. No new performance starts are executed or allowed
by this script. Every input is sha256-verified against the frozen spec
configs/followup_v1/et_workload_time_spec.json before any computation.

Theory under test (frozen in spec):
  LB(capture) = max(sum_e w_e / 8, max_e w_e),  w_e = 0 if n_e==0 else 128*ceil(n_e/128)
  For every layout pi:  max_g W_g(pi) >= LB   (unsplittable expert placement)
  Primary time conversion:  T_hat(pi) = F + c * W_max_pad(pi)
  Diagnostic:               T_hat2    = F + c * W_max_pad + g * N_pad

Calibration: prediction-v1 calibration split (6 capture units, 24 starts).
Validation primary: nearscan evaluation-01 x layers {24,28,44,71}, triple
layouts {identity, bin-r-1.00-1.05, base} = 12 conditions (pre-designated by
batch order + locked layer set; never by timing).
Validation secondary (pre-registered robustness): evaluation-02/03 x same layers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

T95_DF3 = 3.1824463052837078
T95_DF7 = 2.3646242515927849
TRIPLE = {"identity": "identity", "near_balanced": "bin-r-1.00-1.05", "greedy_base": "base"}
PRIMARY_METRIC = "completion_fenced_us_per_forward"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def ci95(values, tq):
    n = len(values)
    if n < 2:
        return {"mean": float(values[0]) if n == 1 else None, "ci95_low": None, "ci95_high": None, "n": n}
    m = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half = tq * sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "ci95_low": m - half, "ci95_high": m + half, "n": n}


def ols(x_rows, y):
    X = np.asarray(x_rows, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ beta
    resid = Y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    return {
        "coefficients": [float(b) for b in beta],
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "resid_sd": float(np.std(resid, ddof=X.shape[1])) if len(resid) > X.shape[1] else None,
        "resid_mae": float(np.mean(np.abs(resid))),
        "resid_max_abs": float(np.max(np.abs(resid))),
        "resid_signed_bias": float(np.mean(resid)),
        "n_points": int(len(Y)),
    }


def workload_from_route_metadata(rm):
    blocks = [int(b) for b in rm["expert_blocks_128"]]
    if len(blocks) != 128:
        raise ValueError("expert_blocks_128 must have 128 entries")
    w = [128 * b for b in blocks]  # zero-expert rule: blocks==0 -> w_e==0
    n_pad = float(sum(w))
    lb = max(n_pad / 8.0, float(max(w)))
    return {"w_e": w, "N_pad": n_pad, "LB_tokens": lb, "max_w_e": float(max(w)),
            "zero_expert_padding_rule": rm.get("zero_expert_padding_rule")}


def verify_locked(spec, root):
    problems = []
    verified = {}
    for name, rec in spec["locked_inputs"].items():
        p = pathlib.Path(rec["path"])
        if not p.exists():
            problems.append(f"missing {name}: {p}")
            continue
        got = sha256_file(p)
        if got != rec["sha256"]:
            problems.append(f"sha mismatch {name}: expected {rec['sha256'][:16]} got {got[:16]}")
        verified[name] = {"path": str(p), "sha256": got, "ok": got == rec["sha256"]}
    if problems:
        raise SystemExit("INPUT VERIFICATION FAILED:\n" + "\n".join(problems))
    return verified


def main():
    root = pathlib.Path("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012")
    spec_path = root / "configs/followup_v1/et_workload_time_spec.json"
    spec = load_json(spec_path)
    spec_sha = sha256_file(spec_path)
    verified = verify_locked(spec, root)

    out_path = root / "results/et_v1/analysis.json"
    if out_path.exists():
        raise SystemExit(f"REFUSE to overwrite existing result: {out_path}")

    # ---------------- calibration side ----------------
    cal_index = load_json(pathlib.Path(spec["locked_inputs"]["calibration_results_index"]["path"]))
    cal_manifests = {}
    for m in spec["calibration_layout_manifests"]:
        cal_manifests[m["capture_id"]] = load_json(pathlib.Path(m["path"]))

    # per capture unit: workload + per-layout W_max
    cal_units = {}
    for cap_id, man in cal_manifests.items():
        wl = workload_from_route_metadata(man["route_metadata"])
        layouts = {}
        for lay in man["layouts"]:
            d = lay["descriptors"]
            wmax = float(max(d["rank_padded_rows"]))
            layouts[lay["layout_id"]] = {
                "W_max_pad": wmax,
                "rank_padded_rows": [float(x) for x in d["rank_padded_rows"]],
                "padded_max_over_mean": float(d["padded_max_over_mean"]),
                "raw_max_over_mean": float(d["raw_max_over_mean"]),
            }
        cal_units[cap_id] = {"workload": wl, "layouts": layouts}

    # per capture x layout: start-level and pooled primary timing
    cal_points = {}  # (cap, layout) -> {"starts": [4 means], "blocks": [...]}
    comp_points = {}  # (cap, layout) -> per-rank comm/gather aggregates
    for res in cal_index["results"]:
        raw = load_json(pathlib.Path(res["path"]))
        cap = raw["capture_id"]
        # consistency: capture_id in raw must equal index record
        if cap != res["capture_id"]:
            raise SystemExit(f"capture_id mismatch in {res['path']}")
        for rec in raw["primary_timing_records"]:
            if rec["pass"] != "primary":
                continue
            key = (cap, rec["layout_id"])
            e = cal_points.setdefault(key, {"block_values": [], "start_means": {}})
            e["block_values"].append(float(rec[PRIMARY_METRIC]))
            sm = e["start_means"].setdefault(res["start_id"], [])
            sm.append(float(rec[PRIMARY_METRIC]))
        for rec in raw["diagnostic_timing_records"]:
            if rec["pass"] != "diagnostic":
                continue
            key = (cap, rec["layout_id"])
            c = comp_points.setdefault(key, {"dispatch": [], "gather": [], "combine": [], "layer": []})
            for rk in rec["ranks"]:
                diu = rk["detailed_iteration_us"]
                c["dispatch"].append(float(np.mean(diu["dispatch_us"])))
                c["gather"].append(float(np.mean(diu["expert_gather_us"])))
                c["combine"].append(float(np.mean(diu["combine_us"])))
                c["layer"].append(float(np.mean(diu["layer_us"])))

    # build 60-point table
    fit_rows = []
    for (cap, lay), e in sorted(cal_points.items()):
        if lay not in cal_units[cap]["layouts"]:
            raise SystemExit(f"layout {lay} missing from manifest of {cap}")
        start_means = [float(np.mean(v)) for _, v in sorted(e["start_means"].items())]
        if len(start_means) != 4:
            raise SystemExit(f"expected 4 starts for {cap}/{lay}, got {len(start_means)}")
        fit_rows.append({
            "capture_id": cap, "layout_id": lay,
            "T_mean_us": float(np.mean(start_means)), "start_means_us": start_means,
            "W_max_pad": cal_units[cap]["layouts"][lay]["W_max_pad"],
            "N_pad": cal_units[cap]["workload"]["N_pad"],
            "LB_tokens": cal_units[cap]["workload"]["LB_tokens"],
        })
    if len(fit_rows) != 60:
        raise SystemExit(f"expected 60 calibration points, got {len(fit_rows)}")

    primary_fit = ols([[1.0, r["W_max_pad"]] for r in fit_rows], [r["T_mean_us"] for r in fit_rows])
    F_us, c_us = primary_fit["coefficients"]
    diag_fit = ols([[1.0, r["W_max_pad"], r["N_pad"]] for r in fit_rows], [r["T_mean_us"] for r in fit_rows])

    # in-sample LB check: T_meas >= F + c*LB (per point; LB layout-invariant)
    lb_violations = []
    for r in fit_rows:
        lb_time = F_us + c_us * r["LB_tokens"]
        if r["T_mean_us"] < lb_time:
            lb_violations.append({"capture_id": r["capture_id"], "layout_id": r["layout_id"],
                                  "T_mean_us": r["T_mean_us"], "LB_time_us": lb_time,
                                  "deficit_us": r["T_mean_us"] - lb_time})

    # component analysis
    gather_rows, gather_y, comm_rows = [], [], {}
    for (cap, lay), c in comp_points.items():
        rows = cal_units[cap]["layouts"][lay]["rank_padded_rows"]
        n_rank = 8
        # per-rank aggregates (means over starts x blocks repetitions)
        per_rank = {}
        for i, key in enumerate(("dispatch", "gather", "combine", "layer")):
            vals = c[key]
            if len(vals) % n_rank != 0:
                raise SystemExit(f"component rank pooling broken for {cap}/{lay}")
            arr = np.asarray(vals).reshape(-1, n_rank)
            per_rank[key] = arr.mean(axis=0)  # 8 per-rank means
            per_rank[key + "_within_var"] = arr.var(axis=0, ddof=1).mean() if arr.shape[0] > 1 else 0.0
        for g in range(n_rank):
            gather_rows.append([1.0, rows[g]])
            gather_y.append(float(per_rank["gather"][g]))
        comm_rows[(cap, lay)] = {
            "comm_rank_mean": float(np.mean(per_rank["dispatch"] + per_rank["combine"])),
            "within_var": float(np.mean([per_rank[k + "_within_var"] for k in ("dispatch", "combine")])),
            "dispatch": float(np.mean(per_rank["dispatch"])),
            "combine": float(np.mean(per_rank["combine"])),
            "layer_minus_sum": float(np.mean(per_rank["layer"] - per_rank["dispatch"] - per_rank["gather"] - per_rank["combine"])),
        }
    gather_fit = ols(gather_rows, gather_y)

    # comm layout-invariance per capture: across-layout var vs within-layout repeat var
    comm_invariance = []
    for cap in sorted(cal_units):
        lays = [l for (c2, l) in comm_rows if c2 == cap]
        across = [comm_rows[(cap, l)]["comm_rank_mean"] for l in sorted(lays)]
        within = [comm_rows[(cap, l)]["within_var"] for l in sorted(lays)]
        a = float(np.var(across, ddof=1)) if len(across) > 1 else 0.0
        b = float(np.mean(within))
        comm_invariance.append({"capture_id": cap, "across_layout_var_us2": a,
                                "mean_within_layout_var_us2": b,
                                "ratio_across_over_within": (a / b) if b > 0 else None,
                                "comm_mean_us": float(np.mean(across)),
                                "N_pad": cal_units[cap]["workload"]["N_pad"]})
    comm_token_fit = ols([[1.0, ci["N_pad"]] for ci in comm_invariance], [ci["comm_mean_us"] for ci in comm_invariance])

    # ---------------- validation side ----------------
    nearscan = load_json(pathlib.Path(spec["locked_inputs"]["nearscan_scan_analysis"]["path"]))
    near_index = load_json(pathlib.Path(spec["locked_inputs"]["nearscan_layout_index"]["path"]))
    near_manifests = {}
    for mrec in near_index["manifests"]:
        near_manifests[mrec["capture_id"]] = load_json(pathlib.Path(mrec["path"]))

    layout_time_rows = {(r["capture_id"], r["layout_id"]): r for r in nearscan["layout_times"]}
    cum_rows = {(r["capture_id"], r["layout_b"]): r for r in nearscan["identity_cumulative_contrasts"]}

    def capture_geometry(cap_id):
        man = near_manifests[cap_id]
        wl = workload_from_route_metadata(man["route_metadata"])
        lays = {}
        for lay in man["layouts"]:
            lays[lay["layout_id"]] = {
                "W_max_pad": float(max(lay["descriptors"]["rank_padded_rows"])),
                "r_raw": float(lay["r_raw"]),
                "padded_max_over_mean": float(lay["padded_max_over_mean"]),
            }
        return wl, lays

    def run_validation(cap_ids, tq, label):
        abs_points, role_errs, q_points, slack_rows = [], {}, [], []
        for cap in cap_ids:
            wl, lays = capture_geometry(cap)
            near_id = TRIPLE["near_balanced"]
            near_fallback = False
            if near_id not in lays:
                near_id = TRIPLE["greedy_base"]
                near_fallback = True
            triple = {"identity": TRIPLE["identity"], "near_balanced": near_id, "greedy_base": TRIPLE["greedy_base"]}
            w_id = lays[TRIPLE["identity"]]["W_max_pad"]
            lb_time = F_us + c_us * wl["LB_tokens"]
            for role, lid in triple.items():
                lt = layout_time_rows.get((cap, lid))
                if lt is None:
                    raise SystemExit(f"missing layout_time row {cap}/{lid}")
                T = float(lt["mean_us"])
                starts = [float(x) for x in lt["start_level_mean_completion_us_per_forward"]]
                that = F_us + c_us * lays[lid]["W_max_pad"]
                err = that - T
                abs_points.append({
                    "capture_id": cap, "role": role, "layout_id": lid,
                    "near_balanced_fallback_to_base": near_fallback if role == "near_balanced" else False,
                    "r_raw": lays[lid]["r_raw"], "padded_max_over_mean": lays[lid]["padded_max_over_mean"],
                    "W_max_pad": lays[lid]["W_max_pad"], "LB_tokens": wl["LB_tokens"], "N_pad": wl["N_pad"],
                    "T_measured_mean_us": T, "T_measured_start_means_us": starts,
                    "T_hat_us": that, "signed_err_us": err, "abs_err_us": abs(err),
                    "rel_err": err / T if T else None,
                })
                role_errs.setdefault(role, []).append(err)
                sl = [s - lb_time for s in starts]
                slack_rows.append({
                    "capture_id": cap, "role": role, "layout_id": lid,
                    "T_hat_LB_us": lb_time,
                    "slack_mean_us": T - lb_time,
                    "slack_ci": ci95(sl, T95_DF3),
                    "slack_share_of_T": (T - lb_time) / T if T else None,
                    "LB_share_of_T": lb_time / T if T else None,
                    "unsplittability_price_tokens": lays[lid]["W_max_pad"] - wl["LB_tokens"],
                    "unsplittability_price_us": c_us * (lays[lid]["W_max_pad"] - wl["LB_tokens"]),
                })
            for role in ("near_balanced", "greedy_base"):
                lid = triple[role]
                cr = cum_rows.get((cap, lid))
                if cr is None:
                    raise SystemExit(f"missing cumulative contrast {cap}/{lid}")
                qhat = c_us * (w_id - lays[lid]["W_max_pad"])
                eff = [float(x) for x in cr["start_level_effects_us"]]
                errs = [qhat - e for e in eff]
                q_points.append({
                    "capture_id": cap, "role": role, "layout_id": lid,
                    "q_hat_us": qhat, "observed_q_mean_us": float(cr["mean_us"]),
                    "observed_q_ci95_low_us": float(cr["ci95_low_us"]), "observed_q_ci95_high_us": float(cr["ci95_high_us"]),
                    "q_err_mean_us": float(np.mean(errs)), "q_err_ci": ci95(errs, T95_DF3),
                    "observed_verdict": cr.get("verdict"),
                })
        role_bias = {role: ci95(errs, tq) for role, errs in role_errs.items()}
        mae = float(np.mean([p["abs_err_us"] for p in abs_points]))
        bias = float(np.mean([p["signed_err_us"] for p in abs_points]))
        maxabs = float(np.max([p["abs_err_us"] for p in abs_points]))
        return {"label": label, "captures": cap_ids, "t_quantile": tq,
                "n_conditions": len(abs_points), "abs_points": abs_points,
                "mae_us": mae, "signed_bias_us": bias, "max_abs_err_us": maxabs,
                "role_signed_bias_ci": role_bias, "q_points": q_points, "slack_rows": slack_rows}

    primary_caps = list(spec["data_roles"]["validation_primary"]["capture_ids"])
    secondary_caps = list(spec["data_roles"]["validation_secondary"]["capture_ids"])
    val_primary = run_validation(primary_caps, T95_DF3, "primary")
    val_secondary = run_validation(secondary_caps, T95_DF7, "secondary_robustness")

    # verdicts (rules frozen in spec)
    role_verdicts = {}
    for role, cb in val_primary["role_signed_bias_ci"].items():
        if cb["ci95_low"] is None:
            v = "insufficient_data"
        elif cb["ci95_low"] <= 0.0 <= cb["ci95_high"]:
            v = "compatible_with_fixed_comm_linear_compute"
        else:
            v = "violated_bias_excludes_zero"
        role_verdicts[role] = {"verdict": v, **{k: cb[k] for k in ("mean", "sd", "ci95_low", "ci95_high", "n")}}
    any_violated = any(x["verdict"] == "violated_bias_excludes_zero" for x in role_verdicts.values())
    naming = "balanced_execution_reference" if any_violated else "time_model_usable_in_validated_scope"

    analysis = {
        "analysis": "et_workload_lower_bound_to_time_conversion",
        "study_id": "h20-placement-et-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec": {"path": str(spec_path), "sha256": spec_sha},
        "analyzer": {"path": __file__, "sha256": sha256_file(pathlib.Path(__file__))},
        "no_new_performance_launches": True,
        "input_verification": verified,
        "part_a_calibration": {
            "n_points": len(fit_rows),
            "primary_model": {"form": "T_us = F + c * W_max_pad", **primary_fit,
                              "F_us": F_us, "c_us_per_padded_token": c_us},
            "diagnostic_model": {"form": "T_us = F + c * W_max_pad + g * N_pad", **diag_fit,
                                 "F_us": diag_fit["coefficients"][0], "c_us_per_padded_token": diag_fit["coefficients"][1],
                                 "g_us_per_padded_token": diag_fit["coefficients"][2]},
            "lb_in_sample_violations": lb_violations,
            "component_analysis": {
                "note": "diagnostic detailed-observation pass; used for structure (slopes/variance), NOT absolute end-to-end claims",
                "expert_gather_vs_rank_padded_rows": {"form": "gather_us = a + c_e * rank_padded_rows", **gather_fit,
                                                       "a_us": gather_fit["coefficients"][0], "c_e_us_per_padded_token": gather_fit["coefficients"][1]},
                "comm_layout_invariance": comm_invariance,
                "comm_vs_tokens": {"form": "comm_us = b0 + b1 * N_pad (6 capture means, descriptive)", **comm_token_fit},
            },
            "calibration_points": fit_rows,
        },
        "part_b_validation_primary": val_primary,
        "part_b_validation_secondary": val_secondary,
        "verdicts": {
            "role_verdicts_primary": role_verdicts,
            "naming_decision": naming,
            "lb_status": "strict workload lower bound retained unconditionally (mathematical; independent of the time model)",
            "exit_condition": "errors and applicability quantified; uncertainty reported as-is; no experiments added to chase significance",
        },
        "interpretation_rules": [
            "T_hat uses F and c fitted on calibration captures ONLY; validation never refits.",
            "CIs on validation errors reflect start/capture variance only; calibration parameter uncertainty is summarized by the in-sample residual sd reported alongside.",
            "A CI crossing zero means 'compatible with zero', not proof of no effect and not equivalence.",
            "Negative errors and negative observed q are preserved exactly.",
            "q_hat = c * (W_id - W_cand); the fixed term F cancels in gains.",
            "Slack = T_measured - (F + c*LB); it includes repeat noise and any model bias; it is NOT a proven achievable gain.",
            "Component (diagnostic-pass) numbers carry detailed-observation overhead; they support structural conclusions (linearity, layout-invariance) only.",
        ],
        "schema_version": 1,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(analysis, f, indent=2)
    tmp.rename(out_path)
    print("WROTE", out_path)
    print("sha256", sha256_file(out_path))
    # concise summary
    pa = analysis["part_a_calibration"]
    print(f"\nPRIMARY MODEL: T_us = {F_us:.3f} + {c_us:.6f} * W_max_pad  (R2={pa['primary_model']['r2']:.6f}, resid sd={pa['primary_model']['resid_sd']:.2f} us, MAE={pa['primary_model']['resid_mae']:.2f} us)")
    dg = pa["diagnostic_model"]
    print(f"DIAG MODEL:    F={dg['F_us']:.3f} c={dg['c_us_per_padded_token']:.6f} g={dg['g_us_per_padded_token']:.6f} (R2={dg['r2']:.6f})")
    ge = pa["component_analysis"]["expert_gather_vs_rank_padded_rows"]
    print(f"EXPERT GATHER: gather_us = {ge['a_us']:.3f} + {ge['c_e_us_per_padded_token']:.6f} * padded_rows (R2={ge['r2']:.6f}, resid sd={ge['resid_sd']:.2f})")
    print(f"LB in-sample violations: {len(pa['lb_in_sample_violations'])}")
    for lbl, v in (("PRIMARY", val_primary), ("SECONDARY", val_secondary)):
        print(f"\n{lbl} validation: {v['n_conditions']} conditions | MAE={v['mae_us']:.2f} us | bias={v['signed_bias_us']:+.2f} us | max|err|={v['max_abs_err_us']:.2f} us")
        for role, cb in v["role_signed_bias_ci"].items():
            print(f"  role {role:14s} bias {cb['mean']:+8.2f} us CI [{cb['ci95_low']:+8.2f},{cb['ci95_high']:+8.2f}]")
    print("\nVERDICTS:", json.dumps(analysis["verdicts"]["role_verdicts_primary"], indent=1))
    print("NAMING:", naming)


if __name__ == "__main__":
    main()
