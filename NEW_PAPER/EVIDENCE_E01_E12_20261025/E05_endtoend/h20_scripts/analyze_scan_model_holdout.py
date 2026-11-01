#!/usr/bin/env python3
"""Frozen out-of-sample model comparison on the near-equilibrium scan holdout.

Applies the five frozen candidate models (locked M0/M1/M2 plus the two
calibration-only scalar comparators S_linear_no_plateau and H_padding_linear)
to the 83 non-identity scan layouts and compares them against the observed
identity-cumulative start-level effects from the frozen scan analysis. No
parameter is refit here; training R^2 is never reported; all signed errors are
preserved; model superiority is claimed only where the capture-clustered
paired-loss CI excludes zero.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any

import numpy as np
from scipy import stats

from prediction_common import sha256_file, write_json_atomic
from prediction_models import MODEL_IDS, direct_effect_features, m2_effect_features, predict_ridge

SCAN_STUDY_ID = "h20-placement-nearscan-v1"
EXPECTED_ROWS = 83
BOOTSTRAP_B = 10000


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--spec", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def check_record(record: dict[str, Any], label: str) -> None:
    path = pathlib.Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"hash differs for {label}: {path}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    spec_path = (args.spec or root / "configs/followup_v1/scan_model_holdout_spec.json").resolve()
    output_path = (args.output or root / "results/nearscan_v1/model_holdout.json").resolve()

    spec = load_json(spec_path)
    if sha256_file(pathlib.Path(__file__).resolve()) != spec["analyzer_implementation"]["sha256"]:
        raise ValueError("analyzer differs from the frozen holdout spec")
    if spec.get("study_id") != SCAN_STUDY_ID:
        raise ValueError("holdout spec belongs to another study")
    if spec.get("status") != "frozen_before_holdout_comparison":
        raise ValueError("holdout spec is not frozen before comparison")

    lock_path = pathlib.Path(spec["model_lock"]["path"])
    check_record(spec["model_lock"], "model_lock")
    lock = load_json(lock_path)
    if lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock is not in locked status")
    for name, record in lock["locked_implementation"].items():
        check_record(record, f"lock:{name}")

    check_record(spec["scan_analysis_input"], "scan_analysis")
    check_record(spec["scan_layout_index"], "scan_layout_index")
    scan = load_json(pathlib.Path(spec["scan_analysis_input"]["path"]))
    layout_index = load_json(pathlib.Path(spec["scan_layout_index"]["path"]))

    manifests: dict[str, dict[str, Any]] = {}
    for record in layout_index["manifests"]:
        check_record(record, f"scan_manifest:{record['capture_id']}")
        manifests[record["capture_id"]] = load_json(pathlib.Path(record["path"]))
    if len(manifests) != 12:
        raise ValueError("manifest count differs")

    observed = {
        (row["capture_id"], row["layout_b"]): row
        for row in scan["identity_cumulative_contrasts"]
    }
    s_value = float(spec["models_compared"]["S_linear_no_plateau"]["S_us_per_unit_delta_r"])
    h_value = float(spec["models_compared"]["H_padding_linear"]["H_us_per_unit_delta_pad"])

    model_names = list(MODEL_IDS) + ["S_linear_no_plateau", "H_padding_linear"]
    rows: list[dict[str, Any]] = []
    for capture_id in sorted(manifests):
        manifest = manifests[capture_id]
        layout_by_id = {layout["layout_id"]: layout for layout in manifest["layouts"]}
        identity = layout_by_id["identity"]["descriptors"]
        r_id = float(identity["raw_max_over_mean"])
        pad_id = float(identity["padded_max_over_mean"])
        for layout in manifest["layouts"]:
            layout_id = layout["layout_id"]
            if layout_id == "identity":
                continue
            key = (capture_id, layout_id)
            if key not in observed:
                raise ValueError(f"missing observed cumulative contrast: {key}")
            candidate = layout["descriptors"]
            preds: dict[str, float] = {}
            for model_id in MODEL_IDS:
                if model_id == "M2_grouped_gemm":
                    names, features = m2_effect_features(
                        identity_descriptors=identity,
                        candidate_descriptors=candidate,
                        runner_model=lock["models"][model_id]["runner_compute_model"],
                    )
                else:
                    names, features = direct_effect_features(
                        model_id,
                        identity_descriptors=identity,
                        candidate_descriptors=candidate,
                    )
                completion = lock["models"][model_id]["completion_model"]
                preds[model_id] = float(predict_ridge(completion, features, names))
            preds["S_linear_no_plateau"] = s_value * (
                r_id - float(candidate["raw_max_over_mean"])
            )
            preds["H_padding_linear"] = h_value * (
                pad_id - float(candidate["padded_max_over_mean"])
            )
            obs_row = observed[key]
            rows.append(
                {
                    "capture_id": capture_id,
                    "layout_id": layout_id,
                    "target_region": obs_row["target_region"],
                    "r_candidate": float(candidate["raw_max_over_mean"]),
                    "padded_candidate": float(candidate["padded_max_over_mean"]),
                    "observed_q_us": float(obs_row["mean_us"]),
                    "observed_verdict": obs_row["verdict"],
                    "predictions_us": preds,
                    "errors_us": {m: float(obs_row["mean_us"]) - preds[m] for m in model_names},
                }
            )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"holdout row count differs: {len(rows)}")

    obs = np.array([row["observed_q_us"] for row in rows])
    captures = np.array([row["capture_id"] for row in rows], dtype=object)
    per_model: dict[str, Any] = {}
    abs_err: dict[str, np.ndarray] = {}
    for model in model_names:
        pred = np.array([row["predictions_us"][model] for row in rows])
        err = obs - pred
        abs_err[model] = np.abs(err)
        rho = stats.spearmanr(pred, obs)
        tau = stats.kendalltau(pred, obs)
        per_model[model] = {
            "mae_us": float(np.mean(np.abs(err))),
            "signed_bias_us": float(np.mean(err)),
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "kendall_tau": float(tau.statistic),
            "kendall_tau_p": float(tau.pvalue),
            "by_region": {},
        }
        for region in sorted({row["target_region"] for row in rows}):
            mask = np.array([row["target_region"] == region for row in rows])
            per_model[model]["by_region"][region] = {
                "n": int(mask.sum()),
                "mae_us": float(np.mean(np.abs(err[mask]))),
                "signed_bias_us": float(np.mean(err[mask])),
            }

    # selection regret per capture over non-identity layouts
    regret: dict[str, Any] = {}
    capture_ids = sorted(manifests)
    for model in model_names:
        per_capture = []
        for capture_id in capture_ids:
            subset = [row for row in rows if row["capture_id"] == capture_id]
            oracle = max(row["observed_q_us"] for row in subset)
            best_pred = max(row["predictions_us"][model] for row in subset)
            selected = next(row for row in subset if row["predictions_us"][model] == best_pred)
            per_capture.append(oracle - selected["observed_q_us"])
        regret[model] = {
            "per_capture_regret_us": [float(value) for value in per_capture],
            "mean_regret_us": float(np.mean(per_capture)),
        }

    # capture-clustered bootstrap for paired MAE differences
    seed = int(spec["metrics"]["paired_model_differences"]["seed"])
    rng = np.random.default_rng(seed)
    unique_captures = np.array(capture_ids, dtype=object)
    cluster_index = {capture: np.where(unique_captures == capture)[0] for capture in capture_ids}
    pairs: dict[str, Any] = {}
    for i, m_i in enumerate(model_names):
        for m_j in model_names[i + 1:]:
            diff = abs_err[m_i] - abs_err[m_j]
            point = float(np.mean(diff))
            draws = np.empty(BOOTSTRAP_B, dtype=np.float64)
            for b in range(BOOTSTRAP_B):
                chosen = rng.choice(capture_ids, size=len(capture_ids), replace=True)
                idx = np.concatenate([cluster_index[c] for c in chosen])
                draws[b] = np.mean(diff[idx])
            low, high = np.percentile(draws, [2.5, 97.5])
            if low > 0.0:
                verdict = "model_i_worse_excludes_zero"
            elif high < 0.0:
                verdict = "model_i_better_excludes_zero"
            else:
                verdict = "compatible_with_zero"
            pairs[f"{m_i}_minus_{m_j}"] = {
                "mean_paired_abs_err_difference_us": point,
                "ci95_low_us": float(low),
                "ci95_high_us": float(high),
                "verdict": verdict,
            }

    payload = {
        "schema_version": 1,
        "study_id": SCAN_STUDY_ID,
        "analysis": "scan_layout_out_of_sample_model_holdout",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "analyzer_implementation": spec["analyzer_implementation"],
        "inputs": {
            "model_lock": spec["model_lock"],
            "scan_analysis_input": spec["scan_analysis_input"],
            "scan_layout_index": spec["scan_layout_index"],
        },
        "interpretation_rules": spec["interpretation_rules"],
        "rows": rows,
        "per_model": per_model,
        "selection_regret": regret,
        "paired_model_differences": pairs,
        "counts": {"holdout_rows": len(rows), "captures": len(capture_ids), "models": len(model_names)},
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite holdout analysis: {output_path}")
    write_json_atomic(output_path, payload)
    print(f"wrote {output_path}")
    for model in model_names:
        print(
            f"{model:20s} MAE {per_model[model]['mae_us']:7.1f} us  bias {per_model[model]['signed_bias_us']:+7.1f} us  "
            f"rho {per_model[model]['spearman_rho']:+.3f}  tau {per_model[model]['kendall_tau']:+.3f}  "
            f"regret {regret[model]['mean_regret_us']:6.1f} us"
        )


if __name__ == "__main__":
    main()
