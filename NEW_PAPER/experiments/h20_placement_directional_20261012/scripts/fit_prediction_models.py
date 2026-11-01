#!/usr/bin/env python3
"""Fit and lock M0/M1/M2 using calibration captures and timing only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np

from prediction_common import STUDY_ID, sha256_file, write_json_atomic
from prediction_models import (
    ALPHA_GRID,
    MODEL_IDS,
    direct_effect_features,
    fit_ridge,
    m2_effect_features,
    m2_rank_feature_matrix,
    predict_ridge,
    select_smallest_alpha_by_grouped_mae,
)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--results-index", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias_observed_minus_predicted": float(np.mean(residual)),
    }


def load_calibration_inputs(
    *,
    root: pathlib.Path,
    protocol_path: pathlib.Path,
    results_index_path: pathlib.Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("study_id") != STUDY_ID or protocol.get("split") != "calibration":
        raise ValueError("model fitting accepts calibration protocol only")
    if protocol.get("status") != "frozen_before_calibration_timing":
        raise ValueError("calibration protocol is not frozen")
    index = json.loads(results_index_path.read_text())
    if index.get("study_id") != STUDY_ID or index.get("split") != "calibration":
        raise ValueError("model fitting accepts calibration timing index only")
    if index.get("status") != "all_calibration_timing_results_validated_and_hash_indexed":
        raise ValueError("calibration timing index is not validated")
    if index.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("timing index references a different calibration protocol")

    manifests: dict[str, dict[str, Any]] = {}
    manifest_records: dict[str, dict[str, Any]] = {}
    for start in protocol["starts"]:
        capture_id = start["capture_id"]
        record = start["layout_manifest"]
        if capture_id in manifest_records and manifest_records[capture_id] != record:
            raise ValueError(f"layout manifest differs across starts: {capture_id}")
        manifest_records[capture_id] = record
    for capture_id, record in manifest_records.items():
        path = pathlib.Path(record["path"]).resolve()
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"calibration layout manifest hash differs: {capture_id}")
        manifest = json.loads(path.read_text())
        if manifest.get("split") != "calibration" or manifest.get("capture_id") != capture_id:
            raise ValueError(f"calibration layout manifest header differs: {capture_id}")
        manifests[capture_id] = manifest

    start_specs = {start["start_id"]: start for start in protocol["starts"]}
    primary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for record in index["results"]:
        path = pathlib.Path(record["path"]).resolve()
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"calibration raw result hash differs: {path}")
        raw = json.loads(path.read_text())
        start = start_specs.get(raw["start_id"])
        if start is None:
            raise ValueError(f"raw result has an unknown start: {raw['start_id']}")
        if raw["capture_id"] != start["capture_id"] or raw["split"] != "calibration":
            raise ValueError(f"raw result metadata differs: {raw['start_id']}")
        manifest = manifests[raw["capture_id"]]
        layout_by_id = {layout["layout_id"]: layout for layout in manifest["layouts"]}

        primary_by_block: dict[int, dict[str, float]] = defaultdict(dict)
        for timing in raw["primary_timing_records"]:
            primary_by_block[int(timing["block"])][timing["layout_id"]] = float(
                timing["completion_fenced_us_per_forward"]
            )
        if sorted(primary_by_block) != list(range(protocol["timing"]["blocks_per_start"])):
            raise ValueError(f"primary blocks differ: {raw['start_id']}")
        for layout_id in protocol["design"]["layout_ids"]:
            if layout_id == "identity":
                continue
            block_effects = [
                values["identity"] - values[layout_id]
                for _, values in sorted(primary_by_block.items())
            ]
            primary_rows.append(
                {
                    "batch_id": raw["batch_id"],
                    "layer_id": int(raw["layer_id"]),
                    "capture_id": raw["capture_id"],
                    "start_id": raw["start_id"],
                    "layout_id": layout_id,
                    "category": layout_by_id[layout_id]["category"],
                    "q_completion_us": float(np.mean(block_effects)),
                    "block_effects_us": block_effects,
                }
            )

        runner_values: dict[tuple[str, int], list[float]] = defaultdict(list)
        for timing in raw["diagnostic_timing_records"]:
            layout_id = timing["layout_id"]
            for rank, rank_record in enumerate(timing["ranks"]):
                values = [float(value) for value in rank_record["runner_run_iteration_us"]]
                if len(values) != protocol["timing"]["timed_forwards_per_activation"]:
                    raise ValueError(f"runner diagnostic cardinality differs: {raw['start_id']}")
                runner_values[(layout_id, rank)].extend(values)
        expected_runner_values = (
            protocol["timing"]["diagnostic_blocks_per_start"]
            * protocol["timing"]["timed_forwards_per_activation"]
        )
        for layout_id in protocol["design"]["layout_ids"]:
            descriptors = layout_by_id[layout_id]["descriptors"]
            feature_names, feature_matrix = m2_rank_feature_matrix(descriptors)
            for rank in range(8):
                values = runner_values[(layout_id, rank)]
                if len(values) != expected_runner_values:
                    raise ValueError(f"runner diagnostic repeats differ: {raw['start_id']}")
                diagnostic_rows.append(
                    {
                        "batch_id": raw["batch_id"],
                        "layer_id": int(raw["layer_id"]),
                        "capture_id": raw["capture_id"],
                        "start_id": raw["start_id"],
                        "layout_id": layout_id,
                        "rank": rank,
                        "feature_names": feature_names,
                        "features": feature_matrix[rank],
                        "runner_run_us": float(np.mean(values)),
                    }
                )
        raw_records.append(record)

    expected_primary = 6 * 4 * 9
    expected_diagnostic = 6 * 4 * 10 * 8
    if len(primary_rows) != expected_primary or len(diagnostic_rows) != expected_diagnostic:
        raise ValueError(
            f"calibration row cardinality differs: primary={len(primary_rows)}, "
            f"diagnostic={len(diagnostic_rows)}"
        )
    return protocol, manifests, primary_rows, diagnostic_rows, raw_records


def descriptor_pair(
    manifests: dict[str, dict[str, Any]], row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout_by_id = {
        layout["layout_id"]: layout
        for layout in manifests[row["capture_id"]]["layouts"]
    }
    return layout_by_id["identity"]["descriptors"], layout_by_id[row["layout_id"]][
        "descriptors"
    ]


def build_direct_matrix(
    model_id: str,
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray]:
    matrix = []
    names: list[str] | None = None
    for row in rows:
        identity, candidate = descriptor_pair(manifests, row)
        row_names, values = direct_effect_features(
            model_id,
            identity_descriptors=identity,
            candidate_descriptors=candidate,
        )
        if names is None:
            names = row_names
        elif names != row_names:
            raise ValueError(f"{model_id} feature order differs across rows")
        matrix.append(values)
    assert names is not None
    return names, np.stack(matrix)


def grouped_cv_direct(
    model_id: str,
    *,
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    feature_names, matrix = build_direct_matrix(model_id, rows, manifests)
    candidates = []
    unique_groups = sorted(set(groups.tolist()))
    for alpha in ALPHA_GRID:
        folds = []
        predictions = np.empty_like(target)
        for held_out in unique_groups:
            train = groups != held_out
            test = groups == held_out
            model = fit_ridge(
                matrix[train],
                target[train],
                feature_names=feature_names,
                alpha=alpha,
                fit_intercept=False,
            )
            predictions[test] = predict_ridge(model, matrix[test], feature_names)
            folds.append(
                {
                    "held_out_request_batch": held_out,
                    "training_observations": int(train.sum()),
                    "validation_observations": int(test.sum()),
                    **error_metrics(target[test], predictions[test]),
                }
            )
        record = {
            "alpha": alpha,
            "folds": folds,
            "mean_group_mae": float(np.mean([fold["mae"] for fold in folds])),
            "pooled": error_metrics(target, predictions),
        }
        candidates.append(((alpha,), record))
    selected_key, selected = select_smallest_alpha_by_grouped_mae(candidates)
    selected_alpha = selected_key[0]
    final_model = fit_ridge(
        matrix,
        target,
        feature_names=feature_names,
        alpha=selected_alpha,
        fit_intercept=False,
    )
    oof = np.empty_like(target)
    for held_out in unique_groups:
        train = groups != held_out
        test = groups == held_out
        fold_model = fit_ridge(
            matrix[train],
            target[train],
            feature_names=feature_names,
            alpha=selected_alpha,
            fit_intercept=False,
        )
        oof[test] = predict_ridge(fold_model, matrix[test], feature_names)
    cv = {
        "split": "three_fold_leave_one_complete_request_batch_out",
        "selection_loss": "mean_of_request_batch_MAE",
        "tie_break": "smallest_alpha",
        "alpha_grid": list(ALPHA_GRID),
        "candidates": [record for _, record in candidates],
        "selected_alpha": selected_alpha,
        "selected_out_of_fold": error_metrics(target, oof),
    }
    return final_model, cv, oof


def diagnostic_arrays(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = rows[0]["feature_names"]
    if any(row["feature_names"] != names for row in rows):
        raise ValueError("M2 runner feature order differs across rows")
    matrix = np.stack([row["features"] for row in rows])
    target = np.asarray([row["runner_run_us"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["batch_id"] for row in rows], dtype=object)
    return names, matrix, target, groups


def build_m2_matrix(
    *,
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    runner_model: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    names: list[str] | None = None
    matrix = []
    cache: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
    for row in rows:
        key = (row["capture_id"], row["layout_id"])
        if key not in cache:
            identity, candidate = descriptor_pair(manifests, row)
            cache[key] = m2_effect_features(
                identity_descriptors=identity,
                candidate_descriptors=candidate,
                runner_model=runner_model,
            )
        row_names, values = cache[key]
        if names is None:
            names = row_names
        elif names != row_names:
            raise ValueError("M2 completion feature order differs across rows")
        matrix.append(values)
    assert names is not None
    return names, np.stack(matrix)


def grouped_cv_m2(
    *,
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    target: np.ndarray,
    groups: np.ndarray,
    diagnostic_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    runner_names, runner_x, runner_y, runner_groups = diagnostic_arrays(diagnostic_rows)
    unique_groups = sorted(set(groups.tolist()))
    if unique_groups != sorted(set(runner_groups.tolist())):
        raise ValueError("completion and runner request-batch groups differ")
    candidates = []
    for runner_alpha in ALPHA_GRID:
        for completion_alpha in ALPHA_GRID:
            predictions = np.empty_like(target)
            folds = []
            for held_out in unique_groups:
                runner_train = runner_groups != held_out
                runner_model = fit_ridge(
                    runner_x[runner_train],
                    runner_y[runner_train],
                    feature_names=runner_names,
                    alpha=runner_alpha,
                    fit_intercept=True,
                )
                feature_names, completion_x = build_m2_matrix(
                    rows=rows,
                    manifests=manifests,
                    runner_model=runner_model,
                )
                train = groups != held_out
                test = groups == held_out
                completion_model = fit_ridge(
                    completion_x[train],
                    target[train],
                    feature_names=feature_names,
                    alpha=completion_alpha,
                    fit_intercept=False,
                )
                predictions[test] = predict_ridge(
                    completion_model, completion_x[test], feature_names
                )
                folds.append(
                    {
                        "held_out_request_batch": held_out,
                        "training_observations": int(train.sum()),
                        "validation_observations": int(test.sum()),
                        **error_metrics(target[test], predictions[test]),
                    }
                )
            record = {
                "runner_alpha": runner_alpha,
                "completion_alpha": completion_alpha,
                "folds": folds,
                "mean_group_mae": float(np.mean([fold["mae"] for fold in folds])),
                "pooled": error_metrics(target, predictions),
            }
            candidates.append(((runner_alpha, completion_alpha), record))
    selected_key, selected = select_smallest_alpha_by_grouped_mae(candidates)
    runner_alpha, completion_alpha = selected_key
    final_runner = fit_ridge(
        runner_x,
        runner_y,
        feature_names=runner_names,
        alpha=runner_alpha,
        fit_intercept=True,
    )
    feature_names, completion_x = build_m2_matrix(
        rows=rows,
        manifests=manifests,
        runner_model=final_runner,
    )
    final_completion = fit_ridge(
        completion_x,
        target,
        feature_names=feature_names,
        alpha=completion_alpha,
        fit_intercept=False,
    )
    oof = np.empty_like(target)
    for held_out in unique_groups:
        runner_train = runner_groups != held_out
        fold_runner = fit_ridge(
            runner_x[runner_train],
            runner_y[runner_train],
            feature_names=runner_names,
            alpha=runner_alpha,
            fit_intercept=True,
        )
        fold_names, fold_x = build_m2_matrix(
            rows=rows,
            manifests=manifests,
            runner_model=fold_runner,
        )
        train = groups != held_out
        test = groups == held_out
        fold_completion = fit_ridge(
            fold_x[train],
            target[train],
            feature_names=fold_names,
            alpha=completion_alpha,
            fit_intercept=False,
        )
        oof[test] = predict_ridge(fold_completion, fold_x[test], fold_names)
    cv = {
        "split": "three_fold_leave_one_complete_request_batch_out",
        "runner_and_completion_stages_refit_inside_each_fold": True,
        "selection_loss": "mean_of_request_batch_MAE",
        "tie_break": "smallest_runner_alpha_then_smallest_completion_alpha",
        "alpha_grid": list(ALPHA_GRID),
        "candidates": [record for _, record in candidates],
        "selected_runner_alpha": runner_alpha,
        "selected_completion_alpha": completion_alpha,
        "selected_out_of_fold": error_metrics(target, oof),
    }
    return final_runner, final_completion, cv, oof


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    results_index_path = args.results_index.resolve()
    protocol, manifests, rows, diagnostic_rows, raw_records = load_calibration_inputs(
        root=root,
        protocol_path=protocol_path,
        results_index_path=results_index_path,
    )
    target = np.asarray([row["q_completion_us"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["batch_id"] for row in rows], dtype=object)
    if sorted(set(groups.tolist())) != ["calibration-01", "calibration-02", "calibration-03"]:
        raise ValueError("calibration request-batch groups differ")

    models: dict[str, Any] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    for model_id in ("M0_raw", "M1_padding_128"):
        model, cv, oof = grouped_cv_direct(
            model_id,
            rows=rows,
            manifests=manifests,
            target=target,
            groups=groups,
        )
        models[model_id] = {
            "completion_model": model,
            "cross_validation": cv,
        }
        oof_predictions[model_id] = oof
    runner_model, completion_model, m2_cv, m2_oof = grouped_cv_m2(
        rows=rows,
        manifests=manifests,
        target=target,
        groups=groups,
        diagnostic_rows=diagnostic_rows,
    )
    models["M2_grouped_gemm"] = {
        "runner_compute_model": runner_model,
        "completion_model": completion_model,
        "cross_validation": m2_cv,
    }
    oof_predictions["M2_grouped_gemm"] = m2_oof

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                **{key: row[key] for key in (
                    "batch_id", "layer_id", "capture_id", "start_id", "layout_id", "category"
                )},
                "observed_q_completion_us": row["q_completion_us"],
                "out_of_fold_predictions_us": {
                    model_id: float(oof_predictions[model_id][index])
                    for model_id in MODEL_IDS
                },
                "out_of_fold_absolute_errors_us": {
                    model_id: abs(
                        row["q_completion_us"] - float(oof_predictions[model_id][index])
                    )
                    for model_id in MODEL_IDS
                },
            }
        )

    implementation_records = protocol["implementation"]
    required_lock_records = (
        "prediction_common_record",
        "prediction_models_record",
        "fit_prediction_models_record",
        "generate_prediction_calibration_layouts_record",
        "generate_prediction_evaluation_layouts_record",
        "prediction_replay_record",
        "index_prediction_timing_results_record",
    )
    for name in required_lock_records:
        record = implementation_records[name]
        path = pathlib.Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"implementation changed before model lock: {name}")

    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "models_locked_before_evaluation_layout_generation_and_timing",
        "locked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "response_definition": (
            "start-level Q = mean over four paired block effects; each block effect is "
            "completion_fenced(identity)-completion_fenced(candidate)"
        ),
        "identity_constraint": (
            "completion models use identity-minus-candidate feature differences, "
            "zero centering, RMS scaling, and no intercept; identity predicts exactly zero"
        ),
        "diagnostic_definition": (
            "M2 runner target is the mean of separate-pass same-device CUDA elapsed "
            "intervals bracketing the real runner.run call"
        ),
        "calibration_protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "calibration_results_index": {
            "path": str(results_index_path),
            "bytes": results_index_path.stat().st_size,
            "sha256": sha256_file(results_index_path),
        },
        "calibration_raw_results": raw_records,
        "calibration_layout_manifests": [
            {
                "capture_id": capture_id,
                "path": next(
                    start["layout_manifest"]["path"]
                    for start in protocol["starts"]
                    if start["capture_id"] == capture_id
                ),
                "sha256": next(
                    start["layout_manifest"]["sha256"]
                    for start in protocol["starts"]
                    if start["capture_id"] == capture_id
                ),
            }
            for capture_id in sorted(manifests)
        ],
        "calibration_only": True,
        "evaluation_routes_outputs_or_timing_used_for_fit": False,
        "model_ids": list(MODEL_IDS),
        "alpha_grid": list(ALPHA_GRID),
        "models": models,
        "out_of_fold_prediction_rows": prediction_rows,
        "evaluation_reserved_seeds": protocol["evaluation_reserved_seeds"],
        "evaluation_layout_design": protocol["evaluation_layout_design"],
        "locked_implementation": {
            name: implementation_records[name] for name in required_lock_records
        },
    }
    write_json_atomic(args.output.resolve(), payload)
    print(f"locked models: {args.output.resolve()}")
    print(f"model_lock_sha256={sha256_file(args.output.resolve())}")
    for model_id in MODEL_IDS:
        print(model_id, models[model_id]["cross_validation"]["selected_out_of_fold"])


if __name__ == "__main__":
    main()
