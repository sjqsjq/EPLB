#!/usr/bin/env python3
"""Frozen model mathematics for the H20 placement prediction study."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from prediction_common import (
    common_communication_features,
    effect_features,
    join_features,
    m0_compute_features,
    m1_compute_features,
    m2_completion_compute_features,
    m2_rank_features,
)

ALPHA_GRID = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4)
MODEL_IDS = ("M0_raw", "M1_padding_128", "M2_grouped_gemm")


def _as_matrix(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("model matrix must be finite and two-dimensional")
    return result


def _as_vector(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("model vector must be finite and one-dimensional")
    return result


def fit_ridge(
    x: Any,
    y: Any,
    *,
    feature_names: list[str],
    alpha: float,
    fit_intercept: bool,
) -> dict[str, Any]:
    matrix = _as_matrix(x)
    target = _as_vector(y)
    if matrix.shape[0] != target.shape[0] or matrix.shape[1] != len(feature_names):
        raise ValueError("ridge matrix/target/feature dimensions differ")
    if matrix.shape[0] == 0 or alpha <= 0.0:
        raise ValueError("ridge requires observations and positive alpha")
    if fit_intercept:
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        target_center = float(target.mean())
        centered_target = target - target_center
        scaling = "training_mean_and_population_sd"
    else:
        center = np.zeros(matrix.shape[1], dtype=np.float64)
        scale = np.sqrt(np.mean(matrix * matrix, axis=0))
        target_center = 0.0
        centered_target = target
        scaling = "zero_center_and_training_root_mean_square"
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (matrix - center) / scale
    gram = standardized.T @ standardized / matrix.shape[0]
    rhs = standardized.T @ centered_target / matrix.shape[0]
    coefficients_standardized = np.linalg.solve(
        gram + float(alpha) * np.eye(matrix.shape[1]), rhs
    )
    coefficients = coefficients_standardized / scale
    intercept = target_center - float(center @ coefficients)
    predictions = matrix @ coefficients + intercept
    residuals = target - predictions
    return {
        "estimator": "ridge_linear_regression",
        "objective": "mean_squared_error_plus_alpha_times_squared_l2_norm_in_frozen_scaled_coordinates",
        "alpha": float(alpha),
        "fit_intercept": bool(fit_intercept),
        "scaling": scaling,
        "feature_names": list(feature_names),
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_standardized": coefficients_standardized.tolist(),
        "coefficients_raw_units": coefficients.tolist(),
        "intercept_raw_units": intercept,
        "training_observations": int(matrix.shape[0]),
        "training_features": int(matrix.shape[1]),
        "training_mae": float(np.mean(np.abs(residuals))),
        "training_rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "training_bias_observed_minus_predicted": float(np.mean(residuals)),
    }


def predict_ridge(model: dict[str, Any], x: Any, feature_names: list[str]) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    one_dimensional = matrix.ndim == 1
    if one_dimensional:
        matrix = matrix.reshape(1, -1)
    matrix = _as_matrix(matrix)
    if feature_names != model["feature_names"]:
        raise ValueError("prediction feature order differs from locked model")
    coefficients = np.asarray(model["coefficients_raw_units"], dtype=np.float64)
    if matrix.shape[1] != coefficients.shape[0]:
        raise ValueError("prediction feature dimension differs from locked model")
    values = matrix @ coefficients + float(model["intercept_raw_units"])
    return values[0] if one_dimensional else values


def direct_effect_features(
    model_id: str,
    *,
    identity_descriptors: dict[str, Any],
    candidate_descriptors: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    if model_id == "M0_raw":
        identity_compute = m0_compute_features(identity_descriptors)
        candidate_compute = m0_compute_features(candidate_descriptors)
    elif model_id == "M1_padding_128":
        identity_compute = m1_compute_features(identity_descriptors)
        candidate_compute = m1_compute_features(candidate_descriptors)
    else:
        raise ValueError(f"direct feature path does not support {model_id}")
    identity = join_features(
        identity_compute, common_communication_features(identity_descriptors)
    )
    candidate = join_features(
        candidate_compute, common_communication_features(candidate_descriptors)
    )
    return effect_features(identity, candidate)


def m2_rank_feature_matrix(descriptors: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    rows = []
    names: list[str] | None = None
    for blocks in descriptors["rank_block_multisets"]:
        row_names, values = m2_rank_features(blocks)
        if names is None:
            names = row_names
        elif names != row_names:
            raise ValueError("M2 rank feature definitions differ")
        rows.append(values)
    assert names is not None
    return names, np.stack(rows)


def m2_effect_features(
    *,
    identity_descriptors: dict[str, Any],
    candidate_descriptors: dict[str, Any],
    runner_model: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    identity_rank_names, identity_rank_matrix = m2_rank_feature_matrix(identity_descriptors)
    candidate_rank_names, candidate_rank_matrix = m2_rank_feature_matrix(candidate_descriptors)
    if identity_rank_names != candidate_rank_names:
        raise ValueError("M2 rank feature definitions differ between layouts")
    identity_runner = predict_ridge(
        runner_model, identity_rank_matrix, identity_rank_names
    )
    candidate_runner = predict_ridge(
        runner_model, candidate_rank_matrix, candidate_rank_names
    )
    identity = join_features(
        m2_completion_compute_features(identity_runner),
        common_communication_features(identity_descriptors),
    )
    candidate = join_features(
        m2_completion_compute_features(candidate_runner),
        common_communication_features(candidate_descriptors),
    )
    return effect_features(identity, candidate)


def select_smallest_alpha_by_grouped_mae(
    candidates: Iterable[tuple[tuple[float, ...], dict[str, Any]]]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    records = list(candidates)
    if not records:
        raise ValueError("no hyperparameter candidates")
    return min(
        records,
        key=lambda item: (
            float(item[1]["mean_group_mae"]),
            *item[0],
        ),
    )
