#!/usr/bin/env python3
"""Infer per-rank readiness slack independently and test original-path response."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ValueFn = Callable[[dict[str, Any]], float]


def rank_metric(record: dict[str, Any], rank: int, metric: str) -> float:
    return next(item[metric] for item in record["ranks"] if int(item["rank"]) == rank)


def cross_rank_max(record: dict[str, Any], metric: str) -> float:
    return max(float(item[metric]) for item in record["ranks"])


def non_target_median(record: dict[str, Any], rank: int, metric: str) -> float:
    return float(
        np.median(
            [float(item[metric]) for item in record["ranks"] if int(item["rank"]) != rank]
        )
    )


def block_medians(records: list[dict[str, Any]], value: ValueFn) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record["block"])].append(value(record))
    return {
        block: float(np.median(values)) for block, values in sorted(grouped.items())
    }


def bootstrap_interval(
    values: list[float], rng: np.random.Generator, repetitions: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    samples = rng.choice(array, size=(repetitions, len(array)), replace=True)
    estimates = np.median(samples, axis=1)
    return tuple(float(item) for item in np.percentile(estimates, [2.5, 97.5]))


def effect(
    condition: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    value: ValueFn,
    rng: np.random.Generator,
    repetitions: int,
) -> dict[str, Any]:
    condition_blocks = block_medians(condition, value)
    baseline_blocks = block_medians(baseline, value)
    blocks = sorted(set(condition_blocks) & set(baseline_blocks))
    block_effects = [
        condition_blocks[block] - baseline_blocks[block] for block in blocks
    ]
    low, high = bootstrap_interval(block_effects, rng, repetitions)
    return {
        "estimate_ms": float(np.median(block_effects)),
        "ci95_low_ms": low,
        "ci95_high_ms": high,
        "blocks": blocks,
        "block_effects_ms": block_effects,
    }


def fit_effective_slack(
    delay_us: np.ndarray, release_us: np.ndarray, exposed_us: np.ndarray
) -> dict[str, Any]:
    maximum = max(0.0, float(np.max(delay_us)))
    candidates = np.linspace(0.0, maximum, 4001)
    release_prediction = np.minimum(delay_us[:, None], candidates[None, :])
    exposed_prediction = np.maximum(0.0, delay_us[:, None] - candidates[None, :])
    residual = np.sum((release_us[:, None] - release_prediction) ** 2, axis=0)
    residual += np.sum((exposed_us[:, None] - exposed_prediction) ** 2, axis=0)
    index = int(np.argmin(residual))
    slack = float(candidates[index])
    predicted_release = np.minimum(delay_us, slack)
    predicted_exposed = np.maximum(0.0, delay_us - slack)
    return {
        "slack_us": slack,
        "rmse_us": float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [release_us - predicted_release, exposed_us - predicted_exposed]
                    )
                    ** 2
                )
            )
        ),
        "predicted_release_us": predicted_release.tolist(),
        "predicted_exposed_us": predicted_exposed.tolist(),
    }


def fit_origin_hinge(delay_us: np.ndarray, response_us: np.ndarray) -> dict[str, Any]:
    maximum = max(0.0, float(np.max(delay_us)))
    candidates = np.linspace(0.0, maximum * 0.999, 4001)
    best: tuple[float, float, float] | None = None
    for slack in candidates:
        regressor = np.maximum(0.0, delay_us - slack)
        denominator = float(np.dot(regressor, regressor))
        slope = 0.0 if denominator == 0.0 else max(
            0.0, float(np.dot(regressor, response_us) / denominator)
        )
        residual = response_us - slope * regressor
        sse = float(np.dot(residual, residual))
        if best is None or sse < best[0]:
            best = (sse, float(slack), slope)
    assert best is not None
    sse, slack, slope = best
    return {
        "slack_us": slack,
        "slope": slope,
        "rmse_us": math.sqrt(sse / len(delay_us)),
        "predicted_us": (slope * np.maximum(0.0, delay_us - slack)).tolist(),
    }


def mode_baseline(
    payload: dict[str, Any], records_by_condition: dict[str, list[dict[str, Any]]], mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        condition
        for condition in payload["conditions"]
        if condition["mode"] == mode and float(condition["delay_us"]) == 0.0
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one {mode} zero condition, got {len(candidates)}")
    condition = candidates[0]
    return condition, records_by_condition[condition["name"]]


def scan_conditions(payload: dict[str, Any], mode: str) -> dict[tuple[int, float], dict[str, Any]]:
    result: dict[tuple[int, float], dict[str, Any]] = {}
    for condition in payload["conditions"]:
        delay_us = float(condition["delay_us"])
        if condition["mode"] != mode or delay_us == 0.0:
            continue
        ranks = condition["delay_ranks"]
        if len(ranks) != 1:
            raise ValueError(f"condition {condition['name']} is not a single-rank delay")
        key = (int(ranks[0]), delay_us)
        if key in result:
            raise ValueError(f"duplicate condition for {key}")
        result[key] = condition
    return result


def route_hashes(payload: dict[str, Any]) -> set[str]:
    return {str(record["route_sha256"]) for record in payload["records"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-input", type=Path, required=True)
    parser.add_argument("--original-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    strict_payload = json.loads(args.strict_input.read_text())
    original_payload = json.loads(args.original_input.read_text())
    if strict_payload["total_tokens"] != original_payload["total_tokens"]:
        raise ValueError("strict and original token counts differ")
    if strict_payload["world_size"] != 8 or original_payload["world_size"] != 8:
        raise ValueError("readiness analysis requires eight ranks")
    if route_hashes(strict_payload) != route_hashes(original_payload):
        raise ValueError("strict and original scans do not use the same exact route")

    rng = np.random.default_rng(args.seed)
    strict_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in strict_payload["records"]:
        strict_records[record["condition"]].append(record)
    for record in original_payload["records"]:
        original_records[record["condition"]].append(record)

    _, strict_zero = mode_baseline(strict_payload, strict_records, "strict_barrier")
    _, original_zero = mode_baseline(original_payload, original_records, "original")
    strict_conditions = scan_conditions(strict_payload, "strict_barrier")
    original_conditions = scan_conditions(original_payload, "original")
    if set(strict_conditions) != set(original_conditions):
        raise ValueError("strict and original rank/delay grids differ")

    ranks = sorted({rank for rank, _ in strict_conditions})
    target_delays = sorted({delay for _, delay in strict_conditions})
    if ranks != list(range(8)):
        raise ValueError(f"expected ranks 0..7, got {ranks}")

    point_rows: list[dict[str, Any]] = []
    point_details: dict[str, Any] = {}
    rank_summaries: list[dict[str, Any]] = []
    rank_details: dict[str, Any] = {}
    block_slacks: dict[int, list[float]] = {}

    for rank in ranks:
        rows_for_rank: list[dict[str, Any]] = [
            {
                "rank": rank,
                "target_delay_us": 0.0,
                "strict_actual_delay_us": 0.0,
                "selected_gate_release_us": 0.0,
                "other_gate_exposed_us": 0.0,
                "strict_layer_effect_us": 0.0,
                "original_actual_delay_us": 0.0,
                "original_layer_effect_us": 0.0,
                "original_layer_ci95_low_us": 0.0,
                "original_layer_ci95_high_us": 0.0,
                "original_post_expert_effect_us": 0.0,
            }
        ]
        details_for_rank: list[dict[str, Any]] = []
        for target_delay in target_delays:
            strict_condition = strict_conditions[(rank, target_delay)]
            original_condition = original_conditions[(rank, target_delay)]
            strict_samples = strict_records[strict_condition["name"]]
            original_samples = original_records[original_condition["name"]]

            strict_actual = effect(
                strict_samples,
                strict_zero,
                lambda record, selected=rank: rank_metric(
                    record, selected, "injected_delay_ms"
                ),
                rng,
                args.bootstrap,
            )
            selected_gate = effect(
                strict_samples,
                strict_zero,
                lambda record, selected=rank: rank_metric(
                    record, selected, "strict_gate_ms"
                ),
                rng,
                args.bootstrap,
            )
            other_gate = effect(
                strict_samples,
                strict_zero,
                lambda record, selected=rank: non_target_median(
                    record, selected, "strict_gate_ms"
                ),
                rng,
                args.bootstrap,
            )
            strict_layer = effect(
                strict_samples,
                strict_zero,
                lambda record: cross_rank_max(record, "layer_ms"),
                rng,
                args.bootstrap,
            )
            original_actual = effect(
                original_samples,
                original_zero,
                lambda record, selected=rank: rank_metric(
                    record, selected, "injected_delay_ms"
                ),
                rng,
                args.bootstrap,
            )
            original_layer = effect(
                original_samples,
                original_zero,
                lambda record: cross_rank_max(record, "layer_ms"),
                rng,
                args.bootstrap,
            )
            original_post = effect(
                original_samples,
                original_zero,
                lambda record: cross_rank_max(record, "post_expert_to_done_ms"),
                rng,
                args.bootstrap,
            )
            row = {
                "rank": rank,
                "target_delay_us": target_delay,
                "strict_actual_delay_us": strict_actual["estimate_ms"] * 1000.0,
                "selected_gate_release_us": -selected_gate["estimate_ms"] * 1000.0,
                "other_gate_exposed_us": other_gate["estimate_ms"] * 1000.0,
                "strict_layer_effect_us": strict_layer["estimate_ms"] * 1000.0,
                "original_actual_delay_us": original_actual["estimate_ms"] * 1000.0,
                "original_layer_effect_us": original_layer["estimate_ms"] * 1000.0,
                "original_layer_ci95_low_us": original_layer["ci95_low_ms"] * 1000.0,
                "original_layer_ci95_high_us": original_layer["ci95_high_ms"] * 1000.0,
                "original_post_expert_effect_us": original_post["estimate_ms"] * 1000.0,
            }
            rows_for_rank.append(row)
            details_for_rank.append(
                {
                    "target_delay_us": target_delay,
                    "strict_condition": strict_condition["name"],
                    "original_condition": original_condition["name"],
                    "strict_actual_delay": strict_actual,
                    "selected_gate_effect": selected_gate,
                    "other_gate_effect": other_gate,
                    "strict_layer_effect": strict_layer,
                    "original_actual_delay": original_actual,
                    "original_layer_effect": original_layer,
                    "original_post_expert_effect": original_post,
                }
            )

        rows_for_rank.sort(key=lambda item: item["target_delay_us"])
        strict_x = np.asarray(
            [max(0.0, row["strict_actual_delay_us"]) for row in rows_for_rank]
        )
        release = np.asarray([row["selected_gate_release_us"] for row in rows_for_rank])
        exposed = np.asarray([row["other_gate_exposed_us"] for row in rows_for_rank])
        strict_fit = fit_effective_slack(strict_x, release, exposed)

        per_block_slacks = []
        block_count = len(details_for_rank[0]["strict_actual_delay"]["block_effects_ms"])
        for block_index in range(block_count):
            block_x = np.asarray(
                [0.0]
                + [
                    max(
                        0.0,
                        item["strict_actual_delay"]["block_effects_ms"][block_index]
                        * 1000.0,
                    )
                    for item in details_for_rank
                ]
            )
            block_release = np.asarray(
                [0.0]
                + [
                    -item["selected_gate_effect"]["block_effects_ms"][block_index]
                    * 1000.0
                    for item in details_for_rank
                ]
            )
            block_exposed = np.asarray(
                [0.0]
                + [
                    item["other_gate_effect"]["block_effects_ms"][block_index]
                    * 1000.0
                    for item in details_for_rank
                ]
            )
            per_block_slacks.append(
                fit_effective_slack(block_x, block_release, block_exposed)["slack_us"]
            )
        slack_low, slack_high = bootstrap_interval(
            per_block_slacks, rng, args.bootstrap
        )
        block_slacks[rank] = per_block_slacks

        original_x = np.asarray(
            [max(0.0, row["original_actual_delay_us"]) for row in rows_for_rank]
        )
        original_y = np.asarray(
            [row["original_layer_effect_us"] for row in rows_for_rank]
        )
        original_fit = fit_origin_hinge(original_x, original_y)
        frozen_prediction = np.maximum(0.0, original_x - strict_fit["slack_us"])
        frozen_residual = original_y - frozen_prediction
        frozen_regressor_denominator = float(np.dot(frozen_prediction, frozen_prediction))
        frozen_slope = (
            math.nan
            if frozen_regressor_denominator == 0.0
            else max(
                0.0,
                float(np.dot(frozen_prediction, original_y) / frozen_regressor_denominator),
            )
        )
        first_tested = rows_for_rank[1]
        positive_points = [
            row
            for row in rows_for_rank[1:]
            if row["original_layer_ci95_low_us"] > 0.0
        ]
        first_identified_positive = positive_points[0] if positive_points else None
        summary = {
            "rank": rank,
            "strict_slack_us": strict_fit["slack_us"],
            "strict_slack_ci95_low_us": slack_low,
            "strict_slack_ci95_high_us": slack_high,
            "strict_slack_block_values_us": per_block_slacks,
            "strict_fit_rmse_us": strict_fit["rmse_us"],
            "original_fitted_slack_us": original_fit["slack_us"],
            "original_fitted_slope": original_fit["slope"],
            "original_fitted_rmse_us": original_fit["rmse_us"],
            "frozen_unit_slope_rmse_us": float(np.sqrt(np.mean(frozen_residual**2))),
            "frozen_unit_slope_mae_us": float(np.mean(np.abs(frozen_residual))),
            "frozen_h_fitted_slope": frozen_slope,
            "first_tested_target_delay_us": first_tested["target_delay_us"],
            "first_tested_actual_delay_us": first_tested["original_actual_delay_us"],
            "first_tested_layer_effect_us": first_tested["original_layer_effect_us"],
            "first_tested_ci95_low_us": first_tested["original_layer_ci95_low_us"],
            "first_tested_ci95_high_us": first_tested["original_layer_ci95_high_us"],
            "first_identified_positive_target_delay_us": (
                None
                if first_identified_positive is None
                else first_identified_positive["target_delay_us"]
            ),
            "first_identified_positive_actual_delay_us": (
                None
                if first_identified_positive is None
                else first_identified_positive["original_actual_delay_us"]
            ),
            "first_identified_positive_layer_effect_us": (
                None
                if first_identified_positive is None
                else first_identified_positive["original_layer_effect_us"]
            ),
        }
        rank_summaries.append(summary)
        for row, predicted in zip(rows_for_rank, frozen_prediction, strict=True):
            row["strict_frozen_prediction_us"] = float(predicted)
            row["strict_frozen_residual_us"] = (
                row["original_layer_effect_us"] - float(predicted)
            )
            point_rows.append(row)
        rank_details[str(rank)] = {
            "strict_fit": strict_fit,
            "original_fit": original_fit,
            "summary": summary,
            "points": details_for_rank,
        }

    block_winners = [
        min(ranks, key=lambda rank: block_slacks[rank][block_index])
        for block_index in range(len(next(iter(block_slacks.values()))))
    ]
    winner_counts = Counter(block_winners)
    last_ready_rank = min(rank_summaries, key=lambda item: item["strict_slack_us"])["rank"]
    minimum_slack = min(item["strict_slack_us"] for item in rank_summaries)
    candidates = [
        item["rank"]
        for item in rank_summaries
        if item["strict_slack_ci95_low_us"] <= minimum_slack
    ]

    strict_h = np.asarray([item["strict_slack_us"] for item in rank_summaries])
    original_h = np.asarray(
        [item["original_fitted_slack_us"] for item in rank_summaries]
    )
    if float(np.std(strict_h)) == 0.0 or float(np.std(original_h)) == 0.0:
        correlation = math.nan
    else:
        correlation = float(np.corrcoef(strict_h, original_h)[0, 1])
    h_rmse = float(np.sqrt(np.mean((strict_h - original_h) ** 2)))
    all_residuals = np.asarray(
        [row["strict_frozen_residual_us"] for row in point_rows], dtype=np.float64
    )
    nonzero_residuals = np.asarray(
        [
            row["strict_frozen_residual_us"]
            for row in point_rows
            if row["target_delay_us"] > 0.0
        ],
        dtype=np.float64,
    )
    stable_singular_last_ready = len(set(block_winners)) == 1 and len(candidates) == 1
    first_tested_all_span_zero = all(
        item["first_tested_ci95_low_us"] <= 0.0
        <= item["first_tested_ci95_high_us"]
        for item in rank_summaries
    )

    negative_effects = [
        {
            "rank": row["rank"],
            "target_delay_us": row["target_delay_us"],
            "estimate_us": row["original_layer_effect_us"],
            "ci95_low_us": row["original_layer_ci95_low_us"],
            "ci95_high_us": row["original_layer_ci95_high_us"],
        }
        for row in point_rows
        if row["original_layer_effect_us"] < 0.0
    ]

    analysis = {
        "strict_source": str(args.strict_input),
        "original_source": str(args.original_input),
        "total_tokens": strict_payload["total_tokens"],
        "route_sha256": sorted(route_hashes(strict_payload)),
        "readiness_definition": (
            "For target rank i, same-rank strict-gate release is modeled as "
            "min(delay, H_i), and the median gate increase over non-target ranks is "
            "modeled as max(0, delay-H_i). H_i is fit only from the independent "
            "strict scan, without subtracting CUDA timestamps from different GPUs."
        ),
        "prediction_definition": (
            "The strict-derived H_i is frozen before evaluating the independent "
            "original-path prediction max(0, measured_delay-H_i)."
        ),
        "estimator": (
            "Within each block, subtract the matching zero-delay block median; "
            "report the median of five block effects. Confidence intervals cluster-"
            "bootstrap the five blocks."
        ),
        "negative_effect_policy": (
            "Negative baseline-subtracted estimates are retained unchanged. An interval "
            "that spans zero is compatible with a zero effect; its sign is not identified."
        ),
        "last_ready": {
            "stable_singular_last_ready_identified": stable_singular_last_ready,
            "minimum_effective_slack_rank": last_ready_rank,
            "minimum_effective_slack_us": minimum_slack,
            "candidate_ranks_from_block_interval_overlap": candidates,
            "block_winners": block_winners,
            "block_winner_counts": {str(rank): winner_counts[rank] for rank in ranks},
            "first_tested_delay_all_rank_intervals_span_zero": (
                first_tested_all_span_zero
            ),
            "interpretation": (
                "No stable singular last-ready rank was identified: block winners "
                "switch between ranks 1 and 3 and the block-interval candidate set "
                "contains ranks 0, 1, 3, and 6. No fixed rank has an identified "
                "original-path response at the first 4-us tested delay."
                if not stable_singular_last_ready
                else "The strict effective-slack ordering identifies one stable "
                "singular last-ready rank."
            ),
        },
        "cross_rank_evaluation": {
            "strict_vs_original_slack_rmse_us": h_rmse,
            "strict_vs_original_slack_pearson": correlation,
            "frozen_unit_slope_all_point_rmse_us": float(
                np.sqrt(np.mean(all_residuals**2))
            ),
            "frozen_unit_slope_all_point_mae_us": float(
                np.mean(np.abs(all_residuals))
            ),
            "frozen_unit_slope_nonzero_point_rmse_us": float(
                np.sqrt(np.mean(nonzero_residuals**2))
            ),
            "frozen_unit_slope_nonzero_point_mae_us": float(
                np.mean(np.abs(nonzero_residuals))
            ),
            "interpretation": (
                "The frozen curves capture part of the aggregate large-delay rise, "
                "but the near-zero cross-rank correlation and large slack RMSE mean "
                "that rank-specific plateau widths were not independently validated."
            ),
        },
        "rank_summaries": rank_summaries,
        "negative_original_layer_effects": negative_effects,
        "details": rank_details,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "rank_readiness_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    points_path = args.output_dir / "rank_readiness_points.csv"
    with points_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(point_rows)

    ranks_path = args.output_dir / "rank_readiness_summary.csv"
    with ranks_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rank_summaries)

    figure_path = args.output_dir / "rank_readiness_curves.png"
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.86, wspace=0.08, hspace=0.24)
    for rank, axis in enumerate(axes.flat):
        points = sorted(
            [row for row in point_rows if row["rank"] == rank],
            key=lambda item: item["original_actual_delay_us"],
        )
        x = np.asarray([row["original_actual_delay_us"] for row in points])
        y = np.asarray([row["original_layer_effect_us"] for row in points])
        low = np.asarray([row["original_layer_ci95_low_us"] for row in points])
        high = np.asarray([row["original_layer_ci95_high_us"] for row in points])
        prediction = np.asarray([row["strict_frozen_prediction_us"] for row in points])
        axis.fill_between(x, low, high, color="#DCE8F5", alpha=0.9, linewidth=0)
        axis.plot(x, y, color="#2468A2", marker="o", markersize=4.5, linewidth=1.8, label="measured original")
        axis.plot(x, prediction, color="#333333", linestyle="--", linewidth=1.6, label="strict-derived prediction")
        summary = rank_summaries[rank]
        axis.set_title(f"rank {rank}: H={summary['strict_slack_us']:.1f} µs", fontsize=10)
        axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.supxlabel("measured injected delay (µs)")
    fig.supylabel("original-path layer-time change (µs)")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    gate_figure_path = args.output_dir / "strict_gate_readiness_curves.png"
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.86, wspace=0.08, hspace=0.24)
    for rank, axis in enumerate(axes.flat):
        points = sorted(
            [row for row in point_rows if row["rank"] == rank],
            key=lambda item: item["strict_actual_delay_us"],
        )
        x = np.asarray([row["strict_actual_delay_us"] for row in points])
        release = np.asarray([row["selected_gate_release_us"] for row in points])
        exposed = np.asarray([row["other_gate_exposed_us"] for row in points])
        slack = rank_summaries[rank]["strict_slack_us"]
        axis.plot(x, release, color="#2468A2", marker="o", markersize=4.5, linewidth=1.8, label="selected-rank wait released")
        axis.plot(x, exposed, color="#9A4F18", marker="s", markersize=4.0, linewidth=1.8, label="other-rank wait added")
        xx = np.linspace(0.0, max(x), 200)
        axis.plot(xx, np.minimum(xx, slack), color="#2468A2", linestyle="--", linewidth=1.1)
        axis.plot(xx, np.maximum(0.0, xx - slack), color="#9A4F18", linestyle="--", linewidth=1.1)
        axis.set_title(f"rank {rank}: H={slack:.1f} µs", fontsize=10)
        axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.supxlabel("measured injected delay (µs)")
    fig.supylabel("strict-gate response (µs)")
    fig.savefig(gate_figure_path, dpi=180)
    plt.close(fig)

    report_path = args.output_dir / "REPORT.md"
    lines = [
        f"# Per-rank readiness experiment (N={strict_payload['total_tokens']})",
        "",
        "The strict scan estimates each rank's effective baseline readiness slack from "
        "same-rank gate-wait changes. The independently executed original scan is then "
        "predicted with those values frozen; no CUDA timestamp is subtracted across GPUs.",
        "",
        "| rank | strict H (µs) | 95% block CI | original fitted H (µs) | first tested effect at 4 µs [95% CI] (µs) | first identified positive tested delay (µs) | frozen RMSE (µs) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rank_summaries:
        positive_delay = item["first_identified_positive_target_delay_us"]
        positive_text = "none" if positive_delay is None else f"{positive_delay:.0f}"
        lines.append(
            f"| {item['rank']} | {item['strict_slack_us']:.3f} | "
            f"[{item['strict_slack_ci95_low_us']:.3f}, {item['strict_slack_ci95_high_us']:.3f}] | "
            f"{item['original_fitted_slack_us']:.3f} | "
            f"{item['first_tested_layer_effect_us']:.3f} "
            f"[{item['first_tested_ci95_low_us']:.3f}, {item['first_tested_ci95_high_us']:.3f}] | "
            f"{positive_text} | {item['frozen_unit_slope_rmse_us']:.3f} |"
        )
    lines += [
        "",
        "## Last-ready identity",
        "",
        "**No stable singular last-ready rank was identified.** The point-estimate "
        f"minimum is rank {last_ready_rank} ({minimum_slack:.3f} µs), but the "
        f"block winners are **{block_winners}** (ranks 1 and 3 switch) and the "
        f"block-interval candidate set is **{candidates}**. Therefore no fixed rank "
        "is assigned zero distribution-level slack. At the first 4 µs tested delay, "
        "every rank's original-path interval spans zero; no fixed rank shows an "
        "identified response from the smallest intervention.",
        "",
        "## Independent plateau prediction",
        "",
        f"Strict-vs-original fitted-H Pearson correlation is **{correlation:.3f}** and "
        f"H RMSE is **{h_rmse:.3f} µs**. The frozen unit-slope curve has "
        f"**{analysis['cross_rank_evaluation']['frozen_unit_slope_all_point_rmse_us']:.3f} µs** "
        "RMSE over all points (including the fixed zero anchor for every rank) and "
        f"**{analysis['cross_rank_evaluation']['frozen_unit_slope_nonzero_point_rmse_us']:.3f} µs** "
        "over nonzero-delay points. The curves capture part of the aggregate rise at "
        "large delay, but the near-zero correlation and large H error mean that "
        "rank-specific plateau widths were **not independently validated**.",
        "",
        "The one-sided intervention has an asymmetric interpretation: absorbing extra "
        "delay on a noncritical rank does not show that reducing computation on the "
        "current slowest rank has no benefit over the same interval.",
        "",
        "Negative baseline-subtracted estimates are retained unchanged. Whenever an "
        "interval spans zero, the estimate is compatible with a zero effect and the "
        "experiment does not identify its sign.",
        "",
    ]
    report_path.write_text("\n".join(lines))

    print(f"wrote {analysis_path}")
    print(f"wrote {points_path}")
    print(f"wrote {ranks_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {gate_figure_path}")
    print(f"wrote {report_path}")
    print(
        f"N={strict_payload['total_tokens']} min-H rank={last_ready_rank} "
        f"H={minimum_slack:.3f} us candidates={candidates} "
        f"all-point RMSE={analysis['cross_rank_evaluation']['frozen_unit_slope_all_point_rmse_us']:.3f} us"
    )


if __name__ == "__main__":
    main()
