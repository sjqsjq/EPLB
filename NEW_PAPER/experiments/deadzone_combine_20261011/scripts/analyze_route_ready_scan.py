#!/usr/bin/env python3
"""Analyze real routes with global-ready deltas and no second H subtraction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ValueFn = Callable[[dict[str, Any]], float]


def rank_metric(record: dict[str, Any], rank: int, metric: str) -> float:
    return next(float(item[metric]) for item in record["ranks"] if int(item["rank"]) == rank)


def rank_ready_interval(record: dict[str, Any], rank: int) -> float:
    item = next(item for item in record["ranks"] if int(item["rank"]) == rank)
    return float(item["dispatch_ms"] + item["expert_ms"] + item["injected_delay_ms"])


def cross_rank_max(record: dict[str, Any], metric: str) -> float:
    return max(float(item[metric]) for item in record["ranks"])


def strict_pre_gate(record: dict[str, Any]) -> float:
    return max(
        float(item["layer_ms"] - item["combine_after_gate_ms"])
        for item in record["ranks"]
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


def condition_map(payload: dict[str, Any]) -> dict[tuple[float, str], dict[str, Any]]:
    result = {}
    for condition in payload["conditions"]:
        key = (float(condition["requested_r"]), str(condition["mode"]))
        if key in result:
            raise ValueError(f"duplicate route/mode condition {key}")
        result[key] = condition
    return result


def reconstruction_bootstrap(
    ready_effects: list[dict[str, Any]],
    rank_summaries: list[dict[str, Any]],
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    route_blocks = np.asarray(
        [item["block_effects_ms"] for item in ready_effects], dtype=np.float64
    ) * 1000.0
    slack_blocks = np.asarray(
        [item["strict_slack_block_values_us"] for item in rank_summaries],
        dtype=np.float64,
    )
    route_indices = rng.integers(
        0, route_blocks.shape[1], size=(repetitions, route_blocks.shape[1])
    )
    slack_indices = rng.integers(
        0, slack_blocks.shape[1], size=(repetitions, slack_blocks.shape[1])
    )
    delta_samples = np.stack(
        [np.median(route_blocks[rank][route_indices], axis=1) for rank in range(8)],
        axis=1,
    )
    slack_samples = np.stack(
        [np.median(slack_blocks[rank][slack_indices], axis=1) for rank in range(8)],
        axis=1,
    )
    baseline_latest = np.max(-slack_samples, axis=1)
    changed_latest = np.max(-slack_samples + delta_samples, axis=1)
    samples = changed_latest - baseline_latest
    return tuple(float(item) for item in np.percentile(samples, [2.5, 97.5]))


def metric_error(rows: list[dict[str, Any]], prediction: str, target: str) -> dict[str, float]:
    residual = np.asarray(
        [float(row[target] - row[prediction]) for row in rows], dtype=np.float64
    )
    return {
        "rmse_us": float(np.sqrt(np.mean(residual**2))),
        "mae_us": float(np.mean(np.abs(residual))),
        "max_abs_error_us": float(np.max(np.abs(residual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--readiness-analysis", type=Path, required=True)
    parser.add_argument("--legacy-analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    readiness = json.loads(args.readiness_analysis.read_text())
    if int(payload["total_tokens"]) != int(readiness["total_tokens"]):
        raise ValueError("route and readiness token counts differ")
    if int(payload["world_size"]) != 8:
        raise ValueError("route-ready analysis requires eight ranks")

    conditions = condition_map(payload)
    ratios = sorted({ratio for ratio, _ in conditions})
    expected_modes = {"original", "strict_barrier"}
    for ratio in ratios:
        modes = {mode for candidate, mode in conditions if candidate == ratio}
        if modes != expected_modes:
            raise ValueError(f"r={ratio} modes are {sorted(modes)}")
    if ratios[0] != 1.0:
        raise ValueError("route scan must start at r=1")

    records_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_condition[record["condition"]].append(record)
    original_baseline = records_by_condition[conditions[(1.0, "original")]["name"]]
    strict_baseline = records_by_condition[
        conditions[(1.0, "strict_barrier")]["name"]
    ]

    readiness_hashes = set(readiness["route_sha256"])
    baseline_hash = payload["condition_routes"][
        conditions[(1.0, "strict_barrier")]["name"]
    ]
    if readiness_hashes != {baseline_hash}:
        raise ValueError("readiness scan and route r=1 baseline hashes differ")

    rank_summaries = sorted(readiness["rank_summaries"], key=lambda item: item["rank"])
    if [int(item["rank"]) for item in rank_summaries] != list(range(8)):
        raise ValueError("readiness analysis does not contain ranks 0..7")
    slack_us = np.asarray(
        [float(item["strict_slack_us"]) for item in rank_summaries],
        dtype=np.float64,
    )

    legacy_h = None
    legacy_key = None
    if args.legacy_analysis is not None:
        legacy = json.loads(args.legacy_analysis.read_text())
        legacy_key = "original-rank0"
        legacy_fit = legacy["fits"]["post_expert_to_done"].get(legacy_key)
        if legacy_fit is None or not math.isfinite(float(legacy_fit["hinge_us"])):
            legacy_key = "original-pooled"
            legacy_fit = legacy["fits"]["post_expert_to_done"][legacy_key]
        legacy_h = float(legacy_fit["hinge_us"])

    rng = np.random.default_rng(args.seed)
    routes = {route["route_sha256"]: route for route in payload["routes"]}
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for ratio in ratios:
        original_condition = conditions[(ratio, "original")]
        strict_condition = conditions[(ratio, "strict_barrier")]
        original_records = records_by_condition[original_condition["name"]]
        strict_records = records_by_condition[strict_condition["name"]]
        route_hash = payload["condition_routes"][original_condition["name"]]
        if route_hash != payload["condition_routes"][strict_condition["name"]]:
            raise ValueError(f"r={ratio} original and strict route hashes differ")
        route = routes[route_hash]

        original_layer = effect(
            original_records,
            original_baseline,
            lambda record: cross_rank_max(record, "layer_ms"),
            rng,
            args.bootstrap,
        )
        strict_layer = effect(
            strict_records,
            strict_baseline,
            lambda record: cross_rank_max(record, "layer_ms"),
            rng,
            args.bootstrap,
        )
        strict_combine = effect(
            strict_records,
            strict_baseline,
            lambda record: cross_rank_max(record, "combine_after_gate_ms"),
            rng,
            args.bootstrap,
        )
        direct_global_ready = effect(
            strict_records,
            strict_baseline,
            strict_pre_gate,
            rng,
            args.bootstrap,
        )
        hot_expert = effect(
            original_records,
            original_baseline,
            lambda record: rank_metric(record, 0, "expert_ms"),
            rng,
            args.bootstrap,
        )
        ready_effects = [
            effect(
                strict_records,
                strict_baseline,
                lambda record, selected=rank: rank_ready_interval(record, selected),
                rng,
                args.bootstrap,
            )
            for rank in range(8)
        ]
        rank_delta_us = np.asarray(
            [item["estimate_ms"] * 1000.0 for item in ready_effects],
            dtype=np.float64,
        )
        baseline_latest = float(np.max(-slack_us))
        reconstructed_global_ready = float(
            np.max(-slack_us + rank_delta_us) - baseline_latest
        )
        reconstructed_low, reconstructed_high = reconstruction_bootstrap(
            ready_effects, rank_summaries, rng, args.bootstrap
        )

        old_prediction = None
        if legacy_h is not None:
            old_prediction = max(
                0.0, hot_expert["estimate_ms"] * 1000.0 - legacy_h
            )

        row = {
            "requested_r": ratio,
            "realized_raw_r": route["raw_max_mean_ratio"],
            "realized_aligned_r": route["aligned_max_mean_ratio"],
            "hot_raw_assignments": route["destination_assignment_counts"][0],
            "hot_aligned_rows": route["destination_aligned_rows"][0],
            "original_layer_effect_us": original_layer["estimate_ms"] * 1000.0,
            "original_layer_ci95_low_us": original_layer["ci95_low_ms"] * 1000.0,
            "original_layer_ci95_high_us": original_layer["ci95_high_ms"] * 1000.0,
            "strict_layer_effect_us": strict_layer["estimate_ms"] * 1000.0,
            "strict_post_gate_combine_effect_us": strict_combine["estimate_ms"] * 1000.0,
            "direct_global_ready_effect_us": direct_global_ready["estimate_ms"] * 1000.0,
            "direct_global_ready_ci95_low_us": direct_global_ready["ci95_low_ms"] * 1000.0,
            "direct_global_ready_ci95_high_us": direct_global_ready["ci95_high_ms"] * 1000.0,
            "reconstructed_global_ready_effect_us": reconstructed_global_ready,
            "reconstructed_global_ready_ci95_low_us": reconstructed_low,
            "reconstructed_global_ready_ci95_high_us": reconstructed_high,
            "hot_expert_delta_us": hot_expert["estimate_ms"] * 1000.0,
            "legacy_hot_expert_minus_h_prediction_us": old_prediction,
            "route_sha256": route_hash,
            "correctness_max_abs": max(
                float(item["correctness_max_abs"])
                for record in original_records + strict_records
                for item in record["ranks"]
            ),
        }
        for rank, delta in enumerate(rank_delta_us):
            row[f"rank{rank}_ready_delta_us"] = float(delta)
        rows.append(row)
        details[f"r={ratio:g}"] = {
            "original_condition": original_condition,
            "strict_condition": strict_condition,
            "route": route,
            "original_layer_effect": original_layer,
            "strict_layer_effect": strict_layer,
            "strict_post_gate_combine_effect": strict_combine,
            "direct_global_ready_effect": direct_global_ready,
            "rank_ready_effects": ready_effects,
            "reconstructed_global_ready_effect_us": reconstructed_global_ready,
            "reconstructed_global_ready_ci95_us": [
                reconstructed_low,
                reconstructed_high,
            ],
            "hot_expert_effect": hot_expert,
            "legacy_hot_expert_minus_h_prediction_us": old_prediction,
        }

    evaluations = {
        "direct_global_ready_to_original_layer": metric_error(
            rows, "direct_global_ready_effect_us", "original_layer_effect_us"
        ),
        "reconstructed_global_ready_to_original_layer": metric_error(
            rows, "reconstructed_global_ready_effect_us", "original_layer_effect_us"
        ),
        "reconstructed_to_direct_global_ready": metric_error(
            rows,
            "reconstructed_global_ready_effect_us",
            "direct_global_ready_effect_us",
        ),
    }
    if legacy_h is not None:
        evaluations["legacy_hot_expert_minus_h_to_original_layer"] = metric_error(
            rows,
            "legacy_hot_expert_minus_h_prediction_us",
            "original_layer_effect_us",
        )

    negative = [
        {
            "requested_r": row["requested_r"],
            "metric": metric,
            "estimate_us": row[field],
            "ci95_low_us": row[low],
            "ci95_high_us": row[high],
        }
        for row in rows
        for metric, field, low, high in (
            (
                "original_layer_effect",
                "original_layer_effect_us",
                "original_layer_ci95_low_us",
                "original_layer_ci95_high_us",
            ),
            (
                "direct_global_ready_effect",
                "direct_global_ready_effect_us",
                "direct_global_ready_ci95_low_us",
                "direct_global_ready_ci95_high_us",
            ),
            (
                "reconstructed_global_ready_effect",
                "reconstructed_global_ready_effect_us",
                "reconstructed_global_ready_ci95_low_us",
                "reconstructed_global_ready_ci95_high_us",
            ),
        )
        if float(row[field]) < 0.0
    ]

    analysis = {
        "source": str(args.input),
        "readiness_source": str(args.readiness_analysis),
        "legacy_source": None if args.legacy_analysis is None else str(args.legacy_analysis),
        "total_tokens": payload["total_tokens"],
        "primary_model": (
            "layer_effect_us = delta_global_latest_ready_us; once the all-rank latest-"
            "ready change is formed, H is not subtracted again"
        ),
        "direct_global_ready_definition": (
            "same-block effect of cross-rank max(layer_ms-combine_after_gate_ms) "
            "on the strict path"
        ),
        "reconstruction_definition": (
            "max_i(-H_i + delta_R_i) - max_i(-H_i), where H_i comes only from "
            "the independent strict rank scan and delta_R_i is the same-rank route "
            "effect in dispatch_ms+expert_ms+injected_delay_ms"
        ),
        "legacy_ablation": {
            "definition": "max(0, hot_rank_expert_delta_us-H_us)",
            "hinge_key": legacy_key,
            "hinge_us": legacy_h,
            "status": (
                "diagnostic ablation only; it ignores dispatch changes, non-hot ranks, "
                "and last-ready switching"
            ),
        },
        "padding_policy": (
            "Raw r is reported only as route metadata. Compute interpretation uses "
            "per-expert 128-row-aligned route manifests and measured per-rank ready deltas; "
            "no delta=S*(r-1) conversion is used."
        ),
        "estimator": (
            "Within each block, subtract the r=1 median for the matching execution mode; "
            "report the median of five block effects and cluster-bootstrap blocks."
        ),
        "negative_effect_policy": (
            "Negative baseline-subtracted estimates are retained unchanged. An interval "
            "that spans zero is compatible with a zero effect; its sign is not identified."
        ),
        "evaluations": evaluations,
        "negative_effects": negative,
        "rows": rows,
        "details": details,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "route_ready_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    csv_path = args.output_dir / "route_ready_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure_path = args.output_dir / "global_ready_to_layer.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    x = np.asarray([row["direct_global_ready_effect_us"] for row in rows])
    y = np.asarray([row["original_layer_effect_us"] for row in rows])
    x_low = np.asarray([row["direct_global_ready_ci95_low_us"] for row in rows])
    x_high = np.asarray([row["direct_global_ready_ci95_high_us"] for row in rows])
    y_low = np.asarray([row["original_layer_ci95_low_us"] for row in rows])
    y_high = np.asarray([row["original_layer_ci95_high_us"] for row in rows])
    axes[0].errorbar(
        x,
        y,
        xerr=np.vstack([x - x_low, x_high - x]),
        yerr=np.vstack([y - y_low, y_high - y]),
        fmt="o",
        color="#2468A2",
        markersize=6,
        capsize=3,
        label="route conditions",
    )
    bound_low = min(0.0, float(np.min(x_low)), float(np.min(y_low)))
    bound_high = max(float(np.max(x_high)), float(np.max(y_high))) * 1.05
    axes[0].plot([bound_low, bound_high], [bound_low, bound_high], color="#333333", linestyle="--", linewidth=1.5, label="identity")
    annotation_offsets = {1.10: (-42, 9), 1.20: (7, -14), 1.40: (6, 5)}
    for row in rows:
        requested_r = round(float(row["requested_r"]), 2)
        if requested_r in annotation_offsets:
            axes[0].annotate(
                f"r={row['realized_raw_r']:.2f}",
                (row["direct_global_ready_effect_us"], row["original_layer_effect_us"]),
                xytext=annotation_offsets[requested_r],
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set(xlabel="direct global-ready change (µs)", ylabel="original layer-time change (µs)", title="Global readiness predicts layer time")
    axes[0].grid(color="#E6E6E6", linewidth=0.7)
    axes[0].legend(frameon=False)

    raw_r = np.asarray([row["realized_raw_r"] for row in rows])
    reconstructed = np.asarray([row["reconstructed_global_ready_effect_us"] for row in rows])
    axes[1].plot(raw_r, y, color="#2468A2", marker="o", linewidth=1.8, label="original layer")
    axes[1].plot(raw_r, x, color="#9A4F18", marker="s", linewidth=1.8, label="direct global ready")
    axes[1].plot(raw_r, reconstructed, color="#555555", marker="^", linewidth=1.8, label="per-rank reconstruction")
    axes[1].axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axes[1].set(xlabel="raw route ratio r (metadata only)", ylabel="baseline-subtracted change (µs)", title="Padding-aware measured response")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = args.output_dir / "REPORT.md"
    direct_eval = evaluations["direct_global_ready_to_original_layer"]
    reconstructed_eval = evaluations["reconstructed_global_ready_to_original_layer"]
    lines = [
        f"# Global-ready real-route analysis (N={payload['total_tokens']})",
        "",
        "The primary delta is the all-rank latest-ready change. It already includes "
        "baseline readiness slack and rank switching, so **H is not subtracted again**.",
        "",
        "| requested r | raw r | aligned r | direct global ready Δ (µs) | reconstructed global ready Δ (µs) | original layer Δ (µs) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['requested_r']:.2f} | {row['realized_raw_r']:.6f} | "
            f"{row['realized_aligned_r']:.6f} | {row['direct_global_ready_effect_us']:.3f} | "
            f"{row['reconstructed_global_ready_effect_us']:.3f} | "
            f"{row['original_layer_effect_us']:.3f} |"
        )
    lines += [
        "",
        f"Direct global-ready → original-layer RMSE: **{direct_eval['rmse_us']:.3f} µs** "
        f"(MAE {direct_eval['mae_us']:.3f} µs).",
        f"Per-rank reconstruction → original-layer RMSE: "
        f"**{reconstructed_eval['rmse_us']:.3f} µs** "
        f"(MAE {reconstructed_eval['mae_us']:.3f} µs).",
        "",
        "Raw r is not converted to time. The route manifest's per-expert 128-row "
        "alignment and measured per-rank ready changes are used, preserving the "
        "DeepGEMM staircase.",
        "",
        "Negative baseline-subtracted estimates are retained unchanged. Whenever an "
        "interval spans zero, the estimate is compatible with a zero effect and the "
        "experiment does not identify its sign.",
        "",
    ]
    if legacy_h is not None:
        legacy_eval = evaluations["legacy_hot_expert_minus_h_to_original_layer"]
        lines += [
            f"Legacy hot-expert-minus-H ablation ({legacy_key}, H={legacy_h:.3f} µs) "
            f"has RMSE **{legacy_eval['rmse_us']:.3f} µs**. It is retained only as a "
            "labeled ablation because it omits dispatch changes, other ranks, and "
            "last-ready switching.",
            "",
        ]
    report_path.write_text("\n".join(lines))

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")
    print(
        f"N={payload['total_tokens']} direct-ready RMSE={direct_eval['rmse_us']:.3f} us "
        f"reconstructed RMSE={reconstructed_eval['rmse_us']:.3f} us"
    )


if __name__ == "__main__":
    main()
