#!/usr/bin/env python3
"""Legacy diagnostic of the hot-rank-only hinge on real routing imbalance."""

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


def rank_metric(record: dict[str, Any], rank: int, metric: str) -> float:
    return next(item[metric] for item in record["ranks"] if item["rank"] == rank)


def cross_rank_max(record: dict[str, Any], metric: str) -> float:
    return max(item[metric] for item in record["ranks"])


def block_medians(
    records: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record["block"])].append(value(record))
    return {
        block: float(np.median(values)) for block, values in sorted(grouped.items())
    }


def effect(
    condition: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    rng: np.random.Generator,
    bootstrap: int,
) -> dict[str, Any]:
    condition_blocks = block_medians(condition, value)
    baseline_blocks = block_medians(baseline, value)
    blocks = sorted(set(condition_blocks) & set(baseline_blocks))
    block_effects = np.asarray(
        [condition_blocks[block] - baseline_blocks[block] for block in blocks],
        dtype=np.float64,
    )
    samples = rng.choice(
        block_effects, size=(bootstrap, len(block_effects)), replace=True
    )
    estimates = np.median(samples, axis=1)
    return {
        "estimate_ms": float(np.median(block_effects)),
        "ci95_low_ms": float(np.percentile(estimates, 2.5)),
        "ci95_high_ms": float(np.percentile(estimates, 97.5)),
        "block_effects_ms": block_effects.tolist(),
        "condition_block_medians_ms": [condition_blocks[block] for block in blocks],
        "baseline_block_medians_ms": [baseline_blocks[block] for block in blocks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hinge-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    hinge_payload = json.loads(args.hinge_analysis.read_text())
    rng = np.random.default_rng(args.seed)

    records_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_condition[record["condition"]].append(record)
    conditions = sorted(payload["conditions"], key=lambda item: item["requested_r"])
    baseline_condition = conditions[0]
    if float(baseline_condition["requested_r"]) != 1.0:
        raise ValueError("the first route condition must be the r=1 baseline")
    baseline = records_by_condition[baseline_condition["name"]]
    hot_ranks = {int(condition["hot_rank"]) for condition in conditions}
    if len(hot_ranks) != 1:
        raise ValueError(f"expected one fixed hot rank, got {sorted(hot_ranks)}")
    hot_rank = hot_ranks.pop()
    hinge_key = f"original-rank{hot_rank}"
    hinge = hinge_payload["fits"]["post_expert_to_done"].get(hinge_key)
    if hinge is None or not math.isfinite(float(hinge["hinge_us"])):
        hinge_key = "original-pooled"
        hinge = hinge_payload["fits"]["post_expert_to_done"][hinge_key]
    hinge_us = float(hinge["hinge_us"])
    hinge_slope = float(hinge["slope"])
    routes = {route["route_sha256"]: route for route in payload["routes"]}

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for condition in conditions:
        name = condition["name"]
        records = records_by_condition[name]
        route_hash = payload["condition_routes"][name]
        route = routes[route_hash]
        delta = effect(
            records,
            baseline,
            lambda record: rank_metric(record, int(condition["hot_rank"]), "expert_ms"),
            rng,
            args.bootstrap,
        )
        layer = effect(
            records,
            baseline,
            lambda record: cross_rank_max(record, "layer_ms"),
            rng,
            args.bootstrap,
        )
        post = effect(
            records,
            baseline,
            lambda record: cross_rank_max(record, "post_expert_to_done_ms"),
            rng,
            args.bootstrap,
        )
        delta_us = delta["estimate_ms"] * 1000.0
        layer_us = layer["estimate_ms"] * 1000.0
        predicted_us = max(0.0, delta_us - hinge_us)
        slope_adjusted_us = hinge_slope * predicted_us
        row = {
            "condition": name,
            "requested_r": condition["requested_r"],
            "realized_raw_r": route["raw_max_mean_ratio"],
            "realized_aligned_r": route["aligned_max_mean_ratio"],
            "hot_raw_assignments": route["destination_assignment_counts"][int(condition["hot_rank"])],
            "hot_aligned_rows": route["destination_aligned_rows"][int(condition["hot_rank"])],
            "delta_hot_expert_us": delta_us,
            "delta_ci95_low_us": delta["ci95_low_ms"] * 1000.0,
            "delta_ci95_high_us": delta["ci95_high_ms"] * 1000.0,
            "layer_effect_us": layer_us,
            "layer_ci95_low_us": layer["ci95_low_ms"] * 1000.0,
            "layer_ci95_high_us": layer["ci95_high_ms"] * 1000.0,
            "post_expert_effect_us": post["estimate_ms"] * 1000.0,
            "predicted_layer_effect_us": predicted_us,
            "residual_us": layer_us - predicted_us,
            "slope_adjusted_prediction_us": slope_adjusted_us,
            "slope_adjusted_residual_us": layer_us - slope_adjusted_us,
            "correctness_max_abs": max(
                rank["correctness_max_abs"]
                for record in records
                for rank in record["ranks"]
            ),
            "route_sha256": route_hash,
        }
        rows.append(row)
        details[name] = {
            "condition": condition,
            "route": route,
            "delta_hot_expert": delta,
            "layer_effect": layer,
            "post_expert_effect": post,
            "unit_slope_prediction_us": predicted_us,
            "unit_slope_residual_us": row["residual_us"],
            "slope_adjusted_prediction_us": slope_adjusted_us,
            "slope_adjusted_residual_us": row["slope_adjusted_residual_us"],
        }

    errors = np.asarray([row["residual_us"] for row in rows], dtype=np.float64)
    adjusted_errors = np.asarray(
        [row["slope_adjusted_residual_us"] for row in rows], dtype=np.float64
    )
    evaluation = {
        "unit_slope": {
            "rmse_us": float(np.sqrt(np.mean(errors**2))),
            "mae_us": float(np.mean(np.abs(errors))),
            "max_abs_error_us": float(np.max(np.abs(errors))),
        },
        "causal_slope_adjusted": {
            "rmse_us": float(np.sqrt(np.mean(adjusted_errors**2))),
            "mae_us": float(np.mean(np.abs(adjusted_errors))),
            "max_abs_error_us": float(np.max(np.abs(adjusted_errors))),
        },
    }
    negative = [
        {
            "condition": row["condition"],
            "metric": metric,
            "estimate_us": row[field],
            "ci95_low_us": row[low],
            "ci95_high_us": row[high],
        }
        for row in rows
        for metric, field, low, high in (
            ("delta_hot_expert", "delta_hot_expert_us", "delta_ci95_low_us", "delta_ci95_high_us"),
            ("layer_effect", "layer_effect_us", "layer_ci95_low_us", "layer_ci95_high_us"),
        )
        if row[field] < 0.0
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "load_scan_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    analysis = {
        "source": str(args.input),
        "hinge_source": str(args.hinge_analysis),
        "total_tokens": payload["total_tokens"],
        "model": "layer_effect_us = max(0, delta_hot_expert_us - H_us)",
        "model_status": (
            "legacy diagnostic ablation only; this hot-rank-only model omits dispatch "
            "changes, non-hot ranks, and last-ready switching"
        ),
        "sensitivity_model": (
            "layer_effect_us = causal_slope * max(0, delta_hot_expert_us - H_us)"
        ),
        "causal_hinge_key": hinge_key,
        "causal_hinge_us": hinge_us,
        "causal_hinge_slope": hinge_slope,
        "estimator": (
            "Within each of five blocks, subtract the r=1 median from the condition "
            "median; report the median of the five block effects. δ uses hot-rank "
            "dispatch-done-to-expert-done CUDA time and ΔT uses cross-rank-max layer time."
        ),
        "negative_effect_policy": (
            "Measured negative baseline-subtracted values are retained. When a cluster "
            "interval spans zero, it is compatible with a zero effect and its sign is "
            "not identified; hinge predictions remain nonnegative by definition."
        ),
        "evaluation": evaluation,
        "negative_effects": negative,
        "rows": rows,
        "details": details,
    }
    json_path = args.output_dir / "load_scan_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    fig, axis = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    x = np.asarray([row["delta_hot_expert_us"] for row in rows])
    y = np.asarray([row["layer_effect_us"] for row in rows])
    xerr = np.vstack(
        [
            x - np.asarray([row["delta_ci95_low_us"] for row in rows]),
            np.asarray([row["delta_ci95_high_us"] for row in rows]) - x,
        ]
    )
    yerr = np.vstack(
        [
            y - np.asarray([row["layer_ci95_low_us"] for row in rows]),
            np.asarray([row["layer_ci95_high_us"] for row in rows]) - y,
        ]
    )
    axis.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3, label="measured")
    maximum = max(hinge_us * 1.5, float(max(x, default=0.0)) * 1.15)
    xx = np.linspace(min(0.0, float(min(x, default=0.0))), maximum, 300)
    yy = np.maximum(0.0, xx - hinge_us)
    axis.plot(xx, yy, "k--", label=f"(δ−{hinge_us:.1f} µs)₊")
    axis.plot(
        xx,
        hinge_slope * yy,
        color="0.4",
        linestyle=":",
        label=f"{hinge_slope:.3f}·(δ−H)₊ sensitivity",
    )
    for row in rows:
        axis.annotate(
            f"r={row['realized_raw_r']:.3f}",
            (row["delta_hot_expert_us"], row["layer_effect_us"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.axvline(0.0, color="0.6", linewidth=0.8)
    axis.set(
        xlabel="measured hot-rank expert-phase increase δ (µs)",
        ylabel="measured layer-time increase (µs)",
        title=f"Legacy hot-rank-only diagnostic, N={payload['total_tokens']}",
    )
    axis.legend()
    figure_path = args.output_dir / "delta_to_layer_time.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = args.output_dir / "REPORT.md"
    lines = [
        f"# Legacy hot-rank-only route ablation (N={payload['total_tokens']})",
        "",
        "**Status: diagnostic ablation only.** This analysis uses only the hot rank's "
        "expert interval and omits dispatch changes, non-hot ranks, baseline readiness "
        "ordering, and last-ready switching. The all-rank global-ready analysis is the "
        "primary route model.",
        "",
        f"Delay-derived hinge ({hinge_key}): **H={hinge_us:.3f} µs**, slope={hinge_slope:.3f}.",
        f"Unit-slope hinge RMSE: **{evaluation['unit_slope']['rmse_us']:.3f} µs**; "
        f"MAE: **{evaluation['unit_slope']['mae_us']:.3f} µs**.",
        f"Delay-fit slope sensitivity RMSE: "
        f"**{evaluation['causal_slope_adjusted']['rmse_us']:.3f} µs**; "
        f"MAE: **{evaluation['causal_slope_adjusted']['mae_us']:.3f} µs**.",
        "",
        "| requested r | raw r | aligned r | δ hot expert (µs) | layer Δ (µs) | predicted (µs) | residual (µs) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['requested_r']:.2f} | {row['realized_raw_r']:.6f} | "
            f"{row['realized_aligned_r']:.6f} | {row['delta_hot_expert_us']:.3f} | "
            f"{row['layer_effect_us']:.3f} | {row['predicted_layer_effect_us']:.3f} | "
            f"{row['residual_us']:.3f} |"
        )
    lines += [
        "",
        "Every token keeps exactly seven destination ranks at every r; only the second "
        "expert assignment moves. The source→destination unique-row matrix is therefore fixed.",
        "",
        "Negative baseline-subtracted measurements are retained. Values whose five-block "
        "cluster interval spans zero are compatible with a zero effect; their sign is "
        "not identified by this experiment.",
        "",
    ]
    report_path.write_text("\n".join(lines))

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")
    print(
        f"N={payload['total_tokens']} H={hinge_us:.3f} us slope={hinge_slope:.3f} "
        f"RMSE={evaluation['unit_slope']['rmse_us']:.3f} us "
        f"MAE={evaluation['unit_slope']['mae_us']:.3f} us "
        f"correctness={max(row['correctness_max_abs'] for row in rows):.6g}"
    )


if __name__ == "__main__":
    main()
