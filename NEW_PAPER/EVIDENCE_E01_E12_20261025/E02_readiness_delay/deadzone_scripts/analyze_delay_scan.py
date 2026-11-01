#!/usr/bin/env python3
"""Block-aware analysis for the causal DeepEP combine delay scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    "layer_ms",
    "post_expert_to_done_ms",
    "dispatch_ms",
    "expert_ms",
    "strict_gate_ms",
    "combine_after_gate_ms",
)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def block_bootstrap_ci(
    block_effects: list[float], rng: np.random.Generator, repetitions: int
) -> tuple[float, float]:
    values = np.asarray(block_effects, dtype=np.float64)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    return percentile_interval(np.median(samples, axis=1))


def record_cross_rank(record: dict[str, Any], metric: str) -> float:
    return max(rank[metric] for rank in record["ranks"])


def record_selected_ranks(
    record: dict[str, Any], metric: str, ranks: list[int]
) -> float:
    by_rank = {rank["rank"]: rank[metric] for rank in record["ranks"]}
    return max(by_rank[rank] for rank in ranks)


def block_medians(
    records: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]
) -> dict[int, float]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_block[record["block"]].append(value(record))
    return {
        block: float(np.median(values)) for block, values in sorted(by_block.items())
    }


def effect_summary(
    condition: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    rng: np.random.Generator,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    condition_blocks = block_medians(condition, value)
    baseline_blocks = block_medians(baseline, value)
    blocks = sorted(set(condition_blocks) & set(baseline_blocks))
    effects = [condition_blocks[block] - baseline_blocks[block] for block in blocks]
    estimate = float(np.median(effects))
    low, high = block_bootstrap_ci(effects, rng, bootstrap_repetitions)
    return {
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "block_effects": effects,
        "condition_block_medians": [condition_blocks[block] for block in blocks],
        "baseline_block_medians": [baseline_blocks[block] for block in blocks],
    }


def fit_origin_hinge(x: list[float], y: list[float]) -> dict[str, Any]:
    x_values = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    y_values = np.asarray(y, dtype=np.float64)
    if len(x_values) < 3 or float(np.max(x_values)) <= 0:
        return {"hinge_us": math.nan, "slope": math.nan, "rmse_us": math.nan}
    candidates = np.linspace(0.0, float(np.max(x_values)) * 0.999, 4001)
    best: tuple[float, float, float] | None = None
    for hinge in candidates:
        regressor = np.maximum(0.0, x_values - hinge)
        denominator = float(np.dot(regressor, regressor))
        slope = 0.0 if denominator == 0.0 else max(
            0.0, float(np.dot(regressor, y_values) / denominator)
        )
        residual = y_values - slope * regressor
        sse = float(np.dot(residual, residual))
        if best is None or sse < best[0]:
            best = (sse, hinge, slope)
    assert best is not None
    sse, hinge, slope = best
    return {
        "hinge_us": hinge,
        "slope": slope,
        "rmse_us": math.sqrt(sse / len(x_values)),
        "x_us": x_values.tolist(),
        "y_us": y_values.tolist(),
        "predicted_us": (slope * np.maximum(0.0, x_values - hinge)).tolist(),
    }


def infer_c0_us(conditions: list[dict[str, Any]]) -> float:
    estimates = [
        condition["delay_us"] / condition["delay_ratio"]
        for condition in conditions
        if condition.get("delay_ratio") not in (None, 0.0)
    ]
    if not estimates:
        raise ValueError("conditions do not carry delay_ratio metadata")
    return float(np.median(estimates))


def condition_rank_label(condition: dict[str, Any], world_size: int) -> str:
    ranks = condition["delay_ranks"]
    if condition.get("delay_ratio") == 0:
        return "zero"
    if len(ranks) == world_size:
        return "all"
    if len(ranks) == 1:
        return str(ranks[0])
    return "-".join(map(str, ranks))


def fit_rank_curves(rows: list[dict[str, Any]], effect_field: str) -> dict[str, Any]:
    fits: dict[str, Any] = {}
    for mode in ("original", "strict_barrier"):
        for rank in (0, 3, 7):
            points = [
                row
                for row in rows
                if row["mode"] == mode
                and (
                    row["rank"] == str(rank)
                    or (row["rank"] == "zero" and row["target_ratio"] == 0)
                )
            ]
            points.sort(key=lambda row: row["target_ratio"])
            fits[f"{mode}-rank{rank}"] = fit_origin_hinge(
                [max(0.0, row["actual_delay_us"]) for row in points],
                [row[effect_field] for row in points],
            )

        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for ratio in sorted(
            {
                row["target_ratio"]
                for row in rows
                if row["mode"] == mode
                and (row["rank"] in {"0", "3", "7"} or row["rank"] == "zero")
            }
        ):
            if ratio == 0:
                pooled_x.append(0.0)
                pooled_y.append(0.0)
                continue
            points = [
                row
                for row in rows
                if row["mode"] == mode
                and row["rank"] in {"0", "3", "7"}
                and row["target_ratio"] == ratio
            ]
            pooled_x.append(float(np.median([row["actual_delay_us"] for row in points])))
            pooled_y.append(float(np.median([row[effect_field] for row in points])))
        fits[f"{mode}-pooled"] = fit_origin_hinge(pooled_x, pooled_y)
    return fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_size = int(payload["world_size"])
    c0_us = infer_c0_us(payload["conditions"])
    rng = np.random.default_rng(args.seed)

    records_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_condition[record["condition"]].append(record)
    conditions = {condition["name"]: condition for condition in payload["conditions"]}
    baselines = {
        "original": records_by_condition["original-zero"],
        "strict_barrier": records_by_condition["barrier-zero"],
    }

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for name, condition in conditions.items():
        condition_records = records_by_condition[name]
        mode = condition["mode"]
        baseline_records = baselines[mode]
        ranks = condition["delay_ranks"]
        if condition.get("delay_ratio") == 0:
            # Zero is a shared virtual point for every single-rank curve.
            ranks_for_delay = [0]
        else:
            ranks_for_delay = ranks

        delay_effect = effect_summary(
            condition_records,
            baseline_records,
            lambda record, selected=ranks_for_delay: record_selected_ranks(
                record, "injected_delay_ms", selected
            ),
            rng,
            args.bootstrap,
        )
        metric_effects = {
            metric: effect_summary(
                condition_records,
                baseline_records,
                lambda record, key=metric: record_cross_rank(record, key),
                rng,
                args.bootstrap,
            )
            for metric in METRICS
        }
        correctness = max(
            rank["correctness_max_abs"]
            for record in condition_records
            for rank in record["ranks"]
        )
        ratio = condition.get("delay_ratio")
        row: dict[str, Any] = {
            "condition": name,
            "mode": mode,
            "rank": condition_rank_label(condition, world_size),
            "delay_ranks": ",".join(map(str, ranks)),
            "target_ratio": ratio,
            "target_delay_us": condition["delay_us"],
            "actual_delay_us": delay_effect["estimate"] * 1000.0,
            "actual_delay_ci95_low_us": delay_effect["ci95_low"] * 1000.0,
            "actual_delay_ci95_high_us": delay_effect["ci95_high"] * 1000.0,
            "actual_ratio": delay_effect["estimate"] * 1000.0 / c0_us,
            "correctness_max_abs": correctness,
        }
        for metric, summary in metric_effects.items():
            prefix = metric.removesuffix("_ms")
            row[f"{prefix}_effect_us"] = summary["estimate"] * 1000.0
            row[f"{prefix}_ci95_low_us"] = summary["ci95_low"] * 1000.0
            row[f"{prefix}_ci95_high_us"] = summary["ci95_high"] * 1000.0
        rows.append(row)
        detailed[name] = {
            "condition": condition,
            "delay_effect_ms": delay_effect,
            "metric_effects_ms": metric_effects,
            "correctness_max_abs": correctness,
        }

    rows.sort(
        key=lambda row: (
            row["mode"],
            row["rank"],
            -1.0 if row["target_ratio"] is None else row["target_ratio"],
        )
    )

    csv_path = args.output_dir / "delay_scan_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    layer_fits = fit_rank_curves(rows, "layer_effect_us")
    mechanism_fits = fit_rank_curves(rows, "post_expert_to_done_effect_us")

    all_rank_points = [
        row for row in rows if row["mode"] == "original" and row["rank"] == "all"
    ]
    zero_row = next(
        row
        for row in rows
        if row["mode"] == "original" and row["target_ratio"] == 0
    )
    all_rank_points = [zero_row] + sorted(
        all_rank_points, key=lambda row: row["target_ratio"]
    )
    x_all = np.asarray(
        [max(0.0, row["actual_delay_us"]) for row in all_rank_points], dtype=float
    )
    y_all = np.asarray([row["layer_effect_us"] for row in all_rank_points], dtype=float)
    all_rank_denominator = float(np.dot(x_all, x_all))
    if all_rank_denominator == 0.0:
        all_rank_slope = math.nan
        all_rank_rmse = math.nan
    else:
        all_rank_slope = float(np.dot(x_all, y_all) / all_rank_denominator)
        all_rank_rmse = float(
            np.sqrt(np.mean((y_all - all_rank_slope * x_all) ** 2))
        )
    all_rank_linear_fit = {
        "slope": all_rank_slope,
        "rmse_us": all_rank_rmse,
        "x_us": x_all.tolist(),
        "y_us": y_all.tolist(),
    }

    negative_effects = [
        {
            "condition": row["condition"],
            "layer_effect_us": row["layer_effect_us"],
            "ci95_low_us": row["layer_ci95_low_us"],
            "ci95_high_us": row["layer_ci95_high_us"],
        }
        for row in rows
        if row["layer_effect_us"] < 0.0
    ]
    analysis = {
        "source": str(args.input),
        "total_tokens": payload["total_tokens"],
        "c0_us": c0_us,
        "estimator": (
            "For each condition and block, subtract the same-block median of the "
            "matching path baseline; report the median across five block effects."
        ),
        "ci": (
            f"95% percentile interval from {args.bootstrap} cluster bootstraps of "
            "the five block effects; repetitions remain nested in their blocks."
        ),
        "negative_effect_policy": (
            "Negative baseline-subtracted effects are retained, never clipped. "
            "When an interval spans zero, it is compatible with a zero effect; "
            "the experiment does not identify the sign."
        ),
        "fits": {
            "layer_time": layer_fits,
            "post_expert_to_done": mechanism_fits,
            "original_all_ranks_layer_linear": all_rank_linear_fit,
        },
        "negative_layer_effects": negative_effects,
        "conditions": detailed,
    }
    analysis_path = args.output_dir / "delay_scan_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    colors = {0: "#1f77b4", 3: "#ff7f0e", 7: "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for axis, mode, title in (
        (axes[0, 0], "original", "Original asynchronous combine"),
        (axes[0, 1], "strict_barrier", "Strict GPU-ready barrier"),
    ):
        for rank in (0, 3, 7):
            points = [
                row
                for row in rows
                if row["mode"] == mode and row["rank"] == str(rank)
            ]
            points.sort(key=lambda row: row["target_ratio"])
            x = [0.0] + [row["actual_delay_us"] / c0_us for row in points]
            y = [0.0] + [row["layer_effect_us"] / c0_us for row in points]
            low = [0.0] + [row["layer_ci95_low_us"] / c0_us for row in points]
            high = [0.0] + [row["layer_ci95_high_us"] / c0_us for row in points]
            axis.plot(x, y, marker="o", color=colors[rank], label=f"late rank {rank}")
            axis.fill_between(x, low, high, color=colors[rank], alpha=0.12)
        fit = layer_fits[f"{mode}-pooled"]
        xx = np.linspace(0.0, 2.05, 300)
        yy = fit["slope"] * np.maximum(0.0, xx * c0_us - fit["hinge_us"]) / c0_us
        axis.plot(xx, yy, color="black", linestyle="--", label="pooled hinge fit")
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set(title=title, xlabel="actual injected Δ / C0", ylabel="layer-time change / C0")
        axis.legend(fontsize=8)

    axis = axes[1, 0]
    x = [row["actual_delay_us"] / c0_us for row in all_rank_points]
    y = [row["layer_effect_us"] / c0_us for row in all_rank_points]
    axis.plot(x, y, marker="o", color="#9467bd", label="all ranks delayed")
    xx = np.linspace(0.0, max(x), 200)
    axis.plot(xx, all_rank_slope * xx, "k--", label=f"origin slope={all_rank_slope:.2f}")
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set(
        title="All-rank delay control (original path)",
        xlabel="actual injected Δ / C0",
        ylabel="layer-time change / C0",
    )
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    validation_points = [row for row in rows if row["target_ratio"] not in (None, 0)]
    for mode, marker in (("original", "o"), ("strict_barrier", "s")):
        points = [row for row in validation_points if row["mode"] == mode and row["rank"] != "all"]
        axis.scatter(
            [row["target_delay_us"] for row in points],
            [row["actual_delay_us"] for row in points],
            s=24,
            alpha=0.7,
            marker=marker,
            label=mode,
        )
    maximum = max(row["target_delay_us"] for row in validation_points)
    axis.plot([0.0, maximum], [0.0, maximum], "k--", label="target = actual")
    axis.set(
        title="GPU delay calibration in measured iterations",
        xlabel="target differential (µs)",
        ylabel="measured differential (µs)",
    )
    axis.legend(fontsize=8)

    figure_path = args.output_dir / "delay_scan_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = args.output_dir / "REPORT.md"
    original = layer_fits["original-pooled"]
    strict = layer_fits["strict_barrier-pooled"]
    original_mechanism = mechanism_fits["original-pooled"]
    strict_mechanism = mechanism_fits["strict_barrier-pooled"]
    report_path.write_text(
        "\n".join(
            [
                f"# Delay scan analysis (N={payload['total_tokens']})",
                "",
                f"- Calibration C0: **{c0_us:.3f} µs**",
                f"- Original layer-time hinge: **H={original['hinge_us']:.3f} µs** "
                f"(**{original['hinge_us'] / c0_us:.3f} C0**), slope={original['slope']:.3f}",
                f"- Strict-barrier layer-time hinge: **H={strict['hinge_us']:.3f} µs** "
                f"(**{strict['hinge_us'] / c0_us:.3f} C0**), slope={strict['slope']:.3f}",
                f"- Original post-expert hinge (mechanism metric): "
                f"**H={original_mechanism['hinge_us']:.3f} µs** "
                f"(**{original_mechanism['hinge_us'] / c0_us:.3f} C0**), "
                f"slope={original_mechanism['slope']:.3f}",
                f"- Strict-barrier post-expert hinge (mechanism control): "
                f"**H={strict_mechanism['hinge_us']:.3f} µs** "
                f"(**{strict_mechanism['hinge_us'] / c0_us:.3f} C0**), "
                f"slope={strict_mechanism['slope']:.3f}",
                f"- All-rank control slope: **{all_rank_slope:.3f}**",
                f"- Maximum correctness error: **{max(row['correctness_max_abs'] for row in rows):.6g}**",
                "",
                "Negative baseline-subtracted effects are retained. When an interval spans zero, "
                "it is compatible with a zero effect; the experiment does not identify the sign.",
                "",
                f"Raw summary: `{csv_path.name}`",
                f"Machine-readable analysis: `{analysis_path.name}`",
                f"Figure: `{figure_path.name}`",
                "",
            ]
        )
    )
    print(f"wrote {csv_path}")
    print(f"wrote {analysis_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")
    print(
        f"N={payload['total_tokens']} C0={c0_us:.3f} us; "
        f"original H={original['hinge_us']:.3f} us slope={original['slope']:.3f}; "
        f"strict H={strict['hinge_us']:.3f} us slope={strict['slope']:.3f}; "
        f"post H original/strict={original_mechanism['hinge_us']:.3f}/"
        f"{strict_mechanism['hinge_us']:.3f} us; all-rank slope={all_rank_slope:.3f}"
    )


if __name__ == "__main__":
    main()
