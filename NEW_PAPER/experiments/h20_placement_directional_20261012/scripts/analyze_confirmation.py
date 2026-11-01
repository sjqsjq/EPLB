#!/usr/bin/env python3
"""Validate and analyze the four-start H20 placement confirmation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
from typing import Any, Iterable

from scipy import stats

T_CRIT_95_DF3 = 3.182446305284263


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--csv-output", type=pathlib.Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: Iterable[float], *, label: str) -> dict[str, Any]:
    samples = [float(value) for value in values]
    require(len(samples) == 4, f"{label}: expected four process-start values")
    mean = statistics.fmean(samples)
    standard_deviation = statistics.stdev(samples)
    standard_error = standard_deviation / math.sqrt(4)
    half_width = T_CRIT_95_DF3 * standard_error
    if standard_error == 0.0:
        p_two_sided = 0.0 if mean != 0.0 else 1.0
    else:
        p_two_sided = float(2.0 * stats.t.sf(abs(mean / standard_error), df=3))
    return {
        "n_independent_process_starts": 4,
        "values_by_start": samples,
        "mean_us": mean,
        "median_us": statistics.median(samples),
        "standard_deviation_us": standard_deviation,
        "standard_error_us": standard_error,
        "minimum_us": min(samples),
        "maximum_us": max(samples),
        "ci_level": 0.95,
        "degrees_of_freedom": 3,
        "ci_method": (
            "two-sided Student-t interval over four independent process-start means"
        ),
        "ci_low_us": mean - half_width,
        "ci_high_us": mean + half_width,
        "p_two_sided_unadjusted": p_two_sided,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return adjusted


def classify_interval(record: dict[str, Any], epsilon_us: float) -> str:
    low = record["ci_low_us"]
    high = record["ci_high_us"]
    if low > epsilon_us:
        return "materially_faster"
    if high < -epsilon_us:
        return "materially_slower"
    if low >= -epsilon_us and high <= epsilon_us:
        return "practically_equivalent_within_frozen_margin"
    if low <= 0.0 <= high:
        return "compatible_with_zero_not_equivalent"
    return "directional_but_practical_margin_unresolved"


def flatten_rank_records(value: Any, *, label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list) and len(value) == 8, f"{label}: expected 8 ranks")
    records: list[dict[str, Any]] = []
    for rank, rank_records in enumerate(value):
        require(isinstance(rank_records, list), f"{label}: rank {rank} is not a list")
        require(
            all(record["rank"] == rank for record in rank_records),
            f"{label}: rank labels differ at rank {rank}",
        )
        records.extend(rank_records)
    return records


def validate_run(
    raw: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    start: dict[str, Any],
) -> None:
    confirmation = protocol["confirmation"]
    layout_ids = confirmation["layout_ids"]
    require(raw["schema_version"] == 2, "confirmation raw schema changed")
    require(raw["phase"] == "confirm", "raw phase is not confirm")
    require(raw["start_id"] == start["start_id"], "start ID differs")
    require(raw["selected_layouts"] == layout_ids, "layout order differs")
    require(
        raw["placement_manifest_sha256"]
        == confirmation["placement_manifest_sha256"],
        "placement manifest hash differs",
    )
    followup = raw["followup_protocol"]
    require(followup["protocol_id"] == protocol["protocol_id"], "protocol ID differs")
    require(followup["sha256"] == protocol_sha256, "protocol hash differs")
    require(
        followup["replay_placement_sha256"]
        == protocol["implementation"]["replay_placement_sha256"],
        "replay source hash differs",
    )
    require(
        followup["tolerance_file_sha256"]
        == confirmation["tolerance_file_sha256"],
        "tolerance hash differs",
    )

    inventory = raw["gpu_inventory"]
    require(len(inventory) == 8, "GPU inventory does not contain eight ranks")
    require(
        [item["rank"] for item in inventory] == list(range(8)),
        "GPU rank inventory differs",
    )
    require(
        all(item["name"] == protocol["scope"]["accelerator_model"] for item in inventory),
        "non-H20 accelerator in run",
    )
    require(len({item["pid"] for item in inventory}) == 8, "rank PIDs are not unique")

    expected_snapshots = {item["rank"]: item for item in confirmation["snapshot_files"]}
    require(len(raw["snapshot_metadata"]) == 8, "snapshot metadata rank count differs")
    for item in raw["snapshot_metadata"]:
        expected = expected_snapshots[item["rank"]]
        require(item["file_sha256"] == expected["sha256"], "snapshot hash differs")
        require(item["bytes"] == expected["bytes"], "snapshot byte count differs")

    execution = raw["execution"]
    require(execution["deep_ep_mode"] == "normal", "DeepEP mode is not normal")
    require("DeepGemm" in execution["runner"], "runner is not DeepGEMM")
    audit = raw["intervention_audit"]
    require(audit["only_logical_to_physical_placement_changes"], "intervention changed")
    require(audit["logical_routes_and_weights_frozen"], "logical routes not frozen")
    require(audit["weight_migration_before_warmup_and_timing"], "migration timing changed")
    require(audit["jit_and_initial_warmup_before_timing"], "initial warmup missing")
    require(
        audit["identical_warmup_after_every_timed_activation"],
        "per-activation warmup missing",
    )
    require(audit["common_delay_floor_us"] is None, "common delay floor present")
    require(audit["differential_delay_us"] is None, "differential delay present")
    require(
        not audit["mid_layer_collective_between_expert_and_combine"],
        "mid-layer gate present",
    )
    require(not any(audit["source_scan"].values()), "run_once source audit failed")

    measurement = raw["measurement"]
    expected_measurement = {
        "initial_warmup_forwards_per_layout": confirmation[
            "initial_warmup_forwards_per_layout"
        ],
        "activation_warmup_forwards_per_condition": confirmation[
            "activation_warmup_forwards_per_condition"
        ],
        "blocks": confirmation["blocks_per_start"],
        "iterations_per_block": confirmation["timed_forwards_per_condition"],
        "random_seed": start["random_seed"],
    }
    for key, expected in expected_measurement.items():
        require(measurement[key] == expected, f"measurement {key} differs")
    require(raw["condition_schedule"] == start["schedules"], "schedule differs")

    initial_warmup = flatten_rank_records(
        raw["warmup_aligned_count_checks"], label="initial warmup"
    )
    require(len(initial_warmup) == 8 * 13, "initial warmup record count differs")
    for rank in range(8):
        ids = [record["layout_id"] for record in initial_warmup if record["rank"] == rank]
        require(ids == layout_ids, f"rank {rank} initial warmup layout order differs")
    for record in initial_warmup:
        require(
            record["actual_aligned_counts"] == record["expected_aligned_counts"],
            "initial warmup aligned counts differ",
        )

    activation = flatten_rank_records(
        raw["activation_warmup_aligned_count_checks"], label="activation warmup"
    )
    expected_activation_records = 8 * confirmation["blocks_per_start"] * 13
    require(
        len(activation) == expected_activation_records,
        "activation warmup record count differs",
    )
    for record in activation:
        require(
            record["forwards"]
            == confirmation["activation_warmup_forwards_per_condition"],
            "activation warmup count differs",
        )
        require(
            record["actual_aligned_counts"] == record["expected_aligned_counts"],
            "activation warmup aligned counts differ",
        )
    for rank in range(8):
        for block in range(confirmation["blocks_per_start"]):
            records = [
                record
                for record in activation
                if record["rank"] == rank and record["block"] == block
            ]
            records.sort(key=lambda record: record["condition_order"])
            require(
                [f"{record['layout_id']}:minimal" for record in records]
                == start["schedules"][block],
                f"rank {rank} block {block} activation order differs",
            )

    correctness = raw["correctness"]["records"]
    require(len(correctness) == 8 * 13, "correctness record count differs")
    require(
        {(record["rank"], record["layout_id"]) for record in correctness}
        == {(rank, layout_id) for rank in range(8) for layout_id in layout_ids},
        "correctness rank/layout coverage differs",
    )
    for record in correctness:
        for comparison in ("versus_captured_output", "versus_identity_replay"):
            metrics = record[comparison]
            require(metrics["nonfinite_pair_count"] == 0, "nonfinite correctness output")
            require(metrics["isclose_failure_count"] == 0, "correctness tolerance failure")
            require(metrics["atol"] == confirmation["atol"], "atol changed")
            require(metrics["rtol"] == confirmation["rtol"], "rtol changed")

    timing = raw["timing_records"]
    require(len(timing) == 5 * 13, "timing record count differs")
    for block in range(5):
        records = [record for record in timing if record["block"] == block]
        records.sort(key=lambda record: record["condition_order"])
        require(
            [f"{record['layout_id']}:{record['observation_mode']}" for record in records]
            == start["schedules"][block],
            f"timing order differs in block {block}",
        )
        require(all(record["iterations"] == 50 for record in records), "iterations changed")
        require(
            all(record["completion_fenced_us_per_forward"] > 0 for record in records),
            "nonpositive completion timing",
        )
    require(
        all(
            item["initial"] == item["final"]
            for item in raw["frozen_tensor_hashes"]
        ),
        "frozen tensors changed during replay",
    )


def paired_values(
    raw: dict[str, Any], layout_id: str, field: str
) -> tuple[list[float], float, float]:
    records = raw["timing_records"]
    by_key = {(record["block"], record["layout_id"]): record for record in records}
    blocks = raw["measurement"]["blocks"]
    effects = [
        by_key[(block, "identity")][field] - by_key[(block, layout_id)][field]
        for block in range(blocks)
    ]
    layout_values = [by_key[(block, layout_id)][field] for block in range(blocks)]
    identity_values = [by_key[(block, "identity")][field] for block in range(blocks)]
    return effects, statistics.fmean(layout_values), statistics.fmean(identity_values)


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = (args.protocol or root / "configs" / "followup_v1" / "protocol.json").resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol_sha256 = sha256_file(protocol_path)
    require(protocol["status"] == "frozen_before_confirmation_timing", "protocol not frozen")
    require(
        sha256_file(pathlib.Path(__file__).resolve())
        == protocol["implementation"]["analyze_confirmation_sha256"],
        "analyzer source differs from frozen protocol",
    )
    confirmation = protocol["confirmation"]
    starts = confirmation["starts"]
    require(len(starts) == 4, "protocol must contain four starts")
    require(len({start["random_seed"] for start in starts}) == 4, "seeds not distinct")

    runs: list[dict[str, Any]] = []
    raw_files = []
    for start in starts:
        path = (root / start["output_relative_path"]).resolve()
        failure_path = path.with_suffix(".failure.json")
        require(not failure_path.exists(), f"recorded failed start exists: {failure_path}")
        require(path.exists(), f"missing confirmation result: {path}")
        raw = json.loads(path.read_text())
        validate_run(
            raw,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            start=start,
        )
        runs.append(raw)
        raw_files.append(
            {
                "start_id": start["start_id"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    rank_pid_sets = [{item["pid"] for item in raw["gpu_inventory"]} for raw in runs]
    for left in range(4):
        for right in range(left + 1, 4):
            require(
                rank_pid_sets[left].isdisjoint(rank_pid_sets[right]),
                "rank PID reused across starts; independent-start audit failed",
            )
    intervals = [
        (
            dt.datetime.fromisoformat(raw["started_at_utc"]),
            dt.datetime.fromisoformat(raw["finished_at_utc"]),
        )
        for raw in runs
    ]
    require(
        all(intervals[index][1] <= intervals[index + 1][0] for index in range(3)),
        "confirmation starts overlap or are out of order",
    )
    baseline_hashes = [item["initial"] for item in runs[0]["frozen_tensor_hashes"]]
    require(
        all(
            [item["initial"] for item in raw["frozen_tensor_hashes"]]
            == baseline_hashes
            for raw in runs[1:]
        ),
        "frozen tensor hashes differ across starts",
    )

    manifest_path = pathlib.Path(runs[0]["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest_layouts = {item["layout_id"]: item for item in manifest["layouts"]}
    epsilon_us = confirmation["practical_equivalence_margin_us"]
    effects = []
    p_values: dict[str, float] = {}
    for layout_id in confirmation["layout_ids"]:
        if layout_id == "identity":
            continue
        completion_start_values = []
        proxy_start_values = []
        completion_blocks_by_start = []
        proxy_blocks_by_start = []
        absolute_by_start = []
        identity_by_start = []
        percent_by_start = []
        for raw in runs:
            block_effects, absolute, identity = paired_values(
                raw, layout_id, "completion_fenced_us_per_forward"
            )
            proxy_effects, _, _ = paired_values(raw, layout_id, "t_proxy_us_per_forward")
            completion_blocks_by_start.append(block_effects)
            proxy_blocks_by_start.append(proxy_effects)
            completion_start_values.append(statistics.fmean(block_effects))
            proxy_start_values.append(statistics.fmean(proxy_effects))
            absolute_by_start.append(absolute)
            identity_by_start.append(identity)
            percent_by_start.append(
                statistics.fmean(
                    effect / (effect + layout_time) * 100.0
                    for effect, layout_time in zip(
                        block_effects,
                        [
                            record["completion_fenced_us_per_forward"]
                            for record in sorted(
                                (
                                    record
                                    for record in raw["timing_records"]
                                    if record["layout_id"] == layout_id
                                ),
                                key=lambda record: record["block"],
                            )
                        ],
                    )
                )
            )
        completion = mean_ci(completion_start_values, label=layout_id)
        proxy = mean_ci(proxy_start_values, label=f"{layout_id} proxy")
        completion["classification_using_frozen_epsilon"] = classify_interval(
            completion, epsilon_us
        )
        p_values[layout_id] = completion["p_two_sided_unadjusted"]
        raw_layout = manifest_layouts[layout_id]
        effects.append(
            {
                "layout_id": layout_id,
                "category": raw_layout["category"],
                "candidate_id": raw_layout["candidate_id"],
                "swap": raw_layout["swap"],
                "route_shape_metadata": {
                    key: raw_layout["metrics"][key]
                    for key in (
                        "rank_raw_assignments",
                        "rank_aligned_rows",
                        "raw_max_assignments",
                        "aligned_max_rows",
                        "raw_r_max_over_mean",
                        "aligned_r_max_over_mean",
                    )
                },
                "completion_fenced_q_identity_minus_layout": completion,
                "t_proxy_q_identity_minus_layout": proxy,
                "all_20_paired_block_effects_us": [
                    value for values in completion_blocks_by_start for value in values
                ],
                "paired_block_effects_by_start_us": completion_blocks_by_start,
                "t_proxy_paired_block_effects_by_start_us": proxy_blocks_by_start,
                "absolute_completion_mean_by_start_us": absolute_by_start,
                "identity_completion_mean_by_start_us": identity_by_start,
                "percent_effect_by_start": percent_by_start,
            }
        )
    holm = holm_adjust(p_values)
    for effect in effects:
        effect["completion_fenced_q_identity_minus_layout"][
            "p_two_sided_holm_adjusted_across_12_layouts"
        ] = holm[effect["layout_id"]]

    a2_a3_by_start = []
    a2_a3_blocks = []
    for raw in runs:
        by_key = {
            (record["block"], record["layout_id"]): record
            for record in raw["timing_records"]
        }
        values = [
            by_key[(block, "A2")]["completion_fenced_us_per_forward"]
            - by_key[(block, "A3")]["completion_fenced_us_per_forward"]
            for block in range(5)
        ]
        a2_a3_blocks.append(values)
        a2_a3_by_start.append(statistics.fmean(values))
    matched_contrast = mean_ci(a2_a3_by_start, label="Q(A3)-Q(A2)")
    matched_contrast["definition"] = "Q(A3)-Q(A2) = T(A2)-T(A3)"
    matched_contrast["paired_block_effects_by_start_us"] = a2_a3_blocks
    matched_contrast["classification_using_frozen_epsilon"] = classify_interval(
        matched_contrast, epsilon_us
    )

    identity_start_means = []
    for raw in runs:
        values = [
            record["completion_fenced_us_per_forward"]
            for record in raw["timing_records"]
            if record["layout_id"] == "identity"
        ]
        identity_start_means.append(statistics.fmean(values))
    identity = mean_ci(identity_start_means, label="identity timing")

    summary = {
        "schema_version": 1,
        "experiment": "h20_directional_expert_placement_confirmation",
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "protocol_id": protocol["protocol_id"],
            "pilot_base_commit": protocol["pilot_base_commit"],
        },
        "scope": protocol["scope"],
        "design": {
            "independent_process_starts": 4,
            "blocks_per_start": 5,
            "timed_forwards_per_layout_per_block": 50,
            "activation_warmup_forwards_per_condition": 10,
            "layout_count": 13,
            "inferential_unit": "independent process start",
            "primary_ci": "two-sided Student-t with n=4 and df=3",
            "block_values_are_descriptive_not_independent_starts": True,
            "best_start_or_repeat_selection": False,
        },
        "raw_files": raw_files,
        "identity_completion_fenced_timing": identity,
        "practical_equivalence_margin": {
            "epsilon_us": epsilon_us,
            "status": "frozen before confirmation timing",
            "rule": (
                "Only an interval wholly inside [-epsilon,+epsilon] supports "
                "practical equivalence; crossing zero alone does not."
            ),
        },
        "effects": effects,
        "matched_aggregate_load_contrast": matched_contrast,
        "multiplicity": {
            "family": "all 12 layout-versus-identity contrasts",
            "method": "Holm adjustment of two-sided t-test p-values",
            "estimates_and_unadjusted_95pct_intervals_retained": True,
        },
        "interpretation_limits": protocol["interpretation_limits"],
    }

    output = (args.output or root / "results" / "confirmatory_v1" / "summary.json").resolve()
    csv_output = (
        args.csv_output or root / "results" / "confirmatory_v1" / "layout_effects.csv"
    ).resolve()
    write_json_atomic(output, summary)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layout_id",
                "category",
                "mean_us",
                "ci_low_us",
                "ci_high_us",
                "p_two_sided_unadjusted",
                "p_two_sided_holm_adjusted",
                "classification_using_frozen_epsilon",
                "start_01_us",
                "start_02_us",
                "start_03_us",
                "start_04_us",
            ],
        )
        writer.writeheader()
        for effect in effects:
            estimate = effect["completion_fenced_q_identity_minus_layout"]
            writer.writerow(
                {
                    "layout_id": effect["layout_id"],
                    "category": effect["category"],
                    "mean_us": estimate["mean_us"],
                    "ci_low_us": estimate["ci_low_us"],
                    "ci_high_us": estimate["ci_high_us"],
                    "p_two_sided_unadjusted": estimate["p_two_sided_unadjusted"],
                    "p_two_sided_holm_adjusted": estimate[
                        "p_two_sided_holm_adjusted_across_12_layouts"
                    ],
                    "classification_using_frozen_epsilon": estimate[
                        "classification_using_frozen_epsilon"
                    ],
                    **{
                        f"start_{index + 1:02d}_us": value
                        for index, value in enumerate(estimate["values_by_start"])
                    },
                }
            )
    print(f"CONFIRMATION_SUMMARY={output}")
    print(f"CONFIRMATION_CSV={csv_output}")


if __name__ == "__main__":
    main()
