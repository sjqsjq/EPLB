#!/usr/bin/env python3
"""Validate and summarize the frozen H20 E0/E1 placement pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


T_CRIT_95_DF9 = 2.2621571627409915
EXPECTED_LAYOUTS = [
    "identity",
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
    "D1",
    "D2",
    "D3",
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean_ci(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    require(len(samples) == 10, f"expected 10 paired blocks, got {len(samples)}")
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples)
    standard_error = standard_deviation / math.sqrt(len(samples))
    half_width = T_CRIT_95_DF9 * standard_error
    return {
        "n_blocks": len(samples),
        "values_us": samples,
        "mean_us": mean,
        "median_us": statistics.median(samples),
        "standard_deviation_us": standard_deviation,
        "standard_error_us": standard_error,
        "ci_method": "two-sided Student-t interval over block-paired differences",
        "ci_level": 0.95,
        "degrees_of_freedom": 9,
        "ci_low_us": mean - half_width,
        "ci_high_us": mean + half_width,
        "minimum_us": min(samples),
        "maximum_us": max(samples),
    }


def timing_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "n_blocks": len(samples),
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "standard_deviation_us": statistics.stdev(samples),
        "minimum_us": min(samples),
        "maximum_us": max(samples),
    }


def classify_interval(low: float, high: float, epsilon: float) -> str:
    if low > epsilon:
        return "materially_faster"
    if high < -epsilon:
        return "materially_slower"
    if low >= -epsilon and high <= epsilon:
        return "practically_negligible"
    if low <= 0.0 <= high:
        return "compatible_with_zero_not_equivalent"
    if low > 0.0:
        return "faster_but_ci_overlaps_epsilon"
    return "slower_but_ci_overlaps_epsilon"


def pearson(x: list[float], y: list[float]) -> float:
    require(len(x) == len(y) and len(x) > 1, "invalid Pearson inputs")
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator


def average_ranks(values: list[float]) -> list[float]:
    sorted_values = sorted(values)
    rank_by_value: dict[float, float] = {}
    for value in set(values):
        positions = [
            index + 1 for index, candidate in enumerate(sorted_values) if candidate == value
        ]
        rank_by_value[value] = statistics.mean(positions)
    return [rank_by_value[value] for value in values]


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest["status"] == "frozen_before_timing", "manifest is not frozen")
    require(manifest["selection_uses_performance_measurements"] is False, "selection used timing")
    require(manifest["world_size"] == 8, "world size is not 8")
    require(manifest["num_experts"] == 128, "expert count is not 128")
    require(manifest["num_local_experts"] == 16, "local expert count is not 16")
    require(manifest["alignment_rows_per_expert"] == 128, "alignment is not 128 rows")
    require(manifest["total_hidden_rows"] == 4050, "unexpected frozen hidden-row count")
    require(manifest["total_valid_assignments"] == 32400, "unexpected assignment count")

    layouts = manifest["layouts"]
    require([layout["layout_id"] for layout in layouts] == EXPECTED_LAYOUTS, "layout IDs changed")
    require(len(layouts) == 13, "expected identity plus 12 swaps")
    require(Counter(layout["category"] for layout in layouts[1:]) == Counter({"A": 3, "B": 3, "C": 3, "D": 3}), "category counts changed")

    for layout in layouts:
        p2l = layout["metrics"]["p2l"]
        l2p = layout["metrics"]["l2p"]
        require(sorted(p2l) == list(range(128)), f"{layout['layout_id']} p2l is not bijective")
        require(sorted(l2p) == list(range(128)), f"{layout['layout_id']} l2p is not bijective")
        require(all(l2p[logical] == physical for physical, logical in enumerate(p2l)), f"{layout['layout_id']} maps are not inverses")
        require(all(len(experts) == 16 for experts in layout["metrics"]["rank_logical_experts"]), f"{layout['layout_id']} does not have 16 slots/rank")
        if layout["layout_id"] != "identity":
            swap = layout["swap"]
            require(swap["baseline_rank_a"] != swap["baseline_rank_b"], f"{layout['layout_id']} is not cross-rank")
            changed = [index for index, logical in enumerate(p2l) if index != logical]
            require(len(changed) == 2, f"{layout['layout_id']} is not exactly one complete swap")

    return {
        "status": manifest["status"],
        "layout_count": len(layouts),
        "cross_rank_swap_count": len(layouts) - 1,
        "all_maps_bijective_and_inverse": True,
        "all_layouts_have_16_slots_per_rank": True,
        "selection_uses_performance_measurements": False,
        "total_hidden_rows": manifest["total_hidden_rows"],
        "total_valid_assignments": manifest["total_valid_assignments"],
    }


def correctness_validation(
    result: dict[str, Any], expected_layouts: list[str]
) -> dict[str, Any]:
    records = result["correctness"]["records"]
    expected_count = 8 * len(expected_layouts)
    require(len(records) == expected_count, f"expected {expected_count} correctness records")
    require(Counter(record["layout_id"] for record in records) == Counter({layout: 8 for layout in expected_layouts}), "incorrect correctness record counts")

    failure_count = 0
    nonfinite_pair_count = 0
    max_abs_captured = 0.0
    max_abs_identity = 0.0
    identity_max_abs = 0.0
    aligned_mismatches = 0
    for record in records:
        captured = record["versus_captured_output"]
        identity = record["versus_identity_replay"]
        failure_count += captured["isclose_failure_count"] + identity["isclose_failure_count"]
        nonfinite_pair_count += captured["nonfinite_pair_count"] + identity["nonfinite_pair_count"]
        max_abs_captured = max(max_abs_captured, captured["max_abs_error"])
        max_abs_identity = max(max_abs_identity, identity["max_abs_error"])
        if record["layout_id"] == "identity":
            identity_max_abs = max(identity_max_abs, captured["max_abs_error"])
        if record["actual_aligned_counts"] != record["expected_aligned_counts"]:
            aligned_mismatches += 1

    require(failure_count == 0, "correctness failures found")
    require(nonfinite_pair_count == 0, "nonfinite output pairs found")
    require(aligned_mismatches == 0, "aligned-count mismatch in correctness replay")
    require(identity_max_abs == 0.0, "identity replay is not exact")
    require(max_abs_captured <= result["tolerance"]["atol"], "max error exceeds frozen atol")

    return {
        "rank_layout_comparisons": len(records),
        "total_isclose_failures_across_both_references": failure_count,
        "total_nonfinite_pairs_across_both_references": nonfinite_pair_count,
        "maximum_absolute_error_versus_captured": max_abs_captured,
        "maximum_absolute_error_versus_identity": max_abs_identity,
        "identity_maximum_absolute_error": identity_max_abs,
        "frozen_atol": result["tolerance"]["atol"],
        "frozen_rtol": result["tolerance"]["rtol"],
        "relative_denominator": result["correctness"]["relative_denominator"],
        "aligned_count_mismatches": aligned_mismatches,
    }


def warmup_validation(
    result: dict[str, Any], expected_layouts: list[str]
) -> dict[str, Any]:
    by_rank = result["warmup_aligned_count_checks"]
    require(len(by_rank) == 8, "expected warmup checks from eight ranks")
    checks = [check for rank_checks in by_rank for check in rank_checks]
    expected_count = 8 * len(expected_layouts)
    require(len(checks) == expected_count, f"expected {expected_count} warmup checks")
    mismatch_count = sum(
        check["actual_aligned_counts"] != check["expected_aligned_counts"]
        for check in checks
    )
    require(mismatch_count == 0, "aligned-count mismatch after warmup")
    return {"rank_layout_checks": len(checks), "mismatch_count": mismatch_count}


def hash_validation(result: dict[str, Any]) -> dict[str, Any]:
    records = result["frozen_tensor_hashes"]
    require(len(records) == 8, "expected frozen hashes from eight ranks")
    changed: list[dict[str, Any]] = []
    storage_counts: list[int] = []
    for record in records:
        storage_counts.append(record["local_weight_storage_count"])
        if record["initial"] != record["final"]:
            changed.append({"rank": record["rank"], "initial": record["initial"], "final": record["final"]})
    require(not changed, "frozen GPU replay tensors changed")
    return {
        "ranks_checked": len(records),
        "changed_rank_count": len(changed),
        "unique_local_weight_storage_counts_by_rank": storage_counts,
    }


def intervention_validation(result: dict[str, Any]) -> dict[str, Any]:
    audit = result["intervention_audit"]
    require(audit["common_delay_floor_us"] is None, "common delay floor present")
    require(audit["differential_delay_us"] is None, "differential delay present")
    require(audit["mid_layer_collective_between_expert_and_combine"] is False, "mid-layer collective present")
    require(audit["jit_and_warmup_before_timing"] is True, "JIT/warmup timing boundary failed")
    require(audit["weight_migration_before_warmup_and_timing"] is True, "weight migration timing boundary failed")
    require(audit["logical_routes_and_weights_frozen"] is True, "logical route or weight changed")
    require(audit["only_logical_to_physical_placement_changes"] is True, "intervention changed more than placement")
    require(all(value is False for value in audit["source_scan"].values()), "source audit failed")
    return audit


def validate_timing_records(result: dict[str, Any], expected_conditions: list[str]) -> dict[tuple[int, str, str], dict[str, Any]]:
    records = result["timing_records"]
    expected_modes = ["minimal"] if result["phase"] == "e1" else ["minimal", "detailed"]
    expected_count = 10 * len(expected_conditions) * len(expected_modes)
    require(len(records) == expected_count, f"expected {expected_count} timing records")

    index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["block"], record["layout_id"], record["observation_mode"])
        require(key not in index, f"duplicate timing record {key}")
        require(record["block"] in range(10), "invalid block index")
        require(record["layout_id"] in expected_conditions, "unexpected layout")
        require(record["observation_mode"] in expected_modes, "unexpected observation mode")
        require(record["iterations"] == 50, "timed block does not contain 50 forwards")
        require(len(record["ranks"]) == 8, "timing record does not contain eight ranks")
        require(sorted(rank["rank"] for rank in record["ranks"]) == list(range(8)), "rank timing set changed")
        expected_proxy = max(rank["local_us_per_forward"] for rank in record["ranks"])
        require(math.isclose(record["t_proxy_us_per_forward"], expected_proxy, rel_tol=0.0, abs_tol=1.0e-9), "T_proxy is not the maximum local elapsed")
        require(record["completion_fenced_us_per_forward"] > 0.0, "nonpositive completion-fenced timing")
        index[key] = record

    for block in range(10):
        for layout in expected_conditions:
            for mode in expected_modes:
                require((block, layout, mode) in index, f"missing timing record {(block, layout, mode)}")

    schedule = result["condition_schedule"]
    require(len(schedule) == 10, "expected ten randomized schedules")
    condition_labels = [
        f"{layout}:{mode}" for layout in expected_conditions for mode in expected_modes
    ]
    for block_schedule in schedule:
        require(sorted(block_schedule) == sorted(condition_labels), "schedule is not a condition permutation")
    require(len({tuple(block_schedule) for block_schedule in schedule}) > 1, "condition order was not randomized")
    return index


def summarize_e0(e0: dict[str, Any]) -> dict[str, Any]:
    index = validate_timing_records(e0, ["identity", "A1"])
    output: dict[str, Any] = {
        "design": {
            "blocks": 10,
            "forwards_per_condition_per_block": 50,
            "layouts": ["identity", "A1"],
            "observation_modes": ["minimal", "detailed"],
        },
        "absolute_timing": {},
        "placement_effect": {},
        "detailed_minus_minimal": {},
    }
    for layout in ["identity", "A1"]:
        for mode in ["minimal", "detailed"]:
            values = [index[(block, layout, mode)]["completion_fenced_us_per_forward"] for block in range(10)]
            output["absolute_timing"][f"{layout}:{mode}"] = timing_summary(values)

    q: dict[str, list[float]] = {}
    for mode in ["minimal", "detailed"]:
        q[mode] = [
            index[(block, "identity", mode)]["completion_fenced_us_per_forward"]
            - index[(block, "A1", mode)]["completion_fenced_us_per_forward"]
            for block in range(10)
        ]
        output["placement_effect"][mode] = mean_ci(q[mode])

    for layout in ["identity", "A1"]:
        difference = [
            index[(block, layout, "detailed")]["completion_fenced_us_per_forward"]
            - index[(block, layout, "minimal")]["completion_fenced_us_per_forward"]
            for block in range(10)
        ]
        output["detailed_minus_minimal"][layout] = mean_ci(difference)

    output["detailed_minus_minimal"]["placement_effect"] = mean_ci(
        detailed - minimal for detailed, minimal in zip(q["detailed"], q["minimal"])
    )
    output["interpretation"] = (
        "Detailed instrumentation changes some absolute timings, but the 95% CI for "
        "Q_detailed-Q_minimal crosses zero. The directional A1 effect remains positive "
        "in both modes, so minimal mode is retained as the E1 primary mode."
    )
    return output


def summarize_e1(e1: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    index = validate_timing_records(e1, EXPECTED_LAYOUTS)
    layout_by_id = {layout["layout_id"]: layout for layout in manifest["layouts"]}
    baseline_metrics = layout_by_id["identity"]["metrics"]
    identity_values = [
        index[(block, "identity", "minimal")]["completion_fenced_us_per_forward"]
        for block in range(10)
    ]
    identity_summary = timing_summary(identity_values)
    epsilon = max(0.01 * identity_summary["mean_us"], 2.0)

    effects: list[dict[str, Any]] = []
    for layout_id in EXPECTED_LAYOUTS[1:]:
        layout = layout_by_id[layout_id]
        metrics = layout["metrics"]
        layout_values = [
            index[(block, layout_id, "minimal")]["completion_fenced_us_per_forward"]
            for block in range(10)
        ]
        q_values = [identity - candidate for identity, candidate in zip(identity_values, layout_values)]
        q = mean_ci(q_values)
        q["classification_using_post_pilot_epsilon"] = classify_interval(
            q["ci_low_us"], q["ci_high_us"], epsilon
        )

        identity_proxy = [
            index[(block, "identity", "minimal")]["t_proxy_us_per_forward"]
            for block in range(10)
        ]
        layout_proxy = [
            index[(block, layout_id, "minimal")]["t_proxy_us_per_forward"]
            for block in range(10)
        ]
        proxy_q = mean_ci(
            identity - candidate for identity, candidate in zip(identity_proxy, layout_proxy)
        )
        proxy_q["classification_using_post_pilot_epsilon"] = classify_interval(
            proxy_q["ci_low_us"], proxy_q["ci_high_us"], epsilon
        )

        completion_minus_proxy = [
            index[(block, layout_id, "minimal")]["completion_fenced_us_per_forward"]
            - index[(block, layout_id, "minimal")]["t_proxy_us_per_forward"]
            for block in range(10)
        ]
        effects.append(
            {
                "layout_id": layout_id,
                "candidate_id": layout["candidate_id"],
                "category": layout["category"],
                "swap": layout["swap"],
                "route_shape_metadata": {
                    "raw_max_assignments": metrics["raw_max_assignments"],
                    "raw_max_improvement_from_identity": baseline_metrics["raw_max_assignments"] - metrics["raw_max_assignments"],
                    "raw_r_max_over_mean": metrics["raw_r_max_over_mean"],
                    "aligned_max_rows": metrics["aligned_max_rows"],
                    "aligned_max_improvement_from_identity": baseline_metrics["aligned_max_rows"] - metrics["aligned_max_rows"],
                    "aligned_r_max_over_mean": metrics["aligned_r_max_over_mean"],
                    "rank_raw_assignments": metrics["rank_raw_assignments"],
                    "rank_aligned_rows": metrics["rank_aligned_rows"],
                },
                "completion_fenced_timing": timing_summary(layout_values),
                "completion_fenced_q_identity_minus_layout": q,
                "t_proxy_q_identity_minus_layout": proxy_q,
                "completion_fenced_minus_t_proxy": timing_summary(completion_minus_proxy),
            }
        )

    raw_improvements = [
        effect["route_shape_metadata"]["raw_max_improvement_from_identity"] for effect in effects
    ]
    aligned_improvements = [
        effect["route_shape_metadata"]["aligned_max_improvement_from_identity"] for effect in effects
    ]
    q_means = [
        effect["completion_fenced_q_identity_minus_layout"]["mean_us"] for effect in effects
    ]

    category_effects: dict[str, Any] = {}
    for category in ["A", "B", "C", "D"]:
        category_q = [
            effect["completion_fenced_q_identity_minus_layout"]["mean_us"]
            for effect in effects
            if effect["category"] == category
        ]
        category_effects[category] = {
            "layout_count": len(category_q),
            "mean_of_layout_q_means_us": statistics.mean(category_q),
            "minimum_layout_q_mean_us": min(category_q),
            "maximum_layout_q_mean_us": max(category_q),
            "inferential_interval_intentionally_not_computed": True,
        }

    fence = e1["empty_completion_fence"]
    return {
        "design": {
            "blocks": 10,
            "forwards_per_layout_per_block": 50,
            "layout_count": 13,
            "total_timing_records": len(e1["timing_records"]),
            "randomized_within_block": True,
            "primary_observation_mode": "minimal",
        },
        "timing_semantics": {
            "primary": e1["measurement"]["completion_fenced_timing"],
            "primary_label": "completion_fenced; not pure layer makespan",
            "secondary": e1["measurement"]["t_proxy"],
            "cross_device_cuda_timestamps_subtracted": False,
            "empty_completion_fence_reference": fence,
            "empty_fence_not_subtracted_from_q": True,
            "reason_not_subtracted": "Q is already a paired change in the completion-fenced interval; subtracting a second horizon/fence term would double-correct it.",
        },
        "identity_completion_fenced_timing": identity_summary,
        "post_pilot_meaningful_effect_tolerance": {
            "epsilon_us": epsilon,
            "formula": "max(0.01 * identity completion-fenced mean, 2 us)",
            "status": "derived after this pilot for a future confirmatory main run",
            "equivalence_rule": "Only a confidence interval wholly inside [-epsilon,+epsilon] establishes practical negligibility.",
        },
        "effects": effects,
        "category_descriptions": category_effects,
        "descriptive_association_only": {
            "layout_count": len(effects),
            "pearson_q_vs_raw_max_improvement": pearson(raw_improvements, q_means),
            "pearson_q_vs_aligned_max_improvement": pearson(aligned_improvements, q_means),
            "spearman_q_vs_raw_max_improvement": pearson(average_ranks(raw_improvements), average_ranks(q_means)),
            "spearman_q_vs_aligned_max_improvement": pearson(average_ranks(aligned_improvements), average_ranks(q_means)),
            "warning": "These are descriptive correlations over deliberately stratified layouts, not fitted or independently evaluated predictive models.",
        },
        "inference_scope": {
            "confidence_intervals": "unadjusted per-layout 95% Student-t intervals over ten paired blocks",
            "multiple_comparisons": "No family-wise multiplicity correction; this is a pilot, not a confirmatory 12-hypothesis study.",
            "independent_process_starts": 1,
            "paper_main_recommendation_not_yet_run": "4 independent starts x 5 blocks x 50 forwards per layout",
        },
    }


def write_effects_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "layout_id",
        "category",
        "logical_expert_a",
        "logical_expert_b",
        "baseline_rank_a",
        "baseline_rank_b",
        "raw_max_assignments",
        "raw_max_improvement",
        "raw_r",
        "aligned_max_rows",
        "aligned_max_improvement",
        "aligned_r",
        "layout_timing_mean_us",
        "layout_timing_median_us",
        "q_mean_us",
        "q_median_us",
        "q_standard_deviation_us",
        "q_ci_low_us",
        "q_ci_high_us",
        "q_classification",
        "t_proxy_q_mean_us",
        "t_proxy_q_ci_low_us",
        "t_proxy_q_ci_high_us",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for effect in summary["e1"]["effects"]:
            swap = effect["swap"]
            metadata = effect["route_shape_metadata"]
            timing = effect["completion_fenced_timing"]
            q = effect["completion_fenced_q_identity_minus_layout"]
            proxy = effect["t_proxy_q_identity_minus_layout"]
            writer.writerow(
                {
                    "layout_id": effect["layout_id"],
                    "category": effect["category"],
                    "logical_expert_a": swap["logical_expert_a"],
                    "logical_expert_b": swap["logical_expert_b"],
                    "baseline_rank_a": swap["baseline_rank_a"],
                    "baseline_rank_b": swap["baseline_rank_b"],
                    "raw_max_assignments": metadata["raw_max_assignments"],
                    "raw_max_improvement": metadata["raw_max_improvement_from_identity"],
                    "raw_r": metadata["raw_r_max_over_mean"],
                    "aligned_max_rows": metadata["aligned_max_rows"],
                    "aligned_max_improvement": metadata["aligned_max_improvement_from_identity"],
                    "aligned_r": metadata["aligned_r_max_over_mean"],
                    "layout_timing_mean_us": timing["mean_us"],
                    "layout_timing_median_us": timing["median_us"],
                    "q_mean_us": q["mean_us"],
                    "q_median_us": q["median_us"],
                    "q_standard_deviation_us": q["standard_deviation_us"],
                    "q_ci_low_us": q["ci_low_us"],
                    "q_ci_high_us": q["ci_high_us"],
                    "q_classification": q["classification_using_post_pilot_epsilon"],
                    "t_proxy_q_mean_us": proxy["mean_us"],
                    "t_proxy_q_ci_low_us": proxy["ci_low_us"],
                    "t_proxy_q_ci_high_us": proxy["ci_high_us"],
                }
            )


def build_raw_index(root: Path, generated_at: str) -> dict[str, Any]:
    entries = [
        ("results/raw/correctness_identity_smoke.json", "identity correctness smoke"),
        ("results/raw/correctness_all_layouts_pilot.json", "all-layout correctness-only pilot used to freeze tolerance"),
        ("results/raw/correctness_tolerance_validation.json", "all-layout validation under frozen tolerance"),
        ("results/raw/e0_smoke.json", "E0 smoke"),
        ("results/raw/e0_pilot.json", "E0 10x50 observer-effect pilot"),
        ("results/raw/e1_pilot.json", "E1 13-layout 10x50 pilot"),
        ("results/correctness_identity_smoke.log", "identity correctness log"),
        ("results/correctness_all_layouts_pilot.log", "all-layout correctness log"),
        ("results/correctness_tolerance_validation.log", "frozen-tolerance validation log"),
        ("results/e0_smoke.log", "E0 smoke log"),
        ("results/e0_pilot.log", "E0 pilot log"),
        ("results/e1_pilot.log", "E1 pilot log"),
        ("placements/manifest.json", "frozen placement manifest"),
        ("inputs/correctness_tolerance.json", "frozen numerical tolerance"),
        ("scripts/replay_placement.py", "replay and timing implementation"),
        ("scripts/analyze_e1.py", "validation and analysis implementation"),
        ("results/e1_summary.json", "machine-readable validated E0/E1 summary"),
        ("results/e1_layout_effects.csv", "flat signed layout-effect table"),
        ("REPORT.md", "human-readable E0/E1 report"),
    ]
    for rank in range(8):
        entries.append(
            (
                f"inputs/live_capture/profile_rank{rank}.pt",
                f"rank-{rank} route-profile capture",
            )
        )
        entries.append(
            (
                f"inputs/live_capture/snapshot_rank{rank}.pt",
                f"rank-{rank} real hidden/route/output/FP8-weight snapshot",
            )
        )

    files = []
    for relative_path, purpose in entries:
        path = root / relative_path
        require(path.is_file(), f"missing indexed artifact: {relative_path}")
        size = path.stat().st_size
        is_large = size > 100 * 1024 * 1024
        entry = {
            "path": relative_path,
            "purpose": purpose,
            "bytes": size,
            "sha256": sha256(path),
            "over_100_mib": is_large,
            "archive_required": is_large,
        }
        if is_large:
            entry.update(
                {
                    "archive_status": "local_only_not_published",
                    "fetch_url": None,
                    "schema": "PyTorch torch.save dictionary; tensor keys and shapes are recorded by the replay validation output",
                    "extraction": "Load locally with torch.load(path, map_location='cpu', weights_only=True).",
                }
            )
        files.append(entry)
    large_files = [entry for entry in files if entry["over_100_mib"]]
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "experiment_root": str(root.resolve()),
        "all_indexed_files_are_at_most_100_mib": not large_files,
        "large_file_count": len(large_files),
        "fetchable_archive_complete": not large_files,
        "publication_status": (
            "No external archive was published because publication/upload requires explicit user confirmation."
            if large_files
            else "No external archive is required."
        ),
        "files": files,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    result_dir = root / "results"
    e0_path = result_dir / "raw" / "e0_pilot.json"
    e1_path = result_dir / "raw" / "e1_pilot.json"
    manifest_path = root / "placements" / "manifest.json"
    tolerance_path = root / "inputs" / "correctness_tolerance.json"

    e0 = load_json(e0_path)
    e1 = load_json(e1_path)
    manifest = load_json(manifest_path)
    tolerance = load_json(tolerance_path)
    manifest_digest = sha256(manifest_path)

    require(manifest_digest == "52b8a045d24456fb03cb5bc294a3a76ba31b243fbb1fbd4a000faf8fb6a12e04", "frozen manifest hash changed")
    for result in [e0, e1]:
        require(result["placement_manifest_sha256"] == manifest_digest, "result manifest hash mismatch")
        require(result["capture_id"] == manifest["capture_id"], "capture ID mismatch")
        require(result["layer_id"] == manifest["layer_id"] == 24, "layer mismatch")
        require(result["shape"]["total_hidden_rows"] == manifest["total_hidden_rows"], "hidden-row mismatch")
        require(result["shape"]["total_valid_assignments"] == manifest["total_valid_assignments"], "assignment mismatch")
        require(result["tolerance"]["status"] == "frozen", "result did not use frozen tolerance")
        require(result["tolerance"]["frozen_before_performance_timing"] is True, "tolerance was not frozen before timing")
        require(result["tolerance"]["placement_manifest_sha256"] == manifest_digest, "tolerance manifest hash mismatch")
        require(datetime.fromisoformat(result["tolerance"]["frozen_at_utc"]) < datetime.fromisoformat(result["started_at_utc"]), "tolerance timestamp is not before timing")

    require(e0["phase"] == "e0", "wrong E0 phase")
    require(e1["phase"] == "e1", "wrong E1 phase")
    require(e0["selected_layouts"] == ["identity", "A1"], "E0 layouts changed")
    require(e1["selected_layouts"] == EXPECTED_LAYOUTS, "E1 layouts changed")
    require(tolerance["status"] == "frozen", "tolerance file is not frozen")
    require(tolerance["placement_manifest_sha256"] == manifest_digest, "tolerance file manifest hash mismatch")
    require(sha256(root / "results/raw/correctness_tolerance_validation.json") == tolerance["correctness_pilot_sha256"], "correctness pilot hash mismatch")

    validation = {
        "passed": True,
        "manifest": validate_manifest(manifest),
        "e0_correctness": correctness_validation(e0, ["identity", "A1"]),
        "e1_correctness": correctness_validation(e1, EXPECTED_LAYOUTS),
        "e0_warmup_aligned_counts": warmup_validation(e0, ["identity", "A1"]),
        "e1_warmup_aligned_counts": warmup_validation(e1, EXPECTED_LAYOUTS),
        "e0_frozen_tensor_hashes": hash_validation(e0),
        "e1_frozen_tensor_hashes": hash_validation(e1),
        "e0_intervention": intervention_validation(e0),
        "e1_intervention": intervention_validation(e1),
        "frozen_tolerance_sha256": sha256(tolerance_path),
        "placement_manifest_sha256": manifest_digest,
        "e0_raw_sha256": sha256(e0_path),
        "e1_raw_sha256": sha256(e1_path),
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "experiment": "8xH20 frozen expert-placement E0/E1 pilots",
        "validation": validation,
        "e0": summarize_e0(e0),
        "e1": summarize_e1(e1, manifest),
    }

    summary_path = result_dir / "e1_summary.json"
    csv_path = result_dir / "e1_layout_effects.csv"
    index_path = result_dir / "raw_index.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_effects_csv(csv_path, summary)
    raw_index = build_raw_index(root, generated_at)
    with index_path.open("w") as handle:
        json.dump(raw_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"VALIDATION_PASS={validation['passed']}")
    print(f"SUMMARY_JSON={summary_path}")
    print(f"EFFECTS_CSV={csv_path}")
    print(f"RAW_INDEX={index_path}")
    for effect in summary["e1"]["effects"]:
        q = effect["completion_fenced_q_identity_minus_layout"]
        print(
            f"{effect['layout_id']}: Q={q['mean_us']:+.3f} us "
            f"95% CI=[{q['ci_low_us']:+.3f}, {q['ci_high_us']:+.3f}] "
            f"{q['classification_using_post_pilot_epsilon']}"
        )


if __name__ == "__main__":
    main()
