#!/usr/bin/env python3
"""Freeze the low-r stopping-study layouts: deterministic greedy base + 12 residual candidates.

Design (FOLLOWUP_PROTOCOL.md section C, frozen before any timing):
- Use only locked evaluation captures and the locked M2 model.
- Base: deterministic low-r placement via first-improvement greedy legal cross-rank
  swaps from identity. Lexicographic score:
    (max raw rank assignments, max padded rank rows,
     sorted-descending raw rank load vector, placement map sha256)
  Legal swaps are enumerated in deterministic lexicographic slot order
  (slot_a ascending, slot_b > slot_a ascending, cross-rank only); the first swap
  that strictly improves the lexicographic score is applied and the scan restarts;
  the walk stops when no legal single swap improves the score. The full path is
  recorded and the final base is frozen here, before timing.
- Residual candidates: every legal cross-rank single swap from the base is scored
  with the locked M2 model only (predicted signed gain versus base, positive =
  candidate faster), then 12 are frozen across three strata:
    gain-01..04  four largest predicted signed gains (tie-break: map sha256),
    near-01..04  four predictions closest to zero (signs retained),
    neg-01..04   four most negative predictions.
  Strata are filled in that order; already-selected maps are skipped with
  deterministic advancement inside the stratum. Measured latency is never used.
- The identity map is included in the manifest as correctness reference only and
  is not a timed layout.

No performance timing is read or written by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generate_prediction_evaluation_layouts import m2_score
from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    WORLD_SIZE,
    RouteMetadata,
    indexed_units,
    layout_descriptors,
    load_capture_index,
    load_tensor_payload,
    placement_map_sha256,
    route_metadata_from_payloads,
    sha256_file,
    validate_forward_payload,
    validate_p2l,
    invert_p2l,
    verify_file_record,
    write_json_atomic,
)

STOPPING_STUDY_ID = "h20-placement-stopping-v1"
GENERATOR_VERSION = "stopping-layouts-v1"
IDENTITY_P2L = tuple(range(NUM_EXPERTS))
CANDIDATE_COUNT = 12
STRATA = (
    ("gain", "four_largest_locked_m2_predicted_signed_gains"),
    ("near", "four_locked_m2_predictions_closest_to_zero_signs_retained"),
    ("neg", "four_most_negative_locked_m2_predictions"),
)
OBJECTIVE = {
    "primary": "max_raw_rank_assignments",
    "tie_breakers": [
        "max_padded_rank_rows_128_zero_aware",
        "sorted_descending_raw_rank_load_vector",
        "placement_map_sha256",
    ],
    "direction": "minimize_lexicographic_tuple",
    "swap_enumeration_order": "slot_a ascending, slot_b>slot_a ascending, cross-rank only",
    "rule": "first_improvement_scan_restarts_after_each_applied_swap",
    "stop": "no_legal_single_swap_improves_the_lexicographic_score",
}
LOCKED_IMPLEMENTATION_REQUIRED = (
    "prediction_common_record",
    "prediction_models_record",
    "generate_prediction_evaluation_layouts_record",
    "generate_prediction_calibration_layouts_record",
)


def aligned_rows(count: int) -> int:
    return 0 if count == 0 else 128 * ((count + 127) // 128)


def rank_loads(p2l: tuple[int, ...], counts: tuple[int, ...]) -> tuple[list[int], list[int]]:
    raw = [0] * WORLD_SIZE
    padded = [0] * WORLD_SIZE
    for slot, logical in enumerate(p2l):
        rank = slot // LOCAL_EXPERTS
        raw[rank] += counts[logical]
        padded[rank] += aligned_rows(counts[logical])
    return raw, padded


def score_prefix(raw: list[int], padded: list[int]) -> tuple:
    return (max(raw), max(padded), tuple(sorted(raw, reverse=True)))


def score_full(p2l: tuple[int, ...], raw: list[int], padded: list[int]) -> tuple:
    return score_prefix(raw, padded) + (placement_map_sha256(p2l),)


def legal_swaps() -> list[tuple[int, int]]:
    pairs = []
    for slot_a in range(NUM_EXPERTS):
        for slot_b in range(slot_a + 1, NUM_EXPERTS):
            if slot_a // LOCAL_EXPERTS != slot_b // LOCAL_EXPERTS:
                pairs.append((slot_a, slot_b))
    return pairs


def swapped(p2l: tuple[int, ...], slot_a: int, slot_b: int) -> tuple[int, ...]:
    values = list(p2l)
    values[slot_a], values[slot_b] = values[slot_b], values[slot_a]
    return tuple(values)


def greedy_base(metadata: RouteMetadata) -> dict[str, Any]:
    counts = metadata.expert_counts
    aligned = [aligned_rows(count) for count in counts]
    p2l = IDENTITY_P2L
    raw, padded = rank_loads(p2l, counts)
    current = score_full(p2l, raw, padded)
    steps: list[dict[str, Any]] = []
    scans = 0
    pairs = legal_swaps()
    improved = True
    while improved:
        improved = False
        scans += 1
        for slot_a, slot_b in pairs:
            rank_a, rank_b = slot_a // LOCAL_EXPERTS, slot_b // LOCAL_EXPERTS
            la, lb = p2l[slot_a], p2l[slot_b]
            new_raw = list(raw)
            new_padded = list(padded)
            new_raw[rank_a] += counts[lb] - counts[la]
            new_raw[rank_b] += counts[la] - counts[lb]
            new_padded[rank_a] += aligned[lb] - aligned[la]
            new_padded[rank_b] += aligned[la] - aligned[lb]
            prefix = score_prefix(new_raw, new_padded)
            if prefix > current[:3]:
                continue
            candidate_p2l = swapped(p2l, slot_a, slot_b)
            if prefix == current[:3]:
                candidate_hash = placement_map_sha256(candidate_p2l)
                if candidate_hash >= current[3]:
                    continue
            # strict lexicographic improvement
            steps.append(
                {
                    "step": len(steps) + 1,
                    "slot_a": slot_a,
                    "slot_b": slot_b,
                    "logical_a": la,
                    "logical_b": lb,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "score_before": [current[0], current[1], list(current[2]), current[3]],
                    "score_after_prefix": [prefix[0], prefix[1], list(prefix[2])],
                }
            )
            p2l = candidate_p2l
            raw, padded = new_raw, new_padded
            current = score_full(p2l, raw, padded)
            steps[-1]["score_after"] = [current[0], current[1], list(current[2]), current[3]]
            improved = True
            break
    return {
        "p2l": p2l,
        "steps": steps,
        "step_count": len(steps),
        "scans": scans,
        "final_lexicographic_score": [current[0], current[1], list(current[2]), current[3]],
    }


def build_manifest(
    *,
    unit,
    metadata: RouteMetadata,
    model_lock: dict[str, Any],
    model_lock_sha256: str,
    capture_index_sha256: str,
) -> dict[str, Any]:
    identity_descriptors = layout_descriptors(metadata, IDENTITY_P2L)
    base_result = greedy_base(metadata)
    base_p2l = validate_p2l(base_result["p2l"])
    base_descriptors = layout_descriptors(metadata, base_p2l)
    base_q = m2_score(
        identity_descriptors=identity_descriptors,
        candidate_descriptors=base_descriptors,
        model_lock=model_lock,
    )
    base_hash = placement_map_sha256(base_p2l)

    pool: list[tuple[float, str, tuple[int, ...], int, int]] = []
    excluded = {"identity_map": 0, "base_map": 0}
    for slot_a, slot_b in legal_swaps():
        candidate = swapped(base_p2l, slot_a, slot_b)
        candidate_hash = placement_map_sha256(candidate)
        if candidate_hash == base_hash:
            excluded["base_map"] += 1
            continue
        if candidate_hash == placement_map_sha256(IDENTITY_P2L):
            excluded["identity_map"] += 1
            continue
        descriptors = layout_descriptors(metadata, candidate)
        q = m2_score(
            identity_descriptors=identity_descriptors,
            candidate_descriptors=descriptors,
            model_lock=model_lock,
        )
        pool.append((q - base_q, candidate_hash, candidate, slot_a, slot_b))

    selected: list[dict[str, Any]] = []
    chosen_hashes: set[str] = set()
    selection_log: list[dict[str, Any]] = []
    for stratum_index, (prefix, description) in enumerate(STRATA):
        if prefix == "gain":
            ordered = sorted(pool, key=lambda item: (-item[0], item[1]))
        elif prefix == "near":
            ordered = sorted(pool, key=lambda item: (abs(item[0]), item[1]))
        else:
            ordered = sorted(pool, key=lambda item: (item[0], item[1]))
        advances = 0
        taken = 0
        for gain, candidate_hash, candidate, slot_a, slot_b in ordered:
            if taken == 4:
                break
            if candidate_hash in chosen_hashes:
                advances += 1
                continue
            chosen_hashes.add(candidate_hash)
            taken += 1
            selected.append(
                {
                    "layout_id": f"{prefix}-{taken:02d}",
                    "stratum": description,
                    "predicted_signed_gain_us": gain,
                    "map_sha256": candidate_hash,
                    "p2l": candidate,
                    "slot_a": slot_a,
                    "slot_b": slot_b,
                }
            )
        if taken != 4:
            raise RuntimeError(f"stratum {prefix} selected {taken} < 4 candidates")
        selection_log.append(
            {
                "stratum": description,
                "layout_ids": [f"{prefix}-{index:02d}" for index in range(1, 5)],
                "sort_key": {
                    "gain": "(-predicted_signed_gain_us, map_sha256)",
                    "near": "(abs(predicted_signed_gain_us), map_sha256)",
                    "neg": "(predicted_signed_gain_us, map_sha256)",
                }[prefix],
                "dedup_advancements": advances,
                "pool_size": len(pool),
            }
        )

    layouts = [
        {
            "layout_id": "identity",
            "category": "identity_reference",
            "map_sha256": placement_map_sha256(IDENTITY_P2L),
            "mapping_semantics": "p2l[physical_slot]=logical_expert; l2p is inverse",
            "p2l": list(IDENTITY_P2L),
            "l2p": list(invert_p2l(IDENTITY_P2L)),
            "descriptors": identity_descriptors,
            "role": "correctness_reference_only_not_timed",
        },
        {
            "layout_id": "base",
            "category": "low_r_greedy_lexicographic_base",
            "map_sha256": base_hash,
            "mapping_semantics": "p2l[physical_slot]=logical_expert; l2p is inverse",
            "p2l": list(base_p2l),
            "l2p": list(invert_p2l(base_p2l)),
            "descriptors": base_descriptors,
            "locked_m2_q_identity_minus_base_us": base_q,
        },
    ]
    for record in selected:
        p2l = record["p2l"]
        slot_a, slot_b = record["slot_a"], record["slot_b"]
        layouts.append(
            {
                "layout_id": record["layout_id"],
                "category": "residual_swap",
                "stratum": record["stratum"],
                "map_sha256": record["map_sha256"],
                "mapping_semantics": "p2l[physical_slot]=logical_expert; l2p is inverse",
                "p2l": list(p2l),
                "l2p": list(invert_p2l(p2l)),
                "swap": {
                    "slot_a": slot_a,
                    "slot_b": slot_b,
                    "logical_a": base_p2l[slot_a],
                    "logical_b": base_p2l[slot_b],
                    "rank_a": slot_a // LOCAL_EXPERTS,
                    "rank_b": slot_b // LOCAL_EXPERTS,
                    "semantics": "exchange logical experts between base owners",
                },
                "locked_m2_predicted_signed_gain_us": record["predicted_signed_gain_us"],
                "locked_m2_q_identity_minus_candidate_us": base_q + record["predicted_signed_gain_us"],
                "descriptors": layout_descriptors(metadata, p2l),
            }
        )

    layout_ids = [layout["layout_id"] for layout in layouts]
    expected_ids = ["identity", "base"] + [
        f"{prefix}-{index:02d}" for prefix, _ in STRATA for index in range(1, 5)
    ]
    if layout_ids != expected_ids:
        raise RuntimeError("layout id order differs from frozen design")

    return {
        "schema_version": 1,
        "study_id": STOPPING_STUDY_ID,
        "split": "stopping",
        "capture_id": unit.capture_id,
        "batch_id": unit.batch_id,
        "layer_id": unit.layer_id,
        "selection_uses_performance_timing": False,
        "designated_capture_rule": "all locked evaluation-split capture units; no subset selection",
        "route_metadata": {
            "hidden_rows_by_source_rank": list(metadata.hidden_rows_by_source_rank),
            "total_hidden_rows": metadata.total_hidden_rows,
            "total_assignments": metadata.total_assignments,
            "source_expert_counts": [list(row) for row in metadata.source_expert_counts],
            "expert_counts": list(metadata.expert_counts),
            "expert_blocks_128": list(metadata.expert_blocks),
            "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
        },
        "base_objective": OBJECTIVE,
        "base_greedy_path": {
            "start": "identity",
            "step_count": base_result["step_count"],
            "scans": base_result["scans"],
            "terminated": "no_legal_single_swap_improves_the_lexicographic_score",
            "steps": base_result["steps"],
            "final_lexicographic_score": base_result["final_lexicographic_score"],
        },
        "candidate_enumeration": {
            "legal_cross_rank_swaps_from_base": len(pool) + excluded["identity_map"] + excluded["base_map"],
            "pool_after_map_uniqueness_exclusions": len(pool),
            "exclusions": excluded,
            "scoring": "locked_M2 predicted completion effect only (identity-anchored effect features); predicted_signed_gain_us = Q(identity-candidate) - Q(identity-base)",
            "selection_uses_performance_timing": False,
        },
        "selection_log": selection_log,
        "layouts": layouts,
        "model_lock": {"sha256": model_lock_sha256},
        "capture_index": {"sha256": capture_index_sha256},
        "generation": {
            "generator": str(pathlib.Path(__file__).resolve()),
            "generator_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "generator_version": GENERATOR_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-index", type=pathlib.Path, default=None)
    parser.add_argument("--model-lock", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    capture_index_path = (args.capture_index or root / "inputs/prediction_v1/capture_index.json").resolve()
    model_lock_path = (args.model_lock or root / "configs/followup_v1/prediction_model_lock.json").resolve()
    output_dir = (args.output_dir or root / "inputs/stopping_v1/layouts").resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen stopping layouts: {output_dir}")

    model_lock_sha256 = sha256_file(model_lock_path)
    model_lock = json.loads(model_lock_path.read_text())
    if model_lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock status differs")
    for name in LOCKED_IMPLEMENTATION_REQUIRED:
        record = model_lock["locked_implementation"][name]
        path = pathlib.Path(record["path"])
        if not path.exists():
            path = root / "scripts" / path.name
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"locked implementation differs on disk: {name}")

    capture_index_sha256 = sha256_file(capture_index_path)
    index = load_capture_index(capture_index_path)
    units = indexed_units(index, split="evaluation")
    if len(units) != 12:
        raise ValueError(f"expected 12 locked evaluation capture units, found {len(units)}")

    staging = output_dir.with_name(output_dir.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        records = []
        all_hashes: set[str] = set()
        for unit in units:
            payloads = []
            for rank, record in enumerate(unit.forward_records):
                path = verify_file_record(record)
                payload = load_tensor_payload(path)
                validate_forward_payload(
                    payload, unit=unit, rank=rank, path=path, allowed_splits={"evaluation"}
                )
                payloads.append(payload)
            metadata = route_metadata_from_payloads(unit, payloads)
            manifest = build_manifest(
                unit=unit,
                metadata=metadata,
                model_lock=model_lock,
                model_lock_sha256=model_lock_sha256,
                capture_index_sha256=capture_index_sha256,
            )
            filename = f"{unit.capture_id}.json"
            path = staging / filename
            write_json_atomic(path, manifest)
            hashes = [layout["map_sha256"] for layout in manifest["layouts"]]
            timed = set(hashes[1:])
            overlap = all_hashes & timed
            if overlap:
                raise ValueError(f"timed map duplicated across captures: {overlap}")
            all_hashes |= timed
            records.append(
                {
                    "batch_id": unit.batch_id,
                    "layer_id": unit.layer_id,
                    "capture_id": unit.capture_id,
                    "path": str((output_dir / filename).resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "base_map_sha256": hashes[1],
                    "greedy_path_steps": manifest["base_greedy_path"]["step_count"],
                    "layout_map_sha256s": hashes,
                }
            )
            print(
                f"{unit.capture_id}: base_steps={manifest['base_greedy_path']['step_count']} "
                f"gains=[{manifest['layouts'][2]['locked_m2_predicted_signed_gain_us']:.1f} .. "
                f"{manifest['layouts'][10]['locked_m2_predicted_signed_gain_us']:.1f}] us"
            )
        index_payload = {
            "schema_version": 1,
            "study_id": STOPPING_STUDY_ID,
            "status": "all_stopping_layouts_frozen_before_timing",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_lock": {
                "path": str(model_lock_path),
                "bytes": model_lock_path.stat().st_size,
                "sha256": model_lock_sha256,
            },
            "capture_index": {"path": str(capture_index_path), "sha256": capture_index_sha256},
            "generator": {
                "path": str(pathlib.Path(__file__).resolve()),
                "bytes": pathlib.Path(__file__).resolve().stat().st_size,
                "sha256": sha256_file(pathlib.Path(__file__).resolve()),
                "version": GENERATOR_VERSION,
            },
            "counts": {
                "capture_units": len(records),
                "layouts_per_capture": 14,
                "timed_layouts_per_capture": 13,
                "residual_candidates_per_capture": CANDIDATE_COUNT,
                "unique_timed_map_hashes": len(all_hashes),
            },
            "manifests": records,
        }
        write_json_atomic(staging / "index.json", index_payload)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"generated {len(records)} stopping layout manifests: {output_dir}")
    print(f"index_sha256={sha256_file(output_dir / 'index.json')}")


if __name__ == "__main__":
    main()
