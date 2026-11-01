#!/usr/bin/env python3
"""Generate route-only calibration layouts before any prediction-study timing."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import random
import shutil
from typing import Iterable

import numpy as np

from prediction_common import (
    IDENTITY_P2L,
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    STUDY_ID,
    WORLD_SIZE,
    RouteMetadata,
    canonicalize_rank_members,
    indexed_units,
    load_capture_index,
    load_tensor_payload,
    make_placement,
    placement_map_sha256,
    route_metadata_from_payloads,
    sha256_file,
    validate_forward_payload,
    verify_file_record,
)

GENERATOR_VERSION = "calibration-layouts-v1"
SHAPE_POOL_SIZE = 2048
SHAPE_RANDOM_PERMUTATIONS = 1024
SHAPE_SWAP_DEPTH_CYCLE = 64


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--capture-index", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    return parser.parse_args()


def derived_seed(capture_id: str, purpose: str, attempt: int = 0) -> int:
    text = f"{STUDY_ID}|{GENERATOR_VERSION}|{capture_id}|{purpose}|{attempt}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def tie_value(seed: int, *values: int) -> int:
    text = "|".join(str(value) for value in (seed, *values))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def random_balanced_layout(seed: int) -> tuple[int, ...]:
    experts = list(range(NUM_EXPERTS))
    random.Random(seed).shuffle(experts)
    return canonicalize_rank_members(
        experts[rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS]
        for rank in range(WORLD_SIZE)
    )


def lpt_layout(weights: Iterable[int], seed: int) -> tuple[int, ...]:
    weight_values = tuple(int(value) for value in weights)
    if len(weight_values) != NUM_EXPERTS:
        raise ValueError("LPT requires one weight per expert")
    order = sorted(
        range(NUM_EXPERTS),
        key=lambda expert: (-weight_values[expert], tie_value(seed, expert)),
    )
    members: list[list[int]] = [[] for _ in range(WORLD_SIZE)]
    loads = [0] * WORLD_SIZE
    for expert in order:
        available = [rank for rank in range(WORLD_SIZE) if len(members[rank]) < LOCAL_EXPERTS]
        rank = min(
            available,
            key=lambda value: (
                loads[value],
                len(members[value]),
                tie_value(seed, expert, value),
            ),
        )
        members[rank].append(expert)
        loads[rank] += weight_values[expert]
    return canonicalize_rank_members(members)


def rank_members(p2l: Iterable[int]) -> list[list[int]]:
    values = tuple(int(value) for value in p2l)
    return [
        list(values[rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS])
        for rank in range(WORLD_SIZE)
    ]


def rank_load_score(members: list[list[int]], weights: tuple[int, ...]) -> tuple[int, ...]:
    loads = [sum(weights[expert] for expert in rank_membership) for rank_membership in members]
    return tuple(sorted(loads, reverse=True))


def local_search_layout(
    weights: Iterable[int], initial: Iterable[int]
) -> tuple[tuple[int, ...], int]:
    weight_values = tuple(int(value) for value in weights)
    members = rank_members(initial)
    current = rank_load_score(members, weight_values)
    iterations = 0
    while True:
        best: tuple[tuple[int, ...], str, int, int, int, int] | None = None
        for left_rank in range(WORLD_SIZE):
            left_load = sum(weight_values[expert] for expert in members[left_rank])
            for right_rank in range(left_rank + 1, WORLD_SIZE):
                right_load = sum(weight_values[expert] for expert in members[right_rank])
                unchanged = [
                    sum(weight_values[expert] for expert in members[rank])
                    for rank in range(WORLD_SIZE)
                    if rank not in (left_rank, right_rank)
                ]
                for left_slot, left_expert in enumerate(members[left_rank]):
                    for right_slot, right_expert in enumerate(members[right_rank]):
                        candidate_loads = unchanged + [
                            left_load - weight_values[left_expert] + weight_values[right_expert],
                            right_load - weight_values[right_expert] + weight_values[left_expert],
                        ]
                        score = tuple(sorted(candidate_loads, reverse=True))
                        if score >= current:
                            continue
                        candidate_members = [list(values) for values in members]
                        candidate_members[left_rank][left_slot] = right_expert
                        candidate_members[right_rank][right_slot] = left_expert
                        candidate_p2l = canonicalize_rank_members(candidate_members)
                        key = (
                            score,
                            placement_map_sha256(candidate_p2l),
                            left_rank,
                            right_rank,
                            left_expert,
                            right_expert,
                        )
                        if best is None or key < best:
                            best = key
        if best is None:
            return canonicalize_rank_members(members), iterations
        _, _, left_rank, right_rank, left_expert, right_expert = best
        left_slot = members[left_rank].index(left_expert)
        right_slot = members[right_rank].index(right_expert)
        members[left_rank][left_slot] = right_expert
        members[right_rank][right_slot] = left_expert
        members[left_rank].sort()
        members[right_rank].sort()
        current = rank_load_score(members, weight_values)
        iterations += 1
        if iterations > NUM_EXPERTS * NUM_EXPERTS:
            raise RuntimeError("strict local-search descent exceeded safety bound")


def objective_designs(
    *,
    metadata: RouteMetadata,
    category: str,
    weights: tuple[int, ...],
    excluded_hashes: set[str],
) -> list[tuple[tuple[int, ...], dict[str, object]]]:
    designs: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def retain(p2l: tuple[int, ...], record: dict[str, object]) -> None:
        map_hash = placement_map_sha256(p2l)
        if map_hash in excluded_hashes:
            return
        excluded_hashes.add(map_hash)
        designs.append((p2l, record))

    lpt_seed = derived_seed(metadata.capture_id, f"{category}-lpt")
    retain(
        lpt_layout(weights, lpt_seed),
        {
            "algorithm": "capacity_constrained_lpt",
            "seed": lpt_seed,
            "objective": f"lexicographic_sorted_descending_{category}_rank_loads",
            "within_rank_slot_order": "ascending_logical_expert_id",
        },
    )

    identity_result, identity_iterations = local_search_layout(weights, IDENTITY_P2L)
    retain(
        identity_result,
        {
            "algorithm": "strict_best_cross_rank_swap_descent",
            "initialization": "identity",
            "iterations": identity_iterations,
            "objective": f"lexicographic_sorted_descending_{category}_rank_loads",
            "candidate_tie_break": "ascending_p2l_sha256",
        },
    )

    attempt = 0
    while len(designs) < 3 and attempt < 256:
        seed = derived_seed(metadata.capture_id, f"{category}-random-local-search", attempt)
        initial = random_balanced_layout(seed)
        result, iterations = local_search_layout(weights, initial)
        retain(
            result,
            {
                "algorithm": "strict_best_cross_rank_swap_descent",
                "initialization": "deterministic_balanced_random",
                "attempt": attempt,
                "seed": seed,
                "iterations": iterations,
                "objective": f"lexicographic_sorted_descending_{category}_rank_loads",
                "candidate_tie_break": "ascending_p2l_sha256",
            },
        )
        attempt += 1
    if len(designs) != 3:
        raise RuntimeError(f"could not produce three unique {category} layouts")
    return designs


def apply_random_cross_rank_swaps(seed: int, swaps: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    members = rank_members(IDENTITY_P2L)
    for _ in range(swaps):
        left_rank, right_rank = rng.sample(range(WORLD_SIZE), 2)
        left_slot = rng.randrange(LOCAL_EXPERTS)
        right_slot = rng.randrange(LOCAL_EXPERTS)
        members[left_rank][left_slot], members[right_rank][right_slot] = (
            members[right_rank][right_slot],
            members[left_rank][left_slot],
        )
    return canonicalize_rank_members(members)


def shape_vector(metadata: RouteMetadata, p2l: tuple[int, ...]) -> np.ndarray:
    vectors = []
    for rank in range(WORLD_SIZE):
        experts = p2l[rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS]
        blocks = sorted((metadata.expert_blocks[expert] for expert in experts), reverse=True)
        vectors.append(tuple(blocks))
    vectors.sort(reverse=True)
    return np.asarray([value for vector in vectors for value in vector], dtype=np.float64)


def grouped_shape_designs(
    *,
    metadata: RouteMetadata,
    anchors: list[tuple[int, ...]],
    excluded_hashes: set[str],
) -> list[tuple[tuple[int, ...], dict[str, object]]]:
    candidates: dict[str, tuple[int, ...]] = {}
    for index in range(SHAPE_POOL_SIZE):
        seed = derived_seed(metadata.capture_id, "shape-pool", index)
        if index < SHAPE_RANDOM_PERMUTATIONS:
            p2l = random_balanced_layout(seed)
        else:
            swaps = 1 + (index - SHAPE_RANDOM_PERMUTATIONS) % SHAPE_SWAP_DEPTH_CYCLE
            p2l = apply_random_cross_rank_swaps(seed, swaps)
        map_hash = placement_map_sha256(p2l)
        if map_hash not in excluded_hashes:
            candidates.setdefault(map_hash, p2l)
    if len(candidates) < 3:
        raise RuntimeError("grouped-shape candidate pool has fewer than three maps")

    hashes = sorted(candidates)
    matrix = np.stack([shape_vector(metadata, candidates[value]) for value in hashes])
    anchor_matrix = np.stack([shape_vector(metadata, value) for value in anchors])
    combined = np.concatenate((matrix, anchor_matrix), axis=0)
    scale = combined.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    normalized = matrix / scale
    normalized_anchors = anchor_matrix / scale

    selected: list[tuple[tuple[int, ...], dict[str, object]]] = []
    selected_vectors = [value for value in normalized_anchors]
    available = set(range(len(hashes)))
    for selection_index in range(3):
        best_index = None
        best_distance = -1.0
        for candidate_index in sorted(available, key=lambda value: hashes[value]):
            distance = min(
                float(np.sum((normalized[candidate_index] - anchor) ** 2))
                for anchor in selected_vectors
            )
            if distance > best_distance + 1.0e-12:
                best_index = candidate_index
                best_distance = distance
        assert best_index is not None
        available.remove(best_index)
        map_hash = hashes[best_index]
        p2l = candidates[map_hash]
        excluded_hashes.add(map_hash)
        selected_vectors.append(normalized[best_index])
        selected.append(
            (
                p2l,
                {
                    "algorithm": "maximin_complete_grouped_shape_multiset_coverage",
                    "selection_index": selection_index,
                    "squared_standardized_min_distance": best_distance,
                    "candidate_pool_size_requested": SHAPE_POOL_SIZE,
                    "candidate_pool_unique_eligible": len(hashes),
                    "pool_random_permutations": SHAPE_RANDOM_PERMUTATIONS,
                    "pool_identity_swap_trajectories": SHAPE_POOL_SIZE - SHAPE_RANDOM_PERMUTATIONS,
                    "swap_depth_cycle": SHAPE_SWAP_DEPTH_CYCLE,
                    "feature": "rank-label-invariant sorted 8x16 expert 128-row block-count multiset",
                    "standardization": "candidate-plus-anchor population SD; zero SD replaced by one",
                    "candidate_tie_break": "ascending_p2l_sha256",
                },
            )
        )
    return selected


def placement_record(placement) -> dict[str, object]:
    return {
        "layout_id": placement.layout_id,
        "category": placement.category,
        "map_sha256": placement.map_sha256,
        "mapping_semantics": "p2l[physical_slot]=logical_expert; l2p is inverse",
        "p2l": list(placement.p2l),
        "l2p": list(placement.l2p),
        "generation": placement.generation,
        "descriptors": placement.descriptors,
    }


def build_manifest(
    *,
    unit,
    metadata: RouteMetadata,
    capture_index_path: pathlib.Path,
    capture_index_sha256: str,
) -> dict[str, object]:
    placements = [
        make_placement(
            layout_id="identity",
            category="identity",
            p2l=IDENTITY_P2L,
            generation={
                "algorithm": "identity",
                "within_rank_slot_order": "ascending_logical_expert_id",
            },
            metadata=metadata,
        )
    ]
    excluded = {placements[0].map_sha256}
    raw_designs = objective_designs(
        metadata=metadata,
        category="raw",
        weights=metadata.expert_counts,
        excluded_hashes=excluded,
    )
    for index, (p2l, generation) in enumerate(raw_designs, 1):
        placements.append(
            make_placement(
                layout_id=f"raw-{index:02d}",
                category="raw_load",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    padding_designs = objective_designs(
        metadata=metadata,
        category="padding",
        weights=metadata.expert_blocks,
        excluded_hashes=excluded,
    )
    for index, (p2l, generation) in enumerate(padding_designs, 1):
        placements.append(
            make_placement(
                layout_id=f"padding-{index:02d}",
                category="padding_128",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    shape_designs = grouped_shape_designs(
        metadata=metadata,
        anchors=[placement.p2l for placement in placements],
        excluded_hashes=excluded,
    )
    for index, (p2l, generation) in enumerate(shape_designs, 1):
        placements.append(
            make_placement(
                layout_id=f"shape-{index:02d}",
                category="grouped_shape_coverage",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    expected_ids = [
        "identity",
        "raw-01",
        "raw-02",
        "raw-03",
        "padding-01",
        "padding-02",
        "padding-03",
        "shape-01",
        "shape-02",
        "shape-03",
    ]
    if [placement.layout_id for placement in placements] != expected_ids:
        raise ValueError("calibration layout IDs differ from the frozen design")
    if len({placement.map_sha256 for placement in placements}) != len(placements):
        raise ValueError(f"duplicate placement map in {unit.capture_id}")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "generated_before_calibration_timing",
        "split": "calibration",
        "capture_id": unit.capture_id,
        "batch_id": unit.batch_id,
        "layer_id": unit.layer_id,
        "selection_uses_performance_timing": False,
        "purpose": "model_identification_not_measured_winner_selection",
        "capture_index": {
            "path": str(capture_index_path),
            "sha256": capture_index_sha256,
        },
        "forward_files": list(unit.forward_records),
        "route_metadata": {
            "hidden_rows_by_source_rank": list(metadata.hidden_rows_by_source_rank),
            "total_hidden_rows": metadata.total_hidden_rows,
            "total_assignments": metadata.total_assignments,
            "source_expert_counts": [list(row) for row in metadata.source_expert_counts],
            "expert_counts": list(metadata.expert_counts),
            "expert_blocks_128": list(metadata.expert_blocks),
            "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
        },
        "generator": {
            "version": GENERATOR_VERSION,
            "seed_derivation": "uint64 big-endian SHA256(study_id|version|capture_id|purpose|attempt) prefix",
            "within_rank_slot_order": "ascending logical expert ID",
            "raw_objective": "lexicographic ascending sorted-descending vector of raw rank assignment totals",
            "padding_objective": "lexicographic ascending sorted-descending vector of rank total 128-row expert blocks",
            "strict_swap_descent": "cross-rank single swaps only; no performance information",
            "grouped_shape_pool_size": SHAPE_POOL_SIZE,
        },
        "layouts": [placement_record(placement) for placement in placements],
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    capture_index_path = (
        args.capture_index or root / "inputs/prediction_v1/capture_index.json"
    ).resolve()
    output_dir = (
        args.output_dir or root / "inputs/prediction_v1/calibration_layouts"
    ).resolve()
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration layouts: {output_dir}")
    if staging.exists():
        raise FileExistsError(f"stale calibration-layout staging directory: {staging}")

    index = load_capture_index(capture_index_path)
    units = indexed_units(index, split="calibration")
    if len(units) != 6:
        raise ValueError(f"expected six calibration capture units, found {len(units)}")
    capture_index_sha256 = sha256_file(capture_index_path)
    manifests: list[tuple[str, dict[str, object]]] = []
    for unit in units:
        payloads = []
        identity_hashes = set()
        for rank, record in enumerate(unit.forward_records):
            path = verify_file_record(record)
            payload = load_tensor_payload(path)
            meta = validate_forward_payload(
                payload,
                unit=unit,
                rank=rank,
                path=path,
                allowed_splits={"calibration"},
            )
            identity_hashes.add(meta["identity_map_sha256"])
            payloads.append(payload)
        if len(identity_hashes) != 1:
            raise ValueError(f"identity map differs across {unit.capture_id}")
        metadata = route_metadata_from_payloads(unit, payloads)
        manifests.append(
            (
                f"{unit.capture_id}.json",
                build_manifest(
                    unit=unit,
                    metadata=metadata,
                    capture_index_path=capture_index_path,
                    capture_index_sha256=capture_index_sha256,
                ),
            )
        )

    staging.mkdir(parents=True)
    try:
        records = []
        for filename, payload in manifests:
            path = staging / filename
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            records.append(
                {
                    "batch_id": payload["batch_id"],
                    "layer_id": payload["layer_id"],
                    "capture_id": payload["capture_id"],
                    "path": str((output_dir / filename).resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "layout_map_sha256s": [
                        layout["map_sha256"] for layout in payload["layouts"]
                    ],
                }
            )
        index_payload = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "all_calibration_layouts_generated_before_timing",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "capture_index": {
                "path": str(capture_index_path),
                "sha256": capture_index_sha256,
            },
            "generator": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": sha256_file(pathlib.Path(__file__).resolve()),
                "version": GENERATOR_VERSION,
            },
            "counts": {
                "capture_units": len(records),
                "layouts_per_capture": 10,
                "nonidentity_layouts_per_capture": 9,
            },
            "manifests": records,
        }
        (staging / "index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"generated {len(manifests)} calibration layout manifests: {output_dir}")
    print(f"index_sha256={sha256_file(output_dir / 'index.json')}")


if __name__ == "__main__":
    main()
