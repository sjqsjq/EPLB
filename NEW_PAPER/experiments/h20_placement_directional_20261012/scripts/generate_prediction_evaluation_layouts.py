#!/usr/bin/env python3
"""Generate held-out evaluation maps only after the calibration model lock exists."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import shutil
from typing import Any, Callable, Iterable

from generate_prediction_calibration_layouts import (
    apply_random_cross_rank_swaps,
    local_search_layout,
    lpt_layout,
    random_balanced_layout,
)
from prediction_common import (
    IDENTITY_P2L,
    STUDY_ID,
    canonical_sha256,
    indexed_units,
    layout_descriptors,
    load_capture_index,
    load_tensor_payload,
    make_placement,
    placement_map_sha256,
    route_metadata_from_payloads,
    sha256_file,
    validate_forward_payload,
    verify_file_record,
)
from prediction_models import m2_effect_features, predict_ridge

GENERATOR_VERSION = "evaluation-layouts-v1"
M2_POOL_SIZE = 4096
M2_RANDOM_PERMUTATIONS = 2048
M2_SWAP_DEPTH_CYCLE = 128
RANDOM_POOL_SIZE = 2048


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--capture-index", type=pathlib.Path)
    parser.add_argument("--calibration-layout-index", type=pathlib.Path)
    parser.add_argument("--model-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    return parser.parse_args()


def seeded_candidates(
    *,
    base_seed: int,
    size: int,
    random_permutations: int,
    swap_depth_cycle: int,
) -> dict[str, tuple[int, ...]]:
    candidates: dict[str, tuple[int, ...]] = {}
    for index in range(size):
        seed = (int(base_seed) + index * 0x9E3779B1) % (2**63 - 1)
        if index < random_permutations:
            p2l = random_balanced_layout(seed)
        else:
            swaps = 1 + (index - random_permutations) % swap_depth_cycle
            p2l = apply_random_cross_rank_swaps(seed, swaps)
        candidates.setdefault(placement_map_sha256(p2l), p2l)
    return candidates


def first_unique_objective_layout(
    *,
    weights: tuple[int, ...],
    category: str,
    variant: str,
    base_seed: int,
    forbidden: set[str],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    for attempt in range(256):
        seed = (base_seed + attempt * 1_000_003) % (2**63 - 1)
        if variant == "lpt":
            p2l = lpt_layout(weights, seed)
            iterations = None
            algorithm = "capacity_constrained_lpt"
        elif variant == "random_swap_descent":
            initial = random_balanced_layout(seed)
            p2l, iterations = local_search_layout(weights, initial)
            algorithm = "strict_best_cross_rank_swap_descent"
        else:
            raise ValueError(variant)
        map_hash = placement_map_sha256(p2l)
        if map_hash in forbidden:
            continue
        forbidden.add(map_hash)
        return p2l, {
            "algorithm": algorithm,
            "category": category,
            "variant": variant,
            "attempt": attempt,
            "seed": seed,
            "iterations": iterations,
            "objective": f"lexicographic_sorted_descending_{category}_rank_loads",
            "selection_uses_performance_timing": False,
        }
    raise RuntimeError(f"could not generate unseen {category}/{variant} map")


def m2_score(
    *,
    identity_descriptors: dict[str, Any],
    candidate_descriptors: dict[str, Any],
    model_lock: dict[str, Any],
) -> float:
    model = model_lock["models"]["M2_grouped_gemm"]
    names, values = m2_effect_features(
        identity_descriptors=identity_descriptors,
        candidate_descriptors=candidate_descriptors,
        runner_model=model["runner_compute_model"],
    )
    return float(predict_ridge(model["completion_model"], values, names))


def select_m2_layouts(
    *,
    metadata,
    identity_descriptors: dict[str, Any],
    base_seed: int,
    forbidden: set[str],
    model_lock: dict[str, Any],
) -> list[tuple[tuple[int, ...], dict[str, Any]]]:
    pool = seeded_candidates(
        base_seed=base_seed,
        size=M2_POOL_SIZE,
        random_permutations=M2_RANDOM_PERMUTATIONS,
        swap_depth_cycle=M2_SWAP_DEPTH_CYCLE,
    )
    scored = []
    for map_hash, p2l in pool.items():
        if map_hash in forbidden:
            continue
        descriptors = layout_descriptors(metadata, p2l)
        predicted = m2_score(
            identity_descriptors=identity_descriptors,
            candidate_descriptors=descriptors,
            model_lock=model_lock,
        )
        scored.append((-predicted, map_hash, p2l))
    scored.sort(key=lambda value: (value[0], value[1]))
    if len(scored) < 2:
        raise RuntimeError("locked-M2 candidate pool has fewer than two unseen maps")
    selected = []
    for rank, (negative_prediction, map_hash, p2l) in enumerate(scored[:2], 1):
        forbidden.add(map_hash)
        selected.append(
            (
                p2l,
                {
                    "algorithm": "top_locked_M2_prediction_from_reserved_route_only_pool",
                    "selection_rank": rank,
                    "locked_predicted_q_us": -negative_prediction,
                    "pool_size_requested": M2_POOL_SIZE,
                    "pool_unique_maps": len(pool),
                    "eligible_unseen_maps": len(scored),
                    "pool_random_permutations": M2_RANDOM_PERMUTATIONS,
                    "pool_identity_swap_trajectories": M2_POOL_SIZE - M2_RANDOM_PERMUTATIONS,
                    "swap_depth_cycle": M2_SWAP_DEPTH_CYCLE,
                    "tie_break": "ascending_p2l_sha256",
                    "selection_uses_performance_timing": False,
                },
            )
        )
    return selected


def select_stratified_random_layouts(
    *,
    metadata,
    base_seed: int,
    forbidden: set[str],
) -> list[tuple[tuple[int, ...], dict[str, Any]]]:
    pool = seeded_candidates(
        base_seed=base_seed,
        size=RANDOM_POOL_SIZE,
        random_permutations=RANDOM_POOL_SIZE,
        swap_depth_cycle=1,
    )
    scored = []
    for map_hash, p2l in pool.items():
        if map_hash in forbidden:
            continue
        raw_r = float(layout_descriptors(metadata, p2l)["raw_max_over_mean"])
        scored.append((raw_r, map_hash, p2l))
    scored.sort(key=lambda value: (value[0], value[1]))
    quartile = len(scored) // 4
    strata = (
        ("lower_raw_r_quartile", scored[:quartile]),
        ("upper_raw_r_quartile", scored[-quartile:]),
    )
    selected = []
    for stratum_index, (stratum_name, records) in enumerate(strata):
        eligible = [record for record in records if record[1] not in forbidden]
        if not eligible:
            raise RuntimeError(f"no unseen maps in {stratum_name}")
        rng = random.Random(base_seed + 10_000_019 * (stratum_index + 1))
        chosen = eligible[rng.randrange(len(eligible))]
        raw_r, map_hash, p2l = chosen
        forbidden.add(map_hash)
        selected.append(
            (
                p2l,
                {
                    "algorithm": "reserved_seed_random_selection_within_raw_r_quartile",
                    "stratum": stratum_name,
                    "seed": base_seed + 10_000_019 * (stratum_index + 1),
                    "pool_size_requested": RANDOM_POOL_SIZE,
                    "pool_unique_maps": len(pool),
                    "eligible_stratum_maps": len(eligible),
                    "selected_raw_max_over_mean": raw_r,
                    "selection_uses_performance_timing": False,
                },
            )
        )
    return selected


def calibration_hashes(path: pathlib.Path) -> set[str]:
    index = json.loads(path.read_text())
    if index.get("status") != "all_calibration_layouts_generated_before_timing":
        raise ValueError("calibration layout index is not frozen input")
    hashes = set()
    for record in index["manifests"]:
        manifest_path = pathlib.Path(record["path"])
        if manifest_path.stat().st_size != record["bytes"] or sha256_file(manifest_path) != record["sha256"]:
            raise ValueError(f"calibration layout manifest differs: {manifest_path}")
        values = json.loads(manifest_path.read_text())["layouts"]
        hashes.update(layout["map_sha256"] for layout in values if layout["layout_id"] != "identity")
    return hashes


def placement_record(placement, model_lock: dict[str, Any], identity_descriptors) -> dict[str, Any]:
    predicted = (
        0.0
        if placement.layout_id == "identity"
        else m2_score(
            identity_descriptors=identity_descriptors,
            candidate_descriptors=placement.descriptors,
            model_lock=model_lock,
        )
    )
    return {
        "layout_id": placement.layout_id,
        "category": placement.category,
        "map_sha256": placement.map_sha256,
        "mapping_semantics": "p2l[physical_slot]=logical_expert; l2p is inverse",
        "p2l": list(placement.p2l),
        "l2p": list(placement.l2p),
        "generation": placement.generation,
        "locked_m2_predicted_q_us": predicted,
        "descriptors": placement.descriptors,
    }


def build_manifest(
    *,
    unit,
    metadata,
    capture_index_path: pathlib.Path,
    capture_index_sha256: str,
    calibration_layout_index: pathlib.Path,
    calibration_layout_index_sha256: str,
    calibration_map_hashes: set[str],
    model_lock_path: pathlib.Path,
    model_lock: dict[str, Any],
    model_lock_sha256: str,
    seeds: dict[str, int],
) -> dict[str, Any]:
    identity = make_placement(
        layout_id="identity",
        category="identity",
        p2l=IDENTITY_P2L,
        generation={"algorithm": "identity"},
        metadata=metadata,
    )
    placements = [identity]
    forbidden = set(calibration_map_hashes)
    forbidden.add(identity.map_sha256)
    raw_specs = (
        ("raw-01", "lpt"),
        ("raw-02", "random_swap_descent"),
    )
    for layout_id, variant in raw_specs:
        p2l, generation = first_unique_objective_layout(
            weights=metadata.expert_counts,
            category="raw",
            variant=variant,
            base_seed=int(seeds[layout_id]),
            forbidden=forbidden,
        )
        placements.append(
            make_placement(
                layout_id=layout_id,
                category="raw_load",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    padding_specs = (
        ("padding-01", "lpt"),
        ("padding-02", "random_swap_descent"),
    )
    for layout_id, variant in padding_specs:
        p2l, generation = first_unique_objective_layout(
            weights=metadata.expert_blocks,
            category="padding",
            variant=variant,
            base_seed=int(seeds[layout_id]),
            forbidden=forbidden,
        )
        placements.append(
            make_placement(
                layout_id=layout_id,
                category="padding_128",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    m2_designs = select_m2_layouts(
        metadata=metadata,
        identity_descriptors=identity.descriptors,
        base_seed=int(seeds["m2_pool"]),
        forbidden=forbidden,
        model_lock=model_lock,
    )
    for index, (p2l, generation) in enumerate(m2_designs, 1):
        placements.append(
            make_placement(
                layout_id=f"m2-{index:02d}",
                category="locked_m2",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    random_designs = select_stratified_random_layouts(
        metadata=metadata,
        base_seed=int(seeds["random_pool"]),
        forbidden=forbidden,
    )
    for index, (p2l, generation) in enumerate(random_designs, 1):
        placements.append(
            make_placement(
                layout_id=f"random-{index:02d}",
                category="stratified_random",
                p2l=p2l,
                generation=generation,
                metadata=metadata,
            )
        )
    expected_ids = [
        "identity",
        "raw-01",
        "raw-02",
        "padding-01",
        "padding-02",
        "m2-01",
        "m2-02",
        "random-01",
        "random-02",
    ]
    if [placement.layout_id for placement in placements] != expected_ids:
        raise ValueError("evaluation layout IDs/order differ")
    nonidentity_hashes = {placement.map_sha256 for placement in placements[1:]}
    if len(nonidentity_hashes) != 8 or nonidentity_hashes & calibration_map_hashes:
        raise ValueError("evaluation maps are duplicated or seen in calibration")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "generated_after_model_lock_before_evaluation_timing",
        "split": "evaluation",
        "capture_id": unit.capture_id,
        "batch_id": unit.batch_id,
        "layer_id": unit.layer_id,
        "selection_uses_performance_timing": False,
        "evaluation_timing_observed_before_generation": False,
        "model_lock": {
            "path": str(model_lock_path),
            "sha256": model_lock_sha256,
            "status": model_lock["status"],
        },
        "capture_index": {
            "path": str(capture_index_path),
            "sha256": capture_index_sha256,
        },
        "calibration_layout_index": {
            "path": str(calibration_layout_index),
            "sha256": calibration_layout_index_sha256,
            "nonidentity_map_hash_count": len(calibration_map_hashes),
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
        "reserved_seeds": seeds,
        "generator": {
            "version": GENERATOR_VERSION,
            "m2_pool_size": M2_POOL_SIZE,
            "m2_random_permutations": M2_RANDOM_PERMUTATIONS,
            "m2_identity_swap_trajectories": M2_POOL_SIZE - M2_RANDOM_PERMUTATIONS,
            "m2_swap_depth_cycle": M2_SWAP_DEPTH_CYCLE,
            "stratified_random_pool_size": RANDOM_POOL_SIZE,
            "within_rank_slot_order": "ascending logical expert ID",
        },
        "layouts": [
            placement_record(placement, model_lock, identity.descriptors)
            for placement in placements
        ],
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    capture_index_path = (
        args.capture_index or root / "inputs/prediction_v1/capture_index.json"
    ).resolve()
    calibration_layout_index = (
        args.calibration_layout_index
        or root / "inputs/prediction_v1/calibration_layouts/index.json"
    ).resolve()
    model_lock_path = args.model_lock.resolve()
    output_dir = (
        args.output_dir or root / "inputs/prediction_v1/evaluation_layouts"
    ).resolve()
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation layouts: {output_dir}")
    if staging.exists():
        raise FileExistsError(f"stale evaluation-layout staging directory: {staging}")

    model_lock = json.loads(model_lock_path.read_text())
    if model_lock.get("study_id") != STUDY_ID or model_lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("models are not locked before evaluation")
    implementation = model_lock["locked_implementation"][
        "generate_prediction_evaluation_layouts_record"
    ]
    if sha256_file(pathlib.Path(__file__).resolve()) != implementation["sha256"]:
        raise ValueError("evaluation layout generator changed after model lock")
    model_lock_sha256 = sha256_file(model_lock_path)
    capture_index_sha256 = sha256_file(capture_index_path)
    calibration_layout_index_sha256 = sha256_file(calibration_layout_index)
    forbidden_calibration = calibration_hashes(calibration_layout_index)

    index = load_capture_index(capture_index_path)
    units = indexed_units(index, split="evaluation")
    if len(units) != 12:
        raise ValueError(f"expected 12 evaluation capture units, found {len(units)}")
    seeds_by_capture = model_lock["evaluation_reserved_seeds"]
    if set(seeds_by_capture) != {unit.capture_id for unit in units}:
        raise ValueError("model lock reserved-seed capture set differs")

    manifests = []
    for unit in units:
        payloads = []
        for rank, record in enumerate(unit.forward_records):
            path = verify_file_record(record)
            payload = load_tensor_payload(path)
            validate_forward_payload(
                payload,
                unit=unit,
                rank=rank,
                path=path,
                allowed_splits={"evaluation"},
            )
            payloads.append(payload)
        metadata = route_metadata_from_payloads(unit, payloads)
        manifests.append(
            (
                f"{unit.capture_id}.json",
                build_manifest(
                    unit=unit,
                    metadata=metadata,
                    capture_index_path=capture_index_path,
                    capture_index_sha256=capture_index_sha256,
                    calibration_layout_index=calibration_layout_index,
                    calibration_layout_index_sha256=calibration_layout_index_sha256,
                    calibration_map_hashes=forbidden_calibration,
                    model_lock_path=model_lock_path,
                    model_lock=model_lock,
                    model_lock_sha256=model_lock_sha256,
                    seeds=seeds_by_capture[unit.capture_id],
                ),
            )
        )

    staging.mkdir(parents=True)
    try:
        records = []
        all_evaluation_hashes = set()
        for filename, payload in manifests:
            path = staging / filename
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            hashes = [layout["map_sha256"] for layout in payload["layouts"]]
            nonidentity = set(hashes[1:])
            if all_evaluation_hashes & nonidentity:
                raise ValueError("nonidentity map duplicated across evaluation captures")
            all_evaluation_hashes.update(nonidentity)
            records.append(
                {
                    "batch_id": payload["batch_id"],
                    "layer_id": payload["layer_id"],
                    "capture_id": payload["capture_id"],
                    "path": str((output_dir / filename).resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "layout_map_sha256s": hashes,
                }
            )
        index_payload = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "all_evaluation_layouts_frozen_after_model_lock_before_timing",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_lock": {
                "path": str(model_lock_path),
                "bytes": model_lock_path.stat().st_size,
                "sha256": model_lock_sha256,
            },
            "capture_index": {
                "path": str(capture_index_path),
                "sha256": capture_index_sha256,
            },
            "calibration_layout_index": {
                "path": str(calibration_layout_index),
                "sha256": calibration_layout_index_sha256,
            },
            "generator": {
                "path": str(pathlib.Path(__file__).resolve()),
                "bytes": pathlib.Path(__file__).resolve().stat().st_size,
                "sha256": sha256_file(pathlib.Path(__file__).resolve()),
                "version": GENERATOR_VERSION,
            },
            "counts": {
                "capture_units": len(records),
                "layouts_per_capture": 9,
                "nonidentity_layouts_per_capture": 8,
                "unique_nonidentity_map_hashes": len(all_evaluation_hashes),
            },
            "calibration_nonidentity_maps_excluded": len(forbidden_calibration),
            "manifests": records,
        }
        (staging / "index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"generated {len(manifests)} evaluation layout manifests: {output_dir}")
    print(f"index_sha256={sha256_file(output_dir / 'index.json')}")


if __name__ == "__main__":
    main()
