#!/usr/bin/env python3
"""Shared immutable-input, layout, and feature logic for the H20 CD-X0 study.

CD-X0 study copy of prediction_common.py (source sha256
049405c8a660c126c42df076e9a69cfb48b5b858b9008a75fda50d195e8acb6d).

Provenance: verbatim copy with exactly four CD-X0 adaptations, all recorded in
configs/followup_v1/cdx0_amendment_v1.json (amendment_5 / histogram bound /
index expectations). prediction_common.py itself is NEVER modified; frozen prior
protocols keep verifying against the original hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re
from typing import Any, Iterable

import numpy as np
import torch

STUDY_ID = "h20-placement-cdx0-v1"
WORLD_SIZE = 8
NUM_EXPERTS = 128
LOCAL_EXPERTS = NUM_EXPERTS // WORLD_SIZE
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
ALIGNMENT = 128
IDENTITY_P2L = tuple(range(NUM_EXPERTS))
MAX_BLOCKS_PER_EXPERT = 256
FP8_DTYPE = torch.float8_e4m3fn
WEIGHT_FIELDS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


@dataclasses.dataclass(frozen=True)
class IndexedCaptureUnit:
    batch_id: str
    split: str
    layer_id: int
    capture_id: str
    total_hidden_rows: int
    total_assignments: int
    hidden_rows_by_source_rank: tuple[int, ...]
    forward_records: tuple[dict[str, Any], ...]
    raw: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RouteMetadata:
    batch_id: str
    split: str
    layer_id: int
    capture_id: str
    hidden_rows_by_source_rank: tuple[int, ...]
    source_expert_counts: tuple[tuple[int, ...], ...]
    expert_counts: tuple[int, ...]
    expert_blocks: tuple[int, ...]
    total_hidden_rows: int
    total_assignments: int


@dataclasses.dataclass(frozen=True)
class Placement:
    layout_id: str
    category: str
    p2l: tuple[int, ...]
    l2p: tuple[int, ...]
    map_sha256: str
    generation: dict[str, Any]
    descriptors: dict[str, Any]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_file_record(record: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != int(record["bytes"]):
        raise ValueError(f"indexed size mismatch for {path}: {size} != {record['bytes']}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(
            f"indexed SHA-256 mismatch for {path}: {actual} != {record['sha256']}"
        )
    return path


def load_capture_index(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported prediction capture-index schema")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("capture index belongs to a different study")
    if payload.get("status") != "all_cdx0_capture_inputs_validated_and_hash_indexed":
        raise ValueError("cdx0 captures are not fully validated and indexed")
    expected_counts = {
        "capture_units": 16,
        "forward_rank_files": 128,
        "expert_rank_files": 16,
    }
    if payload.get("counts") != expected_counts:
        raise ValueError(f"capture-index counts differ: {payload.get('counts')!r}")
    return payload


def _rank_from_path(path: str, stem: str) -> int:
    match = re.search(rf"/{re.escape(stem)}(\d+)\.pt$", path)
    if not match:
        raise ValueError(f"cannot parse rank from indexed path: {path}")
    rank = int(match.group(1))
    if rank not in range(WORLD_SIZE):
        raise ValueError(f"indexed rank is out of range: {path}")
    return rank


def indexed_units(index: dict[str, Any], *, split: str | None = None) -> list[IndexedCaptureUnit]:
    units: list[IndexedCaptureUnit] = []
    for raw in index["capture_units"]:
        if split is not None and raw["split"] != split:
            continue
        records: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        for record in raw["forward_files"]:
            rank = _rank_from_path(record["path"], "forward_rank")
            if records[rank] is not None:
                raise ValueError(f"duplicate forward rank in {raw['capture_id']}")
            records[rank] = record
        if any(record is None for record in records):
            raise ValueError(f"missing forward rank in {raw['capture_id']}")
        unit = IndexedCaptureUnit(
            batch_id=str(raw["batch_id"]),
            split=str(raw["split"]),
            layer_id=int(raw["layer_id"]),
            capture_id=str(raw["capture_id"]),
            total_hidden_rows=int(raw["total_hidden_rows"]),
            total_assignments=int(raw["total_assignments"]),
            hidden_rows_by_source_rank=tuple(
                int(value) for value in raw["hidden_rows_by_source_rank"]
            ),
            forward_records=tuple(record for record in records if record is not None),
            raw=raw,
        )
        if len(unit.hidden_rows_by_source_rank) != WORLD_SIZE:
            raise ValueError(f"wrong source-rank cardinality in {unit.capture_id}")
        if unit.total_assignments != unit.total_hidden_rows * TOP_K:
            raise ValueError(f"assignment total differs in {unit.capture_id}")
        units.append(unit)
    return sorted(units, key=lambda unit: (unit.batch_id, unit.layer_id))


def indexed_expert_records(index: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for record in index["expert_files"]:
        path = record["path"]
        layer_match = re.search(r"/layer-(\d{3})/expert_rank", path)
        if not layer_match:
            raise ValueError(f"cannot parse layer from indexed expert path: {path}")
        layer_id = int(layer_match.group(1))
        rank = _rank_from_path(path, "expert_rank")
        key = (layer_id, rank)
        if key in records:
            raise ValueError(f"duplicate indexed expert artifact: {key}")
        records[key] = record
    if len(records) != 16:
        raise ValueError(f"expected 16 indexed expert artifacts, found {len(records)}")
    return records


def load_tensor_payload(path: pathlib.Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def validate_forward_payload(
    payload: dict[str, Any],
    *,
    unit: IndexedCaptureUnit,
    rank: int,
    path: pathlib.Path,
    allowed_splits: set[str],
) -> dict[str, Any]:
    expected_header = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "capture_id": unit.capture_id,
        "mode": "snapshot_split",
        "rank": rank,
        "layer_id": unit.layer_id,
        "identity_map_required": True,
        "weight_block_size": [ALIGNMENT, ALIGNMENT],
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_header.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"forward header mismatch at {path}: {mismatches}")
    context = payload.get("capture_context", {})
    if unit.split not in allowed_splits:
        raise ValueError(
            f"split {unit.split!r} is sealed for this operation: {unit.capture_id}"
        )
    if context.get("split") != unit.split or context.get("batch_id") != unit.batch_id:
        raise ValueError(f"forward split/batch provenance mismatch at {path}")
    rows = unit.hidden_rows_by_source_rank[rank]
    if payload.get("hidden_rows") != rows:
        raise ValueError(f"forward row count mismatch at {path}")
    expected_shapes = {
        "hidden_states": (rows, HIDDEN_SIZE),
        "logical_topk_ids": (rows, TOP_K),
        "logical_topk_weights": (rows, TOP_K),
        "routed_output": (rows, HIDDEN_SIZE),
    }
    expected_dtypes = {
        "hidden_states": torch.bfloat16,
        "logical_topk_ids": torch.int32,
        "logical_topk_weights": torch.float32,
        "routed_output": torch.bfloat16,
    }
    for name, shape in expected_shapes.items():
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"missing forward tensor {name} at {path}")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(
                f"forward tensor {name} shape/contiguity mismatch at {path}: "
                f"{tuple(tensor.shape)}, contiguous={tensor.is_contiguous()}"
            )
        if tensor.dtype != expected_dtypes[name]:
            raise ValueError(f"forward tensor {name} dtype mismatch at {path}")
    ids = payload["logical_topk_ids"].to(torch.int64)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS):
        raise ValueError(f"logical route ID out of range at {path}")
    sorted_ids = ids.sort(dim=1).values
    if ids.numel() and bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
        raise ValueError(f"duplicate expert in one top-k row at {path}")
    return {
        "path": str(path),
        "rank": rank,
        "hidden_rows": rows,
        "assignments": ids.numel(),
        "identity_map_sha256": payload["identity_map_sha256"],
        "expert_artifact_path": str(pathlib.Path(payload["expert_artifact_path"]).resolve()),
    }


def validate_expert_payload(
    payload: dict[str, Any],
    *,
    layer_id: int,
    rank: int,
    path: pathlib.Path,
    identity_map_sha256: str,
) -> dict[str, Any]:
    expected_header = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "artifact_type": "immutable_layer_expert_weights",
        "rank": rank,
        "layer_id": layer_id,
        "identity_map_required": True,
        "identity_map_sha256": identity_map_sha256,
        "logical_expert_ids": list(
            range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
        ),
        "weight_block_size": [ALIGNMENT, ALIGNMENT],
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_header.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"expert header mismatch at {path}: {mismatches}")
    expected_shapes = {
        "w13_weight": (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "w13_weight_scale_inv": (LOCAL_EXPERTS, 24, 32),
        "w2_weight_scale_inv": (LOCAL_EXPERTS, 32, 12),
    }
    expected_dtypes = {
        "w13_weight": FP8_DTYPE,
        "w2_weight": FP8_DTYPE,
        "w13_weight_scale_inv": torch.float32,
        "w2_weight_scale_inv": torch.float32,
    }
    for name, shape in expected_shapes.items():
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"missing expert tensor {name} at {path}")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(
                f"expert tensor {name} shape/contiguity mismatch at {path}: "
                f"{tuple(tensor.shape)}, contiguous={tensor.is_contiguous()}"
            )
        if tensor.dtype != expected_dtypes[name]:
            raise ValueError(f"expert tensor {name} dtype mismatch at {path}")
    return {
        "path": str(path),
        "rank": rank,
        "layer_id": layer_id,
        "logical_expert_ids": expected_header["logical_expert_ids"],
    }


def route_metadata_from_payloads(
    unit: IndexedCaptureUnit,
    payloads: Iterable[dict[str, Any]],
) -> RouteMetadata:
    payload_list = list(payloads)
    if len(payload_list) != WORLD_SIZE:
        raise ValueError(f"expected eight forward payloads for {unit.capture_id}")
    source_counts: list[list[int]] = []
    for rank, payload in enumerate(payload_list):
        ids = payload["logical_topk_ids"].to(torch.int64).flatten()
        counts = torch.bincount(ids, minlength=NUM_EXPERTS).tolist()
        if sum(counts) != unit.hidden_rows_by_source_rank[rank] * TOP_K:
            raise ValueError(f"source assignment count mismatch in {unit.capture_id}")
        source_counts.append([int(value) for value in counts])
    expert_counts = [sum(row[expert] for row in source_counts) for expert in range(NUM_EXPERTS)]
    expert_blocks = [zero_aware_blocks(count) for count in expert_counts]
    if max(expert_counts) > unit.total_hidden_rows:
        raise ValueError(f"top-k uniqueness bound violated in {unit.capture_id}")
    if max(expert_blocks) > MAX_BLOCKS_PER_EXPERT:
        raise ValueError(f"fixed grouped-GEMM histogram bound exceeded in {unit.capture_id}")
    if sum(expert_counts) != unit.total_assignments:
        raise ValueError(f"global assignment count mismatch in {unit.capture_id}")
    return RouteMetadata(
        batch_id=unit.batch_id,
        split=unit.split,
        layer_id=unit.layer_id,
        capture_id=unit.capture_id,
        hidden_rows_by_source_rank=unit.hidden_rows_by_source_rank,
        source_expert_counts=tuple(tuple(row) for row in source_counts),
        expert_counts=tuple(expert_counts),
        expert_blocks=tuple(expert_blocks),
        total_hidden_rows=unit.total_hidden_rows,
        total_assignments=unit.total_assignments,
    )


def zero_aware_aligned_rows(count: int) -> int:
    if count < 0:
        raise ValueError("expert count cannot be negative")
    return 0 if count == 0 else ALIGNMENT * math.ceil(count / ALIGNMENT)


def zero_aware_blocks(count: int) -> int:
    return zero_aware_aligned_rows(count) // ALIGNMENT


def validate_p2l(values: Iterable[int]) -> tuple[int, ...]:
    p2l = tuple(int(value) for value in values)
    if len(p2l) != NUM_EXPERTS or sorted(p2l) != list(range(NUM_EXPERTS)):
        raise ValueError("p2l must be a bijection over 128 experts")
    return p2l


def invert_p2l(values: Iterable[int]) -> tuple[int, ...]:
    p2l = validate_p2l(values)
    inverse = [0] * NUM_EXPERTS
    for physical, logical in enumerate(p2l):
        inverse[logical] = physical
    return tuple(inverse)


def canonicalize_rank_members(rank_members: Iterable[Iterable[int]]) -> tuple[int, ...]:
    ranks = [tuple(sorted(int(value) for value in members)) for members in rank_members]
    if len(ranks) != WORLD_SIZE or any(len(members) != LOCAL_EXPERTS for members in ranks):
        raise ValueError("rank membership must contain eight sets of 16 experts")
    return validate_p2l(value for members in ranks for value in members)


def placement_map_sha256(p2l: Iterable[int]) -> str:
    return canonical_sha256({"mapping_semantics": "p2l[physical]=logical", "p2l": list(validate_p2l(p2l))})


def layout_descriptors(metadata: RouteMetadata, p2l_values: Iterable[int]) -> dict[str, Any]:
    p2l = validate_p2l(p2l_values)
    l2p = invert_p2l(p2l)
    owner = [l2p[logical] // LOCAL_EXPERTS for logical in range(NUM_EXPERTS)]
    raw_rank = [0] * WORLD_SIZE
    padded_rank = [0] * WORLD_SIZE
    padding_waste_rank = [0] * WORLD_SIZE
    active_rank = [0] * WORLD_SIZE
    rank_block_multisets: list[list[int]] = []
    rank_expert_counts: list[list[int]] = []
    rank_logical_experts: list[list[int]] = []
    for rank in range(WORLD_SIZE):
        experts = list(p2l[rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS])
        counts = [metadata.expert_counts[expert] for expert in experts]
        blocks = [metadata.expert_blocks[expert] for expert in experts]
        aligned = [block * ALIGNMENT for block in blocks]
        raw_rank[rank] = sum(counts)
        padded_rank[rank] = sum(aligned)
        padding_waste_rank[rank] = sum(a - n for a, n in zip(aligned, counts))
        active_rank[rank] = sum(block > 0 for block in blocks)
        rank_block_multisets.append(sorted(blocks, reverse=True))
        rank_expert_counts.append(counts)
        rank_logical_experts.append(experts)
    source_destination = [[0] * WORLD_SIZE for _ in range(WORLD_SIZE)]
    for source in range(WORLD_SIZE):
        for logical, count in enumerate(metadata.source_expert_counts[source]):
            source_destination[source][owner[logical]] += count
    send_total = [sum(row) for row in source_destination]
    receive_total = [sum(source_destination[source][destination] for source in range(WORLD_SIZE)) for destination in range(WORLD_SIZE)]
    send_remote = [send_total[source] - source_destination[source][source] for source in range(WORLD_SIZE)]
    receive_remote = [receive_total[destination] - source_destination[destination][destination] for destination in range(WORLD_SIZE)]
    nonempty_pairs = sum(value > 0 for row in source_destination for value in row)
    nonempty_remote_pairs = sum(
        source != destination and source_destination[source][destination] > 0
        for source in range(WORLD_SIZE)
        for destination in range(WORLD_SIZE)
    )
    remote_assignments = sum(send_remote)
    total_assignments = sum(send_total)
    if total_assignments != metadata.total_assignments:
        raise ValueError("layout communication matrix lost assignments")
    communication = {
        "source_destination_assignments": source_destination,
        "send_total_assignments": send_total,
        "receive_total_assignments": receive_total,
        "send_remote_assignments": send_remote,
        "receive_remote_assignments": receive_remote,
        "nonempty_source_destination_pairs": nonempty_pairs,
        "nonempty_remote_source_destination_pairs": nonempty_remote_pairs,
        "total_assignments": total_assignments,
        "remote_assignments": remote_assignments,
        "remote_dispatch_hidden_bytes": remote_assignments * HIDDEN_SIZE * 2,
        "remote_combine_hidden_bytes": remote_assignments * HIDDEN_SIZE * 2,
        "remote_route_metadata_bytes": remote_assignments * 8,
    }
    return {
        "p2l": list(p2l),
        "l2p": list(l2p),
        "rank_logical_experts": rank_logical_experts,
        "rank_expert_counts": rank_expert_counts,
        "rank_raw_assignments": raw_rank,
        "rank_padded_rows": padded_rank,
        "rank_padding_waste_rows": padding_waste_rank,
        "rank_active_experts": active_rank,
        "rank_block_multisets": rank_block_multisets,
        "raw_max_over_mean": max(raw_rank) / (sum(raw_rank) / WORLD_SIZE),
        "padded_max_over_mean": max(padded_rank) / (sum(padded_rank) / WORLD_SIZE),
        "communication": communication,
    }


def make_placement(
    *,
    layout_id: str,
    category: str,
    p2l: Iterable[int],
    generation: dict[str, Any],
    metadata: RouteMetadata,
) -> Placement:
    canonical = validate_p2l(p2l)
    return Placement(
        layout_id=layout_id,
        category=category,
        p2l=canonical,
        l2p=invert_p2l(canonical),
        map_sha256=placement_map_sha256(canonical),
        generation=generation,
        descriptors=layout_descriptors(metadata, canonical),
    )


def common_communication_features(descriptors: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    communication = descriptors["communication"]
    names: list[str] = []
    values: list[float] = []
    for prefix, field in (
        ("comm_send_total", "send_total_assignments"),
        ("comm_receive_total", "receive_total_assignments"),
        ("comm_send_remote", "send_remote_assignments"),
        ("comm_receive_remote", "receive_remote_assignments"),
    ):
        vector = communication[field]
        for rank, value in enumerate(vector):
            names.append(f"{prefix}_rank{rank}")
            values.append(float(value))
    scalar_fields = (
        ("comm_nonempty_pairs", "nonempty_source_destination_pairs"),
        ("comm_nonempty_remote_pairs", "nonempty_remote_source_destination_pairs"),
        ("comm_total_assignments", "total_assignments"),
        ("comm_remote_assignments", "remote_assignments"),
        ("comm_remote_dispatch_hidden_bytes", "remote_dispatch_hidden_bytes"),
        ("comm_remote_combine_hidden_bytes", "remote_combine_hidden_bytes"),
        ("comm_remote_route_metadata_bytes", "remote_route_metadata_bytes"),
    )
    for name, field in scalar_fields:
        names.append(name)
        values.append(float(communication[field]))
    return names, np.asarray(values, dtype=np.float64)


def m0_compute_features(descriptors: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    sorted_loads = sorted(descriptors["rank_raw_assignments"], reverse=True)
    names = [f"m0_raw_rank_sorted_{index}" for index in range(WORLD_SIZE)]
    values = [float(value) for value in sorted_loads]
    names.append("m0_raw_rank_max")
    values.append(float(max(sorted_loads)))
    return names, np.asarray(values, dtype=np.float64)


def m1_compute_features(descriptors: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for prefix, field in (
        ("m1_padded_rank_sorted", "rank_padded_rows"),
        ("m1_padding_waste_rank_sorted", "rank_padding_waste_rows"),
        ("m1_active_experts_rank_sorted", "rank_active_experts"),
    ):
        vector = sorted(descriptors[field], reverse=True)
        for index, value in enumerate(vector):
            names.append(f"{prefix}_{index}")
            values.append(float(value))
    names.append("m1_padded_rank_max")
    values.append(float(max(descriptors["rank_padded_rows"])))
    return names, np.asarray(values, dtype=np.float64)


def m2_rank_features(block_multiset: Iterable[int]) -> tuple[list[str], np.ndarray]:
    blocks = sorted((int(value) for value in block_multiset), reverse=True)
    if len(blocks) != LOCAL_EXPERTS:
        raise ValueError("M2 requires all 16 local expert shapes")
    if min(blocks) < 0 or max(blocks) > MAX_BLOCKS_PER_EXPERT:
        raise ValueError("M2 block count is outside frozen histogram bounds")
    histogram = [blocks.count(value) for value in range(MAX_BLOCKS_PER_EXPERT + 1)]
    active = [value for value in blocks if value > 0]
    active_groups = len(active)
    total_blocks = sum(active)
    max_blocks = max(active, default=0)
    mean_active = float(np.mean(active)) if active else 0.0
    dispersion = float(np.std(active, ddof=0)) if active else 0.0
    names = [f"m2_block_histogram_{value}" for value in range(MAX_BLOCKS_PER_EXPERT + 1)]
    values: list[float] = [float(value) for value in histogram]
    summaries = (
        ("m2_active_groups", active_groups),
        ("m2_total_blocks", total_blocks),
        ("m2_max_blocks_per_expert", max_blocks),
        ("m2_mean_active_blocks", mean_active),
        ("m2_active_block_dispersion", dispersion),
        ("m2_interaction_active_x_total", active_groups * total_blocks),
        ("m2_interaction_active_x_max", active_groups * max_blocks),
        ("m2_interaction_total_x_dispersion", total_blocks * dispersion),
    )
    for name, value in summaries:
        names.append(name)
        values.append(float(value))
    return names, np.asarray(values, dtype=np.float64)


def m2_completion_compute_features(predicted_runner_us: Iterable[float]) -> tuple[list[str], np.ndarray]:
    values_by_rank = np.asarray(list(predicted_runner_us), dtype=np.float64)
    if values_by_rank.shape != (WORLD_SIZE,) or not np.isfinite(values_by_rank).all():
        raise ValueError("M2 completion aggregation requires eight finite rank predictions")
    ordered = np.sort(values_by_rank)[::-1]
    names = [f"m2_runner_rank_sorted_{index}" for index in range(WORLD_SIZE)]
    values = ordered.tolist()
    names.extend(("m2_runner_max", "m2_runner_std", "m2_runner_range"))
    values.extend((float(ordered[0]), float(np.std(ordered)), float(ordered[0] - ordered[-1])))
    return names, np.asarray(values, dtype=np.float64)


def join_features(*parts: tuple[list[str], np.ndarray]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[np.ndarray] = []
    for part_names, part_values in parts:
        if len(part_names) != len(part_values):
            raise ValueError("feature names and values differ in length")
        names.extend(part_names)
        values.append(np.asarray(part_values, dtype=np.float64))
    if len(names) != len(set(names)):
        raise ValueError("feature names are not unique")
    return names, np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def effect_features(
    identity: tuple[list[str], np.ndarray],
    candidate: tuple[list[str], np.ndarray],
) -> tuple[list[str], np.ndarray]:
    identity_names, identity_values = identity
    candidate_names, candidate_values = candidate
    if identity_names != candidate_names:
        raise ValueError("identity and candidate feature definitions differ")
    return identity_names, np.asarray(identity_values) - np.asarray(candidate_values)
