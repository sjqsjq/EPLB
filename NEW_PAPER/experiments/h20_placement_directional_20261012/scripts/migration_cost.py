#!/usr/bin/env python3
"""Measure real full-tensor expert migration cost for frozen stopping swaps.

Frozen design (FOLLOWUP_PROTOCOL.md section C, "Separate real migration experiment"):
- Migration is a separate transaction, never part of execution-only Q. The old
  bench_p2p_cost.py artifact is inadmissible (it reported w13+w2 bytes while
  moving only w13 and omitted scale tensors).
- For every measured swap the complete FP8 w13 and w2 tensors plus both scale
  tensors are exchanged between their actual H20 owners over the real
  point-to-point fabric (NCCL batch_isend_irecv on uint8 views; no pointer
  aliasing, no pre-resident duplicate copies: staging destinations are zeroed
  before every timed transfer and every received tensor is sha256-verified
  against the canonical artifact hash outside timing).
- Allocation, communicators, staging buffers, canonical hashes, and CUDA events
  are prepared before timing. The timed transaction begins after a start
  alignment, includes the actual transfer, completion acknowledgement, and
  mapping publication, and ends on rank 0's monotonic clock after every rank
  acknowledged readiness. No cross-device timestamp subtraction: the coordinator
  wall time is the primary cost C; same-device CUDA event intervals around each
  rank's local transfer issue+wait are recorded separately as diagnostics.
- Each frozen single swap and the actual multi-swap batch run with 50 fixed
  forward/reverse transaction pairs per start; all repetitions are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import socket
import time
from typing import Any

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")

import torch
import torch.distributed as dist

from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    WEIGHT_FIELDS,
    WORLD_SIZE,
    indexed_expert_records,
    indexed_units,
    invert_p2l,
    load_capture_index,
    load_tensor_payload,
    placement_map_sha256,
    sha256_file,
    validate_expert_payload,
    validate_forward_payload,
    verify_file_record,
    write_json_atomic,
)

STOPPING_STUDY_ID = "h20-placement-stopping-v1"
PROTOCOL_STATUS = "frozen_before_stopping_migration_timing"
PAIRS_PER_TRANSACTION = 50
EXPECTED_BYTES = {
    "w13_weight": 2 * 1536 * 4096 * 1,
    "w2_weight": 4096 * 1536 * 1,
    "w13_weight_scale_inv": 24 * 32 * 4,
    "w2_weight_scale_inv": 32 * 12 * 4,
}
BYTES_PER_EXPERT = sum(EXPECTED_BYTES.values())
FULL_HASH_PAIRS = (1, PAIRS_PER_TRANSACTION)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--start-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = args.protocol.resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported stopping migration protocol schema")
    if payload.get("study_id") != STOPPING_STUDY_ID:
        raise ValueError("stopping migration protocol belongs to another study")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("stopping migration protocol is not frozen before timing")
    implementation = payload["implementation"]
    checks = {
        pathlib.Path(__file__).resolve(): implementation["migration_cost_record"]["sha256"],
        pathlib.Path(__file__).with_name("prediction_common.py").resolve(): implementation[
            "prediction_common_record"
        ]["sha256"],
    }
    for implementation_path, expected in checks.items():
        actual = sha256_file(implementation_path)
        if actual != expected:
            raise ValueError(
                f"frozen implementation hash differs for {implementation_path}: "
                f"{actual} != {expected}"
            )
    starts = {record["start_id"]: record for record in payload["starts"]}
    if args.start_id not in starts:
        raise ValueError(f"start ID is absent from protocol: {args.start_id}")
    start = starts[args.start_id]
    expected_output = (args.root / start["output_relative_path"]).resolve()
    if args.output.resolve() != expected_output:
        raise ValueError(
            f"output path differs from frozen start: {args.output.resolve()} != {expected_output}"
        )
    return payload, start


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def merkle(entries: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(entries)).encode()).hexdigest()


def align(marker: torch.Tensor, group) -> None:
    marker.fill_(1.0)
    dist.all_reduce(marker, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()


def preconnect_p2p(*, rank: int, device, group) -> dict[str, Any]:
    """Untimed one-byte all-pairs exchange that establishes every NCCL P2P
    channel before any timed transfer, per the frozen design requirement that
    communicators are prepared outside the timed region."""
    byte = torch.zeros(1, dtype=torch.uint8, device=device)
    ops = []
    for peer in range(WORLD_SIZE):
        if peer == rank:
            continue
        ops.append(dist.P2POp(dist.isend, byte, peer, group=group))
        ops.append(dist.P2POp(dist.irecv, byte, peer, group=group))
    t0 = time.monotonic_ns()
    requests = dist.batch_isend_irecv(ops)
    for request in requests:
        request.wait()
    torch.cuda.synchronize(device)
    wall_us = (time.monotonic_ns() - t0) / 1000.0
    dist.barrier(group=group)
    return {
        "mode": "untimed_one_byte_all_pairs_batch_isend_irecv",
        "peers": WORLD_SIZE - 1,
        "local_wall_us": wall_us,
        "performed_before_any_timed_transfer": True,
    }


class MigrationState:
    def __init__(self, *, rank: int, device, group):
        self.rank = rank
        self.device = device
        self.group = group
        self.canonical: dict[tuple[int, str], torch.Tensor] = {}
        self.canonical_hashes: dict[str, str] = {}  # "logical:field" -> sha256
        self.staging: dict[tuple[str, int, str, str], torch.Tensor] = {}  # (tx, logical, field, slot)
        self.views: dict[int, torch.Tensor] = {}  # id(tensor) -> uint8 flat view
        self.live: dict[int, dict[str, torch.Tensor]] = {}  # logical -> field -> resident tensor
        self.owners: list[int] = [logical // LOCAL_EXPERTS for logical in range(NUM_EXPERTS)]

    def uint8_view(self, tensor: torch.Tensor) -> torch.Tensor:
        key = id(tensor)
        if key not in self.views:
            self.views[key] = tensor.detach().view(-1).view(torch.uint8)
        return self.views[key]

    def allocate_staging(self, transaction_id: str, logical: int, field: str, slot: str) -> torch.Tensor:
        key = (transaction_id, logical, field, slot)
        if key not in self.staging:
            self.staging[key] = torch.zeros(
                EXPECTED_BYTES[field], dtype=torch.uint8, device=self.device
            )
            self.views[id(self.staging[key])] = self.staging[key]
        return self.staging[key]

    def owners_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.owners).encode()).hexdigest()


def publish_and_ack(
    state: MigrationState,
    new_owners: list[int],
    switches: list[tuple[int, str, torch.Tensor]],
    sent_logicals: list[int],
    marker: torch.Tensor,
) -> None:
    """Inside-timed-region: ack transfer, publish mapping, verify agreement, ready ack."""
    dist.all_reduce(marker, op=dist.ReduceOp.SUM, group=state.group)  # completion ack
    gathered: list[Any] = [None] * WORLD_SIZE
    local_owners = list(new_owners)
    dist.all_gather_object(gathered, (state.rank, local_owners), group=state.group)
    tables = {rank: owners for rank, owners in gathered}
    reference = tables[0]
    agree = all(tables[rank] == reference for rank in range(WORLD_SIZE))
    payload: list[Any] = [agree]
    dist.broadcast_object_list(payload, src=0, group=state.group)
    if not payload[0]:
        raise ValueError("mapping publication disagreement across ranks")
    for logical, field, tensor in switches:
        if logical not in state.live:
            state.live[logical] = {}
        state.live[logical][field] = tensor
    for logical in sent_logicals:
        state.live.pop(logical, None)
    state.owners = list(new_owners)
    marker.fill_(1.0)
    dist.all_reduce(marker, op=dist.ReduceOp.SUM, group=state.group)  # readiness ack


def run_direction(
    *,
    state: MigrationState,
    transaction: dict[str, Any],
    direction: str,
    moves: list[dict[str, Any]],
    new_owners: list[int],
    marker: torch.Tensor,
    pair_index: int,
) -> dict[str, Any]:
    """One timed forward or reverse transaction direction."""
    # Untimed setup: zero every destination buffer for this direction.
    local_recvs: list[tuple[int, str, torch.Tensor, int]] = []
    local_sends: list[tuple[int, str, torch.Tensor, int]] = []
    for move in moves:
        logical = int(move["logical"])
        from_rank = int(move["from_rank"])
        to_rank = int(move["to_rank"])
        slot = "fwd" if direction == "forward" else "rev"
        if to_rank == state.rank:
            for field in WEIGHT_FIELDS:
                buffer = state.allocate_staging(transaction["transaction_id"], logical, field, slot)
                buffer.zero_()
                local_recvs.append((logical, field, buffer, from_rank))
        if from_rank == state.rank:
            if logical not in state.live:
                raise ValueError(
                    f"frozen move sends logical {logical} from rank {state.rank} "
                    "but the expert is not resident there"
                )
            for field in WEIGHT_FIELDS:
                source = state.live[logical][field]
                local_sends.append((logical, field, source, to_rank))
    torch.cuda.synchronize(state.device)

    events = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    align(marker, state.group)
    coordinator_t0 = time.monotonic_ns() if state.rank == 0 else None
    events[0].record()
    ops = []
    for logical, field, tensor, peer in local_sends:
        ops.append(dist.P2POp(dist.isend, state.uint8_view(tensor), peer, group=state.group))
    for logical, field, buffer, peer in local_recvs:
        ops.append(dist.P2POp(dist.irecv, state.uint8_view(buffer), peer, group=state.group))
    requests = dist.batch_isend_irecv(ops) if ops else []
    for request in requests:
        request.wait()
    events[1].record()
    switches = [
        (logical, field, buffer)
        for logical, field, buffer, _peer in local_recvs
    ]
    sent_logicals = sorted({logical for logical, _field, _tensor, _peer in local_sends})
    publish_and_ack(state, new_owners, switches, sent_logicals, marker)
    coordinator_t1 = time.monotonic_ns() if state.rank == 0 else None

    # Untimed verification: received-tensor sha256 versus canonical hashes.
    received_entries = []
    full_hashes: dict[str, str] = {}
    for logical, field, buffer, _peer in local_recvs:
        digest = sha256_tensor(buffer[: EXPECTED_BYTES[field]].view(torch.uint8))
        expected = state.canonical_hashes[f"{logical}:{field}"]
        if digest != expected:
            raise ValueError(
                f"post-transfer hash mismatch: logical={logical} field={field} "
                f"direction={direction} pair={pair_index}"
            )
        received_entries.append(f"{logical}:{field}:{digest}")
        full_hashes[f"{logical}:{field}"] = digest
    local_merkle = merkle(received_entries)
    torch.cuda.synchronize(state.device)
    cuda_us = events[0].elapsed_time(events[1]) * 1000.0
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered,
        (
            state.rank,
            local_merkle,
            cuda_us,
            len(local_recvs),
            sorted(full_hashes.items()) if pair_index in FULL_HASH_PAIRS else [],
        ),
        group=state.group,
    )
    gathered = sorted(gathered, key=lambda entry: entry[0])
    if state.rank == 0:
        # Independent recomputation of every receiving rank's merkle from the
        # frozen move list and the canonical hash table (untimed).
        expected_entries: dict[int, list[str]] = {r: [] for r in range(WORLD_SIZE)}
        for move in moves:
            logical = int(move["logical"])
            to_rank = int(move["to_rank"])
            for field in WEIGHT_FIELDS:
                expected_entries[to_rank].append(
                    f"{logical}:{field}:{state.canonical_hashes[f'{logical}:{field}']}"
                )
        for entry_rank, entry_merkle, _cuda, _count, _hashes in gathered:
            if entry_merkle != merkle(expected_entries[entry_rank]):
                raise ValueError(
                    f"received merkle differs from frozen-move expectation: rank={entry_rank} "
                    f"direction={direction} pair={pair_index}"
                )
    result = {
        "direction": direction,
        "pair_index": pair_index,
        "coordinator_wall_us": (
            (coordinator_t1 - coordinator_t0) / 1000.0
            if coordinator_t0 is not None
            else None
        ),
        "local_cuda_event_us": cuda_us,
        "local_received_merkle": local_merkle,
        "local_received_count": len(local_recvs),
        "local_sent_count": len(local_sends),
        "ranks": gathered,
    }
    if pair_index in FULL_HASH_PAIRS:
        result["full_received_hashes"] = {str(entry[0]): entry[4] for entry in gathered}
    return result


def owners_after(owners: list[int], moves: list[dict[str, Any]]) -> list[int]:
    updated = list(owners)
    for move in moves:
        updated[int(move["logical"])] = int(move["to_rank"])
    return updated


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    protocol, start = load_protocol(args)
    output_path = args.output.resolve()
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite start output: {output_path}")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise ValueError(f"migration cost requires eight ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(list(range(world_size)), backend="nccl")
    torch.set_grad_enabled(False)

    properties = torch.cuda.get_device_properties(device)
    gpu_inventory: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gpu_inventory,
        {
            "rank": rank,
            "pid": os.getpid(),
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        },
        group=group,
    )
    if any(record["name"] != "NVIDIA H20" for record in gpu_inventory):
        raise ValueError("migration cost requires exactly eight NVIDIA H20 GPUs")

    capture_index_record = protocol["capture_index"]
    capture_index_path = pathlib.Path(capture_index_record["path"]).resolve()
    if sha256_file(capture_index_path) != capture_index_record["sha256"]:
        raise ValueError("capture-index hash differs from migration protocol")
    index = load_capture_index(capture_index_path)
    units = {unit.capture_id: unit for unit in indexed_units(index, split="evaluation")}
    unit = units[start["capture_id"]]
    if unit.batch_id != start["batch_id"] or unit.layer_id != start["layer_id"]:
        raise ValueError("start/capture metadata mismatch")
    expert_records = indexed_expert_records(index)

    local_forward_path = verify_file_record(unit.forward_records[rank])
    local_forward = load_tensor_payload(local_forward_path)
    forward_meta = validate_forward_payload(
        local_forward, unit=unit, rank=rank, path=local_forward_path, allowed_splits={"evaluation"}
    )
    local_expert_record = expert_records[(unit.layer_id, rank)]
    local_expert_path = verify_file_record(local_expert_record)
    local_expert = load_tensor_payload(local_expert_path)
    expert_meta = validate_expert_payload(
        local_expert,
        layer_id=unit.layer_id,
        rank=rank,
        path=local_expert_path,
        identity_map_sha256=forward_meta["identity_map_sha256"],
    )

    layout_record = start["layout_manifest"]
    layout_path = pathlib.Path(layout_record["path"]).resolve()
    if sha256_file(layout_path) != layout_record["sha256"]:
        raise ValueError(f"layout manifest hash differs: {layout_path}")
    manifest = json.loads(layout_path.read_text())
    if manifest.get("split") != "stopping" or manifest.get("capture_id") != unit.capture_id:
        raise ValueError("stopping layout manifest mismatch")
    layouts = {layout["layout_id"]: layout for layout in manifest["layouts"]}
    base_p2l = tuple(int(value) for value in layouts["base"]["p2l"])
    base_l2p = invert_p2l(base_p2l)
    if placement_map_sha256(base_p2l) != layouts["base"]["map_sha256"]:
        raise ValueError("base map hash differs")

    state = MigrationState(rank=rank, device=device, group=group)
    local_hashes: dict[str, str] = {}
    for slot, logical in enumerate(range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)):
        tensors = {field: local_expert[field][slot].to(device).contiguous() for field in WEIGHT_FIELDS}
        for field in WEIGHT_FIELDS:
            tensor = tensors[field]
            if tensor.numel() * tensor.element_size() != EXPECTED_BYTES[field]:
                raise ValueError(f"expert tensor byte count differs: {field}")
            state.canonical[(logical, field)] = tensor
            digest = sha256_tensor(tensor)
            local_hashes[f"{logical}:{field}"] = digest
        state.live[logical] = {field: state.canonical[(logical, field)] for field in WEIGHT_FIELDS}
    gathered_hashes: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_hashes, local_hashes, group=group)
    for shard in gathered_hashes:
        state.canonical_hashes.update(shard)
    if len(state.canonical_hashes) != NUM_EXPERTS * len(WEIGHT_FIELDS):
        raise ValueError("canonical hash table is incomplete")

    transactions = protocol["transactions"][unit.capture_id]
    if [tx["transaction_id"] for tx in transactions] != start["transaction_order"]:
        raise ValueError("transaction order differs from frozen start")
    for transaction in transactions:
        if transaction["kind"] == "multi_swap_batch":
            expected_moves = sorted(
                logical for logical in range(NUM_EXPERTS)
                if logical // LOCAL_EXPERTS != base_l2p[logical] // LOCAL_EXPERTS
            )
            actual_moves = sorted(int(move["logical"]) for move in transaction["moves"])
            if actual_moves != expected_moves:
                raise ValueError("multi-swap batch moves differ from base displacement set")
        for move in transaction["moves"]:
            if int(transaction["bytes_per_expert"]) != BYTES_PER_EXPERT:
                raise ValueError("protocol byte accounting differs from tensor sizes")

    marker = torch.ones(1, dtype=torch.float32, device=device)
    preconnect = preconnect_p2p(rank=rank, device=device, group=group)
    started_at = dt.datetime.now(dt.timezone.utc)

    setup_records = []
    transaction_results = []
    for transaction in transactions:
        moves_forward = [
            {"logical": int(move["logical"]), "from_rank": int(move["from_rank"]), "to_rank": int(move["to_rank"])}
            for move in transaction["moves"]
        ]
        moves_reverse = [
            {"logical": move["logical"], "from_rank": move["to_rank"], "to_rank": move["from_rank"]}
            for move in moves_forward
        ]
        owners_before = list(state.owners)
        owners_mid = owners_after(owners_before, moves_forward)
        owners_after_pair = owners_after(owners_mid, moves_reverse)
        if owners_after_pair != owners_before:
            raise ValueError("forward/reverse pair does not restore the prior placement")
        pairs = []
        for pair_index in range(1, PAIRS_PER_TRANSACTION + 1):
            forward = run_direction(
                state=state, transaction=transaction, direction="forward",
                moves=moves_forward, new_owners=owners_mid, marker=marker, pair_index=pair_index,
            )
            reverse = run_direction(
                state=state, transaction=transaction, direction="reverse",
                moves=moves_reverse, new_owners=owners_after_pair, marker=marker, pair_index=pair_index,
            )
            if rank == 0:
                pairs.append(
                    {
                        "pair_index": pair_index,
                        "forward_wall_us": forward["coordinator_wall_us"],
                        "reverse_wall_us": reverse["coordinator_wall_us"],
                        "forward_cuda_us_by_rank": [entry[2] for entry in forward["ranks"]],
                        "reverse_cuda_us_by_rank": [entry[2] for entry in reverse["ranks"]],
                        "forward_merkle_by_rank": {str(entry[0]): entry[1] for entry in forward["ranks"]},
                        "reverse_merkle_by_rank": {str(entry[0]): entry[1] for entry in reverse["ranks"]},
                        "forward_received_counts": {str(entry[0]): entry[3] for entry in forward["ranks"]},
                        "hashes_all_matched_canonical": True,
                    }
                )
                if pair_index in FULL_HASH_PAIRS:
                    pairs[-1]["forward_full_hashes"] = forward["full_received_hashes"]
                    pairs[-1]["reverse_full_hashes"] = reverse["full_received_hashes"]
            del forward, reverse
        if transaction["kind"] == "multi_swap_batch":
            # Untimed setup migration identity -> base: after the multi-batch
            # pairs the placement is identity again; the single swaps are frozen
            # relative to base owners, so the real tensors are migrated to base
            # once here (outside all timed regions).
            moves = [
                {"logical": int(move["logical"]), "from_rank": int(move["from_rank"]), "to_rank": int(move["to_rank"])}
                for move in transaction["moves"]
            ]
            new_owners = owners_after(list(state.owners), moves)
            record = run_direction(
                state=state, transaction={"transaction_id": "setup-identity-to-base"},
                direction="forward", moves=moves, new_owners=new_owners,
                marker=marker, pair_index=0,
            )
            record.pop("coordinator_wall_us")
            setup_records.append({"role": "untimed_setup_migration_identity_to_base", **record})
        if rank == 0:
            walls_f = [pair["forward_wall_us"] for pair in pairs]
            walls_r = [pair["reverse_wall_us"] for pair in pairs]
            transaction_results.append(
                {
                    "transaction_id": transaction["transaction_id"],
                    "kind": transaction["kind"],
                    "moves": transaction["moves"],
                    "bytes_per_expert": transaction["bytes_per_expert"],
                    "bytes_per_direction": transaction["bytes_per_direction"],
                    "pairs": pairs,
                    "summary": {
                        "forward_wall_us": {"mean": sum(walls_f) / len(walls_f), "min": min(walls_f), "max": max(walls_f)},
                        "reverse_wall_us": {"mean": sum(walls_r) / len(walls_r), "min": min(walls_r), "max": max(walls_r)},
                        "all_pairs_retained": len(pairs),
                    },
                }
            )

    # Untimed end-of-start integrity: canonical tensors unchanged; live table = base owners.
    canonical_drift = []
    for (logical, field), tensor in sorted(state.canonical.items()):
        digest = sha256_tensor(tensor)
        if digest != state.canonical_hashes[f"{logical}:{field}"]:
            canonical_drift.append(f"{logical}:{field}")
    if canonical_drift:
        raise ValueError(f"canonical expert tensors changed during migration: {canonical_drift}")
    live_drift = []
    owned = {logical for logical in range(NUM_EXPERTS) if state.owners[logical] == rank}
    if set(state.live) != owned:
        raise ValueError("live resident set differs from published ownership")
    for logical in sorted(owned):
        for field in WEIGHT_FIELDS:
            digest = sha256_tensor(state.live[logical][field])
            if digest != state.canonical_hashes[f"{logical}:{field}"]:
                live_drift.append(f"{logical}:{field}")
    if live_drift:
        raise ValueError(f"live resident tensors corrupted: {live_drift}")
    expected_owners = [base_l2p[logical] // LOCAL_EXPERTS for logical in range(NUM_EXPERTS)]
    if state.owners != expected_owners:
        raise ValueError("end-of-start placement is not the frozen base")

    finished_at = dt.datetime.now(dt.timezone.utc)
    if rank == 0:
        result = {
            "schema_version": 1,
            "study_id": STOPPING_STUDY_ID,
            "experiment": "h20_placement_stopping_migration_cost",
            "split": "stopping_migration",
            "start_id": start["start_id"],
            "capture_id": unit.capture_id,
            "batch_id": unit.batch_id,
            "layer_id": unit.layer_id,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "hostname": socket.gethostname(),
            "gpu_inventory": gpu_inventory,
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol.resolve()),
                "status": protocol["status"],
            },
            "capture_index": capture_index_record,
            "layout_manifest": {
                "path": str(layout_path),
                "sha256": layout_record["sha256"],
                "base_map_sha256": layouts["base"]["map_sha256"],
            },
            "expert_artifact": {**expert_meta, **local_expert_record},
            "canonical_hash_table_sha256": merkle(
                [f"{key}:{value}" for key, value in state.canonical_hashes.items()]
            ),
            "byte_accounting": {
                "per_tensor_bytes": EXPECTED_BYTES,
                "per_expert_bytes": BYTES_PER_EXPERT,
                "scale_tensors_included": True,
                "old_bench_p2p_cost_admissible": False,
            },
            "timing_semantics": {
                "primary": "rank-0 monotonic wall from post-alignment t0 through transfer, completion ack, mapping publication, and readiness ack",
                "cuda_events": "same-device CUDA event interval around local transfer issue+wait; diagnostic only",
                "cross_device_cuda_timestamp_subtraction": False,
                "communicators_preconnected_before_timing": True,
                "setup_excluded_from_timing": True,
                "hash_verification_excluded_from_timing": True,
                "aliasing_or_preresident_copies": False,
                "staging_zeroed_before_every_timed_transfer": True,
            },
            "p2p_preconnect": preconnect,
            "untimed_setup": setup_records,
            "transactions": transaction_results,
            "end_of_start_integrity": {
                "canonical_tensors_unchanged": True,
                "live_resident_tensors_verified": True,
                "final_placement_equals_frozen_base": True,
            },
        }
        core_write(output_path, result)
        print(f"wrote {output_path}")

    dist.barrier(group=group)
    dist.destroy_process_group()


def core_write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


if __name__ == "__main__":
    main()
