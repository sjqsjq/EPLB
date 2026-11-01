#!/usr/bin/env python3
"""Deterministic routes for the fixed-communication real-load scan."""

from __future__ import annotations

import hashlib
from array import array
from typing import Any

import torch


TOP_K = 8
LOCAL_EXPERTS = 16
EXPERT_ALIGNMENT = 128
ROUTE_ONE_PER_RANK = "one_per_rank"
ROUTE_FIXED7_ONE_DOUBLE = "fixed7_one_double"
EXPERT_POLICY_PACKED_HOT = "packed_hot_balanced_cold"


def _balanced_counts(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def _align(value: int, alignment: int = EXPERT_ALIGNMENT) -> int:
    return 0 if value == 0 else ((value + alignment - 1) // alignment) * alignment


def _ratio_max_mean(values: list[int]) -> float:
    return max(values) / (sum(values) / len(values))


def _ratio_max_min(values: list[int]) -> float:
    return max(values) / min(values)


def _allocate_experts(
    row_degrees: list[int], expert_degrees: list[int]
) -> list[list[int]]:
    """Realize a simple bipartite degree sequence via Havel--Hakimi."""
    if sum(row_degrees) != sum(expert_degrees):
        raise ValueError("row and expert degree sums differ")
    if any(degree < 0 or degree > 2 for degree in row_degrees):
        raise ValueError("each destination row must have degree zero, one, or two")

    capacity = list(row_degrees)
    assignments = [[] for _ in row_degrees]
    remaining = list(expert_degrees)
    while max(remaining, default=0) > 0:
        expert = min(
            (index for index, degree in enumerate(remaining) if degree == max(remaining)),
            default=-1,
        )
        degree = remaining[expert]
        candidates = sorted(
            (token for token, value in enumerate(capacity) if value > 0),
            key=lambda token: (-capacity[token], token),
        )
        if degree > len(candidates):
            raise ValueError(
                f"non-graphical route: expert {expert} needs {degree} distinct rows, "
                f"only {len(candidates)} remain"
            )
        for token in candidates[:degree]:
            assignments[token].append(expert)
            capacity[token] -= 1
        remaining[expert] = 0

    if any(capacity):
        raise ValueError(f"unfilled route row capacities: {capacity}")
    for token_assignments in assignments:
        if len(token_assignments) != len(set(token_assignments)):
            raise ValueError("one token selected the same local expert twice")
    actual = [0] * len(expert_degrees)
    for token_assignments in assignments:
        for expert in token_assignments:
            actual[expert] += 1
    if actual != expert_degrees:
        raise ValueError(f"expert degree mismatch: expected {expert_degrees}, got {actual}")
    return assignments


def _route_manifest(
    *,
    ids: list[list[int]],
    destination_degrees: list[list[int]],
    expert_counts: list[list[int]],
    total_tokens: int,
    world_size: int,
    route_kind: str,
    requested_r: float | None,
    target_hot_assignments: int | None,
    hot_rank: int | None,
    expert_policy: str | None,
) -> dict[str, Any]:
    destination_counts = [sum(row[rank] for row in destination_degrees) for rank in range(world_size)]
    unique_counts = [
        sum(row[rank] > 0 for row in destination_degrees) for rank in range(world_size)
    ]
    aligned_expert_counts = [
        [_align(value) for value in rank_counts] for rank_counts in expert_counts
    ]
    aligned_destination_counts = [sum(rank_counts) for rank_counts in aligned_expert_counts]
    local_tokens = total_tokens // world_size
    communication = []
    for source in range(world_size):
        begin = source * local_tokens
        end = begin + local_tokens
        communication.append(
            [
                sum(destination_degrees[token][destination] > 0 for token in range(begin, end))
                for destination in range(world_size)
            ]
        )

    flattened = array("i", (expert for row in ids for expert in row))
    route_hash = hashlib.sha256(flattened.tobytes()).hexdigest()
    return {
        "route_kind": route_kind,
        "requested_r": requested_r,
        "target_hot_assignments": target_hot_assignments,
        "hot_rank": hot_rank,
        "expert_policy": expert_policy,
        "route_sha256": route_hash,
        "destination_assignment_counts": destination_counts,
        "destination_unique_token_counts": unique_counts,
        "source_destination_unique_rows": communication,
        "expert_raw_counts": expert_counts,
        "expert_aligned_counts": aligned_expert_counts,
        "destination_aligned_rows": aligned_destination_counts,
        "raw_max_mean_ratio": _ratio_max_mean(destination_counts),
        "raw_max_min_ratio": _ratio_max_min(destination_counts),
        "aligned_max_mean_ratio": _ratio_max_mean(aligned_destination_counts),
        "aligned_max_min_ratio": _ratio_max_min(aligned_destination_counts),
    }


def _one_per_rank(total_tokens: int, world_size: int):
    ids = []
    destination_degrees = []
    expert_counts = [[0] * LOCAL_EXPERTS for _ in range(world_size)]
    for token in range(total_tokens):
        local_expert = token % LOCAL_EXPERTS
        ids.append(
            [destination * LOCAL_EXPERTS + local_expert for destination in range(world_size)]
        )
        destination_degrees.append([1] * world_size)
        for destination in range(world_size):
            expert_counts[destination][local_expert] += 1
    return ids, destination_degrees, expert_counts


def _fixed7_one_double(
    total_tokens: int,
    world_size: int,
    target_hot_assignments: int,
    hot_rank: int,
):
    if world_size != TOP_K:
        raise ValueError("fixed7_one_double requires top_k == world_size == 8")
    if hot_rank != 0:
        raise ValueError("the reproducible scan currently fixes hot_rank=0")
    if target_hot_assignments < total_tokens:
        raise ValueError("target hot load must be at least the mean load")

    destination_degrees: list[list[int]] = []
    for token in range(total_tokens):
        omitted = token % world_size
        duplicate = (omitted + 1) % world_size
        degrees = [int(destination != omitted) for destination in range(world_size)]
        degrees[duplicate] += 1
        destination_degrees.append(degrees)

    delta = target_hot_assignments - total_tokens
    candidates = [
        token
        for token, degrees in enumerate(destination_degrees)
        if degrees[hot_rank] == 1
        and degrees[(token % world_size + 1) % world_size] == 2
    ]
    if delta > len(candidates):
        raise ValueError(
            f"requested hot-load increment {delta} exceeds fixed-support capacity "
            f"{len(candidates)}"
        )
    for token in candidates[:delta]:
        duplicate = (token % world_size + 1) % world_size
        destination_degrees[token][duplicate] -= 1
        destination_degrees[token][hot_rank] += 1

    destination_counts = [
        sum(row[destination] for row in destination_degrees)
        for destination in range(world_size)
    ]
    expert_counts = []
    for destination, count in enumerate(destination_counts):
        if destination == hot_rank:
            base = total_tokens // LOCAL_EXPERTS
            counts = [base + delta] + [base] * (LOCAL_EXPERTS - 1)
            if sum(counts) != count:
                raise ValueError("packed hot expert counts do not sum to hot-rank load")
        else:
            counts = _balanced_counts(count, LOCAL_EXPERTS)
        expert_counts.append(counts)

    per_destination_assignments = []
    for destination in range(world_size):
        per_destination_assignments.append(
            _allocate_experts(
                [row[destination] for row in destination_degrees],
                expert_counts[destination],
            )
        )

    ids: list[list[int]] = []
    for token in range(total_tokens):
        row = []
        for destination in range(world_size):
            row.extend(
                destination * LOCAL_EXPERTS + local_expert
                for local_expert in sorted(per_destination_assignments[destination][token])
            )
        if len(row) != TOP_K or len(row) != len(set(row)):
            raise ValueError(f"invalid top-k row for token {token}: {row}")
        ids.append(row)
    return ids, destination_degrees, expert_counts


def build_route(
    *,
    total_tokens: int,
    rank: int,
    world_size: int,
    device: torch.device,
    route_kind: str,
    requested_r: float | None = None,
    target_hot_assignments: int | None = None,
    hot_rank: int | None = None,
    expert_policy: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if total_tokens % world_size:
        raise ValueError("total tokens must be divisible by world size")
    if world_size != TOP_K:
        raise ValueError(f"expected world size {TOP_K}, got {world_size}")

    if route_kind == ROUTE_ONE_PER_RANK:
        ids, destination_degrees, expert_counts = _one_per_rank(total_tokens, world_size)
    elif route_kind == ROUTE_FIXED7_ONE_DOUBLE:
        if target_hot_assignments is None or hot_rank is None:
            raise ValueError("fixed7_one_double requires target_hot_assignments and hot_rank")
        if expert_policy != EXPERT_POLICY_PACKED_HOT:
            raise ValueError(f"unsupported expert policy {expert_policy!r}")
        ids, destination_degrees, expert_counts = _fixed7_one_double(
            total_tokens, world_size, target_hot_assignments, hot_rank
        )
    else:
        raise ValueError(f"unknown route kind {route_kind!r}")

    manifest = _route_manifest(
        ids=ids,
        destination_degrees=destination_degrees,
        expert_counts=expert_counts,
        total_tokens=total_tokens,
        world_size=world_size,
        route_kind=route_kind,
        requested_r=requested_r,
        target_hot_assignments=target_hot_assignments,
        hot_rank=hot_rank,
        expert_policy=expert_policy,
    )
    local_tokens = total_tokens // world_size
    begin = rank * local_tokens
    end = begin + local_tokens
    topk_ids = torch.tensor(ids[begin:end], dtype=torch.int32, device=device).contiguous()
    topk_weights = torch.full(
        (local_tokens, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device=device,
    )
    return topk_ids, topk_weights, manifest
