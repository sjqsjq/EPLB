#!/usr/bin/env python3
"""Freeze identity plus 12 route-only cross-rank expert swaps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pathlib
import random
import statistics
from typing import Any, Callable

import torch


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty list")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


def invert_map(p2l: list[int]) -> list[int]:
    if sorted(p2l) != list(range(len(p2l))):
        raise ValueError("physical-to-logical map is not a permutation")
    l2p = [-1] * len(p2l)
    for physical, logical in enumerate(p2l):
        l2p[logical] = physical
    return l2p


def metric_record(
    *,
    p2l: list[int],
    expert_counts: list[int],
    expert_aligned: list[int],
    source_expert_counts: list[list[int]],
    world_size: int,
) -> dict[str, Any]:
    experts = len(p2l)
    local_experts = experts // world_size
    l2p = invert_map(p2l)
    rank_raw = []
    rank_aligned = []
    rank_logical_experts = []
    for rank in range(world_size):
        logical = p2l[rank * local_experts : (rank + 1) * local_experts]
        rank_logical_experts.append(logical)
        rank_raw.append(sum(expert_counts[expert] for expert in logical))
        rank_aligned.append(sum(expert_aligned[expert] for expert in logical))
    source_to_destination = [[0] * world_size for _ in range(world_size)]
    for source in range(world_size):
        for logical, count in enumerate(source_expert_counts[source]):
            destination = l2p[logical] // local_experts
            source_to_destination[source][destination] += count
    destination_totals = [
        sum(source_to_destination[source][destination] for source in range(world_size))
        for destination in range(world_size)
    ]
    if destination_totals != rank_raw:
        raise ValueError("communication matrix and rank loads disagree")
    raw_mean = statistics.fmean(rank_raw)
    aligned_mean = statistics.fmean(rank_aligned)
    return {
        "p2l": p2l,
        "l2p": l2p,
        "rank_logical_experts": rank_logical_experts,
        "rank_raw_assignments": rank_raw,
        "rank_aligned_rows": rank_aligned,
        "raw_max_assignments": max(rank_raw),
        "aligned_max_rows": max(rank_aligned),
        "raw_r_max_over_mean": max(rank_raw) / raw_mean,
        "aligned_r_max_over_mean": max(rank_aligned) / aligned_mean,
        "source_to_destination_assignments": source_to_destination,
    }


def continuous_pick(
    candidates: list[dict[str, Any]],
    count: int,
    score: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    return sorted(candidates, key=score)[:count]


def main() -> None:
    args = parse_args()
    if args.experts % args.world_size != 0:
        raise ValueError("experts must be divisible by world size")
    local_experts = args.experts // args.world_size
    paths = [args.snapshot_dir / f"snapshot_rank{rank}.pt" for rank in range(args.world_size)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing snapshot files: {missing}")

    source_topk: list[torch.Tensor] = []
    snapshot_meta = []
    capture_ids = set()
    layer_ids = set()
    map_hashes = set()
    for rank, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["rank"] != rank:
            raise ValueError(f"snapshot rank mismatch at {rank}")
        if not payload.get("identity_map_required"):
            raise ValueError("snapshot was not marked as identity-map capture")
        expected_logical = list(range(rank * local_experts, (rank + 1) * local_experts))
        if payload["logical_expert_ids"] != expected_logical:
            raise ValueError("snapshot local weights are not identity-mapped")
        topk_ids = payload["logical_topk_ids"].to(torch.int64)
        topk_weights = payload["logical_topk_weights"]
        hidden = payload["hidden_states"]
        if topk_ids.ndim != 2 or topk_ids.shape[1] != 8:
            raise ValueError(f"unexpected top-k shape on rank {rank}: {topk_ids.shape}")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("top-k IDs and weights disagree")
        if hidden.shape[0] != topk_ids.shape[0]:
            raise ValueError("hidden rows and top-k rows disagree")
        if topk_ids.numel() and (topk_ids.min() < 0 or topk_ids.max() >= args.experts):
            raise ValueError("logical top-k ID is out of range")
        source_topk.append(topk_ids.clone())
        capture_ids.add(payload["capture_id"])
        layer_ids.add(int(payload["layer_id"]))
        map_hashes.add(payload["identity_map_sha256"])
        snapshot_meta.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "hidden_rows": int(hidden.shape[0]),
                "num_token_non_padded": payload["num_token_non_padded"],
            }
        )
        del payload, topk_weights, hidden
        gc.collect()
    if len(capture_ids) != 1 or len(layer_ids) != 1 or len(map_hashes) != 1:
        raise ValueError("snapshot capture IDs, layers, or map hashes disagree")

    source_expert_counts = [
        torch.bincount(topk.flatten(), minlength=args.experts)[: args.experts]
        .to(torch.int64)
        .tolist()
        for topk in source_topk
    ]
    expert_counts = [
        sum(source_expert_counts[source][expert] for source in range(args.world_size))
        for expert in range(args.experts)
    ]
    expert_aligned = [
        math.ceil(count / args.alignment) * args.alignment for count in expert_counts
    ]
    baseline_p2l = list(range(args.experts))
    baseline = metric_record(
        p2l=baseline_p2l,
        expert_counts=expert_counts,
        expert_aligned=expert_aligned,
        source_expert_counts=source_expert_counts,
        world_size=args.world_size,
    )

    candidates = []
    for expert_a in range(args.experts):
        rank_a = expert_a // local_experts
        for expert_b in range(expert_a + 1, args.experts):
            rank_b = expert_b // local_experts
            if rank_a == rank_b:
                continue
            p2l = baseline_p2l.copy()
            p2l[expert_a], p2l[expert_b] = p2l[expert_b], p2l[expert_a]
            metrics = metric_record(
                p2l=p2l,
                expert_counts=expert_counts,
                expert_aligned=expert_aligned,
                source_expert_counts=source_expert_counts,
                world_size=args.world_size,
            )
            candidates.append(
                {
                    "candidate_id": f"swap_e{expert_a:03d}_e{expert_b:03d}",
                    "expert_a": expert_a,
                    "expert_b": expert_b,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "raw_max_improvement": baseline["raw_max_assignments"]
                    - metrics["raw_max_assignments"],
                    "aligned_max_improvement": baseline["aligned_max_rows"]
                    - metrics["aligned_max_rows"],
                    "raw_r_improvement": baseline["raw_r_max_over_mean"]
                    - metrics["raw_r_max_over_mean"],
                    "aligned_r_improvement": baseline["aligned_r_max_over_mean"]
                    - metrics["aligned_r_max_over_mean"],
                    "raw_vector_l1": sum(
                        abs(left - right)
                        for left, right in zip(
                            baseline["rank_raw_assignments"],
                            metrics["rank_raw_assignments"],
                            strict=True,
                        )
                    ),
                    "aligned_vector_l1": sum(
                        abs(left - right)
                        for left, right in zip(
                            baseline["rank_aligned_rows"],
                            metrics["rank_aligned_rows"],
                            strict=True,
                        )
                    ),
                    "metrics": metrics,
                }
            )
    candidates.sort(key=lambda item: item["candidate_id"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selection_log: dict[str, Any] = {}

    def remaining() -> list[dict[str, Any]]:
        return [item for item in candidates if item["candidate_id"] not in selected_ids]

    pool_a = [
        item
        for item in remaining()
        if item["raw_max_improvement"] > 0
        and item["aligned_max_improvement"] > 0
    ]
    picks_a = sorted(
        pool_a,
        key=lambda item: (
            -item["aligned_max_improvement"],
            -item["raw_max_improvement"],
            item["candidate_id"],
        ),
    )[:3]
    fallback_a = len(picks_a) < 3
    if fallback_a:
        need = 3 - len(picks_a)
        excluded = {item["candidate_id"] for item in picks_a}
        fallback_pool = [item for item in remaining() if item["candidate_id"] not in excluded]
        picks_a += continuous_pick(
            fallback_pool,
            need,
            lambda item: (
                -item["raw_r_improvement"] - item["aligned_r_improvement"],
                item["candidate_id"],
            ),
        )
    for index, item in enumerate(picks_a, 1):
        item["category"] = "A"
        item["layout_id"] = f"A{index}"
        selected.append(item)
        selected_ids.add(item["candidate_id"])
    a_raw = [item["raw_max_improvement"] for item in picks_a]
    a_aligned = [item["aligned_max_improvement"] for item in picks_a]
    a_min, a_max = min(a_raw), max(a_raw)
    a_median = statistics.median(a_raw)
    a_aligned_median = statistics.median(a_aligned)
    selection_log["A"] = {
        "rule": "positive raw and padded max improvement; rank by padded then raw improvement",
        "eligible_count": len(pool_a),
        "fallback_used": fallback_a,
    }

    pool_b_all = [item for item in remaining() if item["raw_max_improvement"] > 0]
    b_raw_tolerance = max(1.0, abs(a_median) * 0.10)
    b_padded_separation = args.alignment
    pool_b = [
        item
        for item in pool_b_all
        if abs(item["raw_max_improvement"] - a_median) <= b_raw_tolerance
        and abs(item["aligned_max_improvement"] - a_aligned_median)
        >= b_padded_separation
    ]
    picks_b = sorted(
        pool_b,
        key=lambda item: (
            abs(item["aligned_max_improvement"]),
            abs(item["raw_max_improvement"] - a_median),
            item["candidate_id"],
        ),
    )[:3]
    fallback_b = len(picks_b) < 3
    if fallback_b:
        need = 3 - len(picks_b)
        excluded = {item["candidate_id"] for item in picks_b}
        fallback_pool = [
            item for item in pool_b_all if item["candidate_id"] not in excluded
        ]
        picks_b += continuous_pick(
            fallback_pool,
            need,
            lambda item: (
                abs(item["raw_max_improvement"] - a_median),
                abs(item["aligned_max_improvement"]),
                item["candidate_id"],
            ),
        )
    for index, item in enumerate(picks_b, 1):
        item["category"] = "B"
        item["layout_id"] = f"B{index}"
        selected.append(item)
        selected_ids.add(item["candidate_id"])
    selection_log["B"] = {
        "rule": (
            "raw max improvement within 10% of the selected A median and padded "
            "max improvement separated from A by at least one 128-row step; "
            "minimize padded improvement, then raw-distance"
        ),
        "A_raw_improvement_range": [a_min, a_max],
        "A_raw_improvement_median": a_median,
        "A_aligned_improvement_median": a_aligned_median,
        "raw_comparability_tolerance": b_raw_tolerance,
        "minimum_padded_separation": b_padded_separation,
        "eligible_count": len(pool_b),
        "fallback_used": fallback_b,
    }

    raw_abs_values = [abs(item["raw_r_improvement"]) for item in remaining()]
    raw_q25 = percentile(raw_abs_values, 0.25)
    pool_c = [
        item
        for item in remaining()
        if abs(item["raw_r_improvement"]) <= raw_q25
        and item["aligned_vector_l1"] > 0
    ]
    picks_c = sorted(
        pool_c,
        key=lambda item: (
            -item["aligned_vector_l1"],
            -abs(item["aligned_r_improvement"]),
            abs(item["raw_r_improvement"]),
            item["candidate_id"],
        ),
    )[:3]
    fallback_c = len(picks_c) < 3
    if fallback_c:
        need = 3 - len(picks_c)
        excluded = {item["candidate_id"] for item in picks_c}
        fallback_pool = [item for item in remaining() if item["candidate_id"] not in excluded]
        picks_c += continuous_pick(
            fallback_pool,
            need,
            lambda item: (
                abs(item["raw_r_improvement"]),
                -item["aligned_vector_l1"],
                item["candidate_id"],
            ),
        )
    for index, item in enumerate(picks_c, 1):
        item["category"] = "C"
        item["layout_id"] = f"C{index}"
        selected.append(item)
        selected_ids.add(item["candidate_id"])
    selection_log["C"] = {
        "rule": (
            "bottom quartile of absolute raw-r change; maximize aligned-load "
            "vector change, then aligned-r change"
        ),
        "absolute_raw_r_change_q25": raw_q25,
        "eligible_count": len(pool_c),
        "fallback_used": fallback_c,
    }

    pool_d = remaining()
    if len(pool_d) < 3:
        raise ValueError("fewer than three candidates remain for category D")
    picks_d = random.Random(args.seed).sample(pool_d, 3)
    picks_d.sort(key=lambda item: item["candidate_id"])
    for index, item in enumerate(picks_d, 1):
        item["category"] = "D"
        item["layout_id"] = f"D{index}"
        selected.append(item)
        selected_ids.add(item["candidate_id"])
    selection_log["D"] = {
        "rule": "uniform sample without replacement from all remaining legal swaps",
        "eligible_count": len(pool_d),
        "seed": args.seed,
        "fallback_used": False,
    }

    layouts = [
        {
            "layout_id": "identity",
            "category": "baseline",
            "candidate_id": "identity",
            "swap": None,
            "metrics": baseline,
        }
    ]
    for item in selected:
        layouts.append(
            {
                "layout_id": item["layout_id"],
                "category": item["category"],
                "candidate_id": item["candidate_id"],
                "swap": {
                    "logical_expert_a": item["expert_a"],
                    "logical_expert_b": item["expert_b"],
                    "baseline_rank_a": item["rank_a"],
                    "baseline_rank_b": item["rank_b"],
                },
                "selection_evidence": {
                    key: item[key]
                    for key in (
                        "raw_max_improvement",
                        "aligned_max_improvement",
                        "raw_r_improvement",
                        "aligned_r_improvement",
                        "raw_vector_l1",
                        "aligned_vector_l1",
                    )
                },
                "metrics": item["metrics"],
            }
        )
    if len(layouts) != 13 or len(selected_ids) != 12:
        raise AssertionError("expected identity plus 12 unique swaps")

    result = {
        "schema_version": 1,
        "status": "frozen_before_timing",
        "capture_id": next(iter(capture_ids)),
        "layer_id": next(iter(layer_ids)),
        "identity_service_map_sha256": next(iter(map_hashes)),
        "world_size": args.world_size,
        "num_experts": args.experts,
        "num_local_experts": local_experts,
        "top_k": 8,
        "alignment_rows_per_expert": args.alignment,
        "raw_r_definition": "max rank assignments / mean rank assignments",
        "selection_uses_performance_measurements": False,
        "selection_seed": args.seed,
        "legal_cross_rank_swap_count": len(candidates),
        "snapshot_files": snapshot_meta,
        "total_hidden_rows": sum(item["hidden_rows"] for item in snapshot_meta),
        "total_valid_assignments": sum(expert_counts),
        "source_expert_counts": source_expert_counts,
        "expert_counts": expert_counts,
        "expert_aligned_counts": expert_aligned,
        "selection_log": selection_log,
        "layouts": layouts,
    }
    write_json_atomic(args.output, result)
    print(
        f"frozen {len(layouts)} layouts from {len(candidates)} legal swaps; "
        f"layer={result['layer_id']} hidden_rows={result['total_hidden_rows']}"
    )
    for layout in layouts:
        print(
            layout["layout_id"],
            layout["candidate_id"],
            f"raw_r={layout['metrics']['raw_r_max_over_mean']:.6f}",
            f"aligned_r={layout['metrics']['aligned_r_max_over_mean']:.6f}",
        )


if __name__ == "__main__":
    main()
