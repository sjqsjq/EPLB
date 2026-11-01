#!/usr/bin/env python3
"""Validate route-only profile dumps and preselect one layer without timing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

import torch


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--target-output", type=pathlib.Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=94)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--alignment", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experts % args.world_size != 0:
        raise ValueError("experts must be divisible by world size")
    local_experts = args.experts // args.world_size
    paths = [args.capture_dir / f"profile_rank{rank}.pt" for rank in range(args.world_size)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing profile files: {missing}")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    capture_ids = {payload["capture_id"] for payload in payloads}
    if len(capture_ids) != 1:
        raise ValueError(f"profile capture IDs disagree: {capture_ids}")
    for rank, payload in enumerate(payloads):
        if payload["rank"] != rank:
            raise ValueError(f"profile rank mismatch at {rank}: {payload['rank']}")
        if payload["expected_layers"] != args.layers:
            raise ValueError("profile layer count mismatch")
        if payload["num_experts"] != args.experts:
            raise ValueError("profile expert count mismatch")
        if len(payload["layers"]) != args.layers:
            raise ValueError("incomplete profile")

    layer_summaries = []
    for layer_id in range(args.layers):
        per_source = []
        hidden_rows = []
        non_padded = []
        for rank_payload in payloads:
            layer = rank_payload["layers"][layer_id]
            if layer["layer_id"] != layer_id:
                raise ValueError("profile layer ordering mismatch")
            counts = [int(value) for value in layer["logical_expert_counts"]]
            if len(counts) != args.experts:
                raise ValueError("logical expert count length mismatch")
            per_source.append(counts)
            hidden_rows.append(int(layer["hidden_rows"]))
            non_padded.append(layer["num_token_non_padded"])
        expert_counts = [
            sum(per_source[source][expert] for source in range(args.world_size))
            for expert in range(args.experts)
        ]
        rank_raw = [
            sum(expert_counts[rank * local_experts : (rank + 1) * local_experts])
            for rank in range(args.world_size)
        ]
        expert_aligned = [
            math.ceil(count / args.alignment) * args.alignment
            for count in expert_counts
        ]
        rank_aligned = [
            sum(expert_aligned[rank * local_experts : (rank + 1) * local_experts])
            for rank in range(args.world_size)
        ]
        raw_mean = sum(rank_raw) / args.world_size
        aligned_mean = sum(rank_aligned) / args.world_size
        layer_summaries.append(
            {
                "layer_id": layer_id,
                "hidden_rows_by_source_rank": hidden_rows,
                "num_token_non_padded_by_source_rank": non_padded,
                "total_hidden_rows": sum(hidden_rows),
                "total_assignments": sum(expert_counts),
                "expert_counts": expert_counts,
                "expert_aligned_counts": expert_aligned,
                "rank_raw_assignments": rank_raw,
                "rank_aligned_rows": rank_aligned,
                "raw_r_max_over_mean": max(rank_raw) / raw_mean,
                "aligned_r_max_over_mean": max(rank_aligned) / aligned_mean,
            }
        )
    selected = min(
        layer_summaries,
        key=lambda item: (-item["raw_r_max_over_mean"], item["layer_id"]),
    )
    result = {
        "schema_version": 1,
        "capture_id": next(iter(capture_ids)),
        "selection_rule": (
            "select the layer with maximum baseline identity-placement raw "
            "rank assignments divided by the eight-rank mean; break ties by "
            "the smallest layer ID; no placement timing is used"
        ),
        "raw_r_definition": "max rank assignments / mean rank assignments",
        "alignment_rows_per_expert": args.alignment,
        "profile_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ],
        "selected_layer": selected["layer_id"],
        "selected_layer_summary": selected,
        "layers": layer_summaries,
    }
    write_json_atomic(args.output, result)
    write_json_atomic(
        args.target_output,
        {
            "schema_version": 1,
            "selected_layer": selected["layer_id"],
            "selection_rule": result["selection_rule"],
            "profile_summary_sha256": sha256(args.output),
        },
    )
    print(
        f"selected_layer={selected['layer_id']} "
        f"raw_r={selected['raw_r_max_over_mean']:.6f} "
        f"total_hidden_rows={selected['total_hidden_rows']}"
    )


if __name__ == "__main__":
    main()
