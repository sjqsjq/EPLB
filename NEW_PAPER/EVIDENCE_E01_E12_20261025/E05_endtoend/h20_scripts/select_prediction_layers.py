#!/usr/bin/env python3
"""Select bridge, calibration, and held-out layers from screening routes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

import torch


BRIDGE_LAYER = 24
THIRDS = (
    ("early", 0, 31),
    ("middle", 31, 62),
    ("late", 62, 94),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json_no_clobber(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite layer artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale layer-artifact temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=pathlib.Path, required=True)
    parser.add_argument("--request-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--layer-manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=94)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--alignment", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layers != 94:
        raise ValueError("prediction v1 freezes exactly 94 routed layers")
    if args.world_size != 8 or args.experts != 128:
        raise ValueError("prediction v1 is restricted to 8 ranks and 128 experts")
    if args.experts % args.world_size != 0:
        raise ValueError("experts must be divisible by world size")
    local_experts = args.experts // args.world_size

    request_manifest = json.loads(args.request_manifest.read_text())
    if request_manifest.get("study_id") != "h20-placement-prediction-v1":
        raise ValueError("request manifest has the wrong study ID")
    screening = [
        batch for batch in request_manifest["batches"]
        if batch["split"] == "screening"
    ]
    if len(screening) != 1:
        raise ValueError("prediction v1 requires exactly one screening batch")

    paths = [
        args.capture_dir / f"profile_rank{rank}.pt"
        for rank in range(args.world_size)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing profile files: {missing}")
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    capture_ids = {payload["capture_id"] for payload in payloads}
    if len(capture_ids) != 1:
        raise ValueError(f"profile capture IDs disagree: {capture_ids}")
    contexts = {
        json.dumps(payload["capture_context"], sort_keys=True)
        for payload in payloads
    }
    if len(contexts) != 1:
        raise ValueError("profile capture contexts disagree across ranks")
    context = payloads[0]["capture_context"]
    if context["batch_id"] != screening[0]["batch_id"]:
        raise ValueError("profile did not use the frozen screening batch")
    if context["prediction_request_manifest"]["sha256"] != sha256(
        args.request_manifest
    ):
        raise ValueError("profile request-manifest hash mismatch")
    for rank, payload in enumerate(payloads):
        if payload.get("schema_version") != 2:
            raise ValueError("profile does not use split-capture schema v2")
        if payload["study_id"] != "h20-placement-prediction-v1":
            raise ValueError("profile has the wrong study ID")
        if payload["rank"] != rank:
            raise ValueError(f"profile rank mismatch at {rank}")
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
            if len(counts) != args.experts or any(value < 0 for value in counts):
                raise ValueError("invalid logical-expert counts")
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
            0 if count == 0 else math.ceil(count / args.alignment) * args.alignment
            for count in expert_counts
        ]
        rank_aligned = [
            sum(expert_aligned[rank * local_experts : (rank + 1) * local_experts])
            for rank in range(args.world_size)
        ]
        raw_mean = sum(rank_raw) / args.world_size
        aligned_mean = sum(rank_aligned) / args.world_size
        if raw_mean <= 0 or aligned_mean <= 0:
            raise ValueError("screening produced an empty routed layer")
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

    third_winners = []
    for third_name, begin, end in THIRDS:
        ranked = sorted(
            layer_summaries[begin:end],
            key=lambda item: (-item["raw_r_max_over_mean"], item["layer_id"]),
        )
        winner = next(item for item in ranked if item["layer_id"] != BRIDGE_LAYER)
        third_winners.append(
            {
                "third": third_name,
                "range_begin_inclusive": begin,
                "range_end_exclusive": end,
                "winner": winner,
                "bridge_duplicate_resolved": ranked[0]["layer_id"] == BRIDGE_LAYER,
                "top_ranked_layer_before_bridge_resolution": ranked[0]["layer_id"],
            }
        )
    winner_summaries = [item["winner"] for item in third_winners]
    second_calibration = min(
        winner_summaries,
        key=lambda item: (-item["raw_r_max_over_mean"], item["layer_id"]),
    )
    calibration_ids = {BRIDGE_LAYER, second_calibration["layer_id"]}
    selected_ids = {BRIDGE_LAYER, *(item["layer_id"] for item in winner_summaries)}
    if len(selected_ids) != 4 or len(calibration_ids) != 2:
        raise AssertionError("layer selection did not produce four unique layers")

    third_by_layer = {
        item["winner"]["layer_id"]: item["third"] for item in third_winners
    }
    selected_layers = []
    for layer_id in sorted(selected_ids):
        summary = layer_summaries[layer_id]
        calibration = layer_id in calibration_ids
        selected_layers.append(
            {
                "layer_id": layer_id,
                "bridge": layer_id == BRIDGE_LAYER,
                "depth_third_winner": third_by_layer.get(layer_id),
                "calibration": calibration,
                "held_out": not calibration,
                "raw_r_max_over_mean": summary["raw_r_max_over_mean"],
                "aligned_r_max_over_mean": summary["aligned_r_max_over_mean"],
                "total_hidden_rows": summary["total_hidden_rows"],
                "total_assignments": summary["total_assignments"],
            }
        )

    common = {
        "schema_version": 1,
        "study_id": "h20-placement-prediction-v1",
        "status": "layers_frozen_from_screening_routes_before_model_timing",
        "selection_uses_performance_timing": False,
        "selection_rule": {
            "depth_thirds": [
                {
                    "name": name,
                    "begin_inclusive": begin,
                    "end_exclusive": end,
                }
                for name, begin, end in THIRDS
            ],
            "bridge_layer": BRIDGE_LAYER,
            "depth_winner": (
                "within each depth third rank layers by descending screening "
                "identity-placement raw max/mean imbalance and then ascending "
                "layer ID; if layer 24 would duplicate the bridge, take the "
                "next-ranked layer in that same third"
            ),
            "calibration_assignment": (
                "use layer 24 plus the depth-third winner with largest screening "
                "raw max/mean imbalance, breaking ties by lower layer ID; reserve "
                "the other two depth-third winners as completely held-out layers"
            ),
        },
        "request_manifest": file_record(args.request_manifest),
        "screening_batch": screening[0],
        "screening_capture_id": next(iter(capture_ids)),
        "profile_files": [file_record(path) for path in paths],
        "alignment_rows_per_expert": args.alignment,
        "selected_layers": selected_layers,
    }
    full_summary = {
        **common,
        "raw_r_definition": "max identity-placement rank assignments / eight-rank mean",
        "third_winners": third_winners,
        "layers": layer_summaries,
    }
    write_json_no_clobber(args.output, full_summary)
    layer_manifest = {
        **common,
        "profile_summary": file_record(args.output),
    }
    write_json_no_clobber(args.layer_manifest_output, layer_manifest)
    print(
        "selected="
        + ",".join(
            f"L{item['layer_id']}:{'cal' if item['calibration'] else 'held'}"
            for item in selected_layers
        )
        + f" layer_manifest_sha256={sha256(args.layer_manifest_output)}"
    )


if __name__ == "__main__":
    main()
