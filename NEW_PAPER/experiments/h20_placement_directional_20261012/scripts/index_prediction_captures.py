#!/usr/bin/env python3
"""Validate and hash-index every frozen prediction-study capture artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import torch


STUDY_ID = "h20-placement-prediction-v1"
WORLD_SIZE = 8
TOP_K = 8
HIDDEN_SIZE = 4096
NUM_EXPERTS = 128
LOCAL_EXPERTS = 16
WEIGHT_BLOCK_SIZE = [128, 128]
EXPERT_SHAPES = {
    "w13_weight": (LOCAL_EXPERTS, 3072, HIDDEN_SIZE),
    "w2_weight": (LOCAL_EXPERTS, HIDDEN_SIZE, 1536),
    "w13_weight_scale_inv": (LOCAL_EXPERTS, 24, 32),
    "w2_weight_scale_inv": (LOCAL_EXPERTS, 32, 12),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_tensor(
    payload: dict[str, Any], name: str, *, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    return value


def all_finite(tensor: torch.Tensor) -> bool:
    if tensor.dtype == torch.float8_e4m3fn:
        raw = tensor.view(torch.uint8).flatten()
        chunk_size = 8 * 1024 * 1024
        return all(
            not bool(((raw[begin : begin + chunk_size] & 0x7F) == 0x7F).any())
            for begin in range(0, raw.numel(), chunk_size)
        )
    return bool(torch.isfinite(tensor).all())


def write_json_no_clobber(path: pathlib.Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite capture index: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale capture-index temporary exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=pathlib.Path, required=True)
    parser.add_argument("--weight-root", type=pathlib.Path, required=True)
    parser.add_argument("--response-root", type=pathlib.Path, required=True)
    parser.add_argument("--trigger-root", type=pathlib.Path, required=True)
    parser.add_argument("--request-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--layer-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--profile-summary", type=pathlib.Path, required=True)
    parser.add_argument("--capture-protocol", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_manifest = load_json(args.request_manifest)
    layer_manifest = load_json(args.layer_manifest)
    capture_protocol = load_json(args.capture_protocol)
    if request_manifest.get("study_id") != STUDY_ID:
        raise ValueError("request manifest has the wrong study ID")
    if layer_manifest.get("study_id") != STUDY_ID:
        raise ValueError("layer manifest has the wrong study ID")
    if capture_protocol.get("study_id") != STUDY_ID:
        raise ValueError("capture protocol has the wrong study ID")
    if layer_manifest["profile_summary"] != file_record(args.profile_summary):
        raise ValueError("profile summary differs from the layer manifest")

    batches = {item["batch_id"]: item for item in request_manifest["batches"]}
    layers = {
        int(item["layer_id"]): item for item in layer_manifest["selected_layers"]
    }
    calibration_layers = sorted(
        layer_id for layer_id, item in layers.items() if item["calibration"]
    )
    all_layers = sorted(layers)
    if len(calibration_layers) != 2 or len(all_layers) != 4:
        raise ValueError("expected two calibration and four total selected layers")

    profile_records = []
    profile_dir = args.capture_root / "screening-01" / "profile"
    for rank in range(WORLD_SIZE):
        path = profile_dir / f"profile_rank{rank}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("capture_id") != "pred-v1-screening-01-profile":
            raise ValueError(f"unexpected profile capture ID at rank {rank}")
        if payload.get("rank") != rank or len(payload.get("layers", [])) != 94:
            raise ValueError(f"invalid profile rank/layer metadata at rank {rank}")
        profile_records.append(file_record(path))

    unit_records = []
    used_expert_paths: set[pathlib.Path] = set()
    for batch_id, batch in batches.items():
        if batch["split"] == "screening":
            continue
        expected_layers = (
            calibration_layers if batch["split"] == "calibration" else all_layers
        )
        expected_row_multiset = sorted(
            int(item["token_count"]) for item in batch["requests"]
        )
        for layer_id in expected_layers:
            capture_id = f"pred-v1-{batch_id}-layer-{layer_id:03d}"
            unit_dir = args.capture_root / batch_id / f"layer-{layer_id:03d}"
            trigger_path = args.trigger_root / f"{capture_id}.json"
            response_path = args.response_root / f"{capture_id}.json"
            trigger = load_json(trigger_path)
            response = load_json(response_path)
            if trigger.get("capture_id") != capture_id:
                raise ValueError(f"trigger capture ID mismatch for {capture_id}")
            if response.get("capture_id") != capture_id:
                raise ValueError(f"response capture ID mismatch for {capture_id}")
            if response.get("batch_id") != batch_id:
                raise ValueError(f"response batch mismatch for {capture_id}")
            if response["capture_trigger"]["sha256"] != sha256(trigger_path):
                raise ValueError(f"response trigger hash mismatch for {capture_id}")
            responses = response.get("responses", [])
            if len(responses) != batch["request_count"] or any(
                item.get("status") != 200 or "error" in item for item in responses
            ):
                raise ValueError(f"incomplete server response set for {capture_id}")

            forward_records = []
            hidden_rows = []
            for rank in range(WORLD_SIZE):
                forward_path = unit_dir / f"forward_rank{rank}.pt"
                payload = torch.load(
                    forward_path, map_location="cpu", weights_only=False
                )
                expected_header = {
                    "schema_version": 2,
                    "study_id": STUDY_ID,
                    "capture_id": capture_id,
                    "mode": "snapshot_split",
                    "rank": rank,
                    "layer_id": layer_id,
                    "identity_map_required": True,
                    "identity_map_sha256": capture_protocol["identity_map"][
                        "sha256"
                    ],
                    "weight_block_size": WEIGHT_BLOCK_SIZE,
                }
                mismatches = {
                    name: (payload.get(name), value)
                    for name, value in expected_header.items()
                    if payload.get(name) != value
                }
                if mismatches:
                    raise ValueError(
                        f"forward metadata mismatch at {forward_path}: {mismatches}"
                    )
                context = payload["capture_context"]
                if context["batch_id"] != batch_id:
                    raise ValueError(f"forward batch mismatch at {forward_path}")
                if context["layer_record"] != layers[layer_id]:
                    raise ValueError(f"forward layer record mismatch at {forward_path}")
                if context["prediction_request_manifest"]["sha256"] != sha256(
                    args.request_manifest
                ):
                    raise ValueError(
                        f"forward request-manifest mismatch at {forward_path}"
                    )
                if context["layer_manifest"]["sha256"] != sha256(
                    args.layer_manifest
                ):
                    raise ValueError(f"forward layer-manifest mismatch at {forward_path}")

                rows = int(payload["hidden_rows"])
                hidden = require_tensor(
                    payload, "hidden_states", shape=(rows, HIDDEN_SIZE)
                )
                weights = require_tensor(
                    payload, "logical_topk_weights", shape=(rows, TOP_K)
                )
                ids = require_tensor(
                    payload, "logical_topk_ids", shape=(rows, TOP_K)
                )
                output = require_tensor(
                    payload, "routed_output", shape=(rows, HIDDEN_SIZE)
                )
                for name, tensor in (
                    ("hidden_states", hidden),
                    ("logical_topk_weights", weights),
                    ("routed_output", output),
                ):
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"nonfinite {name} at {forward_path}")
                if int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS:
                    raise ValueError(f"out-of-range logical route ID at {forward_path}")
                if payload["num_token_non_padded"] != rows:
                    raise ValueError(f"hidden/non-padded row mismatch at {forward_path}")
                expected_expert_path = (
                    args.weight_root
                    / f"layer-{layer_id:03d}"
                    / f"expert_rank{rank}.pt"
                ).resolve()
                if pathlib.Path(payload["expert_artifact_path"]) != expected_expert_path:
                    raise ValueError(f"expert path mismatch at {forward_path}")
                used_expert_paths.add(expected_expert_path)
                hidden_rows.append(rows)
                forward_records.append(file_record(forward_path))

            if sorted(hidden_rows) != expected_row_multiset:
                raise ValueError(
                    f"source-rank row multiset differs from frozen requests for {capture_id}: "
                    f"{sorted(hidden_rows)} != {expected_row_multiset}"
                )
            if sum(hidden_rows) != batch["total_input_tokens"]:
                raise ValueError(f"total hidden rows differ for {capture_id}")
            unit_records.append(
                {
                    "capture_id": capture_id,
                    "batch_id": batch_id,
                    "split": batch["split"],
                    "layer_id": layer_id,
                    "hidden_rows_by_source_rank": hidden_rows,
                    "total_hidden_rows": sum(hidden_rows),
                    "total_assignments": sum(hidden_rows) * TOP_K,
                    "trigger": file_record(trigger_path),
                    "response": file_record(response_path),
                    "forward_files": forward_records,
                }
            )

    expert_records = []
    expected_expert_count = len(all_layers) * WORLD_SIZE
    if len(used_expert_paths) != expected_expert_count:
        raise ValueError(
            f"used {len(used_expert_paths)} expert artifacts, expected {expected_expert_count}"
        )
    for path in sorted(used_expert_paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        layer_id = int(payload.get("layer_id", -1))
        rank = int(payload.get("rank", -1))
        expected_header = {
            "schema_version": 2,
            "study_id": STUDY_ID,
            "artifact_type": "immutable_layer_expert_weights",
            "rank": rank,
            "layer_id": layer_id,
            "identity_map_required": True,
            "identity_map_sha256": capture_protocol["identity_map"]["sha256"],
            "logical_expert_ids": list(
                range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
            ),
            "weight_block_size": WEIGHT_BLOCK_SIZE,
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected_header.items()
            if payload.get(name) != value
        }
        if mismatches or layer_id not in layers or not 0 <= rank < WORLD_SIZE:
            raise ValueError(f"expert metadata mismatch at {path}: {mismatches}")
        for name, expected_shape in EXPERT_SHAPES.items():
            tensor = require_tensor(payload, name, shape=expected_shape)
            expected_dtype = (
                torch.float8_e4m3fn if name.endswith("weight") else torch.float32
            )
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"invalid dtype for {name} at {path}: {tensor.dtype}"
                )
            if not all_finite(tensor):
                raise ValueError(f"nonfinite expert tensor {name} at {path}")
        expert_records.append(file_record(path))

    expected_units = 3 * 2 + 3 * 4
    if len(unit_records) != expected_units:
        raise ValueError(f"validated {len(unit_records)} units, expected {expected_units}")
    output = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "all_prediction_capture_inputs_validated_and_hash_indexed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capture_protocol": file_record(args.capture_protocol),
        "request_manifest": file_record(args.request_manifest),
        "layer_manifest": file_record(args.layer_manifest),
        "profile_summary": file_record(args.profile_summary),
        "counts": {
            "profile_rank_files": len(profile_records),
            "capture_units": len(unit_records),
            "forward_rank_files": sum(
                len(item["forward_files"]) for item in unit_records
            ),
            "expert_rank_files": len(expert_records),
        },
        "profile_files": profile_records,
        "capture_units": unit_records,
        "expert_files": expert_records,
        "validation": {
            "all_server_responses_http_200": True,
            "all_real_hidden_and_output_values_finite": True,
            "all_route_ids_in_range": True,
            "all_request_token_count_multisets_exact": True,
            "all_expert_tensors_present_and_finite": True,
            "calibration_and_evaluation_request_groups_disjoint": True,
            "evaluation_capture_not_opened_for_model_fitting": True,
        },
    }
    write_json_no_clobber(args.output, output)
    print(
        f"validated {len(unit_records)} units, "
        f"{len(expert_records)} expert files; index_sha256={sha256(args.output)}"
    )


if __name__ == "__main__":
    main()
