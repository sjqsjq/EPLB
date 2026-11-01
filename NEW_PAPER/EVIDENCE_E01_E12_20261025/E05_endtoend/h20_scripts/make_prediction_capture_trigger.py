#!/usr/bin/env python3
"""Atomically arm one hash-bound prediction-study profile or snapshot capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("profile", "snapshot_split"), required=True
    )
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--request-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--trigger", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--weight-output-dir", type=pathlib.Path)
    parser.add_argument("--identity-map", type=pathlib.Path, required=True)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument(
        "--layer-role", choices=("bridge", "calibration", "held_out")
    )
    parser.add_argument("--layer-manifest", type=pathlib.Path)
    parser.add_argument("--min-hidden-rows", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_hidden_rows <= 0:
        raise ValueError("minimum hidden rows must be positive")
    if args.mode == "profile":
        if args.target_layer is not None or args.layer_role is not None:
            raise ValueError("profile trigger cannot specify a target layer or role")
        if args.weight_output_dir is not None or args.layer_manifest is not None:
            raise ValueError("profile trigger cannot specify snapshot artifacts")
    else:
        if args.target_layer is None or args.layer_role is None:
            raise ValueError("split snapshot requires target layer and layer role")
        if args.weight_output_dir is None or args.layer_manifest is None:
            raise ValueError("split snapshot requires weight and layer manifests")

    request_manifest = json.loads(args.request_manifest.read_text())
    if request_manifest.get("study_id") != "h20-placement-prediction-v1":
        raise ValueError("request manifest has the wrong study ID")
    if request_manifest.get("status") != "frozen_before_screening_route_profile":
        raise ValueError("request manifest is not in the frozen pre-profile state")
    matches = [
        batch for batch in request_manifest["batches"]
        if batch["batch_id"] == args.batch_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate batch ID {args.batch_id!r}")
    batch = matches[0]
    request_file = pathlib.Path(batch["request_file"]["path"])
    if file_record(request_file) != batch["request_file"]:
        raise ValueError("request file metadata differs from frozen manifest")
    if args.mode == "profile" and batch["split"] != "screening":
        raise ValueError("route profile must use the frozen screening split")
    if args.mode == "snapshot_split" and batch["split"] == "screening":
        raise ValueError("screening batch cannot become a model-study snapshot")

    layer_record = None
    if args.mode == "snapshot_split":
        layer_manifest = json.loads(args.layer_manifest.read_text())
        if layer_manifest.get("study_id") != "h20-placement-prediction-v1":
            raise ValueError("layer manifest has the wrong study ID")
        selected = {
            int(item["layer_id"]): item for item in layer_manifest["selected_layers"]
        }
        if args.target_layer not in selected:
            raise ValueError("target layer is absent from frozen selected layers")
        layer_record = selected[args.target_layer]
        expected_role = "held_out" if layer_record["held_out"] else "calibration"
        if args.target_layer == 24:
            allowed_roles = {"bridge", "calibration"}
        else:
            allowed_roles = {expected_role}
        if args.layer_role not in allowed_roles:
            raise ValueError(
                f"layer role {args.layer_role!r} conflicts with frozen role {expected_role!r}"
            )
        if batch["split"] == "calibration" and not layer_record["calibration"]:
            raise ValueError("calibration split can capture only calibration layers")

    capture_context = {
        "prediction_request_manifest": file_record(args.request_manifest),
        "batch_id": batch["batch_id"],
        "split": batch["split"],
        "request_file": batch["request_file"],
        "request_source_lines_zero_based": [
            item["source_line_index_zero_based"] for item in batch["requests"]
        ],
        "dataset": request_manifest["dataset"],
        "model": request_manifest["model"],
        "layer_role": args.layer_role,
        "layer_record": layer_record,
        "layer_manifest": (
            file_record(args.layer_manifest)
            if args.layer_manifest is not None else None
        ),
    }
    payload = {
        "schema_version": 2,
        "study_id": "h20-placement-prediction-v1",
        "mode": args.mode,
        "capture_id": args.capture_id,
        "output_dir": str(args.output_dir.resolve()),
        "weight_output_dir": (
            str(args.weight_output_dir.resolve())
            if args.weight_output_dir is not None else None
        ),
        "identity_map": file_record(args.identity_map),
        "identity_map_sha256": sha256(args.identity_map),
        "identity_map_required": True,
        "expected_layers": 94,
        "num_experts": 128,
        "min_hidden_rows": args.min_hidden_rows,
        "target_layer": args.target_layer,
        "capture_context": capture_context,
    }
    expected_files = (
        [args.output_dir / f"profile_rank{rank}.pt" for rank in range(8)]
        if args.mode == "profile"
        else [args.output_dir / f"forward_rank{rank}.pt" for rank in range(8)]
    )
    existing = [str(path) for path in expected_files if path.exists()]
    if existing:
        raise FileExistsError(f"capture outputs already exist: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.weight_output_dir is not None:
        args.weight_output_dir.mkdir(parents=True, exist_ok=True)
    args.trigger.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.trigger.with_suffix(args.trigger.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale trigger temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.trigger)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
