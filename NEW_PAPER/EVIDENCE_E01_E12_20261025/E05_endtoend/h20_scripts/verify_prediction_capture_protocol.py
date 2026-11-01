#!/usr/bin/env python3
"""Verify every immutable input and implementation hash before live capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


STUDY_ID = "h20-placement-prediction-v1"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("study_id") != STUDY_ID:
        raise ValueError("capture protocol has the wrong study ID")
    if protocol.get("status") != "frozen_before_screening_route_profile":
        raise ValueError("capture protocol is not in its frozen pre-profile state")

    records = [
        protocol["followup_protocol"],
        protocol["request_manifest"],
        protocol["identity_map"],
        protocol["runtime_environment"],
        *protocol["request_files"],
        *protocol["model_files"],
        *protocol["implementation"].values(),
    ]
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("capture protocol contains duplicate immutable file records")
    for expected in records:
        path = pathlib.Path(expected["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_record(path)
        if actual != expected:
            raise ValueError(
                f"immutable capture input changed: {path}; "
                f"expected={expected}, actual={actual}"
            )

    request_manifest = json.loads(
        pathlib.Path(protocol["request_manifest"]["path"]).read_text()
    )
    if request_manifest.get("study_id") != STUDY_ID:
        raise ValueError("frozen request manifest has the wrong study ID")
    if request_manifest.get("status") != "frozen_before_screening_route_profile":
        raise ValueError("request manifest is not frozen before screening")
    if len(request_manifest.get("batches", [])) != 7:
        raise ValueError("prediction v1 requires exactly seven request batches")
    splits = [batch["split"] for batch in request_manifest["batches"]]
    if splits.count("screening") != 1:
        raise ValueError("prediction v1 requires one screening batch")
    if splits.count("calibration") != 3 or splits.count("evaluation") != 3:
        raise ValueError("prediction v1 requires three calibration/evaluation batches")
    if any(batch["request_count"] != 8 for batch in request_manifest["batches"]):
        raise ValueError("every prediction batch must contain eight requests")

    print(
        f"verified {len(records)} immutable records; "
        f"protocol_sha256={sha256(args.protocol)}"
    )


if __name__ == "__main__":
    main()
