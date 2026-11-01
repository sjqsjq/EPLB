#!/usr/bin/env python3
"""SETUP-only: extract real route metadata from the 16 indexed CD-X0 captures.

Reads inputs/cdx0_v1/capture_index.json, validates and loads the eight real
forward payloads per capture unit, and derives the route metadata
(expert_counts / zero-aware 128-blocks / per-source-rank counts) exactly as
the replay harness rebuilds it collectively at execution time. One JSON per
capture is written under inputs/cdx0_v1/route_metadata/. Deterministic,
CPU-only, no timing, no GPU, no layout or performance information.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cdx0_common import (  # noqa: E402
    STUDY_ID,
    WORLD_SIZE,
    RouteMetadata,
    indexed_units,
    load_capture_index,
    load_tensor_payload,
    route_metadata_from_payloads,
    sha256_file,
    validate_forward_payload,
    verify_file_record,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    return parser.parse_args()


def metadata_record(metadata: RouteMetadata) -> dict[str, Any]:
    return {
        "hidden_rows_by_source_rank": list(metadata.hidden_rows_by_source_rank),
        "total_hidden_rows": metadata.total_hidden_rows,
        "total_assignments": metadata.total_assignments,
        "source_expert_counts": [list(row) for row in metadata.source_expert_counts],
        "expert_counts": list(metadata.expert_counts),
        "expert_blocks_128": list(metadata.expert_blocks),
        "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    index_path = root / "inputs/cdx0_v1/capture_index.json"
    out_dir = root / "inputs/cdx0_v1/route_metadata"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_capture_index(index_path)
    units = indexed_units(index)
    if len(units) != 16:
        raise ValueError(f"expected 16 capture units, found {len(units)}")

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for unit in sorted(units, key=lambda u: u.capture_id):
        payloads = []
        forward_records = []
        for rank in range(WORLD_SIZE):
            record = unit.forward_records[rank]
            path = verify_file_record(record)
            payload = load_tensor_payload(path)
            validate_forward_payload(
                payload,
                unit=unit,
                rank=rank,
                path=path,
                allowed_splits={"calibration", "evaluation"},
            )
            payloads.append(payload)
            forward_records.append({"path": str(path), "sha256": record["sha256"]})
        metadata = route_metadata_from_payloads(unit, payloads)
        if metadata.hidden_rows_by_source_rank != unit.hidden_rows_by_source_rank:
            raise ValueError(f"per-rank row mismatch in {unit.capture_id}")
        document = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "capture_id": unit.capture_id,
            "batch_id": unit.batch_id,
            "split": unit.split,
            "domain": unit.raw["domain"],
            "layer_id": unit.layer_id,
            "created_at_utc": created_at,
            "capture_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
            "forward_files": forward_records,
            "route_metadata": metadata_record(metadata),
        }
        out_path = out_dir / f"{unit.capture_id}.json"
        write_json_atomic(out_path, document)
        print(
            f"{unit.capture_id}: rows={metadata.total_hidden_rows} "
            f"assignments={metadata.total_assignments} "
            f"max_blocks={max(metadata.expert_blocks)} "
            f"zero_experts={sum(1 for c in metadata.expert_counts if c == 0)}"
        )
    print(f"wrote {len(units)} route-metadata files under {out_dir}")


if __name__ == "__main__":
    main()
