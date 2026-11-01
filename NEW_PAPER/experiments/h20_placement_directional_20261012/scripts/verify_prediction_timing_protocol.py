#!/usr/bin/env python3
"""Verify every immutable record in a frozen prediction timing protocol."""

from __future__ import annotations

import argparse
import json
import pathlib

from prediction_common import STUDY_ID, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"))
    return parser.parse_args()


def verify_record(record: dict, label: str) -> None:
    path = pathlib.Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"size mismatch for {label}: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {label}: {path}: {actual}")


def main() -> None:
    args = parse_args()
    path = args.protocol.resolve()
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 1 or protocol.get("study_id") != STUDY_ID:
        raise ValueError("unsupported prediction timing protocol")
    split = protocol.get("split")
    if args.split is not None and split != args.split:
        raise ValueError(f"protocol split is {split!r}, expected {args.split!r}")
    if protocol.get("status") != f"frozen_before_{split}_timing":
        raise ValueError("prediction timing protocol is not frozen")

    records: list[tuple[dict, str]] = []
    for name in (
        "followup_protocol",
        "capture_index",
        "layout_index",
        "tolerance_file",
    ):
        records.append((protocol[name], name))
    for name, record in protocol["implementation"].items():
        if name.endswith("_record"):
            records.append((record, name))
    seen_layouts = set()
    for start in protocol["starts"]:
        record = start["layout_manifest"]
        key = (record["path"], record["sha256"])
        if key not in seen_layouts:
            records.append((record, f"layout_manifest:{start['capture_id']}"))
            seen_layouts.add(key)
    for record, label in records:
        verify_record(record, label)

    expected_ids = protocol["design"]["layout_ids"]
    blocks = int(protocol["timing"]["blocks_per_start"])
    start_ids = set()
    output_paths = set()
    starts_by_capture: dict[str, int] = {}
    for start in protocol["starts"]:
        if start["start_id"] in start_ids:
            raise ValueError(f"duplicate start ID: {start['start_id']}")
        start_ids.add(start["start_id"])
        if start["output_relative_path"] in output_paths:
            raise ValueError(f"duplicate output path: {start['output_relative_path']}")
        output_paths.add(start["output_relative_path"])
        starts_by_capture[start["capture_id"]] = starts_by_capture.get(start["capture_id"], 0) + 1
        for pass_name in ("primary", "diagnostic"):
            schedules = start[f"{pass_name}_schedules"]
            if len(schedules) != blocks:
                raise ValueError(f"wrong {pass_name} block count in {start['start_id']}")
            for schedule in schedules:
                if len(schedule) != len(expected_ids) or set(schedule) != set(expected_ids):
                    raise ValueError(f"invalid {pass_name} schedule in {start['start_id']}")
    if len(starts_by_capture) != int(protocol["design"]["capture_units"]):
        raise ValueError("capture-unit cardinality differs")
    if any(
        count != int(protocol["design"]["starts_per_capture"])
        for count in starts_by_capture.values()
    ):
        raise ValueError("fresh-start cardinality differs")
    print(
        f"verified {len(records)} immutable records and {len(start_ids)} starts; "
        f"protocol_sha256={sha256_file(path)}"
    )


if __name__ == "__main__":
    main()
