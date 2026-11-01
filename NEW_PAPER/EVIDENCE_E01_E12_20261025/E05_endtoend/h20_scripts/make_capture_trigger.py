#!/usr/bin/env python3
"""Atomically arm the live SGLang route or replay-snapshot capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("profile", "snapshot"), required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--trigger", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--identity-map", type=pathlib.Path, required=True)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--min-hidden-rows", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "snapshot" and args.target_layer is None:
        raise ValueError("snapshot mode requires --target-layer")
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "capture_id": args.capture_id,
        "output_dir": str(args.output_dir.resolve()),
        "identity_map_path": str(args.identity_map.resolve()),
        "identity_map_sha256": sha256(args.identity_map),
        "identity_map_required": True,
        "expected_layers": 94,
        "num_experts": 128,
        "min_hidden_rows": args.min_hidden_rows,
        "target_layer": args.target_layer,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trigger.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.trigger.with_suffix(args.trigger.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.trigger)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
