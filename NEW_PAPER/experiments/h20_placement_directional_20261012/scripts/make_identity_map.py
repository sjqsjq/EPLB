#!/usr/bin/env python3
"""Write the explicit one-copy identity expert map used for capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--layers", type=int, default=94)
    parser.add_argument("--experts", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layers <= 0 or args.experts <= 0:
        raise ValueError("layers and experts must be positive")
    row = list(range(args.experts))
    payload = {"physical_to_logical_map": [row.copy() for _ in range(args.layers)]}
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(f"sha256={hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
