#!/usr/bin/env python3
"""Write one-condition plans for representative Nsight Systems timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    total_tokens = int(calibration["total_tokens"])
    c0_ms = calibration["summary"]["original-zero"]["cross_rank_max"][
        "combine_after_gate_ms"
    ]["median"]
    specs = (
        ("original", "zero", 0.0),
        ("original", "inside", 0.05),
        ("original", "outside", 0.40),
        ("strict_barrier", "zero", 0.0),
        ("strict_barrier", "inside", 0.05),
        ("strict_barrier", "outside", 0.40),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mode, region, ratio in specs:
        mode_label = "original" if mode == "original" else "barrier"
        name = f"timeline-{mode_label}-{region}"
        condition = {
            "name": name,
            "mode": mode,
            "delay_us": ratio * c0_ms * 1000.0,
            "delay_ratio": ratio,
            "delay_ranks": [3],
        }
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps({"conditions": [condition]}, indent=2) + "\n")
        manifest.append({"name": name, "path": str(path), "condition": condition})
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_calibration": str(args.calibration),
                "total_tokens": total_tokens,
                "c0_us": c0_ms * 1000.0,
                "inside_definition": "0.05*C0, below both coarse and merged rank-3 hinges",
                "outside_definition": "0.40*C0, above both coarse and merged rank-3 hinges",
                "conditions": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {manifest_path}: C0={c0_ms * 1000.0:.3f} us")


if __name__ == "__main__":
    main()
