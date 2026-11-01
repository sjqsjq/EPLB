#!/usr/bin/env python3
"""Generate rank-0 delay conditions on the exact r=1 fixed7 load route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RATIOS = {
    256: (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.20, 2.00),
    4096: (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.80, 1.20, 2.00),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    total_tokens = int(calibration["total_tokens"])
    c0_ms = calibration["summary"]["original-zero"]["cross_rank_max"][
        "combine_after_gate_ms"
    ]["median"]
    route = {
        "route_kind": "fixed7_one_double",
        "requested_r": 1.0,
        "target_hot_assignments": total_tokens,
        "hot_rank": 0,
        "expert_policy": "packed_hot_balanced_cold",
    }
    conditions = [
        {
            "name": "original-zero",
            "mode": "original",
            "delay_us": 0.0,
            "delay_ratio": 0.0,
            "delay_ranks": [0],
        }
        | route,
        {
            "name": "barrier-zero",
            "mode": "strict_barrier",
            "delay_us": 0.0,
            "delay_ratio": 0.0,
            "delay_ranks": [0],
        }
        | route,
    ]
    for ratio in RATIOS[total_tokens][1:]:
        label = f"{ratio:g}"
        delay_us = ratio * c0_ms * 1000.0
        for mode, prefix in (("original", "original"), ("strict_barrier", "barrier")):
            conditions.append(
                {
                    "name": f"{prefix}-rank0-x{label}",
                    "mode": mode,
                    "delay_us": delay_us,
                    "delay_ratio": ratio,
                    "delay_ranks": [0],
                }
                | route
            )
    payload = {
        "source_calibration": str(args.calibration),
        "total_tokens": total_tokens,
        "normalization": {"c0_original_ms": c0_ms},
        "route": route,
        "ratios": list(RATIOS[total_tokens]),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: N={total_tokens} C0={c0_ms * 1000.0:.3f} us "
        f"conditions={len(conditions)}"
    )


if __name__ == "__main__":
    main()
