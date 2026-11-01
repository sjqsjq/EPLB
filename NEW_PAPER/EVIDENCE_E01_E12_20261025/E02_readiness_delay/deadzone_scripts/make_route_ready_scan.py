#!/usr/bin/env python3
"""Generate interleaved original/strict conditions for the real-route scan."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


DEFAULT_RATIOS = ("1", "1.02", "1.05", "1.10", "1.20", "1.40")
MODES = (("original", "original"), ("strict_barrier", "strict"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, choices=(256, 4096), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", default=",".join(DEFAULT_RATIOS))
    args = parser.parse_args()

    ratio_labels = tuple(piece.strip() for piece in args.ratios.split(",") if piece.strip())
    if not ratio_labels or Decimal(ratio_labels[0]) != Decimal("1"):
        raise ValueError("ratios must start at 1")

    conditions = []
    realized = []
    for label in ratio_labels:
        requested = Decimal(label)
        target = int(
            (requested * args.total_tokens).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        route = {
            "route_kind": "fixed7_one_double",
            "requested_r": float(requested),
            "target_hot_assignments": target,
            "hot_rank": 0,
            "expert_policy": "packed_hot_balanced_cold",
        }
        for mode, prefix in MODES:
            conditions.append(
                {
                    "name": f"route-r{label}-{prefix}",
                    "mode": mode,
                    "delay_us": 0.0,
                    "delay_ranks": [0],
                }
                | route
            )
        realized.append(
            {
                "requested_r": float(requested),
                "target_hot_assignments": target,
                "realized_raw_max_mean_r": target / args.total_tokens,
            }
        )

    payload = {
        "experiment": "route_ready_scan",
        "total_tokens": args.total_tokens,
        "definition": "r=max_d L_d / mean_d L_d; mean_d L_d equals total_tokens",
        "communication_control": (
            "every token has seven unique destination ranks and one destination "
            "receives two distinct experts; destination support is invariant across r"
        ),
        "modes": [mode for mode, _ in MODES],
        "conditions": conditions,
        "realized": realized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: N={args.total_tokens} ratios={len(ratio_labels)} "
        f"conditions={len(conditions)}"
    )


if __name__ == "__main__":
    main()
