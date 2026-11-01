#!/usr/bin/env python3
"""Generate fixed-communication real-routing imbalance conditions."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


DEFAULT_RATIOS = ("1", "1.02", "1.05", "1.10", "1.20", "1.40")


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
        actual = target / args.total_tokens
        conditions.append(
            {
                "name": f"route-r{label}",
                "mode": "original",
                "delay_us": 0.0,
                "delay_ranks": [0],
                "route_kind": "fixed7_one_double",
                "requested_r": float(requested),
                "target_hot_assignments": target,
                "hot_rank": 0,
                "expert_policy": "packed_hot_balanced_cold",
            }
        )
        realized.append(
            {
                "requested_r": float(requested),
                "target_hot_assignments": target,
                "realized_raw_max_mean_r": actual,
            }
        )

    payload = {
        "total_tokens": args.total_tokens,
        "definition": "r=max_d L_d / mean_d L_d; mean_d L_d equals total_tokens",
        "communication_control": (
            "every token has seven unique destination ranks and one destination "
            "receives two distinct experts; destination support is invariant across r"
        ),
        "conditions": conditions,
        "realized": realized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}: conditions={len(conditions)}")
    for item in realized:
        print(
            f"requested={item['requested_r']:.2f} "
            f"target={item['target_hot_assignments']} "
            f"realized={item['realized_raw_max_mean_r']:.9f}"
        )


if __name__ == "__main__":
    main()
