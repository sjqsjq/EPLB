#!/usr/bin/env python3
"""Generate exact-route, one-rank-at-a-time readiness-delay conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DELAYS_US = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
MODES = ("original", "strict_barrier")


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())
    if not values:
        raise ValueError("delay list is empty")
    if values[0] != 0.0 or any(item < 0.0 for item in values):
        raise ValueError("delays must be nonnegative and start at zero")
    if tuple(sorted(set(values))) != values:
        raise ValueError("delays must be unique and strictly increasing")
    return values


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not values:
        raise ValueError("rank list is empty")
    if tuple(sorted(set(values))) != values or any(rank < 0 or rank >= 8 for rank in values):
        raise ValueError("ranks must be unique, increasing, and in [0, 7]")
    return values


def delay_label(delay_us: float) -> str:
    return f"{delay_us:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, choices=(256, 4096), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--delays-us",
        default=",".join(f"{value:g}" for value in DEFAULT_DELAYS_US),
    )
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(8)))
    args = parser.parse_args()

    delays_us = parse_csv_floats(args.delays_us)
    ranks = parse_csv_ints(args.ranks)
    prefix = "original" if args.mode == "original" else "strict"
    route = {
        "route_kind": "fixed7_one_double",
        "requested_r": 1.0,
        "target_hot_assignments": args.total_tokens,
        "hot_rank": 0,
        "expert_policy": "packed_hot_balanced_cold",
    }

    conditions = [
        {
            "name": f"{prefix}-zero",
            "mode": args.mode,
            "delay_us": 0.0,
            "delay_ranks": [0],
        }
        | route
    ]
    for rank in ranks:
        for delay_us in delays_us[1:]:
            conditions.append(
                {
                    "name": f"{prefix}-rank{rank}-d{delay_label(delay_us)}us",
                    "mode": args.mode,
                    "delay_us": delay_us,
                    "delay_ranks": [rank],
                }
                | route
            )

    payload = {
        "experiment": "rank_readiness_scan",
        "purpose": (
            "strict mode independently estimates per-rank readiness slack; "
            "original mode measures the response with those estimates frozen"
        ),
        "total_tokens": args.total_tokens,
        "mode": args.mode,
        "ranks": list(ranks),
        "delays_us": list(delays_us),
        "route": route,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: N={args.total_tokens} mode={args.mode} "
        f"ranks={len(ranks)} delays={len(delays_us)} conditions={len(conditions)}"
    )


if __name__ == "__main__":
    main()
