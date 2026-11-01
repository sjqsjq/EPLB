#!/usr/bin/env python3
"""Generate the fixed Δ/C0 condition plan from a baseline calibration run."""

from __future__ import annotations

import argparse
import json
import pathlib


DEFAULT_RATIOS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0)


def parse_csv_numbers(value: str, cast):
    return tuple(cast(piece) for piece in value.split(",") if piece)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ranks", default="0,3,7")
    parser.add_argument("--ratios", default=",".join(map(str, DEFAULT_RATIOS)))
    parser.add_argument(
        "--strict-ranks",
        default="0,3,7",
        help="selected ranks repeated under the strict GPU-ready barrier",
    )
    parser.add_argument(
        "--no-all-rank-control",
        action="store_true",
        help="omit original-path all-rank-delay controls",
    )
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    summary = calibration["summary"]
    c0_original_ms = summary["original-zero"]["cross_rank_max"][
        "combine_after_gate_ms"
    ]["median"]
    c0_barrier_ms = summary["barrier-zero"]["cross_rank_max"][
        "combine_after_gate_ms"
    ]["median"]
    c0_us = c0_original_ms * 1000.0

    ratios = parse_csv_numbers(args.ratios, float)
    if not ratios or ratios[0] != 0.0 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must start with 0 and be nonnegative")
    ranks = parse_csv_numbers(args.ranks, int)
    strict_ranks = parse_csv_numbers(args.strict_ranks, int)

    conditions = [
        {
            "name": "original-zero",
            "mode": "original",
            "delay_us": 0.0,
            "delay_ranks": [ranks[0]],
            "delay_ratio": 0.0,
        },
        {
            "name": "barrier-zero",
            "mode": "strict_barrier",
            "delay_us": 0.0,
            "delay_ranks": [strict_ranks[0]],
            "delay_ratio": 0.0,
        },
    ]

    for ratio in ratios[1:]:
        delay_us = ratio * c0_us
        ratio_label = f"{ratio:g}"
        for rank in ranks:
            conditions.append(
                {
                    "name": f"original-rank{rank}-x{ratio_label}",
                    "mode": "original",
                    "delay_us": delay_us,
                    "delay_ranks": [rank],
                    "delay_ratio": ratio,
                }
            )
        for rank in strict_ranks:
            conditions.append(
                {
                    "name": f"barrier-rank{rank}-x{ratio_label}",
                    "mode": "strict_barrier",
                    "delay_us": delay_us,
                    "delay_ranks": [rank],
                    "delay_ratio": ratio,
                }
            )
        if not args.no_all_rank_control:
            conditions.append(
                {
                    "name": f"original-all-ranks-x{ratio_label}",
                    "mode": "original",
                    "delay_us": delay_us,
                    "delay_ranks": "all",
                    "delay_ratio": ratio,
                }
            )

    payload = {
        "source_calibration": str(args.calibration),
        "total_tokens": calibration["total_tokens"],
        "normalization": {
            "definition": (
                "C0 is the median across repetitions of the cross-rank maximum "
                "gate-done-to-combine-done CUDA-event interval on original-zero"
            ),
            "c0_original_ms": c0_original_ms,
            "c0_barrier_ms": c0_barrier_ms,
            "delay_ratios_use": "c0_original_ms for both paths",
        },
        "ratios": list(ratios),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: C0={c0_us:.3f} us, "
        f"barrier baseline={c0_barrier_ms * 1000.0:.3f} us, "
        f"conditions={len(conditions)}"
    )


if __name__ == "__main__":
    main()
