#!/usr/bin/env python3
"""Validate the final deadzone scan artifacts without rewriting measurements."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


SCAN_SPECS = (
    ("calibration", 2, 5, 100),
    ("delay_scan", 51, 5, 100),
    ("delay_dense", 56, 5, 100),
    ("matching_route_delay", 24, 5, 100),
    ("load_scan", 6, 5, 100),
)
READINESS_SCAN_SPECS = (
    ("readiness_strict", 73, 5, 100),
    ("readiness_original", 73, 5, 100),
    ("route_ready", 12, 5, 100),
)
TIMELINE_NAMES = (
    "timeline-original-zero",
    "timeline-original-inside",
    "timeline-original-outside",
    "timeline-barrier-zero",
    "timeline-barrier-inside",
    "timeline-barrier-outside",
)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_raw(
    path: Path,
    *,
    total_tokens: int,
    condition_count: int,
    blocks: int,
    iterations: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    prefix = str(path)
    conditions = payload["conditions"]
    records = payload["records"]
    _require(payload["total_tokens"] == total_tokens, f"{prefix}: total_tokens")
    _require(payload["world_size"] == 8, f"{prefix}: world_size")
    _require(len(conditions) == condition_count, f"{prefix}: condition count")
    _require(len(records) == condition_count * blocks * iterations, f"{prefix}: record count")
    _require(payload["measurement"]["blocks"] == blocks, f"{prefix}: blocks")
    _require(
        payload["measurement"]["iterations_per_block"] == iterations,
        f"{prefix}: iterations_per_block",
    )

    names = [condition["name"] for condition in conditions]
    _require(len(names) == len(set(names)), f"{prefix}: duplicate condition names")
    counts: collections.Counter[tuple[str, int]] = collections.Counter()
    repetitions: dict[tuple[str, int], set[int]] = collections.defaultdict(set)
    max_abs = 0.0
    condition_routes = payload.get("condition_routes", {})
    routes = {
        route["route_sha256"]: route for route in payload.get("routes", [])
    }

    for record in records:
        condition = record["condition"]
        block = int(record["block"])
        key = (condition, block)
        counts[key] += 1
        repetitions[key].add(int(record["repetition"]))
        _require(condition in names, f"{prefix}: unknown record condition {condition}")
        rank_records = record["ranks"]
        _require(len(rank_records) == 8, f"{prefix}: rank record count")
        _require(
            sorted(int(rank_record["rank"]) for rank_record in rank_records)
            == list(range(8)),
            f"{prefix}: rank IDs",
        )
        max_abs = max(
            max_abs,
            *(float(rank_record["correctness_max_abs"]) for rank_record in rank_records),
        )
        if condition_routes:
            expected_hash = condition_routes[condition]
            _require(record["route_sha256"] == expected_hash, f"{prefix}: record route hash")
            _require(expected_hash in routes, f"{prefix}: missing route manifest")

    for name in names:
        for block in range(blocks):
            key = (name, block)
            _require(counts[key] == iterations, f"{prefix}: {key} count")
            _require(
                repetitions[key] == set(range(iterations)),
                f"{prefix}: {key} repetition IDs",
            )
    _require(max_abs == 0.0, f"{prefix}: correctness max {max_abs}")

    for route_hash, route in routes.items():
        _require(
            route["deepep_expert_aligned_counts"] == route["expert_aligned_counts"],
            f"{prefix}: DeepEP aligned counts disagree for {route_hash}",
        )

    return {
        "path": prefix,
        "conditions": condition_count,
        "records": len(records),
        "rank_observations": len(records) * 8,
        "correctness_max_abs": max_abs,
        "route_hashes": sorted(routes),
        "routes": list(routes.values()),
        "condition_routes": condition_routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-timelines", action="store_true")
    parser.add_argument(
        "--include-readiness",
        action="store_true",
        help="also require the N=4096 all-rank readiness and global-ready scans",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    output = args.output or results / "validation.json"

    validations = []
    route_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for stem, condition_count, blocks, iterations in SCAN_SPECS:
        for total_tokens in (256, 4096):
            path = results / f"{stem}_N{total_tokens}.json"
            validation = validate_raw(
                path,
                total_tokens=total_tokens,
                condition_count=condition_count,
                blocks=blocks,
                iterations=iterations,
            )
            validations.append(validation)
            if stem in {"matching_route_delay", "load_scan"}:
                route_payloads[(stem, total_tokens)] = validation

    route_checks = []
    for total_tokens in (256, 4096):
        matching = route_payloads[("matching_route_delay", total_tokens)]
        load = route_payloads[("load_scan", total_tokens)]
        matching_hashes = set(matching["route_hashes"])
        baseline_hash = load["condition_routes"]["route-r1"]
        _require(
            matching_hashes == {baseline_hash},
            f"N={total_tokens}: matching scan and load baseline route mismatch",
        )
        matrices = {
            json.dumps(route["source_destination_unique_rows"], separators=(",", ":"))
            for route in load["routes"]
        }
        _require(len(matrices) == 1, f"N={total_tokens}: communication matrix changed")
        expected_rows = 28 if total_tokens == 256 else 448
        matrix = load["routes"][0]["source_destination_unique_rows"]
        _require(
            matrix == [[expected_rows] * 8 for _ in range(8)],
            f"N={total_tokens}: unexpected source-destination rows",
        )
        route_checks.append(
            {
                "total_tokens": total_tokens,
                "matching_baseline_route_sha256": baseline_hash,
                "source_destination_unique_rows": matrix,
            }
        )

    if args.include_readiness:
        readiness_payloads: dict[str, dict[str, Any]] = {}
        for stem, condition_count, blocks, iterations in READINESS_SCAN_SPECS:
            path = results / f"{stem}_N4096.json"
            validation = validate_raw(
                path,
                total_tokens=4096,
                condition_count=condition_count,
                blocks=blocks,
                iterations=iterations,
            )
            validations.append(validation)
            readiness_payloads[stem] = validation

        load_baseline_hash = route_payloads[("load_scan", 4096)]["condition_routes"][
            "route-r1"
        ]
        strict_hashes = set(readiness_payloads["readiness_strict"]["route_hashes"])
        original_hashes = set(readiness_payloads["readiness_original"]["route_hashes"])
        _require(
            strict_hashes == original_hashes == {load_baseline_hash},
            "N=4096: readiness scans do not match the exact r=1 load route",
        )

        strict_raw = json.loads((results / "readiness_strict_N4096.json").read_text())
        original_raw = json.loads((results / "readiness_original_N4096.json").read_text())
        for raw, expected_mode, label in (
            (strict_raw, "strict_barrier", "strict"),
            (original_raw, "original", "original"),
        ):
            nonzero = [
                condition
                for condition in raw["conditions"]
                if float(condition["delay_us"]) > 0.0
            ]
            _require(
                {condition["mode"] for condition in raw["conditions"]}
                == {expected_mode},
                f"readiness {label}: unexpected execution mode",
            )
            _require(
                {int(condition["delay_ranks"][0]) for condition in nonzero}
                == set(range(8)),
                f"readiness {label}: not all ranks were scanned",
            )
            per_rank_delays = {
                rank: {
                    float(condition["delay_us"])
                    for condition in nonzero
                    if int(condition["delay_ranks"][0]) == rank
                }
                for rank in range(8)
            }
            _require(
                len({tuple(sorted(values)) for values in per_rank_delays.values()}) == 1,
                f"readiness {label}: rank delay grids differ",
            )

        route_ready_raw = json.loads((results / "route_ready_N4096.json").read_text())
        route_mode_pairs: dict[float, set[str]] = collections.defaultdict(set)
        for condition in route_ready_raw["conditions"]:
            route_mode_pairs[float(condition["requested_r"])].add(condition["mode"])
        _require(
            set(route_mode_pairs) == {1.0, 1.02, 1.05, 1.1, 1.2, 1.4},
            "route-ready: unexpected ratio grid",
        )
        _require(
            all(modes == {"original", "strict_barrier"} for modes in route_mode_pairs.values()),
            "route-ready: each ratio must have original and strict modes",
        )
        route_ready_baseline = readiness_payloads["route_ready"]["condition_routes"][
            "route-r1-strict"
        ]
        _require(
            route_ready_baseline == load_baseline_hash,
            "route-ready r=1 does not match the exact load baseline route",
        )

        readiness_analysis = json.loads(
            (results / "readiness_analysis_N4096" / "rank_readiness_analysis.json").read_text()
        )
        _require(
            [int(item["rank"]) for item in readiness_analysis["rank_summaries"]]
            == list(range(8)),
            "readiness analysis: rank summaries",
        )
        route_analysis = json.loads(
            (results / "route_ready_analysis_N4096" / "route_ready_analysis.json").read_text()
        )
        _require(
            "H is not subtracted again" in route_analysis["primary_model"],
            "route-ready analysis does not state the no-second-H rule",
        )
        _require(
            "delta=S*(r-1)" in route_analysis["padding_policy"],
            "route-ready analysis does not reject raw-r linear conversion",
        )
        route_checks.append(
            {
                "total_tokens": 4096,
                "readiness_route_sha256": load_baseline_hash,
                "readiness_ranks": list(range(8)),
                "route_ready_ratios": sorted(route_mode_pairs),
            }
        )

    if args.include_timelines:
        for name in TIMELINE_NAMES:
            path = results / f"{name}_N4096.json"
            validations.append(
                validate_raw(
                    path,
                    total_tokens=4096,
                    condition_count=1,
                    blocks=1,
                    iterations=1,
                )
            )
            reports = sorted((root / "traces" / "N4096").glob(f"{name}-rank*.nsys-rep"))
            _require(len(reports) == 8, f"{name}: expected eight Nsight reports")
            _require(all(report.stat().st_size > 0 for report in reports), f"{name}: empty Nsight report")

    summary = {
        "status": "PASS",
        "files": validations,
        "route_checks": route_checks,
        "totals": {
            "raw_files": len(validations),
            "records": sum(item["records"] for item in validations),
            "rank_observations": sum(item["rank_observations"] for item in validations),
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"PASS: {summary['totals']['raw_files']} files, "
        f"{summary['totals']['records']} records, "
        f"{summary['totals']['rank_observations']} rank observations"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
