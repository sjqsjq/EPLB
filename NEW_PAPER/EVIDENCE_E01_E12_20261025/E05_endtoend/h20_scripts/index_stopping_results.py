#!/usr/bin/env python3
"""Validate and hash-index the frozen low-r stopping execution campaign.

Checks mirror index_prediction_timing_results_v2.py (two-tier correctness gate,
schedule/order validation, source audit, hash re-verification) with the stopping
protocol binding: thirteen timed layouts (base + twelve residual candidates,
identity excluded from timing) across five primary blocks; correctness covers all
fourteen layouts including the identity tier-1 reference. There is no diagnostic
pass in the stopping execution protocol.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from typing import Any

from prediction_common import sha256_file, write_json_atomic

STOPPING_STUDY_ID = "h20-placement-stopping-v1"


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def finite_positive(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"nonpositive/nonfinite timing at {context}: {value!r}")
    return result


def validate_schedule_records(
    records: list[dict[str, Any]],
    *,
    schedules: list[list[str]],
    pass_name: str,
    detailed: bool,
    iterations: int,
) -> None:
    expected = [
        (block, order, layout_id)
        for block, schedule in enumerate(schedules)
        for order, layout_id in enumerate(schedule)
    ]
    actual = [
        (int(record["block"]), int(record["condition_order"]), record["layout_id"])
        for record in records
    ]
    if actual != expected:
        raise ValueError(f"{pass_name} timing order differs from frozen schedules")
    for index, record in enumerate(records):
        context = f"{pass_name}[{index}]"
        if record.get("pass") != pass_name:
            raise ValueError(f"wrong pass label at {context}")
        if record.get("observation_mode") != ("detailed" if detailed else "minimal"):
            raise ValueError(f"wrong observer mode at {context}")
        if int(record.get("iterations", -1)) != iterations:
            raise ValueError(f"wrong iteration count at {context}")
        finite_positive(record["completion_fenced_us_per_forward"], context)
        finite_positive(record["t_proxy_us_per_forward"], context)
        ranks = record.get("ranks", [])
        if len(ranks) != 8:
            raise ValueError(f"wrong rank cardinality at {context}")
        for rank, rank_record in enumerate(ranks):
            finite_positive(rank_record["local_us_per_forward"], f"{context}.rank{rank}")
            if detailed:
                values = rank_record.get("runner_run_iteration_us")
                if not isinstance(values, list) or len(values) != iterations:
                    raise ValueError(f"wrong runner diagnostic count at {context}.rank{rank}")
                for iteration, value in enumerate(values):
                    finite_positive(value, f"{context}.rank{rank}.runner[{iteration}]")
                summary = rank_record.get("runner_run_summary_us", {})
                if int(summary.get("count", -1)) != iterations:
                    raise ValueError(f"wrong runner diagnostic summary at {context}.rank{rank}")


def validate_result(
    raw: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_path: pathlib.Path,
    start: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "study_id": STOPPING_STUDY_ID,
        "experiment": "h20_placement_low_r_stopping_execution",
        "split": protocol["split"],
        "start_id": start["start_id"],
        "capture_id": start["capture_id"],
        "batch_id": start["batch_id"],
        "layer_id": start["layer_id"],
    }
    mismatches = {
        key: (raw.get(key), value)
        for key, value in expected.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"raw result header mismatch for {start['start_id']}: {mismatches}")
    if raw.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError(f"protocol hash differs in {start['start_id']}")
    if raw.get("layout_manifest", {}).get("sha256") != start["layout_manifest"]["sha256"]:
        raise ValueError(f"layout manifest hash differs in {start['start_id']}")
    if raw.get("timing_config") != protocol["timing"]:
        raise ValueError(f"timing config differs in {start['start_id']}")
    if raw.get("primary_schedules") != start["primary_schedules"]:
        raise ValueError(f"primary schedules differ in {start['start_id']}")
    if sorted(raw.get("timed_layout_ids", [])) != sorted(start["primary_schedules"][0]):
        raise ValueError(f"timed layout ids differ in {start['start_id']}")
    gpu_inventory = raw.get("gpu_inventory", [])
    if len(gpu_inventory) != 8 or any(item.get("name") != "NVIDIA H20" for item in gpu_inventory):
        raise ValueError(f"GPU inventory differs in {start['start_id']}")
    audit = raw.get("source_audit", {})
    prohibited = (
        "distributed_collective_in_layer",
        "sleep_or_spin_in_layer",
        "synthetic_delay_symbol_in_layer",
        "mid_layer_collective_or_gate",
        "artificial_delay_floor",
    )
    if any(audit.get(field) for field in prohibited):
        raise ValueError(f"source audit failed in {start['start_id']}: {audit}")
    gate = protocol.get("correctness_gate")
    if not isinstance(gate, dict):
        raise ValueError(f"protocol lacks the two-tier correctness gate for {start['start_id']}")
    if raw.get("correctness_gate") != gate:
        raise ValueError(f"raw correctness gate differs from protocol in {start['start_id']}")
    strict = gate["identity_versus_captured"]
    candidate = gate["candidate"]
    if float(strict["atol"]) != 2.0**-8 or float(strict["rtol"]) != 0.0:
        raise ValueError(f"tier-1 strictness relaxed in protocol for {start['start_id']}")
    correctness = raw.get("correctness", {}).get("records", [])
    timed_count = len(start["primary_schedules"][0])
    if timed_count != 13:
        raise ValueError(f"stopping protocol must time thirteen layouts in {start['start_id']}")
    if len(correctness) != (timed_count + 1) * 8:
        raise ValueError(f"correctness cardinality differs in {start['start_id']}")
    if sum(record["layout_id"] == "identity" for record in correctness) != 8:
        raise ValueError(f"identity tier-1 reference missing in {start['start_id']}")
    for record in correctness:
        for comparison in ("versus_captured_output", "versus_identity_replay"):
            metrics = record[comparison]
            if int(metrics["nonfinite_pair_count"]) > int(strict["max_nonfinite_pair_count"]):
                raise ValueError(
                    f"nonfinite outputs in {start['start_id']}: "
                    f"rank={record['rank']} layout={record['layout_id']} {comparison}"
                )
            if metrics.get("atol") != candidate["atol"] or metrics.get("rtol") != candidate["rtol"]:
                raise ValueError(f"correctness tolerance differs in {start['start_id']}")
            elements = int(metrics["elements"])
            if elements <= 0:
                raise ValueError(f"empty correctness comparison in {start['start_id']}")
            if int(metrics["isclose_failure_count"]) / elements > float(
                candidate["max_isclose_failure_fraction"]
            ):
                raise ValueError(
                    f"candidate sparsity cap exceeded in {start['start_id']}: "
                    f"rank={record['rank']} layout={record['layout_id']} {comparison}"
                )
            if (
                record["layout_id"] == "identity"
                and comparison == "versus_captured_output"
                and float(metrics["max_abs_error"]) > float(strict["atol"])
            ):
                raise ValueError(f"tier-1 identity strict bound exceeded in {start['start_id']}")
    for record in raw.get("frozen_tensor_hashes", []):
        if record.get("initial") != record.get("final"):
            raise ValueError(f"frozen tensor hash changed in {start['start_id']}")
    if len(raw.get("frozen_tensor_hashes", [])) != 8:
        raise ValueError(f"frozen tensor hash cardinality differs in {start['start_id']}")
    iterations = int(protocol["timing"]["timed_forwards_per_activation"])
    validate_schedule_records(
        raw.get("primary_timing_records", []),
        schedules=start["primary_schedules"],
        pass_name="primary",
        detailed=False,
        iterations=iterations,
    )
    return {
        "start_id": start["start_id"],
        "capture_id": start["capture_id"],
        "batch_id": start["batch_id"],
        "layer_id": start["layer_id"],
        "primary_records": len(raw["primary_timing_records"]),
        "correctness_records": len(correctness),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("study_id") != STOPPING_STUDY_ID:
        raise ValueError("timing protocol belongs to another study")
    if protocol.get("status") != "frozen_before_stopping_timing":
        raise ValueError("stopping timing protocol is not frozen")
    records = []
    for start in protocol["starts"]:
        result_path = (root / start["output_relative_path"]).resolve()
        failure_path = pathlib.Path(str(result_path).removesuffix(".json") + ".failure.json")
        if failure_path.exists():
            raise ValueError(f"retained failed start blocks indexing: {failure_path}")
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        raw = json.loads(result_path.read_text())
        validation = validate_result(
            raw,
            protocol=protocol,
            protocol_path=protocol_path,
            start=start,
        )
        records.append(
            {
                **validation,
                "path": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            }
        )
    expected_starts = int(protocol["design"]["capture_units"]) * int(
        protocol["design"]["starts_per_capture"]
    )
    if len(records) != expected_starts:
        raise ValueError(f"result count differs: {len(records)} != {expected_starts}")
    output = {
        "schema_version": 1,
        "study_id": STOPPING_STUDY_ID,
        "split": protocol["split"],
        "status": f"all_{protocol['split']}_timing_results_validated_and_hash_indexed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "counts": {
            "starts": len(records),
            "primary_records": sum(record["primary_records"] for record in records),
            "correctness_records": sum(record["correctness_records"] for record in records),
        },
        "results": records,
    }
    write_json_atomic(args.output.resolve(), output)
    print(
        f"validated {len(records)} {protocol['split']} starts; "
        f"index_sha256={sha256_file(args.output.resolve())}"
    )


if __name__ == "__main__":
    main()
