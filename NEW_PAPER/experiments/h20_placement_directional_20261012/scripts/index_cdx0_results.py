#!/usr/bin/env python3
# CD-X0 indexer for h20-placement-cdx0-v1: minimal-diff patched copy of the
# frozen index_crosstab_results.py (sha256 aa5139ba975cf52e08eb6e5d93b1ef282671a9a76898a66a88a39acfb7ee9db5). Changes: study/status
# identifiers, experiment label, fixed 5-column set, cdx0 role/order checks,
# flagged-duplicate map validation replacing the distinct-map and r_raw
# monotonicity checks (cross-domain bases are not required to beat identity
# imbalance on a foreign row; r_raw stays descriptive), and retired-start
# handling: a failed start is indexed as retired (failure record hashed)
# instead of blocking indexing (continue-on-failure discipline, amendment v1).
# All correctness, schedule, audit and hash validations inherited VERBATIM.
"""Validate and hash-index the frozen CD-X0 cross-domain replay campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from typing import Any

from cdx0_common import sha256_file, write_json_atomic

CDX0_STUDY_ID = "h20-placement-cdx0-v1"
CDX0_LAYOUT_IDS = (
    "identity",
    "base-of-sharegpt",
    "base-of-prover",
    "base-of-book",
    "base-of-code",
)


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
        "study_id": CDX0_STUDY_ID,
        "experiment": "h20_placement_cdx0_cross_domain_pilot",
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
    if timed_count != len(CDX0_LAYOUT_IDS):
        raise ValueError(f"cdx0 timing set size differs in {start['start_id']}")
    if len(correctness) != timed_count * 8:
        raise ValueError(f"correctness cardinality differs in {start['start_id']}")
    if sum(record["layout_id"] == "identity" for record in correctness) != 8:
        raise ValueError(f"identity tier-1 reference missing in {start['start_id']}")
    manifest_block = raw.get("layout_manifest", {})
    layout_ids = manifest_block.get("layout_ids", [])
    roles = manifest_block.get("layout_roles", {})
    if sorted(layout_ids) != sorted(start["primary_schedules"][0]):
        raise ValueError(f"manifest layout ids differ from schedules in {start['start_id']}")
    if sorted(roles) != sorted(layout_ids):
        raise ValueError(f"layout role keys differ in {start['start_id']}")
    if sum(role == "anchor_identity" for role in roles.values()) != 1:
        raise ValueError(f"identity anchor role missing in {start['start_id']}")
    if sum(role == "cdx0_base_column" for role in roles.values()) != 4:
        raise ValueError(f"cdx0 base column role count differs in {start['start_id']}")
    if tuple(layout_ids) != CDX0_LAYOUT_IDS:
        raise ValueError(f"manifest column order differs in {start['start_id']}")
    step_index = manifest_block.get("layout_step_index", {})
    if int(step_index.get("identity", -1)) != 0:
        raise ValueError(f"identity step index differs in {start['start_id']}")
    map_sha256s = manifest_block.get("map_sha256s", [])
    if len(map_sha256s) != len(CDX0_LAYOUT_IDS):
        raise ValueError(f"map sha cardinality differs in {start['start_id']}")
    sha_owner: dict[str, str] = {}
    observed_duplicates: set[tuple[str, str]] = set()
    for layout_id, digest in zip(layout_ids, map_sha256s):
        if digest in sha_owner:
            observed_duplicates.add(tuple(sorted((sha_owner[digest], layout_id))))
        else:
            sha_owner[digest] = layout_id
    flagged_duplicates = {
        tuple(sorted((entry["layout_a"], entry["layout_b"])))
        for entry in manifest_block.get("duplicate_base_maps", [])
    }
    if observed_duplicates != flagged_duplicates:
        raise ValueError(
            f"duplicate-map flag differs from observed duplicates in {start['start_id']}: "
            f"observed={sorted(observed_duplicates)} flagged={sorted(flagged_duplicates)}"
        )
    if any("identity" in pair for pair in flagged_duplicates):
        raise ValueError(f"identity map duplicated by a base column in {start['start_id']}")
    r_raw = manifest_block.get("layout_r_raw", {})
    for column in CDX0_LAYOUT_IDS:
        value = float(r_raw[column])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"nonpositive/nonfinite r_raw in {start['start_id']}: {column}")
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
    if protocol.get("study_id") != CDX0_STUDY_ID:
        raise ValueError("timing protocol belongs to another study")
    if protocol.get("status") != "frozen_before_cdx0_timing":
        raise ValueError("cdx0 timing protocol is not frozen")
    records = []
    retired = []
    for start in protocol["starts"]:
        result_path = (root / start["output_relative_path"]).resolve()
        failure_path = (root / start["failure_relative_path"]).resolve()
        if failure_path.exists() and result_path.exists():
            raise ValueError(f"start has both failure and result records: {start['start_id']}")
        if failure_path.exists():
            # Retired failed start: retained verbatim, never retried under this
            # frozen campaign (continue-on-failure discipline, amendment v1).
            failure_payload = json.loads(failure_path.read_text())
            if failure_payload.get("start_id") != start["start_id"]:
                raise ValueError(f"failure record start mismatch: {failure_path}")
            if failure_payload.get("retry_under_same_frozen_campaign_prohibited") is not True:
                raise ValueError(f"failure record lacks no-retry flag: {failure_path}")
            retired.append(
                {
                    "start_id": start["start_id"],
                    "capture_id": start["capture_id"],
                    "batch_id": start["batch_id"],
                    "layer_id": start["layer_id"],
                    "status": "retired_failed_start",
                    "path": str(failure_path),
                    "bytes": failure_path.stat().st_size,
                    "sha256": sha256_file(failure_path),
                }
            )
            continue
        if not result_path.is_file():
            raise FileNotFoundError(
                f"start has neither result nor failure record: {result_path}"
            )
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
    if len(records) + len(retired) != expected_starts:
        raise ValueError(
            f"result count differs: {len(records)} valid + {len(retired)} retired "
            f"!= {expected_starts}"
        )
    output = {
        "schema_version": 1,
        "study_id": CDX0_STUDY_ID,
        "split": protocol["split"],
        "status": "cdx0_timing_results_validated_and_hash_indexed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "counts": {
            "starts_expected": expected_starts,
            "starts_valid": len(records),
            "starts_retired": len(retired),
            "primary_records": sum(record["primary_records"] for record in records),
            "correctness_records": sum(record["correctness_records"] for record in records),
        },
        "results": records,
        "retired_starts": retired,
    }
    write_json_atomic(args.output.resolve(), output)
    print(
        f"validated {len(records)} cdx0 starts ({len(retired)} retired); "
        f"index_sha256={sha256_file(args.output.resolve())}"
    )


if __name__ == "__main__":
    main()
