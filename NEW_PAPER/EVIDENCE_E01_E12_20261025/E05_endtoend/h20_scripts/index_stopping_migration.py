#!/usr/bin/env python3
"""Validate and hash-index the frozen stopping migration-cost campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from typing import Any

from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    invert_p2l,
    sha256_file,
    write_json_atomic,
)

STOPPING_STUDY_ID = "h20-placement-stopping-v1"
BYTES_PER_EXPERT = 2 * 1536 * 4096 + 4096 * 1536 + 24 * 32 * 4 + 32 * 12 * 4
PAIRS_PER_TRANSACTION = 50


def finite_positive(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"nonpositive/nonfinite timing at {context}: {value!r}")
    return result


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
        "experiment": "h20_placement_stopping_migration_cost",
        "split": "stopping_migration",
        "start_id": start["start_id"],
        "capture_id": start["capture_id"],
        "batch_id": start["batch_id"],
        "layer_id": start["layer_id"],
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"raw migration header mismatch for {start['start_id']}: {mismatches}")
    if raw.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError(f"protocol hash differs in {start['start_id']}")
    if raw.get("layout_manifest", {}).get("sha256") != start["layout_manifest"]["sha256"]:
        raise ValueError(f"layout manifest hash differs in {start['start_id']}")
    gpu_inventory = raw.get("gpu_inventory", [])
    if len(gpu_inventory) != 8 or any(item.get("name") != "NVIDIA H20" for item in gpu_inventory):
        raise ValueError(f"GPU inventory differs in {start['start_id']}")
    expert = raw.get("expert_artifact", {})
    expert_path = pathlib.Path(expert["path"])
    if sha256_file(expert_path) != expert["sha256"] or expert_path.stat().st_size != expert["bytes"]:
        raise ValueError(f"expert artifact differs on disk in {start['start_id']}")
    accounting = raw.get("byte_accounting", {})
    if int(accounting.get("per_expert_bytes", -1)) != BYTES_PER_EXPERT:
        raise ValueError(f"byte accounting differs in {start['start_id']}")
    if not accounting.get("scale_tensors_included") or accounting.get("old_bench_p2p_cost_admissible"):
        raise ValueError(f"byte accounting policy differs in {start['start_id']}")
    semantics = raw.get("timing_semantics", {})
    for field in ("cross_device_cuda_timestamp_subtraction", "aliasing_or_preresident_copies"):
        if semantics.get(field):
            raise ValueError(f"prohibited timing semantics in {start['start_id']}: {field}")
    for field in ("setup_excluded_from_timing", "hash_verification_excluded_from_timing", "staging_zeroed_before_every_timed_transfer", "communicators_preconnected_before_timing"):
        if not semantics.get(field):
            raise ValueError(f"required timing semantics missing in {start['start_id']}: {field}")
    preconnect = raw.get("p2p_preconnect", {})
    if preconnect.get("performed_before_any_timed_transfer") is not True:
        raise ValueError(f"p2p preconnect record differs in {start['start_id']}")
    if not (isinstance(preconnect.get("local_wall_us"), (int, float)) and preconnect["local_wall_us"] > 0):
        raise ValueError(f"p2p preconnect wall differs in {start['start_id']}")
    setup = raw.get("untimed_setup", [])
    if len(setup) != 1 or setup[0].get("role") != "untimed_setup_migration_identity_to_base":
        raise ValueError(f"untimed setup record differs in {start['start_id']}")
    if "coordinator_wall_us" in setup[0]:
        raise ValueError(f"setup migration retained a coordinator wall value in {start['start_id']}")

    manifest = json.loads(pathlib.Path(start["layout_manifest"]["path"]).read_text())
    base_p2l = tuple(int(value) for value in manifest_layout(manifest, "base")["p2l"])
    base_l2p = invert_p2l(base_p2l)

    transactions = raw.get("transactions", [])
    if [tx["transaction_id"] for tx in transactions] != start["transaction_order"]:
        raise ValueError(f"transaction order differs in {start['start_id']}")
    if len(transactions) != 13:
        raise ValueError(f"transaction count differs in {start['start_id']}")
    pair_count = 0
    for tx in transactions:
        context = f"{start['start_id']}/{tx['transaction_id']}"
        pairs = tx.get("pairs", [])
        if len(pairs) != PAIRS_PER_TRANSACTION:
            raise ValueError(f"pair count differs at {context}")
        pair_count += len(pairs)
        if int(tx["bytes_per_expert"]) != BYTES_PER_EXPERT:
            raise ValueError(f"transaction byte accounting differs at {context}")
        if int(tx["bytes_per_direction"]) != len(tx["moves"]) * BYTES_PER_EXPERT:
            raise ValueError(f"transaction direction bytes differ at {context}")
        if tx["kind"] == "multi_swap_batch":
            expected_moves = sorted(
                logical for logical in range(NUM_EXPERTS)
                if logical // LOCAL_EXPERTS != base_l2p[logical] // LOCAL_EXPERTS
            )
            if sorted(int(move["logical"]) for move in tx["moves"]) != expected_moves:
                raise ValueError(f"multi-batch displacement set differs at {context}")
            for move in tx["moves"]:
                if int(move["from_rank"]) != int(move["logical"]) // LOCAL_EXPERTS:
                    raise ValueError(f"multi-batch source rank differs at {context}")
                if int(move["to_rank"]) != base_l2p[int(move["logical"])] // LOCAL_EXPERTS:
                    raise ValueError(f"multi-batch destination rank differs at {context}")
        else:
            layout_id = tx["transaction_id"].removeprefix("swap-")
            record = manifest_layout(manifest, layout_id)
            swap = record["swap"]
            if len(tx["moves"]) != 2:
                raise ValueError(f"single swap must have two moves at {context}")
            moves = {(int(move["logical"]), int(move["from_rank"]), int(move["to_rank"])) for move in tx["moves"]}
            expected = {
                (int(swap["logical_a"]), int(swap["rank_a"]), int(swap["rank_b"])),
                (int(swap["logical_b"]), int(swap["rank_b"]), int(swap["rank_a"])),
            }
            if moves != expected:
                raise ValueError(f"single swap moves differ from frozen manifest at {context}")
        for pair in pairs:
            finite_positive(pair["forward_wall_us"], f"{context}.pair{pair['pair_index']}.forward")
            finite_positive(pair["reverse_wall_us"], f"{context}.pair{pair['pair_index']}.reverse")
            if pair.get("hashes_all_matched_canonical") is not True:
                raise ValueError(f"hash verification flag differs at {context}.pair{pair['pair_index']}")
            for direction in ("forward", "reverse"):
                merkle = pair[f"{direction}_merkle_by_rank"]
                cuda = pair[f"{direction}_cuda_us_by_rank"]
                if len(merkle) != 8 or len(cuda) != 8:
                    raise ValueError(f"rank cardinality differs at {context}.pair{pair['pair_index']}")
                for value in cuda:
                    finite_positive(value, f"{context}.pair{pair['pair_index']}.{direction}.cuda")
            if pair["pair_index"] in (1, PAIRS_PER_TRANSACTION):
                for direction in ("forward", "reverse"):
                    hashes = pair.get(f"{direction}_full_hashes")
                    if not isinstance(hashes, dict) or len(hashes) != 8:
                        raise ValueError(f"full hash record differs at {context}.pair{pair['pair_index']}")
    integrity = raw.get("end_of_start_integrity", {})
    for field in ("canonical_tensors_unchanged", "live_resident_tensors_verified", "final_placement_equals_frozen_base"):
        if integrity.get(field) is not True:
            raise ValueError(f"end-of-start integrity differs in {start['start_id']}: {field}")
    return {
        "start_id": start["start_id"],
        "capture_id": start["capture_id"],
        "batch_id": start["batch_id"],
        "layer_id": start["layer_id"],
        "transactions": len(transactions),
        "pairs": pair_count,
    }


def manifest_layout(manifest: dict[str, Any], layout_id: str) -> dict[str, Any]:
    for layout in manifest["layouts"]:
        if layout["layout_id"] == layout_id:
            return layout
    raise ValueError(f"layout missing from manifest: {layout_id}")


def main() -> None:
    args_parser = argparse.ArgumentParser()
    root_default = pathlib.Path(__file__).resolve().parents[1]
    args_parser.add_argument("--root", type=pathlib.Path, default=root_default)
    args_parser.add_argument("--protocol", type=pathlib.Path, required=True)
    args_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = args_parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("study_id") != STOPPING_STUDY_ID:
        raise ValueError("migration protocol belongs to another study")
    if protocol.get("status") != "frozen_before_stopping_migration_timing":
        raise ValueError("migration protocol is not frozen")
    records = []
    for start in protocol["starts"]:
        result_path = (root / start["output_relative_path"]).resolve()
        failure_path = pathlib.Path(str(result_path).removesuffix(".json") + ".failure.json")
        if failure_path.exists():
            raise ValueError(f"retained failed start blocks indexing: {failure_path}")
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        raw = json.loads(result_path.read_text())
        validation = validate_result(raw, protocol=protocol, protocol_path=protocol_path, start=start)
        records.append(
            {
                **validation,
                "path": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            }
        )
    expected = int(protocol["design"]["capture_units"]) * int(protocol["design"]["starts_per_capture"])
    if len(records) != expected:
        raise ValueError(f"result count differs: {len(records)} != {expected}")
    output = {
        "schema_version": 1,
        "study_id": STOPPING_STUDY_ID,
        "split": "stopping_migration",
        "status": "all_stopping_migration_timing_results_validated_and_hash_indexed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "counts": {
            "starts": len(records),
            "transactions": sum(record["transactions"] for record in records),
            "pairs": sum(record["pairs"] for record in records),
        },
        "results": records,
    }
    write_json_atomic(args.output.resolve(), output)
    print(
        f"validated {len(records)} stopping migration starts; "
        f"index_sha256={sha256_file(args.output.resolve())}"
    )


if __name__ == "__main__":
    main()
