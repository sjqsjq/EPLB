#!/usr/bin/env python3
"""Freeze (or verify) the stopping execution and migration protocols.

Both protocols are frozen BEFORE any stopping timing. The frozen stopping_study
spec inside configs/followup_v1/protocol.json is asserted verbatim; every input
(model lock, stopping layout manifests, tolerance v2, capture index) is hash
verified; implementation records cover every script the launchers will run.
--verify-only regenerates all deterministic content and compares it against the
frozen protocol files without writing anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
from typing import Any

from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    STUDY_ID as SOURCE_STUDY_ID,
    invert_p2l,
    sha256_file,
    write_json_atomic,
)

STOPPING_STUDY_ID = "h20-placement-stopping-v1"
EXECUTION_PROTOCOL_ID = "h20-placement-stopping-execution-v1"
EXECUTION_GENERATOR = "stopping-execution-protocol-v1"
MIGRATION_PROTOCOL_ID = "h20-placement-stopping-migration-v1"
MIGRATION_GENERATOR = "stopping-migration-protocol-v1"
EXECUTION_STATUS = "frozen_before_stopping_timing"
MIGRATION_STATUS = "frozen_before_stopping_migration_timing"
TIMED_LAYOUT_IDS = (
    "base",
    "gain-01", "gain-02", "gain-03", "gain-04",
    "near-01", "near-02", "near-03", "near-04",
    "neg-01", "neg-02", "neg-03", "neg-04",
)
MANIFEST_LAYOUT_IDS = ("identity",) + TIMED_LAYOUT_IDS
BYTES_PER_EXPERT = 2 * 1536 * 4096 + 4096 * 1536 + 24 * 32 * 4 + 32 * 12 * 4
PER_TENSOR_BYTES = {
    "w13_weight": 2 * 1536 * 4096,
    "w2_weight": 4096 * 1536,
    "w13_weight_scale_inv": 24 * 32 * 4,
    "w2_weight_scale_inv": 32 * 12 * 4,
}
PAIRS_PER_TRANSACTION = 50
EXECUTION_IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "prediction_models_record": "prediction_models.py",
    "replay_placement_record": "replay_placement.py",
    "stopping_replay_record": "stopping_replay.py",
    "index_stopping_results_record": "index_stopping_results.py",
    "generate_stopping_layouts_record": "generate_stopping_layouts.py",
    "generate_prediction_evaluation_layouts_record": "generate_prediction_evaluation_layouts.py",
    "run_stopping_execution_record": "run_stopping_execution.sh",
}
MIGRATION_IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "migration_cost_record": "migration_cost.py",
    "index_stopping_migration_record": "index_stopping_migration.py",
    "generate_stopping_layouts_record": "generate_stopping_layouts.py",
    "run_stopping_migration_record": "run_stopping_migration.sh",
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--execution-protocol", type=pathlib.Path)
    parser.add_argument("--migration-protocol", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def derived_seed(generator_version: str, *parts: str) -> int:
    text = "|".join((STOPPING_STUDY_ID, generator_version, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def schedule(seed: int, block: int) -> list[str]:
    order = list(range(len(TIMED_LAYOUT_IDS)))
    random.Random(seed + block).shuffle(order)
    return [TIMED_LAYOUT_IDS[index] for index in order]


def validate_inputs(root: pathlib.Path) -> dict[str, Any]:
    followup_path = root / "configs/followup_v1/protocol.json"
    followup = json.loads(followup_path.read_text())
    spec = followup["stopping_study"]
    expected_spec = {
        "activation_warmup": 10,
        "base": "deterministic low-r lexicographic greedy legal-swap local optimum",
        "blocks_per_start": 5,
        "candidate_count_per_capture": 12,
        "candidate_strata": [
            "four largest locked-M2 predicted gains",
            "four locked-M2 predictions closest to zero",
            "four most negative locked-M2 predictions",
        ],
        "migration": {
            "complete_w13_w2_and_both_scale_tensors": True,
            "forward_reverse_pairs_per_start": 50,
            "old_bench_p2p_cost_is_inadmissible": True,
            "separate_from_execution_timing": True,
            "starts": 4,
        },
        "practical_equivalence_margin_us": 22.948043518066406,
        "starts": 4,
        "timed_forwards": 50,
    }
    drift = {key: (spec.get(key), value) for key, value in expected_spec.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"frozen stopping_study spec drift: {drift}")

    lock_path = root / "configs/followup_v1/prediction_model_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock is not in locked status")
    lock_sha = sha256_file(lock_path)
    for name, record in lock["locked_implementation"].items():
        actual = sha256_file(pathlib.Path(record["path"]))
        if actual != record["sha256"]:
            raise ValueError(f"locked implementation changed since model lock ({name})")

    tolerance_path = root / "configs/followup_v1/prediction_tolerance_v2.json"
    tolerance = json.loads(tolerance_path.read_text())
    if tolerance.get("status") != "frozen_before_evaluation_attempt02_timing":
        raise ValueError("tolerance v2 status differs")
    gate = tolerance["correctness_gate"]
    if float(gate["identity_versus_captured"]["atol"]) != 2.0**-8 or float(gate["identity_versus_captured"]["rtol"]) != 0.0:
        raise ValueError("tier-1 strictness must equal the pilot tolerance")

    layout_index_path = root / "inputs/stopping_v1/layouts/index.json"
    layout_index = json.loads(layout_index_path.read_text())
    if layout_index.get("status") != "all_stopping_layouts_frozen_before_timing":
        raise ValueError("stopping layout index is not frozen")
    if layout_index.get("model_lock", {}).get("sha256") != lock_sha:
        raise ValueError("stopping layout index references a different model lock")
    counts = layout_index.get("counts", {})
    if (
        counts.get("capture_units") != 12
        or counts.get("layouts_per_capture") != 14
        or counts.get("timed_layouts_per_capture") != 13
        or counts.get("residual_candidates_per_capture") != 12
        or counts.get("unique_timed_map_hashes") != 156
    ):
        raise ValueError(f"stopping layout index counts differ: {counts}")
    manifests: dict[str, dict[str, Any]] = {}
    manifest_payloads: dict[str, dict[str, Any]] = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"stopping layout manifest hash differs: {record['capture_id']}")
        manifest = json.loads(path.read_text())
        capture_id = record["capture_id"]
        if manifest.get("capture_id") != capture_id or manifest.get("split") != "stopping":
            raise ValueError(f"manifest header differs: {capture_id}")
        if manifest.get("selection_uses_performance_timing") is not False:
            raise ValueError(f"manifest selection discipline differs: {capture_id}")
        if tuple(layout["layout_id"] for layout in manifest["layouts"]) != MANIFEST_LAYOUT_IDS:
            raise ValueError(f"manifest layout ids differ: {capture_id}")
        if manifest.get("model_lock", {}).get("sha256") != lock_sha:
            raise ValueError(f"manifest references a different model lock: {capture_id}")
        manifests[capture_id] = record
        manifest_payloads[capture_id] = manifest
    if len(manifests) != 12:
        raise ValueError("expected twelve stopping layout manifests")

    capture_index_path = root / "inputs/prediction_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    if capture_index.get("status") != "all_prediction_capture_inputs_validated_and_hash_indexed":
        # status string recorded at capture freeze; accept only the frozen one
        raise ValueError(f"capture index status differs: {capture_index.get('status')}")

    return {
        "followup_path": followup_path,
        "followup": followup,
        "spec": spec,
        "lock_path": lock_path,
        "lock_sha": lock_sha,
        "tolerance_path": tolerance_path,
        "gate": gate,
        "layout_index_path": layout_index_path,
        "manifests": manifests,
        "manifest_payloads": manifest_payloads,
        "capture_index_path": capture_index_path,
    }


def transactions_for(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    base = next(layout for layout in manifest["layouts"] if layout["layout_id"] == "base")
    base_p2l = tuple(int(value) for value in base["p2l"])
    base_l2p = invert_p2l(base_p2l)
    moves = [
        {
            "logical": logical,
            "from_rank": logical // LOCAL_EXPERTS,
            "to_rank": base_l2p[logical] // LOCAL_EXPERTS,
        }
        for logical in range(NUM_EXPERTS)
        if logical // LOCAL_EXPERTS != base_l2p[logical] // LOCAL_EXPERTS
    ]
    transactions = [
        {
            "transaction_id": "multi-batch",
            "kind": "multi_swap_batch",
            "description": "actual identity-to-base multi-swap batch (base greedy displacement set)",
            "moves": moves,
            "bytes_per_expert": BYTES_PER_EXPERT,
            "bytes_per_direction": len(moves) * BYTES_PER_EXPERT,
            "displaced_expert_count": len(moves),
        }
    ]
    for layout in manifest["layouts"][2:]:
        swap = layout["swap"]
        rank_a, rank_b = int(swap["rank_a"]), int(swap["rank_b"])
        logical_a, logical_b = int(swap["logical_a"]), int(swap["logical_b"])
        if rank_a == rank_b:
            raise ValueError("residual swap is not cross-rank")
        transactions.append(
            {
                "transaction_id": f"swap-{layout['layout_id']}",
                "kind": "single_swap",
                "stratum": layout["stratum"],
                "map_sha256": layout["map_sha256"],
                "locked_m2_predicted_signed_gain_us": layout["locked_m2_predicted_signed_gain_us"],
                "logical_a": logical_a,
                "logical_b": logical_b,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "moves": [
                    {"logical": logical_a, "from_rank": rank_a, "to_rank": rank_b},
                    {"logical": logical_b, "from_rank": rank_b, "to_rank": rank_a},
                ],
                "bytes_per_expert": BYTES_PER_EXPERT,
                "bytes_per_direction": 2 * BYTES_PER_EXPERT,
            }
        )
    if len(transactions) != 13:
        raise ValueError("transaction count differs from frozen design")
    return transactions


def build_execution(context: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    spec = context["spec"]
    followup = context["followup"]
    manifests = context["manifests"]
    starts = []
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        record = manifests[capture_id]
        for start_number in range(1, int(spec["starts"]) + 1):
            start_id = f"stop-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(EXECUTION_GENERATOR, capture_id, f"stopping-execution-start-{start_number}")
            starts.append(
                {
                    "start_id": start_id,
                    "capture_id": capture_id,
                    "batch_id": record["batch_id"],
                    "layer_id": record["layer_id"],
                    "layout_manifest": {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    },
                    "random_seed": seed,
                    "primary_schedules": [
                        schedule(seed, block) for block in range(int(spec["blocks_per_start"]))
                    ],
                    "output_relative_path": f"results/stopping_v1/execution/raw/{start_id}.json",
                    "log_relative_path": f"results/stopping_v1/execution/logs/{start_id}.log",
                    "failure_relative_path": f"results/stopping_v1/execution/raw/{start_id}.failure.json",
                }
            )
    implementation = {
        name: file_record(root / "scripts" / filename)
        for name, filename in EXECUTION_IMPLEMENTATION_FILES.items()
    }
    lock = json.loads(context["lock_path"].read_text())
    for name in sorted(lock["locked_implementation"]):
        if name in implementation and implementation[name]["sha256"] != lock["locked_implementation"][name]["sha256"]:
            raise ValueError(f"implementation hash differs from model lock: {name}")
    return {
        "schema_version": 1,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "study_id": STOPPING_STUDY_ID,
        "source_study_id": SOURCE_STUDY_ID,
        "status": EXECUTION_STATUS,
        "split": "stopping",
        "attempt": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": EXECUTION_GENERATOR,
            "seed_derivation": "uint64 big-endian SHA256(stopping_study|version|parts) mod 2^62",
        },
        "scope": followup["scope"],
        "followup_protocol": file_record(context["followup_path"]),
        "stopping_study_spec": spec,
        "capture_index": file_record(context["capture_index_path"]),
        "layout_index": file_record(context["layout_index_path"]),
        "layout_manifests": [manifests[capture_id] for capture_id in sorted(manifests)],
        "model_lock": file_record(context["lock_path"]),
        "tolerance_file": file_record(context["tolerance_path"]),
        "correctness_gate": context["gate"],
        "implementation": implementation,
        "design": {
            "capture_units": 12,
            "starts_per_capture": int(spec["starts"]),
            "fresh_start_semantics": (
                "one torchrun process group per start; every start reloads all "
                "immutable artifacts and rebuilds weights before warmup"
            ),
            "timed_layout_ids": list(TIMED_LAYOUT_IDS),
            "identity_role": "correctness_reference_only_never_timed",
            "response_definition": (
                "start-level Q(candidate vs base) = mean of the five paired block effects; "
                "each block effect is completion_fenced_us_per_forward(base) - "
                "completion_fenced_us_per_forward(candidate) within the same randomized block"
            ),
            "practical_equivalence_margin_us": float(spec["practical_equivalence_margin_us"]),
            "equivalence_rule": (
                "only an interval wholly inside [-epsilon,+epsilon] supports practical "
                "equivalence; crossing zero alone does not"
            ),
            "report_all_signed_effects_including_negative": True,
            "no_best_measured_prefix_selection": True,
            "zero_crossing_is_not_a_deadzone": True,
            "no_slack_subtracted_from_measured_q": True,
            "no_refit_before_or_during_stopping": True,
            "layout_purpose": "low_r_residual_swap_stopping_measurement",
        },
        "timing": {
            "initial_warmup_forwards_per_layout": 10,
            "activation_warmup_forwards": int(spec["activation_warmup"]),
            "timed_forwards_per_activation": int(spec["timed_forwards"]),
            "blocks_per_start": int(spec["blocks_per_start"]),
            "diagnostic_blocks_per_start": 0,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "schedule_algorithm": (
                "random.Random(seed + block).shuffle over the frozen timed layout "
                "order (base + twelve residual candidates; identity is never timed)"
            ),
            "primary_metric": "completion_fenced_us_per_forward",
            "diagnostic_pass": False,
        },
        "result_index_relative_path": "results/stopping_v1/execution/index.json",
        "prohibitions": followup["prohibitions"],
        "starts": starts,
    }


def build_migration(context: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    spec = context["spec"]
    followup = context["followup"]
    manifests = context["manifests"]
    payloads = context["manifest_payloads"]
    transactions = {
        capture_id: transactions_for(payloads[capture_id]) for capture_id in sorted(manifests)
    }
    starts = []
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        record = manifests[capture_id]
        for start_number in range(1, int(spec["migration"]["starts"]) + 1):
            start_id = f"mig-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(MIGRATION_GENERATOR, capture_id, f"stopping-migration-start-{start_number}")
            starts.append(
                {
                    "start_id": start_id,
                    "capture_id": capture_id,
                    "batch_id": record["batch_id"],
                    "layer_id": record["layer_id"],
                    "layout_manifest": {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    },
                    "random_seed": seed,
                    "transaction_order": [tx["transaction_id"] for tx in transactions[capture_id]],
                    "pairs_per_transaction": int(spec["migration"]["forward_reverse_pairs_per_start"]),
                    "output_relative_path": f"results/stopping_v1/migration/raw/{start_id}.json",
                    "log_relative_path": f"results/stopping_v1/migration/logs/{start_id}.log",
                    "failure_relative_path": f"results/stopping_v1/migration/raw/{start_id}.failure.json",
                }
            )
    implementation = {
        name: file_record(root / "scripts" / filename)
        for name, filename in MIGRATION_IMPLEMENTATION_FILES.items()
    }
    return {
        "schema_version": 1,
        "protocol_id": MIGRATION_PROTOCOL_ID,
        "study_id": STOPPING_STUDY_ID,
        "source_study_id": SOURCE_STUDY_ID,
        "status": MIGRATION_STATUS,
        "split": "stopping_migration",
        "attempt": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": MIGRATION_GENERATOR,
            "seed_derivation": "uint64 big-endian SHA256(stopping_study|version|parts) mod 2^62",
        },
        "scope": followup["scope"],
        "followup_protocol": file_record(context["followup_path"]),
        "stopping_study_spec": spec,
        "capture_index": file_record(context["capture_index_path"]),
        "layout_index": file_record(context["layout_index_path"]),
        "layout_manifests": [manifests[capture_id] for capture_id in sorted(manifests)],
        "model_lock": file_record(context["lock_path"]),
        "implementation": implementation,
        "transactions": transactions,
        "design": {
            "capture_units": 12,
            "starts_per_capture": int(spec["migration"]["starts"]),
            "transactions_per_capture": 13,
            "pairs_per_transaction": int(spec["migration"]["forward_reverse_pairs_per_start"]),
            "complete_w13_w2_and_both_scale_tensors": bool(
                spec["migration"]["complete_w13_w2_and_both_scale_tensors"]
            ),
            "old_bench_p2p_cost_is_inadmissible": bool(
                spec["migration"]["old_bench_p2p_cost_is_inadmissible"]
            ),
            "separate_from_execution_timing": bool(spec["migration"]["separate_from_execution_timing"]),
            "transaction_order_note": (
                "multi-batch pairs run first from the identity state; one untimed setup "
                "migration then moves real tensors to base before the twelve single swaps"
            ),
            "timed_region": (
                "begins after start alignment on rank 0 monotonic clock; includes actual "
                "transfer, completion ack, mapping publication; ends after every rank "
                "acknowledged readiness; no cross-device timestamp subtraction"
            ),
            "communicator_preconnect": (
                "an untimed one-byte all-pairs batch_isend_irecv exchange establishes "
                "every NCCL P2P channel before the first timed transfer; its wall is "
                "recorded separately and never enters any transaction timing"
            ),
            "verification": (
                "staging destinations zeroed before every timed transfer; every received "
                "tensor sha256-verified against the canonical artifact hash outside timing; "
                "rank 0 independently recomputes each receiving rank's merkle from the frozen "
                "move list; full per-tensor hashes retained for pairs 1 and 50"
            ),
            "aliasing_or_preresident_copies": False,
            "all_repetitions_retained": True,
            "fastest_repeat_bandwidth_reporting": False,
        },
        "byte_accounting": {
            "per_tensor_bytes": PER_TENSOR_BYTES,
            "bytes_per_expert": BYTES_PER_EXPERT,
        },
        "result_index_relative_path": "results/stopping_v1/migration/index.json",
        "prohibitions": followup["prohibitions"],
        "starts": starts,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    execution_path = (
        args.execution_protocol or root / "configs/followup_v1/stopping_execution_protocol.json"
    ).resolve()
    migration_path = (
        args.migration_protocol or root / "configs/followup_v1/stopping_migration_protocol.json"
    ).resolve()
    context = validate_inputs(root)
    execution = build_execution(context, root)
    migration = build_migration(context, root)
    if args.verify_only:
        if not execution_path.exists() or not migration_path.exists():
            raise FileNotFoundError("verify-only requires both frozen protocols")
        frozen_execution = json.loads(execution_path.read_text())
        frozen_migration = json.loads(migration_path.read_text())
        for frozen, regenerated, label, status, protocol_id in (
            (frozen_execution, execution, "execution", EXECUTION_STATUS, EXECUTION_PROTOCOL_ID),
            (frozen_migration, migration, "migration", MIGRATION_STATUS, MIGRATION_PROTOCOL_ID),
        ):
            if frozen.get("status") != status or frozen.get("protocol_id") != protocol_id:
                raise ValueError(f"{label}: frozen protocol identity differs")
            for key in ("starts", "design", "prohibitions", "stopping_study_spec", "scope"):
                if key in regenerated and frozen.get(key) != regenerated[key]:
                    raise ValueError(f"{label}: frozen {key} differs from deterministic regeneration")
            if "transactions" in regenerated and frozen.get("transactions") != regenerated["transactions"]:
                raise ValueError(f"{label}: frozen transactions differ from deterministic regeneration")
            if "timing" in regenerated and frozen.get("timing") != regenerated["timing"]:
                raise ValueError(f"{label}: frozen timing differs from deterministic regeneration")
            if "correctness_gate" in regenerated and frozen.get("correctness_gate") != regenerated["correctness_gate"]:
                raise ValueError(f"{label}: frozen correctness gate differs")
            for name, record in frozen.get("implementation", {}).items():
                actual = sha256_file(pathlib.Path(record["path"]))
                if actual != record["sha256"]:
                    raise ValueError(f"{label}: implementation hash differs on disk: {name}")
            generator = frozen.get("generator", {})
            if sha256_file(pathlib.Path(generator["path"])) != generator["sha256"]:
                raise ValueError(f"{label}: generator script changed since freeze")
            print(
                f"{label} protocol verified: {len(frozen['implementation'])} implementation "
                f"records, {len(frozen['starts'])} starts, sha256={sha256_file(execution_path if label=='execution' else migration_path)}"
            )
        return
    if execution_path.exists() or migration_path.exists():
        raise FileExistsError("refusing to overwrite frozen stopping protocols")
    write_json_atomic(execution_path, execution)
    write_json_atomic(migration_path, migration)
    print(f"frozen {execution_path}")
    print(f"execution_protocol_sha256={sha256_file(execution_path)}")
    print(f"frozen {migration_path}")
    print(f"migration_protocol_sha256={sha256_file(migration_path)}")
    print(f"execution starts={len(execution['starts'])} migration starts={len(migration['starts'])}")


if __name__ == "__main__":
    main()
