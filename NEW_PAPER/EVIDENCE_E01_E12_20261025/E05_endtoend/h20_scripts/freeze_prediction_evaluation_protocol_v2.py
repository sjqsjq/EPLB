#!/usr/bin/env python3
"""Freeze evaluation timing protocol attempt-02 with the two-tier correctness gate.

Attempt-01 was retired after its frozen strict correctness gate blocked
eval-04-start-1 on isolated one-ulp bf16 flips inherent to placement-permuted
FP8 grouped-GEMM/combine reordering (see results/prediction_v1/evaluation/
retired_attempt01.json). Attempt-02 reruns all 48 starts fresh under a new
frozen protocol; tier-1 identity-versus-captured strictness is unchanged, and
no attempt-01 timing value is used anywhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
from typing import Any

from prediction_common import STUDY_ID, sha256_file, write_json_atomic

PROTOCOL_ID = "h20-placement-prediction-evaluation-v2"
GENERATOR_VERSION = "evaluation-protocol-v2"
LAYOUT_IDS = (
    "identity",
    "raw-01",
    "raw-02",
    "padding-01",
    "padding-02",
    "m2-01",
    "m2-02",
    "random-01",
    "random-02",
)
LOCK_STATUS = "models_locked_before_evaluation_layout_generation_and_timing"
LAYOUT_INDEX_STATUS = "all_evaluation_layouts_frozen_after_model_lock_before_timing"
IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "prediction_models_record": "prediction_models.py",
    "prediction_replay_record": "prediction_replay.py",
    "prediction_replay_v2_record": "prediction_replay_v2.py",
    "replay_placement_record": "replay_placement.py",
    "fit_prediction_models_record": "fit_prediction_models.py",
    "generate_prediction_calibration_layouts_record": "generate_prediction_calibration_layouts.py",
    "generate_prediction_evaluation_layouts_record": "generate_prediction_evaluation_layouts.py",
    "index_prediction_timing_results_record": "index_prediction_timing_results.py",
    "index_prediction_timing_results_v2_record": "index_prediction_timing_results_v2.py",
    "verify_prediction_timing_protocol_record": "verify_prediction_timing_protocol.py",
    "run_prediction_evaluation_record": "run_prediction_evaluation.sh",
    "run_prediction_evaluation_v2_record": "run_prediction_evaluation_v2.sh",
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def derived_seed(*parts: str) -> int:
    text = "|".join((STUDY_ID, GENERATOR_VERSION, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def schedule(seed: int, block: int, offset: int = 0) -> list[str]:
    order = list(range(len(LAYOUT_IDS)))
    random.Random(seed + offset + block).shuffle(order)
    return [LAYOUT_IDS[index] for index in order]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (
        args.output
        or root / "configs/followup_v1/prediction_evaluation_protocol_attempt02.json"
    ).resolve()

    followup = root / "configs/followup_v1/protocol.json"
    capture_index = root / "inputs/prediction_v1/capture_index.json"
    tolerance_v1 = root / "configs/followup_v1/prediction_tolerance.json"
    tolerance_v2_path = root / "configs/followup_v1/prediction_tolerance_v2.json"
    lock_path = root / "configs/followup_v1/prediction_model_lock.json"
    layout_index_path = root / "inputs/prediction_v1/evaluation_layouts/index.json"
    retirement_path = root / "results/prediction_v1/evaluation/retired_attempt01.json"
    attempt01_protocol = root / "configs/followup_v1/prediction_evaluation_protocol.json"

    tolerance_v2 = json.loads(tolerance_v2_path.read_text())
    if tolerance_v2.get("status") != "frozen_before_evaluation_attempt02_timing":
        raise ValueError("tolerance v2 is not frozen before attempt-02 timing")
    gate = tolerance_v2["correctness_gate"]
    if float(gate["identity_versus_captured"]["atol"]) != 2.0**-8 or float(
        gate["identity_versus_captured"]["rtol"]
    ) != 0.0:
        raise ValueError("tier-1 strictness must equal the pilot tolerance")

    lock = json.loads(lock_path.read_text())
    if lock.get("status") != LOCK_STATUS:
        raise ValueError("model lock is not in locked status")
    lock_sha = sha256_file(lock_path)
    for name, record in lock["locked_implementation"].items():
        actual = sha256_file(pathlib.Path(record["path"]))
        if actual != record["sha256"]:
            raise ValueError(f"locked implementation changed since model lock ({name})")

    retirement = json.loads(retirement_path.read_text())
    if retirement.get("status") != "evaluation_protocol_attempt01_retired_correctness_gate_blocked_campaign":
        raise ValueError("attempt-01 retirement record is missing or in wrong status")
    if retirement.get("timed_values_persisted_for_failed_start") is not False:
        raise ValueError("retirement record must state no timed values were persisted")

    layout_index = json.loads(layout_index_path.read_text())
    if layout_index.get("status") != LAYOUT_INDEX_STATUS:
        raise ValueError("evaluation layout index is not in frozen post-lock status")
    if layout_index.get("model_lock", {}).get("sha256") != lock_sha:
        raise ValueError("evaluation layout index references a different model lock")
    counts = layout_index.get("counts", {})
    if (
        counts.get("capture_units") != 12
        or counts.get("layouts_per_capture") != 9
        or counts.get("unique_nonidentity_map_hashes") != 96
    ):
        raise ValueError(f"evaluation layout index counts differ: {counts}")
    manifests = {record["capture_id"]: record for record in layout_index["manifests"]}
    if len(manifests) != 12:
        raise ValueError("expected twelve evaluation layout manifests")
    for capture_id, record in manifests.items():
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"evaluation layout manifest hash differs: {capture_id}")
        manifest = json.loads(path.read_text())
        if manifest.get("model_lock", {}).get("sha256") != lock_sha:
            raise ValueError(f"manifest references a different model lock: {capture_id}")
        if manifest.get("split") != "evaluation":
            raise ValueError(f"manifest split differs: {capture_id}")
        if tuple(layout["layout_id"] for layout in manifest["layouts"]) != LAYOUT_IDS:
            raise ValueError(f"manifest layout IDs/order differ: {capture_id}")

    starts = []
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        manifest_record = manifests[capture_id]
        for start_number in range(1, 5):
            start_id = f"eval-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(capture_id, f"timing-start-{start_number}")
            starts.append(
                {
                    "start_id": start_id,
                    "capture_id": capture_id,
                    "batch_id": manifest_record["batch_id"],
                    "layer_id": manifest_record["layer_id"],
                    "layout_manifest": {
                        "path": manifest_record["path"],
                        "bytes": manifest_record["bytes"],
                        "sha256": manifest_record["sha256"],
                    },
                    "random_seed": seed,
                    "primary_schedules": [schedule(seed, block) for block in range(4)],
                    "diagnostic_schedules": [
                        schedule(seed, block, offset=1_000_003) for block in range(4)
                    ],
                    "output_relative_path": f"results/prediction_v1/evaluation/attempt02/raw/{start_id}.json",
                    "log_relative_path": f"results/prediction_v1/evaluation/attempt02/logs/{start_id}.log",
                    "failure_relative_path": f"results/prediction_v1/evaluation/attempt02/raw/{start_id}.failure.json",
                }
            )

    implementation = {
        name: file_record(root / "scripts" / filename)
        for name, filename in IMPLEMENTATION_FILES.items()
    }
    for name in sorted(lock["locked_implementation"]):
        locked = lock["locked_implementation"][name]
        if implementation[name]["sha256"] != locked["sha256"]:
            raise ValueError(f"implementation hash differs from model lock: {name}")

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "study_id": STUDY_ID,
        "status": "frozen_before_evaluation_timing",
        "split": "evaluation",
        "attempt": 2,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
            "seed_derivation": "uint64 big-endian SHA256(study|version|parts) mod 2^62",
        },
        "scope": json.loads(followup.read_text())["scope"],
        "followup_protocol": file_record(followup),
        "capture_index": file_record(capture_index),
        "layout_index": file_record(layout_index_path),
        "tolerance_file": file_record(tolerance_v2_path),
        "superseded_tolerance_file_v1": file_record(tolerance_v1),
        "model_lock": file_record(lock_path),
        "retired_attempt01": file_record(retirement_path),
        "attempt01_protocol": file_record(attempt01_protocol),
        "correctness_gate": gate,
        "correctness_gate_amendment": {
            "reason": (
                "attempt-01 strict absolute gate blocked on isolated one-ulp bf16 flips "
                "inherent to placement-permuted FP8 grouped-GEMM/combine reordering"
            ),
            "evidence": [
                "calibration correctness records (pre-evaluation): identity-vs-captured max_abs=0; candidate-vs-identity nonzero in 3456/3456 comparisons, max 0.0029296875",
                "bf16 8-bit mantissa: one rounding quantum is < 2^-7 relative; absolute 2^-8 bound fails at magnitudes >= 1",
                "attempt-01 failure record: rank 0 only, layouts raw-01/02+padding-01/02, 2-6 elements of 16,613,376, max_abs exactly 2^-7, zero nonfinite, identity/m2/random passed",
            ],
            "tier1_never_relaxed": True,
            "timing_outcomes_used_for_amendment": False,
            "attempt01_completed_starts_excluded_from_all_analysis": True,
        },
        "implementation": implementation,
        "design": {
            "capture_units": 12,
            "starts_per_capture": 4,
            "fresh_start_semantics": (
                "one torchrun process group per start; every start reloads all "
                "immutable artifacts and rebuilds weights before warmup"
            ),
            "layout_ids": list(LAYOUT_IDS),
            "response_definition": (
                "start-level Q = mean of four paired block effects; each block effect is "
                "completion_fenced_us_per_forward(identity) - completion_fenced_us_per_forward(candidate)"
            ),
            "layout_purpose": (
                "out_of_sample_model_comparison_on_unseen_captures_and_unseen_layouts_after_lock"
            ),
            "no_refit_before_or_during_evaluation": True,
            "all_starts_rerun_fresh_under_this_protocol": True,
        },
        "timing": {
            "initial_warmup_forwards_per_layout": 10,
            "activation_warmup_forwards": 10,
            "timed_forwards_per_activation": 20,
            "blocks_per_start": 4,
            "diagnostic_blocks_per_start": 4,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "schedule_algorithm": (
                "random.Random(seed + offset + block).shuffle over the frozen layout "
                "order; offset 0 for the primary pass, 1000003 for the diagnostic pass"
            ),
            "primary_metric": "completion_fenced_us_per_forward",
            "diagnostic_metric": "same-device CUDA elapsed interval bracketing real runner.run",
            "diagnostic_substitution_for_primary": False,
        },
        "result_index_relative_path": "results/prediction_v1/evaluation/attempt02/index.json",
        "prohibitions": json.loads(followup.read_text())["prohibitions"],
        "starts": starts,
    }
    write_json_atomic(output, payload)
    print(f"frozen {output}")
    print(f"protocol_sha256={sha256_file(output)}")
    print(f"starts={len(starts)} captures={len(manifests)} attempt=2")


if __name__ == "__main__":
    main()
