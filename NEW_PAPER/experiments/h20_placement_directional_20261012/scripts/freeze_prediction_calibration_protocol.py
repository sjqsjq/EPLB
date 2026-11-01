#!/usr/bin/env python3
"""Freeze the calibration timing protocol and reserved evaluation seeds before any timing."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
from typing import Any

from prediction_common import STUDY_ID, sha256_file, write_json_atomic

PROTOCOL_ID = "h20-placement-prediction-calibration-v1"
GENERATOR_VERSION = "calibration-protocol-v1"
LAYOUT_IDS = (
    "identity",
    "raw-01",
    "raw-02",
    "raw-03",
    "padding-01",
    "padding-02",
    "padding-03",
    "shape-01",
    "shape-02",
    "shape-03",
)
EVALUATION_LAYOUT_IDS = (
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
EVALUATION_SEED_PURPOSES = (
    "raw-01",
    "raw-02",
    "padding-01",
    "padding-02",
    "m2_pool",
    "random_pool",
)
IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "prediction_models_record": "prediction_models.py",
    "prediction_replay_record": "prediction_replay.py",
    "replay_placement_record": "replay_placement.py",
    "fit_prediction_models_record": "fit_prediction_models.py",
    "generate_prediction_calibration_layouts_record": "generate_prediction_calibration_layouts.py",
    "generate_prediction_evaluation_layouts_record": "generate_prediction_evaluation_layouts.py",
    "index_prediction_timing_results_record": "index_prediction_timing_results.py",
    "verify_prediction_timing_protocol_record": "verify_prediction_timing_protocol.py",
    "run_prediction_calibration_record": "run_prediction_calibration.sh",
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
        or root / "configs/followup_v1/prediction_calibration_protocol.json"
    ).resolve()
    tolerance_path = root / "configs/followup_v1/prediction_tolerance.json"
    if not tolerance_path.exists():
        write_json_atomic(
            tolerance_path,
            {
                "schema_version": 1,
                "study_id": STUDY_ID,
                "status": "frozen_before_prediction_timing",
                "atol": 2.0**-8,
                "rtol": 0.0,
                "semantics": (
                    "torch.isclose against both the captured routed output and the "
                    "identity replay; identical to the pilot tolerance; never relaxed"
                ),
                "relative_denominator_floor": 1.0e-6,
            },
        )

    followup = root / "configs/followup_v1/protocol.json"
    capture_index = root / "inputs/prediction_v1/capture_index.json"
    layout_index_path = root / "inputs/prediction_v1/calibration_layouts/index.json"
    layout_index = json.loads(layout_index_path.read_text())
    if layout_index["status"] != "all_calibration_layouts_generated_before_timing":
        raise ValueError("calibration layout index is not in frozen status")
    manifests = {record["capture_id"]: record for record in layout_index["manifests"]}
    if len(manifests) != 6:
        raise ValueError("expected six calibration layout manifests")

    index_payload = json.loads(capture_index.read_text())
    evaluation_captures = sorted(
        unit["capture_id"]
        for unit in index_payload["capture_units"]
        if unit["split"] == "evaluation"
    )
    if len(evaluation_captures) != 12:
        raise ValueError("expected twelve evaluation capture units")

    starts = []
    for capture_index_number, capture_id in enumerate(sorted(manifests), 1):
        manifest_record = manifests[capture_id]
        for start_number in range(1, 5):
            start_id = f"cal-{capture_index_number:02d}-start-{start_number}"
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
                    "primary_schedules": [
                        schedule(seed, block) for block in range(4)
                    ],
                    "diagnostic_schedules": [
                        schedule(seed, block, offset=1_000_003) for block in range(4)
                    ],
                    "output_relative_path": f"results/prediction_v1/calibration/raw/{start_id}.json",
                    "log_relative_path": f"results/prediction_v1/calibration/logs/{start_id}.log",
                    "failure_relative_path": f"results/prediction_v1/calibration/raw/{start_id}.failure.json",
                }
            )

    evaluation_reserved_seeds = {
        capture_id: {
            purpose: derived_seed(capture_id, f"evaluation-reserved-{purpose}")
            for purpose in EVALUATION_SEED_PURPOSES
        }
        for capture_id in evaluation_captures
    }

    implementation = {
        name: file_record(root / "scripts" / filename)
        for name, filename in IMPLEMENTATION_FILES.items()
    }

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "study_id": STUDY_ID,
        "status": "frozen_before_calibration_timing",
        "split": "calibration",
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
        "tolerance_file": file_record(tolerance_path),
        "implementation": implementation,
        "design": {
            "capture_units": 6,
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
            "layout_purpose": "model_identification_not_measured_winner_selection",
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
        "result_index_relative_path": "results/prediction_v1/calibration/index.json",
        "evaluation_reserved_seeds": evaluation_reserved_seeds,
        "evaluation_layout_design": {
            "layout_ids": list(EVALUATION_LAYOUT_IDS),
            "generation_order": (
                "after model lock; identity, two raw-load, two padding, two locked-M2 "
                "top-predicted, two stratified-random layouts per capture"
            ),
            "unseen_requirement": (
                "every nonidentity evaluation map hash must be absent from all "
                "calibration manifests and unique across evaluation captures"
            ),
            "timing_never_determines_candidate_membership": True,
        },
        "prohibitions": json.loads(followup.read_text())["prohibitions"],
        "starts": starts,
    }
    write_json_atomic(output, payload)
    print(f"frozen {output}")
    print(f"protocol_sha256={sha256_file(output)}")
    print(f"starts={len(starts)} reserved_seed_captures={len(evaluation_reserved_seeds)}")


if __name__ == "__main__":
    main()
