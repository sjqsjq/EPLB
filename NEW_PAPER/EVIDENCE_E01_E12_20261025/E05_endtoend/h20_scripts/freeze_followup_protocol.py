#!/usr/bin/env python3
"""Freeze follow-up-v1 hashes and the four-start confirmation schedule."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import random

PILOT_BASE_COMMIT = "8d4a21a77cc6960484f2fed75da8902d920b1f8f"
PROTOCOL_ID = "h20-placement-followup-v1"
START_COUNT = 4
BLOCKS_PER_START = 5


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(start_id: str) -> int:
    material = f"{PROTOCOL_ID}:confirmation:{start_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    output = root / "configs" / "followup_v1" / "protocol.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")

    manifest_path = root / "placements" / "manifest.json"
    tolerance_path = root / "inputs" / "correctness_tolerance.json"
    replay_path = root / "scripts" / "replay_placement.py"
    analyzer_path = root / "scripts" / "analyze_confirmation.py"
    launcher_path = root / "scripts" / "run_confirmation.sh"
    design_path = root / "FOLLOWUP_PROTOCOL.md"
    manifest = json.loads(manifest_path.read_text())
    tolerance = json.loads(tolerance_path.read_text())
    layout_ids = [item["layout_id"] for item in manifest["layouts"]]
    assert layout_ids == [
        "identity",
        "A1",
        "A2",
        "A3",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
        "C3",
        "D1",
        "D2",
        "D3",
    ]
    assert manifest["status"] == "frozen_before_timing"
    assert tolerance["status"] == "frozen"
    assert tolerance["atol"] == 0.00390625 and tolerance["rtol"] == 0.0

    starts = []
    for index in range(1, START_COUNT + 1):
        start_id = f"start-{index:02d}"
        seed = derived_seed(start_id)
        schedules = []
        for block in range(BLOCKS_PER_START):
            order = list(range(len(layout_ids)))
            random.Random(seed + block).shuffle(order)
            schedules.append([f"{layout_ids[item]}:minimal" for item in order])
        starts.append(
            {
                "start_id": start_id,
                "random_seed": seed,
                "schedules": schedules,
                "output_relative_path": f"results/confirmatory_v1/raw/{start_id}.json",
                "log_relative_path": f"results/confirmatory_v1/logs/{start_id}.log",
            }
        )
    assert len({item["random_seed"] for item in starts}) == START_COUNT

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_confirmation_timing",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_base_commit": PILOT_BASE_COMMIT,
        "design_document": {
            "path": str(design_path),
            "sha256": sha256_file(design_path),
        },
        "scope": {
            "node_count": 1,
            "accelerator_count": 8,
            "accelerator_model": "NVIDIA H20",
            "expert_parallel_size": 8,
            "deep_ep_mode": "normal",
            "moe_runner": "DeepGEMM FP8",
            "model": "Qwen3-235B-A22B-FP8",
            "geometry": {
                "hidden_size": 4096,
                "intermediate_size": 1536,
                "routed_experts": 128,
                "local_experts_per_rank": 16,
                "top_k": 8,
                "weight_block": [128, 128],
            },
        },
        "implementation": {
            "replay_placement_path": str(replay_path),
            "replay_placement_sha256": sha256_file(replay_path),
            "analyze_confirmation_path": str(analyzer_path),
            "analyze_confirmation_sha256": sha256_file(analyzer_path),
            "run_confirmation_path": str(launcher_path),
            "run_confirmation_sha256": sha256_file(launcher_path),
        },
        "confirmation": {
            "capture_id": manifest["capture_id"],
            "layer_id": 24,
            "placement_manifest_path": str(manifest_path),
            "placement_manifest_sha256": sha256_file(manifest_path),
            "tolerance_file_path": str(tolerance_path),
            "tolerance_file_sha256": sha256_file(tolerance_path),
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
            "snapshot_files": manifest["snapshot_files"],
            "layout_ids": layout_ids,
            "independent_process_starts": START_COUNT,
            "blocks_per_start": BLOCKS_PER_START,
            "timed_forwards_per_condition": 50,
            "initial_warmup_forwards_per_layout": 10,
            "activation_warmup_forwards_per_condition": 10,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "randomization": {
                "algorithm": "Python random.Random(seed + block).shuffle over manifest layout order",
                "seed_derivation": (
                    "first eight bytes of SHA-256("
                    "'h20-placement-followup-v1:confirmation:<start-id>'), "
                    "big-endian, modulo 2^31-1"
                ),
                "frozen_before_timing": True,
            },
            "starts": starts,
            "primary_estimand": "Q = completion(identity) - completion(layout)",
            "inferential_unit": "independent process start",
            "start_reduction": "mean of five block-paired effects",
            "primary_interval": {
                "method": "two-sided Student-t",
                "n": 4,
                "degrees_of_freedom": 3,
                "confidence_level": 0.95,
                "critical_value": 3.182446305284263,
            },
            "multiplicity_secondary": (
                "Holm-adjusted two-sided t-test p-values across all 12 identity contrasts"
            ),
            "practical_equivalence_margin_us": 22.948043518066406,
            "equivalence_rule": (
                "interval wholly inside [-epsilon,+epsilon]; zero crossing alone is not equivalence"
            ),
            "best_repeat_selection": False,
            "pilot_included_in_primary_inference": False,
        },
        "prediction_study": {
            "status": "design_rule_frozen_exact_capture_and_layout_manifests_pending",
            "request_batches": {
                "screening": 1,
                "calibration": 3,
                "evaluation": 3,
                "requests_per_batch": 8,
                "must_be_disjoint_from_e1_and_between_splits": True,
            },
            "layers": {
                "count": 4,
                "include_layer_24": True,
                "calibration_layers": 2,
                "held_out_layers": 2,
                "selection_uses_screening_route_metadata_only": True,
            },
            "capture_units": {"calibration": 6, "evaluation": 12},
            "models": ["M0_raw", "M1_padding_128", "M2_grouped_gemm"],
            "common_communication_covariates": True,
            "calibration_evaluation_timing_overlap": False,
            "starts": 4,
            "blocks_per_start": 4,
            "activation_warmup": 10,
            "timed_forwards": 20,
        },
        "stopping_study": {
            "status": "design_rule_frozen_exact_residual_manifest_pending_locked_model",
            "base": "deterministic low-r lexicographic greedy legal-swap local optimum",
            "candidate_count_per_capture": 12,
            "candidate_strata": [
                "four largest locked-M2 predicted gains",
                "four locked-M2 predictions closest to zero",
                "four most negative locked-M2 predictions",
            ],
            "starts": 4,
            "blocks_per_start": 5,
            "activation_warmup": 10,
            "timed_forwards": 50,
            "practical_equivalence_margin_us": 22.948043518066406,
            "migration": {
                "separate_from_execution_timing": True,
                "complete_w13_w2_and_both_scale_tensors": True,
                "starts": 4,
                "forward_reverse_pairs_per_start": 50,
                "old_bench_p2p_cost_is_inadmissible": True,
            },
        },
        "interpretation_limits": [
            "Confirmation remains one e089-centered case on one layer and forward.",
            "Padding-aware superiority requires locked out-of-sample model loss, not E1 correlation.",
            "Fixed-aggregate-load structural timing remains unresolved unless the new interval resolves it.",
            "A stopping region requires directly measured low-r residual swaps and an equivalence interval.",
            "Completion-fenced timing includes the end fence; observer effects remain explicit.",
        ],
        "prohibitions": {
            "artificial_delay_floor": True,
            "differential_delay": True,
            "sleep_or_spin": True,
            "mid_layer_collective_or_gate": True,
            "cross_device_timestamp_subtraction": True,
            "best_start_or_repeat_selection": True,
            "overwrite_existing_raw_result": True,
            "push_without_separate_instruction": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)
    print(sha256_file(output))


if __name__ == "__main__":
    main()
