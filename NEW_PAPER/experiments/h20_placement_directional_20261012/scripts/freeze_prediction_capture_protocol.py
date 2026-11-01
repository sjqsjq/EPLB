#!/usr/bin/env python3
"""Freeze exact source, inputs, paths, and rules before prediction profiling."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
from typing import Any


STUDY_ID = "h20-placement-prediction-v1"
IMPLEMENTATION_FILES = {
    "freeze_prediction_requests": "scripts/freeze_prediction_requests.py",
    "capture_patch": (
        "scripts/prediction_capture_site/placement_prediction_capture_patch.py"
    ),
    "capture_sitecustomize": "scripts/prediction_capture_site/sitecustomize.py",
    "make_capture_trigger": "scripts/make_prediction_capture_trigger.py",
    "select_layers": "scripts/select_prediction_layers.py",
    "launch_capture_server": "scripts/launch_prediction_capture_server.sh",
    "send_prediction_batch": "scripts/send_prediction_batch.py",
    "index_prediction_captures": "scripts/index_prediction_captures.py",
    "verify_capture_protocol": "scripts/verify_prediction_capture_protocol.py",
    "run_prediction_capture": "scripts/run_prediction_capture.sh",
    "freeze_capture_protocol": "scripts/freeze_prediction_capture_protocol.py",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json_no_clobber(path: pathlib.Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale protocol temporary exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    output = root / "configs/followup_v1/prediction_capture_protocol.json"
    request_manifest_path = root / "configs/followup_v1/prediction_requests.json"
    followup_protocol_path = root / "configs/followup_v1/protocol.json"
    identity_map_path = root / "placements/identity_service_map.json"
    environment_path = pathlib.Path("/workspace/logs/env_235b.sh")
    request_manifest = json.loads(request_manifest_path.read_text())
    followup_protocol = json.loads(followup_protocol_path.read_text())
    if request_manifest.get("study_id") != STUDY_ID:
        raise ValueError("request manifest has the wrong study ID")
    if request_manifest.get("status") != "frozen_before_screening_route_profile":
        raise ValueError("request manifest is not frozen before screening")
    if followup_protocol.get("protocol_id") != "h20-placement-followup-v1":
        raise ValueError("follow-up protocol has the wrong ID")
    if followup_protocol["prediction_study"]["capture_units"] != {
        "calibration": 6,
        "evaluation": 12,
    }:
        raise ValueError("follow-up protocol capture-unit counts changed")

    batches = request_manifest["batches"]
    if [item["split"] for item in batches] != [
        "screening",
        "calibration",
        "calibration",
        "calibration",
        "evaluation",
        "evaluation",
        "evaluation",
    ]:
        raise ValueError("request split order differs from prediction v1")
    if any(item["request_count"] != 8 for item in batches):
        raise ValueError("every frozen batch must contain exactly eight requests")

    implementation = {
        name: file_record(root / relative)
        for name, relative in IMPLEMENTATION_FILES.items()
    }
    model_files = list(request_manifest["model"]["files"].values())
    request_files = [batch["request_file"] for batch in batches]
    capture_root = root / "inputs/prediction_v1/captures"
    weight_root = root / "inputs/prediction_v1/expert_weights"
    trigger_root = root / "inputs/prediction_v1/triggers"
    layer_root = root / "inputs/prediction_v1/layers"
    response_root = root / "results/prediction_v1/capture_responses"
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "frozen_before_screening_route_profile",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "followup_protocol": file_record(followup_protocol_path),
        "request_manifest": file_record(request_manifest_path),
        "request_files": request_files,
        "identity_map": file_record(identity_map_path),
        "runtime_environment": file_record(environment_path),
        "model_path": request_manifest["model"]["path"],
        "model_files": model_files,
        "implementation": implementation,
        "scope": {
            "node_count": 1,
            "accelerator_count": 8,
            "accelerator_model": "NVIDIA H20",
            "ep": 8,
            "deep_ep_mode": "normal",
            "moe_runner_backend": "deep_gemm",
            "model": "Qwen3-235B-A22B-FP8",
            "dtype": "bfloat16",
            "quantization": "fp8",
            "routed_layers": 94,
            "routed_experts": 128,
            "local_experts_per_rank": 16,
            "top_k": 8,
            "hidden_size": 4096,
            "intermediate_size": 1536,
            "weight_block_size": [128, 128],
            "grouped_gemm_alignment_rows_per_expert": 128,
        },
        "server": {
            "launch_arguments": [
                "--tp", "8", "--dp", "8", "--ep-size", "8",
                "--enable-dp-attention",
                "--moe-a2a-backend", "deepep",
                "--deepep-mode", "normal",
                "--moe-runner-backend", "deep_gemm",
                "--dtype", "bfloat16",
                "--quantization", "fp8",
                "--mem-fraction-static", "0.8",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--watchdog-timeout", "600",
                "--init-expert-location", str(identity_map_path.resolve()),
            ],
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
                "SGLANG_CI_DISABLE_MOE_FUSED_FUNC": "1",
            },
            "identity_mapping_required": True,
            "profile_and_snapshot_are_not_performance_timing": True,
        },
        "screening_profile": {
            "capture_id": "pred-v1-screening-01-profile",
            "batch_id": "screening-01",
            "output_dir": str((capture_root / "screening-01/profile").resolve()),
            "response_path": str(
                (response_root / "pred-v1-screening-01-profile.json").resolve()
            ),
            "trigger_archive_path": str(
                (trigger_root / "pred-v1-screening-01-profile.json").resolve()
            ),
            "rank_files": 8,
            "expected_layers_per_rank": 94,
            "selection_uses_performance_timing": False,
        },
        "layer_selection": {
            "bridge_layer": 24,
            "depth_thirds": [
                {"name": "early", "begin_inclusive": 0, "end_exclusive": 31},
                {"name": "middle", "begin_inclusive": 31, "end_exclusive": 62},
                {"name": "late", "begin_inclusive": 62, "end_exclusive": 94},
            ],
            "depth_winner_order": [
                "descending screening identity-placement raw max/mean rank assignment imbalance",
                "ascending layer ID",
            ],
            "bridge_duplicate_rule": (
                "if layer 24 is the first early-third winner, take the next-ranked "
                "early-third layer as that third's winner"
            ),
            "calibration_assignment": (
                "layer 24 plus the depth-third winner with largest screening raw "
                "max/mean imbalance, ties by lower layer ID"
            ),
            "held_out_assignment": (
                "the remaining two depth-third winners are completely held out"
            ),
            "selected_layer_count": 4,
            "calibration_layer_count": 2,
            "held_out_layer_count": 2,
            "profile_summary_path": str(
                (layer_root / "screening_profile_summary.json").resolve()
            ),
            "layer_manifest_path": str(
                (layer_root / "selected_layers.json").resolve()
            ),
        },
        "snapshot_matrix": {
            "calibration": {
                "batch_ids": [
                    item["batch_id"] for item in batches
                    if item["split"] == "calibration"
                ],
                "layers": "both frozen calibration layers",
                "units": 6,
            },
            "evaluation": {
                "batch_ids": [
                    item["batch_id"] for item in batches
                    if item["split"] == "evaluation"
                ],
                "layers": "all four frozen selected layers",
                "units": 12,
            },
            "capture_id_template": "pred-v1-{batch_id}-layer-{layer_id:03d}",
            "forward_output_template": str(
                (capture_root / "{batch_id}/layer-{layer_id:03d}/forward_rank{rank}.pt").resolve()
            ),
            "expert_output_template": str(
                (weight_root / "layer-{layer_id:03d}/expert_rank{rank}.pt").resolve()
            ),
            "trigger_archive_template": str(
                (trigger_root / "pred-v1-{batch_id}-layer-{layer_id:03d}.json").resolve()
            ),
            "response_template": str(
                (response_root / "pred-v1-{batch_id}-layer-{layer_id:03d}.json").resolve()
            ),
            "forward_rank_files_per_unit": 8,
            "expert_rank_files_per_selected_layer": 8,
            "storage": (
                "each real forward stores hidden states, logical top-k IDs and "
                "weights, and routed output; each selected layer stores immutable "
                "w13, w2, and both FP8 scale tensors once per rank"
            ),
        },
        "expected_artifact_counts": {
            "profile_rank_files": 8,
            "snapshot_units": 18,
            "forward_rank_files": 144,
            "expert_rank_files": 32,
            "trigger_archives": 19,
            "response_files": 19,
        },
        "current_trigger_path": str(
            (root / "inputs/prediction_v1/capture_trigger.json").resolve()
        ),
        "capture_index_path": str(
            (root / "inputs/prediction_v1/capture_index.json").resolve()
        ),
        "model_lock_boundary": {
            "evaluation_snapshots_may_be_captured_and_mechanically_validated": True,
            "evaluation_routes_outputs_or_timing_may_not_be_used_for_fitting": True,
            "M0_M1_M2_code_features_coefficients_transformations_and_hyperparameters_must_lock_before_evaluation_timing": True,
            "evaluation_layouts_must_be_generated_only_from_reserved_seeds_and_locked_models": True,
        },
        "invariants": {
            "real_requests_and_real_routed_layer_forwards_only": True,
            "logical_routes_and_weights_frozen_within_each_replay": True,
            "expert_weights_and_both_scale_tensors_frozen_within_each_replay": True,
            "source_ranks_frozen_within_each_replay": True,
            "only_logical_expert_to_physical_slot_bijection_changes": True,
            "zero_expert_padding_rule": "0 if n_e == 0 else 128*ceil(n_e/128)",
            "common_20us_delay_floor": False,
            "differential_delay": False,
            "sleep_or_spin": False,
            "collective_or_gate_between_expert_execution_and_combine": False,
            "capture_artifacts_may_not_be_overwritten": True,
            "best_repeat_selection": False,
            "push_or_external_publication_authorized": False,
        },
    }
    write_json_no_clobber(output, payload)
    print(output)
    print(sha256(output))


if __name__ == "__main__":
    main()
