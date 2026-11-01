#!/usr/bin/env python3
"""Replay one split prediction capture under a frozen calibration/evaluation schedule."""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import pathlib
import socket
from typing import Any

# These must be fixed before importing SGLang through replay_placement.
os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "512")
os.environ.setdefault("SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "1")

import torch
import torch.distributed as dist

import replay_placement as core
from prediction_common import (
    ALIGNMENT,
    LOCAL_EXPERTS,
    STUDY_ID,
    WEIGHT_FIELDS,
    WORLD_SIZE,
    IndexedCaptureUnit,
    indexed_expert_records,
    indexed_units,
    layout_descriptors,
    load_capture_index,
    load_tensor_payload,
    route_metadata_from_payloads,
    sha256_file,
    validate_expert_payload,
    validate_forward_payload,
    verify_file_record,
)

ATOL = 2.0**-8
RTOL = 0.0
EXPECTED_LAYOUT_IDS_CALIBRATION = (
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


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--start-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = args.protocol.resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported prediction timing protocol schema")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("prediction timing protocol belongs to another study")
    if payload.get("status") not in {
        "frozen_before_calibration_timing",
        "frozen_before_evaluation_timing",
    }:
        raise ValueError("prediction timing protocol is not frozen before timing")
    implementation = payload["implementation"]
    checks = {
        pathlib.Path(__file__).resolve(): implementation["prediction_replay_record"]["sha256"],
        pathlib.Path(__file__).with_name("prediction_common.py").resolve(): implementation[
            "prediction_common_record"
        ]["sha256"],
        pathlib.Path(__file__).with_name("replay_placement.py").resolve(): implementation[
            "replay_placement_record"
        ]["sha256"],
    }
    for implementation_path, expected in checks.items():
        actual = sha256_file(implementation_path)
        if actual != expected:
            raise ValueError(
                f"frozen implementation hash differs for {implementation_path}: "
                f"{actual} != {expected}"
            )
    starts = {record["start_id"]: record for record in payload["starts"]}
    if args.start_id not in starts:
        raise ValueError(f"start ID is absent from protocol: {args.start_id}")
    start = starts[args.start_id]
    expected_output = (args.root / start["output_relative_path"]).resolve()
    if args.output.resolve() != expected_output:
        raise ValueError(
            f"output path differs from frozen start: {args.output.resolve()} != {expected_output}"
        )
    return payload, start


def find_unit(index: dict[str, Any], capture_id: str) -> IndexedCaptureUnit:
    matches = [unit for unit in indexed_units(index) if unit.capture_id == capture_id]
    if len(matches) != 1:
        raise ValueError(f"capture index has {len(matches)} matches for {capture_id}")
    return matches[0]


def load_layouts(
    *,
    path: pathlib.Path,
    expected_sha256: str,
    unit: IndexedCaptureUnit,
    metadata,
    split: str,
) -> tuple[dict[str, Any], list[core.Layout]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"layout manifest hash differs: {path}")
    payload = json.loads(path.read_text())
    expected_header = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "split": split,
        "capture_id": unit.capture_id,
        "batch_id": unit.batch_id,
        "layer_id": unit.layer_id,
        "selection_uses_performance_timing": False,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_header.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"layout manifest header mismatch: {mismatches}")
    expected_route = {
        "hidden_rows_by_source_rank": list(metadata.hidden_rows_by_source_rank),
        "total_hidden_rows": metadata.total_hidden_rows,
        "total_assignments": metadata.total_assignments,
        "source_expert_counts": [list(row) for row in metadata.source_expert_counts],
        "expert_counts": list(metadata.expert_counts),
        "expert_blocks_128": list(metadata.expert_blocks),
        "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
    }
    if payload.get("route_metadata") != expected_route:
        raise ValueError("layout manifest route metadata differs from split capture")
    layouts = []
    hashes = set()
    for record in payload["layouts"]:
        p2l = tuple(int(value) for value in record["p2l"])
        l2p = tuple(int(value) for value in record["l2p"])
        if core.Layout(
            layout_id=record["layout_id"],
            category=record["category"],
            swap=None,
            p2l=p2l,
            l2p=l2p,
            metrics=record["descriptors"],
        ).p2l != p2l:
            raise AssertionError("unreachable layout construction failure")
        if sorted(p2l) != list(range(core.NUM_EXPERTS)):
            raise ValueError(f"layout {record['layout_id']} is not bijective")
        if any(l2p[logical] != physical for physical, logical in enumerate(p2l)):
            raise ValueError(f"layout {record['layout_id']} inverse differs")
        recomputed = layout_descriptors(metadata, p2l)
        if recomputed != record["descriptors"]:
            raise ValueError(f"layout descriptors differ for {record['layout_id']}")
        if record["map_sha256"] in hashes:
            raise ValueError("layout manifest contains duplicate maps")
        hashes.add(record["map_sha256"])
        layouts.append(
            core.Layout(
                layout_id=record["layout_id"],
                category=record["category"],
                swap=None,
                p2l=p2l,
                l2p=l2p,
                metrics=record["descriptors"],
            )
        )
    if split == "calibration" and tuple(layout.layout_id for layout in layouts) != EXPECTED_LAYOUT_IDS_CALIBRATION:
        raise ValueError("calibration layout IDs/order differ from frozen design")
    if sum(layout.layout_id == "identity" for layout in layouts) != 1:
        raise ValueError("layout manifest must contain exactly one identity")
    return payload, layouts


def build_split_replay_state(
    *,
    local_forward: dict[str, Any],
    local_expert: dict[str, Any],
    layouts: list[core.Layout],
    metadata,
    expert_records: dict[tuple[int, int], dict[str, Any]],
    identity_map_sha256: str,
    rank: int,
    device: torch.device,
) -> core.ReplayState:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    hidden_states = local_forward["hidden_states"].to(device)
    logical_topk_ids = local_forward["logical_topk_ids"].to(device)
    logical_topk_weights = local_forward["logical_topk_weights"].to(device)
    captured_output = local_forward["routed_output"].to(device)
    frozen_hashes = {
        "hidden_states": core.tensor_sha256(local_forward["hidden_states"]),
        "logical_topk_ids": core.tensor_sha256(local_forward["logical_topk_ids"]),
        "logical_topk_weights": core.tensor_sha256(local_forward["logical_topk_weights"]),
        "captured_output": core.tensor_sha256(local_forward["routed_output"]),
    }

    topk_by_layout = {}
    for layout in layouts:
        l2p = torch.tensor(layout.l2p, dtype=torch.int32, device=device)
        physical_ids = l2p[logical_topk_ids.to(torch.int64)].contiguous()
        topk_by_layout[layout.layout_id] = StandardTopKOutput(
            logical_topk_weights, physical_ids, None
        )

    base_tensors = {field: local_expert[field].to(device) for field in WEIGHT_FIELDS}
    local_start = rank * LOCAL_EXPERTS
    base_key = tuple(range(local_start, local_start + LOCAL_EXPERTS))
    tensors_by_local_map = {base_key: base_tensors}
    quant_by_local_map = {base_key: core.make_quant_info(base_tensors)}
    expert_cache: dict[int, dict[str, Any]] = {rank: local_expert}

    def source_payload(source_rank: int) -> dict[str, Any]:
        if source_rank not in expert_cache:
            record = expert_records[(metadata.layer_id, source_rank)]
            path = verify_file_record(record)
            payload = load_tensor_payload(path)
            validate_expert_payload(
                payload,
                layer_id=metadata.layer_id,
                rank=source_rank,
                path=path,
                identity_map_sha256=identity_map_sha256,
            )
            expert_cache[source_rank] = payload
        return expert_cache[source_rank]

    quant_by_layout = {}
    for layout in layouts:
        local_map = tuple(layout.p2l[local_start : local_start + LOCAL_EXPERTS])
        if local_map not in quant_by_local_map:
            tensors = {field: value.clone() for field, value in base_tensors.items()}
            for destination_slot, logical_expert in enumerate(local_map):
                physical_expert = local_start + destination_slot
                if logical_expert == physical_expert:
                    continue
                source_rank, source_slot = divmod(logical_expert, LOCAL_EXPERTS)
                source = source_payload(source_rank)
                for field in WEIGHT_FIELDS:
                    tensors[field][destination_slot].copy_(
                        source[field][source_slot].to(device)
                    )
            tensors_by_local_map[local_map] = tensors
            quant_by_local_map[local_map] = core.make_quant_info(tensors)
        quant_by_layout[layout.layout_id] = quant_by_local_map[local_map]

    aligned_counts = [block * ALIGNMENT for block in metadata.expert_blocks]
    expected_aligned_by_layout = {
        layout.layout_id: [
            aligned_counts[layout.p2l[physical]]
            for physical in range(local_start, local_start + LOCAL_EXPERTS)
        ]
        for layout in layouts
    }
    torch.cuda.synchronize(device)
    return core.ReplayState(
        hidden_states=hidden_states,
        logical_topk_ids=logical_topk_ids,
        logical_topk_weights=logical_topk_weights,
        captured_output=captured_output,
        topk_by_layout=topk_by_layout,
        quant_by_layout=quant_by_layout,
        expected_aligned_by_layout=expected_aligned_by_layout,
        frozen_hashes=frozen_hashes,
        weight_storage_count=len(tensors_by_local_map),
    )


def run_schedule(
    *,
    pass_name: str,
    schedules: list[list[str]],
    layout_by_id: dict[str, core.Layout],
    state: core.ReplayState,
    dispatcher,
    runner,
    start_marker: torch.Tensor,
    end_marker: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
    activation_warmup: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
    observation_mode = "minimal" if pass_name == "primary" else "detailed"
    records: list[dict[str, Any]] = []
    local_warmups: list[dict[str, Any]] = []
    for block, schedule in enumerate(schedules):
        if set(schedule) != set(layout_by_id) or len(schedule) != len(layout_by_id):
            raise ValueError(f"{pass_name} block {block} is not one permutation of layouts")
        for condition_order, layout_id in enumerate(schedule):
            layout = layout_by_id[layout_id]
            local_warmups.append(
                core.warm_activation(
                    layout=layout,
                    state=state,
                    dispatcher=dispatcher,
                    runner=runner,
                    marker=start_marker,
                    group=group,
                    warmup=activation_warmup,
                    rank=rank,
                    block=block,
                    condition_order=condition_order,
                )
            )
            record = core.timed_block(
                layout=layout,
                observation_mode=observation_mode,
                iterations=iterations,
                state=state,
                dispatcher=dispatcher,
                runner=runner,
                start_marker=start_marker,
                end_marker=end_marker,
                group=group,
                rank=rank,
            )
            if rank == 0:
                assert record is not None
                record.update(
                    {
                        "pass": pass_name,
                        "block": block,
                        "condition_order": condition_order,
                    }
                )
                if pass_name == "diagnostic":
                    for rank_record in record["ranks"]:
                        rank_record["runner_run_iteration_us"] = rank_record[
                            "detailed_iteration_us"
                        ]["expert_gather_us"]
                        rank_record["runner_run_summary_us"] = rank_record[
                            "detailed_summary_us"
                        ]["expert_gather_us"]
                records.append(record)
    gathered_warmups: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_warmups, local_warmups, group=group)
    return records, gathered_warmups, schedules


def source_audit() -> dict[str, Any]:
    base = core.source_audit()
    run_body = inspect.getsource(core.run_once)
    if "runner.run" not in run_body or "_combine_core" not in run_body:
        raise ValueError("reused real-forward operator path is incomplete")
    if run_body.index("runner.run") > run_body.index("_combine_core"):
        raise ValueError("reused expert/combine order changed")
    return {
        **base,
        "reuses_replay_placement_run_once": True,
        "runner_before_combine": True,
        "mid_layer_collective_or_gate": False,
        "artificial_delay_floor": False,
    }


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    protocol, start = load_protocol(args)
    output_path = args.output.resolve()
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite start output: {output_path}")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise ValueError(f"prediction replay requires eight ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(list(range(world_size)), backend="nccl")
    torch.set_grad_enabled(False)
    torch.manual_seed(int(start["random_seed"]) + rank)

    properties = torch.cuda.get_device_properties(device)
    local_gpu = {
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    gpu_inventory: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gpu_inventory, local_gpu, group=group)
    if any(record["name"] != "NVIDIA H20" for record in gpu_inventory):
        raise ValueError("prediction replay requires exactly eight NVIDIA H20 GPUs")

    core.configure_sglang_globals()
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPConfig

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SGLang JIT DeepGEMM is not enabled")

    capture_index_record = protocol["capture_index"]
    capture_index_path = pathlib.Path(capture_index_record["path"]).resolve()
    if sha256_file(capture_index_path) != capture_index_record["sha256"]:
        raise ValueError("capture-index hash differs from timing protocol")
    index = load_capture_index(capture_index_path)
    unit = find_unit(index, start["capture_id"])
    if unit.split != protocol["split"] or unit.batch_id != start["batch_id"] or unit.layer_id != start["layer_id"]:
        raise ValueError("start/capture metadata mismatch")
    expert_records = indexed_expert_records(index)

    local_forward_path = verify_file_record(unit.forward_records[rank])
    local_forward = load_tensor_payload(local_forward_path)
    forward_meta = validate_forward_payload(
        local_forward,
        unit=unit,
        rank=rank,
        path=local_forward_path,
        allowed_splits={protocol["split"]},
    )
    local_expert_record = expert_records[(unit.layer_id, rank)]
    local_expert_path = verify_file_record(local_expert_record)
    local_expert = load_tensor_payload(local_expert_path)
    expert_meta = validate_expert_payload(
        local_expert,
        layer_id=unit.layer_id,
        rank=rank,
        path=local_expert_path,
        identity_map_sha256=forward_meta["identity_map_sha256"],
    )
    if forward_meta["expert_artifact_path"] != str(local_expert_path):
        raise ValueError("forward snapshot references a different expert artifact")
    local_integrity = {
        "rank": rank,
        "forward": {**forward_meta, **unit.forward_records[rank]},
        "expert": {**expert_meta, **local_expert_record},
    }
    gathered_integrity: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_integrity, local_integrity, group=group)
    identity_hashes = {record["forward"]["identity_map_sha256"] for record in gathered_integrity}
    if len(identity_hashes) != 1:
        raise ValueError("identity-map hash differs across ranks")
    identity_map_sha256 = identity_hashes.pop()

    # Route metadata is rebuilt collectively from the eight real forward artifacts.
    local_route_ids = local_forward["logical_topk_ids"]
    gathered_route_ids: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_route_ids, local_route_ids, group=group)
    route_payloads = [{"logical_topk_ids": value} for value in gathered_route_ids]
    metadata = route_metadata_from_payloads(unit, route_payloads)

    layout_record = start["layout_manifest"]
    layout_path = pathlib.Path(layout_record["path"]).resolve()
    layout_payload, layouts = load_layouts(
        path=layout_path,
        expected_sha256=layout_record["sha256"],
        unit=unit,
        metadata=metadata,
        split=protocol["split"],
    )
    layout_by_id = {layout.layout_id: layout for layout in layouts}

    state = build_split_replay_state(
        local_forward=local_forward,
        local_expert=local_expert,
        layouts=layouts,
        metadata=metadata,
        expert_records=expert_records,
        identity_map_sha256=identity_map_sha256,
        rank=rank,
        device=device,
    )
    dispatcher, runner = core.build_layer(group, unit.layer_id)
    start_marker = torch.ones(1, dtype=torch.float32, device=device)
    end_marker = torch.ones(1, dtype=torch.float32, device=device)

    started_at = dt.datetime.now(dt.timezone.utc)
    initial_warmup = int(protocol["timing"]["initial_warmup_forwards_per_layout"])
    warmup_records = core.warm_layouts(
        layouts=layouts,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        marker=start_marker,
        group=group,
        warmup=initial_warmup,
        rank=rank,
    )
    gathered_warmup: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_warmup, warmup_records, group=group)

    correctness_records = core.run_correctness(
        layouts=layouts,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        marker=start_marker,
        group=group,
        rank=rank,
        atol=ATOL,
        rtol=RTOL,
    )
    core.validate_correctness_gate(correctness_records, rank=rank, group=group)

    empty_fence = core.calibrate_empty_fence(
        warmup=int(protocol["timing"]["empty_fence_warmup"]),
        repetitions=int(protocol["timing"]["empty_fence_repetitions"]),
        start_marker=start_marker,
        end_marker=end_marker,
        group=group,
        rank=rank,
    )
    activation_warmup = int(protocol["timing"]["activation_warmup_forwards"])
    iterations = int(protocol["timing"]["timed_forwards_per_activation"])
    primary_records, primary_warmups, primary_schedules = run_schedule(
        pass_name="primary",
        schedules=start["primary_schedules"],
        layout_by_id=layout_by_id,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        start_marker=start_marker,
        end_marker=end_marker,
        group=group,
        rank=rank,
        activation_warmup=activation_warmup,
        iterations=iterations,
    )
    diagnostic_records, diagnostic_warmups, diagnostic_schedules = run_schedule(
        pass_name="diagnostic",
        schedules=start["diagnostic_schedules"],
        layout_by_id=layout_by_id,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        start_marker=start_marker,
        end_marker=end_marker,
        group=group,
        rank=rank,
        activation_warmup=activation_warmup,
        iterations=iterations,
    )

    final_hashes = {
        "hidden_states": core.tensor_sha256(state.hidden_states),
        "logical_topk_ids": core.tensor_sha256(state.logical_topk_ids),
        "logical_topk_weights": core.tensor_sha256(state.logical_topk_weights),
        "captured_output": core.tensor_sha256(state.captured_output),
    }
    if final_hashes != state.frozen_hashes:
        raise ValueError("frozen real-forward tensors changed during replay")
    gathered_hashes: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_hashes,
        {
            "rank": rank,
            "initial": state.frozen_hashes,
            "final": final_hashes,
            "local_weight_storage_count": state.weight_storage_count,
        },
        group=group,
    )
    finished_at = dt.datetime.now(dt.timezone.utc)

    if rank == 0:
        if len(primary_records) != len(layouts) * len(start["primary_schedules"]):
            raise ValueError("primary timing record cardinality differs")
        if len(diagnostic_records) != len(layouts) * len(start["diagnostic_schedules"]):
            raise ValueError("diagnostic timing record cardinality differs")
        deep_ep_config = DeepEPConfig.get_instance()
        result = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "experiment": "h20_placement_out_of_sample_prediction",
            "split": protocol["split"],
            "start_id": start["start_id"],
            "capture_id": unit.capture_id,
            "batch_id": unit.batch_id,
            "layer_id": unit.layer_id,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "hostname": socket.gethostname(),
            "gpu_inventory": gpu_inventory,
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol.resolve()),
                "status": protocol["status"],
            },
            "capture_index": capture_index_record,
            "layout_manifest": {
                **layout_record,
                "status": layout_payload["status"],
                "layout_ids": [layout.layout_id for layout in layouts],
                "map_sha256s": [record["map_sha256"] for record in layout_payload["layouts"]],
            },
            "input_integrity": gathered_integrity,
            "route_metadata": layout_payload["route_metadata"],
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "source_audit": source_audit(),
            "operator_order": [
                "dispatcher.dispatch_a",
                "dispatcher.dispatch_b",
                "runner.run",
                "Buffer.capture",
                "dispatcher._combine_core",
                "completion_event.current_stream_wait",
            ],
            "timing_semantics": {
                "primary": "GPU-0 same-clock completion-fenced block after untimed NCCL start alignment; no fence subtraction",
                "diagnostic": "separate pass; same-device CUDA events around real runner.run; not substituted for primary",
                "cross_device_cuda_timestamp_subtraction": False,
                "mid_layer_collective_or_gate": False,
                "artificial_delay_or_sleep": False,
            },
            "timing_config": protocol["timing"],
            "primary_schedules": primary_schedules,
            "diagnostic_schedules": diagnostic_schedules,
            "initial_warmup": gathered_warmup,
            "primary_activation_warmup": primary_warmups,
            "diagnostic_activation_warmup": diagnostic_warmups,
            "correctness": {"records": correctness_records},
            "empty_fence_diagnostic": empty_fence,
            "primary_timing_records": primary_records,
            "diagnostic_timing_records": diagnostic_records,
            "frozen_tensor_hashes": gathered_hashes,
            "execution": {
                "dispatcher": "SGLang _DeepEPDispatcherImplNormal(async_finish=True)",
                "runner": "SGLang DeepGemmRunnerCore contiguous FP8 grouped GEMM",
                "combine": (
                    "DeepEP Buffer.combine with Buffer.capture previous_event and "
                    "completion_event.current_stream_wait"
                ),
                "deep_ep_mode": "normal",
                "deep_ep_num_sms": deep_ep_config.num_sms,
                "cuda_graph": False,
            },
        }
        core.write_json_atomic(output_path, result)
        print(f"wrote {output_path}")

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
