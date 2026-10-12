#!/usr/bin/env python3
"""Replay one captured H20 MoE layer under frozen physical expert layouts."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import pathlib
import random
import socket
import statistics
from typing import Any

# These match the capture server and must precede SGLang imports.
os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "512")
os.environ.setdefault("SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "1")

import torch
import torch.distributed as dist

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
NUM_EXPERTS = 128
TOP_K = 8
WEIGHT_BLOCK = 128
WORLD_SIZE = 8
LOCAL_EXPERTS = NUM_EXPERTS // WORLD_SIZE
RELATIVE_DENOMINATOR_FLOOR = 1.0e-6
DETAILED_EVENT_NAMES = (
    "start",
    "dispatch_done",
    "expert_gather_done",
    "combine_input_ready",
    "combine_done",
)
WEIGHT_FIELDS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


@dataclasses.dataclass(frozen=True)
class Layout:
    layout_id: str
    category: str
    swap: dict[str, Any] | None
    p2l: tuple[int, ...]
    l2p: tuple[int, ...]
    metrics: dict[str, Any]


@dataclasses.dataclass
class ReplayState:
    hidden_states: torch.Tensor
    logical_topk_ids: torch.Tensor
    logical_topk_weights: torch.Tensor
    captured_output: torch.Tensor
    topk_by_layout: dict[str, Any]
    quant_by_layout: dict[str, Any]
    expected_aligned_by_layout: dict[str, list[int]]
    frozen_hashes: dict[str, str]
    weight_storage_count: int


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("correctness", "e0", "e1"), required=True)
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--snapshot-dir", type=pathlib.Path)
    parser.add_argument("--tolerance-file", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--layouts", help="comma-separated layout IDs; default is phase protocol")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations-per-block", type=int, default=50)
    parser.add_argument("--empty-fence-warmup", type=int, default=5)
    parser.add_argument("--empty-fence-repetitions", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20261012)
    parser.add_argument("--e0-swap", default="A1")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def configure_sglang_globals() -> None:
    from sglang.srt.layers.moe import utils as moe_utils

    moe_utils.MOE_A2A_BACKEND = moe_utils.MoeA2ABackend.DEEPEP
    moe_utils.MOE_RUNNER_BACKEND = moe_utils.MoeRunnerBackend.DEEP_GEMM
    moe_utils.DEEPEP_MODE = moe_utils.DeepEPMode.NORMAL
    moe_utils.DEEPEP_CONFIG = ""
    moe_utils.IS_TBO_ENABLED = False
    moe_utils.IS_SBO_ENABLED = False


def load_layouts(path: pathlib.Path) -> tuple[dict[str, Any], list[Layout]]:
    payload = json.loads(path.read_text())
    if payload["status"] != "frozen_before_timing":
        raise ValueError(f"placement manifest is not frozen: {payload['status']!r}")
    expected = {
        "world_size": WORLD_SIZE,
        "num_experts": NUM_EXPERTS,
        "num_local_experts": LOCAL_EXPERTS,
        "top_k": TOP_K,
        "alignment_rows_per_expert": WEIGHT_BLOCK,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"manifest {key}={payload[key]!r}, expected {value!r}")
    if payload["selection_uses_performance_measurements"]:
        raise ValueError("layout selection used performance measurements")

    layouts = []
    for raw in payload["layouts"]:
        metrics = raw["metrics"]
        p2l = tuple(int(value) for value in metrics["p2l"])
        l2p = tuple(int(value) for value in metrics["l2p"])
        if sorted(p2l) != list(range(NUM_EXPERTS)):
            raise ValueError(f"{raw['layout_id']} p2l is not bijective")
        if sorted(l2p) != list(range(NUM_EXPERTS)):
            raise ValueError(f"{raw['layout_id']} l2p is not bijective")
        if any(l2p[logical] != physical for physical, logical in enumerate(p2l)):
            raise ValueError(f"{raw['layout_id']} maps are not inverses")
        rank_experts = metrics["rank_logical_experts"]
        if len(rank_experts) != WORLD_SIZE or any(
            len(experts) != LOCAL_EXPERTS for experts in rank_experts
        ):
            raise ValueError(f"{raw['layout_id']} does not preserve 16 slots per rank")
        layouts.append(
            Layout(
                layout_id=raw["layout_id"],
                category=raw["category"],
                swap=raw["swap"],
                p2l=p2l,
                l2p=l2p,
                metrics=metrics,
            )
        )
    if [layout.layout_id for layout in layouts].count("identity") != 1:
        raise ValueError("manifest must contain exactly one identity layout")
    return payload, layouts


def selected_layouts(args: argparse.Namespace, layouts: list[Layout]) -> list[Layout]:
    by_id = {layout.layout_id: layout for layout in layouts}
    if args.layouts:
        ids = [piece for piece in args.layouts.split(",") if piece]
    elif args.phase in {"correctness", "e1"}:
        ids = [layout.layout_id for layout in layouts]
    else:
        ids = ["identity", args.e0_swap]
    unknown = [layout_id for layout_id in ids if layout_id not in by_id]
    if unknown:
        raise ValueError(f"unknown layouts: {unknown}")
    if "identity" not in ids:
        ids.insert(0, "identity")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate layout selection: {ids}")
    return [by_id[layout_id] for layout_id in ids]


def load_snapshot(path: pathlib.Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def validate_local_snapshot(
    snapshot: dict[str, Any],
    *,
    rank: int,
    manifest: dict[str, Any],
    path: pathlib.Path,
) -> dict[str, Any]:
    if snapshot["mode"] != "snapshot" or snapshot["rank"] != rank:
        raise ValueError(f"wrong snapshot payload in {path}")
    if snapshot["capture_id"] != manifest["capture_id"]:
        raise ValueError("snapshot capture ID differs from placement manifest")
    if snapshot["layer_id"] != manifest["layer_id"]:
        raise ValueError("snapshot layer differs from placement manifest")
    if snapshot["identity_map_sha256"] != manifest["identity_service_map_sha256"]:
        raise ValueError("snapshot service-map hash differs from placement manifest")
    expected_logical = list(range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS))
    if snapshot["logical_expert_ids"] != expected_logical:
        raise ValueError(f"rank {rank} snapshot expert ownership is not identity")
    if snapshot["weight_block_size"] != [WEIGHT_BLOCK, WEIGHT_BLOCK]:
        raise ValueError("snapshot block quantization shape differs from 128x128")

    expected_shapes = {
        "hidden_states": (snapshot["hidden_rows"], HIDDEN_SIZE),
        "logical_topk_ids": (snapshot["hidden_rows"], TOP_K),
        "logical_topk_weights": (snapshot["hidden_rows"], TOP_K),
        "routed_output": (snapshot["hidden_rows"], HIDDEN_SIZE),
        "w13_weight": (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "w13_weight_scale_inv": (LOCAL_EXPERTS, 24, 32),
        "w2_weight_scale_inv": (LOCAL_EXPERTS, 32, 12),
    }
    for field, shape in expected_shapes.items():
        value = snapshot[field]
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(
                f"rank {rank} {field} shape/contiguity mismatch: "
                f"{tuple(value.shape)}, contiguous={value.is_contiguous()}"
            )
    expected_dtypes = {
        "hidden_states": torch.bfloat16,
        "logical_topk_ids": torch.int32,
        "logical_topk_weights": torch.float32,
        "routed_output": torch.bfloat16,
        "w13_weight": torch.float8_e4m3fn,
        "w2_weight": torch.float8_e4m3fn,
        "w13_weight_scale_inv": torch.float32,
        "w2_weight_scale_inv": torch.float32,
    }
    for field, dtype in expected_dtypes.items():
        if snapshot[field].dtype != dtype:
            raise ValueError(f"rank {rank} {field} dtype is {snapshot[field].dtype}")
    ids = snapshot["logical_topk_ids"]
    if ids.numel() != snapshot["hidden_rows"] * TOP_K:
        raise ValueError("top-k assignment count is inconsistent")
    if int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS:
        raise ValueError("logical top-k IDs are out of range")
    return {
        "rank": rank,
        "path": str(path),
        "hidden_rows": snapshot["hidden_rows"],
        "assignments": ids.numel(),
        "capture_id": snapshot["capture_id"],
        "layer_id": snapshot["layer_id"],
        "identity_map_sha256": snapshot["identity_map_sha256"],
    }


def build_layer(group: dist.ProcessGroup, layer_id: int):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplNormal
    from sglang.srt.layers.moe.utils import DeepEPMode, MoeRunnerBackend

    dispatcher = _DeepEPDispatcherImplNormal(
        async_finish=True,
        group=group,
        router_topk=TOP_K,
        permute_fusion=False,
        num_experts=NUM_EXPERTS,
        num_local_experts=LOCAL_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL,
    )
    config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=LOCAL_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        layer_id=layer_id,
        top_k=TOP_K,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=True,
        routed_scaling_factor=None,
    )
    runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config)
    return dispatcher, runner


def make_quant_info(tensors: dict[str, torch.Tensor]):
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

    return DeepGemmMoeQuantInfo(
        w13_weight=tensors["w13_weight"],
        w2_weight=tensors["w2_weight"],
        use_fp8=True,
        w13_scale=tensors["w13_weight_scale_inv"],
        w2_scale=tensors["w2_weight_scale_inv"],
        block_shape=[WEIGHT_BLOCK, WEIGHT_BLOCK],
    )


def copy_remote_expert(
    destination: dict[str, torch.Tensor],
    *,
    destination_slot: int,
    logical_expert: int,
    snapshot_dir: pathlib.Path,
    device: torch.device,
) -> None:
    source_rank, source_slot = divmod(logical_expert, LOCAL_EXPERTS)
    source = load_snapshot(snapshot_dir / f"snapshot_rank{source_rank}.pt")
    for field in WEIGHT_FIELDS:
        destination[field][destination_slot].copy_(source[field][source_slot].to(device))


def build_replay_state(
    *,
    local_snapshot: dict[str, Any],
    layouts: list[Layout],
    manifest: dict[str, Any],
    snapshot_dir: pathlib.Path,
    rank: int,
    device: torch.device,
) -> ReplayState:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    hidden_states = local_snapshot["hidden_states"].to(device)
    logical_topk_ids = local_snapshot["logical_topk_ids"].to(device)
    logical_topk_weights = local_snapshot["logical_topk_weights"].to(device)
    captured_output = local_snapshot["routed_output"].to(device)
    frozen_hashes = {
        "hidden_states": tensor_sha256(local_snapshot["hidden_states"]),
        "logical_topk_ids": tensor_sha256(local_snapshot["logical_topk_ids"]),
        "logical_topk_weights": tensor_sha256(local_snapshot["logical_topk_weights"]),
        "captured_output": tensor_sha256(local_snapshot["routed_output"]),
    }

    topk_by_layout = {}
    for layout in layouts:
        l2p = torch.tensor(layout.l2p, dtype=torch.int32, device=device)
        physical_ids = l2p[logical_topk_ids.to(torch.int64)].contiguous()
        topk_by_layout[layout.layout_id] = StandardTopKOutput(
            logical_topk_weights, physical_ids, None
        )

    base_tensors = {field: local_snapshot[field].to(device) for field in WEIGHT_FIELDS}
    local_start = rank * LOCAL_EXPERTS
    base_key = tuple(range(local_start, local_start + LOCAL_EXPERTS))
    tensors_by_local_map = {base_key: base_tensors}
    quant_by_local_map = {base_key: make_quant_info(base_tensors)}
    quant_by_layout = {}
    for layout in layouts:
        local_map = tuple(layout.p2l[local_start : local_start + LOCAL_EXPERTS])
        if local_map not in quant_by_local_map:
            tensors = {field: value.clone() for field, value in base_tensors.items()}
            for local_slot, logical_expert in enumerate(local_map):
                physical_expert = local_start + local_slot
                if logical_expert != physical_expert:
                    copy_remote_expert(
                        tensors,
                        destination_slot=local_slot,
                        logical_expert=logical_expert,
                        snapshot_dir=snapshot_dir,
                        device=device,
                    )
            tensors_by_local_map[local_map] = tensors
            quant_by_local_map[local_map] = make_quant_info(tensors)
        quant_by_layout[layout.layout_id] = quant_by_local_map[local_map]

    aligned_counts = [int(value) for value in manifest["expert_aligned_counts"]]
    expected_aligned_by_layout = {}
    for layout in layouts:
        expected_aligned_by_layout[layout.layout_id] = [
            aligned_counts[layout.p2l[physical]]
            for physical in range(local_start, local_start + LOCAL_EXPERTS)
        ]

    torch.cuda.synchronize(device)
    return ReplayState(
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


def new_detailed_events() -> dict[str, torch.cuda.Event]:
    return {
        name: torch.cuda.Event(enable_timing=True) for name in DETAILED_EVENT_NAMES
    }


def run_once(
    *,
    dispatcher,
    runner,
    hidden_states: torch.Tensor,
    topk_output,
    quant_info,
    events: dict[str, torch.cuda.Event] | None = None,
):
    from deep_ep import Buffer

    if events is not None:
        events["start"].record()
    dispatch_a = dispatcher.dispatch_a(hidden_states, topk_output)
    dispatch_output = dispatcher.dispatch_b(*dispatch_a)
    if events is not None:
        events["dispatch_done"].record()

    combine_input = runner.run(dispatch_output, quant_info)
    if events is not None:
        events["expert_gather_done"].record()
        events["combine_input_ready"].record()

    previous_event = Buffer.capture()
    combined, completion_event = dispatcher._combine_core(
        combine_input.hidden_states, previous_event
    )
    completion_event.current_stream_wait()
    dispatcher.handle = None
    dispatcher.src2dst = None
    if events is not None:
        events["combine_done"].record()
    return combined, dispatch_output.num_recv_tokens_per_expert


def actual_aligned_counts(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [int(value) for value in values]


def start_alignment(marker: torch.Tensor, group: dist.ProcessGroup) -> None:
    # This collective is a block-boundary stream alignment and is outside timing.
    dist.all_reduce(marker, op=dist.ReduceOp.MAX, group=group)


def warm_layouts(
    *,
    layouts: list[Layout],
    state: ReplayState,
    dispatcher,
    runner,
    marker: torch.Tensor,
    group: dist.ProcessGroup,
    warmup: int,
    rank: int,
) -> list[dict[str, Any]]:
    results = []
    for layout in layouts:
        start_alignment(marker, group)
        counts = None
        for _ in range(warmup):
            _, counts = run_once(
                dispatcher=dispatcher,
                runner=runner,
                hidden_states=state.hidden_states,
                topk_output=state.topk_by_layout[layout.layout_id],
                quant_info=state.quant_by_layout[layout.layout_id],
            )
        torch.cuda.synchronize()
        actual = actual_aligned_counts(counts)
        expected = state.expected_aligned_by_layout[layout.layout_id]
        if actual != expected:
            raise ValueError(
                f"rank {rank} layout {layout.layout_id} DeepEP aligned counts differ: "
                f"expected {expected}, got {actual}"
            )
        results.append(
            {
                "layout_id": layout.layout_id,
                "rank": rank,
                "actual_aligned_counts": actual,
                "expected_aligned_counts": expected,
            }
        )
    return results


def comparison_metrics(
    value: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None,
    rtol: float | None,
) -> dict[str, Any]:
    value_fp32 = value.float()
    reference_fp32 = reference.float()
    difference = (value_fp32 - reference_fp32).abs()
    denominator = reference_fp32.abs().clamp_min(RELATIVE_DENOMINATOR_FLOOR)
    finite_value = torch.isfinite(value_fp32)
    finite_reference = torch.isfinite(reference_fp32)
    metrics: dict[str, Any] = {
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float((difference / denominator).max().item()),
        "relative_denominator": (
            f"max(abs(reference), {RELATIVE_DENOMINATOR_FLOOR:.1e})"
        ),
        "value_nan_count": int(torch.isnan(value_fp32).sum().item()),
        "value_posinf_count": int(torch.isposinf(value_fp32).sum().item()),
        "value_neginf_count": int(torch.isneginf(value_fp32).sum().item()),
        "reference_nan_count": int(torch.isnan(reference_fp32).sum().item()),
        "reference_posinf_count": int(torch.isposinf(reference_fp32).sum().item()),
        "reference_neginf_count": int(torch.isneginf(reference_fp32).sum().item()),
        "nonfinite_pair_count": int((~(finite_value & finite_reference)).sum().item()),
        "elements": value.numel(),
    }
    if (atol is None) != (rtol is None):
        raise ValueError("atol and rtol must both be provided or both be omitted")
    if atol is not None and rtol is not None:
        failures = ~torch.isclose(
            value_fp32,
            reference_fp32,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
        metrics.update(
            {
                "atol": atol,
                "rtol": rtol,
                "isclose_failure_count": int(failures.sum().item()),
            }
        )
    else:
        metrics.update({"atol": None, "rtol": None, "isclose_failure_count": None})
    return metrics


def run_correctness(
    *,
    layouts: list[Layout],
    state: ReplayState,
    dispatcher,
    runner,
    marker: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
    atol: float | None,
    rtol: float | None,
) -> list[dict[str, Any]] | None:
    by_id = {layout.layout_id: layout for layout in layouts}
    identity = by_id["identity"]

    start_alignment(marker, group)
    identity_output, identity_counts = run_once(
        dispatcher=dispatcher,
        runner=runner,
        hidden_states=state.hidden_states,
        topk_output=state.topk_by_layout[identity.layout_id],
        quant_info=state.quant_by_layout[identity.layout_id],
    )
    torch.cuda.synchronize()
    identity_reference = identity_output.clone()
    if actual_aligned_counts(identity_counts) != state.expected_aligned_by_layout["identity"]:
        raise ValueError(f"rank {rank} identity aligned counts changed after warmup")

    local_records = []
    for layout in layouts:
        start_alignment(marker, group)
        output, counts = run_once(
            dispatcher=dispatcher,
            runner=runner,
            hidden_states=state.hidden_states,
            topk_output=state.topk_by_layout[layout.layout_id],
            quant_info=state.quant_by_layout[layout.layout_id],
        )
        torch.cuda.synchronize()
        actual = actual_aligned_counts(counts)
        expected = state.expected_aligned_by_layout[layout.layout_id]
        if actual != expected:
            raise ValueError(
                f"rank {rank} layout {layout.layout_id} aligned counts changed"
            )
        local_records.append(
            {
                "rank": rank,
                "layout_id": layout.layout_id,
                "category": layout.category,
                "swap": layout.swap,
                "logical_route_ids_sha256": state.frozen_hashes["logical_topk_ids"],
                "logical_route_weights_sha256": state.frozen_hashes[
                    "logical_topk_weights"
                ],
                "expected_physical_experts": list(
                    layout.p2l[
                        rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS
                    ]
                ),
                "actual_aligned_counts": actual,
                "expected_aligned_counts": expected,
                "versus_captured_output": comparison_metrics(
                    output, state.captured_output, atol=atol, rtol=rtol
                ),
                "versus_identity_replay": comparison_metrics(
                    output, identity_reference, atol=atol, rtol=rtol
                ),
            }
        )

    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_records, group=group)
    if rank == 0:
        return [record for rank_records in gathered for record in rank_records]
    return None


def load_tolerance(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> tuple[float | None, float | None, dict[str, Any] | None]:
    if args.phase == "correctness":
        if (args.atol is None) != (args.rtol is None):
            raise ValueError("correctness pilot requires both --atol and --rtol or neither")
        return args.atol, args.rtol, None
    if args.atol is not None or args.rtol is not None:
        raise ValueError("timed phases read a frozen tolerance file; flags are prohibited")
    path = args.tolerance_file or args.root / "inputs" / "correctness_tolerance.json"
    payload = json.loads(path.read_text())
    required = {
        "frozen_before_performance_timing": True,
        "capture_id": manifest["capture_id"],
        "layer_id": manifest["layer_id"],
        "placement_manifest_sha256": manifest_sha256,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(
                f"tolerance file {key}={payload.get(key)!r}, expected {value!r}"
            )
    return float(payload["atol"]), float(payload["rtol"]), payload


def validate_correctness_gate(
    records: list[dict[str, Any]] | None,
    *,
    rank: int,
    group: dist.ProcessGroup,
) -> None:
    failure = False
    message = ""
    if rank == 0:
        assert records is not None
        failures = []
        for record in records:
            for comparison in ("versus_captured_output", "versus_identity_replay"):
                metrics = record[comparison]
                if metrics["nonfinite_pair_count"] or metrics["isclose_failure_count"]:
                    failures.append(
                        {
                            "rank": record["rank"],
                            "layout_id": record["layout_id"],
                            "comparison": comparison,
                            "nonfinite_pair_count": metrics["nonfinite_pair_count"],
                            "isclose_failure_count": metrics["isclose_failure_count"],
                            "max_abs_error": metrics["max_abs_error"],
                        }
                    )
        failure = bool(failures)
        if failures:
            message = f"correctness gate failed: {failures[:8]}"
    payload: list[Any] = [(failure, message)]
    dist.broadcast_object_list(payload, src=0, group=group)
    if payload[0][0]:
        raise ValueError(payload[0][1])


def timed_block(
    *,
    layout: Layout,
    observation_mode: str,
    iterations: int,
    state: ReplayState,
    dispatcher,
    runner,
    start_marker: torch.Tensor,
    end_marker: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
) -> dict[str, Any] | None:
    detailed = observation_mode == "detailed"
    detailed_events = [new_detailed_events() for _ in range(iterations)] if detailed else []
    local_start = torch.cuda.Event(enable_timing=True)
    local_end = torch.cuda.Event(enable_timing=True)
    reference_start = torch.cuda.Event(enable_timing=True) if rank == 0 else None
    reference_end = torch.cuda.Event(enable_timing=True) if rank == 0 else None

    start_alignment(start_marker, group)
    if rank == 0:
        reference_start.record()
    local_start.record()
    output = None
    for index in range(iterations):
        output, _ = run_once(
            dispatcher=dispatcher,
            runner=runner,
            hidden_states=state.hidden_states,
            topk_output=state.topk_by_layout[layout.layout_id],
            quant_info=state.quant_by_layout[layout.layout_id],
            events=detailed_events[index] if detailed else None,
        )
    local_end.record()

    # This is after every local combine completion. It is the only timed
    # cross-rank fence and is not inserted between expert compute and combine.
    dist.reduce(end_marker, dst=0, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        reference_end.record()
    torch.cuda.synchronize()

    local_block_us = local_start.elapsed_time(local_end) * 1000.0
    local_record: dict[str, Any] = {
        "rank": rank,
        "local_block_us": local_block_us,
        "local_us_per_forward": local_block_us / iterations,
        "output_abs_max": float(output.float().abs().max().item()),
    }
    if detailed:
        raw = {
            "layer_us": [],
            "dispatch_us": [],
            "expert_gather_us": [],
            "combine_handoff_us": [],
            "combine_us": [],
        }
        for events in detailed_events:
            raw["layer_us"].append(
                events["start"].elapsed_time(events["combine_done"]) * 1000.0
            )
            raw["dispatch_us"].append(
                events["start"].elapsed_time(events["dispatch_done"]) * 1000.0
            )
            raw["expert_gather_us"].append(
                events["dispatch_done"].elapsed_time(events["expert_gather_done"])
                * 1000.0
            )
            raw["combine_handoff_us"].append(
                events["expert_gather_done"].elapsed_time(
                    events["combine_input_ready"]
                )
                * 1000.0
            )
            raw["combine_us"].append(
                events["combine_input_ready"].elapsed_time(events["combine_done"])
                * 1000.0
            )
        local_record["detailed_iteration_us"] = raw
        local_record["detailed_summary_us"] = {
            name: distribution(values) for name, values in raw.items()
        }

    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_record, group=group)
    if rank != 0:
        return None
    completion_fenced_us = reference_start.elapsed_time(reference_end) * 1000.0
    return {
        "layout_id": layout.layout_id,
        "category": layout.category,
        "swap": layout.swap,
        "observation_mode": observation_mode,
        "iterations": iterations,
        "completion_fenced_block_us": completion_fenced_us,
        "completion_fenced_us_per_forward": completion_fenced_us / iterations,
        "t_proxy_block_us": max(record["local_block_us"] for record in gathered),
        "t_proxy_us_per_forward": max(
            record["local_us_per_forward"] for record in gathered
        ),
        "ranks": gathered,
    }


def calibrate_empty_fence(
    *,
    warmup: int,
    repetitions: int,
    start_marker: torch.Tensor,
    end_marker: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
) -> dict[str, Any] | None:
    reference_values = []
    local_values: list[list[float]] = []
    total = warmup + repetitions
    for index in range(total):
        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        reference_start = torch.cuda.Event(enable_timing=True) if rank == 0 else None
        reference_end = torch.cuda.Event(enable_timing=True) if rank == 0 else None
        start_alignment(start_marker, group)
        if rank == 0:
            reference_start.record()
        local_start.record()
        local_end.record()
        dist.reduce(end_marker, dst=0, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            reference_end.record()
        torch.cuda.synchronize()
        local_us = local_start.elapsed_time(local_end) * 1000.0
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_us, group=group)
        if rank == 0 and index >= warmup:
            reference_values.append(
                reference_start.elapsed_time(reference_end) * 1000.0
            )
            local_values.append([float(value) for value in gathered])
    if rank == 0:
        return {
            "semantics": (
                "GPU-0 same-clock empty interval after untimed NCCL start alignment "
                "through completion of one NCCL reduce"
            ),
            "warmup": warmup,
            "repetitions": repetitions,
            "reference_us": reference_values,
            "reference_summary_us": distribution(reference_values),
            "per_rank_local_event_overhead_us": local_values,
        }
    return None


def condition_schedule(
    *,
    phase: str,
    layouts: list[Layout],
    e0_swap: str,
) -> list[tuple[Layout, str]]:
    by_id = {layout.layout_id: layout for layout in layouts}
    if phase == "e0":
        return [
            (by_id["identity"], "minimal"),
            (by_id[e0_swap], "minimal"),
            (by_id["identity"], "detailed"),
            (by_id[e0_swap], "detailed"),
        ]
    return [(layout, "minimal") for layout in layouts]


def summarize_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['layout_id']}:{record['observation_mode']}"
        grouped.setdefault(key, []).append(record)
    result = {}
    for key, values in grouped.items():
        result[key] = {
            "blocks": len(values),
            "completion_fenced_us_per_forward": distribution(
                [value["completion_fenced_us_per_forward"] for value in values]
            ),
            "t_proxy_us_per_forward": distribution(
                [value["t_proxy_us_per_forward"] for value in values]
            ),
        }
    return result


def source_audit() -> dict[str, Any]:
    body = inspect.getsource(run_once)
    prohibited = {
        "distributed_collective_in_layer": "dist." in body,
        "sleep_or_spin_in_layer": "sleep" in body or "spin" in body,
        "synthetic_delay_symbol_in_layer": "delay" in body,
    }
    if any(prohibited.values()):
        raise ValueError(f"run_once intervention audit failed: {prohibited}")
    return prohibited


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    manifest_path = (args.manifest or args.root / "placements" / "manifest.json").resolve()
    snapshot_dir = (
        args.snapshot_dir or args.root / "inputs" / "live_capture"
    ).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest, all_layouts = load_layouts(manifest_path)
    layouts = selected_layouts(args, all_layouts)
    atol, rtol, tolerance_payload = load_tolerance(
        args, manifest=manifest, manifest_sha256=manifest_sha256
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise ValueError(f"this experiment requires {WORLD_SIZE} ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(list(range(world_size)), backend="nccl")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.random_seed + rank)

    configure_sglang_globals()
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPConfig

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SGLang JIT DeepGEMM is not enabled")

    local_path = snapshot_dir / f"snapshot_rank{rank}.pt"
    local_snapshot = load_snapshot(local_path)
    local_snapshot_meta = validate_local_snapshot(
        local_snapshot, rank=rank, manifest=manifest, path=local_path
    )
    gathered_snapshot_meta: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_snapshot_meta, local_snapshot_meta, group=group)
    if rank == 0:
        if sum(item["hidden_rows"] for item in gathered_snapshot_meta) != manifest[
            "total_hidden_rows"
        ]:
            raise ValueError("snapshot hidden-row total differs from manifest")
        if sum(item["assignments"] for item in gathered_snapshot_meta) != manifest[
            "total_valid_assignments"
        ]:
            raise ValueError("snapshot assignment total differs from manifest")

    state = build_replay_state(
        local_snapshot=local_snapshot,
        layouts=layouts,
        manifest=manifest,
        snapshot_dir=snapshot_dir,
        rank=rank,
        device=device,
    )
    dispatcher, runner = build_layer(group, manifest["layer_id"])
    start_marker = torch.ones(1, dtype=torch.float32, device=device)
    end_marker = torch.ones(1, dtype=torch.float32, device=device)

    started_at = dt.datetime.now(dt.timezone.utc)
    warmup_records = warm_layouts(
        layouts=layouts,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        marker=start_marker,
        group=group,
        warmup=args.warmup,
        rank=rank,
    )
    gathered_warmup: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_warmup, warmup_records, group=group)

    correctness_records = run_correctness(
        layouts=layouts,
        state=state,
        dispatcher=dispatcher,
        runner=runner,
        marker=start_marker,
        group=group,
        rank=rank,
        atol=atol,
        rtol=rtol,
    )
    if args.phase != "correctness":
        validate_correctness_gate(correctness_records, rank=rank, group=group)

    empty_fence = None
    timing_records: list[dict[str, Any]] = []
    schedule_records: list[list[str]] = []
    if args.phase != "correctness":
        empty_fence = calibrate_empty_fence(
            warmup=args.empty_fence_warmup,
            repetitions=args.empty_fence_repetitions,
            start_marker=start_marker,
            end_marker=end_marker,
            group=group,
            rank=rank,
        )
        conditions = condition_schedule(
            phase=args.phase, layouts=layouts, e0_swap=args.e0_swap
        )
        for block in range(args.blocks):
            if rank == 0:
                order = list(range(len(conditions)))
                random.Random(args.random_seed + block).shuffle(order)
            else:
                order = None
            payload = [order]
            dist.broadcast_object_list(payload, src=0, group=group)
            order = payload[0]
            if rank == 0:
                schedule_records.append(
                    [
                        f"{conditions[index][0].layout_id}:"
                        f"{conditions[index][1]}"
                        for index in order
                    ]
                )
            for condition_order, condition_index in enumerate(order):
                layout, observation_mode = conditions[condition_index]
                record = timed_block(
                    layout=layout,
                    observation_mode=observation_mode,
                    iterations=args.iterations_per_block,
                    state=state,
                    dispatcher=dispatcher,
                    runner=runner,
                    start_marker=start_marker,
                    end_marker=end_marker,
                    group=group,
                    rank=rank,
                )
                if rank == 0:
                    record.update(
                        {
                            "block": block,
                            "condition_order": condition_order,
                            "condition_index": condition_index,
                        }
                    )
                    timing_records.append(record)

    final_hashes = {
        "hidden_states": tensor_sha256(state.hidden_states),
        "logical_topk_ids": tensor_sha256(state.logical_topk_ids),
        "logical_topk_weights": tensor_sha256(state.logical_topk_weights),
        "captured_output": tensor_sha256(state.captured_output),
    }
    if final_hashes != state.frozen_hashes:
        raise ValueError("frozen source tensors changed during replay")
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
        output = args.output
        if output is None:
            stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
            output = args.root / "results" / "raw" / f"{args.phase}_{stamp}.json"
        output = output.resolve()
        deep_ep_config = DeepEPConfig.get_instance()
        payload = {
            "schema_version": 1,
            "experiment": "h20_directional_expert_placement",
            "phase": args.phase,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "hostname": socket.gethostname(),
            "manifest_path": str(manifest_path),
            "placement_manifest_sha256": manifest_sha256,
            "capture_id": manifest["capture_id"],
            "layer_id": manifest["layer_id"],
            "selected_layouts": [layout.layout_id for layout in layouts],
            "snapshot_metadata": gathered_snapshot_meta,
            "frozen_tensor_hashes": gathered_hashes,
            "shape": {
                "total_hidden_rows": manifest["total_hidden_rows"],
                "total_valid_assignments": manifest["total_valid_assignments"],
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_experts": NUM_EXPERTS,
                "num_local_experts": LOCAL_EXPERTS,
                "top_k": TOP_K,
                "weight_block": [WEIGHT_BLOCK, WEIGHT_BLOCK],
            },
            "intervention_audit": {
                "only_logical_to_physical_placement_changes": True,
                "logical_routes_and_weights_frozen": True,
                "weight_migration_before_warmup_and_timing": True,
                "jit_and_warmup_before_timing": True,
                "common_delay_floor_us": None,
                "differential_delay_us": None,
                "mid_layer_collective_between_expert_and_combine": False,
                "source_scan": source_audit(),
            },
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
            "tolerance": tolerance_payload
            if tolerance_payload is not None
            else {
                "atol": atol,
                "rtol": rtol,
                "status": "pilot_only_not_frozen" if atol is None else "caller_supplied_pilot",
                "relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR,
            },
            "measurement": {
                "warmup_forwards_per_layout": args.warmup,
                "blocks": 0 if args.phase == "correctness" else args.blocks,
                "iterations_per_block": (
                    0 if args.phase == "correctness" else args.iterations_per_block
                ),
                "random_seed": args.random_seed,
                "local_timing": "same-GPU CUDA event subtraction only",
                "t_proxy": "max rank-local block elapsed; not a calibrated global makespan",
                "completion_fenced_timing": (
                    "GPU-0 same-clock interval after untimed NCCL start alignment "
                    "through an NCCL end reduce enqueued after local completion"
                ),
            },
            "warmup_aligned_count_checks": gathered_warmup,
            "correctness": {
                "records": correctness_records,
                "relative_denominator": (
                    f"max(abs(reference), {RELATIVE_DENOMINATOR_FLOOR:.1e})"
                ),
            },
            "empty_completion_fence": empty_fence,
            "condition_schedule": schedule_records,
            "timing_records": timing_records,
            "timing_summary": summarize_timing(timing_records)
            if timing_records
            else {},
            "software": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
        }
        write_json_atomic(output, payload)
        print(f"RESULT_JSON={output}", flush=True)
        if args.phase == "correctness":
            max_captured = max(
                record["versus_captured_output"]["max_abs_error"]
                for record in correctness_records
            )
            max_identity = max(
                record["versus_identity_replay"]["max_abs_error"]
                for record in correctness_records
            )
            print(
                f"CORRECTNESS max_abs_vs_captured={max_captured:.9g} "
                f"max_abs_vs_identity={max_identity:.9g}",
                flush=True,
            )
        else:
            for key, summary in payload["timing_summary"].items():
                completion = summary["completion_fenced_us_per_forward"]
                proxy = summary["t_proxy_us_per_forward"]
                print(
                    f"{key}: completion_fenced median={completion['median']:.3f} us "
                    f"T_proxy median={proxy['median']:.3f} us",
                    flush=True,
                )

    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
