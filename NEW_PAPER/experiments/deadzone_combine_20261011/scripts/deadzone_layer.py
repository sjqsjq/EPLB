#!/usr/bin/env python3
"""Actual-path DeepEP-normal + DeepGEMM-FP8 one-layer deadzone harness.

Launch with torchrun. The selected rank's delay is enqueued after the real
DeepGEMM down projection and DeepEP-normal ep_gather, immediately before the
Buffer.capture() event consumed by asynchronous DeepEP combine.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import socket
import statistics
from typing import Any

# These must be set before importing SGLang's DeepGEMM modules.
os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "512")
os.environ.setdefault("SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "1")

import torch
import torch.distributed as dist

from gpu_delay import GpuDelay
from route_imbalance import (
    EXPERT_POLICY_PACKED_HOT,
    ROUTE_FIXED7_ONE_DOUBLE,
    ROUTE_ONE_PER_RANK,
    build_route,
)


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
NUM_EXPERTS = 128
TOP_K = 8
WEIGHT_BLOCK = 128
EXPECTED_WORLD_SIZE = 8
LOCAL_EXPERTS = NUM_EXPERTS // EXPECTED_WORLD_SIZE
# Keep every launch above the empty-kernel latency floor. The selected rank
# spins for FLOOR + Δ while all other ranks spin for FLOOR, so the measured
# differential remains Δ even for 0.05*C0 targets of only a few microseconds.
DELAY_FLOOR_US = 20.0
METRIC_NAMES = (
    "layer_ms",
    "dispatch_ms",
    "expert_ms",
    "injected_delay_ms",
    "strict_gate_ms",
    "combine_after_gate_ms",
    "post_expert_to_done_ms",
    "correctness_max_abs",
)


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    mode: str
    delay_us: float
    delay_ranks: tuple[int, ...]
    delay_ratio: float | None = None
    route_kind: str = ROUTE_ONE_PER_RANK
    requested_r: float | None = None
    target_hot_assignments: int | None = None
    hot_rank: int | None = None
    expert_policy: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], world_size: int) -> "Condition":
        mode = str(value.get("mode", "original"))
        if mode not in {"original", "strict_barrier"}:
            raise ValueError(f"Unknown mode {mode!r}")
        ranks_value = value.get("delay_ranks", [0])
        if ranks_value == "all":
            ranks = tuple(range(world_size))
        elif isinstance(ranks_value, int):
            ranks = (ranks_value,)
        else:
            ranks = tuple(int(rank) for rank in ranks_value)
        if any(rank < 0 or rank >= world_size for rank in ranks):
            raise ValueError(f"Invalid delay ranks {ranks} for world size {world_size}")
        delay_us = float(value.get("delay_us", 0.0))
        if delay_us < 0:
            raise ValueError("delay_us must be nonnegative")
        name = str(
            value.get(
                "name",
                f"{mode}-delay{delay_us:g}us-ranks{'-'.join(map(str, ranks))}",
            )
        )
        delay_ratio_value = value.get("delay_ratio")
        delay_ratio = (
            None if delay_ratio_value is None else float(delay_ratio_value)
        )
        route_kind = str(value.get("route_kind", ROUTE_ONE_PER_RANK))
        if route_kind not in {ROUTE_ONE_PER_RANK, ROUTE_FIXED7_ONE_DOUBLE}:
            raise ValueError(f"Unknown route kind {route_kind!r}")
        requested_r_value = value.get("requested_r")
        requested_r = None if requested_r_value is None else float(requested_r_value)
        target_value = value.get("target_hot_assignments")
        target_hot_assignments = None if target_value is None else int(target_value)
        hot_rank_value = value.get("hot_rank")
        hot_rank = None if hot_rank_value is None else int(hot_rank_value)
        expert_policy_value = value.get("expert_policy")
        expert_policy = None if expert_policy_value is None else str(expert_policy_value)
        if route_kind == ROUTE_FIXED7_ONE_DOUBLE:
            if requested_r is None or target_hot_assignments is None or hot_rank is None:
                raise ValueError(
                    "fixed7_one_double requires requested_r, "
                    "target_hot_assignments, and hot_rank"
                )
            if expert_policy != EXPERT_POLICY_PACKED_HOT:
                raise ValueError(f"Unknown expert policy {expert_policy!r}")
            if hot_rank < 0 or hot_rank >= world_size:
                raise ValueError(f"Invalid hot rank {hot_rank}")
        return cls(
            name=name,
            mode=mode,
            delay_us=delay_us,
            delay_ranks=ranks,
            delay_ratio=delay_ratio,
            route_kind=route_kind,
            requested_r=requested_r,
            target_hot_assignments=target_hot_assignments,
            hot_rank=hot_rank,
            expert_policy=expert_policy,
        )

    def route_key(self) -> tuple[Any, ...]:
        return (
            self.route_kind,
            self.requested_r,
            self.target_hot_assignments,
            self.hot_rank,
            self.expert_policy,
        )

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["delay_ranks"] = list(self.delay_ranks)
        return result


def parse_rank_spec(value: str, world_size: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(world_size))
    ranks = tuple(int(piece) for piece in value.split(",") if piece)
    if not ranks:
        raise ValueError("delay rank list is empty")
    if any(rank < 0 or rank >= world_size for rank in ranks):
        raise ValueError(f"Invalid delay ranks {ranks} for world size {world_size}")
    return ranks


def load_conditions(args: argparse.Namespace, world_size: int) -> list[Condition]:
    if args.conditions_json is None:
        return [
            Condition(
                name=args.condition_name
                or f"{args.mode}-delay{args.delay_us:g}us-ranks{args.delay_ranks}",
                mode=args.mode,
                delay_us=args.delay_us,
                delay_ranks=parse_rank_spec(args.delay_ranks, world_size),
            )
        ]
    payload = json.loads(pathlib.Path(args.conditions_json).read_text())
    raw_conditions = payload["conditions"] if isinstance(payload, dict) else payload
    conditions = [Condition.from_dict(value, world_size) for value in raw_conditions]
    if not conditions:
        raise ValueError("conditions JSON contains no conditions")
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError(f"condition names must be unique: {names}")
    return conditions


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)

    result: dict[str, Any] = {}
    for name, condition_records in by_condition.items():
        metric_summary: dict[str, Any] = {}
        for metric in METRIC_NAMES:
            values = [
                max(rank_metrics[metric] for rank_metrics in record["ranks"])
                for record in condition_records
            ]
            metric_summary[metric] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p05": percentile(values, 0.05),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
        result[name] = {
            "repetitions": len(condition_records),
            "cross_rank_max": metric_summary,
        }
    return result


def configure_sglang_globals() -> None:
    # The isolated runner does not construct ServerArgs, so initialize the same
    # global backend choices explicitly before MoeRunner construction.
    from sglang.srt.layers.moe import utils as moe_utils

    moe_utils.MOE_A2A_BACKEND = moe_utils.MoeA2ABackend.DEEPEP
    moe_utils.MOE_RUNNER_BACKEND = moe_utils.MoeRunnerBackend.DEEP_GEMM
    moe_utils.DEEPEP_MODE = moe_utils.DeepEPMode.NORMAL
    moe_utils.DEEPEP_CONFIG = ""
    moe_utils.IS_TBO_ENABLED = False
    moe_utils.IS_SBO_ENABLED = False


def make_balanced_route(
    total_tokens: int, rank: int, world_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_tokens % world_size != 0:
        raise ValueError("total tokens must be divisible by world size")
    local_tokens = total_tokens // world_size
    global_token = rank * local_tokens + torch.arange(local_tokens, device=device)
    destination_rank = torch.arange(world_size, device=device)
    local_expert = global_token[:, None] % LOCAL_EXPERTS
    topk_ids = (destination_rank[None, :] * LOCAL_EXPERTS + local_expert).to(torch.int32)
    topk_weights = torch.full(
        (local_tokens, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device=device,
    )
    return topk_ids.contiguous(), topk_weights


def create_weights(device: torch.device):
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

    # Constant, finite FP8 data removes host RNG cost while preserving the exact
    # grouped DeepGEMM shapes and memory traffic of one sharded Qwen3 MoE layer.
    weight_value = 2.0**-9
    w13 = torch.full(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        weight_value,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2 = torch.full(
        (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        weight_value,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w13_scale = torch.ones(
        (
            LOCAL_EXPERTS,
            (2 * INTERMEDIATE_SIZE + WEIGHT_BLOCK - 1) // WEIGHT_BLOCK,
            (HIDDEN_SIZE + WEIGHT_BLOCK - 1) // WEIGHT_BLOCK,
        ),
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.ones(
        (
            LOCAL_EXPERTS,
            (HIDDEN_SIZE + WEIGHT_BLOCK - 1) // WEIGHT_BLOCK,
            (INTERMEDIATE_SIZE + WEIGHT_BLOCK - 1) // WEIGHT_BLOCK,
        ),
        dtype=torch.float32,
        device=device,
    )
    return DeepGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[WEIGHT_BLOCK, WEIGHT_BLOCK],
    )


def build_layer(group: dist.ProcessGroup):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        _DeepEPDispatcherImplNormal,
    )
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
        layer_id=0,
        top_k=TOP_K,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=True,
    )
    runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config)
    return dispatcher, runner


def run_layer_once(
    *,
    dispatcher,
    runner,
    quant_info,
    hidden_states: torch.Tensor,
    topk_output,
    condition: Condition,
    rank: int,
    group: dist.ProcessGroup,
    strict_marker: torch.Tensor,
    delay: GpuDelay,
    record_events: bool,
):
    from deep_ep import Buffer

    events = {
        name: torch.cuda.Event(enable_timing=record_events)
        for name in (
            "start",
            "dispatch_done",
            "expert_done",
            "delay_done",
            "gate_done",
            "end",
        )
    }

    torch.cuda.nvtx.range_push(f"deadzone_layer:{condition.name}")
    events["start"].record()

    torch.cuda.nvtx.range_push("dispatch")
    dispatch_a = dispatcher.dispatch_a(hidden_states, topk_output)
    dispatch_output = dispatcher.dispatch_b(*dispatch_a)
    events["dispatch_done"].record()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("experts_deepgemm_fp8")
    combine_input = runner.run(dispatch_output, quant_info)
    events["expert_done"].record()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("injected_gpu_delay")
    # Launch exactly one identical-symbol kernel on every rank in every
    # condition. All ranks use the same floor above empty-kernel latency;
    # selected ranks add Δ, making even sub-10-us differentials measurable.
    differential_us = condition.delay_us if rank in condition.delay_ranks else 0.0
    delay(round((DELAY_FLOOR_US + differential_us) * 1000.0))
    events["delay_done"].record()
    torch.cuda.nvtx.range_pop()

    if condition.mode == "strict_barrier":
        torch.cuda.nvtx.range_push("strict_gpu_ready_allreduce")
        # NCCL orders the collective after this rank's producer stream. MAX
        # keeps the persistent marker equal to one, avoiding timed resets.
        dist.all_reduce(strict_marker, op=dist.ReduceOp.MAX, group=group)
        torch.cuda.nvtx.range_pop()
    events["gate_done"].record()

    torch.cuda.nvtx.range_push("deepep_combine")
    previous_event = Buffer.capture()
    combined, completion_event = dispatcher._combine_core(
        combine_input.hidden_states, previous_event
    )
    completion_event.current_stream_wait()
    dispatcher.handle = None
    dispatcher.src2dst = None
    events["end"].record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    return combined, events, dispatch_output.num_recv_tokens_per_expert


def event_metrics(events: dict[str, torch.cuda.Event], correctness: float) -> list[float]:
    elapsed = lambda left, right: events[left].elapsed_time(events[right])
    return [
        elapsed("start", "end"),
        elapsed("start", "dispatch_done"),
        elapsed("dispatch_done", "expert_done"),
        elapsed("expert_done", "delay_done"),
        elapsed("delay_done", "gate_done"),
        elapsed("gate_done", "end"),
        elapsed("expert_done", "end"),
        correctness,
    ]


def gpu_start_gate(marker: torch.Tensor, group: dist.ProcessGroup) -> None:
    dist.all_reduce(marker, op=dist.ReduceOp.MAX, group=group)
    torch.cuda.synchronize()


def calibrate_delay_kernel(
    delay: GpuDelay,
    delay_values_us: list[float],
    rank: int,
    group: dist.ProcessGroup,
    world_size: int,
    repetitions: int,
) -> dict[str, Any] | None:
    if repetitions <= 0:
        return {} if rank == 0 else None
    result: dict[str, Any] = {}
    for delay_us in sorted(set(delay_values_us)):
        local_values = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            delay(round((DELAY_FLOOR_US + delay_us) * 1000.0))
            end.record()
            end.synchronize()
            local_values.append(start.elapsed_time(end))
        local_median = torch.tensor(
            [statistics.median(local_values)], dtype=torch.float64, device="cuda"
        )
        gathered = [torch.empty_like(local_median) for _ in range(world_size)]
        dist.all_gather(gathered, local_median, group=group)
        if rank == 0:
            values = [item.item() for item in gathered]
            result[f"{delay_us:g}"] = {
                "target_differential_us": delay_us,
                "command_spin_us": DELAY_FLOOR_US + delay_us,
                "per_rank_median_us": [value * 1000.0 for value in values],
                "cross_rank_median_us": statistics.median(values) * 1000.0,
                "cross_rank_min_us": min(values) * 1000.0,
                "cross_rank_max_us": max(values) * 1000.0,
            }
    if rank == 0:
        zero = result.get("0")
        if zero is not None:
            zero_by_rank = zero["per_rank_median_us"]
            for item in result.values():
                net = [
                    value - baseline
                    for value, baseline in zip(
                        item["per_rank_median_us"], zero_by_rank, strict=True
                    )
                ]
                item["per_rank_net_us"] = net
                item["cross_rank_net_median_us"] = statistics.median(net)
                item["cross_rank_net_min_us"] = min(net)
                item["cross_rank_net_max_us"] = max(net)
        return result
    return None


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, choices=(256, 4096), default=256)
    parser.add_argument("--mode", choices=("original", "strict_barrier"), default="original")
    parser.add_argument("--delay-us", type=float, default=0.0)
    parser.add_argument("--delay-ranks", default="0", help="comma-separated ranks or all")
    parser.add_argument("--condition-name")
    parser.add_argument("--conditions-json", type=pathlib.Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--iterations-per-block", type=int, default=5)
    parser.add_argument("--delay-calibration-repetitions", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20261011)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--profile-capture-condition",
        help="wrap this measured condition in an all-rank NVTX capture range",
    )
    parser.add_argument("--no-correctness-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(f"This experiment requires 8 ranks, got {world_size}")
    if args.total_tokens % world_size != 0:
        raise ValueError("total tokens must be divisible by world size")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.new_group(list(range(world_size)), backend="nccl")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.random_seed + rank)

    configure_sglang_globals()
    from deep_ep import Buffer
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPConfig, DeepEPBuffer
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SGLang JIT DeepGEMM is not enabled")

    conditions = load_conditions(args, world_size)
    delay = GpuDelay()
    delay_calibration = calibrate_delay_kernel(
        delay,
        [condition.delay_us for condition in conditions],
        rank,
        group,
        world_size,
        args.delay_calibration_repetitions,
    )

    route_outputs: dict[tuple[Any, ...], Any] = {}
    route_manifests: dict[tuple[Any, ...], dict[str, Any]] = {}
    for condition in conditions:
        route_key = condition.route_key()
        if route_key in route_outputs:
            continue
        topk_ids, topk_weights, manifest = build_route(
            total_tokens=args.total_tokens,
            rank=rank,
            world_size=world_size,
            device=device,
            route_kind=condition.route_kind,
            requested_r=condition.requested_r,
            target_hot_assignments=condition.target_hot_assignments,
            hot_rank=condition.hot_rank,
            expert_policy=condition.expert_policy,
        )
        route_outputs[route_key] = StandardTopKOutput(topk_weights, topk_ids, None)
        route_manifests[route_key] = manifest

    local_tokens = args.total_tokens // world_size
    hidden_states = torch.randn(
        (local_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    quant_info = create_weights(device)
    dispatcher, runner = build_layer(group)
    strict_marker = torch.ones(1, dtype=torch.float32, device=device)
    start_marker = torch.ones(1, dtype=torch.float32, device=device)

    reference_condition = Condition("reference", "original", 0.0, (0,))
    reference_outputs: dict[tuple[Any, ...], torch.Tensor] = {}

    # Warm every unique padded route shape independently. Besides excluding
    # compilation and allocator effects, the last pass validates DeepEP's
    # actual 128-aligned expert counts against the route manifest.
    for route_key, topk_output in route_outputs.items():
        actual_aligned_counts = None
        for warmup_index in range(args.warmup + 1):
            gpu_start_gate(start_marker, group)
            output, warmup_events, actual_aligned_counts = run_layer_once(
                dispatcher=dispatcher,
                runner=runner,
                quant_info=quant_info,
                hidden_states=hidden_states,
                topk_output=topk_output,
                condition=reference_condition,
                rank=rank,
                group=group,
                strict_marker=strict_marker,
                delay=delay,
                record_events=False,
            )
            warmup_events["end"].synchronize()
            if warmup_index == args.warmup:
                reference_outputs[route_key] = output.clone()

        local_actual = [int(value) for value in actual_aligned_counts]
        gathered_actual: list[Any] = [None] * world_size
        dist.all_gather_object(gathered_actual, local_actual, group=group)
        expected = route_manifests[route_key]["expert_aligned_counts"]
        if gathered_actual != expected:
            raise ValueError(
                f"DeepEP aligned expert counts disagree with route manifest: "
                f"expected {expected}, got {gathered_actual}"
            )
        route_manifests[route_key]["deepep_expert_aligned_counts"] = gathered_actual

        if any(
            condition.mode == "strict_barrier" and condition.route_key() == route_key
            for condition in conditions
        ):
            strict_warmup = Condition("strict-warmup", "strict_barrier", 0.0, (0,))
            for _ in range(2):
                gpu_start_gate(start_marker, group)
                output, warmup_events, _ = run_layer_once(
                    dispatcher=dispatcher,
                    runner=runner,
                    quant_info=quant_info,
                    hidden_states=hidden_states,
                    topk_output=topk_output,
                    condition=strict_warmup,
                    rank=rank,
                    group=group,
                    strict_marker=strict_marker,
                    delay=delay,
                    record_events=False,
                )
                warmup_events["end"].synchronize()

    records: list[dict[str, Any]] = []
    started_at = dt.datetime.now(dt.timezone.utc)
    for block in range(args.blocks):
        # Interleave one observation from every condition per cycle. A single
        # condition-major chunk let GPU clock/power drift contaminate all 100
        # repetitions of that condition; independently shuffled cycles spread
        # such drift across conditions while retaining five explicit blocks.
        if rank == 0:
            orders = []
            for repetition in range(args.iterations_per_block):
                order = list(range(len(conditions)))
                random.Random(
                    args.random_seed + block * args.iterations_per_block + repetition
                ).shuffle(order)
                orders.append(order)
        else:
            orders = None
        orders_payload = [orders]
        dist.broadcast_object_list(orders_payload, src=0, group=group)
        orders = orders_payload[0]

        for repetition, order in enumerate(orders):
            for condition_index in order:
                condition = conditions[condition_index]
                profile_capture = args.profile_capture_condition == condition.name
                if profile_capture:
                    # cudaProfilerStart/Stop is a more reliable Nsight capture
                    # trigger than matching a dynamic Python NVTX string.  Keep
                    # the named range as an annotation in the saved report.
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("deadzone_profile_capture")
                gpu_start_gate(start_marker, group)
                route_key = condition.route_key()
                topk_output = route_outputs[route_key]
                output, events, _ = run_layer_once(
                    dispatcher=dispatcher,
                    runner=runner,
                    quant_info=quant_info,
                    hidden_states=hidden_states,
                    topk_output=topk_output,
                    condition=condition,
                    rank=rank,
                    group=group,
                    strict_marker=strict_marker,
                    delay=delay,
                    record_events=True,
                )
                events["end"].synchronize()
                if profile_capture:
                    # Keep every process's outer NVTX range open until all GPU
                    # work has completed, so Nsight does not stop at the first
                    # rank to leave the inner layer range.
                    dist.barrier(group=group)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                correctness = 0.0
                if not args.no_correctness_check:
                    correctness = (
                        output.float() - reference_outputs[route_key].float()
                    ).abs().max().item()
                local_metrics = torch.tensor(
                    event_metrics(events, correctness),
                    dtype=torch.float64,
                    device=device,
                )
                gathered_metrics = [torch.empty_like(local_metrics) for _ in range(world_size)]
                dist.all_gather(gathered_metrics, local_metrics, group=group)
                if rank == 0:
                    rank_records = []
                    for source_rank, tensor in enumerate(gathered_metrics):
                        values = tensor.cpu().tolist()
                        rank_records.append(
                            {"rank": source_rank}
                            | dict(zip(METRIC_NAMES, values, strict=True))
                        )
                    records.append(
                        {
                            "block": block,
                            "condition_order": order,
                            "condition_index": condition_index,
                            "condition": condition.name,
                            "mode": condition.mode,
                            "delay_us": condition.delay_us,
                            "delay_ratio": condition.delay_ratio,
                            "delay_ranks": list(condition.delay_ranks),
                            "route_sha256": route_manifests[route_key]["route_sha256"],
                            "requested_r": condition.requested_r,
                            "repetition": repetition,
                            "ranks": rank_records,
                        }
                    )

    finished_at = dt.datetime.now(dt.timezone.utc)
    buffer = DeepEPBuffer._buffer
    deep_ep_config = DeepEPConfig.get_instance()
    if rank == 0:
        sglang_deepep_path = pathlib.Path(
            "/opt/conda/lib/python3.11/site-packages/sglang/srt/layers/moe/token_dispatcher/deepep.py"
        )
        metadata = {
            "experiment": "deadzone_combine_actual_path",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "total_tokens": args.total_tokens,
            "local_tokens": local_tokens,
            "model_shape": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_experts": NUM_EXPERTS,
                "num_local_experts": LOCAL_EXPERTS,
                "top_k": TOP_K,
                "weight_block": [WEIGHT_BLOCK, WEIGHT_BLOCK],
            },
            "routes": list(route_manifests.values()),
            "condition_routes": {
                condition.name: route_manifests[condition.route_key()]["route_sha256"]
                for condition in conditions
            },
            "execution": {
                "dispatcher": "SGLang _DeepEPDispatcherImplNormal(async_finish=True)",
                "runner": "SGLang DeepGemmRunnerCore contiguous FP8 grouped GEMM",
                "combine": "DeepEP Buffer.combine with previous_event and allocate_on_comm_stream",
                "deep_ep_num_sms": deep_ep_config.num_sms,
                "deep_ep_buffer_num_sms": Buffer.num_sms,
                "expert_alignment": 128,
                "strict_gate": "torch.distributed NCCL all_reduce(MAX) on persistent CUDA marker",
                "delay_instrumentation": (
                    "one deadzone_gpu_delay_ns launch per rank and iteration; "
                    f"all ranks spin for {DELAY_FLOOR_US:g} us and selected ranks "
                    "spin for the same floor plus the requested differential"
                ),
                "delay_floor_us": DELAY_FLOOR_US,
                "condition_schedule": "one independently shuffled all-condition cycle per repetition",
            },
            "software": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "sglang_deepep_sha256": sha256(sglang_deepep_path),
            },
            "measurement": {
                "warmup": args.warmup,
                "blocks": args.blocks,
                "iterations_per_block": args.iterations_per_block,
                "random_seed": args.random_seed,
                "timing": "same-device CUDA events; cross-rank statistic is max elapsed interval",
            },
            "conditions": [condition.as_dict() for condition in conditions],
            "delay_kernel_calibration": delay_calibration,
            "records": records,
            "summary": summarize_records(records),
        }
        output_path = args.output
        if output_path is None:
            root = pathlib.Path(__file__).resolve().parents[1]
            stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
            output_path = root / "results" / f"deadzone_N{args.total_tokens}_{stamp}.json"
        write_json_atomic(output_path, metadata)
        print(f"RESULT_JSON={output_path}", flush=True)
        for name, summary in metadata["summary"].items():
            layer = summary["cross_rank_max"]["layer_ms"]
            correctness = summary["cross_rank_max"]["correctness_max_abs"]["max"]
            print(
                f"{name}: layer median={layer['median']:.6f} ms "
                f"p95={layer['p95']:.6f} ms max_abs_diff={correctness:.6g}",
                flush=True,
            )

    # The SGLang-owned DeepEP buffer is constructed with
    # explicitly_destroy=False, so Buffer.destroy() is intentionally invalid.
    # Match the server lifecycle: drain all ranks, release NCCL groups, then let
    # process teardown release the buffer runtime.
    dist.barrier(group=group)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
