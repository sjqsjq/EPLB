#!/usr/bin/env python3
"""
Benchmark for EPLB expert weight swapping on NVL72.

Measures the time cost of expert weight transfer operations:
  A: Same-GPU memcpy (baseline)
  B: Intra-node P2P (1 expert, NVLink)
  C: Cross-node P2P (1 expert, MNNVL)
  D: Intra-node batch swap (N experts)
  E: Cross-node batch swap (N experts)
  F: Full layer rebalance via update_expert_weights_single_layer()

Launch via run_script.sh:
    ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
        --command "python bench_expert_swap.py --model-preset deepseek_v3"

    ./run_script.sh --master-ip 11.139.21.90 --cur-node 2 \
        --command "python bench_expert_swap.py --model-preset deepseek_v3"
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import P2POp

SGLANG_PYTHON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sglang", "python"
)


@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    dtype_name: str

    @property
    def dtype(self):
        return {
            "fp8": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.dtype_name]

    @property
    def dtype_size(self):
        return {"fp8": 1, "bf16": 2, "fp16": 2}[self.dtype_name]

    def expert_bytes(self):
        w13 = 2 * self.moe_intermediate_size * self.hidden_size * self.dtype_size
        w2 = self.hidden_size * self.moe_intermediate_size * self.dtype_size
        return w13 + w2

    def expert_mb(self):
        return self.expert_bytes() / (1024 * 1024)

    def w13_shape(self, num_local):
        return (num_local, 2 * self.moe_intermediate_size, self.hidden_size)

    def w2_shape(self, num_local):
        return (num_local, self.hidden_size, self.moe_intermediate_size)


MODEL_PRESETS = {
    "deepseek_v3": ModelConfig("DeepSeek-V3", 7168, 2048, 256, "fp8"),
    "qwen3_480b": ModelConfig("Qwen3-480B", 5120, 3072, 128, "fp8"),
    "qwen3_30b": ModelConfig("Qwen3-30B-A3B", 4096, 2560, 128, "fp8"),
}


@dataclass
class BenchResult:
    scenario: str
    num_experts_swapped: int
    data_volume_mb: float
    avg_ms: float
    min_ms: float
    max_ms: float
    bandwidth_gbps: float


# ---------------------------------------------------------------------------
# Timing utility (adapted from deepep_utils.py bench())
# ---------------------------------------------------------------------------


def bench_fn(fn, num_warmups=10, num_tests=30):
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(num_warmups):
        fn()

    cache.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    if len(times) > 1:
        times = times[1:]
    return float(np.mean(times)), float(np.min(times)), float(np.max(times))


def compute_bandwidth(data_bytes, avg_ms):
    if avg_ms <= 0:
        return 0.0
    return (data_bytes / 1e9) / (avg_ms / 1e3)


# ---------------------------------------------------------------------------
# Weight creation
# ---------------------------------------------------------------------------


def create_expert_weights(model, num_local, device="cuda"):
    w13 = torch.randn(model.w13_shape(num_local), dtype=torch.float32, device=device)
    w2 = torch.randn(model.w2_shape(num_local), dtype=torch.float32, device=device)
    if model.dtype == torch.float8_e4m3fn:
        w13 = w13.clamp(-448, 448).to(torch.float8_e4m3fn)
        w2 = w2.clamp(-448, 448).to(torch.float8_e4m3fn)
    else:
        w13 = w13.to(model.dtype)
        w2 = w2.to(model.dtype)
    return [w13, w2]


def create_single_expert_tensors(model, n=1, device="cuda"):
    w13 = torch.randn(n, *model.w13_shape(1)[1:], dtype=torch.float32, device=device)
    w2 = torch.randn(n, *model.w2_shape(1)[1:], dtype=torch.float32, device=device)
    if model.dtype == torch.float8_e4m3fn:
        w13 = w13.clamp(-448, 448).to(torch.float8_e4m3fn)
        w2 = w2.clamp(-448, 448).to(torch.float8_e4m3fn)
    else:
        w13 = w13.to(model.dtype)
        w2 = w2.to(model.dtype)
    return w13, w2


# ---------------------------------------------------------------------------
# Scenario A: Same-GPU copy
# ---------------------------------------------------------------------------


def run_scenario_a(model, num_local, rank, args):
    if rank == 0:
        print("  [A] Running same-GPU copy benchmark...")

    weights = create_expert_weights(model, max(num_local, 2))
    dst_w13 = torch.empty_like(weights[0][0:1])
    dst_w2 = torch.empty_like(weights[1][0:1])

    def fn():
        dst_w13.copy_(weights[0][1:2])
        dst_w2.copy_(weights[1][1:2])

    avg, mn, mx = bench_fn(fn, args.num_warmups, args.num_tests)
    bw = compute_bandwidth(model.expert_bytes(), avg)

    del weights, dst_w13, dst_w2
    torch.cuda.empty_cache()

    result = BenchResult("A: Same-GPU copy", 1, model.expert_mb(), avg, mn, mx, bw)
    if rank == 0:
        print(f"      {avg:.3f} ms, {bw:.1f} GB/s")
    return [result]


# ---------------------------------------------------------------------------
# Scenario B/C: Single-expert P2P
# ---------------------------------------------------------------------------


def run_scenario_p2p(model, rank, world_size, gpus_per_node, cross_node, args):
    if cross_node:
        label = "C: Cross-node P2P (MNNVL)"
        if world_size <= gpus_per_node:
            if rank == 0:
                print("  [C] Skipped: need >= 2 nodes for cross-node test")
            return [BenchResult(label, 1, model.expert_mb(), -1, -1, -1, -1)]
        rank_a, rank_b = 0, gpus_per_node
    else:
        label = "B: Intra-node P2P"
        if world_size < 2:
            if rank == 0:
                print("  [B] Skipped: need >= 2 GPUs")
            return [BenchResult(label, 1, model.expert_mb(), -1, -1, -1, -1)]
        rank_a, rank_b = 0, 1

    if rank == 0:
        print(f"  [{label[0]}] Running {label} (rank {rank_a} <-> rank {rank_b})...")

    is_participant = rank == rank_a or rank == rank_b

    if is_participant:
        peer = rank_b if rank == rank_a else rank_a
        send_w13, send_w2 = create_single_expert_tensors(model, n=1)
        recv_w13 = torch.empty_like(send_w13)
        recv_w2 = torch.empty_like(send_w2)

        def fn():
            ops = [
                P2POp(dist.isend, send_w13, peer),
                P2POp(dist.isend, send_w2, peer),
                P2POp(dist.irecv, recv_w13, peer),
                P2POp(dist.irecv, recv_w2, peer),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

        avg, mn, mx = bench_fn(fn, args.num_warmups, args.num_tests)
        bw = compute_bandwidth(model.expert_bytes(), avg)
        del send_w13, send_w2, recv_w13, recv_w2
        torch.cuda.empty_cache()
    else:
        avg, mn, mx, bw = 0, 0, 0, 0

    dist.barrier()

    result_tensor = torch.tensor([avg, mn, mx, bw], dtype=torch.float64, device="cuda")
    dist.broadcast(result_tensor, src=rank_a)
    avg, mn, mx, bw = result_tensor.tolist()

    if rank == 0:
        print(f"      {avg:.3f} ms, {bw:.1f} GB/s")

    return [BenchResult(label, 1, model.expert_mb(), avg, mn, mx, bw)]


# ---------------------------------------------------------------------------
# Scenario D/E: Batch P2P swap (N experts)
# ---------------------------------------------------------------------------


def run_scenario_batch(model, num_local, rank, world_size, gpus_per_node, cross_node, args):
    if cross_node:
        label_prefix = "E: Cross-node batch"
        tag = "E"
        if world_size <= gpus_per_node:
            if rank == 0:
                print("  [E] Skipped: need >= 2 nodes")
            return []
        rank_a, rank_b = 0, gpus_per_node
    else:
        label_prefix = "D: Intra-node batch"
        tag = "D"
        if world_size < 2:
            if rank == 0:
                print("  [D] Skipped: need >= 2 GPUs")
            return []
        rank_a, rank_b = 0, 1

    if rank == 0:
        print(f"  [{tag}] Running {label_prefix} (rank {rank_a} <-> rank {rank_b})...")

    is_participant = rank == rank_a or rank == rank_b
    results = []

    max_n = min(args.max_swap_experts, num_local)
    n_values = []
    n = 1
    while n <= max_n:
        n_values.append(n)
        n *= 2
    if max_n not in n_values:
        n_values.append(max_n)

    for n_experts in n_values:
        if is_participant:
            peer = rank_b if rank == rank_a else rank_a
            send_w13, send_w2 = create_single_expert_tensors(model, n=n_experts)
            recv_w13 = torch.empty_like(send_w13)
            recv_w2 = torch.empty_like(send_w2)

            def fn(sw13=send_w13, sw2=send_w2, rw13=recv_w13, rw2=recv_w2, p=peer, ne=n_experts):
                ops = []
                for i in range(ne):
                    ops.append(P2POp(dist.isend, sw13[i : i + 1].contiguous(), p))
                    ops.append(P2POp(dist.isend, sw2[i : i + 1].contiguous(), p))
                    ops.append(P2POp(dist.irecv, rw13[i : i + 1].contiguous(), p))
                    ops.append(P2POp(dist.irecv, rw2[i : i + 1].contiguous(), p))
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()

            avg, mn, mx = bench_fn(fn, args.num_warmups, args.num_tests)
            vol = model.expert_mb() * n_experts
            bw = compute_bandwidth(model.expert_bytes() * n_experts, avg)
            del send_w13, send_w2, recv_w13, recv_w2
            torch.cuda.empty_cache()
        else:
            avg, mn, mx, bw, vol = 0, 0, 0, 0, 0

        dist.barrier()

        result_tensor = torch.tensor(
            [avg, mn, mx, bw, model.expert_mb() * n_experts],
            dtype=torch.float64,
            device="cuda",
        )
        dist.broadcast(result_tensor, src=rank_a)
        avg, mn, mx, bw, vol = result_tensor.tolist()

        results.append(
            BenchResult(f"{label_prefix} (N={n_experts})", n_experts, vol, avg, mn, mx, bw)
        )
        if rank == 0:
            print(f"      N={n_experts:>3}: {avg:.3f} ms, {bw:.1f} GB/s")

    return results


# ---------------------------------------------------------------------------
# Scenario F: Full layer rebalance via update_expert_weights_single_layer
# ---------------------------------------------------------------------------


def create_swap_maps(num_physical_experts, num_local_experts, world_size, swap_fraction):
    old_map = list(range(num_physical_experts))
    new_map = old_map.copy()

    num_to_swap = max(1, int(num_local_experts * swap_fraction))

    for gpu in range(0, world_size, 2):
        if gpu + 1 >= world_size:
            break
        for j in range(num_to_swap):
            slot_a = gpu * num_local_experts + j
            slot_b = (gpu + 1) * num_local_experts + j
            if slot_a < num_physical_experts and slot_b < num_physical_experts:
                new_map[slot_a], new_map[slot_b] = new_map[slot_b], new_map[slot_a]

    return old_map, new_map


def run_scenario_f(model, num_local, rank, world_size, gpus_per_node, args):
    if rank == 0:
        print("  [F] Running full layer rebalance benchmark...")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _standalone_updater import (
        update_expert_weights_single_layer,
        create_temp_buffers,
    )

    num_physical = num_local * world_size
    results = []

    for frac_label, frac in [("10%", 0.1), ("50%", 0.5), ("100%", 1.0)]:
        weights = create_expert_weights(model, num_local)
        temp_bufs = create_temp_buffers(weights)

        old_map, new_map = create_swap_maps(num_physical, num_local, world_size, frac)

        for _ in range(min(5, args.num_warmups)):
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=temp_bufs,
                old_physical_to_logical_map=old_map,
                new_physical_to_logical_map=new_map,
                num_local_physical_experts=num_local,
                num_gpu_per_node=gpus_per_node,
                rank=rank,
                world_size=world_size,
            )
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=temp_bufs,
                old_physical_to_logical_map=new_map,
                new_physical_to_logical_map=old_map,
                num_local_physical_experts=num_local,
                num_gpu_per_node=gpus_per_node,
                rank=rank,
                world_size=world_size,
            )

        torch.cuda.synchronize()
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
        cache.zero_()

        num_tests = args.num_tests
        times = []
        use_forward = True
        for _ in range(num_tests):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if use_forward:
                update_expert_weights_single_layer(
                    routed_experts_weights=weights,
                    temp_buffers=temp_bufs,
                    old_physical_to_logical_map=old_map,
                    new_physical_to_logical_map=new_map,
                    num_local_physical_experts=num_local,
                    num_gpu_per_node=gpus_per_node,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                update_expert_weights_single_layer(
                    routed_experts_weights=weights,
                    temp_buffers=temp_bufs,
                    old_physical_to_logical_map=new_map,
                    new_physical_to_logical_map=old_map,
                    num_local_physical_experts=num_local,
                    num_gpu_per_node=gpus_per_node,
                    rank=rank,
                    world_size=world_size,
                )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            use_forward = not use_forward

        times_arr = np.array(times)
        if len(times_arr) > 1:
            times_arr = times_arr[1:]
        avg = float(np.mean(times_arr))
        mn = float(np.min(times_arr))
        mx = float(np.max(times_arr))

        num_swapped = max(1, int(num_local * frac))
        num_pairs = world_size // 2
        total_experts_moved = num_swapped * num_pairs * 2
        vol = model.expert_mb() * total_experts_moved
        bw = compute_bandwidth(model.expert_bytes() * total_experts_moved, avg)

        del weights, temp_bufs, cache
        torch.cuda.empty_cache()

        results.append(
            BenchResult(
                f"F: Full rebalance ({frac_label})",
                total_experts_moved,
                vol,
                avg,
                mn,
                mx,
                bw,
            )
        )
        if rank == 0:
            print(f"      {frac_label} change: {avg:.3f} ms, {total_experts_moved} experts moved")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results(results, model, world_size, nnodes, gpus_per_node):
    w13_mb = 2 * model.moe_intermediate_size * model.hidden_size * model.dtype_size / (1024 * 1024)
    w2_mb = model.hidden_size * model.moe_intermediate_size * model.dtype_size / (1024 * 1024)

    print()
    print("=" * 95)
    print("Expert Weight Swap Benchmark - NVL72")
    print(
        f"Model: {model.name} | dtype: {model.dtype_name} | "
        f"GPUs: {world_size} ({nnodes} nodes x {gpus_per_node})"
    )
    print(f"Expert size: ~{model.expert_mb():.1f} MB (w13: {w13_mb:.1f} MB + w2: {w2_mb:.1f} MB)")
    print("=" * 95)

    header = (
        f"{'Scenario':<35} | {'N_exp':>5} | {'Volume(MB)':>10} | "
        f"{'Avg(ms)':>8} | {'Min(ms)':>8} | {'Max(ms)':>8} | {'BW(GB/s)':>8}"
    )
    print(header)
    print("-" * 95)

    for r in results:
        if r.avg_ms < 0:
            print(
                f"{r.scenario:<35} | {'N/A':>5} | {'N/A':>10} | "
                f"{'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
            )
        else:
            print(
                f"{r.scenario:<35} | {r.num_experts_swapped:>5} | "
                f"{r.data_volume_mb:>10.1f} | {r.avg_ms:>8.3f} | "
                f"{r.min_ms:>8.3f} | {r.max_ms:>8.3f} | {r.bandwidth_gbps:>8.1f}"
            )

    print("=" * 95)


def save_results(results, output_path, model, world_size, nnodes, gpus_per_node):
    data = {
        "model": model.name,
        "dtype": model.dtype_name,
        "hidden_size": model.hidden_size,
        "moe_intermediate_size": model.moe_intermediate_size,
        "n_routed_experts": model.n_routed_experts,
        "expert_mb": model.expert_mb(),
        "world_size": world_size,
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
        "results": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Worker function (one per GPU, spawned by mp.spawn)
# ---------------------------------------------------------------------------


def worker(local_rank, args):
    node_rank = int(os.environ.get("RANK", "0"))
    nnodes = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12345")

    gpus_per_node = args.gpus_per_node
    global_rank = node_rank * gpus_per_node + local_rank
    global_world_size = nnodes * gpus_per_node

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=global_world_size,
        rank=global_rank,
    )
    torch.cuda.set_device(local_rank)

    if args.model_preset != "custom":
        preset = MODEL_PRESETS[args.model_preset]
        model = ModelConfig(
            preset.name,
            preset.hidden_size,
            preset.moe_intermediate_size,
            preset.n_routed_experts,
            preset.dtype_name,
        )
    else:
        model = ModelConfig(
            "custom",
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            args.dtype or "fp8",
        )
    if args.dtype:
        model.dtype_name = args.dtype

    num_local = args.num_local_experts
    if num_local is None:
        num_local = max(1, model.n_routed_experts // global_world_size)

    if global_rank == 0:
        print()
        print("=" * 60)
        print("Expert Weight Swap Benchmark - NVL72")
        print("=" * 60)
        print(f"  Model:           {model.name}")
        print(f"  Hidden size:     {model.hidden_size}")
        print(f"  Intermediate:    {model.moe_intermediate_size}")
        print(f"  Routed experts:  {model.n_routed_experts}")
        print(f"  Dtype:           {model.dtype_name}")
        print(f"  Experts per GPU: {num_local}")
        print(f"  GPUs:            {global_world_size} ({nnodes} nodes x {gpus_per_node})")
        print(f"  Expert size:     {model.expert_mb():.1f} MB")
        print(f"  Scenarios:       {args.scenarios}")
        print(f"  Warmups:         {args.num_warmups}")
        print(f"  Test iters:      {args.num_tests}")
        print()

    scenarios = [s.lower() for s in args.scenarios]
    if "all" in scenarios:
        scenarios = ["a", "b", "c", "d", "e", "f"]

    all_results = []

    if "a" in scenarios:
        dist.barrier()
        all_results.extend(run_scenario_a(model, num_local, global_rank, args))

    if "b" in scenarios:
        dist.barrier()
        all_results.extend(
            run_scenario_p2p(model, global_rank, global_world_size, gpus_per_node, False, args)
        )

    if "c" in scenarios:
        dist.barrier()
        all_results.extend(
            run_scenario_p2p(model, global_rank, global_world_size, gpus_per_node, True, args)
        )

    if "d" in scenarios:
        dist.barrier()
        all_results.extend(
            run_scenario_batch(
                model, num_local, global_rank, global_world_size, gpus_per_node, False, args
            )
        )

    if "e" in scenarios:
        dist.barrier()
        all_results.extend(
            run_scenario_batch(
                model, num_local, global_rank, global_world_size, gpus_per_node, True, args
            )
        )

    if "f" in scenarios:
        dist.barrier()
        all_results.extend(
            run_scenario_f(model, num_local, global_rank, global_world_size, gpus_per_node, args)
        )

    if global_rank == 0:
        print_results(all_results, model, global_world_size, nnodes, gpus_per_node)
        save_results(all_results, args.output, model, global_world_size, nnodes, gpus_per_node)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EPLB expert weight swapping on NVL72"
    )

    parser.add_argument(
        "--model-preset",
        choices=list(MODEL_PRESETS.keys()) + ["custom"],
        default="deepseek_v3",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=2048)
    parser.add_argument("--n-routed-experts", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["fp8", "bf16", "fp16"],
        default=None,
        help="Override dtype (default: use model preset)",
    )

    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument(
        "--num-local-experts",
        type=int,
        default=None,
        help="Experts per GPU (default: n_routed_experts / world_size)",
    )
    parser.add_argument("--max-swap-experts", type=int, default=16)

    parser.add_argument("--num-warmups", type=int, default=10)
    parser.add_argument("--num-tests", type=int, default=30)

    parser.add_argument("--output", type=str, default="result/bench_expert_swap.json")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["all"],
        help="Scenarios to run: a b c d e f or all",
    )

    args = parser.parse_args()

    mp.spawn(worker, args=(args,), nprocs=args.gpus_per_node)


if __name__ == "__main__":
    main()
