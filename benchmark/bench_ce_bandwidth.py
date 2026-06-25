#!/usr/bin/env python3
"""
Benchmark CE (Copy Engine) DMA bandwidth on NVL72.

Key difference from bench_expert_swap.py:
  - Uses tensor.copy_() which triggers cudaMemcpyPeerAsync (CE DMA)
  - NOT using NCCL dist.isend/irecv (which uses SM + different routing)
  
This measures the true CE bandwidth that will be used for expert weight
replication in our EPLB pipeline.

Launch via run_script.sh:
    ./run_script.sh --master-ip <IP> --cur-node 2 \
        --command "python bench_ce_bandwidth.py --model-preset qwen3_30b"
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


MODEL_PRESETS = {
    "deepseek_v3": ModelConfig("DeepSeek-V3", 7168, 2048, 256, "fp8"),
    "qwen3_480b": ModelConfig("Qwen3-480B", 5120, 3072, 128, "fp8"),
    "qwen3_30b": ModelConfig("Qwen3-30B-A3B", 4096, 2560, 128, "fp8"),
}


def bench_fn(fn, num_warmups=10, num_tests=30):
    """Benchmark with CUDA events timing."""
    torch.cuda.synchronize()
    # L2 cache flush
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(num_warmups):
        fn()

    cache.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        cache.zero_()  # flush L2 between iterations
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    if len(times) > 1:
        times = times[1:]  # drop first
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    return avg, mn, mx


def worker(local_rank, args):
    node_rank = int(os.environ.get("RANK", "0"))
    nnodes = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12345")

    gpus_per_node = args.gpus_per_node
    global_rank = node_rank * gpus_per_node + local_rank
    world_size = nnodes * gpus_per_node

    # Init process group (needed for barrier synchronization only)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank,
    )
    torch.cuda.set_device(local_rank)

    model = MODEL_PRESETS[args.model_preset]
    expert_bytes = model.expert_bytes()
    expert_mb = model.expert_mb()

    if global_rank == 0:
        print()
        print("=" * 70)
        print("CE (Copy Engine) DMA Bandwidth Benchmark - NVL72")
        print("=" * 70)
        print(f"  Model:        {model.name}")
        print(f"  Expert size:  {expert_mb:.1f} MB")
        print(f"  GPUs:         {world_size} ({nnodes} nodes × {gpus_per_node})")
        print(f"  Method:       tensor.copy_() → cudaMemcpyPeerAsync (CE DMA)")
        print()

    results = []

    # ═══════════════════════════════════════════════════════════════
    # Test 1: Same-GPU copy (baseline, HBM bandwidth)
    # ═══════════════════════════════════════════════════════════════
    if global_rank == 0:
        print("[A] Same-GPU copy (HBM bandwidth baseline)")
    
    src = torch.randn(expert_bytes // 4, dtype=torch.float32, device="cuda")
    src = src.view(torch.uint8)[:expert_bytes]
    dst = torch.empty_like(src)

    def fn_same_gpu():
        dst.copy_(src)

    avg, mn, mx = bench_fn(fn_same_gpu, args.num_warmups, args.num_tests)
    bw = (expert_bytes / 1e9) / (avg / 1e3)
    if global_rank == 0:
        print(f"    {avg:.3f} ms, {bw:.1f} GB/s")
    results.append({
        "scenario": "A: Same-GPU copy",
        "src_rank": global_rank, "dst_rank": global_rank,
        "same_node": True,
        "expert_mb": expert_mb, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
        "bandwidth_gbps": bw,
    })
    del src, dst
    torch.cuda.empty_cache()
    dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # Test 2: CE P2P between all interesting GPU pairs
    # ═══════════════════════════════════════════════════════════════
    
    # Define pairs to test
    # For NVL72: test within same tray (4 GPUs) and across trays
    pairs = []
    
    if world_size >= 2:
        # Same-tray pairs
        pairs.append((0, 1, "same-tray (rank0↔rank1)"))
        if gpus_per_node >= 4:
            pairs.append((0, 3, "same-tray (rank0↔rank3)"))
    
    if world_size > gpus_per_node:
        # Cross-tray pairs (different "nodes" as defined by gpus_per_node)
        pairs.append((0, gpus_per_node, f"cross-tray (rank0↔rank{gpus_per_node})"))
        if world_size > 2 * gpus_per_node:
            far_rank = 2 * gpus_per_node
            pairs.append((0, far_rank, f"cross-tray-far (rank0↔rank{far_rank})"))
    
    for rank_a, rank_b, label in pairs:
        if rank_a >= world_size or rank_b >= world_size:
            continue
        
        is_a = (global_rank == rank_a)
        is_b = (global_rank == rank_b)
        is_participant = is_a or is_b

        if global_rank == 0:
            node_a = rank_a // gpus_per_node
            node_b = rank_b // gpus_per_node
            same = "YES" if node_a == node_b else "NO"
            print(f"\n[CE] {label}")
            print(f"     node(rank{rank_a})={node_a}, node(rank{rank_b})={node_b}, same_tray={same}")

        if is_participant:
            peer_rank = rank_b if is_a else rank_a
            peer_local = peer_rank % gpus_per_node
            peer_device = torch.device(f"cuda:{peer_local}")
            my_device = torch.device(f"cuda:{local_rank}")

            # Enable P2P access
            if torch.cuda.can_access_peer(local_rank, peer_local):
                # Note: on NVL72 with NVSwitch, P2P should work across all GPUs
                pass  # P2P is typically enabled by default with NVSwitch

            # Allocate on peer device, then copy to local
            # Method: use torch.cuda.set_device to allocate on remote, then copy_
            
            # For CE P2P, we need tensors on different GPUs
            # rank_a creates src tensor, rank_b creates dst tensor
            # rank_a does: dst_on_a.copy_(src_on_b)  ← this triggers CE P2P

            # Each participant allocates their local tensor
            local_tensor = torch.randn(
                expert_bytes // 4, dtype=torch.float32, device=my_device
            ).view(torch.uint8)[:expert_bytes]
            
            # We'll use NCCL send/recv just to set up the remote reference,
            # but the actual benchmark uses copy_()
            
            # Simpler approach: both ranks allocate, rank_a copies from rank_b
            # But copy_() across ranks requires the tensors to be on accessible devices
            
            # On NVL72 with NVSwitch, we can use cudaMemcpyPeer directly
            # In PyTorch, this is: dst.copy_(src) where src and dst are on different GPUs
            # But both must be visible to the same process
            
            # Since we use mp.spawn with gpus_per_node processes per node,
            # each process only sees 1 GPU. For cross-GPU copy_, we need
            # both GPUs visible to the same process.
            
            # Workaround: use dist.send/recv for setup, measure with CUDA events
            # OR: restructure to single-process multi-GPU
            
            del local_tensor
        
        dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # Test 2 (revised): Single-process multi-GPU CE benchmark
    # Only rank 0 runs this, accessing multiple GPUs
    # ═══════════════════════════════════════════════════════════════
    
    if global_rank == 0:
        print("\n" + "=" * 70)
        print("CE DMA P2P: Single-process multi-GPU test (rank 0 only)")
        print("=" * 70)
        
        # On NVL72, rank 0 (local_rank 0) can access other GPUs in the same 
        # tray via P2P. For cross-tray, we need cudaMemcpyPeer.
        
        num_local_gpus = min(torch.cuda.device_count(), gpus_per_node)
        print(f"  Visible GPUs from rank 0: {num_local_gpus}")
        
        # Test all local GPU pairs
        for gpu_b in range(1, num_local_gpus):
            # Check P2P access
            can_p2p = torch.cuda.can_access_peer(0, gpu_b)
            print(f"\n  [CE] GPU 0 → GPU {gpu_b} (P2P accessible: {can_p2p})")
            
            if not can_p2p:
                print(f"       Skipped: no P2P access")
                continue
            
            # Allocate src on GPU 0, dst on GPU gpu_b
            src = torch.randn(expert_bytes // 4, dtype=torch.float32, device="cuda:0")
            src = src.view(torch.uint8)[:expert_bytes]
            
            dst = torch.empty(expert_bytes, dtype=torch.uint8, device=f"cuda:{gpu_b}")
            
            def fn_p2p(s=src, d=dst):
                d.copy_(s)
            
            avg, mn, mx = bench_fn(fn_p2p, args.num_warmups, args.num_tests)
            bw = (expert_bytes / 1e9) / (avg / 1e3)
            print(f"       {avg:.3f} ms, {bw:.1f} GB/s")
            
            results.append({
                "scenario": f"CE P2P: GPU 0 → GPU {gpu_b}",
                "src_rank": 0, "dst_rank": gpu_b,
                "same_node": True,
                "expert_mb": expert_mb, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
                "bandwidth_gbps": bw,
            })
            
            del src, dst
            torch.cuda.empty_cache()
        
        # Test bidirectional (concurrent)
        if num_local_gpus >= 2:
            print(f"\n  [CE] Bidirectional: GPU 0 ↔ GPU 1")
            
            src_0 = torch.randn(expert_bytes // 4, dtype=torch.float32, device="cuda:0").view(torch.uint8)[:expert_bytes]
            src_1 = torch.randn(expert_bytes // 4, dtype=torch.float32, device="cuda:1").view(torch.uint8)[:expert_bytes]
            dst_0 = torch.empty(expert_bytes, dtype=torch.uint8, device="cuda:0")
            dst_1 = torch.empty(expert_bytes, dtype=torch.uint8, device="cuda:1")
            
            def fn_bidi(s0=src_0, s1=src_1, d0=dst_0, d1=dst_1):
                # Both directions simultaneously
                d1.copy_(s0)  # GPU 0 → GPU 1
                d0.copy_(s1)  # GPU 1 → GPU 0
            
            avg, mn, mx = bench_fn(fn_bidi, args.num_warmups, args.num_tests)
            bw = (2 * expert_bytes / 1e9) / (avg / 1e3)  # total BW (both dirs)
            bw_per_dir = (expert_bytes / 1e9) / (avg / 1e3)
            print(f"       {avg:.3f} ms, {bw:.1f} GB/s total, {bw_per_dir:.1f} GB/s per direction")
            
            results.append({
                "scenario": f"CE P2P Bidi: GPU 0 ↔ GPU 1",
                "src_rank": 0, "dst_rank": 1,
                "same_node": True,
                "expert_mb": 2 * expert_mb, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
                "bandwidth_gbps": bw,
            })
            
            del src_0, src_1, dst_0, dst_1
            torch.cuda.empty_cache()
        
        # Test multi-source to single destination
        if num_local_gpus >= 3:
            for n_sources in [2, 3]:
                if n_sources >= num_local_gpus:
                    continue
                print(f"\n  [CE] {n_sources} sources → GPU 0 (concurrent)")
                
                sources = []
                dsts = []
                for i in range(1, n_sources + 1):
                    s = torch.randn(expert_bytes // 4, dtype=torch.float32, device=f"cuda:{i}").view(torch.uint8)[:expert_bytes]
                    d = torch.empty(expert_bytes, dtype=torch.uint8, device="cuda:0")
                    sources.append(s)
                    dsts.append(d)
                
                def fn_multi(srcs=sources, ds=dsts):
                    for s, d in zip(srcs, ds):
                        d.copy_(s)
                
                avg, mn, mx = bench_fn(fn_multi, args.num_warmups, args.num_tests)
                total_bytes = n_sources * expert_bytes
                bw = (total_bytes / 1e9) / (avg / 1e3)
                bw_each = (expert_bytes / 1e9) / (avg / 1e3)
                print(f"       {avg:.3f} ms, {bw:.1f} GB/s total, {bw_each:.1f} GB/s per source")
                
                results.append({
                    "scenario": f"CE P2P: {n_sources}→1 concurrent",
                    "src_rank": -1, "dst_rank": 0,
                    "same_node": True,
                    "expert_mb": n_sources * expert_mb,
                    "avg_ms": avg, "min_ms": mn, "max_ms": mx,
                    "bandwidth_gbps": bw,
                })
                
                del sources, dsts
                torch.cuda.empty_cache()

    dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # Test 3: NCCL P2P (for comparison with CE)
    # ═══════════════════════════════════════════════════════════════
    
    nccl_pairs = []
    if world_size >= 2:
        nccl_pairs.append((0, 1, "same-tray"))
    if world_size > gpus_per_node:
        nccl_pairs.append((0, gpus_per_node, "cross-tray"))
    if world_size > 2 * gpus_per_node:
        nccl_pairs.append((0, 2 * gpus_per_node, "cross-tray-far"))
    
    for rank_a, rank_b, label in nccl_pairs:
        if rank_a >= world_size or rank_b >= world_size:
            continue
        
        if global_rank == 0:
            print(f"\n[NCCL P2P] {label}: rank {rank_a} ↔ rank {rank_b}")
        
        is_a = (global_rank == rank_a)
        is_b = (global_rank == rank_b)
        
        if is_a or is_b:
            peer = rank_b if is_a else rank_a
            send_buf = torch.randn(expert_bytes // 4, dtype=torch.float32, device="cuda").view(torch.uint8)[:expert_bytes]
            recv_buf = torch.empty_like(send_buf)
            
            def fn_nccl(sb=send_buf, rb=recv_buf, p=peer):
                ops = [
                    dist.P2POp(dist.isend, sb, p),
                    dist.P2POp(dist.irecv, rb, p),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
            
            avg, mn, mx = bench_fn(fn_nccl, args.num_warmups, args.num_tests)
            bw = (expert_bytes / 1e9) / (avg / 1e3)
            
            del send_buf, recv_buf
            torch.cuda.empty_cache()
        else:
            avg, mn, mx, bw = 0, 0, 0, 0
        
        dist.barrier()
        
        # Broadcast results from rank_a
        result_tensor = torch.tensor([avg, mn, mx, bw], dtype=torch.float64, device="cuda")
        dist.broadcast(result_tensor, src=rank_a)
        avg, mn, mx, bw = result_tensor.tolist()
        
        if global_rank == 0:
            print(f"       {avg:.3f} ms, {bw:.1f} GB/s")
        
        results.append({
            "scenario": f"NCCL P2P: {label}",
            "src_rank": rank_a, "dst_rank": rank_b,
            "same_node": rank_a // gpus_per_node == rank_b // gpus_per_node,
            "expert_mb": expert_mb, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
            "bandwidth_gbps": bw,
        })

    dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    if global_rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<40} {'Avg(ms)':>8} {'BW(GB/s)':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r['scenario']:<40} {r['avg_ms']:>8.3f} {r['bandwidth_gbps']:>10.1f}")
        print("=" * 70)
        
        # Save
        output = {
            "model": model.name,
            "expert_mb": expert_mb,
            "world_size": world_size,
            "nnodes": nnodes,
            "gpus_per_node": gpus_per_node,
            "results": results,
        }
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="CE DMA Bandwidth Benchmark for NVL72")
    parser.add_argument("--model-preset", choices=list(MODEL_PRESETS.keys()), default="qwen3_30b")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-warmups", type=int, default=10)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--output", type=str, default="result/bench_ce_bandwidth.json")
    args = parser.parse_args()
    
    mp.spawn(worker, args=(args,), nprocs=args.gpus_per_node)


if __name__ == "__main__":
    main()
