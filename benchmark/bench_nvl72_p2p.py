#!/usr/bin/env python3
"""
NVL72 P2P Bandwidth Benchmark

Tests whether NVL72's NVSwitch full-mesh topology provides uniform bandwidth
across all GPU pairs, or if there is an intra-tray vs cross-tray difference.

Four test phases:
  Phase 0: Topology discovery (nvidia-smi topo, NVLink status, NCCL env)
  Phase 1: CE DMA bandwidth (intra-tray only, tensor.copy_)
  Phase 2: NCCL P2P sweep (all GPU pair categories)
  Phase 3: Data size sweep (bandwidth vs transfer size)

Launch:
    ./run_script.sh --master-ip 90 --cur-node 2 \
        --command "python bench_nvl72_p2p.py"
"""

import argparse
import json
import os
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


GPUS_PER_TRAY = 4


# ─── Topology Info (Phase 0) ────────────────────────────────────

def collect_topology():
    info = {}

    for cmd_name, cmd in [
        ("nvidia_smi_topo", "nvidia-smi topo -m"),
        ("nvlink_status", "nvidia-smi nvlink -s"),
    ]:
        try:
            r = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            info[cmd_name] = r.stdout
            print(f"\n{'='*70}")
            print(f"  {cmd}")
            print(f"{'='*70}")
            print(r.stdout)
        except Exception as e:
            info[cmd_name] = f"Error: {e}"

    nccl_vars = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    info["nccl_env"] = nccl_vars
    print(f"\nNCCL env: {json.dumps(nccl_vars, indent=2)}")
    return info


# ─── CE DMA Tests (Phase 1) ────────────────────────────────────

def run_ce_dma_tests(expert_bytes, warmup, iters):
    """CE DMA between all intra-tray GPU pairs. Runs BEFORE mp.spawn."""
    n_gpus = min(torch.cuda.device_count(), GPUS_PER_TRAY)
    print(f"\n{'='*70}")
    print(f"  Phase 1: CE DMA (tensor.copy_) — {n_gpus} local GPUs")
    print(f"{'='*70}")

    num_elements = expert_bytes // 2  # bfloat16
    results = []

    for src_dev in range(n_gpus):
        for dst_dev in range(src_dev + 1, n_gpus):
            src = torch.randn(num_elements, dtype=torch.bfloat16,
                              device=f"cuda:{src_dev}")
            dst = torch.empty(num_elements, dtype=torch.bfloat16,
                              device=f"cuda:{dst_dev}")

            for _ in range(warmup):
                dst.copy_(src)
                torch.cuda.synchronize(dst_dev)

            times = []
            for _ in range(iters):
                torch.cuda.synchronize(dst_dev)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(dst_dev))
                dst.copy_(src)
                end.record(torch.cuda.current_stream(dst_dev))
                torch.cuda.synchronize(dst_dev)
                times.append(start.elapsed_time(end))

            avg_ms = statistics.mean(times)
            bw = (expert_bytes / 1e9) / (avg_ms / 1e3) if avg_ms > 0 else 0

            results.append({
                "src": src_dev, "dst": dst_dev,
                "avg_ms": round(avg_ms, 3),
                "bw_gbps": round(bw, 1),
            })
            print(f"  GPU {src_dev} → GPU {dst_dev}: {avg_ms:.3f} ms, {bw:.1f} GB/s")
            del src, dst
            torch.cuda.empty_cache()

    if results:
        avg_bw = statistics.mean(r["bw_gbps"] for r in results)
        print(f"  CE DMA Average: {avg_bw:.1f} GB/s")

    return results


# ─── NCCL P2P Measurement Core ─────────────────────────────────

def measure_nccl_p2p(global_rank, src_rank, dst_rank,
                     num_bytes, warmup, iters):
    """Unidirectional NCCL P2P: src → dst. Returns dict on participants."""
    num_elements = num_bytes // 2
    tensor = torch.empty(num_elements, dtype=torch.bfloat16, device="cuda")
    if global_rank == src_rank:
        tensor.normal_()

    is_src = (global_rank == src_rank)
    is_dst = (global_rank == dst_rank)

    for _ in range(warmup):
        if is_src:
            dist.isend(tensor, dst_rank).wait()
        elif is_dst:
            dist.irecv(tensor, src_rank).wait()
        dist.barrier()

    times_ms = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        if is_src:
            dist.isend(tensor, dst_rank).wait()
        elif is_dst:
            dist.irecv(tensor, src_rank).wait()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if is_src or is_dst:
            times_ms.append((t1 - t0) * 1000)
        dist.barrier()

    del tensor
    if (is_src or is_dst) and times_ms:
        avg_ms = statistics.mean(times_ms)
        bw = (num_bytes / 1e9) / (avg_ms / 1e3) if avg_ms > 0 else 0
        return {"avg_ms": round(avg_ms, 3), "bw_gbps": round(bw, 1)}
    return None


def gather_to_rank0(global_rank, src_rank, measurement):
    """Send measurement from src_rank to rank 0."""
    if global_rank == src_rank and src_rank != 0:
        vals = [measurement["avg_ms"], measurement["bw_gbps"]]
        t = torch.tensor(vals, dtype=torch.float64, device="cuda")
        dist.send(t, 0)
        return measurement
    elif global_rank == 0 and src_rank != 0:
        t = torch.empty(2, dtype=torch.float64, device="cuda")
        dist.recv(t, src_rank)
        return {"avg_ms": round(t[0].item(), 3), "bw_gbps": round(t[1].item(), 1)}
    elif global_rank == 0 and src_rank == 0:
        return measurement
    return None


# ─── GPU Pair Generation ───────────────────────────────────────

def generate_pairs(world_size):
    """GPU pairs grouped by topology category."""
    g = GPUS_PER_TRAY
    pairs = {}

    pairs["intra_tray"] = [(i, j) for i in range(g) for j in range(i+1, g)]

    cross_1 = []
    if world_size > g:
        cross_1 = [(0, g), (1, g+1), (2, g+2), (3, g+3),
                    (0, g+1), (1, g), (0, g+3)]
    pairs["cross_1hop"] = [(a, b) for a, b in cross_1
                           if a < world_size and b < world_size]

    cross_2 = []
    if world_size > 2 * g:
        cross_2 = [(0, 2*g), (1, 2*g+1), (3, 2*g+3)]
    pairs["cross_2hop"] = [(a, b) for a, b in cross_2
                           if a < world_size and b < world_size]

    cross_far = []
    for tray_idx in range(3, world_size // g):
        base = tray_idx * g
        cross_far.append((0, base))
        if base + 3 < world_size:
            cross_far.append((3, base + 3))
    pairs["cross_far"] = [(a, b) for a, b in cross_far
                          if a < world_size and b < world_size]

    return {k: v for k, v in pairs.items() if v}


# ─── NCCL P2P Sweep (Phase 2) ──────────────────────────────────

def run_nccl_sweep(global_rank, world_size, expert_bytes, warmup, iters):
    pair_groups = generate_pairs(world_size)

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  Phase 2: NCCL P2P Sweep ({expert_bytes/(1024*1024):.0f} MB)")
        print(f"{'='*70}")

    all_results = {}

    for category, pairs in pair_groups.items():
        cat_results = []
        if global_rank == 0:
            print(f"\n  [{category}]")

        for src, dst in pairs:
            m = measure_nccl_p2p(global_rank, src, dst,
                                 expert_bytes, warmup, iters)

            dist.barrier()

            result = gather_to_rank0(global_rank, src, m)
            dist.barrier()

            if global_rank == 0 and result:
                src_tray = src // GPUS_PER_TRAY
                dst_tray = dst // GPUS_PER_TRAY
                cat_results.append({
                    "src": src, "dst": dst,
                    "src_tray": src_tray, "dst_tray": dst_tray,
                    **result,
                })
                same = "same" if src_tray == dst_tray else "diff"
                print(f"    GPU {src:>2} → GPU {dst:>2}  "
                      f"(tray {src_tray}→{dst_tray}, {same}): "
                      f"{result['avg_ms']:.3f} ms, {result['bw_gbps']:.1f} GB/s")

        all_results[category] = cat_results

    if global_rank == 0:
        for cat, res in all_results.items():
            if res:
                avg_bw = statistics.mean(r["bw_gbps"] for r in res)
                print(f"\n  {cat} average: {avg_bw:.1f} GB/s")

    return all_results


# ─── Data Size Sweep (Phase 3) ──────────────────────────────────

def run_size_sweep(global_rank, world_size, warmup, iters):
    sizes_mb = [0.1, 0.5, 1, 5, 10, 31, 50, 100, 200]
    intra_pair = (0, 1)
    cross_pair = (0, GPUS_PER_TRAY) if world_size > GPUS_PER_TRAY else None

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  Phase 3: Data Size Sweep")
        print(f"{'='*70}")
        print(f"  {'Size(MB)':>10}  {'Intra(GB/s)':>12}", end="")
        if cross_pair:
            print(f"  {'Cross(GB/s)':>12}  {'Ratio':>7}", end="")
        print()
        print(f"  {'-'*50}")

    results = {"intra": [], "cross": []}

    for size_mb in sizes_mb:
        num_bytes = int(size_mb * 1024 * 1024)

        m_intra = measure_nccl_p2p(global_rank, *intra_pair,
                                    num_bytes, warmup, iters)
        dist.barrier()
        r_intra = gather_to_rank0(global_rank, intra_pair[0], m_intra)
        dist.barrier()

        r_cross = None
        if cross_pair:
            m_cross = measure_nccl_p2p(global_rank, *cross_pair,
                                        num_bytes, warmup, iters)
            dist.barrier()
            r_cross = gather_to_rank0(global_rank, cross_pair[0], m_cross)
            dist.barrier()

        if global_rank == 0:
            intra_bw = r_intra["bw_gbps"] if r_intra else 0
            results["intra"].append({"size_mb": size_mb, "bw_gbps": intra_bw})

            line = f"  {size_mb:>10.1f}  {intra_bw:>12.1f}"
            if r_cross:
                cross_bw = r_cross["bw_gbps"]
                ratio = intra_bw / cross_bw if cross_bw > 0 else 0
                results["cross"].append({"size_mb": size_mb, "bw_gbps": cross_bw})
                line += f"  {cross_bw:>12.1f}  {ratio:>6.2f}x"
            print(line)

    return results


# ─── Worker (mp.spawn entry) ───────────────────────────────────

def worker(local_rank, args, ce_results, topo_info):
    node_rank = int(os.environ.get("RANK", "0"))
    nnodes = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12345")

    global_rank = node_rank * GPUS_PER_TRAY + local_rank
    world_size = nnodes * GPUS_PER_TRAY

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank,
    )

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  NVL72 P2P Bandwidth Benchmark")
        print(f"  {world_size} GPUs ({nnodes} trays × {GPUS_PER_TRAY} GPUs)")
        print(f"  Expert size: {args.expert_size_mb:.0f} MB")
        print(f"{'='*70}")

    expert_bytes = int(args.expert_size_mb * 1024 * 1024)

    nccl_results = run_nccl_sweep(
        global_rank, world_size, expert_bytes, args.warmup, args.iters)

    size_results = run_size_sweep(
        global_rank, world_size, args.warmup, args.iters)

    if global_rank == 0:
        # Summary
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")

        bw_by_cat = {}
        for cat, res in nccl_results.items():
            if res:
                bw_by_cat[cat] = statistics.mean(r["bw_gbps"] for r in res)

        if ce_results:
            ce_avg = statistics.mean(r["bw_gbps"] for r in ce_results)
            print(f"  CE DMA (intra-tray):   {ce_avg:.1f} GB/s")

        for cat, bw in bw_by_cat.items():
            print(f"  NCCL P2P ({cat}): {bw:.1f} GB/s")

        if "intra_tray" in bw_by_cat and ce_results:
            ce_avg = statistics.mean(r["bw_gbps"] for r in ce_results)
            nccl_overhead = (1 - bw_by_cat["intra_tray"] / ce_avg) * 100
            print(f"\n  NCCL overhead vs CE (intra): {nccl_overhead:.1f}%")

        cross_cats = [c for c in bw_by_cat if c != "intra_tray"]
        if "intra_tray" in bw_by_cat and cross_cats:
            cross_avg = statistics.mean(bw_by_cat[c] for c in cross_cats)
            ratio = bw_by_cat["intra_tray"] / cross_avg if cross_avg > 0 else 0
            print(f"  Intra/Cross ratio:     {ratio:.2f}x")

        # Save JSON
        output = {
            "config": {
                "world_size": world_size,
                "nnodes": nnodes,
                "gpus_per_tray": GPUS_PER_TRAY,
                "expert_size_mb": args.expert_size_mb,
                "warmup": args.warmup,
                "iters": args.iters,
                "nccl_env": {k: v for k, v in os.environ.items()
                             if k.startswith("NCCL_")},
            },
            "topology": topo_info,
            "ce_dma": ce_results,
            "nccl_p2p": nccl_results,
            "size_sweep": size_results,
        }
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")
        print(f"{'='*70}")

    dist.barrier()
    dist.destroy_process_group()


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NVL72 P2P Bandwidth Benchmark")
    parser.add_argument("--expert-size-mb", type=float, default=31.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-ce", action="store_true",
                        help="Skip CE DMA tests (Phase 1)")
    parser.add_argument("--skip-topo", action="store_true",
                        help="Skip topology collection (Phase 0)")
    parser.add_argument("--output", type=str,
                        default="result/bench_nvl72_p2p.json")
    args = parser.parse_args()

    node_rank = int(os.environ.get("RANK", "0"))

    # Phase 0: Topology (before mp.spawn, only on node 0)
    topo_info = None
    if not args.skip_topo and node_rank == 0:
        topo_info = collect_topology()

    # Phase 1: CE DMA (before mp.spawn, only on node 0)
    ce_results = None
    if not args.skip_ce and node_rank == 0:
        expert_bytes = int(args.expert_size_mb * 1024 * 1024)
        ce_results = run_ce_dma_tests(expert_bytes, args.warmup, args.iters)

    # Phase 2+3: NCCL P2P (inside mp.spawn)
    mp.spawn(worker, nprocs=GPUS_PER_TRAY, args=(args, ce_results, topo_info))


if __name__ == "__main__":
    main()
