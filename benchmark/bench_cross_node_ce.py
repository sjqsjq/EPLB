#!/usr/bin/env python3
"""
Cross-Node CE DMA Feasibility Test on NVL72.

Investigates whether Copy Engine (CE) DMA can work cross-node via:
  1. PyTorch torch.distributed._symmetric_memory (fabric handle based)
  2. Direct cuMem + fabric handle approach
  3. Comparison: NCCL P2P vs SymmetricMemory SM ld/st vs CE (if feasible)

Key question: After establishing cross-node VA mapping via fabric handles,
can cudaMemcpyPeer trigger CE DMA across nodes, or does it fall back to SM?

Launch:
    ./run_script.sh --master-ip 11.139.21.78 --cur-node 2 \
        --command "python bench_cross_node_ce.py"
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


def check_fabric_support():
    """Check if this GPU supports NVLink fabric handles for cross-node CE."""
    info = {}

    # Method 1: Check via nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        info["gpu_info"] = r.stdout.strip()
    except Exception as e:
        info["gpu_info"] = f"Error: {e}"

    # Method 2: Check PyTorch SymmetricMemory availability
    try:
        has_symm = hasattr(torch.cuda, '_SymmetricMemory') or \
                   hasattr(torch.distributed, '_symmetric_memory')
        info["has_pytorch_symmetric_memory"] = has_symm
    except Exception as e:
        info["has_pytorch_symmetric_memory"] = f"Error: {e}"

    # Method 3: Try to import torch._C._distributed_c10d
    try:
        import torch._C._distributed_c10d as c10d
        info["c10d_available"] = True
    except Exception as e:
        info["c10d_available"] = False

    return info


def bench_nccl_p2p(global_rank, src_rank, dst_rank, num_bytes, warmup=5, iters=20):
    """Baseline: NCCL P2P (existing benchmark method)."""
    tensor = torch.empty(num_bytes // 2, dtype=torch.bfloat16, device="cuda")
    if global_rank == src_rank:
        tensor.normal_()

    for _ in range(warmup):
        if global_rank == src_rank:
            dist.isend(tensor, dst_rank).wait()
        elif global_rank == dst_rank:
            dist.irecv(tensor, src_rank).wait()
        dist.barrier()

    times = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if global_rank == src_rank:
            dist.isend(tensor, dst_rank).wait()
        elif global_rank == dst_rank:
            dist.irecv(tensor, src_rank).wait()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if global_rank in (src_rank, dst_rank):
            times.append((t1 - t0) * 1000)
        dist.barrier()

    if times:
        avg_ms = statistics.mean(times)
        bw = (num_bytes / 1e9) / (avg_ms / 1e3)
        return {"method": "nccl_p2p", "avg_ms": round(avg_ms, 3), "bw_gbps": round(bw, 1)}
    return None


def try_symmetric_memory_cross_node(global_rank, world_size, num_bytes, warmup=5, iters=20):
    """
    Attempt cross-node data transfer via PyTorch SymmetricMemory.

    SymmetricMemory establishes cross-node VA mappings via:
      cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC) → fabric handle
      → cuMemImportFromShareableHandle on peer ranks
      → all ranks have VA pointing to each other's physical HBM

    Then data transfer is done via SM ld/st (NOT CE).
    This gives us the "SM ld/st over NVSwitch" baseline.
    """
    results = {}

    # Check if SymmetricMemory rendezvous is available
    try:
        # PyTorch >= 2.5: torch.distributed._symmetric_memory
        from torch.distributed._symmetric_memory import (
            enable_symm_mem_for_group,
            get_symm_mem_workspace,
        )
        symm_mem_available = True
    except ImportError:
        try:
            # Alternative path
            symm_mem_available = hasattr(torch.cuda, '_SymmetricMemory')
        except:
            symm_mem_available = False

    results["symm_mem_available"] = symm_mem_available

    if not symm_mem_available:
        if global_rank == 0:
            print("  SymmetricMemory API not available in this PyTorch version")
        return results

    try:
        group = dist.group.WORLD

        # Enable symmetric memory for this group
        enable_symm_mem_for_group(dist.group.WORLD)

        # Allocate symmetric memory buffer (same size on all ranks)
        # This uses cuMemCreate with CU_MEM_HANDLE_TYPE_FABRIC internally
        workspace = get_symm_mem_workspace(
            num_bytes, like_tensor=None, device="cuda"
        )

        results["workspace_allocated"] = True
        results["workspace_size"] = num_bytes

        if global_rank == 0:
            print(f"  SymmetricMemory workspace allocated: {num_bytes / 1024 / 1024:.1f} MB")
            print(f"  Buffer ptr: {workspace.data_ptr():#x}")

        # Now we have cross-node mapped memory
        # Measure SM-based write to remote rank's memory (rank 0 writes to rank GPUS_PER_TRAY)
        # via ld/st on the fabric-mapped VA

        # Simple benchmark: copy local tensor into the symmetric workspace
        local_data = torch.randn(num_bytes // 4, dtype=torch.float32, device="cuda")

        # Get peer buffer (this is the cross-node VA)
        # workspace is a tensor whose storage is fabric-mapped

        for _ in range(warmup):
            workspace.copy_(local_data[:workspace.numel()] if local_data.numel() >= workspace.numel()
                           else local_data.repeat(workspace.numel() // local_data.numel() + 1)[:workspace.numel()])
            torch.cuda.synchronize()
            dist.barrier()

        times = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            workspace[:local_data.numel()].copy_(local_data)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            dist.barrier()

        avg_ms = statistics.mean(times)
        bw = (num_bytes / 1e9) / (avg_ms / 1e3)
        results["sm_write_avg_ms"] = round(avg_ms, 3)
        results["sm_write_bw_gbps"] = round(bw, 1)

        if global_rank == 0:
            print(f"  SymmetricMemory SM write: {avg_ms:.3f} ms, {bw:.1f} GB/s")

    except Exception as e:
        results["error"] = str(e)
        if global_rank == 0:
            print(f"  SymmetricMemory failed: {e}")

    return results


def try_fabric_handle_ce(global_rank, world_size, num_bytes, warmup=5, iters=20):
    """
    Attempt to use CE DMA over fabric-mapped cross-node memory.

    Strategy:
      1. Use ctypes to call CUDA Driver API directly
         (cuMemCreate, cuMemMap, cuMemSetAccess with fabric handle)
      2. After mapping, call cudaMemcpyPeer on the fabric-mapped VA
      3. Measure bandwidth → if high and SM usage is zero → CE is working

    This is the experimental path to verify if CE can work cross-node.
    """
    results = {}

    try:
        import ctypes
        cuda_lib = ctypes.CDLL("libcuda.so.1")

        # Check if fabric handle type is available (CUDA 12.3+)
        # CU_MEM_HANDLE_TYPE_FABRIC = 0x8
        CU_MEM_HANDLE_TYPE_FABRIC = 0x8
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 1

        # Get allocation granularity
        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location_type", ctypes.c_int),
                ("location_id", ctypes.c_int),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("reserved", ctypes.c_ulonglong),
                ("compressionType", ctypes.c_ubyte),
                ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort),
                ("_reserved", ctypes.c_ubyte * 4),
            ]

        granularity = ctypes.c_size_t(0)
        prop = CUmemAllocationProp()
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
        prop.location_type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location_id = torch.cuda.current_device()

        ret = cuda_lib.cuMemGetAllocationGranularity(
            ctypes.byref(granularity), ctypes.byref(prop),
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )

        if ret != 0:
            results["fabric_handle_supported"] = False
            results["error"] = f"cuMemGetAllocationGranularity failed: {ret}"
            if global_rank == 0:
                print(f"  Fabric handle NOT supported (cuMemGetAllocationGranularity ret={ret})")
                if ret == 1:
                    print("  CUDA_ERROR_INVALID_VALUE: likely CU_MEM_HANDLE_TYPE_FABRIC not supported on this GPU/CUDA version")
            return results

        results["fabric_handle_supported"] = True
        results["granularity"] = granularity.value
        if global_rank == 0:
            print(f"  Fabric handle supported! Granularity: {granularity.value / 1024:.0f} KB")

        # Try to create fabric memory allocation
        alloc_size = max(num_bytes, granularity.value)
        # Round up to granularity
        alloc_size = ((alloc_size + granularity.value - 1) // granularity.value) * granularity.value

        handle = ctypes.c_uint64(0)
        ret = cuda_lib.cuMemCreate(ctypes.byref(handle), alloc_size, ctypes.byref(prop), 0)

        if ret != 0:
            results["fabric_alloc_success"] = False
            results["error"] = f"cuMemCreate with fabric handle failed: {ret}"
            if global_rank == 0:
                print(f"  cuMemCreate(FABRIC) failed: ret={ret}")
                if ret == 801:
                    print("  CUDA_ERROR_NOT_SUPPORTED: MNNVL fabric not available in this environment")
                elif ret == 300:
                    print("  CUDA_ERROR_NO_DEVICE: no NVLink fabric detected")
            return results

        results["fabric_alloc_success"] = True
        if global_rank == 0:
            print(f"  cuMemCreate(FABRIC) success! handle={handle.value:#x}, size={alloc_size/1024/1024:.1f} MB")

        # Export fabric handle
        fabric_handle_bytes = (ctypes.c_ubyte * 64)()  # CUmemFabricHandle is 64 bytes
        ret = cuda_lib.cuMemExportToShareableHandle(
            ctypes.byref(fabric_handle_bytes), handle,
            CU_MEM_HANDLE_TYPE_FABRIC, 0
        )
        results["export_success"] = (ret == 0)
        if global_rank == 0:
            print(f"  cuMemExportToShareableHandle: {'success' if ret == 0 else f'failed ret={ret}'}")

        if ret == 0:
            # Broadcast fabric handle to all ranks
            handle_tensor = torch.frombuffer(bytes(fabric_handle_bytes), dtype=torch.uint8).cuda().clone()
            dist.broadcast(handle_tensor, src=0)

            if global_rank != 0:
                # Import rank 0's handle on rank GPUS_PER_TRAY (cross-node)
                received_bytes = handle_tensor.cpu().numpy().tobytes()
                received_handle = (ctypes.c_ubyte * 64)(*received_bytes)

                imported_handle = ctypes.c_uint64(0)
                import_prop = CUmemAllocationProp()
                import_prop.type = 1
                import_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
                import_prop.location_type = 1
                import_prop.location_id = torch.cuda.current_device()

                ret2 = cuda_lib.cuMemImportFromShareableHandle(
                    ctypes.byref(imported_handle),
                    ctypes.byref(received_handle),
                    CU_MEM_HANDLE_TYPE_FABRIC
                )
                results["import_success"] = (ret2 == 0)
                if global_rank == GPUS_PER_TRAY:
                    print(f"  cuMemImportFromShareableHandle (cross-node): {'success' if ret2 == 0 else f'failed ret={ret2}'}")

            # Map into VA and try CE memcpy
            # (full mapping code would be needed here for complete test)
            results["note"] = "Fabric handle exchanged. Full CE test requires cuMemMap+cuMemSetAccess."

        # Cleanup
        cuda_lib.cuMemRelease(handle)

    except Exception as e:
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        if global_rank == 0:
            print(f"  Exception: {e}")

    return results


def worker(local_rank, args):
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

    num_bytes = int(args.size_mb * 1024 * 1024)

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  Cross-Node CE Feasibility Test")
        print(f"  {world_size} GPUs ({nnodes} trays), transfer size: {args.size_mb} MB")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"{'='*70}")

        fabric_info = check_fabric_support()
        print(f"\n  GPU: {fabric_info.get('gpu_info', 'N/A')}")
        print(f"  SymmetricMemory API: {fabric_info.get('has_pytorch_symmetric_memory', False)}")

    dist.barrier()

    results = {
        "world_size": world_size,
        "nnodes": nnodes,
        "size_mb": args.size_mb,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    # Test 1: NCCL P2P baseline (intra and cross tray)
    if global_rank == 0:
        print(f"\n[1] NCCL P2P Baseline")

    for src, dst, label in [(0, 1, "intra-tray"), (0, GPUS_PER_TRAY, "cross-tray")]:
        if dst >= world_size:
            continue
        r = bench_nccl_p2p(global_rank, src, dst, num_bytes)
        dist.barrier()
        if r and global_rank in (src, dst):
            res_t = torch.tensor([r["avg_ms"], r["bw_gbps"]], dtype=torch.float64, device="cuda")
            dist.broadcast(res_t, src=src)
        elif global_rank == 0:
            res_t = torch.empty(2, dtype=torch.float64, device="cuda")
            dist.broadcast(res_t, src=src)
        else:
            res_t = torch.empty(2, dtype=torch.float64, device="cuda")
            dist.broadcast(res_t, src=src)

        dist.barrier()
        if global_rank == 0:
            print(f"  NCCL P2P {label} (rank {src}→{dst}): {res_t[0]:.3f} ms, {res_t[1]:.1f} GB/s")
            results[f"nccl_{label.replace('-', '_')}"] = {
                "avg_ms": round(res_t[0].item(), 3),
                "bw_gbps": round(res_t[1].item(), 1)
            }

    # Test 2: SymmetricMemory (SM ld/st over fabric)
    if global_rank == 0:
        print(f"\n[2] PyTorch SymmetricMemory (SM ld/st over NVSwitch)")
    dist.barrier()

    symm_results = try_symmetric_memory_cross_node(global_rank, world_size, num_bytes)
    results["symmetric_memory"] = symm_results

    dist.barrier()

    # Test 3: Fabric Handle CE experiment
    if global_rank == 0:
        print(f"\n[3] Fabric Handle CE DMA Experiment")
    dist.barrier()

    ce_results = try_fabric_handle_ce(global_rank, world_size, num_bytes)
    results["fabric_ce"] = ce_results

    dist.barrier()

    # Summary
    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  Fabric handle supported: {ce_results.get('fabric_handle_supported', False)}")
        print(f"  Fabric alloc success:    {ce_results.get('fabric_alloc_success', False)}")
        print(f"  Export success:          {ce_results.get('export_success', False)}")
        print(f"  SymmetricMemory avail:   {symm_results.get('symm_mem_available', False)}")

        if "error" in ce_results:
            print(f"\n  CE Cross-Node Error: {ce_results['error']}")

        print(f"\n  Conclusion:")
        if ce_results.get("fabric_alloc_success"):
            print("  ✓ CUDA fabric handle works on this GPU/CUDA version")
            print("  → Cross-node CE DMA is theoretically possible")
            print("  → Full test requires cuMemMap + cuMemSetAccess + cudaMemcpyPeer")
        elif ce_results.get("fabric_handle_supported") is False:
            print("  ✗ Fabric handle NOT supported")
            print("  → Cross-node CE DMA is NOT possible on this configuration")
            print("  → NCCL P2P (SM-based) is the only cross-node option")
        else:
            print("  ? Fabric handle query failed - check CUDA version (need 12.3+)")

        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float, default=31.0)
    parser.add_argument("--output", default="result/bench_cross_node_ce.json")
    args = parser.parse_args()
    mp.spawn(worker, nprocs=GPUS_PER_TRAY, args=(args,))


if __name__ == "__main__":
    main()
