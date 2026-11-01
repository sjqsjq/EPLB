#!/usr/bin/env python3
"""
Cross-Node CE DMA Complete Test on NVL72.

Both nodes share CliqueId=32766, ClusterUUID=be4677e3...
Fabric is healthy -> attempt full cuMemMap + cudaMemcpyPeer across nodes.

Tests:
  1. NCCL P2P baseline
  2. Full fabric handle path: cuMemCreate → export → import → cuMemMap → cudaMemcpyPeer
  3. If CE works cross-node: measure bandwidth vs NCCL

Launch:
    ./run_script.sh --master-ip 11.139.21.78 --cur-node 2 \
        --command "python bench_cross_node_ce_v2.py"
"""

import argparse
import ctypes
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GPUS_PER_TRAY = 4

# ── CUDA Driver API constants ──────────────────────────────────────────
CU_MEM_HANDLE_TYPE_FABRIC            = 0x8
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
CU_MEM_ALLOCATION_TYPE_PINNED        = 1
CU_MEM_LOCATION_TYPE_DEVICE          = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE   = 1
CUDA_ERROR_INVALID_VALUE             = 400
CUDA_ERROR_NOT_SUPPORTED             = 801
CUDA_ERROR_NOT_INITIALIZED           = 3

cuda = ctypes.CDLL("libcuda.so.1", use_errno=True)
cudart = ctypes.CDLL("libcudart.so", use_errno=True)


# ── CUDA Driver structs ────────────────────────────────────────────────

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type",                   ctypes.c_int),
        ("requestedHandleTypes",   ctypes.c_int),
        ("location",               CUmemLocation),
        ("win32HandleMetaData",    ctypes.c_void_p),
        ("allocFlags_compressionType", ctypes.c_ubyte),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_ubyte),
        ("allocFlags_usage",       ctypes.c_ushort),
        ("allocFlags_reserved",    ctypes.c_ubyte * 4),
    ]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags",    ctypes.c_int),
    ]


def check_ret(ret, op):
    if ret != 0:
        raise RuntimeError(f"{op} failed: CUDA error {ret}")


def bench_fn(fn, warmup=5, iters=20):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.mean(times), min(times), max(times)


# ── Phase 1: NCCL P2P baseline ────────────────────────────────────────

def bench_nccl_p2p(global_rank, src_rank, dst_rank, num_bytes, warmup=5, iters=20):
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
        avg = statistics.mean(times)
        bw = (num_bytes / 1e9) / (avg / 1e3)
        return avg, bw
    return None, None


# ── Phase 2: Full fabric handle path ──────────────────────────────────

def full_fabric_ce_test(global_rank, world_size, num_bytes, warmup=5, iters=20):
    """
    Full path:
      Rank 0 (Node 0, GPU 0):
        cuMemCreate(FABRIC) → cuMemMap(local VA) → cuMemSetAccess
        → export handle → broadcast
      Rank 4 (Node 1, GPU 0):
        receive handle → cuMemImportFromShareableHandle
        → cuMemAddressReserve → cuMemMap(local VA) → cuMemSetAccess
        → cudaMemcpyPeer(local_buf → remote_VA_of_rank0) or reverse
    """
    device_id = torch.cuda.current_device()
    results = {}

    # Only rank 0 and rank GPUS_PER_TRAY participate
    src_rank = 0
    dst_rank = GPUS_PER_TRAY
    if dst_rank >= world_size:
        return {"skipped": "not enough nodes"}

    is_src = (global_rank == src_rank)
    is_dst = (global_rank == dst_rank)

    if not (is_src or is_dst):
        # Other ranks just participate in barriers
        for _ in range(6):
            dist.barrier()
        return {}

    # Step 1: Get granularity
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id

    gran = ctypes.c_size_t(0)
    ret = cuda.cuMemGetAllocationGranularity(ctypes.byref(gran),
                                              ctypes.byref(prop),
                                              CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
    check_ret(ret, "cuMemGetAllocationGranularity")
    alloc_size = ((num_bytes + gran.value - 1) // gran.value) * gran.value
    results["granularity"] = gran.value
    results["alloc_size"] = alloc_size

    if is_src:
        print(f"  [src rank {src_rank}] alloc_size={alloc_size//1024//1024} MB, gran={gran.value//1024} KB")

    # Step 2: Rank 0 allocates fabric memory
    src_handle = ctypes.c_uint64(0)
    if is_src:
        ret = cuda.cuMemCreate(ctypes.byref(src_handle), alloc_size,
                               ctypes.byref(prop), 0)
        check_ret(ret, "cuMemCreate")
        print(f"  [src rank {src_rank}] cuMemCreate OK, handle={src_handle.value:#x}")

    # Step 3: Rank 0 maps it locally
    src_va = ctypes.c_uint64(0)
    if is_src:
        ret = cuda.cuMemAddressReserve(ctypes.byref(src_va), alloc_size, 0, 0, 0)
        check_ret(ret, "cuMemAddressReserve(src)")
        ret = cuda.cuMemMap(src_va, alloc_size, 0, src_handle, 0)
        check_ret(ret, "cuMemMap(src)")

        access = CUmemAccessDesc()
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_id
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        ret = cuda.cuMemSetAccess(src_va, alloc_size, ctypes.byref(access), 1)
        check_ret(ret, "cuMemSetAccess(src)")
        print(f"  [src rank {src_rank}] mapped src VA={src_va.value:#x}")

    # Step 4: Export fabric handle from rank 0
    FABRIC_HANDLE_SIZE = 64
    fabric_bytes = (ctypes.c_ubyte * FABRIC_HANDLE_SIZE)()
    if is_src:
        ret = cuda.cuMemExportToShareableHandle(ctypes.byref(fabric_bytes),
                                                src_handle,
                                                CU_MEM_HANDLE_TYPE_FABRIC, 0)
        check_ret(ret, "cuMemExportToShareableHandle")
        print(f"  [src rank {src_rank}] export OK")

    dist.barrier()  # barrier 1

    # Step 5: Broadcast fabric handle to dst rank via NCCL
    handle_tensor = torch.zeros(FABRIC_HANDLE_SIZE, dtype=torch.uint8, device="cuda")
    if is_src:
        for i in range(FABRIC_HANDLE_SIZE):
            handle_tensor[i] = fabric_bytes[i]
    dist.broadcast(handle_tensor, src=src_rank)
    received = handle_tensor.cpu().numpy().tobytes()

    dist.barrier()  # barrier 2

    # Step 6: Dst rank imports and maps
    dst_va = ctypes.c_uint64(0)
    import_success = False
    if is_dst:
        received_bytes = (ctypes.c_ubyte * FABRIC_HANDLE_SIZE)(*received)
        imported_handle = ctypes.c_uint64(0)

        import_prop = CUmemAllocationProp()
        import_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        import_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
        import_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        import_prop.location.id = device_id

        ret = cuda.cuMemImportFromShareableHandle(
            ctypes.byref(imported_handle),
            ctypes.byref(received_bytes),
            CU_MEM_HANDLE_TYPE_FABRIC
        )
        if ret != 0:
            print(f"  [dst rank {dst_rank}] cuMemImportFromShareableHandle FAILED ret={ret}")
            if ret == CUDA_ERROR_INVALID_VALUE:
                print(f"    → CUDA_ERROR_INVALID_VALUE: GPUs may not be in same fabric domain")
                print(f"    → Check: nvidia-smi -q | grep CliqueId (both nodes should match)")
            results["import_ret"] = ret
            results["import_success"] = False
        else:
            print(f"  [dst rank {dst_rank}] import OK, handle={imported_handle.value:#x}")
            import_success = True
            results["import_success"] = True

            # Map on dst GPU
            ret = cuda.cuMemAddressReserve(ctypes.byref(dst_va), alloc_size, 0, 0, 0)
            check_ret(ret, "cuMemAddressReserve(dst)")
            ret = cuda.cuMemMap(dst_va, alloc_size, 0, imported_handle, 0)
            check_ret(ret, "cuMemMap(dst)")

            access = CUmemAccessDesc()
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device_id
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            ret = cuda.cuMemSetAccess(dst_va, alloc_size, ctypes.byref(access), 1)
            check_ret(ret, "cuMemSetAccess(dst)")
            print(f"  [dst rank {dst_rank}] mapped remote VA={dst_va.value:#x}")

    dist.barrier()  # barrier 3

    # Step 7: If import succeeded, benchmark CE DMA from dst to src's memory
    if is_dst and import_success and dst_va.value != 0:
        # dst_va points to rank 0's physical memory on Node 0
        # We do a local tensor → remote VA copy via cudaMemcpyPeer or tensor.copy_

        # Method A: Use torch tensor view on the mapped VA
        # Since dst_va is in dst's address space but maps to src's HBM over NVLink,
        # writing to it is SM ld/st (not CE). CE would require cudaMemcpyPeer.

        # Method B: Local buf → remote mapped VA via cudaMemcpy (async)
        local_buf = torch.randn(alloc_size // 4, dtype=torch.float32, device="cuda")
        transfer_bytes = min(num_bytes, alloc_size)

        # Test SM ld/st via tensor view on fabric-mapped VA
        try:
            # Create tensor on the mapped VA (fabric-mapped remote memory)
            remote_tensor = torch.zeros(alloc_size // 4, dtype=torch.float32)
            # UNSAFE: create tensor from raw pointer (fabric VA)
            # This is SM ld/st path - writes go over NVLink to Node 0
            remote_ptr = ctypes.cast(dst_va.value, ctypes.POINTER(ctypes.c_float))

            def sm_write_fn():
                # SM kernel writes to fabric-mapped VA
                ctypes.memmove(remote_ptr, local_buf.data_ptr(),
                               transfer_bytes)

            # Actually use cudaMemcpy to the fabric VA (CE path)
            ret = cudart.cudaMemcpy(
                ctypes.c_uint64(dst_va.value),
                ctypes.c_uint64(local_buf.data_ptr()),
                ctypes.c_size_t(transfer_bytes),
                ctypes.c_int(1)  # cudaMemcpyDeviceToDevice
            )
            print(f"  [dst rank {dst_rank}] cudaMemcpy to fabric VA: ret={ret}")

            if ret == 0:
                # Benchmark CE write to fabric VA
                def ce_fabric_write():
                    cudart.cudaMemcpyAsync(
                        ctypes.c_uint64(dst_va.value),
                        ctypes.c_uint64(local_buf.data_ptr()),
                        ctypes.c_size_t(transfer_bytes),
                        ctypes.c_int(1),  # cudaMemcpyDeviceToDevice
                        ctypes.c_uint64(0)  # default stream
                    )
                    cudart.cudaStreamSynchronize(ctypes.c_uint64(0))

                avg_ms, min_ms, max_ms = bench_fn(ce_fabric_write, warmup, iters)
                bw = (transfer_bytes / 1e9) / (avg_ms / 1e3)
                print(f"  [dst rank {dst_rank}] CE fabric write: {avg_ms:.3f} ms, {bw:.1f} GB/s")
                results["ce_fabric_write"] = {
                    "avg_ms": round(avg_ms, 3),
                    "bw_gbps": round(bw, 1),
                    "note": "cudaMemcpy(DeviceToDevice) to fabric-mapped remote VA"
                }
            else:
                print(f"  [dst rank {dst_rank}] cudaMemcpy to fabric VA FAILED ret={ret}")
                results["ce_fabric_write"] = {"error": f"cudaMemcpy failed ret={ret}"}

        except Exception as e:
            print(f"  [dst rank {dst_rank}] CE fabric test exception: {e}")
            results["ce_fabric_write"] = {"error": str(e)}

    dist.barrier()  # barrier 4

    # Step 8: Also test SM write via torch (fabric VA as torch tensor)
    if is_dst and import_success and dst_va.value != 0:
        try:
            # Create torch tensor backed by fabric-mapped VA
            fabric_storage = torch.frombuffer(
                (ctypes.c_byte * alloc_size).from_address(dst_va.value),
                dtype=torch.float32,
                count=alloc_size // 4
            )
            local_src = torch.randn(alloc_size // 4, dtype=torch.float32, device="cuda")

            def sm_fabric_write():
                fabric_storage.copy_(local_src.cpu())  # CPU path - won't work well

            # Better: use direct CUDA memcpy with cudaMemcpyDefault
            def sm_test():
                ret = cudart.cudaMemcpy(
                    ctypes.c_uint64(dst_va.value),
                    ctypes.c_uint64(local_src.data_ptr()),
                    ctypes.c_size_t(min(num_bytes, alloc_size)),
                    ctypes.c_int(4)  # cudaMemcpyDefault
                )
                return ret

            ret = sm_test()
            print(f"  [dst rank {dst_rank}] cudaMemcpy(Default) to fabric VA: ret={ret}")

        except Exception as e:
            print(f"  [dst rank {dst_rank}] SM fabric test exception: {e}")

    dist.barrier()  # barrier 5

    # Cleanup
    if is_dst and dst_va.value != 0:
        cuda.cuMemUnmap(dst_va, alloc_size)
        cuda.cuMemAddressFree(dst_va, alloc_size)

    if is_src and src_va.value != 0:
        cuda.cuMemUnmap(src_va, alloc_size)
        cuda.cuMemAddressFree(src_va, alloc_size)
        cuda.cuMemRelease(src_handle)

    dist.barrier()  # barrier 6

    return results


# ── Worker ─────────────────────────────────────────────────────────────

def worker(local_rank, args):
    node_rank = int(os.environ.get("RANK", "0"))
    nnodes    = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12345")

    global_rank = node_rank * GPUS_PER_TRAY + local_rank
    world_size  = nnodes * GPUS_PER_TRAY

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank,
    )

    num_bytes = int(args.size_mb * 1024 * 1024)
    results = {"world_size": world_size, "nnodes": nnodes, "size_mb": args.size_mb}

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"  NVL72 Fabric CE Test v2")
        print(f"  {world_size} GPUs ({nnodes} trays), {args.size_mb} MB")
        print(f"  CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
        print(f"{'='*70}")

    dist.barrier()

    # Phase 1: NCCL baseline
    if global_rank == 0:
        print("\n[Phase 1] NCCL P2P Baseline")
    for src, dst, label in [(0, 1, "intra-tray"), (0, GPUS_PER_TRAY, "cross-tray")]:
        if dst >= world_size:
            continue
        avg, bw = bench_nccl_p2p(global_rank, src, dst, num_bytes)
        dist.barrier()
        if avg and global_rank in (src, dst):
            t = torch.tensor([avg, bw], dtype=torch.float64, device="cuda")
            dist.broadcast(t, src=src)
        else:
            t = torch.empty(2, dtype=torch.float64, device="cuda")
            dist.broadcast(t, src=src)
        dist.barrier()
        if global_rank == 0:
            print(f"  NCCL {label}: {t[0]:.3f} ms, {t[1]:.1f} GB/s")
            results[f"nccl_{label.replace('-','_')}"] = {"avg_ms": round(t[0].item(),3), "bw_gbps": round(t[1].item(),1)}

    # Phase 2: Full fabric CE test
    if global_rank == 0:
        print("\n[Phase 2] Full Fabric Handle + CE DMA Test")
    dist.barrier()

    try:
        ce_res = full_fabric_ce_test(global_rank, world_size, num_bytes, args.warmup, args.iters)
        results["fabric_ce"] = ce_res
    except Exception as e:
        if global_rank in (0, GPUS_PER_TRAY):
            print(f"  Exception in fabric CE test: {e}")
            import traceback; traceback.print_exc()
        results["fabric_ce"] = {"error": str(e)}
        for _ in range(6):
            try: dist.barrier()
            except: pass

    dist.barrier()

    if global_rank == 0:
        print(f"\n{'='*70}")
        ce = results.get("fabric_ce", {})
        print(f"  Import success:      {ce.get('import_success', 'N/A')}")
        if "ce_fabric_write" in ce:
            cfw = ce["ce_fabric_write"]
            if "bw_gbps" in cfw:
                print(f"  CE fabric write BW:  {cfw['bw_gbps']} GB/s  ({cfw['avg_ms']} ms)")
                nccl_cross = results.get("nccl_cross_tray", {}).get("bw_gbps", 0)
                print(f"  NCCL cross-tray BW:  {nccl_cross} GB/s")
                print(f"  CE / NCCL ratio:     {cfw['bw_gbps']/nccl_cross:.2f}x" if nccl_cross else "")
            else:
                print(f"  CE fabric write:     {cfw}")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {args.output}")
        print(f"{'='*70}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-mb",  type=float, default=31.0)
    p.add_argument("--warmup",   type=int,   default=5)
    p.add_argument("--iters",    type=int,   default=20)
    p.add_argument("--output",   default="result/bench_cross_node_ce_v2.json")
    args = p.parse_args()
    mp.spawn(worker, nprocs=GPUS_PER_TRAY, args=(args,))


if __name__ == "__main__":
    main()
