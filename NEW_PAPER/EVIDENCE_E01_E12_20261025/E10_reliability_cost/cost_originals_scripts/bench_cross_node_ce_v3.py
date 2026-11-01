#!/usr/bin/env python3
"""
Cross-Node CE DMA Test v3 - clean barrier alignment.
All ranks call the same number of barriers in the same order.
"""
import argparse, ctypes, json, os, statistics, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GPUS_PER_TRAY = 4

CU_MEM_HANDLE_TYPE_FABRIC          = 0x8
CU_MEM_ALLOC_GRAN_RECOMMENDED      = 1
CU_MEM_ALLOCATION_TYPE_PINNED      = 1
CU_MEM_LOCATION_TYPE_DEVICE        = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 1

cuda   = ctypes.CDLL("libcuda.so.1")
cudart = ctypes.CDLL("libcudart.so")

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type",                ctypes.c_int),
        ("requestedHandleTypes",ctypes.c_int),
        ("location",            CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags_u64",      ctypes.c_ulonglong),
    ]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


def prop_for_device(device_id):
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = device_id
    return p


def bench_nccl(rank, src, dst, nbytes, warmup=5, iters=20):
    t = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    if rank == src: t.normal_()
    for _ in range(warmup):
        if rank == src: dist.isend(t, dst).wait()
        elif rank == dst: dist.irecv(t, src).wait()
        dist.barrier()
    times = []
    for _ in range(iters):
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        if rank == src: dist.isend(t, dst).wait()
        elif rank == dst: dist.irecv(t, src).wait()
        torch.cuda.synchronize()
        if rank in (src, dst): times.append((time.perf_counter()-t0)*1000)
        dist.barrier()
    if times:
        avg = statistics.mean(times)
        return avg, (nbytes/1e9)/(avg/1e3)
    return None, None


def worker(local_rank, args):
    node_rank   = int(os.environ.get("RANK", "0"))
    nnodes      = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12345")

    global_rank = node_rank * GPUS_PER_TRAY + local_rank
    world_size  = nnodes * GPUS_PER_TRAY
    dev         = local_rank

    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", f"tcp://{master_addr}:{master_port}",
                            world_size=world_size, rank=global_rank)

    nbytes = int(args.size_mb * 1024 * 1024)

    # ── Phase 1: NCCL baseline ──────────────────────────────────────────
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"  Cross-Node CE Test v3  |  {world_size} GPUs ({nnodes} trays)")
        print(f"  CUDA {torch.version.cuda}  |  {args.size_mb} MB")
        print(f"{'='*60}")
        print("\n[1] NCCL P2P baseline")

    results = {}
    for src, dst, label in [(0, 1, "intra"), (0, GPUS_PER_TRAY, "cross")]:
        if dst >= world_size:
            continue
        avg, bw = bench_nccl(global_rank, src, dst, nbytes)
        # gather to rank 0
        if global_rank in (src, dst) and avg is not None:
            t = torch.tensor([avg, bw], dtype=torch.float64, device="cuda")
        else:
            t = torch.zeros(2, dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)   # only src/dst have non-zero
        if global_rank == 0:
            print(f"  NCCL {label}: {t[0]:.3f} ms, {t[1]:.1f} GB/s")
            results[f"nccl_{label}"] = {"ms": round(t[0].item(),3),
                                         "gbps": round(t[1].item(),1)}

    # ── Phase 2: Fabric handle path ─────────────────────────────────────
    if global_rank == 0:
        print("\n[2] Fabric Handle CE DMA Test (rank 0 → rank GPUS_PER_TRAY)")

    FABRIC_SZ = 64
    SRC_RANK  = 0
    DST_RANK  = GPUS_PER_TRAY if GPUS_PER_TRAY < world_size else -1

    # shared state tensors (broadcast results between ranks)
    status   = torch.zeros(4, dtype=torch.int32, device="cuda")
    # status[0]: gran OK, status[1]: alloc OK, status[2]: export OK
    # status[3]: import OK

    alloc_size = 0
    src_handle = ctypes.c_uint64(0)
    src_va     = ctypes.c_uint64(0)
    imp_handle = ctypes.c_uint64(0)
    dst_va     = ctypes.c_uint64(0)

    # ── Step A: rank 0 gets granularity & allocates ─────────────────────
    if global_rank == SRC_RANK:
        p = prop_for_device(dev)
        gran = ctypes.c_size_t(0)
        ret = cuda.cuMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(p), CU_MEM_ALLOC_GRAN_RECOMMENDED)
        if ret == 0:
            alloc_size = ((nbytes + gran.value - 1) // gran.value) * gran.value
            status[0] = 1
            print(f"  [rank 0] gran={gran.value//1024}KB, alloc={alloc_size//1024//1024}MB")

            ret = cuda.cuMemCreate(ctypes.byref(src_handle), alloc_size,
                                   ctypes.byref(p), 0)
            if ret == 0:
                status[1] = 1
                print(f"  [rank 0] cuMemCreate OK handle={src_handle.value:#x}")

                # map locally
                ret = cuda.cuMemAddressReserve(ctypes.byref(src_va), alloc_size, 0, 0, 0)
                if ret != 0: print(f"  [rank {global_rank}] AddressReserve FAILED ret={ret}"); raise RuntimeError(f"AddressReserve {ret}")
                ret = cuda.cuMemMap(src_va, alloc_size, 0, src_handle, 0)
                if ret != 0: print(f"  [rank {global_rank}] cuMemMap FAILED ret={ret}"); raise RuntimeError(f"MemMap {ret}")
                acc = CUmemAccessDesc()
                acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
                acc.location.id   = dev
                acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                ret = cuda.cuMemSetAccess(src_va, alloc_size, ctypes.byref(acc), 1)
                if ret != 0: print(f"  [rank {global_rank}] SetAccess FAILED ret={ret}"); raise RuntimeError(f"SetAccess {ret}")
                print(f"  [rank 0] local VA={src_va.value:#x}")
            else:
                print(f"  [rank 0] cuMemCreate FAILED ret={ret}")
        else:
            print(f"  [rank 0] granularity FAILED ret={ret}")

    # broadcast alloc_size to all ranks so dst_rank knows the size
    alloc_size_t = torch.tensor([alloc_size], dtype=torch.int64, device="cuda")
    dist.broadcast(alloc_size_t, src=SRC_RANK)
    alloc_size = int(alloc_size_t[0].item())

    dist.broadcast(status, src=SRC_RANK)  # barrier 1

    if status[1].item() == 0:
        if global_rank == 0:
            print("  ✗ Alloc failed, skipping CE test")
        dist.barrier(); dist.barrier(); dist.barrier()  # barriers 2,3,4
        dist.destroy_process_group()
        return

    # ── Step B: rank 0 exports fabric handle ────────────────────────────
    handle_t = torch.zeros(FABRIC_SZ, dtype=torch.uint8, device="cuda")
    if global_rank == SRC_RANK:
        fab = (ctypes.c_ubyte * FABRIC_SZ)()
        ret = cuda.cuMemExportToShareableHandle(
            ctypes.byref(fab), src_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0)
        if ret == 0:
            status[2] = 1
            for i in range(FABRIC_SZ):
                handle_t[i] = fab[i]
            print(f"  [rank 0] export OK")
        else:
            print(f"  [rank 0] export FAILED ret={ret}")

    dist.broadcast(status, src=SRC_RANK)    # barrier 2
    dist.broadcast(handle_t, src=SRC_RANK)  # barrier 3

    if status[2].item() == 0:
        if global_rank == 0:
            print("  ✗ Export failed")
        dist.barrier()  # barrier 4
        dist.destroy_process_group()
        return

    # ── Step C: dst rank imports & maps ─────────────────────────────────
    import_ok = False
    if DST_RANK != -1 and global_rank == DST_RANK:
        raw = (ctypes.c_ubyte * FABRIC_SZ)(*handle_t.cpu().numpy().tolist())
        imp_p = prop_for_device(dev)
        ret = cuda.cuMemImportFromShareableHandle(
            ctypes.byref(imp_handle), ctypes.byref(raw),
            CU_MEM_HANDLE_TYPE_FABRIC)
        if ret == 0:
            status[3] = 1
            import_ok = True
            print(f"  [rank {DST_RANK}] import OK handle={imp_handle.value:#x}")

            ret = cuda.cuMemAddressReserve(ctypes.byref(dst_va), alloc_size, 0, 0, 0)
            if ret != 0: print(f"  [rank {global_rank}] FAILED ret={ret}"); raise RuntimeError(str(ret))
            ret = cuda.cuMemMap(dst_va, alloc_size, 0, imp_handle, 0)
            if ret != 0: print(f"  [rank {global_rank}] FAILED ret={ret}"); raise RuntimeError(str(ret))
            acc = CUmemAccessDesc()
            acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id   = dev
            acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            ret = cuda.cuMemSetAccess(dst_va, alloc_size, ctypes.byref(acc), 1)
            if ret != 0: print(f"  [rank {global_rank}] FAILED ret={ret}"); raise RuntimeError(str(ret))
            print(f"  [rank {DST_RANK}] remote VA={dst_va.value:#x}")
        else:
            print(f"  [rank {DST_RANK}] import FAILED ret={ret} "
                  f"({'INVALID_VALUE' if ret==400 else 'NOT_SUPPORTED' if ret==801 else str(ret)})")

    dist.all_reduce(status, op=dist.ReduceOp.MAX)  # barrier 4
    results["import_ok"] = bool(status[3].item())

    # ── Step D: benchmark if import succeeded ────────────────────────────
    if status[3].item() == 1 and DST_RANK != -1 and global_rank == DST_RANK:
        local_src = torch.randn(alloc_size // 4, dtype=torch.float32, device="cuda")
        xfer = min(nbytes, alloc_size)

        # Test both cudaMemcpyDefault and cudaMemcpyPeer to fabric VA
        for memcpy_kind, kind_name in [(4, "cudaMemcpyDefault"), (1, "cudaMemcpyD2D")]:
            # probe first
            probe_ret = cudart.cudaMemcpy(
                ctypes.c_uint64(dst_va.value),
                ctypes.c_uint64(local_src.data_ptr()),
                ctypes.c_size_t(min(4096, xfer)),
                ctypes.c_int(memcpy_kind))
            torch.cuda.synchronize()
            if probe_ret != 0:
                print(f"  [rank {DST_RANK}] {kind_name} probe FAILED ret={probe_ret} - skipping")
                continue

            times = []
            for _ in range(args.warmup):
                cudart.cudaMemcpy(ctypes.c_uint64(dst_va.value),
                                  ctypes.c_uint64(local_src.data_ptr()),
                                  ctypes.c_size_t(xfer), ctypes.c_int(memcpy_kind))
                torch.cuda.synchronize()

            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cudart.cudaMemcpy(ctypes.c_uint64(dst_va.value),
                                  ctypes.c_uint64(local_src.data_ptr()),
                                  ctypes.c_size_t(xfer), ctypes.c_int(memcpy_kind))
                torch.cuda.synchronize()
                times.append((time.perf_counter()-t0)*1000)

            avg = statistics.mean(times)
            bw  = (xfer/1e9)/(avg/1e3)
            print(f"  [rank {DST_RANK}] {kind_name}→fabric VA: {avg:.3f}ms, {bw:.1f}GB/s")
            results[f"ce_fabric_{kind_name}"] = {"ms": round(avg,3), "gbps": round(bw,1)}
            if "ce_fabric" not in results:
                results["ce_fabric"] = results[f"ce_fabric_{kind_name}"]

    # broadcast CE result to rank 0
    ce_t = torch.zeros(2, dtype=torch.float64, device="cuda")
    if DST_RANK != -1 and global_rank == DST_RANK and "ce_fabric" in results:
        ce_t[0] = results["ce_fabric"]["ms"]
        ce_t[1] = results["ce_fabric"]["gbps"]
    dist.all_reduce(ce_t, op=dist.ReduceOp.MAX)  # barrier 5
    if global_rank == 0 and ce_t[1].item() > 0:
        results["ce_fabric_from_rank0"] = {"ms": round(ce_t[0].item(),3),
                                            "gbps": round(ce_t[1].item(),1)}

    # cleanup
    if global_rank == DST_RANK and dst_va.value:
        cuda.cuMemUnmap(dst_va, alloc_size)
        cuda.cuMemAddressFree(dst_va, alloc_size)
    if global_rank == SRC_RANK and src_va.value:
        cuda.cuMemUnmap(src_va, alloc_size)
        cuda.cuMemAddressFree(src_va, alloc_size)
        cuda.cuMemRelease(src_handle)

    dist.barrier()  # barrier 6 - final sync

    if global_rank == 0:
        nccl_cross = results.get("nccl_cross", {}).get("gbps", 0)
        ce_key = "ce_fabric_from_rank0"
        ce_bw  = results.get(ce_key, {}).get("gbps", 0)
        print(f"\n{'='*60}")
        print(f"  Fabric import success: {results.get('import_ok')}")
        if ce_bw:
            print(f"  CE fabric BW:   {ce_bw:.1f} GB/s")
            print(f"  NCCL cross BW:  {nccl_cross:.1f} GB/s")
            if nccl_cross:
                print(f"  CE / NCCL:      {ce_bw/nccl_cross:.2f}x")
        else:
            print(f"  CE fabric BW:   N/A")
        print(f"{'='*60}")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {args.output}")

    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-mb", type=float, default=31.0)
    p.add_argument("--warmup",  type=int,   default=5)
    p.add_argument("--iters",   type=int,   default=20)
    p.add_argument("--output",  default="result/bench_cross_node_ce_v3.json")
    args = p.parse_args()
    mp.spawn(worker, nprocs=GPUS_PER_TRAY, args=(args,))


if __name__ == "__main__":
    main()
