"""
Gloo-mediated (CPU) P2P weight swap.

Real-time NCCL P2P (isend/irecv), even with a provably symmetric,
deterministically-identical op list on every rank, still deadlocks under
heavy concurrent load: it shares the same physical CUDA stream as DeepEP's
own dispatch/combine kernels, and any per-rank timing skew (worse under
heavy load, more queued GPU work) creates a real stream-ordering hazard that
NCCL's blocking P2P cannot resolve on its own.

Gloo (CPU/network backend) NEVER touches the CUDA stream, so it is
structurally immune to this class of hazard — proven stable across every
test in this session for the (much smaller) demand-count sync. This module
extends the same safe channel to the actual expert weight tensors: move to
CPU, exchange via Gloo, copy back to GPU. Slower than NVLink P2P, but this
runs infrequently (throttled to every N forward-pass boundaries) and
correctness/stability matters far more than transfer speed here.
"""
import logging
from typing import List

import torch
import torch.distributed

logger = logging.getLogger(__name__)


def simple_p2p_swap(
    old_map: List[int],
    new_map: List[int],
    weights: List[torch.Tensor],
    rank: int,
    num_gpus: int,
    per_gpu: int,
    gloo_group=None,
):
    """Execute a diff-based weight swap over Gloo (CPU), avoiding any
    interaction with the CUDA stream DeepEP's own collectives run on."""
    assert len(old_map) == len(new_map)
    num_physical = len(old_map)

    changed = [i for i in range(num_physical) if old_map[i] != new_map[i]]
    if not changed:
        return 0

    def owner_and_local(slot):
        return slot // per_gpu, slot % per_gpu

    pairs = []
    for dst_slot in changed:
        needed_logical = new_map[dst_slot]
        src_slot = None
        for j in range(num_physical):
            if old_map[j] == needed_logical:
                src_slot = j
                break
        if src_slot is not None:
            pairs.append((src_slot, dst_slot))
        else:
            logger.warning(f"[simple_p2p_swap] no source found for logical {needed_logical}")

    ops = []
    recv_bufs = {}  # (dst_slot, weight_idx) -> CPU buffer

    for src_slot, dst_slot in pairs:
        src_rank, src_local = owner_and_local(src_slot)
        dst_rank, dst_local = owner_and_local(dst_slot)

        if src_rank == dst_rank:
            if rank == src_rank:
                for w in weights:
                    w[dst_local].copy_(w[src_local])
            continue

        if rank == src_rank:
            for i, w in enumerate(weights):
                cpu_tensor = w[src_local].to("cpu", non_blocking=False).contiguous()
                ops.append(torch.distributed.P2POp(
                    torch.distributed.isend, cpu_tensor, dst_rank, group=gloo_group
                ))
        elif rank == dst_rank:
            for i, w in enumerate(weights):
                buf = torch.empty(w[dst_local].shape, dtype=w[dst_local].dtype, device="cpu")
                recv_bufs[(dst_slot, i)] = buf
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv, buf, src_rank, group=gloo_group
                ))

    if ops:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    for (dst_slot, i), buf in recv_bufs.items():
        _, dst_local = owner_and_local(dst_slot)
        weights[i][dst_local].copy_(buf.to(weights[i].device))

    return len(pairs)
