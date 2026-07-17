"""V2 rebalancer: elastic threshold + global budget.

Key differences from V1:
- Two-tier threshold: high_threshold (trigger) + target_ratio (stop)
- Global budget: max_total_swaps across ALL layers, not per-layer
- Priority: layers sorted by imbalance ratio, worst-first
- Each layer gets exactly 1 swap per decision (greedy: swap the single
  hottest expert on the busiest GPU with the coldest on the least-busy GPU),
  then move to the next-worst layer. This is more surgical than V1's
  multi-round-per-layer approach.
"""
import torch
import logging
from typing import List, NamedTuple

logger = logging.getLogger(__name__)


class SwapOp(NamedTuple):
    layer_id: int
    phys_slot_a: int
    phys_slot_b: int
    rank_a: int
    rank_b: int
    logical_a: int
    logical_b: int
    imbalance: float


def _compute_layer_ratio(lc, p2l, num_ranks, num_local):
    """Compute imbalance ratio for one layer. Returns (ratio, gpu_loads)."""
    gpu_load = [0] * num_ranks
    for r in range(num_ranks):
        s, e = r * num_local, (r + 1) * num_local
        gpu_load[r] = sum(lc[p2l[i]] for i in range(s, e))
    max_load = max(gpu_load)
    avg_load = max(sum(gpu_load) / num_ranks, 1.0)
    return max_load / avg_load, gpu_load


def _make_one_swap(lc, p2l, gpu_load, num_ranks, num_local):
    """Generate exactly one swap op for one layer: hottest expert on busiest
    GPU <-> coldest expert on least-busy GPU. Returns SwapOp or None."""
    rank_hot = gpu_load.index(max(gpu_load))
    rank_cold = gpu_load.index(min(gpu_load))
    if rank_hot == rank_cold:
        return None

    hot_start = rank_hot * num_local
    hot_end = (rank_hot + 1) * num_local
    phys_a = max(range(hot_start, hot_end), key=lambda i: lc[p2l[i]])
    logical_a = p2l[phys_a]

    cold_start = rank_cold * num_local
    cold_end = (rank_cold + 1) * num_local
    phys_b = min(range(cold_start, cold_end), key=lambda i: lc[p2l[i]])
    logical_b = p2l[phys_b]

    ratio = max(gpu_load) / max(sum(gpu_load) / num_ranks, 1.0)
    return SwapOp(
        layer_id=-1,  # filled by caller
        phys_slot_a=phys_a, phys_slot_b=phys_b,
        rank_a=rank_hot, rank_b=rank_cold,
        logical_a=logical_a, logical_b=logical_b,
        imbalance=ratio,
    )


def try_build_swap_plan_v2(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_ranks: int,
    num_local: int,
    high_threshold: float,
    target_ratio: float,
    max_total_swaps: int,
) -> List[SwapOp]:
    """V2 swap planner: elastic threshold + global budget.

    1. Compute ratio for all layers
    2. Filter layers with ratio > high_threshold
    3. Sort by ratio descending (worst-first)
    4. For each selected layer, do 1 swap, check if ratio dropped below
       target_ratio; if not and budget remains, allow another round
    5. Stop when budget exhausted or no more layers above high_threshold
    """
    L = logical_count.shape[0]
    lc_cpu = logical_count.tolist()
    p2l_cpu = physical_to_logical_map.tolist()

    # Phase 1: compute all layer ratios
    layer_ratios = []  # (layer_id, ratio)
    for l in range(L):
        ratio, _ = _compute_layer_ratio(lc_cpu[l], p2l_cpu[l], num_ranks, num_local)
        if ratio > high_threshold:
            layer_ratios.append((l, ratio))

    if not layer_ratios:
        return []

    # Sort worst-first
    layer_ratios.sort(key=lambda x: x[1], reverse=True)

    # Phase 2: greedily assign swaps within global budget
    candidates = []
    budget_remaining = max_total_swaps
    p2l_working = [list(row) for row in p2l_cpu]  # mutable copy for simulation

    # Multiple passes: keep going until budget empty or no improvement
    for _pass in range(max_total_swaps):  # at most max_total_swaps passes
        if budget_remaining <= 0:
            break
        made_progress = False
        for layer_id, _ in layer_ratios:
            if budget_remaining <= 0:
                break
            ratio, gpu_load = _compute_layer_ratio(
                lc_cpu[layer_id], p2l_working[layer_id], num_ranks, num_local
            )
            if ratio <= target_ratio:
                continue  # already good enough
            if ratio <= high_threshold and _pass > 0:
                continue  # only first pass uses high_threshold, subsequent passes only fix remaining > target

            op = _make_one_swap(
                lc_cpu[layer_id], p2l_working[layer_id], gpu_load, num_ranks, num_local
            )
            if op is None:
                continue

            # Simulate the swap in working copy
            p2l_working[layer_id][op.phys_slot_a], p2l_working[layer_id][op.phys_slot_b] = (
                op.logical_b, op.logical_a
            )

            candidates.append(SwapOp(
                layer_id=layer_id,
                phys_slot_a=op.phys_slot_a, phys_slot_b=op.phys_slot_b,
                rank_a=op.rank_a, rank_b=op.rank_b,
                logical_a=op.logical_a, logical_b=op.logical_b,
                imbalance=ratio,
            ))
            budget_remaining -= 1
            made_progress = True

        if not made_progress:
            break

    if candidates:
        ratios_before = [c.imbalance for c in candidates]
        # Compute after ratios for diagnostics
        ratios_after = []
        for c in candidates:
            r, _ = _compute_layer_ratio(
                lc_cpu[c.layer_id], p2l_working[c.layer_id], num_ranks, num_local
            )
            ratios_after.append(r)
        logger.info(
            f"[OEPLB-V2] plan: {len(candidates)} swaps across "
            f"{len(set(c.layer_id for c in candidates))} layers, "
            f"budget_used={max_total_swaps - budget_remaining}/{max_total_swaps}, "
            f"avg_ratio {sum(ratios_before)/len(ratios_before):.3f} -> {sum(ratios_after)/len(ratios_after):.3f}"
        )

    return candidates
