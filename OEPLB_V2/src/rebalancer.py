"""V2.1 rebalancer: compromise between V1 (thorough per-layer correction,
too-permissive threshold=1.15 -> touches ~30% of layers) and V2 (elastic
2-tier threshold but hard-capped at 8 total swaps -> too little correction).

V2.1 design (per user request 2026-07-17):
- Single threshold_ratio (not high/target 2-tier): a layer is selected iff
  its ratio > threshold_ratio, and gets swapped in FULL multi-round fashion
  (like V1's _build_layer_swap_sequence) until ratio drops back below
  threshold_ratio or max_swaps_per_layer rounds are exhausted.
- NO global cap on total swaps across layers -- every layer that qualifies
  gets fully corrected. This was V1's behavior all along when
  max_total_swap_layers was set generously (48); V2 introduced an
  unjustified hard budget=8 that starved correction. V2.1 removes that.
- The only real "compromise" knob is threshold_ratio itself: set it higher
  than V1's 1.15 (which triggered on ~30% of (layer,step) samples) so only
  genuinely bad layers get touched, cutting P2P/NVLink contention, while
  still fully fixing whichever layers DO qualify (not V2's 1-swap-then-move-on).
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


def _build_layer_swap_sequence(lc, p2l_orig, num_ranks, num_local, threshold_ratio, max_swaps_per_layer):
    """Full multi-round swap loop for ONE layer -- swap until ratio drops
    below threshold_ratio or max_swaps_per_layer rounds are used up.
    Identical algorithm to V1 (proven correct); reused here unchanged."""
    p2l = list(p2l_orig)
    used_slots = set()
    ops = []
    initial_ratio = None
    final_ratio = None

    for _ in range(max_swaps_per_layer):
        gpu_load = [0] * num_ranks
        for r in range(num_ranks):
            s, e = r * num_local, (r + 1) * num_local
            gpu_load[r] = sum(lc[p2l[i]] for i in range(s, e))

        max_load = max(gpu_load)
        avg_load = max(sum(gpu_load) / num_ranks, 1.0)
        ratio = max_load / avg_load
        if initial_ratio is None:
            initial_ratio = ratio
        final_ratio = ratio
        if ratio < threshold_ratio:
            break

        rank_hot = gpu_load.index(max_load)
        rank_cold = gpu_load.index(min(gpu_load))
        if rank_hot == rank_cold:
            break

        hot_start, hot_end = rank_hot * num_local, (rank_hot + 1) * num_local
        hot_candidates = [i for i in range(hot_start, hot_end) if i not in used_slots]
        if not hot_candidates:
            break
        phys_a = max(hot_candidates, key=lambda i: lc[p2l[i]])
        logical_a = p2l[phys_a]

        cold_start, cold_end = rank_cold * num_local, (rank_cold + 1) * num_local
        cold_candidates = [i for i in range(cold_start, cold_end) if i not in used_slots]
        if not cold_candidates:
            break
        phys_b = min(cold_candidates, key=lambda i: lc[p2l[i]])
        logical_b = p2l[phys_b]

        ops.append((phys_a, phys_b, rank_hot, rank_cold, logical_a, logical_b, ratio))
        used_slots.add(phys_a)
        used_slots.add(phys_b)
        p2l[phys_a], p2l[phys_b] = logical_b, logical_a

    return ops, initial_ratio, final_ratio


def try_build_swap_plan_v2(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_ranks: int,
    num_local: int,
    threshold_ratio: float,
    max_swaps_per_layer: int,
    max_total_swap_layers: int = 48,  # safety valve only, not a real budget
) -> List[SwapOp]:
    """V2.1: select layers with ratio > threshold_ratio, fully correct each
    (multi-round) via _build_layer_swap_sequence. No artificial global swap
    budget -- max_total_swap_layers is a safety cap (default 48 = all layers,
    i.e. effectively unbounded) to guard against pathological cases, not a
    tuning knob that starves correction like V2's max_total_swaps=8 did."""
    L = logical_count.shape[0]
    lc_cpu = logical_count.tolist()
    p2l_cpu = physical_to_logical_map.tolist()

    per_layer_result = {}
    for l in range(L):
        ops, initial_ratio, final_ratio = _build_layer_swap_sequence(
            lc_cpu[l], p2l_cpu[l], num_ranks, num_local,
            threshold_ratio, max_swaps_per_layer,
        )
        if ops:
            per_layer_result[l] = (ops, initial_ratio, final_ratio)

    if not per_layer_result:
        return []

    sorted_layers = sorted(per_layer_result.keys(),
                           key=lambda l: per_layer_result[l][1], reverse=True)
    selected_layers = sorted_layers[:max_total_swap_layers]

    candidates = []
    diag_initial, diag_final = [], []
    for l in selected_layers:
        ops, initial_ratio, final_ratio = per_layer_result[l]
        diag_initial.append(initial_ratio)
        diag_final.append(final_ratio)
        for (phys_a, phys_b, rank_hot, rank_cold, logical_a, logical_b, ratio) in ops:
            candidates.append(SwapOp(
                layer_id=l, phys_slot_a=phys_a, phys_slot_b=phys_b,
                rank_a=rank_hot, rank_b=rank_cold,
                logical_a=logical_a, logical_b=logical_b,
                imbalance=ratio,
            ))

    if diag_initial:
        logger.info(
            f"[OEPLB-V2.1] plan: {len(candidates)} swaps across {len(selected_layers)} layers "
            f"(no global cap), avg_ratio_before={sum(diag_initial)/len(diag_initial):.3f} "
            f"avg_ratio_after={sum(diag_final)/len(diag_final):.3f} "
            f"max_ratio_before={max(diag_initial):.3f}"
        )
    return candidates
