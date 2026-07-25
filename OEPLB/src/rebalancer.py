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
    imbalance: float  # imbalance ratio before swap

def compute_gpu_load(
    physical_count_layer: torch.Tensor,
    physical_to_logical_layer: torch.Tensor,
    num_ranks: int,
    num_local: int,
) -> torch.Tensor:
    """GPU-tensor version, kept for diagnostic logging call sites.

    BUGFIX: physical_count_layer (formerly misnamed logical_count_layer) is
    indexed by PHYSICAL SLOT, not logical expert id -- topk_ids reaching
    record_next_layer() have ALREADY been converted logical->physical by
    topk_ids_logical_to_physical() inside fused_topk() (topk.py), since
    ep_dispatch_algorithm="static" is set whenever --enable-pb-oeplb is on.
    Physical slot i belongs to rank i//num_local unconditionally -- no p2l
    gather needed here. physical_to_logical_layer is kept in the signature
    for call-site compatibility but is unused for this sum."""
    gpu_load = torch.zeros(num_ranks, dtype=torch.int64,
                           device=physical_count_layer.device)
    for r in range(num_ranks):
        start = r * num_local
        end = (r + 1) * num_local
        gpu_load[r] = physical_count_layer[start:end].sum()
    return gpu_load


def _build_layer_swap_sequence(lc, p2l_orig, num_ranks, num_local, threshold_ratio, max_swaps_per_layer):
    """Run the full multi-round swap loop for ONE layer. Returns
    (swap_ops_without_layer_id, initial_ratio, final_ratio) — swap_ops is the
    COMPLETE sequence needed to bring this layer's ratio below threshold (or
    as far as max_swaps_per_layer rounds allow), NOT truncated.

    BUGFIX: `lc` (load count) is indexed by PHYSICAL SLOT, not logical expert
    id (see compute_gpu_load docstring for why). `lc[i]` gives the historical
    observed count for physical slot i UNDER THE PLACEMENT AT RECORDING TIME.
    `p2l[i]` looks up which logical expert currently occupies slot i.

    Multi-round simulation within one decision: when we simulate swapping
    phys_a and phys_b, the logical expert identities at those two slots swap
    (p2l update, as before) AND so does their associated historical load --
    slot phys_a's future expected demand becomes what phys_b's expert used to
    draw (lc[phys_b]), and vice versa. Both `lc` (local copy) and `p2l` must be
    swapped together each round, or subsequent rounds' gpu_load computation
    silently ignores all earlier simulated swaps in this loop (found via live
    trace-vs-DIAG discrepancy: avg_ratio_before was equal to avg_ratio_after
    every single decision once lc's own indexing bug was fixed without also
    threading the swap through lc across rounds).
    """
    p2l = list(p2l_orig)
    lc = list(lc)
    used_slots = set()
    ops = []
    initial_ratio = None
    final_ratio = None

    prev_ratio = None
    for _ in range(max_swaps_per_layer):
        gpu_load = [0] * num_ranks
        for r in range(num_ranks):
            s, e = r * num_local, (r + 1) * num_local
            gpu_load[r] = sum(lc[i] for i in range(s, e))

        max_load = max(gpu_load)
        avg_load = max(sum(gpu_load) / num_ranks, 1.0)
        ratio = max_load / avg_load
        if initial_ratio is None:
            initial_ratio = ratio
        final_ratio = ratio

        # Stop if below threshold
        if ratio < threshold_ratio:
            break

        # Stop if no improvement from last swap (avoid infinite loop when
        # threshold is unreachable — e.g. one expert is so hot that no
        # swap can bring ratio below threshold)
        if prev_ratio is not None and ratio >= prev_ratio - 0.001:
            break
        prev_ratio = ratio

        rank_hot = gpu_load.index(max_load)
        rank_cold = gpu_load.index(min(gpu_load))
        if rank_hot == rank_cold:
            break

        hot_start, hot_end = rank_hot * num_local, (rank_hot + 1) * num_local
        hot_candidates = [i for i in range(hot_start, hot_end) if i not in used_slots]
        if not hot_candidates:
            break
        phys_a = max(hot_candidates, key=lambda i: lc[i])
        logical_a = p2l[phys_a]

        cold_start, cold_end = rank_cold * num_local, (rank_cold + 1) * num_local
        cold_candidates = [i for i in range(cold_start, cold_end) if i not in used_slots]
        if not cold_candidates:
            break
        phys_b = min(cold_candidates, key=lambda i: lc[i])
        logical_b = p2l[phys_b]

        ops.append((phys_a, phys_b, rank_hot, rank_cold, logical_a, logical_b, ratio))
        used_slots.add(phys_a)
        used_slots.add(phys_b)
        p2l[phys_a], p2l[phys_b] = logical_b, logical_a
        lc[phys_a], lc[phys_b] = lc[phys_b], lc[phys_a]

    return ops, initial_ratio, final_ratio


def try_build_swap_plan(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_ranks: int,
    num_local: int,
    threshold_ratio: float,
    max_swaps_per_layer: int,
    max_total_swap_layers: int = 5,
    max_total_ops: int = 250,
) -> List[SwapOp]:
    """Greedy global-budget swap planner.

    Instead of processing each layer independently then truncating, this
    version uses a global swap budget and greedily allocates each swap to
    whichever layer currently has the HIGHEST imbalance ratio. After each
    single swap op, the layer's ratio is recomputed and it competes again
    with all other layers for the next swap slot. This ensures the budget
    naturally flows to layers that benefit most, and no single stubborn
    layer can monopolize the budget without actually improving.

    Stops when:
    - Global budget (max_total_ops) exhausted, OR
    - The highest-ratio layer is already below threshold_ratio, OR
    - The best available swap would not improve the highest-ratio layer
    """
    L = logical_count.shape[0]
    lc_cpu = [list(row) for row in logical_count.tolist()]
    p2l_cpu = [list(row) for row in physical_to_logical_map.tolist()]
    used_slots = [set() for _ in range(L)]  # per-layer used slots

    candidates = []
    diag_initial_ratios = {}

    def compute_ratio(layer_id):
        lc = lc_cpu[layer_id]
        loads = [sum(lc[r*num_local:(r+1)*num_local]) for r in range(num_ranks)]
        avg = max(sum(loads) / num_ranks, 1.0)
        return max(loads) / avg

    # Initialize ratios for all layers
    layer_ratios = {l: compute_ratio(l) for l in range(L)}

    for _ in range(max_total_ops):
        # Pick the layer with highest current ratio
        best_layer = max(layer_ratios, key=layer_ratios.get)
        ratio = layer_ratios[best_layer]

        if ratio < threshold_ratio:
            break  # all layers below threshold

        if best_layer not in diag_initial_ratios:
            diag_initial_ratios[best_layer] = ratio

        # Try one swap on this layer
        lc = lc_cpu[best_layer]
        p2l = p2l_cpu[best_layer]
        used = used_slots[best_layer]

        loads = [sum(lc[r*num_local:(r+1)*num_local]) for r in range(num_ranks)]
        rank_hot = loads.index(max(loads))
        rank_cold = loads.index(min(loads))

        if rank_hot == rank_cold:
            layer_ratios[best_layer] = 0  # mark as done
            continue

        hot_s, hot_e = rank_hot * num_local, (rank_hot + 1) * num_local
        hot_cands = [i for i in range(hot_s, hot_e) if i not in used]
        if not hot_cands:
            layer_ratios[best_layer] = 0
            continue

        cold_s, cold_e = rank_cold * num_local, (rank_cold + 1) * num_local
        cold_cands = [i for i in range(cold_s, cold_e) if i not in used]
        if not cold_cands:
            layer_ratios[best_layer] = 0
            continue

        phys_a = max(hot_cands, key=lambda i: lc[i])
        phys_b = min(cold_cands, key=lambda i: lc[i])
        logical_a = p2l[phys_a]
        logical_b = p2l[phys_b]

        # Execute swap in simulation
        p2l[phys_a], p2l[phys_b] = logical_b, logical_a
        lc[phys_a], lc[phys_b] = lc[phys_b], lc[phys_a]
        used.add(phys_a)
        used.add(phys_b)

        new_ratio = compute_ratio(best_layer)

        # No-improvement check: if swap didn't help, undo and mark layer done
        if new_ratio >= ratio - 0.001:
            p2l[phys_a], p2l[phys_b] = logical_a, logical_b
            lc[phys_a], lc[phys_b] = lc[phys_b], lc[phys_a]
            used.discard(phys_a)
            used.discard(phys_b)
            layer_ratios[best_layer] = 0
            continue

        candidates.append(SwapOp(
            layer_id=best_layer, phys_slot_a=phys_a, phys_slot_b=phys_b,
            rank_a=rank_hot, rank_b=rank_cold,
            logical_a=logical_a, logical_b=logical_b,
            imbalance=ratio,
        ))
        layer_ratios[best_layer] = new_ratio

    if candidates and diag_initial_ratios:
        final_ratios = {l: compute_ratio(l) for l in diag_initial_ratios}
        layers_touched = len(diag_initial_ratios)
        avg_before = sum(diag_initial_ratios.values()) / len(diag_initial_ratios)
        avg_after = sum(final_ratios[l] for l in diag_initial_ratios) / len(diag_initial_ratios)
        max_before = max(diag_initial_ratios.values())
        max_after = max(final_ratios[l] for l in diag_initial_ratios)
        logger.info(
            f"[PB-OEPLB-DIAG] layers_touched={layers_touched} "
            f"total_ops={len(candidates)} "
            f"avg_ratio_before={avg_before:.3f} "
            f"avg_ratio_after={avg_after:.3f} "
            f"max_ratio_before={max_before:.3f} "
            f"max_ratio_after={max_after:.3f}"
        )
    return candidates