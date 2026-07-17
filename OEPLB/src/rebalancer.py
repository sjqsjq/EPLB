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
    logical_count_layer: torch.Tensor,
    physical_to_logical_layer: torch.Tensor,
    num_ranks: int,
    num_local: int,
) -> torch.Tensor:
    """GPU-tensor version, kept for diagnostic logging call sites."""
    gpu_load = torch.zeros(num_ranks, dtype=torch.int64,
                           device=logical_count_layer.device)
    for r in range(num_ranks):
        start = r * num_local
        end = (r + 1) * num_local
        logical_ids = physical_to_logical_layer[start:end]
        gpu_load[r] = logical_count_layer[logical_ids].sum()
    return gpu_load


def _build_layer_swap_sequence(lc, p2l_orig, num_ranks, num_local, threshold_ratio, max_swaps_per_layer):
    """Run the full multi-round swap loop for ONE layer. Returns
    (swap_ops_without_layer_id, initial_ratio, final_ratio) — swap_ops is the
    COMPLETE sequence needed to bring this layer's ratio below threshold (or
    as far as max_swaps_per_layer rounds allow), NOT truncated.
    """
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


def try_build_swap_plan(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_ranks: int,
    num_local: int,
    threshold_ratio: float,
    max_swaps_per_layer: int,
    max_total_swap_layers: int = 5,
) -> List[SwapOp]:
    """
    v2 (bug fix): max_total_swap_layers now means "max number of DISTINCT
    LAYERS to touch", not "max total swap operations across all layers".

    BUG in the previous implementation: it collected ALL candidate swaps from
    ALL layers into one flat list, sorted by imbalance ratio, and truncated
    to the top `max_total_swap_layers` OPERATIONS globally. With 48 layers
    and most layers needing 2-3 rounds to converge below threshold, this
    spread a fixed budget of e.g. 48 ops across 43+ layers -- leaving ~1
    op/layer on average, which is often NOT ENOUGH to bring a badly-imbalanced
    layer (e.g. ratio=2.0, needs 3 rounds) fully under threshold. Every layer
    got "partially" corrected instead of the worst layers getting FULLY
    corrected. This directly explains why max_swaps_per_layer (3->16) had
    zero measurable effect: the per-layer round budget was never the binding
    constraint -- the global op-count truncation was.

    Fix: compute each layer's COMPLETE swap sequence independently (already
    self-limiting via the per-layer threshold break), rank layers by their
    INITIAL imbalance ratio, and select the top `max_total_swap_layers`
    LAYERS to actually apply -- each of those selected layers gets its FULL
    sequence of swaps (however many rounds it needed), not a truncated one.
    """
    L = logical_count.shape[0]

    logical_count_cpu = logical_count.tolist()
    p2l_cpu = physical_to_logical_map.tolist()

    per_layer_result = {}  # layer_id -> (ops, initial_ratio, final_ratio)
    for l in range(L):
        ops, initial_ratio, final_ratio = _build_layer_swap_sequence(
            logical_count_cpu[l], p2l_cpu[l], num_ranks, num_local,
            threshold_ratio, max_swaps_per_layer,
        )
        if ops:
            per_layer_result[l] = (ops, initial_ratio, final_ratio)

    if not per_layer_result:
        return []

    # Select which LAYERS to touch: prioritize by initial imbalance (worst first)
    sorted_layers = sorted(per_layer_result.keys(),
                           key=lambda l: per_layer_result[l][1], reverse=True)
    selected_layers = sorted_layers[:max_total_swap_layers]

    candidates = []
    diag_initial_ratios, diag_final_ratios = [], []
    for l in selected_layers:
        ops, initial_ratio, final_ratio = per_layer_result[l]
        diag_initial_ratios.append(initial_ratio)
        diag_final_ratios.append(final_ratio)
        for (phys_a, phys_b, rank_hot, rank_cold, logical_a, logical_b, ratio) in ops:
            candidates.append(SwapOp(
                layer_id=l, phys_slot_a=phys_a, phys_slot_b=phys_b,
                rank_a=rank_hot, rank_b=rank_cold,
                logical_a=logical_a, logical_b=logical_b,
                imbalance=ratio,
            ))

    if diag_initial_ratios:
        skipped = len(per_layer_result) - len(selected_layers)
        logger.info(
            f"[PB-OEPLB-DIAG] layers_touched={len(selected_layers)} "
            f"layers_skipped={skipped} total_ops={len(candidates)} "
            f"avg_ratio_before={sum(diag_initial_ratios)/len(diag_initial_ratios):.3f} "
            f"avg_ratio_after={sum(diag_final_ratios)/len(diag_final_ratios):.3f} "
            f"max_ratio_before={max(diag_initial_ratios):.3f} "
            f"max_ratio_after={max(diag_final_ratios):.3f}"
        )
    return candidates
