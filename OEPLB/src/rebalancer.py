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


def compute_gpu_load(
    physical_count_layer: torch.Tensor,
    physical_to_logical_layer: torch.Tensor,
    num_ranks: int,
    num_local: int,
) -> torch.Tensor:
    """GPU-tensor version for diagnostic logging."""
    gpu_load = torch.zeros(num_ranks, dtype=torch.int64,
                           device=physical_count_layer.device)
    for r in range(num_ranks):
        start = r * num_local
        end = (r + 1) * num_local
        gpu_load[r] = physical_count_layer[start:end].sum()
    return gpu_load


def try_build_swap_plan(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_ranks: int,
    num_local: int,
    threshold_ratio: float,
    max_swaps_per_layer: int,
    max_total_swap_layers: int = 94,
    max_total_ops: int = 250,
) -> List[SwapOp]:
    """Greedy global-budget swap planner with exhaustive per-layer search.

    Core loop: pick the layer with highest imbalance ratio, try to find a
    swap pair that improves it. If the obvious "hottest-on-max-rank swap
    coldest-on-min-rank" doesn't work, try other candidate pairs before
    giving up. A layer is only marked exhausted when no remaining untried
    pair can improve its ratio.

    This fixes a critical issue in the previous version: one failed pair
    would mark the entire layer as "done", causing layers with diffuse
    imbalance (spread across many slots rather than one extreme hot-spot)
    to never get effectively rebalanced, leading to steadily climbing
    max_ratio over successive windows.
    """
    L = logical_count.shape[0]
    lc_cpu = [list(row) for row in logical_count.tolist()]
    p2l_cpu = [list(row) for row in physical_to_logical_map.tolist()]

    # Per-layer tracking: slots that have been tried (successfully swapped
    # OR attempted and failed) — prevents re-trying the same pair.
    tried_hot = [set() for _ in range(L)]
    tried_cold = [set() for _ in range(L)]
    exhausted_layers = set()

    candidates = []
    diag_initial_ratios = {}

    def compute_ratio(layer_id):
        lc = lc_cpu[layer_id]
        loads = [sum(lc[r*num_local:(r+1)*num_local]) for r in range(num_ranks)]
        avg = max(sum(loads) / num_ranks, 1.0)
        return max(loads) / avg

    def get_rank_loads(layer_id):
        lc = lc_cpu[layer_id]
        return [sum(lc[r*num_local:(r+1)*num_local]) for r in range(num_ranks)]

    layer_ratios = {l: compute_ratio(l) for l in range(L)}
    layers_touched_set = set()

    for _ in range(max_total_ops):
        # Pick layer with highest ratio (excluding exhausted)
        best_layer = None
        best_ratio = 0
        for l, r in layer_ratios.items():
            if l not in exhausted_layers and r > best_ratio:
                best_layer = l
                best_ratio = r
        if best_layer is None or best_ratio < threshold_ratio:
            break

        if best_layer not in diag_initial_ratios:
            diag_initial_ratios[best_layer] = best_ratio

        if len(layers_touched_set) >= max_total_swap_layers and best_layer not in layers_touched_set:
            break

        ratio = best_ratio
        lc = lc_cpu[best_layer]
        p2l = p2l_cpu[best_layer]

        # Try to find a swap pair that improves this layer.
        # Strategy: iterate through hot ranks (sorted by load desc) and
        # cold ranks (sorted by load asc), trying untried slot pairs.
        loads = get_rank_loads(best_layer)
        avg_load = max(sum(loads) / num_ranks, 1.0)

        # Ranks sorted by load (high to low for hot, low to high for cold)
        ranks_by_load_desc = sorted(range(num_ranks), key=lambda r: loads[r], reverse=True)
        ranks_by_load_asc = sorted(range(num_ranks), key=lambda r: loads[r])

        found_swap = False
        for rank_hot in ranks_by_load_desc:
            if loads[rank_hot] / avg_load < threshold_ratio:
                break  # no rank above threshold anymore

            hot_s, hot_e = rank_hot * num_local, (rank_hot + 1) * num_local
            # Hot candidates: untried slots on hot rank, sorted by load desc
            hot_cands = sorted(
                [i for i in range(hot_s, hot_e) if i not in tried_hot[best_layer]],
                key=lambda i: lc[i], reverse=True
            )
            if not hot_cands:
                continue

            for rank_cold in ranks_by_load_asc:
                if rank_cold == rank_hot:
                    continue
                if loads[rank_cold] >= loads[rank_hot]:
                    break  # cold rank isn't actually lighter

                cold_s, cold_e = rank_cold * num_local, (rank_cold + 1) * num_local
                cold_cands = sorted(
                    [i for i in range(cold_s, cold_e) if i not in tried_cold[best_layer]],
                    key=lambda i: lc[i]
                )
                if not cold_cands:
                    continue

                # Select the pair with maximum load difference
                # (hot_cands already sorted desc, cold_cands sorted asc,
                # so [0]×[0] IS the max-delta pair for this rank combination)
                phys_a = hot_cands[0]
                phys_b = cold_cands[0]

                # Simulate swap
                lc[phys_a], lc[phys_b] = lc[phys_b], lc[phys_a]
                p2l[phys_a], p2l[phys_b] = p2l[phys_b], p2l[phys_a]
                new_ratio = compute_ratio(best_layer)

                if new_ratio < ratio - 0.0005:
                    # Accept swap
                    candidates.append(SwapOp(
                        layer_id=best_layer, phys_slot_a=phys_a, phys_slot_b=phys_b,
                        rank_a=rank_hot, rank_b=rank_cold,
                        logical_a=p2l[phys_b], logical_b=p2l[phys_a],
                        imbalance=ratio,
                    ))
                    tried_hot[best_layer].add(phys_a)
                    tried_cold[best_layer].add(phys_b)
                    layer_ratios[best_layer] = new_ratio
                    layers_touched_set.add(best_layer)
                    found_swap = True
                    break
                else:
                    # Undo swap, mark these slots as tried
                    lc[phys_a], lc[phys_b] = lc[phys_b], lc[phys_a]
                    p2l[phys_a], p2l[phys_b] = p2l[phys_b], p2l[phys_a]
                    tried_hot[best_layer].add(phys_a)
                    tried_cold[best_layer].add(phys_b)

            if found_swap:
                break

        if not found_swap:
            # 3-way cyclic rotation fallback: when no pairwise swap helps,
            # try rotating 3 slots across 3 ranks to break diffuse plateaus.
            # Decomposed into 2 sequential pairwise SwapOps (safe with the
            # evolving-p2l fix in controller.py's try_finish).
            found_3way = False
            if len(ranks_by_load_desc) >= 3 and ratio > threshold_ratio:
                rank_hot = ranks_by_load_desc[0]
                rank_mid = ranks_by_load_desc[len(ranks_by_load_desc)//2]
                rank_cold = ranks_by_load_asc[0]
                if len(set([rank_hot, rank_mid, rank_cold])) == 3:
                    hot_s, hot_e = rank_hot * num_local, (rank_hot+1) * num_local
                    mid_s, mid_e = rank_mid * num_local, (rank_mid+1) * num_local
                    cold_s, cold_e = rank_cold * num_local, (rank_cold+1) * num_local
                    h_cands = [i for i in range(hot_s, hot_e) if i not in tried_hot[best_layer]]
                    m_cands = [i for i in range(mid_s, mid_e) if i not in tried_hot[best_layer] and i not in tried_cold[best_layer]]
                    c_cands = [i for i in range(cold_s, cold_e) if i not in tried_cold[best_layer]]
                    if h_cands and m_cands and c_cands:
                        phys_h = max(h_cands, key=lambda i: lc[i])
                        phys_m = min(m_cands, key=lambda i: lc[i])
                        phys_c = min(c_cands, key=lambda i: lc[i])
                        # Simulate rotation: h=m_orig, m=c_orig, c=h_orig
                        old_h, old_m, old_c = lc[phys_h], lc[phys_m], lc[phys_c]
                        lc[phys_h], lc[phys_m], lc[phys_c] = old_m, old_c, old_h
                        p2l[phys_h], p2l[phys_m], p2l[phys_c] = p2l[phys_h], p2l[phys_m], p2l[phys_c]  # p2l handled by controller
                        new_ratio = compute_ratio(best_layer)
                        if new_ratio < ratio - 0.0005:
                            # Decompose into 2 swaps: swap(h,c) then swap(m,h)
                            # After swap(h,c): phys_h has c_orig, phys_c has h_orig
                            # After swap(m,h): phys_m has h_current(=c_orig), phys_h has m_orig
                            # Final: phys_h=m_orig, phys_m=c_orig, phys_c=h_orig ✓
                            candidates.append(SwapOp(
                                layer_id=best_layer, phys_slot_a=phys_h, phys_slot_b=phys_c,
                                rank_a=rank_hot, rank_b=rank_cold,
                                logical_a=0, logical_b=0,  # unused, controller reads from p2l
                                imbalance=ratio,
                            ))
                            candidates.append(SwapOp(
                                layer_id=best_layer, phys_slot_a=phys_m, phys_slot_b=phys_h,
                                rank_a=rank_mid, rank_b=rank_hot,
                                logical_a=0, logical_b=0,  # unused
                                imbalance=ratio,
                            ))
                            tried_hot[best_layer].add(phys_h)
                            tried_hot[best_layer].add(phys_m)
                            tried_cold[best_layer].add(phys_c)
                            layer_ratios[best_layer] = new_ratio
                            layers_touched_set.add(best_layer)
                            found_3way = True
                        else:
                            # Undo
                            lc[phys_h], lc[phys_m], lc[phys_c] = old_h, old_m, old_c

            if not found_3way:
                exhausted_layers.add(best_layer)

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
