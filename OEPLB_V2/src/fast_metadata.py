import torch
import torch.nn.functional as F


def fast_init_by_mapping(physical_to_logical_map: torch.Tensor, num_logical_experts: int):
    """
    Fast-path replacement for ExpertLocationMetadata.init_by_mapping(), valid only
    when there are NO redundant physical experts (num_physical_experts ==
    num_logical_experts, a pure bijection -- true whenever ep_num_redundant_experts=0,
    the default and PB-OEPLB's only supported configuration in v0.1-v0.3).

    The official init_by_mapping() calls two Python-for-loop-based helpers
    (_compute_logical_to_all_physical_map, compute_logical_to_rank_dispatch_physical_map)
    written for the "rarely called" official EPLB rebalance cadence (default: every
    1000 forward passes). Measured cost for 48 layers x 128 experts: ~85ms + ~290ms
    = ~375ms, entirely GPU->CPU-sync-bound (thousands of .item() calls in nested
    Python loops). PB-OEPLB calls this on every swap decision (much more frequently),
    so this cost must be eliminated, not just amortized.

    With no redundancy, "logical -> all physical candidates" always has exactly
    ONE candidate (the bijection's inverse), and "logical -> rank-dispatch physical"
    has no ambiguity to resolve via nearest-expert search -- both degenerate to the
    same single vectorized inverse-permutation computation.
    """
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    L, P = physical_to_logical_map.shape
    device = physical_to_logical_map.device

    logical_to_physical = torch.empty(L, num_logical_experts, dtype=torch.int64, device=device)
    layer_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, P)
    logical_to_physical[layer_idx, physical_to_logical_map] = (
        torch.arange(P, device=device).unsqueeze(0).expand(L, P)
    )

    logical_to_all_physical_map = F.pad(
        logical_to_physical.unsqueeze(-1), (0, P - 1), value=-1
    )
    logical_to_all_physical_map_num_valid = torch.ones(
        L, num_logical_experts, dtype=torch.int64, device=device
    )

    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical_map,
        physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_cpu=logical_to_all_physical_map.cpu(),
        logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
        logical_to_rank_dispatch_physical_map=logical_to_physical,
    )
