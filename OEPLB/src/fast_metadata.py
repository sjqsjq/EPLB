import torch
import torch.nn.functional as F


def fast_init_by_mapping(physical_to_logical_map: torch.Tensor, num_logical_experts: int, existing_meta=None):
    """
    Fast-path replacement for ExpertLocationMetadata.init_by_mapping(), valid only
    when there are NO redundant physical experts.
    If existing_meta is provided, match its None-ness for optional fields.
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

    # Match None-ness of optional fields from existing metadata
    rank_dispatch = logical_to_physical
    if existing_meta is not None and existing_meta.logical_to_rank_dispatch_physical_map is None:
        rank_dispatch = None

    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical_map,
        physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_cpu=logical_to_all_physical_map.cpu(),
        logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
        logical_to_rank_dispatch_physical_map=rank_dispatch,
    )
