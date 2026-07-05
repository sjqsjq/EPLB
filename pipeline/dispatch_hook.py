"""
Dispatch Hook: remaps topk_ids using the REAL physical-to-logical mapping.

With redundant experts, some experts exist on multiple GPUs.
This hook routes tokens to the GPU with the least load by picking
the physical replica on the target GPU.
"""
import logging
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_remap_count = 0
_fallback_count = 0
_lookup_cache = {}


def _build_lookup_from_sglang(num_physical: int, num_gpus: int, device, layer_id: int = 0):
    """Build lookup table from SGLang's actual physical-to-logical mapping.
    
    Rebuilds each time since expert placement changes per layer after swaps.
    """
    cache_key = (num_physical, num_gpus, str(device), layer_id)
    if cache_key in _lookup_cache:
        return _lookup_cache[cache_key]
    
    try:
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return None
        
        phy2log = metadata.physical_to_logical_map_cpu  # [layers, num_physical]
        # Use specific layer since swaps change layout per-layer
        if layer_id < phy2log.shape[0]:
            phy2log_l0 = phy2log[layer_id].numpy()
        else:
            phy2log_l0 = phy2log[0].numpy()
        
        num_logical = int(phy2log_l0.max()) + 1
        per_gpu = num_physical // num_gpus
        
        lookup = np.full((num_logical, num_gpus), -1, dtype=np.int64)
        
        for phy_id in range(num_physical):
            log_id = int(phy2log_l0[phy_id])
            if log_id < 0 or log_id >= num_logical:
                continue
            gpu = phy_id // per_gpu
            if gpu < num_gpus:
                lookup[log_id, gpu] = phy_id
        
        has_multi = (lookup >= 0).sum(axis=1)
        replicated = (has_multi > 1).sum()
        if len(_lookup_cache) < 4:
            logger.info(
                f"[DispatchHook] Built lookup layer {layer_id}: "
                f"{replicated} experts with replicas, per_gpu={per_gpu}"
            )
        
        result = torch.from_numpy(lookup).to(device)
        _lookup_cache[cache_key] = result
        return result
        
    except Exception as e:
        logger.warning(f"[DispatchHook] Failed to build lookup: {e}")
        return None


def remap_topk_ids(
    topk_ids: torch.Tensor,
    token_to_gpu: np.ndarray,
    num_physical: int,
    num_gpus: int,
    layer_id: int = 0,
) -> torch.Tensor:
    """Remap topk_ids to route tokens to target GPUs using real replicas."""
    global _remap_count, _fallback_count
    
    B, K = topk_ids.shape
    device = topk_ids.device
    per_gpu = num_physical // num_gpus
    
    # Get or build lookup from SGLang's actual mapping
    lookup = _build_lookup_from_sglang(num_physical, num_gpus, device, layer_id)
    if lookup is None:
        return topk_ids
    
    num_logical = lookup.shape[0]
    
    # Convert physical IDs back to logical
    # In trivial mapping: logical = physical % num_logical
    logical_ids = topk_ids % num_logical  # [B, K]
    
    # Target GPU per token
    if len(token_to_gpu) < B:
        padded = np.zeros(B, dtype=np.int64)
        padded[:len(token_to_gpu)] = token_to_gpu
        padded[len(token_to_gpu):] = (topk_ids[len(token_to_gpu):, 0].cpu().numpy() // per_gpu)
        token_to_gpu = padded
    
    target_gpu = torch.from_numpy(token_to_gpu[:B].astype(np.int64)).to(device)
    target_gpu_exp = target_gpu.unsqueeze(1).expand(-1, K)  # [B, K]
    
    # Lookup: for each (logical_expert, target_gpu), get physical ID
    flat_log = logical_ids.reshape(-1).clamp(0, num_logical - 1)
    flat_tgt = target_gpu_exp.reshape(-1).clamp(0, num_gpus - 1)
    
    remapped_flat = lookup[flat_log, flat_tgt]  # [B*K]
    remapped = remapped_flat.reshape(B, K)
    
    mask_valid = remapped >= 0
    new_topk_ids = torch.where(mask_valid, remapped, topk_ids)
    
    _remap_count += int(mask_valid.sum().item())
    _fallback_count += int((~mask_valid).sum().item())
    
    total = _remap_count + _fallback_count
    if total > 0 and total % 100000 < B * K:
        logger.info(
            f"[DispatchHook] remap rate: {_remap_count/total*100:.1f}% "
            f"({_remap_count}/{total})"
        )
    
    return new_topk_ids


def create_pipeline_dispatch_hook(num_experts: int, num_gpus: int):
    """Create pre_dispatch_hook that uses pipeline algorithm results."""
    
    # With redundant experts, num_physical > num_experts
    # We'll detect it from SGLang metadata at runtime
    
    def hook(dispatcher, hidden_states, topk_output):
        if torch.cuda.is_current_stream_capturing():
            return hidden_states, topk_output
        
        import sys
        sys.path.insert(0, "/workspace/EPLB")
        from pipeline.pipeline_manager import PipelineManager
        
        pm = PipelineManager.get_instance()
        if pm is None:
            return hidden_states, topk_output
        
        result = pm.get_last_result()
        if result is None or result.get("token_to_gpu") is None:
            return hidden_states, topk_output
        
        if not hasattr(topk_output, 'topk_ids') or topk_output.topk_ids is None:
            return hidden_states, topk_output
        
        try:
            num_physical = getattr(dispatcher, 'num_experts', num_experts)
            
            # Get current layer from pipeline manager
            current_layer = getattr(pm, '_current_layer', 0)
            
            new_topk_ids = remap_topk_ids(
                topk_output.topk_ids,
                result["token_to_gpu"],
                num_physical,
                num_gpus,
                layer_id=current_layer,
            )
            
            from sglang.srt.layers.moe.topk import StandardTopKOutput
            new_topk_output = StandardTopKOutput(
                topk_weights=topk_output.topk_weights,
                topk_ids=new_topk_ids,
                router_logits=topk_output.router_logits,
            )
            return hidden_states, new_topk_output
            
        except Exception as e:
            global _remap_count
            if _remap_count < 10:
                logger.warning(f"[DispatchHook] remap failed: {e}")
            return hidden_states, topk_output
    
    return hook
