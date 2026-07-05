"""
WeightSwapper: per-layer P2P expert weight transfer.

Uses our OWN simple diff-based P2P executor (simple_p2p_swap.py) instead of
SGLang's general-purpose update_expert_weights_single_layer, whose chunk-based
multi-rank fan-out logic (_ChunkUtils) produced asymmetric send/recv op counts
across ranks for our specific (sparse, low-volume) migration pattern, causing
a permanent batch_isend_irecv deadlock under heavy concurrent load. Since our
own migration ops always resolve to simple, deterministic 1:1 slot pairings
computed identically (Gloo-synced) on every rank, a direct diff-based pairing
is both simpler and provably symmetric.
"""
import logging
import time
from typing import Dict, List

import torch
import torch.distributed

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from .simple_p2p_swap import simple_p2p_swap

logger = logging.getLogger(__name__)


class WeightSwapper:
    def __init__(self, num_experts: int, num_gpus: int, num_local_physical: int):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.num_local_physical = num_local_physical
        self.num_physical = num_local_physical * num_gpus

        self._rank = None
        self._initialized = False
        self._total_swaps = 0
        self._total_time_us = 0.0
        self._call_count = 0

    def _lazy_init(self):
        if self._initialized:
            return
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._initialized = True
        logger.info(f"[WeightSwapper] init: rank={self._rank}, local={self.num_local_physical}")

    def swap_for_layer(
        self,
        layer_id: int,
        new_phy2log: List[int],
        expert_weights_of_layer: Dict[int, List[torch.Tensor]],
        gloo_group=None,
    ) -> bool:
        if layer_id not in expert_weights_of_layer:
            return False

        weights = expert_weights_of_layer[layer_id]
        self._lazy_init()

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return False

        old_phy2log = metadata.physical_to_logical_map_cpu[layer_id].tolist()
        changes = sum(1 for a, b in zip(old_phy2log, new_phy2log) if a != b)
        if changes == 0:
            return True

        t0 = time.perf_counter()
        try:
            num_transfers = simple_p2p_swap(
                old_map=old_phy2log,
                new_map=new_phy2log,
                weights=weights,
                rank=self._rank,
                num_gpus=self.num_gpus,
                per_gpu=self.num_local_physical,
                gloo_group=gloo_group,
            )

            new_map = torch.tensor(new_phy2log, dtype=torch.long,
                                   device=metadata.physical_to_logical_map.device)
            metadata.physical_to_logical_map[layer_id] = new_map
            metadata.physical_to_logical_map_cpu[layer_id] = new_map.cpu()

            from pipeline.dispatch_hook import _lookup_cache
            _lookup_cache.clear()

            elapsed_us = (time.perf_counter() - t0) * 1e6
            self._total_time_us += elapsed_us
            self._call_count += 1
            self._total_swaps += changes

            if self._call_count % 200 == 1:
                logger.info(
                    f"[WeightSwapper] layer {layer_id}: {changes} changes, "
                    f"{num_transfers} P2P transfers, {elapsed_us:.0f}us, total={self._total_swaps}"
                )
            return True

        except Exception as e:
            if self._call_count < 5:
                logger.error(f"[WeightSwapper] layer {layer_id} failed: {e}")
            return False

    def get_stats(self):
        n = self._call_count
        avg = self._total_time_us / n if n > 0 else 0
        return {
            "call_count": n, "total_swaps": self._total_swaps,
            "total_time_ms": round(self._total_time_us / 1000, 2),
            "avg_us": round(avg, 1),
        }
