"""Ground-truth token routing tracer, independent of PBOEPLBController.

Records, for EVERY forward pass and EVERY layer (no sampling, no decay,
no swap-decision logic involved), two histograms:
  - physical_hist[128]: how many tokens landed on each PHYSICAL slot
    (direct bincount of topk_ids, which are physical slot ids -- see
    controller.py's BUGFIX docstrings for why)
  - logical_hist[128]: how many tokens were destined for each LOGICAL
    expert (physical_hist regathered through the CURRENT physical_to_logical_map)

This lets us separate two effects that are conflated by physical_hist alone:
  - logical_hist reveals genuine CONTENT-DRIVEN expert popularity (should be
    stable across forward passes if the "similar requests route similarly"
    premise holds for this dataset), independent of placement.
  - physical_hist reveals the ACTUAL per-GPU load under the CURRENT placement
    -- this is what swap tries to flatten.

Enabled via env var SGLANG_OEPLB_ROUTING_TRACE=1. Auto-flushes to
{SGLANG_OEPLB_ROUTING_TRACE_DIR}/rank{rank}_chunk{N}.npz every
FLUSH_EVERY records to bound memory.
"""
import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_OEPLB_ROUTING_TRACE", "0") == "1"
_OUT_DIR = os.environ.get("SGLANG_OEPLB_ROUTING_TRACE_DIR", "/tmp/routing_trace")
_FLUSH_EVERY = 4000

_tracer = None


class RoutingTracer:
    def __init__(self, num_layers, num_physical_experts, rank):
        self.num_layers = num_layers
        self.num_physical_experts = num_physical_experts
        self.rank = rank
        self.forward_ids = []
        self.layer_ids = []
        self.physical_hists = []
        self.logical_hists = []
        self.forward_mode_is_prefill = []
        self._layer_counter = 0
        self._forward_id = 0
        self._chunk_idx = 0
        self.current_is_prefill = False
        os.makedirs(_OUT_DIR, exist_ok=True)
        logger.info(f"[ROUTING-TRACER] rank={rank} initialized, dumping to {_OUT_DIR}")

    def on_forward_pass_end(self):
        self._forward_id += 1
        self._layer_counter = 0

    def record(self, topk_ids: torch.Tensor, p2l_layer: torch.Tensor, is_prefill: bool):
        if torch.cuda.is_current_stream_capturing():
            return
        layer_id = self._layer_counter % self.num_layers
        self._layer_counter += 1

        flat = topk_ids.reshape(-1)
        mask = flat != -1
        physical_hist = torch.bincount(
            flat.masked_fill(~mask, 0).long(),
            weights=mask.float(),
            minlength=self.num_physical_experts,
        )
        # logical demand: which logical expert does each hit physical slot
        # currently hold? (gather via CURRENT p2l, then bincount)
        logical_ids_hit = p2l_layer[flat.masked_fill(~mask, 0).long()]
        logical_hist = torch.bincount(
            logical_ids_hit.masked_fill(~mask, 0).long(),
            weights=mask.float(),
            minlength=self.num_physical_experts,
        )

        self.forward_ids.append(self._forward_id)
        self.layer_ids.append(layer_id)
        self.physical_hists.append(physical_hist.cpu().numpy().astype(np.int32))
        self.logical_hists.append(logical_hist.cpu().numpy().astype(np.int32))
        self.forward_mode_is_prefill.append(is_prefill)

        if len(self.forward_ids) >= _FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.forward_ids:
            return
        path = os.path.join(_OUT_DIR, f"rank{self.rank}_chunk{self._chunk_idx}.npz")
        np.savez_compressed(
            path,
            forward_ids=np.array(self.forward_ids, dtype=np.int64),
            layer_ids=np.array(self.layer_ids, dtype=np.int32),
            physical_hists=np.stack(self.physical_hists),
            logical_hists=np.stack(self.logical_hists),
            is_prefill=np.array(self.forward_mode_is_prefill, dtype=bool),
        )
        logger.info(f"[ROUTING-TRACER] rank={self.rank} flushed {len(self.forward_ids)} records to {path}")
        self._chunk_idx += 1
        self.forward_ids = []
        self.layer_ids = []
        self.physical_hists = []
        self.logical_hists = []
        self.forward_mode_is_prefill = []


def get_routing_tracer():
    return _tracer


def init_routing_tracer(num_layers, num_physical_experts, rank):
    global _tracer
    if not _ENABLED:
        return None
    _tracer = RoutingTracer(num_layers, num_physical_experts, rank)
    return _tracer


def is_enabled():
    return _ENABLED
