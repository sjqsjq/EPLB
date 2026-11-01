"""Per-request routing tracer for prefill vs decode similarity analysis.

Records per-forward-pass histograms so we can match each request's
prefill routing to its subsequent decode steps.

Enabled via: SGLANG_PER_REQUEST_TRACE=1
Output dir: SGLANG_PER_REQUEST_TRACE_DIR (default /tmp/per_request_trace)
"""
import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_PER_REQUEST_TRACE", "0") == "1"
_OUT_DIR = os.environ.get("SGLANG_PER_REQUEST_TRACE_DIR", "/tmp/per_request_trace")
_FLUSH_EVERY = 200  # flush every N forward passes

_recorder = None


class PerRequestTracer:
    """Records per-forward-pass, per-layer expert routing histograms."""

    def __init__(self, num_layers, num_experts, rank):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.rank = rank
        self._forward_id = 0
        self._layer_counter = 0
        self._is_prefill = False
        self._chunk_idx = 0

        # Per-forward records
        self.records = []  # list of (forward_id, is_prefill, layer_hists[num_layers x num_experts])
        self._current_layers = np.zeros((num_layers, num_experts), dtype=np.int32)

        os.makedirs(_OUT_DIR, exist_ok=True)
        logger.info(f"[PER-REQ-TRACER] rank={rank} L={num_layers} E={num_experts} dir={_OUT_DIR}")

    def set_prefill(self, is_prefill: bool):
        self._is_prefill = is_prefill

    def record_layer(self, topk_ids: torch.Tensor):
        """Called once per layer per forward pass."""
        if torch.cuda.is_current_stream_capturing():
            return
        layer_id = self._layer_counter % self.num_layers
        self._layer_counter += 1

        flat = topk_ids.reshape(-1)
        mask = flat != -1
        hist = torch.bincount(
            flat.masked_fill(~mask, 0).long(),
            weights=mask.float(),
            minlength=self.num_experts,
        ).cpu().numpy().astype(np.int32)

        self._current_layers[layer_id] = hist

    def on_forward_pass_end(self):
        """Called at end of each forward pass."""
        self.records.append({
            'forward_id': self._forward_id,
            'is_prefill': self._is_prefill,
            'layer_hists': self._current_layers.copy(),
        })
        self._forward_id += 1
        self._layer_counter = 0
        self._current_layers[:] = 0

        if len(self.records) >= _FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.records:
            return
        path = os.path.join(_OUT_DIR, f"rank{self.rank}_chunk{self._chunk_idx}.npz")
        forward_ids = np.array([r['forward_id'] for r in self.records], dtype=np.int64)
        is_prefill = np.array([r['is_prefill'] for r in self.records], dtype=bool)
        layer_hists = np.stack([r['layer_hists'] for r in self.records])  # [N, num_layers, num_experts]
        np.savez_compressed(path, forward_ids=forward_ids, is_prefill=is_prefill, layer_hists=layer_hists)
        logger.info(f"[PER-REQ-TRACER] rank={self.rank} flushed {len(self.records)} forwards to {path}")
        self._chunk_idx += 1
        self.records = []

    def finalize(self):
        self.flush()


def get_per_request_tracer():
    return _recorder


def init_per_request_tracer(num_layers, num_experts, rank):
    global _recorder
    if not _ENABLED:
        return None
    _recorder = PerRequestTracer(num_layers, num_experts, rank)
    return _recorder


def is_enabled():
    return _ENABLED
