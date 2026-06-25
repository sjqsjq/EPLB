# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
#
# GPU-only MoE routing distribution recorder.
#
# Purpose:
#   Record per-layer, per-expert token routing counts by hooking the gate
#   output (topk_ids) in each MoE layer's pre-dispatch hook. All statistics
#   are accumulated on GPU via scatter_add_ — zero GPU→CPU sync during
#   inference — so CUDA Graph capture is never broken.
#
#   Dump is triggered lazily: a background thread checks a flag file
#   (touched by the user or a control script) and performs a single
#   GPU→CPU copy + torch.save at that point.
#
# Activated only when SGLANG_ROUTING_RECORD=1.
# ==============================================================================

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)


# ===================== CUDA stream capture detection =====================

_cudart = None


def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _cudart = ctypes.CDLL(name)
            return _cudart
        except OSError:
            continue
    _cudart = False
    return None


def _is_any_stream_capturing() -> bool:
    """Check if current CUDA stream is in graph-capture mode."""
    lib = _load_cudart()
    if not lib:
        try:
            return torch.cuda.is_current_stream_capturing()
        except Exception:
            return False
    try:
        stream_ptr = torch.cuda.current_stream().cuda_stream
        capture_status = ctypes.c_int(0)
        ret = lib.cudaStreamIsCapturing(
            ctypes.c_void_p(stream_ptr), ctypes.byref(capture_status)
        )
        return ret == 0 and capture_status.value == 1
    except Exception:
        return False


# ===================== env helpers =====================

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_routing_record_enabled() -> bool:
    return _env_flag("SGLANG_ROUTING_RECORD", default=False)


# ===================== Global GPU accumulator =====================

def _get_is_extend_in_batch() -> bool:
    """Check if current batch contains extend (prefill) tokens.

    Uses SGLang's internal DP attention state. Returns False if unavailable.
    """
    try:
        from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper
        return _DpGatheredBufferWrapper.get_is_extend_in_batch()
    except Exception:
        return False


class RoutingRecorder:
    """Process-wide singleton that accumulates routing counts on GPU.

    Maintains separate counters for prefill and decode phases:
      - prefill_counts[num_layers, num_experts]: extend (prefill/chunked-prefill)
      - decode_counts[num_layers, num_experts]: decode-only batches

    Each MoE layer's hook calls `record(layer_id, topk_ids, is_prefill)`
    which does a pure-GPU scatter_add_ — no CPU sync, no stream capture conflict.
    """

    _instance: Optional["RoutingRecorder"] = None
    _lock = threading.Lock()

    def __init__(self, num_layers: int, num_experts: int, device: torch.device):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device

        shape = (num_layers, num_experts)
        self._prefill_counts = torch.zeros(shape, dtype=torch.int64, device=device)
        self._decode_counts = torch.zeros(shape, dtype=torch.int64, device=device)

        self._total_steps = 0
        self._prefill_steps = 0
        self._decode_steps = 0
        self._recording = True

        # dump control
        self._output_dir = os.environ.get(
            "SGLANG_ROUTING_RECORD_DIR", "/tmp/routing_record"
        )
        self._dump_flag_path = os.path.join(self._output_dir, ".dump_trigger")
        self._start_dump_watcher()

        logger.info(
            "[RoutingRecorder] Initialized: layers=%d experts=%d device=%s output=%s",
            num_layers, num_experts, device, self._output_dir,
        )

    @classmethod
    def get_or_create(
        cls, num_layers: int, num_experts: int, device: torch.device
    ) -> "RoutingRecorder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(num_layers, num_experts, device)
            return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["RoutingRecorder"]:
        return cls._instance

    def record(self, layer_id: int, topk_ids: torch.Tensor, is_prefill: bool) -> None:
        """Accumulate routing counts — pure GPU ops, no CPU sync.

        Args:
            layer_id: MoE layer index (0-based across the full model).
            topk_ids: [num_tokens, top_k] tensor of expert indices.
            is_prefill: True for extend/prefill batch, False for decode.
        """
        if not self._recording:
            return

        flat_ids = topk_ids.reshape(-1)
        valid_mask = flat_ids >= 0
        valid_ids = flat_ids[valid_mask].to(torch.int64)

        if valid_ids.numel() == 0:
            return

        ones = torch.ones_like(valid_ids)
        target = self._prefill_counts if is_prefill else self._decode_counts
        target[layer_id].scatter_add_(0, valid_ids, ones)

    def increment_step(self, is_prefill: bool) -> None:
        self._total_steps += 1
        if is_prefill:
            self._prefill_steps += 1
        else:
            self._decode_steps += 1

    def dump(self) -> str:
        """One-shot GPU→CPU copy and save. Returns the output file path."""
        prefill_cpu = self._prefill_counts.cpu().clone()
        decode_cpu = self._decode_counts.cpu().clone()

        os.makedirs(self._output_dir, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        filename = f"routing_rank{rank}_{timestamp}_p{self._prefill_steps}d{self._decode_steps}.pt"
        filepath = os.path.join(self._output_dir, filename)

        data = {
            "prefill_counts": prefill_cpu,   # [num_layers, num_experts]
            "decode_counts": decode_cpu,     # [num_layers, num_experts]
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "total_steps": self._total_steps,
            "prefill_steps": self._prefill_steps,
            "decode_steps": self._decode_steps,
            "rank": rank,
        }
        torch.save(data, filepath)
        logger.info(
            "[RoutingRecorder] Dumped to %s (steps: total=%d prefill=%d decode=%d)",
            filepath, self._total_steps, self._prefill_steps, self._decode_steps,
        )
        return filepath

    def reset(self) -> None:
        """Zero all counters."""
        self._prefill_counts.zero_()
        self._decode_counts.zero_()
        self._total_steps = 0
        self._prefill_steps = 0
        self._decode_steps = 0
        logger.info("[RoutingRecorder] Counters reset.")

    def start(self) -> None:
        self._recording = True
        logger.info("[RoutingRecorder] Recording started.")

    def stop(self) -> None:
        self._recording = False
        logger.info("[RoutingRecorder] Recording stopped.")

    # ------------- dump watcher thread -------------

    def _start_dump_watcher(self) -> None:
        """Background thread that watches for a trigger file to dump."""
        os.makedirs(self._output_dir, exist_ok=True)
        thread = threading.Thread(
            target=self._dump_watcher_loop, daemon=True, name="routing-dump-watcher"
        )
        thread.start()

    def _dump_watcher_loop(self) -> None:
        while True:
            time.sleep(2)
            if os.path.exists(self._dump_flag_path):
                try:
                    os.remove(self._dump_flag_path)
                    self.dump()
                except Exception as exc:
                    logger.warning("[RoutingRecorder] Dump watcher error: %s", exc)


# ===================== Forward-entry hook (recommended) =====================
#
# This is the PREFERRED hook point: called directly from FusedMoE.forward()
# at its entry, BEFORE dispatcher.dispatch(). At this point topk_ids contains
# the GATE'S ORIGINAL global expert IDs (0..num_experts-1), not the EP-local
# IDs that would appear after dispatcher remapping.
#
# This solves the problem where Layer 4-60 only see 8 EP-local experts in
# DeepEP low_latency mode.
# =============================================================================

# Track which layer_id is responsible for incrementing step counters.
# Set lazily on the first call to record_routing_from_forward().
_step_tracker_layer_id: Optional[int] = None


def record_routing_from_forward(layer_id: int, topk_ids: "torch.Tensor") -> None:
    """Record routing counts directly from FusedMoE.forward() entry.

    Args:
        layer_id: MoE layer index (global, 0-based across the full model).
        topk_ids: [num_tokens, top_k] tensor — gate's original global expert IDs.

    This is called from layer.py's forward() at the very beginning, before
    dispatcher.dispatch() has any chance to remap the expert IDs.

    Safe during CUDA Graph capture (skipped). Pure-GPU scatter_add_, no CPU sync.
    """
    if not is_routing_record_enabled():
        return
    if _is_any_stream_capturing():
        return
    if topk_ids is None or not isinstance(topk_ids, torch.Tensor):
        return

    recorder = RoutingRecorder.get_instance()
    if recorder is None:
        return

    is_prefill = _get_is_extend_in_batch()
    recorder.record(layer_id, topk_ids, is_prefill)

    # First layer to call this becomes the step tracker (per process).
    global _step_tracker_layer_id
    if _step_tracker_layer_id is None:
        _step_tracker_layer_id = layer_id
    if layer_id == _step_tracker_layer_id:
        recorder.increment_step(is_prefill)


# ===================== Init-time entry point =====================
#
# Called from FusedMoE.__init__ to ensure the global RoutingRecorder singleton
# is created on first MoE layer init. No dispatcher hook is registered anymore
# because it would capture EP-local IDs (post-dispatch) instead of gate's
# original global IDs.
# =============================================================================

def maybe_attach_routing_recorder(fused_moe_layer) -> None:
    """Initialize RoutingRecorder singleton on first MoE layer construction.

    No dispatcher hook is registered — the actual recording happens via
    record_routing_from_forward() called from FusedMoE.forward() entry.
    No-op when SGLANG_ROUTING_RECORD != 1.
    """
    if not is_routing_record_enabled():
        return

    layer_id = getattr(fused_moe_layer, "layer_id", -1)
    num_experts = getattr(fused_moe_layer, "num_experts", 0)
    num_local_experts = getattr(fused_moe_layer, "num_local_experts", 0)

    if layer_id < 0 or num_experts <= 0:
        return

    num_layers = int(os.environ.get("SGLANG_ROUTING_RECORD_NUM_LAYERS", "61"))

    device = next(
        (p.device for p in fused_moe_layer.parameters()), torch.device("cuda")
    )
    RoutingRecorder.get_or_create(num_layers, num_experts, device)

    logger.info(
        "[RoutingRecord] Init done for layer %d (experts=%d, local=%d) — "
        "using forward-entry hook, gate's original global expert IDs.",
        layer_id, num_experts, num_local_experts,
    )

