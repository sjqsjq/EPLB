"""MoETuner profile hook for SGLang.

Env-var triggered, zero interference with the running server.
Set MOETUNER_PROFILE=1 before launch (also set MOETUNER_PROFILE_DIR).
Dumps per-forward, per-layer expert token histograms as .npz shards.

Post-process the shards to obtain:
  P[L, E]         -- expert token counts (sum over forwards)
  R[L-1, E, E]    -- adjacent-layer transitions (top-1 pairing)

The hook is called from topk.py after topk_ids is available.
It infers layer_id by counting mod num_layers per forward.

Assumes profile is run in identity layout (physical_id == logical_id).
"""
import os, logging, numpy as np, torch
logger = logging.getLogger(__name__)

_ENABLED  = os.environ.get("MOETUNER_PROFILE", "0") == "1"
_OUT_DIR  = os.environ.get("MOETUNER_PROFILE_DIR", "/tmp/moetuner_profile")
_CHUNK_FWD = int(os.environ.get("MOETUNER_PROFILE_CHUNK_FWD", "128"))

_hook = None

class MoETunerProfileHook:
    def __init__(self, num_layers: int, num_experts: int, rank: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.rank = rank
        self._layer_counter = 0
        self._forward_id = 0
        self._chunk_idx = 0
        # Per-forward per-layer topk_ids buffer (kept on GPU-side then downsampled)
        # We store only cumulative P[l,e] and R[l-1,e,e] on CPU to keep memory small.
        self.P = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.R = np.zeros((num_layers - 1, num_experts, num_experts), dtype=np.int64)
        # Per-forward temp: last-layer top-1 ids per token
        self._prev_layer_top1 = None
        self._num_forwards_seen = 0
        os.makedirs(_OUT_DIR, exist_ok=True)
        logger.info(f"[MoETunerProfile] rank={rank} L={num_layers} E={num_experts} out={_OUT_DIR}")

    def record(self, topk_ids: torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            return
        # topk_ids: [tokens, topk]
        layer_id = self._layer_counter % self.num_layers
        self._layer_counter += 1

        flat = topk_ids.reshape(-1).long()
        mask = flat != -1
        flat_valid = flat.masked_fill(~mask, 0)
        hist = torch.bincount(flat_valid, weights=mask.float(), minlength=self.num_experts).cpu().numpy().astype(np.int64)
        self.P[layer_id] += hist

        # For R[l-1, e_prev, e_curr]: pair current-token top-1 with previous-layer top-1
        top1 = topk_ids[:, 0].cpu().numpy().astype(np.int32)   # [tokens]
        # ignore -1 tokens
        top1_mask = top1 != -1
        if layer_id > 0 and self._prev_layer_top1 is not None and self._prev_layer_top1.shape[0] == top1.shape[0]:
            prev = self._prev_layer_top1
            m = top1_mask & (prev != -1)
            if m.any():
                np.add.at(self.R[layer_id - 1], (prev[m], top1[m]), 1)
        self._prev_layer_top1 = top1

    def on_forward_pass_end(self):
        self._forward_id += 1
        self._layer_counter = 0
        self._prev_layer_top1 = None
        self._num_forwards_seen += 1
        if self._num_forwards_seen % _CHUNK_FWD == 0:
            self.flush()

    def flush(self):
        path = os.path.join(_OUT_DIR, f"rank{self.rank}_chunk{self._chunk_idx}.npz")
        np.savez_compressed(path, P=self.P, R=self.R,
                            num_forwards=np.int64(self._num_forwards_seen))
        logger.info(f"[MoETunerProfile] flushed {path} (fwd={self._num_forwards_seen})")
        self._chunk_idx += 1

def get_hook():
    return _hook

def init_hook(num_layers: int, num_experts: int, rank: int):
    global _hook
    if not _ENABLED:
        return None
    if _hook is None:
        _hook = MoETunerProfileHook(num_layers, num_experts, rank)
    return _hook
