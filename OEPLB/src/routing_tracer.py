"""Routing tracer for OEPLB analysis."""
import os, logging, numpy as np, torch
logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SGLANG_OEPLB_ROUTING_TRACE", "0") == "1"
_OUT_DIR = os.environ.get("SGLANG_OEPLB_ROUTING_TRACE_DIR", "/tmp/routing_trace")

_tracer = None
class RoutingTracer:
    def __init__(self, num_layers, num_physical_experts, rank):
        self.num_layers = num_layers
        self.num_physical_experts = num_physical_experts
        self.rank = rank
        self.forward_ids, self.layer_ids, self.physical_hists, self.logical_hists, self.forward_mode_is_prefill = [],[],[],[],[]
        self._layer_counter = 0
        self._forward_id = 0
        self._chunk_idx = 0
        self.current_is_prefill = False
        os.makedirs(_OUT_DIR, exist_ok=True)
    def on_forward_pass_end(self):
        self._forward_id += 1; self._layer_counter = 0
    def record(self, topk_ids, p2l_layer, is_prefill):
        if torch.cuda.is_current_stream_capturing(): return
        layer_id = self._layer_counter % self.num_layers; self._layer_counter += 1
        flat = topk_ids.reshape(-1); mask = flat != -1
        physical_hist = torch.bincount(flat.masked_fill(~mask, 0).long(), weights=mask.float(), minlength=self.num_physical_experts)
        logical_ids_hit = p2l_layer[flat.masked_fill(~mask, 0).long()]
        logical_hist = torch.bincount(logical_ids_hit.masked_fill(~mask, 0).long(), weights=mask.float(), minlength=self.num_physical_experts)
        self.forward_ids.append(self._forward_id); self.layer_ids.append(layer_id)
        self.physical_hists.append(physical_hist.cpu().numpy().astype(np.int32))
        self.logical_hists.append(logical_hist.cpu().numpy().astype(np.int32))
        self.forward_mode_is_prefill.append(is_prefill)
        if len(self.forward_ids) >= 4000: self.flush()
    def flush(self):
        if not self.forward_ids: return
        path = os.path.join(_OUT_DIR, f"rank{self.rank}_chunk{self._chunk_idx}.npz")
        np.savez_compressed(path, forward_ids=np.array(self.forward_ids,dtype=np.int64), layer_ids=np.array(self.layer_ids,dtype=np.int32), physical_hists=np.stack(self.physical_hists), logical_hists=np.stack(self.logical_hists), is_prefill=np.array(self.forward_mode_is_prefill,dtype=bool))
        self._chunk_idx += 1; self.forward_ids,self.layer_ids,self.physical_hists,self.logical_hists,self.forward_mode_is_prefill=[],[],[],[],[]

def get_routing_tracer(): return _tracer
def init_routing_tracer(num_layers, num_physical_experts, rank):
    global _tracer
    if not _ENABLED: return None
    _tracer = RoutingTracer(num_layers, num_physical_experts, rank); return _tracer
def is_enabled(): return _ENABLED

_simple_recorder = None
class SimpleRoutingRecorder:
    def __init__(self, num_layers, num_physical_experts, rank):
        self.num_layers = num_layers; self.num_physical_experts = num_physical_experts; self.rank = rank
        self._layer_counter = 0; self._forward_id = 0; self._is_prefill = False; self._chunk_idx = 0
        self._current_hists = np.zeros((num_layers, num_physical_experts), dtype=np.int32)
        self._records = []
        os.makedirs(_OUT_DIR, exist_ok=True)
        logger.info(f"[RECORDER] rank={rank} L={num_layers} E={num_physical_experts}")
    def set_prefill(self, is_prefill): self._is_prefill = is_prefill
    def record_layer(self, topk_ids):
        if torch.cuda.is_current_stream_capturing(): return
        layer_id = self._layer_counter % self.num_layers; self._layer_counter += 1
        flat = topk_ids.reshape(-1); mask = flat != -1
        hist = torch.bincount(flat.masked_fill(~mask, 0).long(), weights=mask.float(), minlength=self.num_physical_experts).cpu().numpy().astype(np.int32)
        self._current_hists[layer_id] = hist
    def on_forward_pass_end(self):
        self._records.append((self._forward_id, self._is_prefill, self._current_hists.copy()))
        self._current_hists[:] = 0; self._forward_id += 1; self._layer_counter = 0
        if len(self._records) >= 200: self._flush()
    def _flush(self):
        if not self._records: return
        path = os.path.join(_OUT_DIR, f"rank{self.rank}_fwd_chunk{self._chunk_idx}.npz")
        np.savez_compressed(path, forward_ids=np.array([r[0] for r in self._records],dtype=np.int64), is_prefill=np.array([r[1] for r in self._records],dtype=bool), layer_hists=np.stack([r[2] for r in self._records]))
        logger.info(f"[RECORDER] rank={self.rank} flushed {len(self._records)} to {path}")
        self._chunk_idx += 1; self._records = []
    def finalize(self): self._flush()
    def save(self): self._flush()

def get_simple_recorder(): return _simple_recorder
def init_simple_recorder(num_layers, num_physical_experts, rank):
    global _simple_recorder
    if not _ENABLED: return None
    _simple_recorder = SimpleRoutingRecorder(num_layers, num_physical_experts, rank); return _simple_recorder
