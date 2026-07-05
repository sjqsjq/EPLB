"""Launch SGLang with the cross-layer prediction pipeline.

Enables the pipeline only AFTER ModelRunner.initialize() (which covers all
CUDA graph capture / dry-run warmup) has fully completed, since running our
cross-rank collectives or P2P weight swaps during warmup — on synthetic data,
with no real rank-to-rank synchronization guarantees — deadlocks against
DeepEP's own communicator.
"""
import os
import sys
import json
import logging

sys.path.insert(0, "/workspace/EPLB")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect model config
model_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--model-path" and i + 1 < len(sys.argv):
        model_path = sys.argv[i + 1]

if model_path and os.path.exists(os.path.join(model_path, "config.json")):
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    num_layers = cfg.get("num_hidden_layers", 48)
    num_experts = cfg.get("num_experts", 128)
    top_k = cfg.get("num_experts_per_tok", 8)
else:
    num_layers, num_experts, top_k = 94, 128, 8

from pipeline import PipelineManager
pm = PipelineManager.initialize(num_layers=num_layers, num_experts=num_experts, top_k=top_k, num_gpus=4)
logger.info(f"Pipeline initialized (disabled until warmup completes): {pm.get_stats()}")

# Monkeypatch: flip pm.enabled = True only after ModelRunner.initialize()
# (which runs all CUDA graph capture/warmup) returns for THIS process.
from sglang.srt.model_executor.model_runner import ModelRunner

_orig_initialize = ModelRunner.initialize

def _patched_initialize(self, *args, **kwargs):
    result = _orig_initialize(self, *args, **kwargs)
    pm2 = PipelineManager.get_instance()
    if pm2 is not None:
        pm2.enabled = True
        logger.info("[Pipeline] enabled — warmup/CUDA graph capture complete")
    return result

ModelRunner.initialize = _patched_initialize

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.entrypoints.http_server import launch_server

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
