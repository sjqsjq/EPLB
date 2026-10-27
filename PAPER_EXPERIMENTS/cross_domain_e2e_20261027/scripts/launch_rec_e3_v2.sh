#!/bin/bash
# launch_rec_e3_v2.sh - prep recording launcher v2 (cross_domain_e2e_20261027).
# v1 (per_token + deepep auto) crashed: _DetailSinglePassGatherer touches
# num_tokens_per_rdma_rank which is None on single-node deepep normal dispatch.
# v2 replicates the PROVEN recipe moetuner_baseline_20260916/launch_profile.sh:
#   stat recorder + deepep normal + cuda-graph-max-bs 128 + mem 0.8
#   + disable-radix-cache + identity layout (no placement, no PB flags).
# stat+normal -> _SelectExpertsSinglePassGatherer: counts logical expert
# selections at topk time (before dispatch) => independent of a2a mode and
# cuda graph; identity layout => logical == physical. Recorder is armed by
# the driver via HTTP endpoints (no metrics flag => no auto start => warmup
# runs without hooks). Dump dir via SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.
# Usage: ART=<dir> bash launch_rec_e3_v2.sh
set -u
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$ART"
mkdir -p "$ART"
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
cd "$ART"
exec python3 -m sglang.launch_server \
  --model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --expert-distribution-recorder-mode stat \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
