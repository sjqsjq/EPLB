#!/bin/bash
# launch_rec_e3.sh - cross_domain_e2e_20261027 prep recording-boot launcher.
# BASE verbatim from launch_arm_rge2e.sh (sealed rev1 BASE + disable-cuda-graph
# + mem 0.8 + disable-radix-cache), PLUS sglang expert-distribution recorder
# flags from the proven recipe baseline_comparison_20260914/launch_rec.sh
# (per_token mode worked under deepep auto). Identity layout, no PB flags,
# no placement: recorded counts = logical expert demand under initial mapping
# (shared calibration input for ES init_by_eplb, DF greedy, MT ILP1).
# The .pt dump lands in cwd; driver sets cwd = ART.
# Usage: ART=<dir> bash launch_rec_e3.sh
set -u
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
BASE="--model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600"
REC="--expert-distribution-recorder-mode per_token --enable-expert-distribution-metrics"
cd "$ART"
exec python3 -m sglang.launch_server $BASE $REC
