#!/bin/bash
# Usage: sweep_one.sh <label> <sync_window> <threshold> <max_swaps>
LABEL=$1; SW=$2; TH=$3; MS=$4
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:${LD_LIBRARY_PATH:-}
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

python3 -m sglang.launch_server --model-path /workspace/Qwen3-30B-A3B-FP8 --tp 4 --dp 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code --enable-pb-oeplb --pb-oeplb-threshold-ratio $TH --pb-oeplb-min-prefill-tokens 1000 --pb-oeplb-sync-window $SW --pb-oeplb-cooldown-steps 5 --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-max-swaps-per-layer $MS > /tmp/sweep_${LABEL}.log 2>&1 &
PID=$!
echo "Server PID=$PID"

# Wait ready
for i in $(seq 1 60); do
  curl -s --max-time 5 http://localhost:30000/health 2>/dev/null | grep -q ok && break
  sleep 10
done

# Warmup + profile + benchmark
python3 /workspace/EPLB/OEPLB/scripts/sweep_run.py $LABEL

# Kill
kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; sleep 5
