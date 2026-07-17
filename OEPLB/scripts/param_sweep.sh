#!/bin/bash
set -e
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:${LD_LIBRARY_PATH:-}
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

BASE="--model-path /workspace/Qwen3-30B-A3B-FP8 --tp 4 --dp 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code"

# Configurations to sweep: threshold,sync_window,max_swaps,max_layers
CONFIGS=(
  "1.05,32,5,48"
  "1.10,32,5,48"
  "1.05,64,3,48"
  "1.10,64,5,48"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r thresh sw ms ml <<< "$cfg"
  label="sweep_t${thresh}_sw${sw}_ms${ms}_ml${ml}"
  echo "=== Starting $label ==="
  
  python3 -m sglang.launch_server $BASE \
    --enable-pb-oeplb \
    --pb-oeplb-threshold-ratio $thresh \
    --pb-oeplb-min-prefill-tokens 1000 \
    --pb-oeplb-sync-window $sw \
    --pb-oeplb-cooldown-steps 5 \
    --pb-oeplb-max-total-swap-layers $ml \
    --pb-oeplb-max-swaps-per-layer $ms \
    > /tmp/sweep_${label}.log 2>&1 &
  PID=$!
  
  # Wait for server ready (max 8 min)
  for i in $(seq 1 48); do
    if curl -s --max-time 5 http://localhost:30000/health 2>/dev/null | grep -q "ok"; then
      echo "$label server ready after $((i*10))s"
      break
    fi
    sleep 10
    if ! kill -0 $PID 2>/dev/null; then
      echo "$label server died!"
      break 2
    fi
  done
  
  # Run benchmark
  cd /workspace/EPLB/OEPLB
  python3 scripts/run_bench.py $label 2>&1 | tail -5
  
  # Cleanup
  kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; sleep 5
  echo "=== $label done ==="
  echo ""
done

echo "=== All sweeps done ==="
