#!/bin/bash
# Send concurrent requests to trigger EPLB rebalance
# Each request generates multiple forward passes (prefill + decode tokens)
# With --eplb-rebalance-num-iterations 200, we need ~200 forward passes to trigger rebalance

SERVER="http://11.139.21.79:34567"
MODEL="/cpfs01/user/nebula_model/llm_weight/DeepSeek-R1"
CONCURRENCY=${1:-8}
TOTAL_REQUESTS=${2:-50}
MAX_TOKENS=${3:-256}

echo "Sending $TOTAL_REQUESTS requests with concurrency=$CONCURRENCY, max_tokens=$MAX_TOKENS"
echo "Target: trigger EPLB rebalance (every 200 iterations)"

PROMPTS=(
  "Explain the theory of general relativity in detail."
  "Write a comprehensive guide to machine learning."
  "Describe the history of quantum computing."
  "What are the main challenges in climate science?"
  "Explain how neural networks work step by step."
  "Describe the architecture of modern CPUs."
  "What is the significance of the Higgs boson?"
  "Explain distributed systems consensus algorithms."
)

send_request() {
    local idx=$1
    local prompt="${PROMPTS[$((idx % ${#PROMPTS[@]}))]}"
    curl -s -o /dev/null -w "req=$idx status=%{http_code} time=%{time_total}s\n" \
      "$SERVER/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"$prompt\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0.7}" &
}

SENT=0
while [ $SENT -lt $TOTAL_REQUESTS ]; do
    BATCH_END=$((SENT + CONCURRENCY))
    if [ $BATCH_END -gt $TOTAL_REQUESTS ]; then
        BATCH_END=$TOTAL_REQUESTS
    fi
    
    for ((i=SENT; i<BATCH_END; i++)); do
        send_request $i
    done
    wait
    
    SENT=$BATCH_END
    echo "--- Completed $SENT/$TOTAL_REQUESTS requests ---"
done

echo ""
echo "All requests sent. Checking logs for EPLB rebalance events..."
sleep 2
grep -c "rebalance" /cpfs01/user/nebula_model/sjq-workspace/benchmark/log/worker/0520_*rank0.log 2>/dev/null && echo "(rebalance events found)" || echo "(no rebalance events yet)"
