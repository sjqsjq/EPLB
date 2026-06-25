#!/bin/bash
#
# bench_eplb_inference.sh - Part 2: Measure EPLB rebalance impact during live inference
#
# This script:
# 1. Starts SGLang server with EPLB enabled (--enable-eplb + redundant experts)
# 2. Sends sustained inference traffic to trigger EPLB rebalancing
# 3. Collects and parses rebalance timing from server logs
#
# Usage: ./bench_eplb_inference.sh [--nodes N] [--master-ip IP] [--model MODEL]
#
# Prerequisites: run_container.sh must have been run first
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
NODES=2
MASTER_IP="11.139.21.90"
CONFIG_FILE="config.yaml"
MODEL_NAME=""  # auto-detect from config
REDUNDANT_EXPERTS=32
REBALANCE_ITERS=200   # trigger rebalance every N iterations (lower = more frequent for testing)
REBALANCE_LAYERS_PER_CHUNK=""  # empty = all layers at once
BENCHMARK_DURATION=120  # seconds to run inference
BATCH_SIZE=8
INPUT_LEN=1024
OUTPUT_LEN=128
PORT=30000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes) NODES="$2"; shift 2 ;;
        --master-ip) MASTER_IP="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --redundant-experts) REDUNDANT_EXPERTS="$2"; shift 2 ;;
        --rebalance-iters) REBALANCE_ITERS="$2"; shift 2 ;;
        --rebalance-layers-per-chunk) REBALANCE_LAYERS_PER_CHUNK="$2"; shift 2 ;;
        --duration) BENCHMARK_DURATION="$2"; shift 2 ;;
        --bs) BATCH_SIZE="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-complete IP
IP_PREFIX="11.139.21"
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    MASTER_IP="${IP_PREFIX}.${MASTER_IP}"
fi

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Part 2: EPLB Rebalance Inference Benchmark${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Master IP:          $MASTER_IP"
echo "  Nodes:              $NODES"
echo "  Redundant experts:  $REDUNDANT_EXPERTS"
echo "  Rebalance interval: every $REBALANCE_ITERS iterations"
echo "  Duration:           ${BENCHMARK_DURATION}s"
echo "  Batch size:         $BATCH_SIZE"
echo "  Input/Output len:   $INPUT_LEN / $OUTPUT_LEN"
echo ""

# ============================================================
# Step 1: Build server launch command
# ============================================================
echo -e "${YELLOW}[1/5] Building server command...${NC}"

GPUS_PER_NODE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['gpus_per_node'])")
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# EP size = total GPUs (all GPUs form one EP group)
EP_SIZE=$TOTAL_GPUS
TP_SIZE=$TOTAL_GPUS
DP_SIZE=$TOTAL_GPUS

EPLB_ARGS="--enable-eplb --ep-num-redundant-experts $REDUNDANT_EXPERTS --eplb-rebalance-num-iterations $REBALANCE_ITERS"
if [ -n "$REBALANCE_LAYERS_PER_CHUNK" ]; then
    EPLB_ARGS="$EPLB_ARGS --eplb-rebalance-layers-per-chunk $REBALANCE_LAYERS_PER_CHUNK"
fi

SERVER_CMD="python -m sglang.launch_server \
    --port $PORT \
    --host 0.0.0.0 \
    --tp-size $TP_SIZE \
    --dp-size $DP_SIZE \
    --ep-size $EP_SIZE \
    --nnodes $NODES \
    --dist-init-addr ${MASTER_IP}:31000 \
    --attention-backend trtllm_mha \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --stream-out \
    --moe-a2a-backend deepep \
    --disable-cuda-graph \
    --decode-log-interval 1 \
    $EPLB_ARGS"

echo "  Server command:"
echo "    $SERVER_CMD"
echo ""

# ============================================================
# Step 2: Start server
# ============================================================
echo -e "${YELLOW}[2/5] Starting SGLang server with EPLB...${NC}"

./run_server.sh --master-ip "$MASTER_IP" --command "$SERVER_CMD" --config "$CONFIG_FILE"
SERVER_STATUS=$?

if [ $SERVER_STATUS -ne 0 ]; then
    echo -e "${RED}Server failed to start!${NC}"
    exit 1
fi

echo -e "${GREEN}  Server is ready.${NC}"
echo ""

# ============================================================
# Step 3: Run inference traffic
# ============================================================
echo -e "${YELLOW}[3/5] Sending inference traffic for ${BENCHMARK_DURATION}s...${NC}"

# Get model path for tokenizer
MODEL_PATH=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
print(config['model_path'] + '/' + config['model_name'])
")

echo "  Model path: $MODEL_PATH"
echo "  Target: http://${MASTER_IP}:${PORT}"
echo ""

# Use timeout to limit duration, run bench_one_batch in loop
BENCH_LOG="result/bench_eplb_inference_traffic.log"
mkdir -p result

# Run multiple rounds to ensure we trigger rebalancing events
NUM_ROUNDS=$((BENCHMARK_DURATION / 10))  # Each round ~10s

for round in $(seq 1 $NUM_ROUNDS); do
    echo "  Round $round/$NUM_ROUNDS..."
    ./run_client.sh --base-url "http://${MASTER_IP}:${PORT}" \
        --bs $BATCH_SIZE --input-len $INPUT_LEN --output-len $OUTPUT_LEN \
        2>&1 | tee -a "$BENCH_LOG" | grep -E "throughput|latency|Throughput" || true
    sleep 2
done

echo ""
echo -e "${GREEN}  Traffic generation complete.${NC}"
echo ""

# ============================================================
# Step 4: Collect and parse EPLB rebalance logs
# ============================================================
echo -e "${YELLOW}[4/5] Parsing EPLB rebalance timing from logs...${NC}"

# Find the latest worker log for rank 0
LATEST_LOG=$(ls -t log/worker/rank0_* 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    # Try alternative log naming
    LATEST_LOG=$(ls -t log/worker/*rank0* 2>/dev/null | head -1)
fi

if [ -z "$LATEST_LOG" ]; then
    echo -e "${RED}  No rank0 log found!${NC}"
    echo "  Trying to fetch log from master node..."
    # Fetch log directly from container
    NODE_LIST=($(cat "tmp/nodelist_${MASTER_IP}"))
    MASTER_NODE="${NODE_LIST[0]}"
    ssh "$MASTER_NODE" "echo '617178Sjq' | sudo -S pouch exec sjq_sglang_benchmark_rank0 cat /tmp/sglang_server.log 2>/dev/null" > "result/server_rank0.log" 2>/dev/null || true
    LATEST_LOG="result/server_rank0.log"
fi

echo "  Log file: $LATEST_LOG"
echo ""

# Parse rebalance events
REBALANCE_LOG="result/bench_eplb_rebalance_timing.txt"
grep -a "EPLBManager.*rebalance" "$LATEST_LOG" > "$REBALANCE_LOG" 2>/dev/null || true

REBALANCE_COUNT=$(grep -c "rebalance end" "$REBALANCE_LOG" 2>/dev/null || echo "0")
echo "  Total rebalance events: $REBALANCE_COUNT"

if [ "$REBALANCE_COUNT" -gt 0 ]; then
    echo ""
    echo "  Rebalance timing breakdown:"
    echo "  ─────────────────────────────────────────────────────────────"
    grep "rebalance end" "$REBALANCE_LOG" | while read -r line; do
        echo "    $line"
    done
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""

    # Extract timing stats
    python3 << 'PYEOF'
import re
import sys
import json
import numpy as np

log_file = sys.argv[1] if len(sys.argv) > 1 else "result/bench_eplb_rebalance_timing.txt"

totals = []
stat_collects = []
algorithms = []
p2p_transfers = []

with open(log_file, 'r') as f:
    for line in f:
        if "rebalance end" not in line:
            continue
        m = re.search(r'total=([\d.]+)s', line)
        if m:
            totals.append(float(m.group(1)))
        m = re.search(r'stat_collect=([\d.]+)s', line)
        if m:
            stat_collects.append(float(m.group(1)))
        m = re.search(r'algorithm=([\d.]+)s', line)
        if m:
            algorithms.append(float(m.group(1)))
        m = re.search(r'p2p_transfer=([\d.]+)s', line)
        if m:
            p2p_transfers.append(float(m.group(1)))

if not totals:
    print("  No timing data found in logs.")
    sys.exit(0)

def stats(arr, name):
    a = np.array(arr)
    return f"  {name:<20}: avg={np.mean(a)*1000:.1f}ms  min={np.min(a)*1000:.1f}ms  max={np.max(a)*1000:.1f}ms  std={np.std(a)*1000:.1f}ms  (n={len(a)})"

print("\n  ═══════════════════════════════════════════════════════")
print("  EPLB Rebalance Timing Statistics")
print("  ═══════════════════════════════════════════════════════")
print(stats(totals, "Total"))
if stat_collects:
    print(stats(stat_collects, "Stat collection"))
if algorithms:
    print(stats(algorithms, "Algorithm"))
if p2p_transfers:
    print(stats(p2p_transfers, "P2P transfer"))
print("  ═══════════════════════════════════════════════════════")

if totals and p2p_transfers:
    avg_total = np.mean(totals) * 1000
    avg_p2p = np.mean(p2p_transfers) * 1000
    print(f"\n  P2P transfer占比: {avg_p2p/avg_total*100:.1f}% of total rebalance time")
    print(f"  Overhead (stat+algo): {avg_total - avg_p2p:.1f}ms")

# Save structured results
results = {
    "num_rebalance_events": len(totals),
    "total_ms": {"avg": np.mean(totals)*1000, "min": np.min(totals)*1000, "max": np.max(totals)*1000},
    "p2p_transfer_ms": {"avg": np.mean(p2p_transfers)*1000, "min": np.min(p2p_transfers)*1000, "max": np.max(p2p_transfers)*1000} if p2p_transfers else None,
    "stat_collect_ms": {"avg": np.mean(stat_collects)*1000, "min": np.min(stat_collects)*1000, "max": np.max(stat_collects)*1000} if stat_collects else None,
    "algorithm_ms": {"avg": np.mean(algorithms)*1000, "min": np.min(algorithms)*1000, "max": np.max(algorithms)*1000} if algorithms else None,
}
with open("result/bench_eplb_rebalance_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to result/bench_eplb_rebalance_stats.json")
PYEOF
    "$REBALANCE_LOG"
else
    echo -e "${YELLOW}  No rebalance events detected. Possible reasons:${NC}"
    echo "    - Server utilization too high (above threshold), rebalance skipped"
    echo "    - Not enough iterations completed to trigger rebalance"
    echo "    - Check log for 'Skipped ep rebalancing' messages"
    echo ""
    SKIP_COUNT=$(grep -c "Skipped ep rebalancing" "$LATEST_LOG" 2>/dev/null || echo "0")
    echo "  'Skipped rebalancing' events: $SKIP_COUNT"
fi

# ============================================================
# Step 5: Summary
# ============================================================
echo ""
echo -e "${YELLOW}[5/5] Summary${NC}"
echo "  ─────────────────────────────────────────"
echo "  Server log:        $LATEST_LOG"
echo "  Rebalance log:     $REBALANCE_LOG"
echo "  Traffic log:       $BENCH_LOG"
echo "  Results:           result/bench_eplb_rebalance_stats.json"
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Part 2 benchmark complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "To stop the server:"
echo "  ./cleanup_containers.sh --master-ip $MASTER_IP"
