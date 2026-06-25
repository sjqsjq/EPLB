#!/bin/bash
#
# run_profiler.sh - Run profiler with HTTP API (Simplified)
#
# Core functionality:
# - Runs 2 fixed test configurations:
#   1. bs=32, input_len=2048, output_len=512
#   2. bs=4, input_len=65536 (64k), output_len=512
# - Test flow: start client → sleep 10s → start profiler → wait for completion → sleep 30s
# - Completion check: poll /get_model_info for running_req and waiting_req
# - Profiler auto-stops (no explicit stop_profile call)
# - Failsafe: If profiler hasn't stopped after 5 minutes, call stop_profile
#
# PD Separation Mode:
# - Use --prefill and --decode to specify separate servers
# - Profiler will be started on both prefill and decode servers
# - Output paths will have prefill/decode suffix
#
# Usage:
#   Normal mode: ./run_profiler.sh --master-ip IP
#   PD mode:     ./run_profiler.sh --prefill IP --decode IP
#

set -e

# Default values
MASTER_IP=""
PREFILL_IP=""
DECODE_IP=""
MAX_POLL_ATTEMPTS=300  # 300 * 2s = 10 minutes max wait for profiling completion
USE_IMAGE=false
IMG_RATIO=""
IMAGE_RESOLUTION="1080p"
IMAGE_COUNT=1
DEBUG=false
ROUTER_PORT="34001"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --prefill)
            PREFILL_IP="$2"
            shift 2
            ;;
        --decode)
            DECODE_IP="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --image)
            USE_IMAGE=true
            shift
            ;;
        --img-ratio)
            IMG_RATIO="$2"
            shift 2
            ;;
        --image-resolution)
            IMAGE_RESOLUTION="$2"
            shift 2
            ;;
        --image-count)
            IMAGE_COUNT="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --master-ip IP [OPTIONS]"
            echo "   OR: $0 --prefill IP --decode IP [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --master-ip IP          Master server IP (normal mode)"
            echo "  --prefill IP            Prefill server IP (PD mode)"
            echo "  --decode IP             Decode server IP (PD mode)"
            echo "  --router-port PORT      Router port for PD mode (default: 34001)"
            echo "  --image                 Use image dataset"
            echo "  --img-ratio RATIO       Vision token ratio (e.g., 0.3)"
            echo "  --image-resolution RES  Image resolution (default: 1080p)"
            echo "  --image-count N         Number of images (default: 1)"
            echo "  --debug                 Enable debug mode"
            exit 1
            ;;
    esac
done

# Auto-complete IP prefix
IP_PREFIX="11.139.21"
if [ -n "$MASTER_IP" ] && [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    MASTER_IP="${IP_PREFIX}.${MASTER_IP}"
    echo "Auto-completed master IP: $MASTER_IP"
fi
if [ -n "$PREFILL_IP" ] && [[ "$PREFILL_IP" =~ ^[0-9]+$ ]]; then
    PREFILL_IP="${IP_PREFIX}.${PREFILL_IP}"
    echo "Auto-completed prefill IP: $PREFILL_IP"
fi
if [ -n "$DECODE_IP" ] && [[ "$DECODE_IP" =~ ^[0-9]+$ ]]; then
    DECODE_IP="${IP_PREFIX}.${DECODE_IP}"
    echo "Auto-completed decode IP: $DECODE_IP"
fi

# Determine mode and validate
if [ -n "$PREFILL_IP" ] && [ -n "$DECODE_IP" ]; then
    PD_MODE=true
    ROUTER_URL="http://${PREFILL_IP}:${ROUTER_PORT}"
    PREFILL_URL="http://${PREFILL_IP}:34567"
    DECODE_URL="http://${DECODE_IP}:34567"
    BASE_URL="$ROUTER_URL"  # For client requests
elif [ -n "$MASTER_IP" ]; then
    PD_MODE=false
    BASE_URL="http://${MASTER_IP}:34567"
else
    echo "Error: Must specify either --master-ip OR (--prefill AND --decode)"
    echo "Usage: $0 --master-ip IP [OPTIONS]"
    echo "   OR: $0 --prefill IP --decode IP [OPTIONS]"
    exit 1
fi

# Validate image mode parameters
if [ "$USE_IMAGE" = true ] && [ -z "$IMG_RATIO" ]; then
    echo "Error: --img-ratio is required when using --image"
    echo "Example: $0 --master-ip 81 --image --img-ratio 0.3"
    exit 1
fi

echo "======================================"
echo "SGLang Profiler Runner"
echo "======================================"
if [ "$PD_MODE" = true ]; then
    echo "Mode: PD Separation"
    echo "Prefill server: $PREFILL_URL"
    echo "Decode server: $DECODE_URL"
    echo "Router: $ROUTER_URL"
else
    echo "Mode: Normal"
    echo "Server: $BASE_URL"
fi
echo "Dataset: $([ "$USE_IMAGE" = true ] && echo "image" || echo "text")"
if [ "$USE_IMAGE" = true ]; then
    echo "Vision Ratio: $IMG_RATIO"
    echo "Image Resolution: $IMAGE_RESOLUTION"
    echo "Image Count: $IMAGE_COUNT"
fi

# Read configuration from config.yaml
CONFIG_FILE="config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

WORKSPACE_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")
MASTER_CONTAINER="sjq_sglang_benchmark_rank0"

echo "Workspace: $WORKSPACE_PATH"

# Get server configuration
echo ""
echo "Fetching server configuration..."

if [ "$PD_MODE" = true ]; then
    # Fetch from both prefill and decode servers
    PREFILL_INFO=$(curl -s ${PREFILL_URL}/server_info)
    DECODE_INFO=$(curl -s ${DECODE_URL}/server_info)

    if [ $? -ne 0 ] || [ -z "$PREFILL_INFO" ] || [ -z "$DECODE_INFO" ]; then
        echo "Error: Failed to fetch server info from prefill or decode server"
        exit 1
    fi

    # Parse prefill configuration
    PREFILL_TP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tp_size', 1))")
    PREFILL_EP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('ep_size', 1))")
    PREFILL_DP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dp_size', 1))")

    # Parse decode configuration
    DECODE_TP=$(echo "$DECODE_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tp_size', 1))")
    DECODE_EP=$(echo "$DECODE_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('ep_size', 1))")
    DECODE_DP=$(echo "$DECODE_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dp_size', 1))")

    MODEL_PATH=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model_path', ''))")

    echo "  Prefill: TP${PREFILL_TP}-EP${PREFILL_EP}-DP${PREFILL_DP}"
    echo "  Decode:  TP${DECODE_TP}-EP${DECODE_EP}-DP${DECODE_DP}"
else
    SERVER_INFO=$(curl -s ${BASE_URL}/server_info)

    if [ $? -ne 0 ] || [ -z "$SERVER_INFO" ]; then
        echo "Error: Failed to fetch server info from $BASE_URL/server_info"
        exit 1
    fi

    TP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tp_size', 1))")
    EP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('ep_size', 1))")
    DP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dp_size', 1))")
    MODEL_PATH=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model_path', ''))")

    echo "  TP size: $TP_SIZE"
    echo "  EP size: $EP_SIZE"
    echo "  DP size: $DP_SIZE"
fi

# Extract model name from model_path (last path segment)
MODEL_NAME=$(basename "$MODEL_PATH")
echo "  Model: $MODEL_NAME"

# Base configuration
TRACE_DIR=/cpfs01/user/nebula_model/sjq-workspace/benchmark/result/traces

if [ "$PD_MODE" = true ]; then
    CONFIG_DIR="${MODEL_NAME}-Prefill-TP${PREFILL_TP}-EP${PREFILL_EP}-DP${PREFILL_DP}-Decode-TP${DECODE_TP}-EP${DECODE_EP}-DP${DECODE_DP}"
else
    CONFIG_DIR="${MODEL_NAME}-TP${TP_SIZE}-EP${EP_SIZE}-DP${DP_SIZE}"
fi

# Fixed test configurations
declare -a TEST_CONFIGS=(
    "64:2048:512"     # bs64, in2048, out512
    "64:65536:512"     # bs64, in65536, out512
)

echo ""
echo "Will run ${#TEST_CONFIGS[@]} profiling tests:"
for i in "${!TEST_CONFIGS[@]}"; do
    IFS=':' read -r BS IN OUT <<< "${TEST_CONFIGS[$i]}"
    echo "  Test $((i+1)): bs${BS} in${IN} out${OUT}"
done
echo ""

# Function: Wait for profiling to complete by checking server queue
wait_for_profiling_done() {
    local server_url=$1
    local server_label=$2

    echo "  Waiting for $server_label profiling to complete..."
    local attempts=0

    while [ $attempts -lt $MAX_POLL_ATTEMPTS ]; do
        # Get queue info from server
        local queue_info=$(curl -s ${server_url}/get_model_info 2>/dev/null || echo "{}")
        local running_reqs=$(echo "$queue_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(sum(w.get('running_req', 0) for w in data.get('worker_info', [])))" 2>/dev/null || echo "999")
        local waiting_reqs=$(echo "$queue_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(sum(w.get('waiting_req', 0) for w in data.get('worker_info', [])))" 2>/dev/null || echo "999")

        # Check if both running and waiting requests are 0
        if [ "$running_reqs" -eq 0 ] && [ "$waiting_reqs" -eq 0 ]; then
            echo "  $server_label profiling completed (running_req=0, waiting_req=0)"
            return 0
        fi

        echo "  $server_label - Running: $running_reqs, Waiting: $waiting_reqs (attempt $((attempts + 1))/$MAX_POLL_ATTEMPTS)"
        sleep 2
        attempts=$((attempts + 1))
    done

    echo "  Warning: Timeout waiting for $server_label profiling completion"
    return 1
}

# Function: Run a single profiling test
run_profiling_test() {
    local BS=$1
    local IN=$2
    local OUT=$3
    local TEST_NUM=$4

    # Convert input length to readable format (2k, 64k, etc)
    local IN_READABLE
    if [ $IN -ge 1024 ]; then
        IN_READABLE="$((IN / 1024))k"
    else
        IN_READABLE="${IN}"
    fi

    # Build subdirectory path with optional image suffix
    local SUB_DIR="${CONFIG_DIR}/bs${BS}_in${IN_READABLE}_out${OUT}"
    if [ "$USE_IMAGE" = true ]; then
        SUB_DIR="${SUB_DIR}_img${IMG_RATIO}"
    fi

    echo ""
    echo "======================================"
    if [ "$USE_IMAGE" = true ]; then
        echo "[Test ${TEST_NUM}/${#TEST_CONFIGS[@]}] bs${BS} in${IN_READABLE} out${OUT} img${IMG_RATIO}"
    else
        echo "[Test ${TEST_NUM}/${#TEST_CONFIGS[@]}] bs${BS} in${IN_READABLE} out${OUT}"
    fi
    echo "======================================"

    # [1/6] Start client in background
    echo "  [1/6] Starting benchmark client..."

    # Build client command based on mode and dataset type
    if [ "$PD_MODE" = true ]; then
        # PD mode: pass prefill and decode URLs
        if [ "$USE_IMAGE" = true ]; then
            CLIENT_CMD="bash run_client.sh --bs $BS --input-len $IN --output-len $OUT --prefill $PREFILL_URL --decode $DECODE_URL --router-port $ROUTER_PORT --image --image-resolution $IMAGE_RESOLUTION --image-count $IMAGE_COUNT"
        else
            CLIENT_CMD="bash run_client.sh --bs $BS --input-len $IN --output-len $OUT --prefill $PREFILL_URL --decode $DECODE_URL --router-port $ROUTER_PORT"
        fi
    else
        # Normal mode: pass base URL
        if [ "$USE_IMAGE" = true ]; then
            CLIENT_CMD="bash run_client.sh --bs $BS --input-len $IN --output-len $OUT --base-url $BASE_URL --image --image-resolution $IMAGE_RESOLUTION --image-count $IMAGE_COUNT"
        else
            CLIENT_CMD="bash run_client.sh --bs $BS --input-len $IN --output-len $OUT --base-url $BASE_URL"
        fi
    fi

    echo "  Command: $CLIENT_CMD"
    $CLIENT_CMD > /tmp/profiler_client_${TEST_NUM}.log 2>&1 &
    CLIENT_PID=$!
    echo "  Client started (PID: $CLIENT_PID)"

    # [2/6] Wait 10 seconds before starting profiler
    echo "  [2/6] Waiting 10 seconds before starting profiler..."
    sleep 10

    # [3/6] Start profiler via HTTP API
    echo "  [3/6] Starting profiler via HTTP API..."
    PROFILER_START_TIME=$(date +%s)  # Record profiler start time

    if [ "$PD_MODE" = true ]; then
        # PD mode: Start profiler on both prefill and decode servers
        local PREFILL_OUT_DIR="${TRACE_DIR}/${SUB_DIR}_prefill"
        local DECODE_OUT_DIR="${TRACE_DIR}/${SUB_DIR}_decode"

        mkdir -p "$PREFILL_OUT_DIR"
        mkdir -p "$DECODE_OUT_DIR"

        echo "  Starting profiler on prefill server..."
        echo "    Output: $PREFILL_OUT_DIR"
        PREFILL_RESPONSE=$(curl -s -X POST ${PREFILL_URL}/start_profile \
            -H "Content-Type: application/json" \
            -d @- <<EOF
{
  "activities": ["CPU", "GPU"],
  "num_steps": 150,
  "output_dir": "${PREFILL_OUT_DIR}"
}
EOF
)

        echo "  Starting profiler on decode server..."
        echo "    Output: $DECODE_OUT_DIR"
        DECODE_RESPONSE=$(curl -s -X POST ${DECODE_URL}/start_profile \
            -H "Content-Type: application/json" \
            -d @- <<EOF
{
  "activities": ["CPU", "GPU"],
  "num_steps": 150,
  "output_dir": "${DECODE_OUT_DIR}"
}
EOF
)

        if [ $? -eq 0 ]; then
            echo "  Profilers started successfully on both servers"
        else
            echo "  Failed to start profilers"
            kill $CLIENT_PID 2>/dev/null || true
            return 1
        fi
    else
        # Normal mode: Start profiler on single server
        local OUT_DIR="${TRACE_DIR}/${SUB_DIR}"
        mkdir -p "$OUT_DIR"
        echo "  Output directory: $OUT_DIR"

        PROFILE_RESPONSE=$(curl -s -X POST ${BASE_URL}/start_profile \
            -H "Content-Type: application/json" \
            -d @- <<EOF
{
  "activities": ["CPU", "GPU"],
  "num_steps": 150,
  "output_dir": "${OUT_DIR}"
}
EOF
)

        if [ $? -eq 0 ]; then
            PROFILE_ID=$(echo "$PROFILE_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('profile_id', ''))" 2>/dev/null || echo "")
            if [ -n "$PROFILE_ID" ]; then
                echo "  Profiler started successfully (profile_id: $PROFILE_ID)"
            else
                echo "  Profiler started successfully"
            fi
        else
            echo "  Failed to start profiler"
            kill $CLIENT_PID 2>/dev/null || true
            return 1
        fi
    fi

    # [4/6] Wait for client to complete
    echo "  [4/6] Waiting for client to complete..."
    wait $CLIENT_PID
    CLIENT_EXIT_CODE=$?

    if [ $CLIENT_EXIT_CODE -eq 0 ]; then
        echo "  Client completed successfully"
    else
        echo "  Warning: Client exited with code $CLIENT_EXIT_CODE"
    fi

    # [5/6] Wait for profiling to complete (check queue)
    echo "  [5/6] Waiting for profiling to complete..."

    if [ "$PD_MODE" = true ]; then
        # Wait for both prefill and decode servers
        wait_for_profiling_done "$PREFILL_URL" "Prefill" &
        PREFILL_WAIT_PID=$!
        wait_for_profiling_done "$DECODE_URL" "Decode" &
        DECODE_WAIT_PID=$!

        wait $PREFILL_WAIT_PID
        wait $DECODE_WAIT_PID
    else
        wait_for_profiling_done "$BASE_URL" "Server"
    fi

    # Failsafe stop: If 5 minutes have passed since profiler start, call stop_profile
    local CURRENT_TIME=$(date +%s)
    local ELAPSED_TIME=$((CURRENT_TIME - PROFILER_START_TIME))
    local FAILSAFE_TIMEOUT=300  # 5 minutes

    if [ $ELAPSED_TIME -ge $FAILSAFE_TIMEOUT ]; then
        echo "  Failsafe: ${ELAPSED_TIME}s elapsed since profiler start, calling stop_profile..."
        if [ "$PD_MODE" = true ]; then
            curl -s -X POST ${PREFILL_URL}/stop_profile > /dev/null 2>&1 || true
            curl -s -X POST ${DECODE_URL}/stop_profile > /dev/null 2>&1 || true
        else
            curl -s -X POST ${BASE_URL}/stop_profile > /dev/null 2>&1 || true
        fi
        echo "  Failsafe stop signal sent"
    else
        echo "  Profiler auto-stopped (elapsed time: ${ELAPSED_TIME}s < ${FAILSAFE_TIMEOUT}s)"
    fi

    # [6/6] Wait 30 seconds for trace files to be saved
    echo "  [6/6] Waiting 30 seconds for trace files to be saved..."
    sleep 30

    echo "  Test ${TEST_NUM} completed successfully"

    # Clean up client log
    rm -f /tmp/profiler_client_${TEST_NUM}.log
}

# Main execution loop
echo "Starting profiling tests..."

for i in "${!TEST_CONFIGS[@]}"; do
    IFS=':' read -r BS IN OUT <<< "${TEST_CONFIGS[$i]}"
    TEST_NUM=$((i+1))

    # Run the profiling test
    run_profiling_test $BS $IN $OUT $TEST_NUM || {
        echo "Test $TEST_NUM failed, continuing to next test..."
    }
done

echo ""
echo "======================================"
echo "All profiling tests completed!"
echo "======================================"
echo "Trace files saved to: $TRACE_DIR/$CONFIG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Check trace files: ls -lh $TRACE_DIR/$CONFIG_DIR/*/"
echo "  2. Analyze traces with profiling tools"
echo ""
