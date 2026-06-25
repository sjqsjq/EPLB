#!/bin/bash
#
# run_client.sh - Run SGLang benchmark client
#
# Usage: ./run_client.sh --base-url URL [options]
#
# Options:
#   --base-url URL         Server base URL (formats: 81 | 11.139.21.81 | http://11.139.21.81:34567)
#   --bs N                 Batch size (default: 32)
#   --input-len N          Input sequence length (default: 2000)
#   --output-len N         Output sequence length (default: 100)
#
# This script:
# 1. Reads model path from config.yaml
# 2. SSHes to master node and runs sglang.bench_one_batch_server in container
# 3. Displays benchmark results
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="config.yaml"
BATCH_SIZE=32
INPUT_LEN=2000
OUTPUT_LEN=100
BASE_URL=""
USE_IMAGE=false
IMAGE_RESOLUTION="1080p"
IMAGE_COUNT=1
PREFILL_URL=""
DECODE_URL=""
ROUTER_PORT="34001"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --image)
            USE_IMAGE=true
            shift
            ;;
        --image-resolution)
            IMAGE_RESOLUTION="$2"
            shift 2
            ;;
        --image-count)
            IMAGE_COUNT="$2"
            shift 2
            ;;
        --prefill)
            PREFILL_URL="$2"
            shift 2
            ;;
        --decode)
            DECODE_URL="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--base-url URL | --prefill URL --decode URL] [--bs N] [--input-len N] [--output-len N] [--image] [--image-resolution RES] [--image-count N]"
            echo ""
            echo "Mode 1: Unified or single PD instance"
            echo "  --base-url URL         Server URL (e.g., http://11.139.21.81:34567)"
            echo ""
            echo "Mode 2: PD disaggregation with router"
            echo "  --prefill URL          Prefill server URL (e.g., http://11.139.21.81:34567)"
            echo "  --decode URL           Decode server URL (e.g., http://11.139.21.87:34567)"
            echo "  --router-port PORT     Router port (default: 34001)"
            exit 1
            ;;
    esac
done

# Auto-complete base-url parameter
IP_PREFIX="11.139.21"
DEFAULT_PORT="34567"

# Check if PD disaggregation mode (--prefill and --decode provided)
if [ -n "$PREFILL_URL" ] && [ -n "$DECODE_URL" ]; then
    echo -e "${YELLOW}PD Disaggregation Mode: Router will be started${NC}"

    # Auto-complete prefill URL
    if [[ "$PREFILL_URL" =~ ^[0-9]+$ ]]; then
        PREFILL_URL="http://${IP_PREFIX}.${PREFILL_URL}:${DEFAULT_PORT}"
    elif [[ "$PREFILL_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        PREFILL_URL="http://${PREFILL_URL}:${DEFAULT_PORT}"
    fi

    # Auto-complete decode URL
    if [[ "$DECODE_URL" =~ ^[0-9]+$ ]]; then
        DECODE_URL="http://${IP_PREFIX}.${DECODE_URL}:${DEFAULT_PORT}"
    elif [[ "$DECODE_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        DECODE_URL="http://${DECODE_URL}:${DEFAULT_PORT}"
    fi

    # Set BASE_URL to router address (requests will be sent here)
    BASE_URL="http://127.0.0.1:${ROUTER_PORT}"

    echo "  Prefill server: $PREFILL_URL"
    echo "  Decode server:  $DECODE_URL"
    echo "  Router will listen on: $BASE_URL"

elif [ -n "$BASE_URL" ]; then
    # Standard mode: single server
    # Case 1: Short format (just last octet) -> http://11.139.21.XX:34567
    if [[ "$BASE_URL" =~ ^[0-9]+$ ]]; then
        BASE_URL="http://${IP_PREFIX}.${BASE_URL}:${DEFAULT_PORT}"
        echo "Auto-completed base URL: $BASE_URL"
    # Case 2: IP only (no http://) -> http://IP:34567
    elif [[ "$BASE_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        BASE_URL="http://${BASE_URL}:${DEFAULT_PORT}"
        echo "Auto-completed base URL: $BASE_URL"
    # Case 3: Full URL (no change needed)
    fi
else
    # Neither mode specified - error
    echo -e "${RED}Error: Must specify either --base-url OR (--prefill AND --decode)${NC}"
    echo ""
    echo "Mode 1: Unified or single PD instance"
    echo "  $0 --base-url URL [options]"
    echo ""
    echo "Mode 2: PD disaggregation with router"
    echo "  $0 --prefill URL --decode URL [options]"
    echo ""
    echo "Examples:"
    echo "  $0 --base-url 81 --bs 32 --input-len 2000"
    echo "  $0 --prefill 81 --decode 87 --bs 32 --input-len 2000"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SGLang Benchmark Client${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Parse configuration
echo -e "${YELLOW}[1/3] Reading configuration...${NC}"
CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

WORKSPACE_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")
MODEL_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_path'])")
MODEL_NAME=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_name'])")
FULL_MODEL_PATH="$MODEL_PATH/$MODEL_NAME"

# Read sudo password from environment variable or config (with default fallback)
SUDO_PASSWORD="${SUDO_PASSWORD:-$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sudo_password', 'Alibaba@12#\$'))")}"

echo "  Model path: $FULL_MODEL_PATH"
echo "  Base URL: $BASE_URL"
echo "  Batch size: $BATCH_SIZE"
echo "  Input length: $INPUT_LEN"
echo "  Output length: $OUTPUT_LEN"
if [ "$USE_IMAGE" = true ]; then
    echo "  Dataset: image"
    echo "  Image resolution: $IMAGE_RESOLUTION"
    echo "  Image count: $IMAGE_COUNT"
else
    echo "  Dataset: random (text)"
fi

# Start router if in PD disaggregation mode
if [ -n "$PREFILL_URL" ] && [ -n "$DECODE_URL" ]; then
    echo -e "${YELLOW}[2/4] Starting router...${NC}"

    # Extract prefill master IP
    PREFILL_IP=$(echo "$PREFILL_URL" | grep -oP 'http://\K[^:]+')

    # Router will run in prefill master container
    ROUTER_CONTAINER="sjq_sglang_prefill_rank0"

    # Kill any existing router process in container
    python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
        "pkill -f 'sglang.*router.*--port $ROUTER_PORT' 2>/dev/null || true" >/dev/null 2>&1 || true
    sleep 1

    # Start router in container (background)
    ROUTER_CMD="cd $WORKSPACE_PATH && PYTHONUNBUFFERED=1 nohup python -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill $PREFILL_URL \
        --decode $DECODE_URL \
        --host 0.0.0.0 \
        --port $ROUTER_PORT \
        > /tmp/sglang_router_${ROUTER_PORT}.log 2>&1 &"

    echo "  Router will run in container: $ROUTER_CONTAINER on $PREFILL_IP"
    echo "  Router command: $ROUTER_CMD"

    python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
        "$ROUTER_CMD" >/dev/null 2>&1 &

    # Update BASE_URL to router address on prefill master
    BASE_URL="http://${PREFILL_IP}:${ROUTER_PORT}"

    # Wait for router to be ready
    echo "  Waiting for router to be ready at $BASE_URL..."
    for i in {1..30}; do
        if curl -s ${BASE_URL}/health >/dev/null 2>&1; then
            echo -e "  ${GREEN}Router is ready!${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}Error: Router failed to start after 30 seconds${NC}"
            echo "Router log (from container):"
            python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
                "tail -50 /tmp/sglang_router_${ROUTER_PORT}.log" 2>&1 || echo "Failed to read router log"
            exit 1
        fi
        sleep 1
    done

    BENCHMARK_STEP="3"
    ROUTER_STARTED=true
else
    BENCHMARK_STEP="2"
    ROUTER_STARTED=false
fi

# Build benchmark command
echo -e "${YELLOW}[$BENCHMARK_STEP/3] Running benchmark...${NC}"

if [ "$USE_IMAGE" = true ]; then
    # Image dataset benchmark using custom script
    BENCH_CMD="python3 test_image_requests.py \
  --base-url $BASE_URL \
  --num-requests $BATCH_SIZE \
  --text-length $INPUT_LEN \
  --output-length $OUTPUT_LEN \
  --image-resolution $IMAGE_RESOLUTION \
  --image-count $IMAGE_COUNT"
else
    # Text-only dataset benchmark using bench_serving
    BENCH_CMD="python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url $BASE_URL \
  --tokenizer $FULL_MODEL_PATH \
  --num-prompts $BATCH_SIZE \
  --request-rate 1000 \
  --dataset-name random \
  --random-input-len $INPUT_LEN \
  --random-output-len $OUTPUT_LEN \
  --warmup-requests 1"
fi

echo "  Command: $BENCH_CMD"
echo ""
echo -e "${GREEN}--- Benchmark Output ---${NC}"

# Run benchmark (in prefill container for PD mode, in master container for unified mode)
if [ "$ROUTER_STARTED" = true ]; then
    # PD mode: run benchmark in prefill container (requests go to router in same container)
    python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
        "cd $WORKSPACE_PATH && $BENCH_CMD" || {
        echo -e "${RED}Benchmark execution failed${NC}"
        # Kill router on exit
        python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
            "pkill -f 'sglang.*router.*--port $ROUTER_PORT' 2>/dev/null || true" >/dev/null 2>&1 || true
        exit 1
    }
else
    # Unified mode: run benchmark in master container via SSH
    MASTER_IP=$(echo "$BASE_URL" | grep -oP 'http://\K[^:]+')
    MASTER_CONTAINER="sjq_sglang_benchmark_rank0"
    python3 ssh_util.py exec_in_container "$MASTER_IP" "$MASTER_CONTAINER" \
        "cd $WORKSPACE_PATH && $BENCH_CMD" || {
        echo -e "${RED}Benchmark execution failed${NC}"
        exit 1
    }
fi
echo -e "${GREEN}--- End Benchmark Output ---${NC}"

# Cleanup router if started
if [ "$ROUTER_STARTED" = true ]; then
    echo ""
    echo "Stopping router in container..."
    python3 ssh_util.py exec_in_container "$PREFILL_IP" "$ROUTER_CONTAINER" \
        "pkill -f 'sglang.*router.*--port $ROUTER_PORT' 2>/dev/null || true" >/dev/null 2>&1 || true
    echo "Router stopped"
fi

# Summary
echo ""
echo -e "${YELLOW}[$(( BENCHMARK_STEP + 1 ))/3] Summary${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Input length: $INPUT_LEN"
echo "  Output length: $OUTPUT_LEN"
echo "  Model: $FULL_MODEL_PATH"
echo "  Server: $BASE_URL"
echo ""
