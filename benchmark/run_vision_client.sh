#!/bin/bash
#
# run_vision_client.sh - Run vision benchmark client
#
# Usage: ./run_vision_client.sh --base-url URL [options]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
CONFIG_FILE="config.yaml"
BASE_URL=""
BATCH_SIZE=32
IMAGE_COUNT=4
TEXT_TOKENS=1000
OUTPUT_TOKENS=512
IMAGE_RESOLUTION="1080p"
SAVE_RESULT=false
OUTPUT_DIR="./result/vision_benchmarks"
VERBOSE=false

# Usage function
usage() {
    cat <<EOF
Usage: $0 --base-url URL [options]

Options:
  --base-url URL           Server URL (formats: 81 | 11.139.21.81 | http://11.139.21.81:34567)
  --batch-size N           Number of concurrent requests (default: 32)
  --image-count N          Images per request (default: 4)
  --text-tokens N          Text tokens per request (default: 1000)
  --output-tokens N        Output tokens (default: 512)
  --image-resolution STR   Image resolution: 480p, 720p, 1080p, 4k, or WxH (default: 1080p = 2040 vision tokens for Qwen-VL)
  --save-result            Save results to JSON
  --output-dir PATH        Output directory (default: ./result/vision_benchmarks)
  --verbose                Verbose logging
  -h, --help               Show this help

Examples:
  # Basic test
  ./run_vision_client.sh --base-url 81

  # Custom batch size
  ./run_vision_client.sh --base-url 81 --batch-size 64

  # Custom configuration
  ./run_vision_client.sh --base-url 81 --batch-size 32 --image-count 8 --text-tokens 2000

  # Save results
  ./run_vision_client.sh --base-url 81 --save-result
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --image-count)
            IMAGE_COUNT="$2"
            shift 2
            ;;
        --text-tokens)
            TEXT_TOKENS="$2"
            shift 2
            ;;
        --output-tokens)
            OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --image-resolution)
            IMAGE_RESOLUTION="$2"
            shift 2
            ;;
        --save-result)
            SAVE_RESULT=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Auto-complete base-url parameter
IP_PREFIX="11.139.21"
DEFAULT_PORT="34567"

if [ -n "$BASE_URL" ]; then
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
fi

# Validate required arguments
if [ -z "$BASE_URL" ]; then
    echo -e "${RED}Error: --base-url is required${NC}"
    usage
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Read workspace path from config
WORKSPACE_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['workspace_path'])" 2>/dev/null)
if [ -z "$WORKSPACE_PATH" ]; then
    echo -e "${YELLOW}Warning: Could not read workspace_path from $CONFIG_FILE, using current directory${NC}"
    WORKSPACE_PATH="$SCRIPT_DIR"
fi

# Build vision client command
VISION_CMD="python3 vision_client.py \
  --base-url $BASE_URL \
  --batch-size $BATCH_SIZE \
  --image-count $IMAGE_COUNT \
  --text-tokens $TEXT_TOKENS \
  --output-tokens $OUTPUT_TOKENS \
  --image-resolution $IMAGE_RESOLUTION"

if [ "$SAVE_RESULT" = true ]; then
    VISION_CMD="$VISION_CMD --save-result --output-dir $OUTPUT_DIR"
fi

if [ "$VERBOSE" = true ]; then
    VISION_CMD="$VISION_CMD --verbose"
fi

echo -e "${GREEN}Running vision benchmark...${NC}"
echo "  Base URL: $BASE_URL"
echo "  Batch size: $BATCH_SIZE"
echo "  Images per request: $IMAGE_COUNT × $IMAGE_RESOLUTION"
echo "  Text tokens: $TEXT_TOKENS"
echo "  Output tokens: $OUTPUT_TOKENS"
echo ""

# Extract master IP from BASE_URL
MASTER_IP=$(echo "$BASE_URL" | grep -oP 'http://\K[^:]+')
MASTER_CONTAINER="sjq_sglang_benchmark_rank0"

# Execute in master container
echo -e "${GREEN}Executing in container $MASTER_CONTAINER on $MASTER_IP...${NC}"
python3 ssh_util.py exec_in_container "$MASTER_IP" "$MASTER_CONTAINER" \
    "cd $WORKSPACE_PATH && $VISION_CMD"
