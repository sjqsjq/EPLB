#!/bin/bash
#
# run_benchmark.sh - Automated benchmark wrapper for SGLang
#
# Usage:
#   ./run_benchmark.sh [OPTIONS]
#
# This script wraps benchmark.py to provide a convenient shell interface
# for running automated benchmark tests across multiple batch sizes and input lengths.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
CONFIG_FILE="config.yaml"
EXTRA_ARGS=""
PREFILL_URL=""
DECODE_URL=""
ROUTER_PORT="34001"

# Print usage
usage() {
    cat << EOF
${GREEN}SGLang Automated Benchmark Tool${NC}

Usage: $0 [OPTIONS]

Options:
  --config FILE              Config file (default: config.yaml)
  --base-url URL             Server base URL (formats: 81 | 11.139.21.81 | http://11.139.21.81:34567)
  --prefill URL              Prefill server URL for PD disaggregation (e.g., http://11.139.21.81:34567)
  --decode URL               Decode server URL for PD disaggregation (e.g., http://11.139.21.87:34567)
  --router-port PORT         Router port for PD mode (default: 34001)
  --backend BACKEND          Override backend (default: sglang)
  --warmup-requests N        Override warmup requests
  --max-queue-requests N     Override queue threshold
  --enable-profiler          Enable profiler (HTTP API)
  --skip-existing            Skip tests with existing results (resume)
  --verbose                  Enable debug logging
  -h, --help                 Show this help message

Examples:
  # Basic usage with short format
  $0 --base-url 81

  # PD disaggregation mode
  $0 --prefill 81 --decode 87

  # Override server URL (full format)
  $0 --base-url http://11.139.21.81:34567

  # Enable profiler and skip existing results
  $0 --base-url 81 --enable-profiler --skip-existing

  # Custom queue threshold
  $0 --base-url 81 --max-queue-requests 20

  # Verbose mode
  $0 --base-url 81 --verbose

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --base-url)
            EXTRA_ARGS="$EXTRA_ARGS --base-url $2"
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
        --backend)
            EXTRA_ARGS="$EXTRA_ARGS --backend $2"
            shift 2
            ;;
        --warmup-requests)
            EXTRA_ARGS="$EXTRA_ARGS --warmup-requests $2"
            shift 2
            ;;
        --max-queue-requests)
            EXTRA_ARGS="$EXTRA_ARGS --max-queue-requests $2"
            shift 2
            ;;
        --enable-profiler)
            EXTRA_ARGS="$EXTRA_ARGS --enable-profiler"
            shift
            ;;
        --skip-existing)
            EXTRA_ARGS="$EXTRA_ARGS --skip-existing"
            shift
            ;;
        --verbose)
            EXTRA_ARGS="$EXTRA_ARGS --verbose"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Read server port from config (fallback to 34567)
SERVER_PORT=$(python3 - <<'PY'
import yaml
try:
    config = yaml.safe_load(open("config.yaml")) or {}
    print(config.get("benchmark", {}).get("server", {}).get("port", 34567))
except Exception:
    print(34567)
PY
)

# Auto-complete base-url parameter
IP_PREFIX="11.139.21"
DEFAULT_PORT="$SERVER_PORT"

# Check if PD disaggregation mode (--prefill and --decode provided)
if [ -n "$PREFILL_URL" ] && [ -n "$DECODE_URL" ]; then
    echo -e "${YELLOW}PD Disaggregation Mode: Router will be used${NC}"

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

    # Extract prefill master IP
    PREFILL_IP=$(echo "$PREFILL_URL" | grep -oP 'http://\K[^:]+')

    # Set BASE_URL to router address on prefill master
    BASE_URL_ARG="http://${PREFILL_IP}:${ROUTER_PORT}"
    EXTRA_ARGS="$EXTRA_ARGS --base-url $BASE_URL_ARG"

    echo "  Prefill server: $PREFILL_URL"
    echo "  Decode server:  $DECODE_URL"
    echo "  Router address: $BASE_URL_ARG"

    PD_MODE=true
else
    BASE_URL_ARG=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--base-url )[^ ]+' || echo "")

    if [ -n "$BASE_URL_ARG" ]; then
        # Case 1: Short format (just last octet) -> http://11.139.21.XX:34567
        if [[ "$BASE_URL_ARG" =~ ^[0-9]+$ ]]; then
            BASE_URL_ARG="http://${IP_PREFIX}.${BASE_URL_ARG}:${DEFAULT_PORT}"
            echo "Auto-completed base URL: $BASE_URL_ARG"
            # Update EXTRA_ARGS with new base-url
            EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s|--base-url [^ ]*|--base-url $BASE_URL_ARG|")
        # Case 2: IP only (no http://) -> http://IP:34567
        elif [[ "$BASE_URL_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            BASE_URL_ARG="http://${BASE_URL_ARG}:${DEFAULT_PORT}"
            echo "Auto-completed base URL: $BASE_URL_ARG"
            EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s|--base-url [^ ]*|--base-url $BASE_URL_ARG|")
        # Case 3: Full URL (no change needed)
        fi
    fi

    PD_MODE=false
fi

# Read workspace from config
WORKSPACE_PATH=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['workspace_path'])")

# Determine container name based on mode
if [ "$PD_MODE" = true ]; then
    MASTER_CONTAINER="sjq_sglang_prefill_rank0"
else
    MASTER_CONTAINER="sjq_sglang_benchmark_rank0"
fi

# Validate required base-url
BASE_URL_ARG=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--base-url )[^ ]+' || echo "")
if [ -z "$BASE_URL_ARG" ]; then
    echo -e "${RED}Error: Must specify either --base-url OR (--prefill AND --decode)${NC}"
    echo "Usage: $0 --base-url URL [OPTIONS]"
    echo "   OR: $0 --prefill URL --decode URL [OPTIONS]"
    echo ""
    echo "URL formats:"
    echo "  Short format: --base-url 81              (auto-completes to http://11.139.21.81:${DEFAULT_PORT})"
    echo "  IP only:      --base-url 11.139.21.81    (auto-completes to http://11.139.21.81:${DEFAULT_PORT})"
    echo "  Full URL:     --base-url http://11.139.21.81:${DEFAULT_PORT}"
    echo ""
    echo "Examples:"
    echo "  $0 --base-url 81"
    echo "  $0 --prefill 81 --decode 87"
    echo "  $0 --base-url 81 --enable-profiler --skip-existing"
    exit 1
fi

# Print header
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SGLang Automated Benchmark${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${BLUE}Config:${NC} $CONFIG_FILE"
echo -e "${BLUE}Workspace:${NC} $WORKSPACE_PATH"
echo -e "${BLUE}Base URL:${NC} $BASE_URL_ARG"

# Check batch size mode
BATCH_SIZE_MODE=$(python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
        bs = config['benchmark']['test_matrix'].get('batch_sizes', [])
        if isinstance(bs, str) and bs.lower() == 'auto':
            print('auto')
        elif isinstance(bs, list) and len(bs) == 0:
            print('auto')
        else:
            print('manual')
except:
    print('unknown')
")

if [ "$BATCH_SIZE_MODE" = "auto" ]; then
    echo -e "${BLUE}Batch size mode:${NC} ${GREEN}AUTO${NC} (will calculate dynamically based on GPU memory)"
elif [ "$BATCH_SIZE_MODE" = "manual" ]; then
    echo -e "${BLUE}Batch size mode:${NC} ${YELLOW}MANUAL${NC} (using config.yaml values)"
else
    echo -e "${BLUE}Batch size mode:${NC} ${RED}UNKNOWN${NC}"
fi

echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check if ssh_util.py exists
if [ ! -f "ssh_util.py" ]; then
    echo -e "${RED}Error: ssh_util.py not found in current directory${NC}"
    exit 1
fi

# Check if benchmark.py exists
if [ ! -f "benchmark.py" ]; then
    echo -e "${RED}Error: benchmark.py not found in current directory${NC}"
    exit 1
fi

# Build command to run IN CONTAINER
if [ "$PD_MODE" = true ]; then
    # In PD mode, pass decode URL for server_info fetching
    CMD="python3 benchmark.py --config $CONFIG_FILE --decode-url $DECODE_URL $EXTRA_ARGS"
else
    CMD="python3 benchmark.py --config $CONFIG_FILE $EXTRA_ARGS"
fi

echo -e "${BLUE}Command (in container):${NC} $CMD"
echo ""

# Start router if in PD disaggregation mode
if [ "$PD_MODE" = true ]; then
    echo -e "${YELLOW}Starting router (if not already running)...${NC}"

    # Check if router is already running
    ROUTER_RUNNING=false
    if curl -s ${BASE_URL_ARG}/health >/dev/null 2>&1; then
        echo -e "  ${GREEN}Router is already running at $BASE_URL_ARG${NC}"
        ROUTER_RUNNING=true
    else
        echo "  Router not detected, starting new instance..."

        # Kill any existing router process in container
        python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
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

        echo "  Router will run in container: $MASTER_CONTAINER on $PREFILL_IP"

        python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
            "$ROUTER_CMD" >/dev/null 2>&1 &

        # Wait for router to be ready
        echo "  Waiting for router to be ready at $BASE_URL_ARG..."
        for i in {1..30}; do
            if curl -s ${BASE_URL_ARG}/health >/dev/null 2>&1; then
                echo -e "  ${GREEN}Router is ready!${NC}"
                ROUTER_RUNNING=true
                break
            fi
            if [ $i -eq 30 ]; then
                echo -e "${RED}Error: Router failed to start after 30 seconds${NC}"
                echo "Router log (from container):"
                python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
                    "tail -50 /tmp/sglang_router_${ROUTER_PORT}.log" 2>&1 || echo "Failed to read router log"
                exit 1
            fi
            sleep 1
        done
    fi
    echo ""
fi

# Execute benchmark in container via ssh_util.py
echo -e "${YELLOW}Starting benchmark in container...${NC}"
echo ""

# Extract master IP from base-url for remote execution
MASTER_IP=$(echo "$BASE_URL_ARG" | grep -oP 'http://\K[^:]+')
python3 ssh_util.py exec_in_container "$MASTER_IP" "$MASTER_CONTAINER" \
    "cd $WORKSPACE_PATH && $CMD"
EXIT_CODE=$?

# Print result
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Results saved to: result/benchmarks/"
    echo ""
    echo "To view results:"
    echo "  cd result/benchmarks/"
    echo "  ls -lh"
elif [ $EXIT_CODE -eq 130 ]; then
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Benchmark interrupted by user${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo ""
    echo "You can resume with --skip-existing:"
    echo "  $0 --skip-existing $EXTRA_ARGS"
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Benchmark failed with code $EXIT_CODE${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo "Check logs above for error details"
fi

exit $EXIT_CODE
