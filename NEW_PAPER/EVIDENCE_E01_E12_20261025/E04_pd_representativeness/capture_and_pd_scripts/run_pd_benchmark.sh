#!/bin/bash
#
# run_pd_benchmark.sh - Automated PD (Prefill-Decode) Separation Benchmark
#
# This script automates the complete workflow for PD separation testing:
#   1. Deploy prefill and decode containers
#   2. Start prefill and decode servers
#   3. Start router
#   4. Run benchmark tests
#   5. Optional cleanup
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
PREFILL_MASTER="81"
DECODE_MASTER="87"
ROUTER_PORT="34001"
NNODES_PREFILL="2"
NNODES_DECODE="2"
SKIP_DEPLOY=false
SKIP_SERVER=false
SKIP_CLEANUP=false
BATCH_SIZES="8,16,32,64"
INPUT_LENS="2048,4096,8192"
OUTPUT_LEN="512"

# Print usage
usage() {
    cat << EOF
${GREEN}SGLang PD Separation Benchmark Tool${NC}

Usage: $0 [OPTIONS]

Options:
  --config FILE              Config file (default: config.yaml)
  --prefill-master IP        Prefill master IP (default: 81, auto-completes to 11.139.21.81)
  --decode-master IP         Decode master IP (default: 87, auto-completes to 11.139.21.87)
  --nnodes-prefill N         Number of nodes for prefill cluster (default: 2)
  --nnodes-decode N          Number of nodes for decode cluster (default: 2)
  --router-port PORT         Router port (default: 34001)
  --batch-sizes BS           Comma-separated batch sizes (default: 8,16,32,64)
  --input-lens LENS          Comma-separated input lengths (default: 2048,4096,8192)
  --output-len LEN           Output length (default: 512)
  --skip-deploy              Skip container deployment (use existing containers)
  --skip-server              Skip server startup (use running servers)
  --skip-cleanup             Skip cleanup after tests
  -h, --help                 Show this help message

Examples:
  # Full workflow (deploy, start servers, run tests, cleanup)
  $0

  # Use existing deployment and servers
  $0 --skip-deploy --skip-server

  # Custom configuration
  $0 --prefill-master 81 --decode-master 87 --nnodes-prefill 2 --nnodes-decode 2

  # Custom test matrix
  $0 --batch-sizes "4,8,16,32" --input-lens "1024,2048,4096" --output-len 256

  # Skip cleanup to inspect results
  $0 --skip-cleanup

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
        --prefill-master)
            PREFILL_MASTER="$2"
            shift 2
            ;;
        --decode-master)
            DECODE_MASTER="$2"
            shift 2
            ;;
        --nnodes-prefill)
            NNODES_PREFILL="$2"
            shift 2
            ;;
        --nnodes-decode)
            NNODES_DECODE="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --input-lens)
            INPUT_LENS="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --skip-server)
            SKIP_SERVER=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
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

# Auto-complete IPs
IP_PREFIX="11.139.21"
if [[ "$PREFILL_MASTER" =~ ^[0-9]+$ ]]; then
    PREFILL_MASTER_IP="${IP_PREFIX}.${PREFILL_MASTER}"
else
    PREFILL_MASTER_IP="$PREFILL_MASTER"
fi

if [[ "$DECODE_MASTER" =~ ^[0-9]+$ ]]; then
    DECODE_MASTER_IP="${IP_PREFIX}.${DECODE_MASTER}"
else
    DECODE_MASTER_IP="$DECODE_MASTER"
fi

# Read server port from config
SERVER_PORT=$(python3 - <<'PY'
import yaml
try:
    config = yaml.safe_load(open("config.yaml")) or {}
    print(config.get("benchmark", {}).get("server", {}).get("port", 34567))
except Exception:
    print(34567)
PY
)

PREFILL_URL="http://${PREFILL_MASTER_IP}:${SERVER_PORT}"
DECODE_URL="http://${DECODE_MASTER_IP}:${SERVER_PORT}"
ROUTER_URL="http://${PREFILL_MASTER_IP}:${ROUTER_PORT}"

# Print header
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}PD Separation Benchmark${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${BLUE}Config:${NC} $CONFIG_FILE"
echo -e "${BLUE}Prefill cluster:${NC} $PREFILL_MASTER_IP ($NNODES_PREFILL nodes)"
echo -e "${BLUE}Decode cluster:${NC} $DECODE_MASTER_IP ($NNODES_DECODE nodes)"
echo -e "${BLUE}Router:${NC} $ROUTER_URL"
echo -e "${BLUE}Test matrix:${NC}"
echo -e "  Batch sizes: $BATCH_SIZES"
echo -e "  Input lengths: $INPUT_LENS"
echo -e "  Output length: $OUTPUT_LEN"
echo ""

# Step 1: Deploy containers
if [ "$SKIP_DEPLOY" = false ]; then
    echo -e "${YELLOW}[1/4] Deploying containers...${NC}"
    echo -e "${BLUE}Deploying prefill containers...${NC}"
    ./run_container.sh --cur-node $NNODES_PREFILL --master-ip $PREFILL_MASTER --pd prefill

    echo -e "${BLUE}Deploying decode containers...${NC}"
    ./run_container.sh --cur-node $NNODES_DECODE --master-ip $DECODE_MASTER --pd decode

    echo -e "${GREEN}✓ Containers deployed${NC}"
    echo ""
else
    echo -e "${YELLOW}[1/4] Skipping container deployment (using existing containers)${NC}"
    echo ""
fi

# Step 2: Start servers
if [ "$SKIP_SERVER" = false ]; then
    echo -e "${YELLOW}[2/4] Starting PD servers...${NC}"

    echo -e "${BLUE}Starting prefill server...${NC}"
    bash run_server.sh --master-ip $PREFILL_MASTER --pd prefill --command \
        "python -m sglang.launch_server \
        --port $SERVER_PORT \
        --tp-size 8 \
        --ep-size 8 \
        --nnodes $NNODES_PREFILL \
        --dist-init-addr ${PREFILL_MASTER_IP}:31000 \
        --attention-backend trtllm_mha \
        --enable-dp-attention \
        --load-balance-method round_robin" &
    PREFILL_PID=$!

    echo -e "${BLUE}Starting decode server...${NC}"
    bash run_server.sh --master-ip $DECODE_MASTER --pd decode --command \
        "python -m sglang.launch_server \
        --port $SERVER_PORT \
        --tp-size 8 \
        --ep-size 8 \
        --nnodes $NNODES_DECODE \
        --dist-init-addr ${DECODE_MASTER_IP}:31000 \
        --attention-backend trtllm_mha \
        --enable-dp-attention \
        --deepep-mode low_latency \
        --prefill-round-robin-balance" &
    DECODE_PID=$!

    echo -e "${BLUE}Waiting for servers to start...${NC}"
    wait $PREFILL_PID
    wait $DECODE_PID

    echo -e "${GREEN}✓ Servers started${NC}"
    echo ""
else
    echo -e "${YELLOW}[2/4] Skipping server startup (using running servers)${NC}"
    echo ""
fi

# Step 3: Start router and run tests
echo -e "${YELLOW}[3/4] Running benchmark tests...${NC}"

# Check if router is running
echo -e "${BLUE}Checking router status...${NC}"
if curl -s ${ROUTER_URL}/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Router is already running${NC}"
else
    echo -e "${BLUE}Starting router...${NC}"
    WORKSPACE_PATH=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['workspace_path'])")

    python3 ssh_util.py exec_in_container "$PREFILL_MASTER_IP" "sjq_sglang_prefill_rank0" \
        "cd $WORKSPACE_PATH && PYTHONUNBUFFERED=1 nohup python -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill $PREFILL_URL \
        --decode $DECODE_URL \
        --host 0.0.0.0 \
        --port $ROUTER_PORT \
        > /tmp/sglang_router_${ROUTER_PORT}.log 2>&1 &" >/dev/null 2>&1 &

    # Wait for router
    echo -e "${BLUE}Waiting for router to be ready...${NC}"
    for i in {1..30}; do
        if curl -s ${ROUTER_URL}/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Router is ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}Error: Router failed to start${NC}"
            exit 1
        fi
        sleep 1
    done
fi

echo ""

# Run tests for each combination
IFS=',' read -ra BS_ARRAY <<< "$BATCH_SIZES"
IFS=',' read -ra IL_ARRAY <<< "$INPUT_LENS"

TOTAL_TESTS=$((${#BS_ARRAY[@]} * ${#IL_ARRAY[@]}))
CURRENT_TEST=0
PASSED_TESTS=0
FAILED_TESTS=0

echo -e "${BLUE}Running $TOTAL_TESTS tests...${NC}"
echo ""

for BS in "${BS_ARRAY[@]}"; do
    for IL in "${IL_ARRAY[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo -e "${YELLOW}[Test $CURRENT_TEST/$TOTAL_TESTS] BS=$BS, Input=$IL, Output=$OUTPUT_LEN${NC}"

        if ./run_client.sh --prefill $PREFILL_MASTER --decode $DECODE_MASTER \
            --bs $BS --input-len $IL --output-len $OUTPUT_LEN; then
            echo -e "${GREEN}✓ Test passed${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}✗ Test failed${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
        echo ""

        # Small delay between tests
        sleep 2
    done
done

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${BLUE}Total tests:${NC} $TOTAL_TESTS"
echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED_TESTS"
fi
echo ""

# Step 4: Cleanup
if [ "$SKIP_CLEANUP" = false ]; then
    echo -e "${YELLOW}[4/4] Cleaning up...${NC}"
    ./cleanup_containers.sh
    echo -e "${GREEN}✓ Cleanup complete${NC}"
else
    echo -e "${YELLOW}[4/4] Skipping cleanup (containers and servers still running)${NC}"
    echo ""
    echo -e "${BLUE}To manually cleanup later:${NC}"
    echo "  ./cleanup_containers.sh"
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All done!${NC}"
echo -e "${GREEN}======================================${NC}"

exit 0
