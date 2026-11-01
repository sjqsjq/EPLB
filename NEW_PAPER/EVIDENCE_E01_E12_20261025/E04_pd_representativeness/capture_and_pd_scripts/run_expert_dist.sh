#!/bin/bash
#
# run_expert_dist.sh - Control SGLang native expert distribution recorder
#
# Usage:
#   ./run_expert_dist.sh start   --master-ip 81 --port 34567   # Start recording
#   ./run_expert_dist.sh stop    --master-ip 81 --port 34567   # Stop recording
#   ./run_expert_dist.sh dump    --master-ip 81 --port 34567   # Dump to CSV files
#   ./run_expert_dist.sh all     --master-ip 81 --port 34567   # Start → wait → dump
#   ./run_expert_dist.sh show    --master-ip 81                # Show dumped CSV files
#
# Output CSV path: <workspace_path>/result/expert_dist/expert_distribution_rank*_timestamp*.csv
# CSV format:  layer_id,expert_id,count
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
MASTER_IP=""
PORT="34567"
CONFIG_FILE="config.yaml"
WAIT_SECONDS=60  # For "all" mode: how long to wait between start and dump

# Parse arguments
ACTION="${1:-}"
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip)  MASTER_IP="$2";      shift 2 ;;
        --port)       PORT="$2";           shift 2 ;;
        --config)     CONFIG_FILE="$2";    shift 2 ;;
        --wait)       WAIT_SECONDS="$2";   shift 2 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Auto-complete IP prefix
IP_PREFIX="11.139.21"
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    MASTER_IP="${IP_PREFIX}.${MASTER_IP}"
fi

if [ -z "$MASTER_IP" ]; then
    echo -e "${RED}Error: --master-ip is required${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 start  --master-ip 81 [--port 34567]              # Start recording"
    echo "  $0 stop   --master-ip 81 [--port 34567]              # Stop recording"
    echo "  $0 dump   --master-ip 81 [--port 34567]              # Dump to CSV"
    echo "  $0 all    --master-ip 81 [--port 34567] [--wait 60]  # Start → wait → dump"
    echo "  $0 show   --master-ip 81                              # Show CSV files"
    exit 1
fi

BASE_URL="http://${MASTER_IP}:${PORT}"

# Read workspace_path from config
WORKSPACE_PATH=""
if [ -f "$CONFIG_FILE" ]; then
    WORKSPACE_PATH=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    print(yaml.safe_load(f).get('workspace_path', ''))
" 2>/dev/null || echo "")
fi
EXPERT_DIST_DIR="${WORKSPACE_PATH:-.}/result/expert_dist"

do_start() {
    echo -e "${CYAN}[Expert Dist] Starting recording on ${BASE_URL}...${NC}"
    RESP=$(curl -s -w "\n%{http_code}" -X POST \
        -H 'Content-Type: application/json' \
        "${BASE_URL}/start_expert_distribution_record" -d '{}' 2>&1)
    HTTP_CODE=$(echo "$RESP" | tail -1)
    BODY=$(echo "$RESP" | head -n -1)
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Recording started${NC}"
        echo "  Response: $BODY"
    else
        echo -e "${RED}✗ Failed (HTTP $HTTP_CODE)${NC}"
        echo "  Response: $BODY"
        return 1
    fi
}

do_stop() {
    echo -e "${CYAN}[Expert Dist] Stopping recording on ${BASE_URL}...${NC}"
    RESP=$(curl -s -w "\n%{http_code}" -X POST \
        -H 'Content-Type: application/json' \
        "${BASE_URL}/stop_expert_distribution_record" -d '{}' 2>&1)
    HTTP_CODE=$(echo "$RESP" | tail -1)
    BODY=$(echo "$RESP" | head -n -1)
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Recording stopped${NC}"
        echo "  Response: $BODY"
    else
        echo -e "${RED}✗ Failed (HTTP $HTTP_CODE)${NC}"
        echo "  Response: $BODY"
        return 1
    fi
}

do_dump() {
    echo -e "${CYAN}[Expert Dist] Dumping records on ${BASE_URL}...${NC}"
    RESP=$(curl -s -w "\n%{http_code}" -X POST \
        -H 'Content-Type: application/json' \
        "${BASE_URL}/dump_expert_distribution_record" -d '{}' 2>&1)
    HTTP_CODE=$(echo "$RESP" | tail -1)
    BODY=$(echo "$RESP" | head -n -1)
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Records dumped${NC}"
        echo "  Response: $BODY"
        echo ""
        echo -e "${YELLOW}Output directory: ${EXPERT_DIST_DIR}${NC}"
        echo "  CSV format: layer_id,expert_id,count"
        echo "  File pattern: expert_distribution_rank*_timestamp*.csv"
    else
        echo -e "${RED}✗ Failed (HTTP $HTTP_CODE)${NC}"
        echo "  Response: $BODY"
        return 1
    fi
}

do_show() {
    echo -e "${CYAN}[Expert Dist] Listing CSV files in ${EXPERT_DIST_DIR}...${NC}"
    echo ""

    # Try to list files (via SSH if workspace is on CPFS)
    if [ -d "$EXPERT_DIST_DIR" ]; then
        FILES=$(ls -lht "$EXPERT_DIST_DIR"/expert_distribution_*.csv 2>/dev/null || true)
    else
        # Try via first node in nodelist
        NODELIST_FILE="${WORKSPACE_PATH}/tmp/nodelist_${MASTER_IP}"
        if [ -f "$NODELIST_FILE" ]; then
            FIRST_NODE=$(head -1 "$NODELIST_FILE")
            FILES=$(python3 ssh_util.py exec_on_node "$FIRST_NODE" \
                "ls -lht $EXPERT_DIST_DIR/expert_distribution_*.csv 2>/dev/null" 2>/dev/null || true)
        else
            FILES=""
        fi
    fi

    if [ -z "$FILES" ]; then
        echo -e "${YELLOW}No CSV files found yet.${NC}"
        echo "  Make sure you have:"
        echo "    1. Started with --expert-distribution-recorder-mode stat"
        echo "    2. Called: $0 start --master-ip ..."
        echo "    3. Sent some requests"
        echo "    4. Called: $0 dump --master-ip ..."
        return 0
    fi

    echo "$FILES"
    echo ""

    # Show head of the latest file
    LATEST=$(echo "$FILES" | head -1 | awk '{print $NF}')
    if [ -n "$LATEST" ]; then
        echo -e "${GREEN}Preview of latest file ($LATEST):${NC}"
        if [ -f "$LATEST" ]; then
            head -20 "$LATEST"
        else
            FIRST_NODE=$(head -1 "${WORKSPACE_PATH}/tmp/nodelist_${MASTER_IP}" 2>/dev/null || echo "")
            if [ -n "$FIRST_NODE" ]; then
                python3 ssh_util.py exec_on_node "$FIRST_NODE" "head -20 $LATEST" 2>/dev/null || true
            fi
        fi
        TOTAL_LINES=$(wc -l < "$LATEST" 2>/dev/null || echo "?")
        echo "  ... ($TOTAL_LINES total lines)"
    fi
}

do_all() {
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Expert Distribution: Full Capture${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    do_start || exit 1
    echo ""
    echo -e "${YELLOW}Recording for ${WAIT_SECONDS} seconds...${NC}"
    echo -e "${YELLOW}(Send requests to the server during this time)${NC}"
    echo ""

    # Countdown
    for i in $(seq "$WAIT_SECONDS" -10 10); do
        echo "  ${i}s remaining..."
        sleep 10
    done
    REMAINING=$((WAIT_SECONDS % 10))
    if [ "$REMAINING" -gt 0 ]; then
        sleep "$REMAINING"
    fi

    echo ""
    do_stop || true
    echo ""
    do_dump || exit 1
    echo ""
    do_show
}

case "$ACTION" in
    start) do_start ;;
    stop)  do_stop ;;
    dump)  do_dump ;;
    show)  do_show ;;
    all)   do_all ;;
    *)
        echo -e "${RED}Error: Unknown action '$ACTION'${NC}"
        echo ""
        echo "Usage:"
        echo "  $0 start  --master-ip 81 [--port 34567]              # Start recording"
        echo "  $0 stop   --master-ip 81 [--port 34567]              # Stop recording"
        echo "  $0 dump   --master-ip 81 [--port 34567]              # Dump to CSV"
        echo "  $0 all    --master-ip 81 [--port 34567] [--wait 60]  # Start → wait → dump"
        echo "  $0 show   --master-ip 81                              # Show CSV files"
        exit 1
        ;;
esac

