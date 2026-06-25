#!/bin/bash
#
# patch_eplb.sh - Copy instrumented eplb_manager.py to all containers
#
# Usage: ./patch_eplb.sh --master-ip IP [--config FILE] [--src FILE]
#
# This script:
# 1. Reads the nodelist for the given master-ip
# 2. Copies the modified eplb_manager.py into each container
# 3. Does NOT modify any other files or scripts
#
# Run this AFTER run_container.sh and BEFORE run_server.sh
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIG_FILE="config.yaml"
MASTER_IP=""
SRC_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip) MASTER_IP="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --src) SRC_FILE="$2"; shift 2 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

if [ -z "$MASTER_IP" ]; then
    echo -e "${RED}Error: --master-ip is required${NC}"
    echo "Usage: $0 --master-ip IP [--config FILE] [--src FILE]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-complete IP
IP_PREFIX="11.139.21"
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    MASTER_IP="${IP_PREFIX}.${MASTER_IP}"
fi

# Default source: sglang_src in workspace
if [ -z "$SRC_FILE" ]; then
    SRC_FILE="$SCRIPT_DIR/sglang_src/python/sglang/srt/eplb/eplb_manager.py"
fi

if [ ! -f "$SRC_FILE" ]; then
    echo -e "${RED}Error: Source file not found: $SRC_FILE${NC}"
    exit 1
fi

# Read nodelist
NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_${MASTER_IP}"
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo -e "${RED}Error: Node list not found: $NODE_LIST_FILE${NC}"
    echo "  Run ./run_container.sh first"
    exit 1
fi

NODES=($(cat "$NODE_LIST_FILE"))
DEST="/sgl-workspace/sglang/python/sglang/srt/eplb/eplb_manager.py"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Patching EPLB Manager${NC}"
echo -e "${GREEN}======================================${NC}"
echo "  Source:  $SRC_FILE"
echo "  Dest:    $DEST"
echo "  Nodes:   ${#NODES[@]}"
echo ""

RANK=0
ALL_OK=true
for NODE in "${NODES[@]}"; do
    CONTAINER="sjq_sglang_benchmark_rank${RANK}"
    echo -n "  Rank $RANK ($NODE:$CONTAINER)... "

    RESULT=$(python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER" \
        "cp $SRC_FILE $DEST && echo OK" 2>&1 | tail -1)

    if echo "$RESULT" | grep -q "OK"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        ALL_OK=false
    fi
    RANK=$((RANK + 1))
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All containers patched successfully.${NC}"
else
    echo -e "${RED}Some containers failed to patch. Check above.${NC}"
    exit 1
fi
