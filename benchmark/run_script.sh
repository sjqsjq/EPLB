#!/bin/bash
#
# run_script.sh - Run a custom command across containers on multiple nodes
#
# Usage: ./run_script.sh --master-ip IP --cur-node N [--command "CMD"] [--container NAME] [--config FILE]
#
# This script:
# 1. Reads node list from the nodelist file (same as run_server.sh)
# 2. Enters each container and executes the specified command
# 3. Sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK env vars automatically
# 4. RANK increments per node, WORLD_SIZE = --cur-node value, MASTER_ADDR = --master-ip
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="config.yaml"
MASTER_IP=""
WORLD_SIZE=""
CONTAINER_NAME="sjq_sglang_benchmark_rank0"  # Will be overridden per rank
MASTER_PORT=12345
CUSTOM_COMMAND=""
PYTHON_SCRIPT="/cpfs01/user/nebula_model/sjq-workspace/benchmark/scripts/test_low_latency.py"
SCRIPT_ARGS="--num-processes 4 --allow-mnnvl --num-tokens 256"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --cur-node)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --command)
            CUSTOM_COMMAND="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 --master-ip IP --cur-node N [--master-port PORT] [--command \"CMD\"] [--container NAME] [--config FILE]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MASTER_IP" ] || [ -z "$WORLD_SIZE" ]; then
    echo -e "${RED}Error: --master-ip and --cur-node are required${NC}"
    echo ""
    echo "Usage: $0 --master-ip IP --cur-node N [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --master-ip IP       Master node IP address (MASTER_ADDR)"
    echo "  --cur-node N         Total number of nodes (WORLD_SIZE)"
    echo ""
    echo "Optional:"
    echo "  --master-port PORT   Master port (default: 12345)"
    echo "  --command \"CMD\"      Custom command to run (overrides default)"
    echo "  --container NAME     Container name prefix (default: sjq_sglang_benchmark_rank)"
    echo "  --config FILE        Config file (default: config.yaml)"
    echo ""
    echo "Examples:"
    echo "  $0 --master-ip 11.139.21.81 --cur-node 2"
    echo "  $0 --master-ip 11.139.21.81 --cur-node 4 --master-port 12345"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-complete IP prefix if only last octet provided
IP_PREFIX="11.13.195"
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    MASTER_IP="${IP_PREFIX}.${MASTER_IP}"
    echo "Auto-completed master IP: $MASTER_IP"
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Distributed Script Runner${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Config: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Parse configuration to get workspace path
echo -e "${YELLOW}[1/4] Reading configuration...${NC}"
CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

WORKSPACE_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")
SUDO_PASSWORD="${SUDO_PASSWORD:-$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sudo_password', 'Alibaba@12#\$'))")}"

echo "  Workspace: $WORKSPACE_PATH"
echo "  Master IP: $MASTER_IP"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $WORLD_SIZE"

# Read node list
echo -e "${YELLOW}[2/4] Reading node list...${NC}"
NODE_LIST_FILE="$WORKSPACE_PATH/tmp/nodelist_${MASTER_IP}"
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo -e "${RED}Error: Node list file not found: $NODE_LIST_FILE${NC}"
    echo "  Please run: ./run_container.sh --cur-node $WORLD_SIZE --master-ip $MASTER_IP"
    exit 1
fi

NODES=($(cat "$NODE_LIST_FILE"))
NNODES=${#NODES[@]}
echo "  Found $NNODES nodes in node list"

# Validate node count matches WORLD_SIZE
if [ "$NNODES" -ne "$WORLD_SIZE" ]; then
    echo -e "${YELLOW}  Warning: Node list has $NNODES nodes but --cur-node specifies $WORLD_SIZE${NC}"
    echo "  Using $NNODES nodes from node list"
fi

# Build the command to execute on each node
echo -e "${YELLOW}[3/4] Preparing command...${NC}"

if [ -n "$CUSTOM_COMMAND" ]; then
    BASE_CMD="$CUSTOM_COMMAND"
    echo "  Using custom command"
else
    BASE_CMD="python $PYTHON_SCRIPT $SCRIPT_ARGS"
    echo "  Using default command: $BASE_CMD"
fi

echo ""

# Launch command on each node
echo -e "${YELLOW}[4/4] Launching on all nodes...${NC}"
RANK=0
for NODE in "${NODES[@]}"; do
    CONTAINER="sjq_sglang_benchmark_rank${RANK}"
    echo "  Launching on $NODE (rank $RANK, container: $CONTAINER)..."

    # Build the full command with env vars set inline
    FULL_CMD="export MASTER_ADDR=${MASTER_IP} && export MASTER_PORT=${MASTER_PORT} && export WORLD_SIZE=${WORLD_SIZE} && export RANK=${RANK} && cd $WORKSPACE_PATH && $BASE_CMD"

    # Print the actual command for debugging
    echo ""
    echo -e "${YELLOW}  [Node $RANK Command]${NC}"
    echo "    MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE RANK=$RANK"
    echo "    $BASE_CMD"
    echo ""

    # Execute in container in background
    python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER" \
        "$FULL_CMD" &

    RANK=$((RANK + 1))
    sleep 0.1
done

# Wait for all background jobs to complete
echo "  Waiting for all nodes to complete..."
wait

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All nodes finished!${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Nodes: $NNODES"
echo "MASTER_ADDR: $MASTER_IP"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""