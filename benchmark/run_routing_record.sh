#!/bin/bash
#
# run_routing_record.sh - Control GPU-based MoE routing distribution recorder
#
# This recorder hooks topk_ids (gate output = real routing decision) via
# pure-GPU scatter_add_, so it works correctly with CUDA Graph and DeepEP
# low_latency mode (unlike the native recorder which only sees masked_m).
#
# Usage:
#   ./run_routing_record.sh dump   --master-ip 90              # Trigger dump on all nodes
#   ./run_routing_record.sh show   --master-ip 90              # List & preview .pt files
#   ./run_routing_record.sh reset  --master-ip 90              # Reset counters (not impl)
#   ./run_routing_record.sh fetch  --master-ip 90              # Copy .pt to local result/
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
MASTER_IP=""
CONFIG_FILE="config.yaml"

# Parse arguments
ACTION="${1:-}"
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip)  MASTER_IP="$2";      shift 2 ;;
        --config)     CONFIG_FILE="$2";    shift 2 ;;
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
    echo "  $0 dump   --master-ip 90   # Trigger dump on all nodes"
    echo "  $0 show   --master-ip 90   # List & preview .pt files"
    echo "  $0 fetch  --master-ip 90   # Copy .pt files to local result/"
    exit 1
fi

# Read config
WORKSPACE_PATH=""
if [ -f "$CONFIG_FILE" ]; then
    WORKSPACE_PATH=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    print(yaml.safe_load(f).get('workspace_path', ''))
" 2>/dev/null || echo "")
fi
ROUTING_RECORD_DIR="${WORKSPACE_PATH:-.}/result/routing_record"

# Read node list
NODELIST_FILE="${WORKSPACE_PATH}/tmp/nodelist_${MASTER_IP}"
if [ ! -f "$NODELIST_FILE" ]; then
    echo -e "${RED}Error: Node list not found: $NODELIST_FILE${NC}"
    echo "  Run run_container.sh first to create the node list."
    exit 1
fi
readarray -t NODES < "$NODELIST_FILE"
NUM_NODES=${#NODES[@]}

do_dump() {
    echo -e "${CYAN}[Routing Record] Triggering dump on ${NUM_NODES} nodes...${NC}"
    echo -e "  Output dir: ${ROUTING_RECORD_DIR}"
    echo ""

    RANK=0
    for NODE in "${NODES[@]}"; do
        CONTAINER_NAME="sjq_sglang_benchmark_rank${RANK}"
        TRIGGER_FILE="${ROUTING_RECORD_DIR}/.dump_trigger"

        echo -n "  Node ${NODE} (rank ${RANK}): "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "touch ${TRIGGER_FILE}" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}triggered${NC}"
        else
            echo -e "${RED}failed${NC}"
        fi
        RANK=$((RANK + 1))
    done

    echo ""
    echo -e "${YELLOW}Waiting 5s for dump to complete...${NC}"
    sleep 5

    do_show
}

do_show() {
    echo -e "${CYAN}[Routing Record] Listing .pt files in ${ROUTING_RECORD_DIR}...${NC}"
    echo ""

    FIRST_NODE="${NODES[0]}"
    CONTAINER_NAME="sjq_sglang_benchmark_rank0"

    # Write a temp python script to shared storage, then exec in container.
    # This avoids all quote-escaping issues in bash -> SSH -> pouch exec chain.
    SHOW_SCRIPT="${ROUTING_RECORD_DIR}/_show_routing.py"
    cat > /tmp/_show_routing.py << 'PYEOF'
import glob, os, sys, datetime

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/routing_record"
patterns = [os.path.join(data_dir, p) for p in ["routing_rank*.pt", "routing_record_*.pt"]]
files = []
for pat in patterns:
    files.extend(glob.glob(pat))
# deduplicate and sort by mtime desc
files = sorted(set(files), key=os.path.getmtime, reverse=True)

if not files:
    print("No .pt files found in", data_dir)
    sys.exit(0)

print(f"Found {len(files)} .pt file(s):")
for f in files:
    sz = os.path.getsize(f)
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  {os.path.basename(f):50s} {sz:>10,} bytes  {ts}")
print()

import torch
latest = files[0]
print(f"Preview of latest: {os.path.basename(latest)}")
data = torch.load(latest, map_location="cpu", weights_only=False)
ne = data.get("num_experts", 0)
nl = data.get("num_layers", 0)
rank = data.get("rank", "?")
print(f"  Rank: {rank}  Layers: {nl}  Experts: {ne}")
ts = data.get("total_steps", 0)
ps = data.get("prefill_steps", 0)
ds = data.get("decode_steps", 0)
print(f"  Steps: total={ts} prefill={ps} decode={ds}")

for phase in ["prefill", "decode"]:
    key = f"{phase}_counts"
    if key not in data:
        print(f"  [{phase.upper()}] key not found")
        continue
    rc = data[key]
    total = rc.sum().item()
    if total == 0:
        print(f"  [{phase.upper()}] No data")
        continue
    print(f"  [{phase.upper()}] Total routed tokens: {total:,}")
    for lid in range(nl):
        lt = rc[lid].sum().item()
        if lt == 0:
            continue
        act = (rc[lid] > 0).sum().item()
        mx = rc[lid].max().item()
        mi = rc[lid].argmax().item()
        mn = rc[lid][rc[lid] > 0].min().item()
        print(f"    L{lid:>2d}: total={lt:>12,} active={act:>3d}/{ne} max=e{mi}({mx:,}) min={mn:,} ratio={mx/max(mn,1):.1f}x")
PYEOF

    # Copy script to shared storage so container can access it
    scp /tmp/_show_routing.py "${FIRST_NODE}:${SHOW_SCRIPT}" 2>/dev/null

    # Execute inside container
    python3 ssh_util.py exec_in_container "$FIRST_NODE" "$CONTAINER_NAME" \
        "python3 ${SHOW_SCRIPT} ${ROUTING_RECORD_DIR}"

    # Cleanup
    rm -f /tmp/_show_routing.py
}

do_fetch() {
    echo -e "${CYAN}[Routing Record] Fetching .pt files to local...${NC}"

    LOCAL_DIR="./result/routing_record"
    mkdir -p "$LOCAL_DIR"

    FIRST_NODE="${NODES[0]}"

    # Direct scp from shared storage (no container needed)
    echo -n "  Copying from ${FIRST_NODE}:${ROUTING_RECORD_DIR}/ ... "
    scp "${FIRST_NODE}:${ROUTING_RECORD_DIR}/routing_rank*.pt" "${LOCAL_DIR}/" 2>/dev/null
    scp "${FIRST_NODE}:${ROUTING_RECORD_DIR}/routing_record_*.pt" "${LOCAL_DIR}/" 2>/dev/null

    echo ""
    echo -e "${GREEN}Files saved to: ${LOCAL_DIR}/${NC}"
    ls -lh "${LOCAL_DIR}"/*.pt 2>/dev/null || echo -e "${YELLOW}No .pt files found${NC}"
}

do_check() {
    echo -e "${CYAN}[Routing Record] Checking setup on all nodes...${NC}"
    echo ""

    SGLANG_FUSED_MOE_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton"

    RANK=0
    for NODE in "${NODES[@]}"; do
        CONTAINER_NAME="sjq_sglang_benchmark_rank${RANK}"
        echo -e "${YELLOW}--- Node ${NODE} (rank ${RANK}, container: ${CONTAINER_NAME}) ---${NC}"

        # 1. Check env var
        echo -n "  SGLANG_ROUTING_RECORD env: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "echo \$SGLANG_ROUTING_RECORD" 2>/dev/null || echo "FAIL"

        # 2. Check routing_logger.py exists
        echo -n "  routing_logger.py: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "test -f ${SGLANG_FUSED_MOE_DIR}/routing_logger.py && echo OK || echo MISSING" 2>/dev/null || echo "FAIL"

        # 3. Check layer.py has routing_logger import
        echo -n "  layer.py patched: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "grep -c routing_logger ${SGLANG_FUSED_MOE_DIR}/layer.py 2>/dev/null || echo 0" 2>/dev/null || echo "FAIL"

        # 4. Check dump watcher thread is running
        echo -n "  dump watcher thread: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "grep -r routing-dump-watcher /proc/*/task/*/comm 2>/dev/null | head -1 | grep -q routing && echo OK || echo NOT_FOUND" 2>/dev/null || echo "FAIL"

        # 5. Check .pt files in output dir
        echo -n "  .pt files: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "ls ${ROUTING_RECORD_DIR}/*.pt 2>/dev/null | wc -l" 2>/dev/null || echo "0"

        # 6. Check server log for routing recorder
        echo -n "  server log RoutingRecord: "
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "grep -c RoutingRecord ${WORKSPACE_PATH}/log/worker/*rank${RANK}*.log 2>/dev/null || echo 0" 2>/dev/null || echo "FAIL"

        echo ""
        RANK=$((RANK + 1))
    done
}

case "$ACTION" in
    dump)  do_dump ;;
    show)  do_show ;;
    fetch) do_fetch ;;
    check) do_check ;;
    *)
        echo ""
        echo "Usage:"
        echo "  $0 dump   --master-ip 90   # Trigger dump on all nodes"
        echo "  $0 show   --master-ip 90   # List & preview .pt files"
        echo "  $0 fetch  --master-ip 90   # Copy .pt files to local result/"
        echo "  $0 check  --master-ip 90   # Check setup status on all nodes"
        echo ""
        echo "Setup steps:"
        echo "  1. Start containers:  ./run_container.sh --cur-node 4 --master-ip 90 --enable-routing-record"
        echo "  2. Launch server:     ./run_server.sh --command '...' --master-ip 90"
        echo "  3. Send requests:     ./single_bench.sh --base-url 90 --bs 256 --input-len 8192"
        echo "  4. Dump results:      $0 dump --master-ip 90"
        echo "  5. Fetch to local:    $0 fetch --master-ip 90"
        exit 1
        ;;
esac

