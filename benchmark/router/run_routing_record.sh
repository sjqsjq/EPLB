#!/bin/bash
#
# run_routing_record.sh - Record MoE expert routing for load balance analysis
#
# Usage:
#   # Via run_script.sh (recommended):
#   ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
#     --command "bash router/run_routing_record.sh --model DeepSeek-R1 --dataset data/my_dataset.csv"
#
#   # With custom text column:
#   ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
#     --command "bash router/run_routing_record.sh --model DeepSeek-R1 \
#       --dataset data/my_dataset.csv --text-column prompt --max-prompts 200"
#

set -e

MODEL_NAME=""
DATASET=""
TEXT_COLUMN="text"
MAX_PROMPTS=100
MAX_SEQ_LEN=0
DTYPE="${DTYPE:-bfloat16}"
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --text-column) TEXT_COLUMN="$2"; shift 2 ;;
        --max-prompts) MAX_PROMPTS="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${CONFIG_FILE:-${WORKSPACE_DIR}/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['model_path'])")
CONFIG_MODEL_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['model_name'])")

MODEL_NAME="${MODEL_NAME:-$CONFIG_MODEL_NAME}"
FULL_MODEL_PATH="${CONFIG_MODEL_PATH}/${MODEL_NAME}"

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required"
    echo "Usage: $0 --dataset <csv_path> [--model NAME] [--text-column COL] [--max-prompts N]"
    exit 1
fi

# Resolve dataset path relative to workspace
if [[ "$DATASET" != /* ]]; then
    DATASET="${WORKSPACE_DIR}/${DATASET}"
fi

OUTPUT_DIR="${WORKSPACE_DIR}/result/expert"
OUTPUT_FILE="${OUTPUT_DIR}/routing_record_${MODEL_NAME}.json"
LOG_FILE="${OUTPUT_DIR}/routing_record.log"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MoE Expert Routing Recorder"
echo "  (Load Balance Analysis)"
echo "=============================================="
echo "Model:       ${FULL_MODEL_PATH}"
echo "Dataset:     ${DATASET}"
echo "Text column: ${TEXT_COLUMN}"
echo "Max prompts: ${MAX_PROMPTS}"
echo "Max seq len: ${MAX_SEQ_LEN}"
echo "Dtype:       ${DTYPE}"
echo "Output:      ${OUTPUT_FILE}"
echo "=============================================="

if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $FULL_MODEL_PATH"
    exit 1
fi
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Installing dependencies..."
pip install accelerate -q 2>/dev/null || true

echo ""
echo "[$(date '+%H:%M:%S')] Starting evaluation in background (to avoid SSH 300s timeout)..."
echo "[$(date '+%H:%M:%S')] NOTE: Model loading may take 5-15 minutes for large models."
echo ""
echo "  Log file: $LOG_FILE"
echo "  Monitor:  tail -f $LOG_FILE"
echo ""

# Run in background with nohup to avoid SSH timeout killing the process.
# All output goes to log file. Use 'tail -f' to monitor.
nohup bash -c "
PYTHONUNBUFFERED=1 python3 '${SCRIPT_DIR}/routing_recorder.py' \
    --model-path '$FULL_MODEL_PATH' \
    --dataset '$DATASET' \
    --text-column '$TEXT_COLUMN' \
    --max-prompts '$MAX_PROMPTS' \
    --max-seq-len '$MAX_SEQ_LEN' \
    --dtype '$DTYPE' \
    --output '$OUTPUT_FILE' \
    --device-map auto \
    --trust-remote-code \
    2>&1 | tee -a '$LOG_FILE'

echo '' >> '$LOG_FILE'
echo '============================================' >> '$LOG_FILE'
echo \"[\$(date '+%H:%M:%S')] COMPLETED. Results: $OUTPUT_FILE\" >> '$LOG_FILE'
echo '============================================' >> '$LOG_FILE'
" > "$LOG_FILE" 2>&1 &

BGPID=$!
echo "[$(date '+%H:%M:%S')] Background PID: $BGPID"
echo "[$(date '+%H:%M:%S')] Waiting 10s to confirm startup..."
sleep 10

# Show first few lines to confirm it started
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "--- First lines of log (confirming startup) ---"
    head -20 "$LOG_FILE"
    echo "--- End preview ---"
fi

echo ""
echo "=============================================="
echo "Script launched successfully!"
echo ""
echo "  [查看实时日志] 用以下命令查看运行进度:"
echo "    ./run_script.sh --master-ip <IP> --cur-node 1 --command \"tail -100 $LOG_FILE\""
echo ""
echo "  [防止300s超时] 脚本已用 nohup 后台运行，SSH 断连不影响任务。"
echo "    重连后用上面的 tail 命令即可查看进度。"
echo ""
echo "  Result: $OUTPUT_FILE"
echo "=============================================="

