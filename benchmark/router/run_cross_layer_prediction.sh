#!/bin/bash
#
# run_cross_layer_prediction.sh - Run cross-layer routing prediction verification
#
# This script runs INSIDE the container environment (launched by run_container.sh).
# It uses a single node with multiple GPUs to load DeepSeek-V3 and verify
# cross-layer expert routing prediction accuracy (Libra/Fate paper).
#
# Usage:
#   # Step 1: Launch container (1 node is enough)
#   ./run_container.sh --cur-node 1 --master-ip 11.139.21.90
#
#   # Step 2: Run this script via run_script.sh (background mode to avoid SSH timeout)
#   ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
#     --command "bash router/run_cross_layer_prediction.sh"
#
#   # Or with custom model:
#   ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
#     --command "bash router/run_cross_layer_prediction.sh --model DeepSeek-V3"
#
#   # Monitor progress (from outside):
#   ./run_script.sh --master-ip 11.139.21.90 --cur-node 1 \
#     --command "tail -f result/expert/cross_layer_prediction.log"
#
# Environment (auto-detected from config.yaml inside container):
#   MODEL_PATH: /cpfs01/user/nebula_model/llm_weight
#   WORKSPACE:  /cpfs01/user/nebula_model/sjq-workspace/benchmark
#

set -e

# Default parameters (can override via CLI)
MODEL_NAME=""
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
CONFIG_FILE=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model NAME] [--dtype TYPE] [--max-new-tokens N] [--num-prompts N] [--config FILE]"
            exit 1
            ;;
    esac
done

# Detect paths (container environment)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Read model_path and model_name from config.yaml (same as other scripts)
CONFIG_FILE="${CONFIG_FILE:-${WORKSPACE_DIR}/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_MODEL_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['model_path'])")
CONFIG_MODEL_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['model_name'])")

# CLI --model overrides config; otherwise use config.yaml
MODEL_NAME="${MODEL_NAME:-$CONFIG_MODEL_NAME}"
FULL_MODEL_PATH="${CONFIG_MODEL_PATH}/${MODEL_NAME}"

# Output
OUTPUT_DIR="${WORKSPACE_DIR}/result/expert"
OUTPUT_FILE="${OUTPUT_DIR}/cross_layer_prediction_${MODEL_NAME}.json"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Cross-Layer Routing Prediction Verification"
echo "  (Libra/Fate Paper Reproduction)"
echo "=============================================="
echo "Model:          ${FULL_MODEL_PATH}"
echo "Dtype:          ${DTYPE}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Num prompts:    ${NUM_PROMPTS}"
echo "Output:         ${OUTPUT_FILE}"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=============================================="

# Verify model path exists
if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $FULL_MODEL_PATH"
    echo "Available models in ${MODEL_BASE_PATH}:"
    ls "$MODEL_BASE_PATH" 2>/dev/null || echo "  (cannot list)"
    exit 1
fi

# Log file for monitoring progress
LOG_FILE="${OUTPUT_DIR}/cross_layer_prediction.log"

echo ""
echo "[$(date '+%H:%M:%S')] Installing dependencies (if missing)..."
pip install accelerate -q 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Dependencies ready."

echo ""
echo "[$(date '+%H:%M:%S')] Starting evaluation in background (to avoid SSH timeout)..."
echo "[$(date '+%H:%M:%S')] NOTE: Model loading may take 5-15 minutes for 671B models."
echo ""
echo "  Log file: $LOG_FILE"
echo "  Monitor:  tail -f $LOG_FILE"
echo ""

# Run in background with nohup to avoid SSH timeout killing the process.
# All output goes to log file. Use 'tail -f' to monitor.
nohup bash -c "
PYTHONUNBUFFERED=1 python3 '${SCRIPT_DIR}/cross_layer_routing_predictor.py' \
    --model-path '$FULL_MODEL_PATH' \
    --dtype '$DTYPE' \
    --max-new-tokens '$MAX_NEW_TOKENS' \
    --num-prompts '$NUM_PROMPTS' \
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
echo "  Monitor: tail -f $LOG_FILE"
echo "  Result:  $OUTPUT_FILE"
echo "=============================================="

