#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL="${PREDICTION_CAPTURE_PROTOCOL:-$ROOT/configs/followup_v1/prediction_capture_protocol.json}"
REQUEST_MANIFEST="$ROOT/configs/followup_v1/prediction_requests.json"
IDENTITY_MAP="$ROOT/placements/identity_service_map.json"
CURRENT_TRIGGER="$ROOT/inputs/prediction_v1/capture_trigger.json"
CAPTURE_ROOT="$ROOT/inputs/prediction_v1/captures"
WEIGHT_ROOT="$ROOT/inputs/prediction_v1/expert_weights"
TRIGGER_ROOT="$ROOT/inputs/prediction_v1/triggers"
LAYER_ROOT="$ROOT/inputs/prediction_v1/layers"
PROFILE_SUMMARY="$LAYER_ROOT/screening_profile_summary.json"
LAYER_MANIFEST="$LAYER_ROOT/selected_layers.json"
CAPTURE_INDEX="$ROOT/inputs/prediction_v1/capture_index.json"
RESULT_ROOT="$ROOT/results/prediction_v1"
RESPONSE_ROOT="$RESULT_ROOT/capture_responses"
LOG_ROOT="$RESULT_ROOT/logs"
SERVER_LOG="$LOG_ROOT/capture_server.log"
RUN_LOG="$LOG_ROOT/capture_orchestrator.log"
LAUNCHER="$ROOT/scripts/launch_prediction_capture_server.sh"
TRIGGER_MAKER="$ROOT/scripts/make_prediction_capture_trigger.py"
SENDER="$ROOT/scripts/send_prediction_batch.py"
SELECTOR="$ROOT/scripts/select_prediction_layers.py"
INDEXER="$ROOT/scripts/index_prediction_captures.py"
VERIFIER="$ROOT/scripts/verify_prediction_capture_protocol.py"
PORT="${PORT:-30000}"
API="http://127.0.0.1:${PORT}/v1/completions"
HEALTH="http://127.0.0.1:${PORT}/health"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in \
  "$CURRENT_TRIGGER" "$CAPTURE_ROOT" "$WEIGHT_ROOT" "$TRIGGER_ROOT" \
  "$LAYER_ROOT" "$CAPTURE_INDEX" "$RESULT_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to start because capture output already exists: $path" >&2
    exit 1
  fi
done

python3 "$VERIFIER" --protocol "$PROTOCOL"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [[ "${#gpu_names[@]}" -ne 8 ]]; then
  echo "expected exactly 8 visible GPUs, found ${#gpu_names[@]}" >&2
  exit 1
fi
for name in "${gpu_names[@]}"; do
  if [[ "$name" != "NVIDIA H20" ]]; then
    echo "unexpected GPU model: $name" >&2
    exit 1
  fi
done
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$compute_pids" ]]; then
  echo "GPU compute processes are already active: $compute_pids" >&2
  exit 1
fi

mkdir -p "$CAPTURE_ROOT" "$WEIGHT_ROOT" "$TRIGGER_ROOT" "$LAYER_ROOT" \
  "$RESPONSE_ROOT" "$LOG_ROOT"
exec > >(tee "$RUN_LOG") 2>&1
printf 'capture_start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'protocol_sha256=%s\n' "$(sha256sum "$PROTOCOL" | awk '{print $1}')"

PREDICTION_CAPTURE_PROTOCOL="$PROTOCOL" \
PREDICTION_CAPTURE_TRIGGER="$CURRENT_TRIGGER" \
PORT="$PORT" setsid "$LAUNCHER" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "capture_server_pid=$SERVER_PID"

ready=0
for _ in $(seq 1 2400); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "capture server exited before readiness" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if python3 - "$HEALTH" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "capture server did not become healthy within 2400 seconds" >&2
  exit 1
fi
echo "capture_server_healthy_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

request_path() {
  python3 - "$REQUEST_MANIFEST" "$1" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
items = [item for item in manifest["batches"] if item["batch_id"] == sys.argv[2]]
if len(items) != 1:
    raise SystemExit(f"unknown or duplicate batch {sys.argv[2]!r}")
print(items[0]["request_file"]["path"])
PY
}

model_path() {
  python3 - "$REQUEST_MANIFEST" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model"]["path"])
PY
}

archive_trigger() {
  local capture_id=$1
  local archive="$TRIGGER_ROOT/${capture_id}.json"
  if [[ -e "$archive" ]]; then
    echo "refusing to overwrite trigger archive: $archive" >&2
    return 1
  fi
  cp --no-clobber "$CURRENT_TRIGGER" "$archive"
  cmp --silent "$CURRENT_TRIGGER" "$archive"
  chmod a-w "$archive"
}

verify_rank_files() {
  local directory=$1
  local stem=$2
  for rank in $(seq 0 7); do
    local path="$directory/${stem}${rank}.pt"
    if [[ ! -s "$path" ]]; then
      echo "missing rank capture: $path" >&2
      return 1
    fi
  done
}

PROFILE_ID="pred-v1-screening-01-profile"
PROFILE_DIR="$CAPTURE_ROOT/screening-01/profile"
python3 "$TRIGGER_MAKER" \
  --mode profile \
  --capture-id "$PROFILE_ID" \
  --batch-id screening-01 \
  --request-manifest "$REQUEST_MANIFEST" \
  --trigger "$CURRENT_TRIGGER" \
  --output-dir "$PROFILE_DIR" \
  --identity-map "$IDENTITY_MAP"
archive_trigger "$PROFILE_ID"
python3 "$SENDER" \
  --requests "$(request_path screening-01)" \
  --request-manifest "$REQUEST_MANIFEST" \
  --batch-id screening-01 \
  --trigger "$TRIGGER_ROOT/${PROFILE_ID}.json" \
  --output "$RESPONSE_ROOT/${PROFILE_ID}.json" \
  --api "$API" \
  --model "$(model_path)"
verify_rank_files "$PROFILE_DIR" profile_rank
python3 "$SELECTOR" \
  --capture-dir "$PROFILE_DIR" \
  --request-manifest "$REQUEST_MANIFEST" \
  --output "$PROFILE_SUMMARY" \
  --layer-manifest-output "$LAYER_MANIFEST"

mapfile -t capture_matrix < <(python3 - "$REQUEST_MANIFEST" "$LAYER_MANIFEST" <<'PY'
import json, pathlib, sys
requests = json.loads(pathlib.Path(sys.argv[1]).read_text())
layers = json.loads(pathlib.Path(sys.argv[2]).read_text())["selected_layers"]
for batch in requests["batches"]:
    if batch["split"] == "screening":
        continue
    for layer in layers:
        if batch["split"] == "calibration" and not layer["calibration"]:
            continue
        role = (
            "bridge" if layer["bridge"]
            else "calibration" if layer["calibration"]
            else "held_out"
        )
        print(f'{batch["batch_id"]}\t{batch["split"]}\t{layer["layer_id"]}\t{role}')
PY
)
if [[ "${#capture_matrix[@]}" -ne 18 ]]; then
  echo "expected 18 snapshot capture units, found ${#capture_matrix[@]}" >&2
  exit 1
fi

for row in "${capture_matrix[@]}"; do
  IFS=$'\t' read -r batch_id split layer_id role <<<"$row"
  printf -v layer_padded '%03d' "$layer_id"
  capture_id="pred-v1-${batch_id}-layer-${layer_padded}"
  output_dir="$CAPTURE_ROOT/$batch_id/layer-$layer_padded"
  echo "capture_unit_start id=$capture_id split=$split role=$role utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 "$TRIGGER_MAKER" \
    --mode snapshot_split \
    --capture-id "$capture_id" \
    --batch-id "$batch_id" \
    --request-manifest "$REQUEST_MANIFEST" \
    --trigger "$CURRENT_TRIGGER" \
    --output-dir "$output_dir" \
    --weight-output-dir "$WEIGHT_ROOT" \
    --identity-map "$IDENTITY_MAP" \
    --target-layer "$layer_id" \
    --layer-role "$role" \
    --layer-manifest "$LAYER_MANIFEST"
  archive_trigger "$capture_id"
  python3 "$SENDER" \
    --requests "$(request_path "$batch_id")" \
    --request-manifest "$REQUEST_MANIFEST" \
    --batch-id "$batch_id" \
    --trigger "$TRIGGER_ROOT/${capture_id}.json" \
    --output "$RESPONSE_ROOT/${capture_id}.json" \
    --api "$API" \
    --model "$(model_path)"
  verify_rank_files "$output_dir" forward_rank
  verify_rank_files "$WEIGHT_ROOT/layer-$layer_padded" expert_rank
  echo "capture_unit_complete id=$capture_id utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

python3 "$INDEXER" \
  --capture-root "$CAPTURE_ROOT" \
  --weight-root "$WEIGHT_ROOT" \
  --response-root "$RESPONSE_ROOT" \
  --trigger-root "$TRIGGER_ROOT" \
  --request-manifest "$REQUEST_MANIFEST" \
  --layer-manifest "$LAYER_MANIFEST" \
  --profile-summary "$PROFILE_SUMMARY" \
  --capture-protocol "$PROTOCOL" \
  --output "$CAPTURE_INDEX"
printf 'capture_complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'capture_index_sha256=%s\n' "$(sha256sum "$CAPTURE_INDEX" | awk '{print $1}')"
