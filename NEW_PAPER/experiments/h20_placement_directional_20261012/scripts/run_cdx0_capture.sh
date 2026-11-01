#!/usr/bin/env bash
# CD-X0 capture orchestrator (SETUP phase; produces the 16 immutable capture units).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/configs/followup_v1/cdx0_requests.json"
IDENTITY_MAP="$ROOT/placements/identity_service_map.json"
CURRENT_TRIGGER="$ROOT/inputs/cdx0_v1/capture_trigger.json"
CAPTURE_ROOT="$ROOT/inputs/cdx0_v1/captures"
WEIGHT_ROOT="$ROOT/inputs/cdx0_v1/expert_weights"
TRIGGER_ROOT="$ROOT/inputs/cdx0_v1/triggers"
FAILURES_ROOT="$ROOT/inputs/cdx0_v1/capture_failures"
RESPONSE_ROOT="$ROOT/results/cdx0_v1/capture_responses"
LOG_ROOT="$ROOT/results/cdx0_v1/logs"
SERVER_LOG="$LOG_ROOT/capture_server.log"
RUN_LOG="$LOG_ROOT/capture_orchestrator.log"
CAPTURE_INDEX="$ROOT/inputs/cdx0_v1/capture_index.json"
LAUNCHER="$ROOT/scripts/launch_cdx0_capture_server.sh"
TRIGGER_MAKER="$ROOT/scripts/make_cdx0_trigger.py"
SENDER="$ROOT/scripts/send_cdx0_batch.py"
INDEXER="$ROOT/scripts/cdx0_index_captures.py"
PORT="${PORT:-30000}"
API="http://127.0.0.1:${PORT}/v1/completions"
HEALTH="http://127.0.0.1:${PORT}/health"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 -- "-$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$CURRENT_TRIGGER" "$CAPTURE_ROOT" "$WEIGHT_ROOT" "$TRIGGER_ROOT" "$CAPTURE_INDEX" "$RESPONSE_ROOT"; do
  if [[ -e "$path" ]]; then echo "refusing to start: output already exists: $path" >&2; exit 1; fi
done
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "expected 8 GPUs, found ${#gpu_names[@]}" >&2; exit 1; }
for name in "${gpu_names[@]}"; do [[ "$name" == "NVIDIA H20" ]] || { echo "unexpected GPU: $name" >&2; exit 1; }; done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" ]]; then
  echo "GPU compute processes already active" >&2; exit 1
fi
mkdir -p "$CAPTURE_ROOT" "$WEIGHT_ROOT" "$TRIGGER_ROOT" "$FAILURES_ROOT" "$RESPONSE_ROOT" "$LOG_ROOT"
exec > >(tee "$RUN_LOG") 2>&1
printf 'cdx0_capture_start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'manifest_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"

CDX0_CAPTURE_TRIGGER="$CURRENT_TRIGGER" PORT="$PORT" setsid "$LAUNCHER" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "capture_server_pid=$SERVER_PID"
ready=0
for _ in $(seq 1 2400); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited before readiness" >&2; tail -200 "$SERVER_LOG" >&2 || true; exit 1; fi
  if python3 - "$HEALTH" <<'PY' >/dev/null 2>&1
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as r:
    if r.status != 200: raise SystemExit(1)
PY
  then ready=1; break; fi
  sleep 1
done
[[ "$ready" -eq 1 ]] || { echo "server not healthy within 2400s" >&2; exit 1; }
echo "capture_server_healthy_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

request_path() {
  python3 - "$MANIFEST" "$1" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
items = [b for b in m["batches"] if b["batch_id"] == sys.argv[2]]
if len(items) != 1: raise SystemExit(f"unknown batch {sys.argv[2]!r}")
print(items[0]["request_file"]["path"])
PY
}
MODEL_PATH="/root/models/ms_cache/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master"

archive_trigger() {
  local cid=$1 archive="$TRIGGER_ROOT/$1.json"
  [[ -e "$archive" ]] && { echo "refusing to overwrite trigger archive: $archive" >&2; return 1; }
  cp --no-clobber "$CURRENT_TRIGGER" "$archive"
  cmp --silent "$CURRENT_TRIGGER" "$archive"
  chmod a-w "$archive"
}
verify_rank_files() {
  local dir=$1 stem=$2
  for rank in $(seq 0 7); do
    [[ -s "$dir/${stem}${rank}.pt" ]] || { echo "missing rank file: $dir/${stem}${rank}.pt" >&2; return 1; }
  done
}

capture_unit() { # $1 batch_id $2 split $3 domain $4 layer $5 attempt(1|2)
  local batch_id=$1 split=$2 domain=$3 layer=$4 attempt=$5
  local cid="cdx0-${split}-${domain}-layer-$(printf '%03d' "$layer")"
  [[ "$attempt" -eq 2 ]] && cid="${cid}-r2"
  local suffix=""; [[ "$attempt" -eq 2 ]] && suffix="-r2"
  local outdir="$CAPTURE_ROOT/$batch_id/layer-$(printf '%03d' "$layer")$suffix"
  echo "capture_unit_start id=$cid attempt=$attempt utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 "$TRIGGER_MAKER" \
    --capture-id "$cid" --batch-id "$batch_id" \
    --request-manifest "$MANIFEST" --trigger "$CURRENT_TRIGGER" \
    --output-dir "$outdir" --weight-output-dir "$WEIGHT_ROOT" \
    --identity-map "$IDENTITY_MAP" --target-layer "$layer" --min-hidden-rows 400
  archive_trigger "$cid"
  python3 "$SENDER" \
    --requests "$(request_path "$batch_id")" --request-manifest "$MANIFEST" \
    --batch-id "$batch_id" --trigger "$TRIGGER_ROOT/${cid}.json" \
    --output "$RESPONSE_ROOT/${cid}.json" --api "$API" --model "$MODEL_PATH"
  verify_rank_files "$outdir" forward_rank
  verify_rank_files "$WEIGHT_ROOT/layer-$(printf '%03d' "$layer")" expert_rank
  echo "capture_unit_complete id=$cid utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

mapfile -t matrix < <(python3 - "$MANIFEST" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
for b in m["batches"]:
    for layer in (24, 71):
        print(f'{b["batch_id"]}\t{b["split"]}\t{b["domain"]}\t{layer}')
PY
)
[[ "${#matrix[@]}" -eq 16 ]] || { echo "expected 16 units, found ${#matrix[@]}" >&2; exit 1; }

for row in "${matrix[@]}"; do
  IFS=$'\t' read -r batch_id split domain layer <<<"$row"
  if capture_unit "$batch_id" "$split" "$domain" "$layer" 1; then
    continue
  fi
  echo "capture_unit_failed id=cdx0-${split}-${domain}-layer-${layer} attempt=1; retiring and re-arming once (-r2)"
  mkdir -p "$FAILURES_ROOT/$batch_id"
  mv "$CAPTURE_ROOT/$batch_id/layer-$(printf '%03d' "$layer")" "$FAILURES_ROOT/$batch_id/" 2>/dev/null || true
  cp "$RESPONSE_ROOT/cdx0-${split}-${domain}-layer-$(printf '%03d' "$layer").json" "$FAILURES_ROOT/$batch_id/" 2>/dev/null || true
  rm -f "$RESPONSE_ROOT/cdx0-${split}-${domain}-layer-$(printf '%03d' "$layer").json"
  capture_unit "$batch_id" "$split" "$domain" "$layer" 2
done

IDENTITY_SHA="$(sha256sum "$IDENTITY_MAP" | awk '{print $1}')"
python3 "$INDEXER" --identity-map-sha256 "$IDENTITY_SHA" --server-log "$SERVER_LOG"
printf 'cdx0_capture_complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'capture_index_sha256=%s\n' "$(sha256sum "$CAPTURE_INDEX" | awk '{print $1}')"
