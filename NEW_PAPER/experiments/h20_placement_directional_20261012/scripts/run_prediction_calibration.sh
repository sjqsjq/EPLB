#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
PROTOCOL="${PREDICTION_CALIBRATION_PROTOCOL:-$ROOT/configs/followup_v1/prediction_calibration_protocol.json}"
RUNNER="$ROOT/scripts/prediction_replay.py"
VERIFY="$ROOT/scripts/verify_prediction_timing_protocol.py"
INDEXER="$ROOT/scripts/index_prediction_timing_results.py"
ENV_FILE=/workspace/logs/env_235b.sh

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "$PROTOCOL" ]] || fail "missing frozen calibration protocol: $PROTOCOL"
[[ -f "$ENV_FILE" ]] || fail "missing frozen runtime environment: $ENV_FILE"
python3 "$VERIFY" --protocol "$PROTOCOL" --split calibration

git -C "$REPO" diff --check -- "$ROOT"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || fail "expected eight GPUs, found ${#GPU_NAMES[@]}"
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == "NVIDIA H20" ]] || fail "expected NVIDIA H20, found $name"
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  fail "GPU compute processes are active before calibration"
fi

RESULT_INDEX_REL="$(python3 - "$PROTOCOL" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['result_index_relative_path'])
PY
)"
RESULT_INDEX="$ROOT/$RESULT_INDEX_REL"
[[ ! -e "$RESULT_INDEX" && ! -e "$RESULT_INDEX.tmp" ]] || fail "result index already exists: $RESULT_INDEX"

python3 - "$ROOT" "$PROTOCOL" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); protocol=json.load(open(sys.argv[2]))
for start in protocol['starts']:
    for key in ('output_relative_path','log_relative_path','failure_relative_path'):
        path=root/start[key]
        if path.exists():
            raise SystemExit(f"refusing pre-existing campaign artifact: {path}")
PY

BASE="$ROOT/results/prediction_v1/calibration"
mkdir -p "$BASE/environment/preflight"
git -C "$REPO" rev-parse HEAD > "$BASE/environment/preflight/git_head.txt"
git -C "$REPO" status --short > "$BASE/environment/preflight/git_status.txt"
git -C "$REPO" diff --binary -- "$ROOT" > "$BASE/environment/preflight/experiment_diff.patch"
nvidia-smi topo -m > "$BASE/environment/preflight/nvidia_topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$BASE/environment/preflight/nvidia_query.csv"
uname -a > "$BASE/environment/preflight/uname.txt"
python3 - <<'PY' > "$BASE/environment/preflight/software.txt"
import platform, torch
import deep_ep, deep_gemm, sglang
print('python', platform.python_version())
print('torch', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('sglang', sglang.__version__)
print('deep_ep', deep_ep.__file__)
print('deep_gemm', deep_gemm.__file__)
PY

# Same frozen environment as the capture and confirmatory runs.
# shellcheck disable=SC1091
source "$ENV_FILE"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
ulimit -c 0

mapfile -t START_ROWS < <(python3 - "$PROTOCOL" <<'PY'
import json,sys
for start in json.load(open(sys.argv[1]))['starts']:
    print('\t'.join(str(start[key]) for key in (
        'start_id','output_relative_path','log_relative_path','failure_relative_path'
    )))
PY
)
EXPECTED_STARTS="$(python3 - "$PROTOCOL" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p['design']['capture_units']*p['design']['starts_per_capture'])
PY
)"
[[ "${#START_ROWS[@]}" -eq "$EXPECTED_STARTS" ]] || fail "protocol start count differs"

for row in "${START_ROWS[@]}"; do
  IFS=$'\t' read -r START_ID OUTPUT_REL LOG_REL FAILURE_REL <<< "$row"
  OUTPUT="$ROOT/$OUTPUT_REL"
  LOG="$ROOT/$LOG_REL"
  FAILURE="$ROOT/$FAILURE_REL"
  META="$BASE/environment/$START_ID"
  [[ ! -e "$OUTPUT" && ! -e "$LOG" && ! -e "$FAILURE" ]] || fail "start artifact already exists: $START_ID"
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
    fail "GPU compute process appeared before $START_ID"
  fi
  mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")" "$(dirname "$FAILURE")" "$META"
  nvidia-smi --query-gpu=index,name,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$META/gpu_before.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$META/compute_processes_before.csv" 2>&1 || true
  printf '%q ' torchrun --standalone --nproc-per-node=8 "$RUNNER" \
    --root "$ROOT" --protocol "$PROTOCOL" --start-id "$START_ID" --output "$OUTPUT" > "$META/command.txt"
  printf '\n' >> "$META/command.txt"
  date --utc --iso-8601=ns > "$META/launched_at_utc.txt"

  set +e
  torchrun --standalone --nproc-per-node=8 "$RUNNER" \
    --root "$ROOT" \
    --protocol "$PROTOCOL" \
    --start-id "$START_ID" \
    --output "$OUTPUT" 2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}
  set -e

  date --utc --iso-8601=ns > "$META/finished_at_utc.txt"
  nvidia-smi --query-gpu=index,name,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$META/gpu_after.csv"
  if [[ "$STATUS" -ne 0 ]]; then
    python3 - "$FAILURE" "$START_ID" "$STATUS" "$LOG" <<'PY'
import datetime,json,pathlib,sys
path,start_id,status,log=sys.argv[1:]
payload={
 'schema_version':1,
 'start_id':start_id,
 'exit_status':int(status),
 'log_path':log,
 'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'performance_values_not_used_for_invalidation':True,
 'retry_under_same_frozen_campaign_prohibited':True,
}
pathlib.Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
    fail "calibration start failed and was retained: $START_ID (exit $STATUS)"
  fi
  [[ -s "$OUTPUT" ]] || fail "start completed without raw output: $START_ID"
  python3 - "$OUTPUT" "$START_ID" <<'PY'
import json,sys
raw=json.load(open(sys.argv[1])); start_id=sys.argv[2]
assert raw['start_id']==start_id and raw['split']=='calibration'
assert len(raw['primary_timing_records'])==40
assert len(raw['diagnostic_timing_records'])==40
assert len(raw['correctness']['records'])==80
PY
  printf 'completed %s: %s\n' "$START_ID" "$OUTPUT"
done

python3 "$INDEXER" --root "$ROOT" --protocol "$PROTOCOL" --output "$RESULT_INDEX"
printf 'calibration campaign complete: %s\n' "$RESULT_INDEX"
