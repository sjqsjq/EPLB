#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
LAUNCHER="${BASH_SOURCE[0]}"
PROTOCOL="$ROOT/configs/followup_v1/protocol.json"
RUNNER="$ROOT/scripts/replay_placement.py"
ANALYZER="$ROOT/scripts/analyze_confirmation.py"
MANIFEST="$ROOT/placements/manifest.json"
TOLERANCE="$ROOT/inputs/correctness_tolerance.json"
ENV_FILE=/workspace/logs/env_235b.sh
RESULT_ROOT="$ROOT/results/confirmatory_v1"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "$PROTOCOL" ]] || fail "missing frozen protocol: $PROTOCOL"
[[ -f "$ENV_FILE" ]] || fail "missing environment file: $ENV_FILE"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail "repository must be clean before confirmation"

python3 - "$ROOT" "$PROTOCOL" "$RUNNER" "$ANALYZER" "$LAUNCHER" "$MANIFEST" "$TOLERANCE" <<'PY'
import hashlib, json, pathlib, sys
root, protocol_path, runner, analyzer, launcher, manifest, tolerance = map(pathlib.Path, sys.argv[1:])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

protocol = json.loads(protocol_path.read_text())
assert protocol["status"] == "frozen_before_confirmation_timing"
checks = {
    runner: protocol["implementation"]["replay_placement_sha256"],
    analyzer: protocol["implementation"]["analyze_confirmation_sha256"],
    launcher: protocol["implementation"]["run_confirmation_sha256"],
    manifest: protocol["confirmation"]["placement_manifest_sha256"],
    tolerance: protocol["confirmation"]["tolerance_file_sha256"],
}
for path, expected in checks.items():
    actual = sha(path)
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")
for item in protocol["confirmation"]["snapshot_files"]:
    path = pathlib.Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise SystemExit(f"snapshot mismatch for rank {item['rank']}: {path}")
PY

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || fail "expected eight GPUs, found ${#GPU_NAMES[@]}"
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == "NVIDIA H20" ]] || fail "expected NVIDIA H20, found $name"
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  fail "GPU compute processes are active before confirmation"
fi

mkdir -p "$RESULT_ROOT/environment" "$RESULT_ROOT/logs" "$RESULT_ROOT/raw"
BASE_METADATA="$RESULT_ROOT/environment/preflight"
mkdir -p "$BASE_METADATA"
git -C "$REPO" rev-parse HEAD > "$BASE_METADATA/git_head.txt"
git -C "$REPO" status --short > "$BASE_METADATA/git_status.txt"
git -C "$REPO" diff --binary > "$BASE_METADATA/git_diff.patch"
nvidia-smi -q -x > "$BASE_METADATA/nvidia_smi.xml"
nvidia-smi topo -m > "$BASE_METADATA/nvidia_topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$BASE_METADATA/nvidia_query.csv"
uname -a > "$BASE_METADATA/uname.txt"
python3 - <<'PY' > "$BASE_METADATA/software.txt"
import platform, torch
import deep_ep, deep_gemm, sglang
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("sglang", sglang.__version__)
print("deep_ep", deep_ep.__file__)
print("deep_gemm", deep_gemm.__file__)
PY

# Shellcheck cannot follow a runtime path; this is the same frozen environment used by E1.
# shellcheck disable=SC1091
source "$ENV_FILE"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
ulimit -c 0

mapfile -t START_IDS < <(python3 - "$PROTOCOL" <<'PY'
import json, sys
for start in json.load(open(sys.argv[1]))["confirmation"]["starts"]:
    print(start["start_id"])
PY
)
[[ "${#START_IDS[@]}" -eq 4 ]] || fail "frozen protocol does not contain four starts"

for START_ID in "${START_IDS[@]}"; do
  read -r SEED OUTPUT_REL LOG_REL < <(python3 - "$PROTOCOL" "$START_ID" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
start = next(item for item in protocol["confirmation"]["starts"] if item["start_id"] == sys.argv[2])
print(start["random_seed"], start["output_relative_path"], start["log_relative_path"])
PY
  )
  OUTPUT="$ROOT/$OUTPUT_REL"
  LOG="$ROOT/$LOG_REL"
  FAILURE="${OUTPUT%.json}.failure.json"
  META="$RESULT_ROOT/environment/$START_ID"
  [[ ! -e "$OUTPUT" && ! -e "$FAILURE" && ! -e "$LOG" ]] || fail "start output already exists: $START_ID"
  mkdir -p "$META" "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
  nvidia-smi -q -x > "$META/nvidia_smi_before.xml"
  nvidia-smi topo -m > "$META/nvidia_topology_before.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$META/compute_processes_before.csv" 2>&1 || true
  nvidia-smi --query-gpu=index,name,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$META/gpu_before.csv"
  printf '%q ' \
    torchrun --standalone --nproc-per-node=8 "$RUNNER" \
    --phase confirm --root "$ROOT" --manifest "$MANIFEST" \
    --tolerance-file "$TOLERANCE" --protocol-file "$PROTOCOL" \
    --start-id "$START_ID" --output "$OUTPUT" --warmup 10 \
    --activation-warmup 10 --blocks 5 --iterations-per-block 50 \
    --empty-fence-warmup 5 --empty-fence-repetitions 20 \
    --random-seed "$SEED" > "$META/command.txt"
  printf '\n' >> "$META/command.txt"
  date --utc --iso-8601=ns > "$META/launched_at_utc.txt"

  set +e
  torchrun --standalone --nproc-per-node=8 "$RUNNER" \
    --phase confirm \
    --root "$ROOT" \
    --manifest "$MANIFEST" \
    --tolerance-file "$TOLERANCE" \
    --protocol-file "$PROTOCOL" \
    --start-id "$START_ID" \
    --output "$OUTPUT" \
    --warmup 10 \
    --activation-warmup 10 \
    --blocks 5 \
    --iterations-per-block 50 \
    --empty-fence-warmup 5 \
    --empty-fence-repetitions 20 \
    --random-seed "$SEED" 2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}
  set -e

  date --utc --iso-8601=ns > "$META/finished_at_utc.txt"
  nvidia-smi -q -x > "$META/nvidia_smi_after.xml"
  nvidia-smi --query-gpu=index,name,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv > "$META/gpu_after.csv"
  if [[ "$STATUS" -ne 0 ]]; then
    python3 - "$FAILURE" "$START_ID" "$STATUS" "$LOG" <<'PY'
import datetime, json, pathlib, sys
path, start_id, status, log = sys.argv[1:]
payload = {
    "schema_version": 1,
    "start_id": start_id,
    "exit_status": int(status),
    "log_path": log,
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "performance_values_not_used_for_invalidation": True,
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    fail "confirmation start failed and was retained: $START_ID (exit $STATUS)"
  fi
  [[ -s "$OUTPUT" ]] || fail "start completed without result JSON: $START_ID"
  python3 - "$OUTPUT" "$START_ID" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
assert raw["phase"] == "confirm" and raw["start_id"] == sys.argv[2]
assert len(raw["timing_records"]) == 65
assert len(raw["correctness"]["records"]) == 104
PY
  printf 'completed %s: %s\n' "$START_ID" "$OUTPUT"
done

python3 "$ANALYZER" --root "$ROOT" --protocol "$PROTOCOL"
printf 'confirmation complete: %s\n' "$RESULT_ROOT/summary.json"
