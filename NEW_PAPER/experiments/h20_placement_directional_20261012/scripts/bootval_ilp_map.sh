#!/bin/bash
# PRE-FREEZE boot validation for the v3 frozen-arm map (SETUP, untimed).
# Boots the exact v3 frozen-arm launcher + configuration (warmup ON, cuda
# graph ON, mem 0.8) with frozen_map_ilp.json, then serves a 500-request
# CALIBRATION subset at concurrency 256. No evaluation requests, no measured
# statistic. Logs and bench json are archived and sha-locked in the v3
# protocol's setup_artifacts. If this fails, the fallback (rotate-like block
# permutation map) is used and this script's log documents why.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
SETUP=$EXP/results/fullservice_v2/setup
DIAG=$EXP/results/fullservice_v2/diagnostics
LAUNCHER=$EXP/configs/followup_v1/fullservice_launchers_v3/launch235b_fixedh20.sh
MAP=$SETUP/frozen_map_ilp.json
SLOG=$DIAG/bootval_ilpmap_server.log
export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
log(){ echo "[$(date -Is)] BOOTVAL $*" | tee -a "$DIAG/bootval.log"; }

gpu_clear_wait(){
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$m" -lt 1000 ] && return 0
    sleep 4
  done
  return 1
}
kill_server(){
  pkill -9 -f "sglang.launch_server" >/dev/null 2>&1
  pkill -9 -f "sglang::" >/dev/null 2>&1
  sleep 5
  gpu_clear_wait
}

[ -f "$MAP" ] || { log "FATAL map missing $MAP"; exit 1; }
kill_server
log "boot $LAUNCHER $MAP"
setsid nohup bash "$LAUNCHER" "$MAP" > "$SLOG" 2>&1 &
brc=0
for i in $(seq 1 600); do
  grep -q "ready to roll" "$SLOG" 2>/dev/null && break
  if grep -q "Traceback" "$SLOG" 2>/dev/null; then brc=2; break; fi
  if [ "$i" -eq 600 ]; then brc=3; fi
  sleep 4
done
if [ "$brc" -ne 0 ]; then
  log "BOOT_FAIL rc=$brc (see $SLOG)"
  kill_server
  exit $brc
fi
v1=$(grep -c "init_expert_location from init_by_mapping" "$SLOG" || true)
log "BOOT_OK init_by_mapping=$v1 (expect 8)"
cd /workspace/EPLB/OEPLB/scripts || exit 9
timeout 1800 python3 run_grid_bench.py bootval_ilpmap "$DIAG/calib_subset500.jsonl" 256 > "$DIAG/bootval_bench.log" 2>&1
benchrc=$?
cp /workspace/EPLB/OEPLB/benchmarks/results/bootval_ilpmap.json "$DIAG/bootval_calibsub500_bench.json" 2>/dev/null
log "BENCH rc=$benchrc json=$DIAG/bootval_calibsub500_bench.json"
kill_server
if [ "$benchrc" -ne 0 ]; then exit 10; fi
python3 - <<'PYCHK'
import json, os
p = os.path.join(os.environ.get("DIAG", "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/fullservice_v2/diagnostics"), "bootval_calibsub500_bench.json")
d = json.load(open(p))
ok, err = d.get("ok"), d.get("errors")
print("BENCHVAL ok=%s errors=%s total_time_s=%s" % (ok, err, d.get("total_time_s")))
raise SystemExit(0 if (ok == 500 and err == 0) else 11)
PYCHK
log "BOOTVAL_PASS map serves 500/500 calibration requests"
exit 0
