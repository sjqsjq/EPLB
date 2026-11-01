#!/bin/bash
# ADW v2.1 r3 BEHAVIOR VALIDATION run (plan §5, user step 5) -- 5th W0 start.
# PREREQUISITE: FREEZE_record_v2p1.md written + LOCK record §8 appended with
# the frozen knobs BEFORE this script is finalized (placeholders __KNOBS__
# substituted). Runs AFTER freeze only. EA_W=16, W_init=16, alpha=0, D off.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUTROOT=$EXP/results/adw_v2p1/r3_logging
OUT=$OUTROOT/behavior
mkdir -p "$OUT"
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$EXP/inputs/fullservice_v2/workload/calibration.jsonl
DLOG=$OUTROOT/driver.log
log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }

log "=== BEHAVIOR VALIDATION RUN (frozen knobs, W live, W_init=16) ==="
sha256sum -c --quiet "$OUTROOT/deployed_shas.txt" || { log "FATAL: sha mismatch vs LOCK record -- not starting behavior run"; exit 9; }
kill_server
export OEPLB_WINDOW_CONTROLLER=21
# ---- FROZEN fitted knobs (from fit_results_v2p1.json / FREEZE_record_v2p1.md) ----
export OEPLB_V21_KAPPA=3875.1768798828125
export OEPLB_V21_Z_LO=22.643858727592757 OEPLB_V21_Z_HI=314.14010272527486 OEPLB_V21_Z_URGENT=942.4203081758246
export OEPLB_V21_THETA_R_LOW=0.010095552687949321 OEPLB_V21_THETA_R_ACT=0.011395643112507271
export OEPLB_V21_N_MIN=12320768
# ---- spec defaults (unchanged) ----
export OEPLB_V21_M_SHRINK=2 OEPLB_V21_M_GROW=3
export OEPLB_V21_DWELL_SHRINK=2 OEPLB_V21_DWELL_GROW=2
export OEPLB_V21_U_MAX=4 OEPLB_V21_W_UNCERTAIN_TARGET=16
export OEPLB_V21_URGENT_EPISODE_CAP=3 OEPLB_V21_BETA=0.5
export OEPLB_V21_LOG=$OUT/v2p1_behavior_rank0.jsonl
export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
export EA_W=16 EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
rm -f "$OEPLB_V21_LOG"
log "BEHAVIOR boot start"
setsid nohup bash "$LAUNCH" > "$OUT/server.log" 2>&1 &
rc=3
for i in $(seq 1 600); do
  grep -q "ready to roll" "$OUT/server.log" 2>/dev/null && { rc=0; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$OUT/server.log" 2>/dev/null && { rc=2; break; }
  sleep 4
done
log "BEHAVIOR boot rc=$rc after ~$((i*4))s"
if [ $rc -ne 0 ]; then tail -30 "$OUT/server.log" >> "$OUT/run.log"; kill_server; exit 1; fi
cd /workspace/EPLB/OEPLB/scripts || exit 9
timeout 3600 python3 run_grid_bench.py v21behavior "$DS" 256 > "$OUT/bench.log" 2>&1
brc=$?
log "BEHAVIOR bench rc=$brc"
cp /workspace/EPLB/OEPLB/benchmarks/results/v21behavior.json "$OUT/" 2>/dev/null
nl=$(wc -l < "$OEPLB_V21_LOG" 2>/dev/null || echo 0)
fatal=0; grep -q "fatal_collective" "$OEPLB_V21_LOG" 2>/dev/null && fatal=1
log "BEHAVIOR jsonl_lines=$nl fatal=$fatal"
kill_server
log "=== BEHAVIOR RUN DONE (brc=$brc nl=$nl fatal=$fatal) ==="
