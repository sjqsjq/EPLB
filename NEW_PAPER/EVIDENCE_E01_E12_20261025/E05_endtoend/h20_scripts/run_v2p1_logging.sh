#!/bin/bash
# ADW v2.1 r3 calibration LOGGING campaign driver (pre-registered plan §1).
# 4 fixed-window runs, order W=8/16/32/64, each = one W0 pass (8037 reqs).
# Inert knobs => W fixed by construction. Failure policy (plan §1.2):
# one pre-registered restart per slot; second fatal failure => STOP (exit 9).
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUTROOT=$EXP/results/adw_v2p1/r3_logging
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$EXP/inputs/fullservice_v2/workload/calibration.jsonl
DLOG=$OUTROOT/driver.log
log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }

run_slot(){
  W=$1; ATTEMPT=$2
  OUT=$OUTROOT/W$W
  mkdir -p "$OUT"
  CLOG=$OUT/run.log
  log "=== SLOT W=$W attempt $ATTEMPT ==="
  # integrity: deployed code must match the locked shas before every boot
  sha256sum -c --quiet "$OUTROOT/deployed_shas.txt" 2>>"$CLOG" || { log "FATAL: sha mismatch vs LOCK record -- stopping"; exit 9; }
  kill_server
  export OEPLB_WINDOW_CONTROLLER=21
  export OEPLB_V21_KAPPA=1.0
  export OEPLB_V21_Z_LO=1e12 OEPLB_V21_Z_HI=2e12 OEPLB_V21_Z_URGENT=3e12
  export OEPLB_V21_THETA_R_LOW=1e-12 OEPLB_V21_THETA_R_ACT=2.0
  export OEPLB_V21_N_MIN=0
  export OEPLB_V21_M_SHRINK=2 OEPLB_V21_M_GROW=3
  export OEPLB_V21_DWELL_SHRINK=2 OEPLB_V21_DWELL_GROW=2
  export OEPLB_V21_U_MAX=4 OEPLB_V21_W_UNCERTAIN_TARGET=16
  export OEPLB_V21_URGENT_EPISODE_CAP=3 OEPLB_V21_BETA=0.5
  export OEPLB_V21_LOG=$OUT/v2p1_rank0.jsonl
  export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
  export EA_W=$W EA_ALPHA=0.0
  unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
  SLOG=$OUT/server_a${ATTEMPT}.log
  rm -f "$OEPLB_V21_LOG"
  log "W=$W boot start (attempt $ATTEMPT)"
  setsid nohup bash "$LAUNCH" > "$SLOG" 2>&1 &
  rc=3
  for i in $(seq 1 600); do
    grep -q "ready to roll" "$SLOG" 2>/dev/null && { rc=0; break; }
    grep -qE "Traceback|CUDA out of memory|RuntimeError" "$SLOG" 2>/dev/null && { rc=2; break; }
    sleep 4
  done
  log "W=$W boot rc=$rc after ~$((i*4))s"
  if [ $rc -ne 0 ]; then tail -30 "$SLOG" >> "$CLOG"; kill_server; return 1; fi
  cd /workspace/EPLB/OEPLB/scripts || { kill_server; return 1; }
  timeout 3600 python3 run_grid_bench.py "v21calib_W${W}" "$DS" 256 > "$OUT/bench_a${ATTEMPT}.log" 2>&1
  brc=$?
  log "W=$W bench rc=$brc"
  cp "/workspace/EPLB/OEPLB/benchmarks/results/v21calib_W${W}.json" "$OUT/bench_a${ATTEMPT}.json" 2>/dev/null
  fatal=0
  grep -q "fatal_collective" "$OEPLB_V21_LOG" 2>/dev/null && fatal=1
  grep -qE "FATAL: det SUM|FATAL: confirm" "$SLOG" 2>/dev/null && fatal=1
  nl=$(wc -l < "$OEPLB_V21_LOG" 2>/dev/null || echo 0)
  log "W=$W jsonl_lines=$nl fatal_collective=$fatal"
  kill_server
  if [ $brc -eq 0 ] && [ "$nl" -gt 0 ] && [ $fatal -eq 0 ]; then
    log "=== SLOT W=$W attempt $ATTEMPT OK ==="
    return 0
  fi
  log "=== SLOT W=$W attempt $ATTEMPT FAILED (brc=$brc nl=$nl fatal=$fatal) ==="
  return 1
}

log "CAMPAIGN START: v2.1 r3 calibration logging (4 slots, order 8/16/32/64)"
for W in 8 16 32 64; do
  run_slot $W 1 && continue
  log "SLOT W=$W failed attempt 1 -> ONE pre-registered restart (plan §1.2)"
  run_slot $W 2 && continue
  log "CAMPAIGN STOP: second failure at slot W=$W (plan §1.2: stop, no further starts)"
  exit 9
done
log "CAMPAIGN DONE: all 4 logging runs completed"
