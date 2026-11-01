#!/bin/bash
# ADW v2 Step-5 §4 BEHAVIOR VALIDATION RUN (pre-registered plan).
# W0=calibration.jsonl 8037 reqs, w_init=16, W LIVE, FROZEN-CANDIDATE knobs
# from fit_results_v2.json (kappa=2806.9 z_lo=3.737 z_hi=19.864
# z_urgent=115.619 theta_h_low=988725 theta_h_act=1065054 n_min=12320768,
# spec defaults for m/dwell/u_max/beta/cap). Rules freeze AFTER this run
# validates behavior+cost; defects are documented, not silently re-tuned.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/adw_v2/calib_step5
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$EXP/inputs/fullservice_v2/workload/calibration.jsonl
CLOG=$OUT/behavior.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }
kill_server
export OEPLB_WINDOW_CONTROLLER=2
export OEPLB_V2_KAPPA=2806.9 OEPLB_V2_Z_LO=3.737 OEPLB_V2_Z_HI=19.864 OEPLB_V2_Z_URGENT=115.619
export OEPLB_V2_M_SHRINK=2 OEPLB_V2_M_GROW=3 OEPLB_V2_N_MIN=12320768 OEPLB_V2_U_MAX=4
export OEPLB_V2_THETA_H_LOW=988725 OEPLB_V2_THETA_H_ACT=1065054
export OEPLB_V2_DWELL_SHRINK=2 OEPLB_V2_DWELL_GROW=2 OEPLB_V2_W_UNCERTAIN_TARGET=16
export OEPLB_V2_URGENT_EPISODE_CAP=3 OEPLB_V2_BETA=0.5
export OEPLB_V2_LOG=$OUT/v2behavior_rank0.jsonl
export EA_W=16 EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
log "BEHAVIOR boot (frozen-candidate knobs, W live)"
setsid nohup bash "$LAUNCH" > "$OUT/behavior_server.log" 2>&1 &
rc=3
for i in $(seq 1 600); do
  grep -q "ready to roll" "$OUT/behavior_server.log" 2>/dev/null && { rc=0; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$OUT/behavior_server.log" 2>/dev/null && { rc=2; break; }
  sleep 4
done
log "BEHAVIOR boot rc=$rc"
if [ $rc -ne 0 ]; then tail -30 "$OUT/behavior_server.log" >> "$CLOG"; kill_server; exit 1; fi
cd /workspace/EPLB/OEPLB/scripts || exit 9
timeout 3600 python3 run_grid_bench.py v2behavior "$DS" 256 > "$OUT/v2behavior.bench.log" 2>&1
log "BEHAVIOR bench rc=$?"
cp /workspace/EPLB/OEPLB/benchmarks/results/v2behavior.json "$OUT/" 2>/dev/null
kill_server
log "BEHAVIOR_DONE"
