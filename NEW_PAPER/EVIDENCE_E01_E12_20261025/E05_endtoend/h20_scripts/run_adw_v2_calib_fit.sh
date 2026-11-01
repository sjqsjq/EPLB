#!/bin/bash
# ADW v2 Step-5 calibration FITTING RUN (pre-registered plan §1).
# W0=calibration.jsonl 8037 requests, W_init=16, alpha=0.0, adaptive OFF.
# FIT-RUN-ONLY knobs: kappa=1.0 (=> logged z == s_change*n_layer),
# z thresholds 1e12/2e12/3e12 so NO shrink/grow can ever trigger (W stays 16
# by construction -> fixed-W observation, no feedback), n_min=0 (uncertain
# path off; every window observable). These are NOT the frozen v2 knobs;
# frozen values come from the §3 fitting rules applied to this run's logs.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/adw_v2/calib_step5
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$EXP/inputs/fullservice_v2/workload/calibration.jsonl
CLOG=$OUT/calib.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }
kill_server
export OEPLB_WINDOW_CONTROLLER=2
export OEPLB_V2_KAPPA=1.0 OEPLB_V2_Z_LO=1e12 OEPLB_V2_Z_HI=2e12 OEPLB_V2_Z_URGENT=3e12
export OEPLB_V2_M_SHRINK=2 OEPLB_V2_M_GROW=3 OEPLB_V2_N_MIN=0 OEPLB_V2_U_MAX=4
export OEPLB_V2_THETA_H_LOW=0 OEPLB_V2_THETA_H_ACT=1e18
export OEPLB_V2_DWELL_SHRINK=2 OEPLB_V2_DWELL_GROW=2 OEPLB_V2_W_UNCERTAIN_TARGET=16
export OEPLB_V2_URGENT_EPISODE_CAP=3 OEPLB_V2_BETA=0.5
export OEPLB_V2_LOG=$OUT/v2calib_rank0.jsonl
export EA_W=16 EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
log "CALIB_FIT boot (W locked 16 by inert thresholds; kappa=1.0 observation; theta_h_low=0 blocks grow)"
setsid nohup bash "$LAUNCH" > "$OUT/calib_server.log" 2>&1 &
rc=3
for i in $(seq 1 600); do
  grep -q "ready to roll" "$OUT/calib_server.log" 2>/dev/null && { rc=0; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$OUT/calib_server.log" 2>/dev/null && { rc=2; break; }
  sleep 4
done
log "CALIB_FIT boot rc=$rc"
if [ $rc -ne 0 ]; then tail -30 "$OUT/calib_server.log" >> "$CLOG"; kill_server; exit 1; fi
cd /workspace/EPLB/OEPLB/scripts || exit 9
timeout 3600 python3 run_grid_bench.py v2calib_fit "$DS" 256 > "$OUT/v2calib_fit.bench.log" 2>&1
log "CALIB_FIT bench rc=$?"
cp /workspace/EPLB/OEPLB/benchmarks/results/v2calib_fit.json "$OUT/" 2>/dev/null
kill_server
log "CALIB_FIT_DONE"
