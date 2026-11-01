#!/bin/bash
# ADW v2 Step-4 smoke #2: exercise the every-16-window cross-rank decision-hash
# all_reduce under ACTUAL W movement. SMOKE-ONLY knob values (kappa=5255 from
# smoke#1 s_change median / 4; theta_h straddling smoke#1 h_w range) - these
# are mechanism-verification placeholders, NOT calibration values. w_init=16
# (grid middle) so grow AND shrink are reachable; 4000 requests -> ~25 windows.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/adw_v2/smoke_step4
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$OUT/smoke4000.jsonl
CLOG=$OUT/smoke.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }
kill_server
export OEPLB_WINDOW_CONTROLLER=2
export OEPLB_V2_KAPPA=5255.0 OEPLB_V2_Z_LO=2.0 OEPLB_V2_Z_HI=4.0 OEPLB_V2_Z_URGENT=8.0
export OEPLB_V2_M_SHRINK=2 OEPLB_V2_M_GROW=3 OEPLB_V2_N_MIN=1000 OEPLB_V2_U_MAX=4
export OEPLB_V2_THETA_H_LOW=500000 OEPLB_V2_THETA_H_ACT=100000
export OEPLB_V2_DWELL_SHRINK=2 OEPLB_V2_DWELL_GROW=2 OEPLB_V2_W_UNCERTAIN_TARGET=16
export OEPLB_V2_URGENT_EPISODE_CAP=3 OEPLB_V2_BETA=0.5
export OEPLB_V2_LOG=$OUT/v2smokeC_rank0.jsonl
export EA_W=16 EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
log "SMOKE_C boot (v2 hash-check-under-movement; w_init=16; SMOKE-ONLY knobs)"
setsid nohup bash "$LAUNCH" > "$OUT/smokeC_server.log" 2>&1 &
rc=3
for i in $(seq 1 600); do
  grep -q "ready to roll" "$OUT/smokeC_server.log" 2>/dev/null && { rc=0; break; }
  grep -qE "Traceback|CUDA out of memory" "$OUT/smokeC_server.log" 2>/dev/null && { rc=2; break; }
  sleep 4
done
log "SMOKE_C boot rc=$rc"
if [ $rc -eq 0 ]; then
  cd /workspace/EPLB/OEPLB/scripts || exit 9
  timeout 1800 python3 run_grid_bench.py v2smokeC "$DS" 256 > "$OUT/v2smokeC.bench.log" 2>&1
  log "SMOKE_C bench rc=$?"
  cp /workspace/EPLB/OEPLB/benchmarks/results/v2smokeC.json "$OUT/" 2>/dev/null
fi
kill_server
log "SMOKE_C_DONE"
