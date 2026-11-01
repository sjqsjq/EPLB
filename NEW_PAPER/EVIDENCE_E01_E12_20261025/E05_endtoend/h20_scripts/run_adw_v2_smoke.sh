#!/bin/bash
# ADW v2 Step-4 verification smokes (NOT result-producing runs; smoke-only knobs).
# Smoke A (V3): v1 legacy path, OEPLB_WINDOW_CONTROLLER unset (default 1),
#   A10-style flags -> legacy [PB-OEPLB-ADW]/[PB-OEPLB-WINDOW] mutations must
#   appear; zero [PB-OEPLB-ADW2] lines.
# Smoke B (V2/V4/V5): OEPLB_WINDOW_CONTROLLER=2, w_init=8 (grid member; chosen
#   so >=16 windows fit the smoke budget and the every-16-window cross-rank
#   decision-hash all_reduce fires), alpha=0.0, adaptive flags OFF,
#   SMOKE-ONLY knob placeholders (kappa=1.0 etc.) - mechanism verification
#   only; Step-5 calibration freezes the real values.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/adw_v2/smoke_step4
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
DS=$OUT/smoke1200.jsonl
CLOG=$OUT/smoke.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }
boot(){ # $1 logfile ; env exported by caller
  setsid nohup bash "$LAUNCH" > "$1" 2>&1 &
  for i in $(seq 1 600); do
    grep -q "ready to roll" "$1" 2>/dev/null && return 0
    grep -qE "Traceback|CUDA out of memory" "$1" 2>/dev/null && return 2
    sleep 4
  done
  return 3
}
bench(){ # $1 label
  cd /workspace/EPLB/OEPLB/scripts || return 9
  timeout 1800 python3 run_grid_bench.py "$1" "$DS" 256 > "$OUT/$1.bench.log" 2>&1
  local rc=$?
  cp "/workspace/EPLB/OEPLB/benchmarks/results/$1.json" "$OUT/" 2>/dev/null
  cd "$EXP/scripts" || true
  return $rc
}

# ---------- Smoke A: v1 identity (default version=1) ----------
kill_server
unset OEPLB_WINDOW_CONTROLLER OEPLB_V2_KAPPA OEPLB_V2_LOG
export EA_W=16 EA_ALPHA=0.0 EA_ADAPTIVE_WINDOW=1
log "SMOKE_A boot (v1 default; W=16 alpha=0.0 ADW=1)"
boot "$OUT/smokeA_server.log"; rc=$?
log "SMOKE_A boot rc=$rc"
if [ $rc -eq 0 ]; then
  bench v2smokeA; log "SMOKE_A bench rc=$?"
fi
kill_server

# ---------- Smoke B: v2 single authority ----------
export OEPLB_WINDOW_CONTROLLER=2
export OEPLB_V2_KAPPA=1.0 OEPLB_V2_Z_LO=2.0 OEPLB_V2_Z_HI=4.0 OEPLB_V2_Z_URGENT=8.0
export OEPLB_V2_M_SHRINK=2 OEPLB_V2_M_GROW=3 OEPLB_V2_N_MIN=1000 OEPLB_V2_U_MAX=4
export OEPLB_V2_THETA_H_LOW=500 OEPLB_V2_THETA_H_ACT=2000
export OEPLB_V2_DWELL_SHRINK=2 OEPLB_V2_DWELL_GROW=2 OEPLB_V2_W_UNCERTAIN_TARGET=8
export OEPLB_V2_URGENT_EPISODE_CAP=3 OEPLB_V2_BETA=0.5
export OEPLB_V2_LOG=$OUT/v2smokeB_rank0.jsonl
export EA_W=8 EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
log "SMOKE_B boot (v2; w_init=8 alpha=0.0 adaptive flags OFF; SMOKE-ONLY knobs)"
boot "$OUT/smokeB_server.log"; rc=$?
log "SMOKE_B boot rc=$rc"
if [ $rc -eq 0 ]; then
  bench v2smokeB; log "SMOKE_B bench rc=$?"
fi
kill_server
log "SMOKES_DONE"
