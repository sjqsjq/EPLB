#!/bin/bash
# Step-6 common functions (protocol: results/adw_v2/step6_formal_protocol_E.md)
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
S6=$EXP/results/step6
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
WLD=$EXP/inputs/fullservice_v2/workload
PB=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
declare -A DS=( [W0]=$WLD/calibration.jsonl [W1]=$WLD/w1_stable_code_4000.jsonl [W2]=$WLD/w2_switch_code_chinese_8000.jsonl )
declare -A NN=( [W0]=8037 [W1]=4000 [W2]=8000 )

verify_hashes(){ # abort driver on any mismatch (protocol §1/§2)
  local ok=1
  echo "$(sha256sum $PB/controller.py)"  | grep -q 2a4dedb22d4bcaba203233c0f9a5ae6c211370f524addb7d166669406690878b || { echo "HASH FAIL controller.py"; ok=0; }
  echo "$(sha256sum $PB/window_v2.py)"    | grep -q 810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74 || { echo "HASH FAIL window_v2.py"; ok=0; }
  echo "$(sha256sum $PB/config.py)"       | grep -q 180fa41eae6e9394186afceb2ceadf9468ebbacda69591fb9b9897a2e6c2680c || { echo "HASH FAIL config.py"; ok=0; }
  echo "$(sha256sum $WLD/calibration.jsonl)" | grep -q 7d53aa0b07dbc9f11f47af44ee4f4106861f5ace6d4eeca1ba41cdf5b01b17f9 || { echo "HASH FAIL W0"; ok=0; }
  echo "$(sha256sum $WLD/w1_stable_code_4000.jsonl)" | grep -q f0391f7cdf8edd734a09c65b331d751052b8b1937a0416b3782a409039fe8578 || { echo "HASH FAIL W1"; ok=0; }
  echo "$(sha256sum $WLD/w2_switch_code_chinese_8000.jsonl)" | grep -q 9d852a756957d3db805fd876ebbad51a0224b9b6c23d23edb5646cdf260362f8 || { echo "HASH FAIL W2"; ok=0; }
  [ $ok -eq 1 ]
}
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){
  pkill -9 -f "sglang.launch_server" >/dev/null 2>&1
  pkill -9 -f "sglang::" >/dev/null 2>&1
  sleep 5
  # INFRA FIX 2026-10-16 (deviation memo step6_deviation_infra_portrace.md):
  # EADDRINUSE :30234 retired the W0_f32 boot; GPU-clear alone is not sufficient.
  # Force-release the dist port and wait until both campaign ports are actually free.
  command -v fuser >/dev/null 2>&1 && fuser -k -n tcp -9 30234 >/dev/null 2>&1
  for i in $(seq 1 45); do
    ss -tan 2>/dev/null | grep -E ":(30234|30000)\b" | grep -q LISTEN || break
    sleep 2
  done
  ss -tan 2>/dev/null | grep -E ":(30234|30000)\b" | grep -q LISTEN && echo "WARN ports still bound"
  gpu_clear_wait || echo "WARN gpus not clear"
  sleep 5
}

# run_start <cell_dir> <label> <cond> <mode:fixed|v2> <W> <logfile> <extralog>
# fixed: controller=1 EA_W=W ; v2: controller=2 frozen knobs + per-start V2_LOG
run_start(){
  local cell=$1 label=$2 cond=$3 mode=$4 W=$5 clog=$6
  local cdir=$S6/$cell; mkdir -p "$cdir"
  kill_server
  local oeplb_extra=""
  if [ "$mode" = "fixed" ]; then
    export OEPLB_WINDOW_CONTROLLER=1
    unset OEPLB_V2_LOG
  else
    export OEPLB_WINDOW_CONTROLLER=2
    export OEPLB_V2_KAPPA=2806.9 OEPLB_V2_Z_LO=3.737 OEPLB_V2_Z_HI=19.864 OEPLB_V2_Z_URGENT=115.619
    export OEPLB_V2_M_SHRINK=${ABL_M_SHRINK:-2} OEPLB_V2_M_GROW=${ABL_M_GROW:-3}
    export OEPLB_V2_N_MIN=12320768 OEPLB_V2_U_MAX=4
    export OEPLB_V2_THETA_H_LOW=${ABL_THETA_LOW:-988725} OEPLB_V2_THETA_H_ACT=${ABL_THETA_ACT:-1065054}
    export OEPLB_V2_DWELL_SHRINK=${ABL_DWELL_S:-2} OEPLB_V2_DWELL_GROW=${ABL_DWELL_G:-2}
    export OEPLB_V2_W_UNCERTAIN_TARGET=16
    export OEPLB_V2_URGENT_EPISODE_CAP=${ABL_CAP:-3} OEPLB_V2_BETA=${ABL_BETA:-0.5}
    export OEPLB_V2_LOG=$cdir/v2_rank0.jsonl
  fi
  export EA_W=$W EA_ALPHA=0.0
  unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
  echo "[$(date -Is)] $cell boot mode=$mode W=$W" | tee -a "$clog"
  setsid nohup bash "$LAUNCH" > "$cdir/server.log" 2>&1 &
  local rc=3
  for i in $(seq 1 600); do
    grep -q "ready to roll" "$cdir/server.log" 2>/dev/null && { rc=0; break; }
    grep -qE "Traceback|CUDA out of memory|RuntimeError" "$cdir/server.log" 2>/dev/null && { rc=2; break; }
    sleep 4
  done
  echo "[$(date -Is)] $cell boot rc=$rc" | tee -a "$clog"
  if [ $rc -ne 0 ]; then tail -30 "$cdir/server.log" >> "$clog"; echo "[$(date -Is)] $cell RETIRED boot_fail" | tee -a "$clog"; kill_server; return 1; fi
  ( cd /workspace/EPLB/OEPLB/scripts && timeout 3600 python3 run_grid_bench.py "$label" "${DS[$cond]}" 256 > "$cdir/bench.log" 2>&1 )
  echo "[$(date -Is)] $cell bench rc=$?" | tee -a "$clog"
  cp /workspace/EPLB/OEPLB/benchmarks/results/$label.json "$cdir/bench.json" 2>/dev/null
  local okc=$(python3 -c "import json;d=json.load(open('$cdir/bench.json'));print(d['ok'],d['errors'])" 2>/dev/null)
  echo "[$(date -Is)] $cell ok/errors: $okc (expected ${NN[$cond]} 0)" | tee -a "$clog"
  # v2-arm integrity: desync / v2 error events retire the start (protocol §6)
  if [ "$mode" = "v2" ] && [ -f "$cdir/v2_rank0.jsonl" ]; then
    if grep -qE '"desync": true|RANK DESYNC' "$cdir/v2_rank0.jsonl" "$cdir/server.log" 2>/dev/null || grep -q "v2 error" "$cdir/server.log" 2>/dev/null; then
      echo "[$(date -Is)] $cell RETIRED desync_or_v2error (artifacts preserved)" | tee -a "$clog"; kill_server; return 2
    fi
  fi
  set -- $okc
  if [ "${1:-0}" -ne "${NN[$cond]}" ] || [ "${2:-1}" -ne 0 ]; then
    echo "[$(date -Is)] $cell RETIRED bench_incomplete" | tee -a "$clog"; kill_server; return 1
  fi
  echo "[$(date -Is)] $cell PASS" | tee -a "$clog"
  kill_server
  return 0
}
