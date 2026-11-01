#!/bin/bash
# E-A calibration grid + C5-dynamic smokes driver.
# Executes configs/followup_v1/ea_v1_calibration_grid_protocol.json (frozen):
#   phase 0: hash-verify every locked artifact (workload, code, launchers, bench tool)
#   phase 1: 3 smokes (A10/A01/A11 signatures) on CALIBRATION requests
#   phase 2: grid W x alpha, 2 independent starts per combo, CALIBRATION requests only
# Evaluation requests are NEVER touched. Failed starts are retained and retired,
# never retried. No best-start selection. Smoke data never enters selection.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/ea_v1/calibration_grid
CONF=$EXP/configs/followup_v1
LAUNCH=$CONF/ea_launchers/launch235b_eagrid.sh
PROT=$CONF/ea_v1_calibration_grid_protocol.json
DSC=$EXP/inputs/fullservice_v2/workload/calibration.jsonl
mkdir -p "$OUT/logs" "$OUT/bench"
STATE=$OUT/campaign_state.json
CLOG=$OUT/campaign.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }

[ -f "$PROT" ] || { log "FATAL protocol missing"; exit 1; }
python3 -c "import json;p=json.load(open('$PROT'));assert p['spec_id']=='h20-placement-ea-v1-calibration-grid' and p['status']=='frozen_before_execution'" \
  || { log "FATAL protocol not frozen"; exit 1; }

# ---- phase 0: verify all locked hashes ----
python3 - "$PROT" <<'PYV' || { log "FATAL lock verification failed"; exit 1; }
import hashlib,json,sys
prot=json.load(open(sys.argv[1]))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()
for key,rec in prot["locked_artifacts"].items():
    got=sha(rec["path"])
    assert got==rec["sha256"], f"{key}: sha differs ({got[:12]} vs {rec['sha256'][:12]}) {rec['path']}"
print("LOCKS_VERIFIED", len(prot["locked_artifacts"]), "artifacts")
PYV
log "PHASE0_OK"

record_state(){
  STATE="$STATE" python3 - "$1" <<'PYSTATE'
import json,os,sys,datetime
p=os.environ["STATE"]
e=json.loads(sys.argv[1]); e["ts"]=datetime.datetime.now().astimezone().isoformat()
st=json.load(open(p)) if os.path.exists(p) else []
st.append(e); json.dump(st,open(p,"w"),indent=1)
PYSTATE
}

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
  if gpu_clear_wait; then return 0; fi
  log "WARN gpus not clear after kill"; return 1
}
boot(){ # $1 logfile (env EA_* already exported)
  setsid nohup bash "$LAUNCH" > "$1" 2>&1 &
  for i in $(seq 1 600); do
    grep -q "ready to roll" "$1" 2>/dev/null && return 0
    grep -q "Traceback" "$1" 2>/dev/null && return 2
    sleep 4
  done
  return 3
}
bench(){ # $1 label $2 dataset
  cd /workspace/EPLB/OEPLB/scripts || return 9
  timeout 3600 python3 run_grid_bench.py "$1" "$2" 256 > "$OUT/logs/$1.bench.log" 2>&1
  local rc=$?
  cp "/workspace/EPLB/OEPLB/benchmarks/results/$1.json" "$OUT/bench/" 2>/dev/null
  cd "$EXP/scripts" || true
  return $rc
}
mech_counts(){ # $1 serverlog -> compact json
  python3 - "$1" <<'PYM'
import re,sys,json
t=open(sys.argv[1],errors='replace').read()
def c(p): return len(re.findall(p,t))
print(json.dumps({
 "adw_ratio_jump":c(r"PB-OEPLB-ADW\] ratio jump"),
 "adw_shrink":c(r"ratio jump [0-9.]+ -> shrink sw to"),
 "adw_decay0_window_fixed":c(r"decay->0 \(clear stale history; window fixed\)"),
 "adw_grow":c(r"PB-OEPLB-ADW\] (?:converged|volatile) 3x -> grow"),
 "window_shift":c(r"PB-OEPLB-WINDOW\] shift confirmed"),
 "window_stable":c(r"PB-OEPLB-WINDOW\] stable confirmed"),
 "reset":c(r"PB-OEPLB-RESET\]"),
 "warmstart":c(r"PB-OEPLB-WARMSTART\]"),
 "calib_tier":c(r"PB-OEPLB-CALIB\]"),
 "refine":c(r"PB-OEPLB-REFINE\]"),
 "arg_adw_true":c(r"pb_oeplb_adaptive_window=True"),
 "arg_add_true":c(r"pb_oeplb_adaptive_decay=True"),
 "tracebacks":c(r"Traceback")}))
PYM
}
run_one(){ # $1 label $2 phase $3 W $4 ALPHA $5 ADW $6 ADD
  local label=$1 phase=$2
  local slog="$OUT/logs/${label}_server.log"
  log "RUN $label phase=$phase W=$3 alpha=$4 adw=$5 add=$6"
  kill_server
  export EA_W=$3 EA_ALPHA=$4 EA_ADAPTIVE_WINDOW=$5 EA_ADAPTIVE_DECAY=$6
  boot "$slog"
  local brc=$?
  if [ $brc -ne 0 ]; then
    log "BOOT_FAIL $label rc=$brc"
    record_state "{\"phase\":\"$phase\",\"run\":\"$label\",\"W\":$3,\"alpha\":$4,\"adw\":$5,\"add\":$6,\"boot_rc\":$brc,\"status\":\"boot_failed\"}"
    kill_server
    return 1
  fi
  bench "$label" "$DSC"
  local benchrc=$?
  local mj bj
  mj=$(mech_counts "$slog")
  bj=$(python3 -c "import json;d=json.load(open('$OUT/bench/$label.json'));print(json.dumps({k:d.get(k) for k in ('ok','errors','total_time_s','tps','total_comp_tokens','lat_mean_ms')}))" 2>/dev/null || echo '{}')
  log "BENCH_DONE $label rc=$benchrc bench=$bj mech=$mj"
  record_state "{\"phase\":\"$phase\",\"run\":\"$label\",\"W\":$3,\"alpha\":$4,\"adw\":$5,\"add\":$6,\"boot_rc\":0,\"bench_rc\":$benchrc,\"bench\":$bj,\"mechanisms\":$mj,\"status\":\"completed\"}"
  kill_server
  return 0
}

log "GRID_START $(sha256sum "$PROT" | cut -c1-16)"
# ---- phase 1: C5-dynamic smokes (log signatures; never used for selection) ----
run_one easmoke_A10_s1 smoke 16 0.5 1 0
run_one easmoke_A01_s1 smoke 16 0.5 0 1
run_one easmoke_A11_s1 smoke 16 0.5 1 1
# ---- phase 2: grid, 2 independent starts per combo, round-robin by start ----
for s in 1 2; do
  for W in 8 16 32 64; do
    for A in 0.0 0.5 0.9; do
      AT=$(echo "$A" | tr '.' 'p')
      run_one "eagrid_W${W}_A${AT}_s${s}" grid "$W" "$A" 0 0
    done
  done
done
log "GRID_DONE"
record_state '{"phase":"campaign","status":"done"}'
