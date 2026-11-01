#!/bin/bash
# h20-placement-fullservice-v3 campaign driver.
# Executes the frozen protocol configs/followup_v1/fullservice_protocol_v3.json:
#   phase 0 (SETUP, untimed): verify v2-lineage calibration counts + the
#           MoETuner-ILP-generated static map by sha256 against the protocol
#   phase 1 (MEASURED): round-robin 3 starts x 3 arms
#           (identity / frozen-static / online PB-OEPLB), every bench on the
#           disjoint EVALUATION requests only.
# The calibration/evaluation split is the frozen hash-based split in
# inputs/fullservice_v2/workload/split_manifest.json; all three files are hash
# verified against the frozen protocol before phase 0. Failed boots/benches are
# RETAINED with status and NEVER retried under this frozen campaign. No
# best-start selection. Logs/bench jsons copied to results.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/fullservice_v3
CONF=$EXP/configs/followup_v1
LAUNCH=$CONF/fullservice_launchers_v3
PROT=$CONF/fullservice_protocol_v3.json
SPLIT=$EXP/inputs/fullservice_v2/workload/split_manifest.json
export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
mkdir -p "$OUT/logs" "$OUT/bench" "$OUT/setup"
STATE=$OUT/campaign_state.json
CLOG=$OUT/campaign.log

log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }

[ -f "$PROT" ] || { log "FATAL protocol missing"; exit 1; }
python3 -c "import json;p=json.load(open('$PROT'));assert p['spec_id']=='h20-placement-fullservice-v3' and p['status']=='frozen_before_execution'" \
  || { log "FATAL protocol not frozen v2"; exit 1; }

# ---- frozen split verification (hash all three files against the protocol) ----
DSC=$(python3 - "$PROT" "$SPLIT" <<'PYV'
import hashlib,json,sys
prot=json.load(open(sys.argv[1])); man=json.load(open(sys.argv[2]))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()
pw=prot['workload']
assert sha(pw['source']['path'])==pw['source']['sha256'], 'source hash differs'
for key in ('calibration','evaluation'):
    assert man[key]['path']==pw[key]['path'], f'{key} path differs from protocol'
    assert sha(man[key]['path'])==man[key]['sha256']==pw[key]['sha256'], f'{key} hash differs'
assert man['calibration']['requests']+man['evaluation']['requests']==pw['source']['num_requests']
print(man['calibration']['path'])
PYV
) || { log "FATAL split verification failed"; exit 1; }
DSE=$(python3 -c "import json;print(json.load(open('$SPLIT'))['evaluation']['path'])")
log "SPLIT_OK calib=$DSC eval=$DSE"

record_state(){ # $1 = json string
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

boot(){ # $1 launcher  $2 logfile  rest: launcher args
  local launcher=$1; local lg=$2; shift 2
  setsid nohup bash "$launcher" "$@" > "$lg" 2>&1 &
  local i
  for i in $(seq 1 600); do
    grep -q "ready to roll" "$lg" 2>/dev/null && return 0
    grep -q "Traceback" "$lg" 2>/dev/null && return 2
    sleep 4
  done
  return 3
}

bench(){ # $1 label  $2 dataset ; returns bench rc
  cd /workspace/EPLB/OEPLB/scripts || return 9
  timeout 3600 python3 run_grid_bench.py "$1" "$2" 256 > "$OUT/logs/$1.bench.log" 2>&1
  local rc=$?
  cp "/workspace/EPLB/OEPLB/benchmarks/results/$1.json" "$OUT/bench/" 2>/dev/null
  cd "$EXP/scripts" || true
  return $rc
}

run_one(){ # $1 arm  $2 label  $3 launcher  rest args
  local arm=$1 label=$2 launcher=$3; shift 3
  local slog="$OUT/logs/${label}_server.log"
  log "RUN $label arm=$arm launcher=$(basename "$launcher")"
  kill_server
  boot "$launcher" "$slog" "$@"
  local brc=$?
  if [ $brc -ne 0 ]; then
    log "BOOT_FAIL $label rc=$brc"
    record_state "{\"phase\":\"measure\",\"run\":\"$label\",\"arm\":\"$arm\",\"boot_rc\":$brc,\"status\":\"boot_failed\"}"
    kill_server
    return 1
  fi
  local v1 v2 v3
  v1=$(grep -c "init_expert_location from init_by_mapping" "$slog" 2>/dev/null || true)
  v2=$(grep -c "PB-OEPLB] Init" "$slog" 2>/dev/null || true)
  v3=$(grep -c "ready to roll" "$slog" 2>/dev/null || true)
  bench "$label" "$DSE"
  local benchrc=$?
  local tb
  tb=$(grep -c "Traceback" "$slog" 2>/dev/null || true)
  log "BENCH_DONE $label rc=$benchrc init_by_mapping=$v1 oeplb_init=$v2 tracebacks=$tb"
  record_state "{\"phase\":\"measure\",\"run\":\"$label\",\"arm\":\"$arm\",\"boot_rc\":0,\"bench_rc\":$benchrc,\"init_by_mapping_lines\":$v1,\"oeplb_init_lines\":$v2,\"server_tracebacks\":$tb,\"status\":\"completed\"}"
  kill_server
  return 0
}

log "CAMPAIGN_START $(sha256sum "$PROT" | cut -c1-16)"

# ---------------- phase 0: setup verification (untimed; lineage from v2) ----------------
# The calibration counts and the ILP-generated static map already exist (v2 setup
# lineage + MoETuner ILP solve). Verify every setup artifact by sha256 against the
# frozen v3 protocol, copy into results/fullservice_v3/setup, then measure.
log "PHASE0 verifying setup artifacts against frozen protocol"
python3 - "$PROT" "$OUT" <<'PYSETUP' || { log "FATAL setup verification failed"; exit 1; }
import hashlib, json, os, shutil, sys
prot = json.load(open(sys.argv[1])); out = sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()
su = prot['setup_artifacts']
os.makedirs(os.path.join(out, 'setup'), exist_ok=True)
for key, rec in su.items():
    src = rec['path']
    assert sha(src) == rec['sha256'], f'{key}: sha differs {src}'
    dst = os.path.join(out, 'setup', os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    assert sha(dst) == rec['sha256'], f'{key}: copy sha differs'
m = json.load(open(su['frozen_map']['path']))['physical_to_logical_map']
assert len(m) == 94 and all(len(l) == 128 and sorted(l) == list(range(128)) for l in m)
print('SETUP_VERIFIED')
PYSETUP
record_state "{\"phase\":\"setup\",\"step\":\"verify_artifacts\",\"status\":\"completed\"}"

# ---------------- phase 1: measured runs on EVALUATION requests, round-robin ----------------
for s in 1 2 3; do
  run_one identity "fs300v3_id_s$s" /workspace/logs/launch235b_identity.sh
  run_one frozen   "fs300v3_fx_s$s" "$LAUNCH/launch235b_fixedh20.sh" "$OUT/setup/frozen_map_ilp.json"
  run_one online   "fs300v3_on_s$s" /workspace/logs/launch235b_oeplb.sh
done

log "CAMPAIGN_DONE"
record_state "{\"phase\":\"campaign\",\"status\":\"done\"}"
