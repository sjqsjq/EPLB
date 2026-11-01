#!/bin/bash
# Five-arm campaign driver (A00/A10/A01/A11/P11).
# Executes configs/followup_v1/ea_v1_campaign_protocol.json (frozen):
#   phase 0: hash-verify every locked artifact
#   phase 1: 25 formal starts = 5 rounds x 5 arms, round-robin rotation
#            (each arm occupies each within-round position exactly once)
# Workload: evaluation.jsonl (7963 real requests; v3 order/concurrency 256/
# service protocol). Calibration requests are NEVER touched here.
# Failed starts are recorded and retired, never retried in this frozen
# campaign. No best-start/repeat selection. Config mismatch = systematic
# protocol violation -> campaign aborts (not a retryable start).
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
OUT=$EXP/results/ea_v1/fivearm_campaign
CONF=$EXP/configs/followup_v1
LAUNCH=$CONF/ea_launchers/launch235b_eacampaign.sh
PROT=$CONF/ea_v1_campaign_protocol.json
DSC=$EXP/inputs/fullservice_v2/workload/evaluation.jsonl
mkdir -p "$OUT/logs" "$OUT/bench" "$OUT/diag"
STATE=$OUT/campaign_state.json
CLOG=$OUT/campaign.log
log(){ echo "[$(date -Is)] $*" | tee -a "$CLOG"; }

[ -f "$PROT" ] || { log "FATAL protocol missing"; exit 1; }
python3 -c "import json;p=json.load(open('$PROT'));assert p['spec_id']=='h20-placement-ea-v1-fivearm-campaign' and p['status']=='frozen_before_execution'" \
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

# ---- schedule from frozen protocol ----
python3 - "$PROT" > "$OUT/schedule.tsv" <<'PYS'
import json,sys
p=json.load(open(sys.argv[1]))
for e in p["schedule"]:
    a=p["arms"][e["arm"]]
    print("\t".join([e["label"],str(e["round"]),str(e["position"]),e["arm"],
                     str(a["W"]),str(a["alpha"]),str(int(a["adw"])),str(int(a["add"])),str(int(a["planner_bi"]))]))
PYS
NS=$(wc -l < "$OUT/schedule.tsv")
[ "$NS" -eq 25 ] || { log "FATAL schedule has $NS entries, expected 25"; exit 1; }

record_state(){
  STATE="$STATE" python3 - "$1" <<'PYSTATE'
import json,os,sys,datetime
p=os.environ["STATE"]
e=json.loads(sys.argv[1]); e["ts"]=datetime.datetime.now().astimezone().isoformat()
st=json.load(open(p)) if os.path.exists(p) else []
st.append(e); json.dump(st,open(p,"w"),indent=1)
PYSTATE
}
prev_status(){ # $1 label -> status string or empty
  python3 - "$STATE" "$1" <<'PYP'
import json,os,sys
p,lab=sys.argv[1],sys.argv[2]
if not os.path.exists(p): print(""); raise SystemExit
for e in json.load(open(p)):
    if e.get("run")==lab and e.get("phase")=="campaign": print(e.get("status","?")); break
else: print("")
PYP
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
config_verify(){ # $1 slog $2 W $3 ALPHA $4 ADW $5 ADD $6 BI
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYC'
import re,sys
slog,W,ALPHA,ADW,ADD,BI=sys.argv[1:7]
t=open(slog,errors='replace').read()
errs=[]
def need(p,desc):
    if not re.search(p,t): errs.append(desc)
need(r"pb_oeplb_sync_window=%s\b"%re.escape(W),"sync_window!=%s"%W)
need(r"pb_oeplb_decay_factor=%s\b"%re.escape(ALPHA),"decay_factor!=%s"%ALPHA)
tw={1:"True",0:"False"}
need(r"pb_oeplb_adaptive_window=%s\b"%tw[int(ADW)],"adaptive_window!=%s"%tw[int(ADW)])
need(r"pb_oeplb_adaptive_decay=%s\b"%tw[int(ADD)],"adaptive_decay!=%s"%tw[int(ADD)])
need(r"pb_oeplb_calibrate_adaptive_sensitivity=False\b","calibrate!=False (C7)")
need(r"pb_oeplb_dead_zone_ratio=0\.0\b","dead_zone_ratio!=0.0 (C6 auto 1.096)")
need(r"pb_oeplb_threshold_ratio=1\.02\b","threshold_ratio!=1.02")
need(r"pb_oeplb_max_total_ops=300\b","max_total_ops!=300")
if int(BI)==1:
    need(r"\[PB-OEPLB-PLANNER\] rule=best_improvement_rank_pair","planner rule != best_improvement_rank_pair")
else:
    need(r"\[PB-OEPLB-PLANNER\] rule=heuristic_first_improvement","planner rule != heuristic_first_improvement")
if errs:
    print("CONFIG_MISMATCH "+"; ".join(errs)); sys.exit(1)
print("CONFIG_OK"); sys.exit(0)
PYC
}
diag_extract(){ # $1 slog $2 outfile
  python3 - "$1" "$2" <<'PYD'
import re,sys,json,hashlib
src,out=sys.argv[1],sys.argv[2]
TS=re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
mech={k:0 for k in ["adw_ratio_jump","adw_shrink","adw_decay0_window_fixed","adw_grow",
 "window_shift","window_stable","reset","warmstart","calib_tier","refine",
 "arg_adw_true","arg_add_true","tracebacks","hookdiag","planner_bi_log","planner_default_log",
 "issued_windows_dp0","verify_move_before","swap_done_lines"]}
window_events=[];reset_events=[];ready_after=[];prof=[];diag=[];walls=[];timing=[];issued=[];moves=[]
args={}
for ln in open(src,errors='replace'):
    m=TS.match(ln); ts=m.group(1) if m else None
    dp0=(" DP0 " in ln)
    if "[PB-OEPLB-ADW]" in ln:
        if "ratio jump" in ln: mech["adw_ratio_jump"]+=1
        if "shrink sw to" in ln: mech["adw_shrink"]+=1
        if "decay->0 (clear stale history; window fixed)" in ln: mech["adw_decay0_window_fixed"]+=1
        if "grow sw to" in ln: mech["adw_grow"]+=1
        if dp0: window_events.append([ts,ln.split("[PB-OEPLB-ADW]",1)[1].strip()])
    elif "[PB-OEPLB-WINDOW]" in ln:
        if "shift confirmed" in ln: mech["window_shift"]+=1
        if "stable confirmed" in ln: mech["window_stable"]+=1
        if dp0: window_events.append([ts,ln.split("[PB-OEPLB-WINDOW]",1)[1].strip()])
    elif "[PB-OEPLB-RESET]" in ln:
        mech["reset"]+=1
        if dp0:
            mm=re.search(r"converged=([\d.]+) prev=([\d.]+) current=([\d.]+)",ln)
            reset_events.append([ts]+([float(x) for x in mm.groups()] if mm else []))
    elif "[PB-OEPLB-WARMSTART]" in ln: mech["warmstart"]+=1
    elif "[PB-OEPLB-CALIB]" in ln: mech["calib_tier"]+=1
    elif "[PB-OEPLB-REFINE]" in ln: mech["refine"]+=1
    elif "[PB-OEPLB-HOOKDIAG]" in ln: mech["hookdiag"]+=1
    elif "[PB-OEPLB-PLANNER]" in ln:
        mech["planner_bi_log" if "best_improvement_rank_pair" in ln else "planner_default_log"]+=1
    elif "Traceback" in ln: mech["tracebacks"]+=1
    if "pb_oeplb_adaptive_window=True" in ln: mech["arg_adw_true"]+=1
    if "pb_oeplb_adaptive_decay=True" in ln: mech["arg_add_true"]+=1
    if "server_args=ServerArgs" in ln and not args:
        for k in ["sync_window","decay_factor","adaptive_window","adaptive_decay",
                  "calibrate_adaptive_sensitivity","dead_zone_ratio","threshold_ratio",
                  "max_total_ops","window_floor","min_swap_ops"]:
            mm=re.search(r"pb_oeplb_%s=([^,\)]+)"%k,ln)
            if mm: args[k]=mm.group(1)
    mm=re.search(r"window#(\d+): issued (\d+) swap\(s\) \((\d+) tok global, dp=(\d+)\)",ln)
    if mm and dp0:
        mech["issued_windows_dp0"]+=1
        issued.append([int(mm.group(1)),int(mm.group(2)),int(mm.group(3)),int(mm.group(4))])
    mm=re.search(r"(\d+) swap\(s\) done \(([\d.]+)ms\)",ln)
    if mm:
        mech["swap_done_lines"]+=1
        walls.append([ts,int(mm.group(1)),float(mm.group(2))])
    mm=re.search(r"\[PB-OEPLB-TIMING\] begin\(\): temp_alloc=([\d.]+)ms batch_isend_irecv=([\d.]+)ms copy=([\d.]+)ms total_begin=([\d.]+)ms n_ops=(\d+)",ln)
    if mm: timing.append([ts,float(mm.group(4))])
    mm=re.search(r"\[PB-OEPLB-PROF\] window#(\d+) calls=(\d+) record=([\d.]+)ms allreduce=([\d.]+)ms planbuild=([\d.]+)ms finalize=([\d.]+)ms total=([\d.]+)ms",ln)
    if mm and dp0:
        prof.append({"window":int(mm.group(1)),"calls":int(mm.group(2)),
                     "record_ms":float(mm.group(3)),"allreduce_ms":float(mm.group(4)),
                     "planbuild_ms":float(mm.group(5)),"finalize_ms":float(mm.group(6)),
                     "total_ms":float(mm.group(7))})
    mm=re.search(r"\[PB-OEPLB-DIAG\] layers_touched=(\d+) total_ops=(\d+) avg_ratio_before=([\d.]+) avg_ratio_after=([\d.]+) max_ratio_before=([\d.]+) max_ratio_after=([\d.]+)",ln)
    if mm and dp0:
        diag.append({"layers_touched":int(mm.group(1)),"total_ops":int(mm.group(2)),
                     "avg_ratio_before":float(mm.group(3)),"avg_ratio_after":float(mm.group(4)),
                     "max_ratio_before":float(mm.group(5)),"max_ratio_after":float(mm.group(6))})
    mm=re.search(r"rank=(\d+) ready after (\d+) forwards",ln)
    if mm: ready_after.append([int(mm.group(1)),int(mm.group(2))])
    mm=re.search(r"\[VERIFY-WEIGHT-MOVE\] rank=(\d+) BEFORE swap layer=(\d+) phys_a=(\d+)\(rank(\d+)\) phys_b=(\d+)\(rank(\d+)\)",ln)
    if mm:
        mech["verify_move_before"]+=1
        moves.append([ts,int(mm.group(2)),int(mm.group(3)),int(mm.group(5))])
seen=set();seq=[]
for mv in moves:
    k=(mv[0],mv[1],min(mv[2],mv[3]),max(mv[2],mv[3]))
    if k in seen: continue
    seen.add(k);seq.append(mv)
last={};rev=[];rep=[]
for ts2,layer,pa,pb in seq:
    uk=(layer,min(pa,pb),max(pa,pb))
    if uk in last:
        lpa,lpb=last[uk]
        (rep if (pa,pb)==(lpa,lpb) else rev).append([ts2,layer,pa,pb])
    last[uk]=(pa,pb)
we={}
for ts2,n,ms in walls:
    e=we.setdefault(ts2,{"n_ranks":0,"wall_ms":0.0,"ops":0})
    e["n_ranks"]+=1;e["wall_ms"]=max(e["wall_ms"],ms);e["ops"]=max(e["ops"],n)
wall_events=[[k,v["wall_ms"],v["ops"],v["n_ranks"]] for k,v in sorted(we.items())]
tg={}
for ts2,tb in timing: tg[ts2]=max(tg.get(ts2,0.0),tb)
res={"source":src,
 "source_sha256":hashlib.sha256(open(src,'rb').read()).hexdigest(),
 "mechanisms":mech,"server_args_pb_oeplb":args,
 "window_events_dp0":window_events,"reset_events_dp0":reset_events,
 "ready_after_forwards":ready_after,"issued_windows_dp0":issued,
 "prof_dp0":prof,"diag_dp0":diag,
 "wall_events":wall_events,
 "wall_summary":{"n_events":len(wall_events),
   "max_wall_ms":max([w[1] for w in wall_events],default=0.0),
   "sum_wall_ms":sum(w[1] for w in wall_events)},
 "timing_begin_max_ms_per_event":sorted(tg.items()),
 "move_sequence_len":len(seq),
 "swap_stats":{"n_ops":len(seq),"n_reversals":len(rev),"n_repeats":len(rep),
   "reversals_sample":rev[:50],"repeats_sample":rep[:50]}}
json.dump(res,open(out,"w"),indent=1)
print("DIAG_OK",out)
PYD
}
run_one(){ # $1 label $2 round $3 pos $4 arm $5 W $6 ALPHA $7 ADW $8 ADD $9 BI
  local label=$1 round=$2 pos=$3 arm=$4 W=$5 ALPHA=$6 ADW=$7 ADD=$8 BI=$9
  local slog="$OUT/logs/${label}_server.log"
  local dfile="$OUT/diag/${label}_diag.json"
  local prev; prev=$(prev_status "$label")
  if [ "$prev" = "completed" ]; then log "SKIP_RESUME $label (already completed; resume, not retry)"; return 0; fi
  if [ -n "$prev" ]; then
    log "FATAL_RETRY_REFUSED $label has prior status=$prev; failed starts are retired, never retried in this frozen campaign"; exit 1
  fi
  log "RUN $label round=$round pos=$pos arm=$arm W=$W alpha=$ALPHA adw=$ADW add=$ADD bi=$BI"
  kill_server
  export EA_W=$W EA_ALPHA=$ALPHA EA_ADAPTIVE_WINDOW=$ADW EA_ADAPTIVE_DECAY=$ADD EA_PLANNER_BI=$BI
  boot "$slog"
  local brc=$?
  if [ $brc -ne 0 ]; then
    log "BOOT_FAIL $label rc=$brc"
    record_state "{\"phase\":\"campaign\",\"run\":\"$label\",\"round\":$round,\"position\":$pos,\"arm\":\"$arm\",\"W\":$W,\"alpha\":$ALPHA,\"adw\":$ADW,\"add\":$ADD,\"planner_bi\":$BI,\"boot_rc\":$brc,\"status\":\"boot_failed\"}"
    kill_server
    return 1
  fi
  local cvout; cvout=$(config_verify "$slog" "$W" "$ALPHA" "$ADW" "$ADD" "$BI")
  if [ $? -ne 0 ]; then
    log "FATAL_CONFIG $label: $cvout"
    record_state "{\"phase\":\"campaign\",\"run\":\"$label\",\"round\":$round,\"position\":$pos,\"arm\":\"$arm\",\"status\":\"config_mismatch\",\"detail\":\"$cvout\"}"
    kill_server
    log "CAMPAIGN_ABORT config mismatch is a systematic protocol violation, not a retryable start"
    exit 1
  fi
  log "CONFIG_OK $label $cvout"
  bench "$label" "$DSC"
  local benchrc=$?
  diag_extract "$slog" "$dfile" || { log "WARN diag_extract failed for $label"; }
  local mj bj dsha
  mj=$(python3 -c "import json;print(json.dumps(json.load(open('$dfile'))['mechanisms']))" 2>/dev/null || echo '{}')
  bj=$(python3 -c "import json;d=json.load(open('$OUT/bench/$label.json'));print(json.dumps({k:d.get(k) for k in ('ok','errors','total_time_s','tps','total_comp_tokens','lat_mean_ms')}))" 2>/dev/null || echo '{}')
  dsha=$(sha256sum "$dfile" 2>/dev/null | cut -d' ' -f1)
  log "BENCH_DONE $label rc=$benchrc bench=$bj mech=$mj"
  record_state "{\"phase\":\"campaign\",\"run\":\"$label\",\"round\":$round,\"position\":$pos,\"arm\":\"$arm\",\"W\":$W,\"alpha\":$ALPHA,\"adw\":$ADW,\"add\":$ADD,\"planner_bi\":$BI,\"boot_rc\":0,\"config_verify\":\"CONFIG_OK\",\"bench_rc\":$benchrc,\"bench\":$bj,\"mechanisms\":$mj,\"diag_file\":\"diag/${label}_diag.json\",\"diag_sha256\":\"$dsha\",\"status\":\"completed\"}"
  kill_server
  return 0
}

log "CAMPAIGN_START $(sha256sum "$PROT" | cut -c1-16)"
while IFS="	" read -r LABEL ROUND POS ARM W ALPHA ADW ADD BI; do
  run_one "$LABEL" "$ROUND" "$POS" "$ARM" "$W" "$ALPHA" "$ADW" "$ADD" "$BI"
done < "$OUT/schedule.tsv"
log "CAMPAIGN_DONE"
record_state '{"phase":"campaign_meta","status":"done"}'
