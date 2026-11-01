#!/bin/bash
# run_final_campaign.sh — final_plan_20261019 serial campaign driver (rev1-pattern).
# Usage: run_final_campaign.sh F0SMOKE|CALIB1|CALIB2|MAIN|P1   (DRY_RUN=1 = walkthrough only)
#
# Governance (identical semantics to sealed run_sp_rev1_dev.sh):
#  - sha locks re-checked before EVERY boot (deps v3 lock, bench client, CSV, stream).
#  - TIMED_CAP consumed at BOOT_OK; RESULT_RETIRE never retried (formal phases).
#  - PRE-REGISTERED smoke exception (F0SMOKE only, registered before any boot):
#    smoke cells produce NO evaluation data, so a bench-level RESULT_RETIRE there
#    is INFRA-class: ONE same-cell re-boot within INFRA_CAP (=3 total per phase).
#    MECH_RETIRE / INTEGRITY_FAIL are NEVER re-bootable => HALT (exit 8) + report.
#  - INFRA_MISSING (boot failures beyond INFRA_CAP) => HALT (exit 8) + report.
#  - Budget violation (timed_used > TIMED_CAP) => HALT (exit 8).
#  - Verdicts never triggered by endpoint VALUES; W/H/S knobs untouched at runtime.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
F=$EXP/results/final_plan_20261019
PHASE=${1:?usage: run_final_campaign.sh F0SMOKE|CALIB1|CALIB2|MAIN|P1}
DRY_RUN=${DRY_RUN:-0}
DEPSHAS=$F/C_configs/deployed_shas_final_plan.txt
LAUNCH=$F/scripts/launch_arm.sh
INTEG=$F/scripts/integrity_check_fp.py
BENCHV2=$F/scripts/bench_client_v2.py
BENCHV2_SHAF=$F/C_configs/bench_client_v2.sha256
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
LOGDIR=$F/logs; mkdir -p "$LOGDIR"
TSV=$LOGDIR/campaign_marks.tsv
INFRA_CAP=3

case "$PHASE" in
  F0SMOKE)
    CSV=$F/D_matrix/run_list_f0_smoke.csv
    CSV_SHA=0b1d1519f77f378738bde3bc3f3e36c0c288eb9306fcd18c9b26ca7378924b54; NROWS=8
    TIMED_CAP=8
    DS=$F/smoke_f0/smoke_slice.jsonl
    DS_SHA=8b224dee4bcd126c93b32826dec8e1ff625ccd86a141149e4e963fa07ddfff90
    NREQ=384; CONC=64; BENCH_TIMEOUT=1800 ;;
  CALIB2)
    CSV=$F/D_matrix/run_list_calib2.csv
    CSV_SHA=95f738594ccfede00ccde65a5271f32bb7a8547f5a6ec96992507dd1340e2e07; NROWS=8
    TIMED_CAP=8
    DS=$F/B_data/streams/calib_confirm.jsonl
    DS_SHA=69ea817a3a20fd1a38deab1d60c0a730cec82caed12b17dd834c953a6e718bcf
    NREQ=1280; CONC=256; BENCH_TIMEOUT=2400 ;;
  CALIB1)
    CSV=$F/D_matrix/run_list_calib1.csv
    CSV_SHA=2c365171d57f668f61f9458eba74ab9990be50bbe042711b6d70567a3c32c401; NROWS=20
    TIMED_CAP=20
    DS=$F/B_data/streams/calib_screen.jsonl
    DS_SHA=d4fa14d2ee28343fa06428e6f9df2d5064ec6273ed1a96fa3e3c04dc8ae0b009
    NREQ=1280; CONC=256; BENCH_TIMEOUT=2400 ;;
  MAIN)
    CSV=$F/D_matrix/run_list_main.csv
    CSV_SHA=c2e06df4b9daa9acfeedda21a84624c5172893babdcec8697aff0bff6aa2dc60; NROWS=168
    TIMED_CAP=168
    # DS/DS_SHA/NREQ/BENCH_TIMEOUT are overridden per-row (cond E1-E4) in the loop; E1 = default
    DS=$F/B_data/streams/e1_code_stable.jsonl
    DS_SHA=dee24c502f7f301d92ce242c671c7a59278fb4749ecef1da74a40c36cb8fc682
    NREQ=3221; CONC=256; BENCH_TIMEOUT=2400 ;;
  P1)
    CSV=$F/D_matrix/run_list_p1.csv
    CSV_SHA=b324f8eac6d0ece81aedf24088e97548059a44fe19a6a1f8847ad6993c3ecbfd; NROWS=32
    TIMED_CAP=32
    DS=$F/B_data/streams/p1_longio.jsonl
    DS_SHA=3c09e0270a60d185c801646c3cebc7157a99086d0c10c8b4e78c5dc615068457
    NREQ=2048; CONC=64; BENCH_TIMEOUT=3600 ;;
  *) echo "unknown phase $PHASE" >&2; exit 9 ;;
esac

log(){ echo "[$(date -u +%FT%TZ)] [$PHASE] $*" | tee -a "$LOGDIR/driver_${PHASE}.log"; }
mark(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$PHASE" "$*" >> "$TSV"; }
sha_of(){ sha256sum "$1" | awk '{print $1}'; }

halt(){ log "HALT: $*"; mark "HALT $*"; kill_server_quiet; exit 8; }

check_locks(){
  sha256sum -c "$DEPSHAS" >/dev/null 2>&1 || { log "FATAL: dep lock v3 mismatch"; return 1; }
  sha256sum -c "$BENCHV2_SHAF" >/dev/null 2>&1 || { log "FATAL: bench client sha mismatch"; return 1; }
  [ "$(sha_of "$CSV")" = "$CSV_SHA" ] || { log "FATAL: run-list sha mismatch"; return 1; }
  [ "$(sha_of "$DS")" = "$DS_SHA" ] || { log "FATAL: stream sha mismatch"; return 1; }
  local n; n=$(( $(wc -l < "$CSV") - 1 ))
  [ "$n" = "$NROWS" ] || { log "FATAL: CSV rows $n != $NROWS"; return 1; }
  return 0; }

gpu_clear_wait(){ local i m; for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$m" -lt 1000 ] && return 0; sleep 4; done
  log "FATAL: GPU not clear after 360s"; return 1; }
kill_server_quiet(){ pkill -9 -f sglang.launch_server >/dev/null 2>&1
  pkill -9 -f 'sglang::' >/dev/null 2>&1; sleep 5
  gpu_clear_wait || return 1; return 0; }
env_clear(){ local v; for v in $(env | grep -oE '^(OEPLB_[A-Za-z0-9_]+|EA_[A-Za-z0-9_]+)=' | tr -d '='); do unset "$v" 2>/dev/null || true; done; }

check_locks || exit 8
log "phase start: rows=$NROWS timed_cap=$TIMED_CAP infra_cap=$INFRA_CAP ds=$DS conc=$CONC dry=$DRY_RUN"
timed_used=0; infra_used=0
declare -A CELL_VERDICT

attempt_cell(){ # $1 seq $2 arm $3 params $4 label $5 ART -> echoes VERDICT via globals VOUT/VERD
  local seq="$1" arm="$2" params="$3" label="$4" ART="$5"
  mkdir -p "$ART"
  { echo "phase=$PHASE seq=$seq arm=$arm params=$params label=$label"
    echo "utc=$(date -u +%FT%TZ) driver=run_final_campaign.sh"
    echo "csv_sha=$CSV_SHA ds_sha=$DS_SHA nreq=$NREQ conc=$CONC bench_timeout=$BENCH_TIMEOUT"
    echo "launcher_sha=$(sha_of "$LAUNCH") integ_sha=$(sha_of "$INTEG") benchv2_sha=$(sha_of "$BENCHV2")"
  } > "$ART/meta.txt"
  env_clear
  ART="$ART" setsid nohup bash "$LAUNCH" $arm $params > "$ART/server.log" 2>&1 &
  SPID=$!
  local i ok=0
  for i in $(seq 1 600); do
    if grep -q "ready to roll" "$ART/server.log" 2>/dev/null; then ok=1; break; fi
    if grep -qE "Traceback|CUDA out of memory|RuntimeError" "$ART/server.log" 2>/dev/null; then break; fi
    kill -0 "$SPID" 2>/dev/null || { sleep 2; break; }
    sleep 4
  done
  if [ "$ok" != 1 ]; then BOOT_RC=1; return; fi
  BOOT_RC=0
  tr '\0' '\n' < "/proc/$SPID/environ" 2>/dev/null | grep -E '^(OEPLB_|EA_|SGLANG_|NVSHMEM_|NCCL_)' | sort > "$ART/meta_env.txt" || true
  log "BOOT_OK $label (pid $SPID)"; mark "BOOT_OK $label"
  ( cd "$F/scripts" && OEPLB_BENCH_OUT="$ART" OEPLB_MODEL="$MODEL" \
      timeout "$BENCH_TIMEOUT" python3 "$BENCHV2" "$label" "$DS" "$CONC" ) > "$ART/bench.log" 2>&1
  [ -f "$ART/${label}.json" ] && cp "$ART/${label}.json" "$ART/bench.json"
  local extra=""
  [ "$arm" = E ] && extra="--eplb-period $(echo "$params" | awk '{print $3}')"
  [ "$arm" = F ] && extra="--expect-shw $(echo "$params" | tr ' ' ',')"
  VOUT=$(python3 "$INTEG" --artifact-dir "$ART" --arm "$arm" --label "$label" \
        --expected-requests "$NREQ" $extra 2>&1)
  echo "$VOUT" > "$ART/verdict.txt"
  VERD=$(echo "$VOUT" | grep -oE 'VERDICT:[A-Z_]+' | tail -1 | cut -d: -f2)
  [ -n "$VERD" ] || VERD=INTEGRITY_FAIL
}

while IFS=, read -r seq cphase cond block arm params label artifact_dir; do
  [ "$cphase" = "$PHASE" ] || continue
  if [ "$PHASE" = MAIN ]; then
    case "$cond" in
      E1) DS=$F/B_data/streams/e1_code_stable.jsonl; DS_SHA=dee24c502f7f301d92ce242c671c7a59278fb4749ecef1da74a40c36cb8fc682; NREQ=3221; BENCH_TIMEOUT=2400;;
      E2) DS=$F/B_data/streams/e2_prover_stable.jsonl; DS_SHA=fa788238fa2dcc90b12b8b2e18ab723509dd92937d1a8ff664ffec47627bc78f; NREQ=4082; BENCH_TIMEOUT=2400;;
      E3) DS=$F/B_data/streams/e3_multidomain_switch.jsonl; DS_SHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879; NREQ=8192; BENCH_TIMEOUT=3600;;
      E4) DS=$F/B_data/streams/e4_fullduplex.jsonl; DS_SHA=a024c5e61d0dbf321dee6245ef3d5f270da7e4374304d910e943cdd0657bc0a6; NREQ=2048; BENCH_TIMEOUT=3600;;
      *) echo "unknown MAIN cond $cond" >&2; exit 9;;
    esac
    CONC=256
  elif [ "$PHASE" = P1 ]; then
    case "$cond" in
      PL) CONC=64;;
      PH) CONC=256;;
      *) echo "unknown P1 cond $cond" >&2; exit 9;;
    esac
  fi
  ART=$F/$artifact_dir
  if grep -qF "VERDICT $label VALID" "$TSV" 2>/dev/null; then
    log "resume-skip $label (already VALID)"; timed_used=$((timed_used+1)); CELL_VERDICT[$label]=VALID; continue; fi
  if [ "$DRY_RUN" = 1 ]; then
    extra=""; [ "$arm" = E ] && extra="--eplb-period $(echo "$params" | awk '{print $3}')"
    [ "$arm" = F ] && extra="--expect-shw $(echo "$params" | tr ' ' ',')"
    echo "DRY seq=$seq arm=$arm params='$params' label=$label ART=$ART"
    echo "  boot: setsid nohup bash $LAUNCH $arm $params > $ART/server.log"
    echo "  bench: OEPLB_BENCH_OUT=$ART timeout $BENCH_TIMEOUT python3 $BENCHV2 $label $DS $CONC"
    echo "  check: python3 $INTEG --artifact-dir $ART --arm $arm --label $label --expected-requests $NREQ $extra"
    continue; fi
  check_locks || halt "lock check failed before seq=$seq"
  [ "$timed_used" -ge "$TIMED_CAP" ] && halt "budget: timed_used=$timed_used at seq=$seq"
  kill_server_quiet || halt "gpu not clear before seq=$seq"
  attempt_cell "$seq" "$arm" "$params" "$label" "$ART"
  if [ "$BOOT_RC" != 0 ] || [ -z "${VERD:-}" ]; then
    log "boot failure seq=$seq $label (infra_used=$infra_used)"; mark "BOOT_FAIL $label"
    kill_server_quiet || halt "gpu not clear after boot failure seq=$seq"
    if [ "$infra_used" -lt "$INFRA_CAP" ]; then
      infra_used=$((infra_used+1)); mark "INFRA_REBOOT $label n=$infra_used"
      log "INFRA re-boot #$infra_used for $label"
      attempt_cell "$seq" "$arm" "$params" "$label" "$ART"
      [ "$BOOT_RC" != 0 ] && { mark "INFRA_MISSING $label"; halt "second boot failure seq=$seq $label (INFRA_CAP path)"; }
    else
      mark "INFRA_MISSING $label"; halt "INFRA_CAP=$INFRA_CAP exhausted at seq=$seq $label"
    fi
  fi
  timed_used=$((timed_used+1))
  log "verdict $label: $VERD"; mark "VERDICT $label $VERD"
  case "$VERD" in
    VALID) CELL_VERDICT[$label]=VALID ;;
    RESULT_RETIRE)
      if [ "$PHASE" = F0SMOKE ]; then
        # pre-registered smoke exception: infra-class, one re-boot within INFRA_CAP
        if [ "$infra_used" -lt "$INFRA_CAP" ]; then
          infra_used=$((infra_used+1)); mark "INFRA_REBOOT $label n=$infra_used smoke-retire"
          log "smoke RESULT_RETIRE -> INFRA re-boot #$infra_used for $label"
          kill_server_quiet || halt "gpu not clear before smoke re-boot $label"
          attempt_cell "$seq" "$arm" "$params" "$label" "$ART"
          if [ "$BOOT_RC" != 0 ]; then mark "INFRA_MISSING $label"; halt "smoke re-boot failed $label"; fi
          timed_used=$((timed_used+1))
          log "verdict(re) $label: $VERD"; mark "VERDICT $label $VERD"
          case "$VERD" in
            VALID) CELL_VERDICT[$label]=VALID ;;
            RESULT_RETIRE) CELL_VERDICT[$label]=RESULT_RETIRE ;;
            *) halt "MECH/INTEGRITY failure after smoke re-boot: $label ($VERD)";;
          esac
        else
          CELL_VERDICT[$label]=RESULT_RETIRE
          halt "smoke RESULT_RETIRE beyond INFRA_CAP at $label"
        fi
      else
        CELL_VERDICT[$label]=RESULT_RETIRE   # formal phase: retired, never retried
      fi ;;
    MECH_RETIRE|INTEGRITY_FAIL)
      CELL_VERDICT[$label]=$VERD
      halt "mechanism/integrity failure at $label: $VERD (see $ART/verdict.txt)" ;;
    *) halt "unknown verdict '$VERD' at $label" ;;
  esac
done < <(tail -n +2 "$CSV")

if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN complete ($NROWS rows)"; exit 0; fi
kill_server_quiet || halt "gpu not clear at phase end"
# phase summary
if [ "$PHASE" = F0SMOKE ]; then SUM=$F/smoke_f0/summary.json
elif [ "$PHASE" = CALIB2 ]; then SUM=$F/calib/stage2/summary_calib2.json
elif [ "$PHASE" = MAIN ]; then SUM=$F/main/summary_main.json
elif [ "$PHASE" = P1 ]; then SUM=$F/p1/summary_p1.json
else SUM=$F/calib/stage1/summary_calib1.json; fi
mkdir -p "$(dirname "$SUM")"
python3 - "$SUM" "$PHASE" "$TSV" <<'PYEOF'
import json,sys,collections
sum_p,phase,tsv=sys.argv[1],sys.argv[2],sys.argv[3]
v=collections.OrderedDict()
for line in open(tsv):
    p=line.rstrip("\n").split("\t")
    if len(p)>=3 and p[1]==phase:
        w=p[2].split()
        if w and w[0]=="VERDICT" and len(w)>=3: v[w[1]]=w[2]
json.dump({"phase":phase,"verdicts":v,"n":len(v)},open(sum_p,"w"),indent=1)
print(sum_p,len(v))
PYEOF
if [ "$PHASE" = F0SMOKE ]; then
  nvalid=$(python3 -c "
import json;d=json.load(open('$SUM'));print(sum(1 for x in d['verdicts'].values() if x=='VALID'))")
  if [ "$nvalid" = "$NROWS" ]; then touch "$F/smoke_f0/SMOKE_PASS"; log "SMOKE_PASS: $nvalid/$NROWS VALID"; mark "SMOKE_PASS"; exit 0
  else log "SMOKE_GATE_FAIL: $nvalid/$NROWS VALID"; mark "SMOKE_GATE_FAIL"; exit 7; fi
fi
case "$PHASE" in
  CALIB1) mkdir -p "$F/calib/stage1"; touch "$F/calib/stage1/CALIB1_DONE";;
  CALIB2) mkdir -p "$F/calib/stage2"; touch "$F/calib/stage2/CALIB2_DONE";;
  MAIN) mkdir -p "$F/main"; touch "$F/main/MAIN_DONE";;
  P1) mkdir -p "$F/p1"; touch "$F/p1/P1_DONE";;
esac
log "$PHASE complete: summary at $SUM"; mark "${PHASE}_DONE"
exit 0
