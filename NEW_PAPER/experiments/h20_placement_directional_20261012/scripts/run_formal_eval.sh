#!/bin/bash
# =============================================================================
# run_formal_eval.sh — ADW v2.1 formal full-service evaluation driver (48 timed
# starts). Generated from PROTOCOL_formal_eval_v2p1_DRAFT.md §3/§8/§10 + locked
# run list; sha-locked before first start (§14). LOCAL governance: no push.
#
# Rules encoded here (protocol §6/§8/§10):
#  - Execute run_list CSV strictly in seq order; failed starts never shift order.
#  - Per-start hygiene: kill server -> GPU clear -> sha256sum -c deployed shas
#    -> export arm env -> boot (poll "ready to roll" <=2400s) -> bench
#    (timeout 3600, concurrency 256) -> copy artifacts -> integrity_check.py
#    -> teardown -> append state marker.
#  - INFRA zero-data boot failure (never ready): ONE same-cell re-boot,
#    campaign cap 4 (flag INFRA-R); second failure -> INFRA_MISSING cell-round.
#  - Result failure (ok<n | errors>0) -> RESULT_RETIRE, never retried.
#  - Mechanism failure / integrity fail -> MECH_RETIRE / INTEGRITY_FAIL.
#  - For-cause halt: A or M with >=2 non-VALID rounds (retirements or
#    INFRA_MISSING) in the SAME condition -> stop campaign, report to author.
#    Halts are NEVER triggered by endpoint values. No interim cross-arm
#    analysis anywhere in this driver; it only logs per-start facts.
#  - Resume: state file is append-only (tab-separated; fields:
#      $1=ts $2=MARKER $3=seq $4=label $5=cond $6=arm $7=verdict/extra ...).
#    A seq with VERDICT is skipped; a seq with START but no VERDICT and an
#    existing bench.json is recovered offline via integrity_check (marked
#    RESUME_RECOVERED); otherwise it is treated as zero-data INFRA re-run.
#  - DRY_RUN=1: parse CSV, resolve env per row, verify paths/labels; no GPU ops.
# =============================================================================
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
FE=$EXP/results/adw_v2p1/formal_eval
OUTROOT=$EXP/results/adw_v2p1/r3_logging
DEPSHAS=$OUTROOT/deployed_shas.txt
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
BENCHROOT=/workspace/EPLB/OEPLB
CSV=$FE/run_list_formal_eval_v2p1.csv
CSV_SHA_LOCKED=b1d8a45e9f2c856bc3c8830d8b5fcd673ed5f90ba58919ac4dc61ce73e6c6ab4
INTEG=$EXP/scripts/integrity_check.py
STATE=$FE/driver_state.tsv
DLOG=$FE/driver.log
DRY_RUN=${DRY_RUN:-0}
INFRA_CAP=4
DS_W1=$EXP/inputs/fullservice_v2/workload/w1_stable_code_4000.jsonl
DS_W2=$EXP/inputs/fullservice_v2/workload/w2_switch_code_chinese_8000.jsonl
DS_W1_SHA=f0391f7cdf8edd734a09c65b331d751052b8b1937a0416b3782a409039fe8578
DS_W2_SHA=9d852a756957d3db805fd876ebbad51a0224b9b6c23d23edb5646cdf260362f8

log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }
mark(){ local f; printf '%s' "$(date -Is)" >> "$STATE"; for f in "$@"; do printf '\t%s' "$f"; done >> "$STATE"; printf '\n' >> "$STATE"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }

export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8

# ---- arm env (verbatim authority: run_v2p1_behavior.sh / run_v2p1_logging.sh) ----
env_common(){
  export EA_ALPHA=0.0
  unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
}
env_v21_defaults(){
  export OEPLB_V21_M_SHRINK=2 OEPLB_V21_M_GROW=3
  export OEPLB_V21_DWELL_SHRINK=2 OEPLB_V21_DWELL_GROW=2
  export OEPLB_V21_U_MAX=4 OEPLB_V21_W_UNCERTAIN_TARGET=16
  export OEPLB_V21_URGENT_EPISODE_CAP=3 OEPLB_V21_BETA=0.5
}
env_A(){
  env_common
  export OEPLB_WINDOW_CONTROLLER=21
  # FROZEN fitted knobs (FREEZE_record_v2p1.md; do not edit)
  export OEPLB_V21_KAPPA=3875.1768798828125
  export OEPLB_V21_Z_LO=22.643858727592757 OEPLB_V21_Z_HI=314.14010272527486 OEPLB_V21_Z_URGENT=942.4203081758246
  export OEPLB_V21_THETA_R_LOW=0.010095552687949321 OEPLB_V21_THETA_R_ACT=0.011395643112507271
  export OEPLB_V21_N_MIN=12320768
  env_v21_defaults
  export OEPLB_V21_LOG=$ART/v2p1_rank0.jsonl
  export EA_W=16
}
env_M(){
  env_common
  export OEPLB_WINDOW_CONTROLLER=21
  # INERT knobs (LOCK_record_v2p1 §5; monitoring-matched fixed mode)
  export OEPLB_V21_KAPPA=1.0
  export OEPLB_V21_Z_LO=1e12 OEPLB_V21_Z_HI=2e12 OEPLB_V21_Z_URGENT=3e12
  export OEPLB_V21_THETA_R_LOW=1e-12 OEPLB_V21_THETA_R_ACT=2.0
  export OEPLB_V21_N_MIN=0
  env_v21_defaults
  export OEPLB_V21_LOG=$ART/v2p1_rank0.jsonl
  export EA_W=16
}
env_legacy(){  # F / L: controller 1, NO v2.1 machinery of any kind
  env_common
  export OEPLB_WINDOW_CONTROLLER=1
  unset OEPLB_V21_KAPPA OEPLB_V21_Z_LO OEPLB_V21_Z_HI OEPLB_V21_Z_URGENT
  unset OEPLB_V21_THETA_R_LOW OEPLB_V21_THETA_R_ACT OEPLB_V21_N_MIN
  unset OEPLB_V21_M_SHRINK OEPLB_V21_M_GROW OEPLB_V21_DWELL_SHRINK OEPLB_V21_DWELL_GROW
  unset OEPLB_V21_U_MAX OEPLB_V21_W_UNCERTAIN_TARGET OEPLB_V21_URGENT_EPISODE_CAP OEPLB_V21_BETA
  unset OEPLB_V21_LOG
  export EA_W=$1
}

# ---- preflight locks ----
got=$(sha256sum "$CSV" | awk '{print $1}')
[ "$got" = "$CSV_SHA_LOCKED" ] || { log "FATAL: run-list sha mismatch ($got) — aborting"; exit 9; }
python3 -m py_compile "$INTEG" || { log "FATAL: integrity checker does not compile"; exit 9; }
nrows=$(python3 -c "import csv;print(sum(1 for _ in csv.reader(open('$CSV')))-1)")
[ "$nrows" = "48" ] || { log "FATAL: run list has $nrows rows, expected 48"; exit 9; }
nuniq=$(python3 -c "import csv;rows=list(csv.reader(open('$CSV')))[1:];print(len(set(r[10] for r in rows)))")
[ "$nuniq" = "48" ] || { log "FATAL: bench labels not unique ($nuniq/48)"; exit 9; }
if [ "$DRY_RUN" != "1" ]; then
  sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed code sha mismatch — aborting"; exit 9; }
  g1=$(sha256sum "$DS_W1" | awk '{print $1}'); g2=$(sha256sum "$DS_W2" | awk '{print $1}')
  [ "$g1" = "$DS_W1_SHA" ] || { log "FATAL: W1 dataset sha mismatch — aborting"; exit 9; }
  [ "$g2" = "$DS_W2_SHA" ] || { log "FATAL: W2 dataset sha mismatch — aborting"; exit 9; }
fi
touch "$STATE" "$DLOG"

infra_reboots=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
declare -A NONVALID
NONVALID[W1]=$(awk -F'\t' '$2=="VERDICT" && $7!="VALID" && $5=="W1" && ($6=="A"||$6=="M"){n++} END{print n+0}' "$STATE")
NONVALID[W2]=$(awk -F'\t' '$2=="VERDICT" && $7!="VALID" && $5=="W2" && ($6=="A"||$6=="M"){n++} END{print n+0}' "$STATE")

log "=== FORMAL EVAL DRIVER start (DRY_RUN=$DRY_RUN, infra_reboots_used=$infra_reboots/$INFRA_CAP, nonvalid W1=${NONVALID[W1]} W2=${NONVALID[W2]}) ==="
[ "$DRY_RUN" = "1" ] || mark CAMPAIGN_START pid=$$ infra_reboots_used=$infra_reboots

csv_tsv=$(python3 -c "
import csv,sys
for r in list(csv.reader(open('$CSV')))[1:]:
    print('\t'.join(r))
")

echo "$csv_tsv" | while IFS=$'\t' read -r seq round cond block_pos arm_pos arm controller ea_w knobs dataset bench_label artifact_dir rank0_jsonl; do
  ART=$EXP/$artifact_dir
  case "$cond" in W1) DS=$DS_W1; NREQ=4000;; W2) DS=$DS_W2; NREQ=8000;; *) log "FATAL bad condition $cond"; exit 9;; esac
  case "$arm" in A|F|L|M) ;; *) log "FATAL bad arm $arm"; exit 9;; esac
  # resume: skip finished seqs
  if awk -F'\t' -v s="$seq" '$2=="VERDICT" && $3==s{f=1} END{exit !f}' "$STATE"; then
    log "seq $seq ($bench_label) already has VERDICT — skip"; continue
  fi
  # resume: interrupted start that already produced bench.json -> offline verdict, no new GPU start
  if awk -F'\t' -v s="$seq" '$2=="START" && $3==s{f=1} END{exit !f}' "$STATE" && [ -f "$ART/bench.json" ]; then
    log "seq $seq interrupted with bench.json present — recovering offline"
    vout=$(python3 "$INTEG" --artifact-dir "$ART" --arm "$arm" --condition "$cond" --label "$bench_label" --expected-requests "$NREQ" 2>&1)
    verdict=$(echo "$vout" | sed -n 's/^VERDICT:\([A-Z_]*\) .*/\1/p' | head -1)
    ttime=$(echo "$vout" | sed -n 's/^VERDICT:[A-Z_]* t=\(.*\)/\1/p' | head -1)
    [ -z "$verdict" ] && { verdict="INTEGRITY_FAIL"; ttime="-"; log "seq $seq recovery checker error: $vout"; }
    mark RESUME_RECOVERED "$seq" "$bench_label" "$cond" "$arm" "$verdict"
    mark VERDICT "$seq" "$bench_label" "$cond" "$arm" "$verdict" t=$ttime recovered=1
    log "seq $seq RESUME_RECOVERED verdict=$verdict t=$ttime"
    if [ "$arm" = "A" ] || [ "$arm" = "M" ]; then
      if [ "$verdict" != "VALID" ]; then
        NONVALID[$cond]=$(( ${NONVALID[$cond]} + 1 ))
        if [ ${NONVALID[$cond]} -ge 2 ]; then
          log "FOR-CAUSE HALT: arm $arm >=2 non-VALID rounds in $cond — stopping per protocol §8."
          mark HALT for_cause "$seq" "$cond" "$arm"
          exit 8
        fi
      fi
    fi
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    case "$arm" in
      A) env_A; envdump="ctrl=$OEPLB_WINDOW_CONTROLLER EA_W=$EA_W knobs=FROZEN log=$OEPLB_V21_LOG";;
      M) env_M; envdump="ctrl=$OEPLB_WINDOW_CONTROLLER EA_W=$EA_W knobs=INERT log=$OEPLB_V21_LOG";;
      F|L) env_legacy "$ea_w"; envdump="ctrl=$OEPLB_WINDOW_CONTROLLER EA_W=$EA_W knobs=-";;
    esac
    [ -f "$DS" ] || { log "DRY FAIL: dataset missing $DS"; exit 9; }
    log "DRY seq=$seq r$round $cond pos$block_pos/$arm_pos arm=$arm csv_ctrl=$controller csv_EA_W=$ea_w $envdump ds=$(basename "$DS") nreq=$NREQ label=$bench_label art=$artifact_dir jsonl=$rank0_jsonl"
    continue
  fi
  mkdir -p "$ART"
  log "=== seq $seq : r$round $cond arm $arm (block_pos $block_pos, arm_pos $arm_pos) label $bench_label ==="
  mark START "$seq" "$bench_label" "$cond" "$arm" infra_reboots_used=$infra_reboots

  verdict=""; ttime="-"; attempt=1
  while :; do
    sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed sha mismatch at seq $seq — aborting campaign"; mark HALT sha_mismatch "$seq"; exit 9; }
    kill_server
    case "$arm" in
      A) env_A;; M) env_M;; F|L) env_legacy "$ea_w";;
    esac
    rm -f "$ART/v2p1_rank0.jsonl"
    rm -f "$BENCHROOT/benchmarks/results/$bench_label.json"
    SLOG=$ART/server.log
    [ $attempt -gt 1 ] && SLOG=$ART/server_reboot$((attempt-1)).log
    log "seq $seq boot start (attempt $attempt)"
    boot_t0=$(date +%s)
    setsid nohup bash "$LAUNCH" > "$SLOG" 2>&1 &
    rc=3
    for i in $(seq 1 600); do
      grep -q "ready to roll" "$SLOG" 2>/dev/null && { rc=0; break; }
      grep -qE "Traceback|CUDA out of memory|RuntimeError" "$SLOG" 2>/dev/null && { rc=2; break; }
      sleep 4
    done
    boot_s=$(( $(date +%s) - boot_t0 ))
    if [ $rc -ne 0 ]; then
      log "seq $seq BOOT FAIL rc=$rc after ${boot_s}s (zero-data INFRA)"
      kill_server
      if [ $attempt -eq 1 ] && [ $infra_reboots -lt $INFRA_CAP ]; then
        infra_reboots=$((infra_reboots+1))
        mark INFRA_REBOOT "$seq" "$bench_label" "$cond" "$arm" used=$infra_reboots/$INFRA_CAP
        log "seq $seq INFRA-R re-boot ($infra_reboots/$INFRA_CAP used)"
        attempt=$((attempt+1)); continue
      fi
      verdict="INFRA_MISSING"; ttime="-"
      mark BOOT_FAIL "$seq" "$bench_label" "$cond" "$arm" rc=$rc attempt=$attempt
      break
    fi
    log "seq $seq boot OK in ${boot_s}s — bench start"
    mark BOOT_OK "$seq" "$bench_label" "$cond" "$arm" boot_s=$boot_s attempt=$attempt
    {
      echo "seq=$seq round=$round condition=$cond block_pos=$block_pos arm_pos=$arm_pos arm=$arm"
      echo "controller=$controller ea_w_csv=$ea_w knobs_csv=$knobs label=$bench_label attempt=$attempt"
      echo "boot_s=$boot_s bench_start=$(date -Is)"
      env | grep -E '^(OEPLB_|EA_)' | sort
    } > "$ART/meta.txt"
    ( cd "$BENCHROOT/scripts" && timeout 3600 python3 run_grid_bench.py "$bench_label" "$DS" 256 ) > "$ART/bench.log" 2>&1
    brc=$?
    echo "bench_rc=$brc bench_end=$(date -Is)" >> "$ART/meta.txt"
    cp "$BENCHROOT/benchmarks/results/$bench_label.json" "$ART/bench.json" 2>/dev/null
    log "seq $seq bench rc=$brc"
    vout=$(python3 "$INTEG" --artifact-dir "$ART" --arm "$arm" --condition "$cond" --label "$bench_label" --expected-requests "$NREQ" 2>&1); irc=$?
    verdict=$(echo "$vout" | sed -n 's/^VERDICT:\([A-Z_]*\) .*/\1/p' | head -1)
    ttime=$(echo "$vout" | sed -n 's/^VERDICT:[A-Z_]* t=\(.*\)/\1/p' | head -1)
    [ -z "$verdict" ] && { verdict="INTEGRITY_FAIL"; ttime="-"; log "seq $seq integrity checker error (rc=$irc): $vout"; }
    log "seq $seq VERDICT=$verdict total_time_s=$ttime (checker rc=$irc)"
    kill_server
    break
  done

  mark VERDICT "$seq" "$bench_label" "$cond" "$arm" "$verdict" t=$ttime attempt=$attempt
  if [ "$arm" = "A" ] || [ "$arm" = "M" ]; then
    if [ "$verdict" != "VALID" ]; then
      NONVALID[$cond]=$(( ${NONVALID[$cond]} + 1 ))
      if [ ${NONVALID[$cond]} -ge 2 ]; then
        log "FOR-CAUSE HALT: arm $arm has >=2 non-VALID rounds in $cond — stopping campaign per protocol §8; report to author."
        mark HALT for_cause "$seq" "$cond" "$arm" w1_nonvalid=${NONVALID[W1]} w2_nonvalid=${NONVALID[W2]}
        exit 8
      fi
    fi
  fi
done
drc=$?
if [ "$DRY_RUN" != "1" ]; then
  if [ $drc -eq 0 ]; then
    nv=$(awk -F'\t' '$2=="VERDICT" && $7=="VALID"{n++} END{print n+0}' "$STATE")
    ir=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
    log "=== CAMPAIGN LOOP FINISHED (valid verdicts: $nv/48; INFRA re-boots used: $ir/$INFRA_CAP; see state for retired/missing) ==="
    mark CAMPAIGN_END valid=$nv infra_reboots_used=$ir
    kill_server
  else
    log "=== CAMPAIGN LOOP EXITED rc=$drc (halt or fatal; state preserved) ==="
  fi
fi
exit $drc
