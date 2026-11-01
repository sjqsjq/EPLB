#!/bin/bash
# =============================================================================
# run_calibration_m1m2m3.sh — Route-B probe calibration driver (M1/M2/M3,
# 15 timed launches MAX). Frozen protocol: PRE_REG_calibration_m1m2m3.md +
# DRAFT_window_admission_method_v1_1.md (spec v1.1-calib). LOCAL ONLY: no push.
#
# Usage: ./run_calibration_m1m2m3.sh M1|M2|M3   (DRY_RUN=1 for dry validation)
#
# Rules encoded (PRE_REG §B/§C/§D):
#  - Frozen seq order; failed starts never shift order; stage argument selects
#    the rows of that stage; rows with a VERDICT are skipped (resume).
#  - Budget: 15 TIMED starts (BOOT_OK = bench will run). Refuses to start
#    beyond the cap. INFRA zero-data boot failures: ONE same-cell re-boot,
#    campaign INFRA cap 3, NOT consuming the 15; cap exceeded -> HALT+report.
#  - RESULT failure (ok<N|errors>0 -> RESULT_RETIRE): retired, never retried,
#    consumes budget, driver CONTINUES (shortfall reported at stage gate).
#  - MECH_RETIRE / INTEGRITY_FAIL: mechanism-class -> HALT immediately (exit 8),
#    report to author. Halts are NEVER triggered by endpoint values.
#  - Per-start hygiene identical to the sealed campaign driver: kill -> GPU
#    clear -> sha256sum -c calib lock -> env -> boot (poll <=2400s) -> bench
#    (timeout 3600, concurrency 256) -> artifacts -> integrity_check.py
#    (--arm A, UNMODIFIED checker) -> teardown -> append-only state.
#  - All runs: controller 21, A-arm frozen knobs (verbatim), EA_W=16,
#    EA_ALPHA=0.0, OEPLB_PROBE_COMMIT=0, OEPLB_PROBE_CALIB_TRIGGER=1.
#  - Stage gates (M1->M2, M2->M3) are evaluated OUTSIDE this driver by the
#    gate scripts (mechanism fields only); this driver only logs facts.
# =============================================================================
set -u
STAGE=${1:-}
case "$STAGE" in M1|M2|M3) ;; *) echo "usage: $0 M1|M2|M3 (DRY_RUN=1 optional)"; exit 2;; esac
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
NV=$EXP/results/adw_v2p1/next_version
CAL=$EXP/results/adw_v2p1/calibration_m1m2m3
DEPSHAS=$NV/deployed_shas_calib_m1m2m3.txt
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
BENCHROOT=/workspace/EPLB/OEPLB
CSV=$CAL/run_list_calibration.csv
CSV_SHA_LOCKED=506ae95ebc36828894bc03558cfd9fb0d734945dbf01f4ffb4b1351be1b18ffa
INTEG=$EXP/scripts/integrity_check.py
STATE=$CAL/calib_state.tsv
DLOG=$CAL/driver.log
DRY_RUN=${DRY_RUN:-0}
INFRA_CAP=3
TIMED_CAP=15
DS_W1=$EXP/inputs/fullservice_v2/workload/w1_stable_code_4000.jsonl
DS_W2=$EXP/inputs/fullservice_v2/workload/w2_switch_code_chinese_8000.jsonl
DS_W1_SHA=f0391f7cdf8edd734a09c65b331d751052b8b1937a0416b3782a409039fe8578
DS_W2_SHA=9d852a756957d3db805fd876ebbad51a0224b9b6c23d23edb5646cdf260362f8

log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }
mark(){ local f; printf '%s' "$(date -Is)" >> "$STATE"; for f in "$@"; do printf '\t%s' "$f"; done >> "$STATE"; printf '\n' >> "$STATE"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }

export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8

# ---- env: A-arm frozen knobs verbatim (authority: run_formal_eval.sh env_A) ----
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
# env_calib MODE EPS_RISK_US FORCE_ABORT_AT
env_calib(){
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
  # Route-B probe (PRE_REG table A; frozen)
  export OEPLB_PROBE_MODE=$1
  export OEPLB_PROBE_R=12 OEPLB_PROBE_NREF=4 OEPLB_PROBE_NCAND=3 OEPLB_PROBE_NFB=4
  export OEPLB_PROBE_EPS_RISK_US=$2
  export OEPLB_PROBE_COMMIT=0
  export OEPLB_PROBE_FORCE_ABORT_AT=$3
  export OEPLB_PROBE_CALIB_TRIGGER=1
  export OEPLB_PROBE_EPS_PRAC_FRAC=0.01
}

# ---- preflight locks ----
got=$(sha256sum "$CSV" | awk '{print $1}')
[ "$got" = "$CSV_SHA_LOCKED" ] || { log "FATAL: run-list sha mismatch ($got) — aborting"; exit 9; }
python3 -m py_compile "$INTEG" || { log "FATAL: integrity checker does not compile"; exit 9; }
nrows=$(python3 -c "import csv;print(sum(1 for _ in csv.reader(open('$CSV')))-1)")
[ "$nrows" = "15" ] || { log "FATAL: run list has $nrows rows, expected 15"; exit 9; }
nuniq=$(python3 -c "import csv;rows=list(csv.reader(open('$CSV')))[1:];print(len(set(r[6] for r in rows)))")
[ "$nuniq" = "15" ] || { log "FATAL: bench labels not unique ($nuniq/15)"; exit 9; }
if [ "$DRY_RUN" != "1" ]; then
  [ -f "$DEPSHAS" ] || { log "FATAL: missing calib sha lock $DEPSHAS"; exit 9; }
  sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed code sha mismatch — aborting"; exit 9; }
  g1=$(sha256sum "$DS_W1" | awk '{print $1}'); g2=$(sha256sum "$DS_W2" | awk '{print $1}')
  [ "$g1" = "$DS_W1_SHA" ] || { log "FATAL: W1 dataset sha mismatch — aborting"; exit 9; }
  [ "$g2" = "$DS_W2_SHA" ] || { log "FATAL: W2 dataset sha mismatch — aborting"; exit 9; }
fi
mkdir -p "$CAL"
touch "$STATE" "$DLOG"

infra_reboots=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
timed_starts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")

log "=== CALIBRATION DRIVER stage=$STAGE start (DRY_RUN=$DRY_RUN, timed_starts_used=$timed_starts/$TIMED_CAP, infra_reboots_used=$infra_reboots/$INFRA_CAP) ==="
[ "$DRY_RUN" = "1" ] || mark STAGE_START "$STAGE" pid=$$ timed_used=$timed_starts infra_used=$infra_reboots

csv_tsv=$(python3 -c "
import csv,sys
for r in list(csv.reader(open('$CSV')))[1:]:
    print('\t'.join(r))
")

echo "$csv_tsv" | while IFS=$'\t' read -r seq stage cond mode eps_risk force_abort bench_label artifact_dir; do
  [ "$stage" = "$STAGE" ] || continue
  ART=$EXP/$artifact_dir
  case "$cond" in W1) DS=$DS_W1; NREQ=4000;; W2) DS=$DS_W2; NREQ=8000;; *) log "FATAL bad condition $cond"; exit 9;; esac
  case "$mode" in off|gate|aaa) ;; *) log "FATAL bad mode $mode"; exit 9;; esac
  if awk -F'\t' -v s="$seq" '$2=="VERDICT" && $3==s{f=1} END{exit !f}' "$STATE"; then
    log "seq $seq ($bench_label) already has VERDICT — skip"; continue
  fi
  # resume: interrupted start that already produced bench.json -> offline verdict
  if awk -F'\t' -v s="$seq" '$2=="START" && $3==s{f=1} END{exit !f}' "$STATE" && [ -f "$ART/bench.json" ]; then
    log "seq $seq interrupted with bench.json present — recovering offline"
    vout=$(python3 "$INTEG" --artifact-dir "$ART" --arm A --condition "$cond" --label "$bench_label" --expected-requests "$NREQ" 2>&1)
    verdict=$(echo "$vout" | sed -n 's/^VERDICT:\([A-Z_]*\) .*/\1/p' | head -1)
    ttime=$(echo "$vout" | sed -n 's/^VERDICT:[A-Z_]* t=\(.*\)/\1/p' | head -1)
    [ -z "$verdict" ] && { verdict="INTEGRITY_FAIL"; ttime="-"; log "seq $seq recovery checker error: $vout"; }
    mark RESUME_RECOVERED "$seq" "$bench_label" "$cond" "$mode" "$verdict"
    mark VERDICT "$seq" "$bench_label" "$cond" "$mode" "$verdict" t=$ttime recovered=1
    log "seq $seq RESUME_RECOVERED verdict=$verdict t=$ttime"
    case "$verdict" in MECH_RETIRE|INTEGRITY_FAIL) log "MECHANISM-CLASS FAILURE at seq $seq — HALT per PRE_REG §C"; mark HALT mechanism "$seq" "$verdict"; exit 8;; esac
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    env_calib "$mode" "$eps_risk" "$force_abort"
    [ -f "$DS" ] || { log "DRY FAIL: dataset missing $DS"; exit 9; }
    log "DRY seq=$seq stage=$stage cond=$cond mode=$mode eps_risk_us=$OEPLB_PROBE_EPS_RISK_US force_abort_at=$OEPLB_PROBE_FORCE_ABORT_AT ctrl=$OEPLB_WINDOW_CONTROLLER EA_W=$EA_W commit=$OEPLB_PROBE_COMMIT calib_trigger=$OEPLB_PROBE_CALIB_TRIGGER knobs=FROZEN log=$OEPLB_V21_LOG ds=$(basename "$DS") nreq=$NREQ label=$bench_label art=$artifact_dir"
    continue
  fi
  # budget check (timed starts)
  timed_starts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")
  if [ "$timed_starts" -ge "$TIMED_CAP" ]; then
    log "BUDGET EXHAUSTED: timed_starts=$timed_starts/$TIMED_CAP — refusing seq $seq; report shortfall"
    mark BUDGET_HALT "$seq" "$bench_label" timed=$timed_starts
    exit 7
  fi
  mkdir -p "$ART"
  log "=== seq $seq : stage=$stage cond=$cond mode=$mode eps_risk_us=$eps_risk force_abort_at=$force_abort label=$bench_label ==="
  mark START "$seq" "$bench_label" "$cond" "$mode" infra_used=$infra_reboots timed_used=$timed_starts

  verdict=""; ttime="-"; attempt=1
  while :; do
    sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed sha mismatch at seq $seq — aborting"; mark HALT sha_mismatch "$seq"; exit 9; }
    kill_server
    env_calib "$mode" "$eps_risk" "$force_abort"
    rm -f "$ART/v2p1_rank0.jsonl" "$ART/probe_run_end.json"
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
        mark INFRA_REBOOT "$seq" "$bench_label" "$cond" "$mode" used=$infra_reboots/$INFRA_CAP
        log "seq $seq INFRA re-boot ($infra_reboots/$INFRA_CAP used; does NOT consume the 15)"
        attempt=$((attempt+1)); continue
      fi
      if [ $infra_reboots -ge $INFRA_CAP ]; then
        log "INFRA CAP EXCEEDED ($infra_reboots/$INFRA_CAP) — HALT and report per PRE_REG §D"
        mark HALT infra_cap "$seq"; exit 7
      fi
      verdict="INFRA_MISSING"; ttime="-"
      mark BOOT_FAIL "$seq" "$bench_label" "$cond" "$mode" rc=$rc attempt=$attempt
      break
    fi
    log "seq $seq boot OK in ${boot_s}s — bench start (timed start #$((timed_starts+1))/$TIMED_CAP)"
    mark BOOT_OK "$seq" "$bench_label" "$cond" "$mode" boot_s=$boot_s attempt=$attempt
    {
      echo "seq=$seq stage=$stage condition=$cond mode=$mode eps_risk_us=$eps_risk force_abort_at=$force_abort"
      echo "label=$bench_label attempt=$attempt boot_s=$boot_s bench_start=$(date -Is)"
      env | grep -E '^(OEPLB_|EA_)' | sort
    } > "$ART/meta.txt"
    ( cd "$BENCHROOT/scripts" && timeout 3600 python3 run_grid_bench.py "$bench_label" "$DS" 256 ) > "$ART/bench.log" 2>&1
    brc=$?
    echo "bench_rc=$brc bench_end=$(date -Is)" >> "$ART/meta.txt"
    cp "$BENCHROOT/benchmarks/results/$bench_label.json" "$ART/bench.json" 2>/dev/null
    log "seq $seq bench rc=$brc"
    vout=$(python3 "$INTEG" --artifact-dir "$ART" --arm A --condition "$cond" --label "$bench_label" --expected-requests "$NREQ" 2>&1); irc=$?
    verdict=$(echo "$vout" | sed -n 's/^VERDICT:\([A-Z_]*\) .*/\1/p' | head -1)
    ttime=$(echo "$vout" | sed -n 's/^VERDICT:[A-Z_]* t=\(.*\)/\1/p' | head -1)
    [ -z "$verdict" ] && { verdict="INTEGRITY_FAIL"; ttime="-"; log "seq $seq integrity checker error (rc=$irc): $vout"; }
    log "seq $seq VERDICT=$verdict total_time_s=$ttime (checker rc=$irc)"
    kill_server
    break
  done

  mark VERDICT "$seq" "$bench_label" "$cond" "$mode" "$verdict" t=$ttime attempt=$attempt
  case "$verdict" in
    MECH_RETIRE|INTEGRITY_FAIL)
      log "MECHANISM-CLASS FAILURE at seq $seq ($verdict) — HALT per PRE_REG §C (mechanism only; never endpoint values); report to author"
      mark HALT mechanism "$seq" "$verdict"
      exit 8;;
    RESULT_RETIRE|INFRA_MISSING)
      log "seq $seq retired ($verdict) — never retried; continues per PRE_REG §D (shortfall reported at stage gate)";;
  esac
done
drc=$?
if [ "$DRY_RUN" != "1" ]; then
  if [ $drc -eq 0 ]; then
    nv=$(awk -F'\t' '$2=="VERDICT" && $7=="VALID"{n++} END{print n+0}' "$STATE")
    ts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")
    ir=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
    log "=== STAGE $STAGE LOOP FINISHED (VALID total: $nv; timed starts used: $ts/$TIMED_CAP; INFRA re-boots: $ir/$INFRA_CAP) — run the stage gate before the next stage ==="
    mark STAGE_END "$STAGE" valid_total=$nv timed_used=$ts infra_used=$ir
  else
    log "=== STAGE $STAGE LOOP EXITED rc=$drc (halt or fatal; state preserved) ==="
  fi
fi
exit $drc
