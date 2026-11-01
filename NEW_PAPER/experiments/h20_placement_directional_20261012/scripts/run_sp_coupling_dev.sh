#!/bin/bash
# =============================================================================
# run_sp_coupling_dev.sh — sp_coupling_v1 driver (plan 2026-10-18 §7).
# LOCAL ONLY: no push. Cloned from run_calibration_m1m2m3.sh (sealed campaign
# hygiene), SP-adapted.
#
# Usage: ./run_sp_coupling_dev.sh SMOKE|DEV   (DRY_RUN=1 for dry validation)
#
# Rules encoded (plan §6/§7 + standing governance):
#  - SMOKE: 3 launches (300-req subset), P0e gate; does NOT consume DEV budget.
#  - DEV: 12 TIMED launches MAX, frozen arm order (sp_arm_order.json ->
#    run_list_sp_dev.csv, sha-locked below). Failed starts never shift order.
#  - Budget: BOOT_OK = bench will run = consumes cap. INFRA zero-data boot
#    failures: ONE same-cell re-boot, INFRA cap 3, NOT consuming the cap;
#    cap exceeded -> HALT+report. RESULT failure (ok<N|errors>0 ->
#    RESULT_RETIRE): retired, never retried, consumes budget, driver CONTINUES.
#  - MECH_RETIRE / INTEGRITY_FAIL: mechanism class -> HALT immediately (exit 8).
#    Halts are NEVER triggered by endpoint values.
#  - Per-start hygiene: kill -> GPU clear -> sha256sum -c v2 lock -> env ->
#    boot (poll <=2400s) -> bench (concurrency 256, O1 dataset, all costs in
#    service time) -> artifacts -> integrity_check_sp.py -> teardown ->
#    append-only state.
#  - All runs: controller 21, planner knobs FROZEN (verbatim v21 authority),
#    EA_W=16, EA_ALPHA=0.0, adaptive OFF, probe OFF. Arms differ ONLY in
#    OEPLB_SP_MODE/SP_S/SP_W (A: adaptive seed 20261018 streak 3 recover 2).
# =============================================================================
set -u
PHASE=${1:-}
case "$PHASE" in SMOKE|DEV) ;; *) echo "usage: $0 SMOKE|DEV (DRY_RUN=1 optional)"; exit 2;; esac
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
SP=$EXP/results/sp_coupling_v1
DEPSHAS=$SP/deployed_shas_sp_coupling_v2.txt
LAUNCH=$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh
BENCHROOT=/workspace/EPLB/OEPLB
INTEG=$EXP/scripts/integrity_check_sp.py
DS_W1=$EXP/inputs/fullservice_v2/workload/w1_stable_code_4000.jsonl
DS_W2=$EXP/inputs/fullservice_v2/workload/w2_switch_code_chinese_8000.jsonl
DS_SM=$EXP/inputs/fullservice_v2/workload/smoke_w1_300.jsonl
DS_W1_SHA=f0391f7cdf8edd734a09c65b331d751052b8b1937a0416b3782a409039fe8578
DS_W2_SHA=9d852a756957d3db805fd876ebbad51a0224b9b6c23d23edb5646cdf260362f8
DS_SM_SHA=073f79c2b0eb793f1f927cc81c54454ddda235080beddee0290a8a5749a1e7e6
DRY_RUN=${DRY_RUN:-0}
INFRA_CAP=3
if [ "$PHASE" = "SMOKE" ]; then
  CSV=$SP/run_list_sp_smoke.csv
  CSV_SHA_LOCKED=0700617dc61238f8a94dcb108d70809f992e7fd9b46b040ae6e6b83e643d6a54
  NROWS_EXPECT=3; TIMED_CAP=3; STATE=$SP/smoke_state.tsv; DLOG=$SP/smoke_driver.log
  BENCH_TIMEOUT=1200
else
  CSV=$SP/run_list_sp_dev.csv
  CSV_SHA_LOCKED=d5ba4739e90e94ab9d337724b5ff9e561538b69ce3721f2948176438b17dcb29
  NROWS_EXPECT=12; TIMED_CAP=12; STATE=$SP/dev_state.tsv; DLOG=$SP/dev_driver.log
  BENCH_TIMEOUT=3600
fi

log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }
mark(){ local f; printf '%s' "$(date -Is)" >> "$STATE"; for f in "$@"; do printf '\t%s' "$f"; done >> "$STATE"; printf '\n' >> "$STATE"; }
gpu_clear_wait(){ for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && return 0; sleep 4; done; return 1; }
kill_server(){ pkill -9 -f "sglang.launch_server" >/dev/null 2>&1; pkill -9 -f "sglang::" >/dev/null 2>&1; sleep 5; gpu_clear_wait || log "WARN gpus not clear"; }

export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8

# ---- env: shared frozen base (planner + budgets UNCHANGED; plan §2/§6) ----
env_common(){
  export EA_ALPHA=0.0
  unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
  export OEPLB_WINDOW_CONTROLLER=21
  # FROZEN fitted knobs (FREEZE_record_v2p1.md; do not edit)
  export OEPLB_V21_KAPPA=3875.1768798828125
  export OEPLB_V21_Z_LO=22.643858727592757 OEPLB_V21_Z_HI=314.14010272527486 OEPLB_V21_Z_URGENT=942.4203081758246
  export OEPLB_V21_THETA_R_LOW=0.010095552687949321 OEPLB_V21_THETA_R_ACT=0.011395643112507271
  export OEPLB_V21_N_MIN=12320768
  export OEPLB_V21_M_SHRINK=2 OEPLB_V21_M_GROW=3
  export OEPLB_V21_DWELL_SHRINK=2 OEPLB_V21_DWELL_GROW=2
  export OEPLB_V21_U_MAX=4 OEPLB_V21_W_UNCERTAIN_TARGET=16
  export OEPLB_V21_URGENT_EPISODE_CAP=3 OEPLB_V21_BETA=0.5
  export OEPLB_V21_LOG=$ART/v2p1_rank0.jsonl
  export EA_W=16
  export OEPLB_PROBE_MODE=off
}
# env_arm ARM: the ONLY difference between arms is the SP block (plan §6:
# fixed arms keep online planning/migration and share ALL infra).
env_arm(){
  env_common
  case "$1" in
    F_dense)  export OEPLB_SP_MODE=fixed OEPLB_SP_S=1 OEPLB_SP_W=16;;
    F_sparse) export OEPLB_SP_MODE=fixed OEPLB_SP_S=4 OEPLB_SP_W=64;;
    A)        export OEPLB_SP_MODE=adaptive OEPLB_SP_S=1 OEPLB_SP_W=16 \
                     OEPLB_SP_SEED=20261018 OEPLB_SP_QUIET_STREAK=3 \
                     OEPLB_SP_RECOVER_MIN_CYCLES=2;;
    *) log "FATAL bad arm $1"; exit 9;;
  esac
}

# ---- preflight locks ----
got=$(sha256sum "$CSV" | awk '{print $1}')
[ "$got" = "$CSV_SHA_LOCKED" ] || { log "FATAL: run-list sha mismatch ($got) — aborting"; exit 9; }
python3 -m py_compile "$INTEG" || { log "FATAL: sp integrity checker does not compile"; exit 9; }
nrows=$(python3 -c "import csv;print(sum(1 for _ in csv.reader(open('$CSV')))-1)")
[ "$nrows" = "$NROWS_EXPECT" ] || { log "FATAL: run list has $nrows rows, expected $NROWS_EXPECT"; exit 9; }
nuniq=$(python3 -c "import csv;rows=list(csv.reader(open('$CSV')))[1:];print(len(set(r[5] for r in rows)))")
[ "$nuniq" = "$NROWS_EXPECT" ] || { log "FATAL: bench labels not unique ($nuniq/$NROWS_EXPECT)"; exit 9; }
if [ "$DRY_RUN" != "1" ]; then
  [ -f "$DEPSHAS" ] || { log "FATAL: missing sp sha lock $DEPSHAS"; exit 9; }
  sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed code sha mismatch — aborting"; exit 9; }
  g1=$(sha256sum "$DS_W1" | awk '{print $1}'); g2=$(sha256sum "$DS_W2" | awk '{print $1}'); gs=$(sha256sum "$DS_SM" | awk '{print $1}')
  [ "$g1" = "$DS_W1_SHA" ] || { log "FATAL: W1 dataset sha mismatch — aborting"; exit 9; }
  [ "$g2" = "$DS_W2_SHA" ] || { log "FATAL: W2 dataset sha mismatch — aborting"; exit 9; }
  [ "$gs" = "$DS_SM_SHA" ] || { log "FATAL: SMOKE dataset sha mismatch — aborting"; exit 9; }
fi
mkdir -p "$SP"
touch "$STATE" "$DLOG"

infra_reboots=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
timed_starts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")

log "=== SP DRIVER phase=$PHASE start (DRY_RUN=$DRY_RUN, timed_starts_used=$timed_starts/$TIMED_CAP, infra_reboots_used=$infra_reboots/$INFRA_CAP) ==="
[ "$DRY_RUN" = "1" ] || mark PHASE_START "$PHASE" pid=$$ timed_used=$timed_starts infra_used=$infra_reboots

csv_tsv=$(python3 -c "
import csv,sys
for r in list(csv.reader(open('$CSV')))[1:]:
    print('\t'.join(r))
")

echo "$csv_tsv" | while IFS=$'\t' read -r seq phase cond block arm bench_label artifact_dir; do
  ART=$EXP/$artifact_dir
  case "$cond" in
    W1) DS=$DS_W1; NREQ=4000;;
    W2) DS=$DS_W2; NREQ=8000;;
    SMOKE) DS=$DS_SM; NREQ=300;;
    *) log "FATAL bad condition $cond"; exit 9;;
  esac
  case "$arm" in F_dense|F_sparse|A) ;; *) log "FATAL bad arm $arm"; exit 9;; esac
  if awk -F'\t' -v s="$seq" '$2=="VERDICT" && $3==s{f=1} END{exit !f}' "$STATE"; then
    log "seq $seq ($bench_label) already has VERDICT — skip"; continue
  fi
  # resume: interrupted start that already produced bench.json -> offline verdict
  if awk -F'\t' -v s="$seq" '$2=="START" && $3==s{f=1} END{exit !f}' "$STATE" && [ -f "$ART/bench.json" ]; then
    log "seq $seq interrupted with bench.json present — recovering offline"
    vout=$(python3 "$INTEG" --artifact-dir "$ART" --arm "$arm" --condition "$cond" --label "$bench_label" --expected-requests "$NREQ" 2>&1)
    verdict=$(echo "$vout" | sed -n 's/^VERDICT:\([A-Z_]*\) .*/\1/p' | head -1)
    ttime=$(echo "$vout" | sed -n 's/^VERDICT:[A-Z_]* t=\(.*\)/\1/p' | head -1)
    [ -z "$verdict" ] && { verdict="INTEGRITY_FAIL"; ttime="-"; log "seq $seq recovery checker error: $vout"; }
    mark RESUME_RECOVERED "$seq" "$bench_label" "$cond" "$arm" "$verdict"
    mark VERDICT "$seq" "$bench_label" "$cond" "$arm" "$verdict" t=$ttime recovered=1
    log "seq $seq RESUME_RECOVERED verdict=$verdict t=$ttime"
    case "$verdict" in MECH_RETIRE|INTEGRITY_FAIL) log "MECHANISM-CLASS FAILURE at seq $seq — HALT per plan §7/§8"; mark HALT mechanism "$seq" "$verdict"; exit 8;; esac
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    env_arm "$arm"
    [ -f "$DS" ] || { log "DRY FAIL: dataset missing $DS"; exit 9; }
    log "DRY seq=$seq phase=$phase cond=$cond block=$block arm=$arm SP_MODE=$OEPLB_SP_MODE SP_S=$OEPLB_SP_S SP_W=$OEPLB_SP_W ctrl=$OEPLB_WINDOW_CONTROLLER EA_W=$EA_W ALPHA=$EA_ALPHA probe=$OEPLB_PROBE_MODE knobs=FROZEN log=$OEPLB_V21_LOG ds=$(basename "$DS") nreq=$NREQ label=$bench_label art=$artifact_dir"
    continue
  fi
  timed_starts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")
  if [ "$timed_starts" -ge "$TIMED_CAP" ]; then
    log "BUDGET EXHAUSTED: timed_starts=$timed_starts/$TIMED_CAP — refusing seq $seq"
    mark BUDGET_HALT "$seq" "$bench_label" timed=$timed_starts
    exit 7
  fi
  mkdir -p "$ART"
  log "=== seq $seq : phase=$phase cond=$cond block=$block arm=$arm label=$bench_label ==="
  mark START "$seq" "$bench_label" "$cond" "$arm" infra_used=$infra_reboots timed_used=$timed_starts

  verdict=""; ttime="-"; attempt=1
  while :; do
    sha256sum -c --quiet "$DEPSHAS" || { log "FATAL: deployed sha mismatch at seq $seq — aborting"; mark HALT sha_mismatch "$seq"; exit 9; }
    kill_server
    env_arm "$arm"
    rm -f "$ART/v2p1_rank0.jsonl" "$ART/sp_run_end.json" "$ART/probe_run_end.json" "$ART/verdict.json"
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
        log "seq $seq INFRA re-boot ($infra_reboots/$INFRA_CAP used; does NOT consume the cap)"
        attempt=$((attempt+1)); continue
      fi
      if [ $infra_reboots -ge $INFRA_CAP ]; then
        log "INFRA CAP EXCEEDED ($infra_reboots/$INFRA_CAP) — HALT and report"
        mark HALT infra_cap "$seq"; exit 7
      fi
      verdict="INFRA_MISSING"; ttime="-"
      mark BOOT_FAIL "$seq" "$bench_label" "$cond" "$arm" rc=$rc attempt=$attempt
      break
    fi
    log "seq $seq boot OK in ${boot_s}s — bench start (timed start #$((timed_starts+1))/$TIMED_CAP)"
    mark BOOT_OK "$seq" "$bench_label" "$cond" "$arm" boot_s=$boot_s attempt=$attempt
    {
      echo "seq=$seq phase=$phase condition=$cond block=$block arm=$arm"
      echo "label=$bench_label attempt=$attempt boot_s=$boot_s bench_start=$(date -Is)"
      env | grep -E '^(OEPLB_|EA_)' | sort
    } > "$ART/meta.txt"
    ( cd "$BENCHROOT/scripts" && timeout $BENCH_TIMEOUT python3 run_grid_bench.py "$bench_label" "$DS" 256 ) > "$ART/bench.log" 2>&1
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
  case "$verdict" in
    MECH_RETIRE|INTEGRITY_FAIL)
      log "MECHANISM-CLASS FAILURE at seq $seq ($verdict) — HALT per plan §7/§8 (mechanism only; never endpoint values); report to author"
      mark HALT mechanism "$seq" "$verdict"
      exit 8;;
    RESULT_RETIRE|INFRA_MISSING)
      log "seq $seq retired ($verdict) — never retried; driver continues (shortfall reported at phase end)";;
  esac
done
drc=$?
if [ "$DRY_RUN" != "1" ]; then
  if [ $drc -eq 0 ]; then
    nv=$(awk -F'\t' '$2=="VERDICT" && $7=="VALID"{n++} END{print n+0}' "$STATE")
    ts=$(awk -F'\t' '$2=="BOOT_OK"{n++} END{print n+0}' "$STATE")
    ir=$(awk -F'\t' '$2=="INFRA_REBOOT"{n++} END{print n+0}' "$STATE")
    log "=== PHASE $PHASE LOOP FINISHED (VALID total: $nv; timed starts used: $ts/$TIMED_CAP; INFRA re-boots: $ir/$INFRA_CAP) ==="
    mark PHASE_END "$PHASE" valid_total=$nv timed_used=$ts infra_used=$ir
  else
    log "=== PHASE $PHASE LOOP EXITED rc=$drc (halt or fatal; state preserved) ==="
  fi
fi
exit $drc
