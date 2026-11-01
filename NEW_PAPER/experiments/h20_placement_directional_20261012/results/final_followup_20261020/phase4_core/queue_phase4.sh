#!/bin/bash
# Phase-4 64-boot core supplement queue (formal ledger, budget 64 of <=104).
# Matrix: supplement_matrix_locked.csv (sha256 d3ff2b09a27a683378b356e9a818a2f209c93af708147e8f31607c3b6044d71e, verified at start; NEVER edited).
# Executed code: frozen v2 (async_swapper 2e4652a0306cb315..., controller 4964cfc8e8af397a...) per
# PHASE4_CODE_VERSION_ADDENDUM.md (CSV code_freeze column holds the pre-round-2 v1 sha; lock kept, no backfill).
# Halt-on-fail: any FAIL (weight/mapping error or migration-class fault) => STOP package,
# return to correctness diagnosis, report blocker (fix rounds 2/2 exhausted; NO third round).
# Resume-safe: rows already PASS in the ledger are skipped, never re-run (no budget waste).
set -u
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
CSV=$G/phase4_core/supplement_matrix_locked.csv
L=$G/phase4_core/LEDGER_phase4.tsv
CSV_SHA=$(sha256sum $CSV | cut -d' ' -f1)
[ "$CSV_SHA" = "d3ff2b09a27a683378b356e9a818a2f209c93af708147e8f31607c3b6044d71e" ] || { echo "ABORT csv_sha_mismatch $CSV_SHA"; exit 2; }
[ -f $L ] || echo -e "seq\tlabel\tstart_utc\tend_utc\trc\tverdict" > $L
grep -q "^# QUEUE_START" $L || echo "# QUEUE_START v2 async=2e4652a0306cb315 controller=4964cfc8e8af397a csv=$CSV_SHA start=$(date -u +%FT%TZ)" >> $L
tail -n +2 $CSV | while IFS=, read -r SEQ PHASE BLOCK STREAM ARM PARAMS LABEL DSPATH DSSHA NREQ CONC BT _CF; do
  if grep -qP "^$SEQ\t$LABEL\t.*\t0\tPASS$" $L; then echo "[skip] $LABEL already PASS"; continue; fi
  case $ARM in
    P|F) MODE=pb;  [ $STREAM = E3 ] && MS=4 || MS=6 ;;
    *)   MODE=plain; MS=0 ;;
  esac
  S=$(date -u +%FT%TZ)
  bash $G/phase4_core/run_supp_boot.sh "$LABEL" "$ARM" "$PARAMS" "$DSPATH" "$DSSHA" "$NREQ" "$CONC" "$BT" "$MODE" "$MS"
  RC=$?
  [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$SEQ\t$LABEL\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  if [ $V != PASS ]; then
    echo -e "-\tHALTED_AT_$LABEL\t-\t$(date -u +%FT%TZ)\t$RC\tSTOP_PACKAGE" >> $L
    echo "HALTED at $LABEL rc=$RC — stop package, correctness diagnosis, blocker report (fix rounds exhausted)"
    exit 1
  fi
  echo "[done] seq=$SEQ $LABEL PASS $(date -u +%FT%TZ)"
done
RC=$?
[ $RC -eq 0 ] && echo -e "-\tPHASE4_ALL\t-\t$(date -u +%FT%TZ)\t0\tQUEUE_PASS" >> $L
exit $RC
