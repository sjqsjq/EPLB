#!/bin/bash
# queue_g3_smoke.sh - 2 smoke boots (author fix1): each boot sequentially
# covers BOTH tasks x 512-row short streams at C256, per-task accounting via
# segment_meta.jsonl. SM_PF first (own facts + P2), SM_ALL second (cross-boot
# P1-P5 gate, reads SM_PF g3_facts.json). Any nonzero rc => STOP the queue;
# NO single-arm patches; any fix requires a new frozen design row in LEDGER.
# GPU authorization: granted 2026-10-30 (all experiments formally open).
set -u
G3=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030
S=$G3/B_data/streams
. $G3/code_lock/stream_shas.env
LEDGER=$G3/LEDGER_g3.tsv
log(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$LEDGER"; }
echo "[queue] SMOKE_G3_START $(date -u +%FT%TZ)"
log QUEUE SMOKE_G3_START
run(){
  L=$1
  echo "[queue] boot $L $(date -u +%FT%TZ)"
  bash $G3/scripts/run_g3_boot.sh "$@"
  RC=$?
  echo "[queue] boot $L rc=$RC $(date -u +%FT%TZ)"
  log "$L" "QUEUE rc=$RC"
  if [ $RC -ne 0 ]; then
    echo "[queue] SMOKE_FAIL at $L rc=$RC - STOP (no single-arm patches)"
    log QUEUE "SMOKE_FAIL_${L}_rc${RC}"
    exit 3
  fi
}
run SM_PF G3_PF $S/arcc_512_smoke.jsonl $SM_ARC_SMOKE 512 256 900 SM_PF_arc $S/prover_512_smoke.jsonl $SM_PROVER_SMOKE 512 SM_PF_prov
run SM_ALL G3_ALL $S/arcc_512_smoke.jsonl $SM_ARC_SMOKE 512 256 900 SM_ALL_arc $S/prover_512_smoke.jsonl $SM_PROVER_SMOKE 512 SM_ALL_prov
echo "[queue] SMOKE_G3_PASS $(date -u +%FT%TZ)"
log QUEUE SMOKE_G3_PASS
