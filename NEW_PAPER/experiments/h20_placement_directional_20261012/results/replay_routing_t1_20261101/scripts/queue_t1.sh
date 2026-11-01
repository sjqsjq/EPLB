#!/bin/bash
# queue_t1.sh - T1 smoke (strict gate) + 6 formal boots, frozen order.
# PROTOCOL_t1 sec.5. rc=2 GUARD_ABORT stop; rc=1 log+continue (INFRA <=2).
set -u
T1=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/replay_routing_t1_20261101
DS=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019/B_data/streams/e3_multidomain_switch.jsonl
DSSHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
SMOKE=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030/B_data/streams/prover_512_smoke.jsonl
SMOKESHA=7d575c9c68075a52bbc2f068ccfa5dbef323ae23e84c81d56f729737af057fba
led(){ printf "%s\tqueue\t%s\n" "$(date -u +%FT%TZ)" "$1" >> /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/replay_routing_t1_20261101/LEDGER_t1.tsv; }
runs(){
  echo "[queue_t1] SMOKE $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $T1/scripts/run_t1_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
  RC=$?
  if [ $RC -ne 0 ]; then echo "[queue_t1] SMOKE_FAIL $1 rc=$RC $(date -u +%FT%TZ)"; led "SMOKE_FAIL at $1 rc=$RC"; exit $RC; fi
}
run(){
  echo "[queue_t1] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $T1/scripts/run_t1_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue_t1] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; led "GUARD_ABORT at $1 rc=2"; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue_t1] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
runs T1_SMOKE_GAP300 GAP300 "$SMOKE" "$SMOKESHA" 512 256 600
run T1_ORIG ORIG "$DS" "$DSSHA" 8192 256 900
run T1_GAP32 GAP32 "$DS" "$DSSHA" 8192 256 900
run T1_GAP128 GAP128 "$DS" "$DSSHA" 8192 256 900
run T1_GAP300 GAP300 "$DS" "$DSSHA" 8192 256 900
run T1_MD300 MD300 "$DS" "$DSSHA" 8192 256 900
run T1_BI300 BI300 "$DS" "$DSSHA" 8192 256 900
echo "[queue_t1] T1_QUEUE_END $(date -u +%FT%TZ)"
led "T1_QUEUE_END 1smoke_6formal"
