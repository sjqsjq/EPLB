#!/bin/bash
# queue_g3_formal.sh - 16 formal boots: 2 tasks x 4 paired blocks x 2 arms,
# arm order alternating per block (PF->ALL, ALL->PF, PF->ALL, ALL->PF), exact
# order frozen in PROTOCOL_g3.md sec.4. C256, full 1024-row streams, O32 real
# generation, temp 0. Precondition: smoke gate PASS (g3_smoke_gate.json).
# rc policy: rc=2 (guard abort) => STOP (contamination risk); rc=1/3 => log
# and continue; zero-data INFRA restarts (<=2) are supervisor-judged at queue
# end, never automatic mid-queue. G3 GPU cap <= 5h hard.
set -u
G3=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030
S=$G3/B_data/streams
. $G3/code_lock/stream_shas.env
LEDGER=$G3/LEDGER_g3.tsv
log(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$LEDGER"; }
GATE=$G3/boots/SM_ALL/g3_smoke_gate.json
if [ ! -f "$GATE" ]; then
  echo "[queue] FORMAL_ABORT smoke gate file missing"
  log QUEUE FORMAL_ABORT_no_smoke_gate; exit 4
fi
GATE_OK=$(python3 -c "import json,sys; g=json.load(open(sys.argv[1])); print(1 if g.get(\"all_pass\") else 0)" "$GATE")
if [ "$GATE_OK" != "1" ]; then
  echo "[queue] FORMAL_ABORT smoke gate not PASS"
  log QUEUE FORMAL_ABORT_smoke_gate_fail; exit 4
fi
ARC=$S/arcc_1024_out32.jsonl
PRO=$S/prover_1024_out32_ie.jsonl
echo "[queue] FORMAL_G3_START $(date -u +%FT%TZ)"
log QUEUE FORMAL_G3_START_16_boots
run(){
  L=$1
  echo "[queue] boot $L $(date -u +%FT%TZ)"
  bash $G3/scripts/run_g3_boot.sh "$@"
  RC=$?
  echo "[queue] boot $L rc=$RC $(date -u +%FT%TZ)"
  log "$L" "QUEUE rc=$RC"
  if [ $RC -eq 2 ]; then
    echo "[queue] GUARD_ABORT at $L rc=2"
    log QUEUE GUARD_ABORT; exit 2
  fi
}
run G3_ARC_PF_b1   G3_PF  $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_PF_b1   "" "" 0 ""
run G3_ARC_ALL_b1  G3_ALL $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_ALL_b1  "" "" 0 ""
run G3_PROV_PF_b1  G3_PF  $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_PF_b1  "" "" 0 ""
run G3_PROV_ALL_b1 G3_ALL $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_ALL_b1 "" "" 0 ""
run G3_ARC_ALL_b2  G3_ALL $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_ALL_b2  "" "" 0 ""
run G3_ARC_PF_b2   G3_PF  $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_PF_b2   "" "" 0 ""
run G3_PROV_ALL_b2 G3_ALL $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_ALL_b2 "" "" 0 ""
run G3_PROV_PF_b2  G3_PF  $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_PF_b2  "" "" 0 ""
run G3_ARC_PF_b3   G3_PF  $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_PF_b3   "" "" 0 ""
run G3_ARC_ALL_b3  G3_ALL $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_ALL_b3  "" "" 0 ""
run G3_PROV_PF_b3  G3_PF  $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_PF_b3  "" "" 0 ""
run G3_PROV_ALL_b3 G3_ALL $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_ALL_b3 "" "" 0 ""
run G3_ARC_ALL_b4  G3_ALL $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_ALL_b4  "" "" 0 ""
run G3_ARC_PF_b4   G3_PF  $ARC $SM_ARC_FULL    1024 256 1800 G3_ARC_PF_b4   "" "" 0 ""
run G3_PROV_ALL_b4 G3_ALL $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_ALL_b4 "" "" 0 ""
run G3_PROV_PF_b4  G3_PF  $PRO $SM_PROVER_FULL 1024 256 1800 G3_PROV_PF_b4  "" "" 0 ""
echo "[queue] FORMAL_G3_COMPLETE 16 boots $(date -u +%FT%TZ)"
log QUEUE FORMAL_G3_COMPLETE_16_boots
