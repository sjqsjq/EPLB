#!/bin/bash
# recovery_admission_e2e_20261027 queue: prep1 = RG GPU validation boot (author order: new-version validation FIRST)
# AUTHORIZATION: author task directive 2026-10-26 (prep <=3, formal 4x7=28,
# zero-data INFRA restarts <=2, no extra rounds based on speed).
# Launch detached:
#   cd /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027/scripts && setsid nohup bash queue_prep1.sh > ../boots/queue_prep1.sh.log 2>&1 &
# rc=2 (guard abort) stops the queue: systemic failure, no auto-retry per PROTOCOL sec.8.
# rc=1 (ready/bench fail) is logged and the queue continues; failed boots are triaged
# by the operator within the INFRA restart budget (never silently dropped).
A2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027
DS=$A2/streams/prover_256tok_out1_x4.jsonl
DSSHA=b312ba50c39abe1a81e85bd16f27f2b7913ae7fffb57550efb5b702fb34aabf0
run(){ L=$1; A=$2; P=$3
  echo "[$(date -u +%FT%TZ)] BOOT $L arm=$A params=$P"
  bash $A2/scripts/run_rge2e_boot.sh "$L" "$A" "$P" "$DS" "$DSSHA" 8192 256 3600
  rc=$?
  echo "[$(date -u +%FT%TZ)] $L rc=$rc"
  if [ $rc -eq 2 ]; then echo "[$(date -u +%FT%TZ)] GUARD_ABORT rc=2 stopping queue"; exit 2; fi
}
run rgv_P1 P 1
echo "[$(date -u +%FT%TZ)] QUEUE_DONE prep1"
