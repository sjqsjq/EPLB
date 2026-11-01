#!/bin/bash
# recovery_admission_e2e_20261027 queue: formal 28 boots in LOCKED order (order/ORDER_TABLE.json, seed 20261027); never reordered by interim results
# AUTHORIZATION: author task directive 2026-10-26 (prep <=3, formal 4x7=28,
# zero-data INFRA restarts <=2, no extra rounds based on speed).
# Launch detached:
#   cd /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027/scripts && setsid nohup bash queue_formal.sh > ../boots/queue_formal.sh.log 2>&1 &
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
# ---- round 1: ES I P1 DF ED P0 MT ----
run rge2e_ES_r1 ES ""
run rge2e_I_r1 I ""
run rge2e_P1_r1 P 1
run rge2e_DF_r1 DF ""
run rge2e_ED_r1 ED ""
run rge2e_P0_r1 P 0
run rge2e_MT_r1 MT ""
# ---- round 2: I P1 DF ED P0 MT ES ----
run rge2e_I_r2 I ""
run rge2e_P1_r2 P 1
run rge2e_DF_r2 DF ""
run rge2e_ED_r2 ED ""
run rge2e_P0_r2 P 0
run rge2e_MT_r2 MT ""
run rge2e_ES_r2 ES ""
# ---- round 3: P1 DF ED P0 MT ES I ----
run rge2e_P1_r3 P 1
run rge2e_DF_r3 DF ""
run rge2e_ED_r3 ED ""
run rge2e_P0_r3 P 0
run rge2e_MT_r3 MT ""
run rge2e_ES_r3 ES ""
run rge2e_I_r3 I ""
# ---- round 4: MT P0 ED DF P1 I ES ----
run rge2e_MT_r4 MT ""
run rge2e_P0_r4 P 0
run rge2e_ED_r4 ED ""
run rge2e_DF_r4 DF ""
run rge2e_P1_r4 P 1
run rge2e_I_r4 I ""
run rge2e_ES_r4 ES ""
echo "[$(date -u +%FT%TZ)] QUEUE_DONE formal28"
