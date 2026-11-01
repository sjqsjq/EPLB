#!/bin/bash
# =====================================================================
# AUTHORIZATION REQUIRED — DO NOT RUN WITHOUT EXPLICIT AUTHOR APPROVAL
# (standing directive: no GPU launches without authorization)
# abl_prover_sd_20261026 phase-1: 2 independent boots, pre-declared
# order P -> F (PROTOCOL §2). Mechanism-exercise check ONLY; no
# performance judgment; anti-tuning clauses PROTOCOL §3 bind this run.
# continue-on-fail: a failed P boot does not skip the F boot.
# =====================================================================
A=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/abl_prover_sd_20261026
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
DSSHA=b4034e3b67ac2925134f123da706117721b7fbfbee47cbf8658f752b6d1d257a
{
  echo "== queue_abl start $(date -u +%FT%TZ)"
  bash $A/run_abl_boot.sh apsd_P_b1 P "" "$DS" "$DSSHA" 2048 256 1800
  echo "== apsd_P_b1 rc=$? $(date -u +%FT%TZ)"
  bash $A/run_abl_boot.sh apsd_F_1_16_16_b1 F "1 16 16" "$DS" "$DSSHA" 2048 256 1800
  echo "== apsd_F_1_16_16_b1 rc=$? $(date -u +%FT%TZ)"
  echo "== queue_abl end $(date -u +%FT%TZ)"
} >> "$A/queue_abl_run.log" 2>&1
