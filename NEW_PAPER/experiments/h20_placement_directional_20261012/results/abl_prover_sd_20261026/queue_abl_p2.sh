#!/bin/bash
# AUTHORIZATION REQUIRED — this queue occupies the GPU for ~24 boots.
# Authorization on record: author "迅速继续" 2026-10-26 (see PHASE2_PREREGISTRATION.md §0, LEDGER_abl.tsv).
# Design frozen in PHASE2_PREREGISTRATION.md §3/§4: 8 blocks x (P -> F1,16,16 -> F4,16,64), continue-on-fail, NO auto-retries.
A=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/abl_prover_sd_20261026
DS=$A/streams_p2/prover_256tok_out1_x4.jsonl
DSSHA=b312ba50c39abe1a81e85bd16f27f2b7913ae7fffb57550efb5b702fb34aabf0
{
echo "== queue_abl_p2 start $(date -u +%FT%TZ)"
for b in 1 2 3 4 5 6 7 8; do
  bash $A/run_abl_boot_p2.sh apsd2_P_b$b P "" "$DS" "$DSSHA" 8192 256 3600
  echo "== apsd2_P_b$b rc=$? $(date -u +%FT%TZ)"
  bash $A/run_abl_boot_p2.sh apsd2_F_1_16_16_b$b F "1 16 16" "$DS" "$DSSHA" 8192 256 3600
  echo "== apsd2_F_1_16_16_b$b rc=$? $(date -u +%FT%TZ)"
  bash $A/run_abl_boot_p2.sh apsd2_F_4_16_64_b$b F "4 16 64" "$DS" "$DSSHA" 8192 256 3600
  echo "== apsd2_F_4_16_64_b$b rc=$? $(date -u +%FT%TZ)"
done
echo "== queue_abl_p2 end $(date -u +%FT%TZ)"
} >> "$A/queue_abl_p2_run.log" 2>&1
