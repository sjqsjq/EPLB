#!/bin/sh
# archive_finalize_xd.sh -- bulk to /data + manifest + GH curated + README. Detached-safe, logged.
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
DATA=/data/minghua/sjq/PAPER_EXPERIMENTS/cross_domain_e2e_20261027
GH=/workspace/EPLB/PAPER_EXPERIMENTS/cross_domain_e2e_20261027
LOG=$X/logs/archive_finalize.log
mkdir -p "$DATA" "$GH" "$X/logs"
echo "[archive] START $(date -u +%FT%TZ)" >> "$LOG"
echo "[archive] bulk copy ~589M to /data (slow disk) ..." >> "$LOG"
cp -r "$X"/. "$DATA"/ 2>> "$LOG"
echo "[archive] bulk rc=$? $(date -u +%FT%TZ)" >> "$LOG"
(cd "$DATA" && find . -type f ! -name ARCHIVE_SHA256SUMS.txt | sort | xargs -r sha256sum > ARCHIVE_SHA256SUMS.txt) && echo "[archive] manifest lines: $(wc -l < "$DATA/ARCHIVE_SHA256SUMS.txt") $(date -u +%FT%TZ)" >> "$LOG"
echo "[archive] curated GH side..." >> "$LOG"
mkdir -p "$GH/code_lock" "$GH/analysis" "$GH/scripts" "$GH/placements" "$GH/notes" "$GH/boots_curated" "$GH/prep_curated/record_boot2" "$GH/prep_curated/record_boot_small" "$GH/prep_curated/profile_build"
cp "$X/PROTOCOL.md" "$X/LEDGER_xd.tsv" "$X/ORDER_TABLE.json" "$GH/"
cp "$X"/code_lock/* "$GH/code_lock/" 2>/dev/null
cp "$X"/analysis/analyze_xd.py "$X"/analysis/timeline_xd.py "$X"/analysis/report_gen_xd.py "$X"/analysis/make_readme_xd.py "$X"/analysis/update_map_xd.py "$X"/analysis/summary.json "$X"/analysis/per_boot_raw.tsv "$X"/analysis/timeline_5_5.json "$X"/analysis/timeline_5_5.md "$X"/analysis/costs_profile_solve.md "$X"/analysis/REPORT_xd.md "$GH/analysis/"
cp "$X"/scripts/launch_arm_xd.sh "$X"/scripts/run_xd_boot.sh "$X"/scripts/mech_check_xd.py "$X"/scripts/queue_formal_xd.sh "$X"/scripts/launch_rec_e3_v2.sh "$X"/scripts/run_rec_continue.sh "$X"/scripts/finalize_prep_watcher.sh "$X"/scripts/finalize_watcher_xd.sh "$X"/scripts/archive_finalize_xd.sh "$GH/scripts/" 2>/dev/null
cp "$X"/placements/*.json "$GH/placements/"
cp "$X"/notes/E3_historical_crosscheck.md "$GH/notes/"
for d in "$X"/boots/I_r* "$X"/boots/ES_r* "$X"/boots/ED_r* "$X"/boots/MT_r* "$X"/boots/DF_r* "$X"/boots/P0_r*; do
  L=$(basename "$d"); mkdir -p "$GH/boots_curated/$L"
  for f in "$L.json" MECH_CHECK.json boot_meta.json boot_meta.json.end sp_run_end.json v2p1_rank0.jsonl bench.log; do
    [ -f "$d/$f" ] && cp "$d/$f" "$GH/boots_curated/$L/"
  done
done
find "$X/prep/record_boot2" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$GH/prep_curated/record_boot2/" \;
find "$X/prep/record_boot" -maxdepth 2 -type f -size -1M -exec cp {} "$GH/prep_curated/record_boot_small/" \; 2>/dev/null
cp "$X"/prep/profile_build/* "$GH/prep_curated/profile_build/" 2>/dev/null
python3 "$X/analysis/make_readme_xd.py" >> "$LOG" 2>&1
echo "[archive] readme rc=$?" >> "$LOG"
du -sh "$GH" >> "$LOG" 2>&1
echo "[archive] DONE $(date -u +%FT%TZ)" >> "$LOG"
