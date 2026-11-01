#!/bin/bash
# archive_finalize.sh -- run AFTER 28/28 formal boots + final analysis.
# 1) bulk copy of the whole campaign dir to /data (detached-safe, timeout guarded)
# 2) SHA-256 manifest of the bulk copy
# 3) curated GitHub-side archive under /workspace/EPLB/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/
set -u
A2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027
DATA=/data/minghua/sjq/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027
GH=/workspace/EPLB/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027
mkdir -p "$DATA" "$GH"
echo "[archive] bulk copy to $DATA (streams + boots + all)"
cp -r "$A2"/. "$DATA"/ 2>/dev/null
echo "[archive] bulk copy rc=$? ; manifest..."
(cd "$DATA" && find . -type f | sort | xargs -r sha256sum > ARCHIVE_SHA256SUMS.txt) && echo "[archive] manifest lines: $(wc -l < "$DATA/ARCHIVE_SHA256SUMS.txt")"
echo "[archive] curated GH side..."
mkdir -p "$GH"/{code_lock,order,analysis,tests_rgv,scripts,placements,boots_curated}
cp "$A2/PROTOCOL.md" "$A2/LEDGER_rge2e.tsv" "$GH"/
cp "$A2"/code_lock/* "$GH/code_lock"/
cp "$A2"/order/ORDER_TABLE.json "$A2"/order/ORDER_TABLE.md "$GH/order"/ 2>/dev/null
cp "$A2"/analysis/analyze_rge2e.py "$A2"/analysis/report_gen.py "$A2"/analysis/summary.json "$A2"/analysis/report.md "$A2"/analysis/per_boot_raw.tsv "$GH/analysis"/ 2>/dev/null
cp "$A2"/tests_rgv/test_branches.py "$A2"/tests_rgv/replay_p_b1.py "$A2"/tests_rgv/branch_results.json "$A2"/tests_rgv/replay_results.json "$GH/tests_rgv"/ 2>/dev/null
cp "$A2"/scripts/launch_arm_rge2e.sh "$A2"/scripts/run_rge2e_boot.sh "$A2"/scripts/mech_check_rge2e.py "$A2"/scripts/queue_prep1.sh "$A2"/scripts/queue_prep23.sh "$A2"/scripts/queue_formal.sh "$A2"/scripts/archive_finalize.sh "$GH/scripts"/
cp "$A2"/placements/*.json "$GH/placements"/
for d in "$A2"/boots/rge2e_* "$A2"/boots/rgv_*; do
  L=$(basename "$d"); mkdir -p "$GH/boots_curated/$L"
  for f in "$L.json" MECH_CHECK.json boot_meta.json boot_meta.json.end sp_run_end.json v2p1_rank0.jsonl; do
    [ -f "$d/$f" ] && cp "$d/$f" "$GH/boots_curated/$L/"
  done
  [ -f "$d/bench.log" ] && cp "$d/bench.log" "$GH/boots_curated/$L/"
done
du -sh "$GH"
echo "[archive] done; README.md to be written next (with final numbers + /data pointer)"
