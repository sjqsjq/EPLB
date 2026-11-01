# EVIDENCE_E01_E12_20261025 — 原始证据清点包（只读审计产物）
Built: 2026-10-25 (UTC). Author directive: read-only inventory per checklist E01-E12; no code,
parameter, or frozen-analysis modification; no GPU launched. All new files live ONLY in this directory.

## Package layout
- E01_count_padding/ ... E12_generalization/  — per-checklist-item subpackages (E12 dir: E12_generalization)
- _COMMON_docs/ — campaign-level docs referenced by multiple E items (TRACE_INDEX, FINAL_PLAN, ERRATA, ONEPAGE, REPORT_followup, OEPLB README/outline, H20 ENVIRONMENT/PLAN/REPORT, paper.md)
- manifest.json / manifest.tsv — every registered source file: e, tag, rel, bytes, mtime, sha256, packaged (relative path inside this package), src (absolute source path), note
- build_package.py + BUILD_LOG.txt + BUILD_STATS.json — the packaging script and its full run log (retained for audit; rerunnable read-only)
- LEDGER_evidence_audit.tsv — append-only audit ledger of THIS inventory (does not touch any frozen LEDGER)
- EVIDENCE_INDEX.md — per-E FOUND/PARTIAL/MISSING verdicts with file-level pointers
- GAPS_AND_RETEST_DESIGN.md — missing fields + supplementary-test DESIGNS (design-only, NOT executed, awaiting author)

## Size policy (applied uniformly)
- Small files (per-root limits, typically <=1-10MB; phase-4 boots copied in FULL as top priority): copied into package
- Larger files: absolute source path + sha256 registered in manifest (path+hash per author instruction)
- Core dumps (~618GB across trees): path + size ONLY, NOT hashed, NOT moved — retained in place as forensic evidence
- Files above per-root hash caps (configs/inputs payloads >200MB): path + size registered, hash skipped (noted per entry)
- SOURCE_MISSING: registered honestly in manifest with note

## Version separation (author directive: 不同版本分包)
- E05_endtoend/phase4_v2_fixed (64 boots, frozen v2 race-fix shas) vs E05_endtoend/final_plan_* (old 168+32, deployed_shas v3) vs E05_endtoend/d38_L512_O1_comparison (Sep, pre/post server_args bugfix)
- E03: theory_a100_oeplb (launch scripts+data) vs a100_theory_validation_newpaper (counts+fits) — complementary copies, THEORY_VALIDATION_A100.md byte-identical (sha 02e1efca...)
- E07/E09: sp_coupling_v1 includes both v1 and REV1 revision docs; adw_v2 vs adw_v2p1 separate tags
- E02: deadzone results include pre_floor_interleave / pre_uniform_kernel variant files alongside finals
- E12: dsv4_recovered_from_transcript/ — see its README_provenance.md: git-object-derived (HEAD 5df703d6, sp_coupling_dev, 2026-10-22T18:05:34Z), NOT transcript-derived; parent-session supplement file adds old-sglang numbers with UNVERIFIED label

## Integrity verification
- manifest sha256 values are of the SOURCE files at build time; packaged copies verified byte-identical by re-hash spot checks (see LEDGER + EVIDENCE_INDEX verification section)
- Key frozen shas re-verified during audit: async_swapper 2e4652a0..., controller 4964cfc8... (guard text inside phase4 run_supp_boot.sh, packaged)

## Totals (build 2026-10-25)
- Manifest entries: 13,171 (13,157 build + 6 sharegpt supplement + 8 EMPTY_DIR registry rows)
- Files copied into package: 11,286 (~772MB); files hashed: 13,025; SOURCE_MISSING: 0; ROOT errors: 0
