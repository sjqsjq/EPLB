# GAPS_AND_RETEST_DESIGN — 缺失字段登记 + 补测设计（全部仅设计，未执行，待作者确认）
Date: 2026-10-25. Governance: migration-repair rounds (2/<=2) exhausted — none of the designs below touch
the frozen deployment; all are offline/CPU-only or deferred pending hardware. No GPU was launched in this audit.

## A. Missing / partial fields (registered honestly)
| # | E | Gap | Status | Evidence that IS present |
|---|---|-----|--------|--------------------------|
| G1 | E03 | H800 §5.11 spin-absorption 11x end-to-end raw tables never archived | MISSING | sjq H800_* Dec-2025 TPxEP grid JSONs (5 dirs, copied) + profiler trees (629/178MB + 626MB tar, path+sha) = parameter-fit SOURCES only; formula-prediction numbers quoted in TRACE_INDEX |
| G2 | E04 | ShareGPT segmentation SCRIPT itself not found (grep sharegpt across all py/sh: only bound_curve.py comment) | MISSING(script)/FOUND(data) | 6 segmented artifacts sharegpt_*.jsonl (182MB, path+sha); adjacent builders make_segmented.py + freeze_requests.py packaged |
| G3 | E04/E12 | fig12 rt2_book_rank0/ EMPTY (book rank0 trace absent) | MISSING | other rt2 tasks complete: 282 npz hashed under E04/rt2_nine_task_npz_HASHONLY |
| G4 | E08 | Per-plan plan_time_s / migration_bytes / acceptance-trajectory fields not found as explicit per-plan records | PARTIAL | nearscan cost_benefit.json (per-single-swap 1360.8-1407.6us), ep_v1 planner_offline_quality.json, ea_v1 fivearm snapshots, eacampaign_code_lock.json |
| G5 | E12 | LEDGER_dsv4 rows 218-249 (old-sglang phase, planb M0-M5 gates, rollback_executed) — appended after last git commit | MISSING on disk | 217/249 rows git-recovered verbatim; client-host transcript extraction instructions embedded in UNRECOVERED__* files |
| G6 | E12 | EFFECT_REPORT_old_sglang.md + bench_old_{i,p}_{e2,e3,e4}.json originals deleted, not in git | MISSING originals | numbers preserved via SUPPLEMENT__old_sglang_wall_ratios_parent_context.md (parent-session quote, UNVERIFIED label) + UNRECOVERED__ notice |
| G7 | cross | Paper headline gain provenance unreconciled: original +15.5% (INVALID, server_args L1673 bug) / repro R1 +18.6% / R2 +19.4% (47.6/39.85) / README L512_O1 +16.5% (23363.5/20061.4) / outline +17.5% & 97.6%-of-oracle / paper.md +17.46% | UNRESOLVED FIELD | all constituent raw JSONs packaged (d38 dir: identity 40.1, oe 46.3, repro bl 39.8/39.9, oe 46.9/47.6, align_oe 47.3/47.9) |
| G8 | E05 | final_plan empty dirs: A_gap_table, E_p1_protocol, F_package, freeze; H20 traces/, logs/ | EMPTY (registered) | gap evidence lives in ea_v1/nearscan_v1 + A-cells; freeze evidence in FF/freeze + C_configs |
| G9 | E05/E09 | H20/configs (41GB, 48 files) + inputs (19GB) + E09 payloads >200MB: path+size registered, hash skipped per cap | PARTIAL by policy | can be hashed on demand (CPU-only, ~minutes) if author requires |
| G10 | E01 | e0 exists as logs only (e0_pilot.log/e0_smoke.log); no e0 json (e1 has csv+json+log) | PARTIAL | e1 full: e1_pilot.json (raw/) + e1_layout_effects.csv + e1_summary.json |
| G11 | E06 | GOOD integer counts are DERIVED arithmetic (good_rate x n, exact, 0/64 flags) not independent re-count from perreq | DERIVED (labeled) | all 64 perreq jsonl packaged in FULL -> independent recount possible (design D6) |

## B. Supplementary test DESIGNS (design-only; NOT executed; awaiting author go/no-go)
### D1 (G1, E03 H800) — requires H800 access; deferred
Re-run spin-absorption end-to-end campaign (11 runs, frozen controller shas, same streams) with raw-table
archival protocol (per-run bench.json + perreq + server.log + window logs into a dated tree). Fallback if no
H800: paper §5.11 cites formula prediction + Dec-2025 profiler sources only, with explicit "raw e2e tables not
archived" limitation sentence. Resources: 8xH800 session (~1 day). NOT EXECUTED.
### D2 (G3, E04) — requires GPU capture; deferred
Re-capture book_4438tok_O10 rank0 routing trace under frozen recorder config (run_routing_record.sh dump),
or drop the fig12 book-rank0 panel and footnote the coverage. Resources: 1 H20 boot + 1 capture run. NOT EXECUTED.
### D3 (G4, E08) — CPU-only offline planner audit; no GPU; needs author go
Input: packaged ea_v1/ep_v1/nearscan snapshots (same initial maps, budgets, stream counts). Protocol: re-run
best-improvement / PB-gap / max-delta planners offline over stored count snapshots; record per-plan
plan_time_s, migration_bytes (rows x hidden x dtype), accepted/rejected trajectory; cross-check against
nearscan cost_benefit.json per-swap costs. Output: planner comparison table with all three variants on
identical inputs. Risk: none to frozen artifacts (read-only inputs, new output dir). NOT EXECUTED.
### D4 (G7) — documentation reconciliation; no execution
Author designates the canonical headline table (R2 +19.4% aligned-params repro vs README +16.5% grid cell vs
outline +17.5%). All raw values packaged; a one-page provenance note can then be frozen into _COMMON_docs. NOT EXECUTED.
### D5 (G5/G6, E12) — client-host transcript extraction; needs author-side shell
Follow embedded instructions in UNRECOVERED__* files: stream-parse the client-host session jsonl for
LEDGER rows 218-249 and EFFECT_REPORT_old_sglang.md content; save as RECOVERED_TRANSCRIPT__* with sha of
source jsonl. Container-side recovery exhausted (git, /tmp, deleted-fd scan all searched). NOT EXECUTED.
### D6 (G11, E06) — CPU-only independent GOOD recount; needs author go
Input: 64 packaged perreq jsonl (327,680 requests) + SLO_REFERENCE_FROZEN.json (E3 ref_p95_TTFT=6.7419s;
E4 7.3127s + TPOT 0.08313s). Protocol: recompute GOOD membership per frozen definition (error==null AND
TTFT<=k*ref AND (E4) TPOT<=k*ref) for k in {0.8,1.0,1.2}; compare integers against DERIVED table
(must match exactly; any mismatch = finding). Runtime ~minutes, pure CPU, read-only. NOT EXECUTED.

## C. Adverse / failure results explicitly RETAINED in this package (author directive)
- DSV4 all-cells-negative campaign (E12, git-recovered): P_fixab best -3.8%/-7.8%/-14.1%; FP-gate PASS (overhead genuine)
- deadzone cross-rank validation FAILURE: Pearson 0.086 / RMSE 27.425us (H_i frozen per-rank NOT validated cross-rank); frozen-H_i ΔR RMSE 21.628 vs direct 6.734
- gb200 aggregate-ratio-not-sufficient: OEPLB ratio 1.002 (171.5tps) vs MoETuner 1.000 (181.6tps); EPLB-static r_eff 1.517 vs accounted 1.011; deadzone r1.02 -4.848us
- d38 original +15.5% INVALID (server_args L1673 dispatch bug; EXPERIMENT_LOG_repro.md packaged)
- reproduce_d38_adaptive.json 4/4 boot_fail; oeplb_fixed_adaptive_results.json pre-fix -6.0%
- nearscan analysis_attempt1_retired (mechanical region label bug) + analyzer_run_attempt1_retired.log
- prediction_capture_failed_preprofile_attempt01 (whole dir packaged)
- smoke_N256_cleanup_failed.json; phase3_prep B1-hang cores (4, in place); INCIDENT x8 hang records; FP main/p1 INCIDENT cores
- phase-4 adverse: E4 P-vs-I tps -226.3, G -0.148; E-arm E4 Goodput all 0 (tpot_p95 3.7x ref); E3 G@1.0 P-vs-I -0.0599
- final_plan P-vs-I overall +2.21% NS (p=0.0718) vs P-vs-E +36.53% (p=3.8e-08)
- sp_coupling v1 bare-threshold exits 26/26 (fixed in rev1 -> 0)
