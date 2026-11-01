# README — provenance of the DSV4 negative-results recovery (E12)
Recovery executed: 2026-10-25T19:56:51Z–19:58Z (UTC), read-only forensic recovery.
Output dir: /workspace/EPLB/NEW_PAPER/EVIDENCE_E01_E12_20261025/E12_generalization/dsv4_recovered_from_transcript/

## 1. MANDATED PROVENANCE STATEMENT (as required by the E12 checklist)
> "Originals deleted from disk 2026-10-25 by author order; models
> (/root/models/DeepSeek-V4-Flash, dsv4_ref_mp8_fp8, dsv4_serve_mp8_m2), envs
> (/opt/dsv4_env, /opt/dsv4_ref_env) and the experiment tree remain deleted —
> NOT recovered. These copies are transcript-derived and were never
> disk-verified against the originals at recovery time (they were verified
> while originals existed, pre-deletion)."

**PROVENANCE CORRECTION (material, must be read with the statement above):**
the copies in this directory are NOT transcript-derived. They are
**git-object-derived**: byte-exact copies of the file blobs committed at
HEAD of /workspace/EPLB/.git before the deletion. The mandated sentence's
first clause was independently re-verified at recovery time and remains
true: /workspace/EPLB/NEW_PAPER/experiments/dsv4_adaptation_20261022/,
/opt/dsv4_env, /opt/dsv4_ref_env are absent; /root/models/ contains only
ms_cache (DeepSeek-V4-Flash, dsv4_ref_mp8_fp8, dsv4_serve_mp8_m2 absent).
Nothing was resurrected into the experiment tree; the only writes were to
this output directory.

## 2. Why NOT transcript-derived (full disclosure)
The tasking specified reconstruction from the session transcript JSONL at:
  /home/admin/.claude/projects/-home-admin--aone-cloud-cli-runner-cwd-98046363a249-workspace/17bd8aca-3c5e-4017-ab8d-079c16661419.jsonl
That path is on the CLIENT host. Every execution tool available to this
recovery session (mcp__local__bash/read/write/glob/grep/stat, and a
general-purpose subagent probe) reaches ONLY the remote GPU container
(hostname mlflow-training-ur7y05-511762-n3-master-0, cwd /workspace);
the container has no /home/admin (its /home is empty), and the MCP file
tools reject the transcript path ("not under any allowed root"). Therefore:
- the transcript was never read, parsed, or modified;
- **sha256 of the source jsonl(s): COULD NOT BE COMPUTED from this sandbox**
  (no other .jsonl session files were reachable either);
- artifacts that existed ONLY post-last-commit and only in the transcript
  (EFFECT_REPORT_old_sglang.md, bench_old_* JSONs, LEDGER rows 218–249)
  are NOT recovered here (see UNRECOVERED__* files and the LEDGER gap marker).

## 3. Actual source used (git object store — read-only)
- Repo: /workspace/EPLB/.git, branch sp_coupling_dev (checked-out HEAD).
- HEAD commit: 5df703d6a951b8a42fe86d2dd99dc03faf76d399,
  2026-10-22T18:05:34Z, "dsv4 fixab: campaign done ledger row (commit 6537f9f2)".
- The 2026-10-25 deletion was filesystem-only (rm -rf); it was never committed.
  The git index still tracks 312 paths under
  NEW_PAPER/experiments/dsv4_adaptation_20261022/ — i.e., verbatim originals
  of everything committed through campaign-done survive in git objects.
- Extraction method: `git cat-file blob HEAD:<path>` redirected into this
  output dir (GIT_OPTIONAL_LOCKS=0). No checkout/reset/clean/stash/fsck; no
  refs, index, objects, working tree, /opt/conda, or experiment tree modified;
  no GPU used. Every recovered file is byte-identical to its committed blob
  (blob SHA-1 + file SHA-256 in _RECOVERY_MANIFEST.tsv).
- Cross-validation: all headline numbers recovered from git match, value for
  value, the expected numbers quoted independently in the E12 tasking brief
  (40.9/42.5 −3.8%, 41.4/44.9 −7.8%, 2763.1/3215.1 −14.1%, −6.4/−11.7/−14.3%,
  −18.8/−17.1/−18.4%, kappa 3875.177, s_change 0.0586, h_rate 0.094,
  61/75 = 53+8, gate PASS) — strong evidence of fidelity to the deleted
  originals for the committed state.

## 4. Per-artifact completeness assessment
| artifact | file here | verdict | notes |
|---|---|---|---|
| LEDGER_dsv4.tsv | RECOVERED__LEDGER_dsv4.tsv | **PARTIAL** | 217/249 rows (36,033 B, blob a0b375d3...). Rows 218–249 appended post-commit (old-sglang phase, planb M0–M5 gate executions, rollback_executed) — transcript-only, NOT recovered; gap marker appended (line 218). Env bring-up tries, FIX-NVSHMEM3/4, FIX-FP4DEQUANT, FIX-MEMFRAC, matrix r1/r2/r3, diag decomposition, fixab patch A+B rows all present. |
| fixab_comparison.md | RECOVERED__fixab_comparison.md | **COMPLETE** | 2,132 B, final campaign version (written 18:05:21Z, committed 18:05:34Z). Median table + patch effect + controller behavior + function-preservation gate PASS. |
| bench_matrix_final_table.md | RECOVERED__bench_matrix_final_table.md | **COMPLETE** | 2,105 B. r1+r2 medians, latency deltas, per-round raw, sp_run_end summaries. |
| EFFECT_REPORT_old_sglang.md | UNRECOVERED__EFFECT_REPORT_old_sglang.md | **NOT RECOVERED** | post-commit, transcript-only. Brief's expected values (4.3→4.1, 4.4→4.1, 229.6→203.5; −6.0/−6.7/−11.4%) recorded there as UNVERIFIED. |
| STATS_PATH_TRACE.md | RECOVERED__STATS_PATH_TRACE.md | **COMPLETE** | 8,966 B (~9KB ✓). z formula, kappa=3875.1768798828125, N_MIN=12320768, E4 61/75=53+8, swap-issuance path, s_change 0.0586, w1 h_rate 0.094 all present. ("jsonl attribution lag" section = §4/§5 attribution semantics.) |
| ENV_NOTES.md | RECOVERED__ENV_NOTES.md | **COMPLETE (as of commit)** | 22,979 B; Phases 1/1b/1c/1d(+1e content per fix-commit msg). Any post-commit old-sglang additions would be transcript-only. |
| planb_assessment.md | RECOVERED__planb_assessment.md | **COMPLETE (as of commit)** | 24,362 B; verdict INFEASIBLE. M0–M5 gate EXECUTION ledger rows are in the missing LEDGER tail. |
| patches/OVERHEAD_ANALYSIS.md | RECOVERED__patches_OVERHEAD_ANALYSIS.md | **COMPLETE** | 9,437 B (not partial). |
| bench_old_{i,p}_{e2,e3,e4}/bench JSONs | UNRECOVERED__bench_old_i_p_e2_e3_e4_json.md | **NOT RECOVERED** | post-commit, transcript-only. |
| 0.5.22-track bench JSON summaries | RECOVERED__bench_<cell>__dsv4_<cell>.json (24 files) | **COMPLETE** | I-arm (e2 r1-r3, e3 r1-r2, e4 r1-r2, static_diag), P_broken-arm (e2 r2, e3 r1-r2, e4 r1-r2, noswap_diag), P_fix (5 cells), P_fixab (3 cells), e2_p1, fixcheck_p_e2, + fixab_gate.json. |
| supplementary tables | RECOVERED__fix_table_final_r3b.md, RECOVERED__fix_comparison_table.md, RECOVERED__bench_matrix_r1_table.md | **COMPLETE** | r3b unpatched-baseline + fix tables. |

## 5. Residue of the deleted campaign still on the container (context only)
/opt/dsv4_env_install.log{,.aliyun_partial}, /opt/dsv4_env_install.sh;
/root/.cache/huggingface/modules/transformers_modules/dsv4_serve_mp8{,_m2};
/root/.tilelang/cache_dsv4; session scratch in /tmp (r2/r3/fix commit-message
drafts, r3_base.sh, clone_diff.txt, ref_b1_runs.sh, tl_verify_clone.py,
keyset_diff*.py, clone.log, ref_smoke.log, convert_mp8.log). The full 312-path
committed tree remains extractable from git objects if the evidence package
later needs more than the artifacts above (not done here — scope discipline).

## 6. Integrity
- _RECOVERY_MANIFEST.tsv: outfile, git blob SHA-1, byte size, SHA-256, git path.
- KEY_NUMBERS.md: consolidated negative-results tables with per-number provenance.
- All RECOVERED__* files are byte-identical to their git blobs; the only
  additions are the single clearly-marked gap line appended to the LEDGER copy
  and the UNRECOVERED__* notices.
