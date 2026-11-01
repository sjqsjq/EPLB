# Author-directed diag #3 — freq6 cross-domain, paper protocol reproduction (2026-10-21)

Author observation: NEW_PAPER cross-domain results (freq6 +8.5%, §5.3.1) vs current system "差很多".
Verified: the paper numbers came from a DIFFERENT launch config than the campaign P arm.

## Paper-era protocol (baseline_comparison_20260914, results.json + launch scripts, originals on /data)
- Workload: freq6 = 6 segments book(4438tok)<->prover(2048tok), N=600, conc=32, O=10 (freq6_bench.py)
- Server: --disable-cuda-graph, mem 0.78, decay 0.5, --pb-oeplb-adaptive-window --pb-oeplb-adaptive-decay
- Statistic: run1=warmup, run2=steady; paper reported medians of repeated runs
- Paper result: identity 4.7 → PB-OEPLB steady 5.1 (+8.5%), "RESET+缩窗32→16→8"
- Campaign P arm differs on EVERY knob: cuda-graph ON (max-bs 128), mem 0.8, decay 0,
  adaptive-window CLI OFF (adaptation moved to v2.1 env knobs OEPLB_SP_MODE/WINDOW_CONTROLLER)
- Model path substituted (old /data/models/Qwen3-235B-A22B-FP8 gone): /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
  (same model as whole campaign). Datasets = paper originals (book_4438tok_O10.jsonl, prover_2048tok_out1.jsonl).

## Results (single boot per cell; run1/run2 tps; ±0.1–0.2 tps ≈ 2–4% single-run noise)
| Boot | Code | Launch config | run1 | run2 steady | steady gain vs matching identity |
|---|---|---|---|---|---|
| f6_I_paper | (import-only) | paper identity flags | 4.5 | 4.5 | — |
| f6_v2_P_paper | v2 frozen | paper P flags (legacy adaptive path) | 4.3 | 4.8 | **+6.7%** |
| f6_I_camp | (import-only) | campaign BASE | 4.7 | 4.8 | — |
| f6_v2_P_camp | v2 frozen | campaign P arm | 4.4 | 4.8 | **0.0%** |
| f6_gh_P_paper | GitHub pristine (ada04359) | paper P flags | 4.1 | 4.9 | **+8.9%** |

## Reading
1. **Paper reproduction PASSES**: GitHub-pristine + paper flags gives +8.9% (paper: +8.5%). The
   published cross-domain claim is faithful to that system+protocol.
2. **The v2 fixes did NOT kill the cross-domain gain**: v2 code under paper flags still gets +6.7%
   (legacy adaptive path with [PB-OEPLB-RESET] intact in v2; 88 RESET lines observed). The
   pristine-vs-v2 gap under paper flags (4.9 vs 4.8 steady) is within single-boot noise.
3. **The campaign launch config is where the gain disappears**: v2 + campaign P flags on the same
   freq6 workload = 4.8 steady = identity (0.0%). The campaign P arm (cuda-graph on, mem 0.8,
   decay 0, adaptive-window CLI off, v2.1 env-knob S/H/W adaptation) does not reproduce the
   paper-era RESET+window-shrink+decay-0.5 behavior, and on this workload its adaptation yields
   no throughput gain. This is consistent with phase-4 E3 (campaign config): P−I tps only +2.7%
   and good_rate@1.0 −6pp.
4. Attribution summary: paper +8.5% ≈ pristine code + paper config. Current campaign weak
   cross-domain ≈ campaign config, NOT the reliability fixes.

## Governance / limits
- author_diag ledger rows f6_* (code column v2/gh_pristine); NOT in formal ≤104 / prep ≤12 pools;
  descriptive only; single boots, not medians — the +6.7 vs +8.9 distinction is NOT established
  (noise), the 0.0% vs +8.9% distinction IS far beyond noise.
- No control-law change made; no parameter search started. Adopting paper-era flags into any
  performance package would be a NEW version requiring author decision (post-freeze governance).
- Containment: bench timeout 900s/run, watchdog 600s, full-dir backup + trap restore;
  RESTORE_v2_f6 OK (deployed shas verified v2: async 2e4652a0, controller 4964cfc8).
