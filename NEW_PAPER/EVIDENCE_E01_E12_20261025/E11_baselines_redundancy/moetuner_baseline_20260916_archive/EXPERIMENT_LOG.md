# MoETuner Baseline Reproduction (2026-09-16)

Reproduce **MoETuner (Go & Mahajan, arXiv:2502.06643)** as offline-optimal baseline
for PB-OEPLB paper §5.3 comparison on Qwen3-235B-A22B-FP8.

## What is MoETuner

- ILP1 clustering of E=128 experts into G=8 clusters (K=16 per cluster) minimizing
  per-layer load imbalance Σ|T_c − T_bar|, on a per-layer basis.
- Uses SGLang native `expert-distribution-recorder` (`--expert-distribution-recorder-mode stat`) to profile
  per-layer expert token counts, then feeds P[L=94, E=128] to Gurobi ILP.
- Emits `placement.json` (physical_to_logical_map only) consumed by
  SGLang `--init-expert-location`. Placement is **frozen** — no online adaptation.
- ILP solver: Gurobi 60s TimeLimit per layer × 94 layers ≈ 17min total.

## Setup (matched 09/14 baseline_comparison_20260914)

- Hardware: 8×H20 96GB, Qwen3-235B-A22B-FP8, TP=DP=EP=8 (16 experts/card)
- SGLang: DeepEP `auto` + DeepGEMM FP8, `--disable-cuda-graph`, `--mem-fraction-static 0.78`,
  `--disable-radix-cache`, `--skip-server-warmup`
- Bench harness: `bench_0914.py` = 09/14 `prover_prefill.py` (endpoint `/generate`,
  `max_new_tokens=1`, no concurrency cap, N=256 concurrent asyncio.gather),
  3 runs, r1 dropped as cold-start, median over r2/r3.
- Freq6 harness: `freq6_bench_json.py` = 09/14 `freq6_bench.py` verbatim
  (conc=32, N=1800, 6 segments book_4438tok/prover_2048tok alternating, O=10),
  3 runs median.

## Datasets

- Same-domain: `single_domain/prover_256tok_out1.jsonl` (pinned 10× per-layer imbalance,
  first 1024 for profile, prover_256tok N=256 for bench)
- Cross-domain: freq6 built inline from `prefill_decode_correlation/book_4438tok_O10.jsonl`
  + `single_domain/prover_2048tok_out1.jsonl` (6 segments of 300 requests each,
  alternating book/prover)

## Placement generated

- `artifacts/placement_pinned_fair.json` — solved from profile on
  `pinned10x_head1024.jsonl` (first half of prover_256tok, disjoint from bench half).
  Total solver time 1015.5s (Gurobi, TimeLimit 60s/layer, MIPGap 0.005).
  mean_imbalance 1.000 (perfect balance achievable on this dataset).
- `artifacts/P_pinned_fair.npz` — profiled P[94, 128] token counts (all-reduced across 8 ranks).

## Results

### E1. Same-domain pinned10x (prover_256tok, N=256, 3 runs, r1 cold dropped)

| method | r1 (cold) | r2 | r3 | median | vs identity |
|---|---|---|---|---|---|
| identity | 15.7 | 55.2 | 56.1 | **55.7** | — |
| MoETuner (pinned_fair placement) | 18.3 | 68.0 | 73.0 | **70.5** | **+13.0% (vs 09/14 identity=62.4)** |

### E2. Cross-domain freq6 (6seg book↔prover_2048tok, N=1800, conc=32, 3 runs median)

| method | r1 | r2 | r3 | median | vs identity |
|---|---|---|---|---|---|
| identity | 4.43 | 4.53 | 4.50 | **4.50** | — |
| MoETuner (pinned_fair placement → freq6) | 4.45 | 4.63 | 4.65 | **4.63** | **−1.5% (vs 09/14 identity=4.7)** |

## Interpretation

- **Same-domain oracle setting**: MoETuner (frozen placement fit to prover profile)
  achieves +26.7% — the largest gain among all baselines tested (09/14 table:
  PB_OEPLB +20.7%, DataForest +18.0%, EPLB_static +9.6%, EPLB_dyn −1%).
  This confirms MoETuner's role as **offline-optimal upper bound** for static
  placement methods when the test distribution matches profile distribution.
- **Cross-domain transfer failure**: dropping the same prover-fit placement into
  freq6 yields only +2.9% — offline placement gained nothing transferable, matching
  the theoretical prediction that per-layer clustering fits a single distribution.
  PB-OEPLB (§5.3.1 freq6 row +8.5%) beats MoETuner cross-domain by adapting online.
- Comparison against 09/14 identity baseline (62.4 req/s) shows today's identity
  is 55.7 (−11% run-to-run variance from same server config). Ratios are the
  invariant metric — MoETuner 70.5 / identity 55.7 = **1.27×**, exceeds
  PB_OEPLB's 75.3 / 62.4 = 1.21× same-domain.

## Files

- `results/` — bench JSON outputs (label, N, tps, time)
- `scripts/` — driver + launch + bench + ILP solver source
- `artifacts/placement_pinned_fair.json` + `.meta.json` — final placement + solver meta
- `artifacts/P_pinned_fair.npz` — profiled token count tensor
- `logs/` — full driver stdout for each phase

## Key implementation notes

1. `--deepep-mode` for profile server: must be `normal`, not `low_latency`
   (low_latency asserts on prefill batch size).
2. SGLang native recorder mode `stat` — no custom hook needed. HTTP endpoints:
   `/start_expert_distribution_record`, `/stop`, `/dump` (POST bodies empty; use wget
   since curl not available in container).
3. `placement.json` only accepts the `physical_to_logical_map` field for `--init-expert-location`.
4. rank0's `.pt` dump already contains all-reduced global sum; no nee