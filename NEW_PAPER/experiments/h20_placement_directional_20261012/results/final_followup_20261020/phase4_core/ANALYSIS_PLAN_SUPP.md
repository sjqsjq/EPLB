# Phase-4 supplement analysis plan — PRE-DECLARED (drafted pre-boots)

Pool: NEW supplement pool (labels s_*), analyzed separately. The old
four-workflow overall G is NOT updated or re-pooled (directive).

## Data per boot
- bench.json (label json): ok/errors, total_time_s, tps, lat percentiles.
- perreq jsonl: per-request TTFT/elapsed/comp_tokens/error.
- Goodput@k (k∈{0.8,1.0,1.2}, primary 1.0) computed with FROZEN reference
  (phase3_prep/SLO_REFERENCE_FROZEN.json, frozen 2026-10-20T09:24:53Z
  before any comparison boot): E3 TTFT-only; E4 TTFT+TPOT.
- PB boots (P/F): v2p1 windows, PLANHASH identity, swap counts (report
  migration activity per arm; P arm S-state behavior reported descriptively).

## Primary hypotheses (Holm family PER STREAM, 2 tests each)
Per stream S ∈ {E3, E4}:
  H1(S): G_P − G_I > 0;  H2(S): G_P − G_E_native > 0
where G = throughput gain metric = tps (bench.json) and Goodput@1.0
reported as co-primary SLO metric (both get Holm within their own family:
tps family {P−I, P−Enative}; Goodput family {P−I, P−Enative}).
Secondary (no Holm, raw p, labeled secondary): P−F per stream.

## Statistics
- n=8 blocks per cell; block-paired deltas (block b: arm_P(b) − arm_X(b)).
- Paired bootstrap over blocks, 10000 draws, percentile 95% CI, two-sided
  p from bootstrap distribution (fraction crossing 0, ×2, capped at 1).
- No sample deletion by mean/p/boot-draw. Failed boots (weight/mapping
  error or migration-class fault): keep failure, stop package, return to
  correctness diagnosis (1 fix round remains); boot NOT silently replaced —
  a rerun (if any) is documented as a new block with reason.
- 不显著≠等价: non-significant results reported with CI, never as equality.

## Reporting
- Per-stream tables: arm × {tps mean±CI, Goodput@k mean±CI, p95 TTFT/TPOT,
  swaps/ windows, S>1 share (P)}.
- Deltas with Holm-adjusted p for primaries; raw p flagged for secondary.
- One-page 可以写/不能写/仍未知 (phase 8 deliverable) references this plan.
