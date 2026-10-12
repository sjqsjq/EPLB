# 8×H20 frozen expert-placement E0/E1 pilot report

- **Date:** 2026-10-12
- **Hardware:** one node, 8×NVIDIA H20, EP=8
- **Execution:** DeepEP normal mode, DeepGEMM FP8, eager replay (no CUDA Graph)
- **Model geometry:** Qwen3-235B-A22B-FP8, 128 routed experts, top-k 8, hidden size 4096, routed intermediate size 1536
- **Captured layer:** routed layer 24
- **Frozen real forward:** 4050 rank-local hidden rows in total and 32,400 valid expert assignments
- **Status:** E0 and the requested 10×50 E1 pilot are complete and pass the frozen correctness/audit gates. The protocol's optional paper-main recommendation of four independent starts has not been run.

## 1. What changed—and what did not

For every layout, the replay used the same captured rank-local hidden states, logical top-k expert IDs, logical top-k weights, source ranks, real packed FP8 expert weights, and FP8 scales. Only the complete bijection between logical experts and physical GPU/local slots changed. Logical IDs were mapped through `l2p` before dispatch, and each logical expert's weight tensors and scales moved together to its destination physical slot.

The frozen placement manifest contains identity plus exactly 12 complete legal cross-rank swaps: three each in categories A, B, C, and D. Selection used route/shape metadata only, before performance timing. All 13 `p2l`/`l2p` pairs are bijective inverses and preserve 16 physical slots on every rank.

There is no common 20 μs delay floor, no differential delay, no sleep/spin, and no collective or gate between expert execution and combine. Weight migration, allocation/setup, DeepGEMM JIT work, and ten warmup forwards per layout occur before timed blocks.

Frozen manifest SHA-256:

```text
52b8a045d24456fb03cb5bc294a3a76ba31b243fbb1fbd4a000faf8fb6a12e04
```

## 2. Correctness and audit gates

| Check | Result |
|---|---:|
| E1 timing records | 130 = 13 layouts × 10 blocks |
| Forwards per timing record | 50 |
| E1 rank-layout correctness comparisons | 104 = 13 × 8 |
| Frozen-tolerance `isclose` failures | 0 |
| Nonfinite output pairs | 0 |
| Identity replay max absolute error vs captured output | 0 |
| All-layout max absolute error vs captured/identity | 0.001953125 (`2^-9`) |
| Frozen tolerance | `atol=0.00390625` (`2^-8`), `rtol=0` |
| Correctness aligned-count mismatches | 0 / 104 |
| Warmup aligned-count mismatches | 0 / 104 |
| Ranks with changed frozen GPU input/route/reference hashes | 0 / 8 |
| Delay/gate/source-scan violations | 0 |

Relative error uses `max(abs(reference), 1e-6)` and is diagnostic only; it is not an acceptance criterion because values near zero make it unstable. The frozen tolerance was written before E0/E1 performance timing and was not relaxed afterward.

## 3. Timing semantics

The primary timing is **completion-fenced**, not a pure layer makespan:

1. an untimed NCCL operation aligns block starts;
2. GPU 0 records a same-clock reference start;
3. all ranks execute the entire 50-forward block;
4. after each rank's local combine completion, all ranks enqueue one NCCL end reduce;
5. GPU 0 records the reference end after the reduce.

Cross-device CUDA timestamps are never subtracted. The secondary `T_proxy` is the maximum of eight same-GPU local elapsed times and is not called a calibrated global makespan.

The E1 empty completion-fence reference had mean 74.643 μs/block, median 66.304 μs/block, and range 54.464–239.808 μs/block (20 repetitions). The mean amortized value is 1.493 μs per forward in a 50-forward block. It is retained and labeled as fence cost; it is **not subtracted** from the layout effects. The measured effect below is already the paired change in the completion-fenced interval, so subtracting another horizon/fence term would double-correct it.

Define

\[
Q(s)=T_{\mathrm{identity}}-T_{\mathrm{layout}(s)}.
\]

Positive `Q` means a layout is faster; negative `Q` means it is slower. Every negative observation is retained.

## 4. E0 observer-effect pilot

E0 used identity and A1, minimal and detailed observation, ten randomized blocks, and 50 forwards per condition per block. The values below are paired over blocks and use two-sided 95% Student-t intervals with 9 degrees of freedom.

| Quantity | Mean (μs/forward) | 95% CI (μs/forward) |
|---|---:|---:|
| `Q_minimal` | +519.545 | [+471.723, +567.367] |
| `Q_detailed` | +487.813 | [+429.380, +546.245] |
| Identity: detailed − minimal | +33.672 | [−18.527, +85.872] |
| A1: detailed − minimal | +65.405 | [+4.246, +126.564] |
| `Q_detailed − Q_minimal` | −31.733 | [−112.401, +48.936] |

Detailed instrumentation changes some absolute timings. However, the interval for its change to the placement effect crosses zero, and A1 remains clearly positive in both modes. Minimal observation is therefore the lower-observer-effect primary mode used for E1.

## 5. E1 signed block-paired effects

Identity completion-fenced timing across ten blocks was 2294.804 μs/forward mean, 2287.523 μs/forward median, and 30.759 μs standard deviation.

The table reports the primary completion-fenced `Q`. Intervals are unadjusted per-layout 95% Student-t intervals over ten block-paired differences; this is a pilot, not a family-wise confirmatory 12-hypothesis analysis.

| Layout | Category | Swap | Raw max | 128-aligned max | Mean `Q` (μs) | Median `Q` (μs) | 95% CI (μs) | Pilot interpretation |
|---|:---:|---|---:|---:|---:|---:|---:|---|
| A1 | A | e016 ↔ e089 | 6781 | 7936 | +475.004 | +495.472 | [+393.850, +556.158] | faster |
| A2 | A | e034 ↔ e089 | 6781 | 7936 | +501.085 | +513.653 | [+457.467, +544.704] | faster |
| A3 | A | e041 ↔ e089 | 6781 | 7936 | +518.123 | +515.580 | [+495.211, +541.036] | faster |
| B1 | B | e089 ↔ e097 | 7046 | 8320 | +439.709 | +436.639 | [+408.660, +470.758] | faster |
| B2 | B | e068 ↔ e089 | 7098 | 8320 | +427.274 | +428.514 | [+388.772, +465.776] | faster |
| B3 | B | e039 ↔ e089 | 6922 | 8192 | +397.150 | +409.488 | [+346.051, +448.250] | faster |
| C1 | C | e000 ↔ e016 | 10093 | 11264 | −7.585 | −3.645 | [−40.888, +25.717] | compatible with zero; not equivalent |
| C2 | C | e000 ↔ e034 | 10093 | 11264 | −20.782 | −18.223 | [−58.898, +17.335] | compatible with zero; not equivalent |
| C3 | C | e000 ↔ e041 | 10093 | 11264 | −67.724 | −40.295 | [−140.673, +5.224] | compatible with zero; not equivalent |
| D1 | D | e000 ↔ e083 | 11626 | 12800 | −243.333 | −241.936 | [−290.976, −195.690] | slower |
| D2 | D | e032 ↔ e055 | 10093 | 11264 | −33.068 | −25.953 | [−72.526, +6.391] | compatible with zero; not equivalent |
| D3 | D | e082 ↔ e104 | 11171 | 12288 | −167.881 | −161.216 | [−205.365, −130.396] | slower |

The secondary `T_proxy` gives essentially the same signed estimates (differences from the primary estimates are below 0.2 μs for every layout), but remains labeled only as a proxy.

A post-pilot tolerance for a future confirmatory main run is

\[
\epsilon=\max(0.01\times2294.804,2\ \mu s)=22.948\ \mu s.
\]

A1–A3 and B1–B3 have intervals wholly above `+epsilon`; D1 and D3 have intervals wholly below `-epsilon`. C1–C3 and D2 cross zero, and none of those intervals lies wholly inside `[−epsilon,+epsilon]`. They are therefore **compatible with zero but do not establish practical equivalence**. In particular, their negative point estimates are not relabeled as noise.

Because this epsilon was derived after the pilot, the classifications are planning/descriptive labels for a future independent-start main run, not preregistered confirmatory decisions on this same pilot.

## 6. What the load descriptors do—and do not show

The identity layout has:

- raw maximum rank assignments: 10,093;
- raw `r=max/mean`: 2.4921;
- independently 128-row-aligned maximum rank work: 11,264 rows;
- aligned `max/mean`: 2.2069.

Category A reduces the raw maximum by 3312 assignments and the aligned maximum by 3328 rows. Category B has a broadly comparable raw-max improvement (2995–3171 assignments) but a smaller/different aligned-max improvement (2944–3072 rows). The three A layout-effect means average +498.071 μs; the three B means average +421.378 μs. This ordering is directionally consistent with the padding-aware descriptor, but category averages are descriptive and no inferential category interval is claimed.

Across all 12 deliberately stratified layouts, the descriptive Pearson correlation between mean `Q` and raw-max improvement is 0.9956; for aligned-max improvement it is 0.9965. Spearman correlations are 0.9537 and 0.9519, respectively. These near-identical values do **not** establish that one model is superior: the layouts were intentionally selected across extreme load strata, the sample is small, and these are not train/test-evaluated models.

Raw `r` remains route metadata, not compute time. The analysis explicitly includes per-expert 128-row padding and the actual aligned grouped-GEMM row shapes. No formula such as `delta = S(r-1)` is used. An M2 grouped-GEMM cost model has not been independently calibrated here, and no cost model is fitted to the same 12 timing effects it would be evaluated against.

Several matched aggregate cases also caution against collapsing placement to one scalar. A2/A3 have identical raw rank vectors, aligned rank vectors, and source→destination assignment matrices, as do C2/C3, yet their point estimates differ. The paired intervals for those within-pair differences cross zero, so this pilot does not prove a local-slot/expert-composition effect; it only shows that aggregate maxima cannot be treated as a complete measured compute-time model without independent calibration.

## 7. Supported conclusions

1. **A real placement intervention has a large directionally consistent effect on this captured H20 layer/forward.** Moving the 3312-assignment hot expert e089 away from the identity bottleneck rank produces approximately 397–518 μs/forward faster completion-fenced execution in all six A/B layouts.
2. **Padding-aware shape information matters to the experimental interpretation.** A and B have broadly similar raw improvements but distinct aligned maxima and distinct average effects. This is useful directional evidence, not a calibrated causal conversion from rows to microseconds.
3. **Changing noncritical composition while leaving the global raw/aligned maximum unchanged does not prove a deadzone.** C1–C3 and D2 have intervals crossing zero. That means compatibility with zero, not proof that all negative values are noise and not proof of a symmetric no-benefit interval.
4. **Worsening the bottleneck can hurt.** D1 and D3 worsen both raw and aligned maxima and show clearly negative `Q`. This is a measured layout effect; no additional slack `H` is subtracted.
5. **Minimal instrumentation is adequate for the E1 primary result.** E0 found no resolved change in the A1 layout effect between minimal and detailed observation, while detailed mode does add some absolute overhead.

## 8. Scope and limitations

- One real captured forward, one routed layer, one request mixture, one node, and one process start were measured.
- The result is specific to 8×H20, EP=8, the recorded DeepEP 20-SM configuration, DeepGEMM FP8, and the recorded software environment. Runtime also warned that NVCC 12.9 is preferred while this environment uses the recorded 12.8 toolchain; that warning was not hidden or retuned after candidate selection.
- Completion-fenced timing includes an end-fence operation and is not called pure layer makespan.
- Ten paired blocks quantify within-start variability but do not replace independent process starts. The protocol recommends four starts × five blocks × 50 forwards for a paper-main confirmation.
- Minimal timing records one CUDA interval around each 50-forward block rather than adding per-forward events. The raw artifact therefore preserves all 130 randomized block aggregates and each block's iteration count, but not 6500 individual-forward latency samples; the paired block is the statistical unit.
- The 12 per-layout intervals are unadjusted for multiplicity.
- The result does not generalize a one-sided delay-slack observation into a symmetric layout deadzone, and no synthetic delay scan was extended.

## 9. Reproducible artifacts

- Raw E1 result: `results/raw/e1_pilot.json`
  - SHA-256: `03cb13c23dc0ecde85a355cdde84b2f7fa7357f983b798e174ff4d275d4ca5d1`
- Machine-readable validated summary: `results/e1_summary.json`
- Layout-effect table: `results/e1_layout_effects.csv`
- Raw/log/code artifact index with byte counts and SHA-256 hashes: `results/raw_index.json`
- Replay implementation: `scripts/replay_placement.py`
- Validation/analysis implementation: `scripts/analyze_e1.py`

The timing JSON/log/code artifacts are below 100 MiB. The eight real forward snapshots are approximately 310 MB each, are excluded from Git, and are indexed locally with byte counts and SHA-256 hashes. They have **not** been uploaded or published because that outward-facing action requires explicit confirmation. Consequently, the protocol's fetchable archive/URL requirement remains incomplete for independent off-machine audit; `results/raw_index.json` records `archive_status=local_only_not_published` and a null fetch URL for each snapshot. This packaging limitation does not alter the local correctness or timing validation, but it must be resolved before claiming a fully reproducible published artifact bundle.
