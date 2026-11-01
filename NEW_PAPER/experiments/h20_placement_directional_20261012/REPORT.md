# 8×H20 frozen expert-placement E0/E1 pilot report

- **Date:** 2026-10-12
- **Hardware:** one node, 8×NVIDIA H20, EP=8
- **Execution:** DeepEP normal mode, DeepGEMM FP8, eager replay (no CUDA Graph)
- **Model geometry:** Qwen3-235B-A22B-FP8, 128 routed experts, top-k 8, hidden size 4096, routed intermediate size 1536
- **Captured layer:** routed layer 24
- **Frozen real forward:** 4050 rank-local hidden rows in total and 32,400 valid expert assignments
- **Status:** E0 and the requested 10×50 E1 pilot are complete and pass the frozen correctness/audit gates. This is a deliberately imbalanced, single-hot-expert placement case from one process start; the four-start confirmation has not been run.

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

Detailed instrumentation changes some absolute timings. The interval for its change to the placement effect crosses zero, and A1 remains clearly positive in both modes, so minimal observation is retained as E1's lower-observer-effect primary mode. E0 supports only the claim that the large A1 effect has the same direction in both modes; it does **not** prove that detailed observation has no effect.

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

All six positive A/B examples revolve around the **same** hot expert, e089, which has 3312 assignments in the frozen route. A1–A3 exchange e089 with e016, e034, and e041, each of which has zero assignments in this forward. The result therefore establishes that moving this one large hot expert is effective in this intentionally imbalanced case; it does not test multiple hot experts, multiple layers, or other load regimes. Layer 24 was selected in advance from route-imbalance metadata rather than performance timing, which avoids performance-based layer selection but does not make this a representative layer sample.

The secondary `T_proxy` gives essentially the same signed estimates (differences from the primary estimates are below 0.2 μs for every layout), but remains labeled only as a proxy.

A post-pilot tolerance for a future confirmatory main run is

\[
\epsilon=\max(0.01\times2294.804,2\ \mu s)=22.948\ \mu s.
\]

A1–A3 and B1–B3 have intervals wholly above `+epsilon`; D1 and D3 have intervals wholly below `-epsilon`. C1–C3 and D2 cross zero, and none of those intervals lies wholly inside `[−epsilon,+epsilon]`. They are therefore **compatible with zero but do not establish practical equivalence**. In particular, their negative point estimates are not relabeled as noise.

Because this epsilon was derived after the pilot, the classifications are planning/descriptive labels for a future independent-start main run, not preregistered confirmatory decisions on this same pilot.

## 6. Load-model interpretation limits

The identity layout has:

- raw maximum rank assignments: 10,093;
- raw `r=max/mean`: 2.4921;
- independently 128-row-aligned maximum rank work: 11,264 rows;
- aligned `max/mean`: 2.2069.

This experiment does **not** show that a padding-aware model is better than raw load. Across the 12 deliberately stratified layouts, the descriptive Pearson correlation between mean `Q` and raw-max improvement is 0.9956, versus 0.9965 for aligned-max improvement. The difference is negligible. Spearman correlations are likewise similar at 0.9537 and 0.9519. These are in-sample descriptive associations over selected layouts, not train/test evaluation of predictive models.

Category A reduces the raw maximum by 3312 assignments and the aligned maximum by 3328 rows. Category B has a broadly comparable raw-max improvement of 2995–3171 assignments and an aligned-max improvement of 2944–3072 rows. A's three layout-effect means average +498.071 μs and B's average +421.378 μs. That ordering is **compatible** with a padding-aware explanation, but A and B simultaneously change raw rank load, expert/slot composition, and source→destination communication. It cannot be attributed specifically to padding.

The logical route is frozen. Consequently, each logical expert keeps exactly the same token count and the same independently padded row count in every layout; placement does not reduce total per-expert work. It redistributes those fixed expert workloads across GPUs and changes their physical slots and communication destinations. Raw `r` remains route metadata rather than compute time, and no formula such as `delta = S(r-1)` is used.

A2 and A3 provide a matched aggregate-load comparison: their raw rank vectors, aligned rank vectors, and source→destination assignment matrices are identical. The point estimate is `Q(A3)-Q(A2)=+17.038 μs`, but the paired 95% interval is `[−29.249,+63.325] μs`. Because it crosses zero, this pilot does **not** demonstrate that expert/slot structure changes time at fixed aggregate `r`. A2/A3 are useful controls for a higher-precision follow-up, not current evidence of a resolved structural effect. The same caution applies to other matched aggregate cases.

An independently calibrated grouped-GEMM cost model has not been evaluated here. Predictive comparisons among raw, padding-aware, and grouped-GEMM models require calibration on nonoverlapping real forwards and evaluation on unseen layouts.

## 7. Supported claim and unanswered stopping question

The most reliable paper claim from this pilot is:

> **Real expert exchanges have directional performance effects, and their benefit must be evaluated through actual execution cost.**

More specifically, moving e089 away from the identity bottleneck rank makes all six e089-centered A/B layouts faster in this captured layer/forward, while D1 and D3 worsen the bottleneck descriptors and are slower. These are placement-case observations, not evidence for general predictive accuracy, an optimal policy, or a theoretical bound.

This experiment has **not** reached or identified a stopping region. After the beneficial A/B swaps, raw `r` remains approximately 1.67–1.75, still far from a nearly balanced layout. C1–C3 and D2 do not lower the global raw or aligned maximum; their intervals cross zero and are not wholly inside the post-pilot `[−epsilon,+epsilon]` negligibility band. They therefore cannot establish a deadzone or a stopping threshold. The pilot answers whether a real exchange can help in this case, not how far balancing should continue.

No one-sided delay-slack observation is converted into a symmetric layout deadzone. Negative estimates remain signed, and no additional slack `H` is subtracted from effects that already measure the change in completion-fenced completion time.

## 8. Pilot scope and limitations

- All six positive layouts move the same 3312-assignment expert e089; multiple hotspots are untested.
- One intentionally imbalanced routed layer was preselected using route metadata. Layer selection did not use performance, but other layers and lower-imbalance regimes are untested.
- One real captured forward, one request mixture, one node, and one process start were measured.
- The result is specific to 8×H20, EP=8, the recorded DeepEP 20-SM configuration, DeepGEMM FP8, and the recorded software environment. Runtime also warned that NVCC 12.9 is preferred while this environment uses the recorded 12.8 toolchain; that warning was not hidden or retuned after candidate selection.
- Completion-fenced timing is one CUDA interval around each 50-forward block and includes the end fence; it is not pure single-layer makespan. The primary and secondary estimates agree closely on the large-effect direction, but neither changes that label.
- Ten paired blocks describe within-start variability and do not replace independent starts. The protocol recommends four starts × five blocks × 50 forwards for confirmation, with every signed repetition retained rather than selecting the best run.
- E0 shows that the large A1 effect keeps its direction under detailed observation. It does not establish absence of observer effect.
- The raw E1 artifact preserves all 130 randomized block aggregates and each block's iteration count, but not 6500 individual-forward latency samples; the paired block is the statistical unit.
- The 12 per-layout intervals are unadjusted for multiplicity.
- No synthetic delay scan was extended.

## 9. Prioritized next experiments

1. **Confirm the existing placement case.** Keep all 13 frozen layouts and run four independent process starts. Apply the same warmup after every layout activation, retain the frozen numerical tolerance, preserve all signed results, and do not select the best repeat.
2. **Test predictive models out of sample.** Use nonoverlapping real forwards to calibrate raw-load, padding-aware, and grouped-GEMM cost models, then predict unseen layouts. Add different hot experts and additional routed layers.
3. **Study stopping behavior directly.** Start from a frozen low-`r` layout, measure the residual benefit of real legal swaps, and only afterward incorporate online migration/exchange cost.

The experimental design and this e089-centered placement case are suitable to write up now. Predictive ability, a stopping threshold, and theoretical upper bounds are not validated conclusions from this pilot.

## 10. Reproducible artifacts

- Raw E1 result: `results/raw/e1_pilot.json`
  - Size: 613,491 bytes
  - SHA-256: `03cb13c23dc0ecde85a355cdde84b2f7fa7357f983b798e174ff4d275d4ca5d1`
  - Repository status: tracked, so all 130 block records and randomized condition schedules can be independently recomputed.
- Machine-readable validated summary: `results/e1_summary.json`
- Layout-effect table: `results/e1_layout_effects.csv`
- Raw/log/code artifact index with byte counts and SHA-256 hashes: `results/raw_index.json`
- Replay implementation: `scripts/replay_placement.py`
- Validation/analysis implementation: `scripts/analyze_e1.py`

The compact E1 timing JSON, logs, code, and summaries are tracked in Git. The eight real forward snapshots are approximately 310 MB each, remain excluded from Git, and are indexed locally with byte counts and SHA-256 hashes. They have **not** been uploaded or published. Consequently, the protocol's fetchable archive/URL requirement remains incomplete for independent off-machine replay; `results/raw_index.json` records `archive_status=local_only_not_published` and a null fetch URL for each snapshot. This packaging limitation does not prevent independent recomputation of the reported timing statistics from the tracked E1 JSON, but the large frozen tensors are still needed to rerun the operators.
