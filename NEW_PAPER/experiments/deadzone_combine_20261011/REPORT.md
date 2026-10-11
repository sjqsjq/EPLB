# Readiness slack and real-route global-ready prediction: final H20 report

## Questions

On 8×H20, EP=8, using the actual DeepEP-normal + DeepGEMM-FP8 layer path with
fixed placement and PB-OEPLB disabled:

1. Can extra delay on a noncritical rank be absorbed by baseline readiness
   slack even when useful early combine progress is forbidden?
2. Is there one stable baseline last-ready rank whose independently estimated
   slack predicts each rank's original-path delay plateau?
3. For real routing changes, does an all-rank latest-ready change explain layer
   time better than the older hot-expert-minus-`H` model?
4. What does this intervention allow us to say—and not say—about the benefit of
   reducing work on the current slowest rank?

## Protocol

The layer has Qwen3-235B-A22B geometry: hidden size 4096, expert intermediate
size 1536, 128 routed experts, top-k 8, and 16 experts per rank. DeepEP normal
mode dispatches FP8 activations with `expert_alignment=128`; DeepGEMM runs the
FP8 grouped gate/up and down GEMMs; SGLang gathers and combines the outputs.

A stream-ordered GPU kernel is injected after expert compute/gather and before
combine dependency capture. Every rank executes a common 20 µs floor. A target
rank receives an additional differential. The strict path inserts a GPU-ordered
NCCL all-rank gate before the unchanged combine.

Every production condition has 5 independently shuffled blocks × 100
repetitions. Every observation stores all eight ranks. Effects are same-block
condition medians minus matching baseline medians, summarized by the median of
five block effects. Confidence intervals cluster-bootstrap blocks. All timing
intervals are formed on the same GPU; timestamps from different GPUs are never
subtracted. Correctness maximum absolute error is zero.

Negative baseline-subtracted values are preserved. When an interval crosses
zero, it is compatible with a zero effect and this experiment does not identify
its sign.

## 1. Earlier one-sided delay control

The all-rank delay control propagates approximately one-for-one:

| tokens | calibrated C0 (µs) | all-rank slope |
|---:|---:|---:|
| 256 | 87.232 | 1.004 |
| 4096 | 211.600 | 0.990 |

The original and strict paths both exhibit a selected-rank delay plateau. Joint
coarse+dense post-expert fits are:

| tokens | original H (µs) | original slope | strict H (µs) | strict slope |
|---:|---:|---:|---:|---:|
| 256 | 16.729 | 0.866 | 20.181 | 0.885 |
| 4096 | 18.075 | 0.940 | 18.711 | 0.933 |

Under the strict gate, post-gate combine duration remains nearly constant while
the target rank first releases its own gate wait and, at larger delays, adds
wait to other ranks. Thus useful early combine progress is not required for
this **one-sided delay tolerance**. Baseline rank-ready skew is sufficient.

This conclusion is directional. Adding delay to a currently noncritical rank
without changing makespan does **not** imply that reducing work on the current
slowest rank has no benefit for the same duration. The earlier plateau cannot,
by itself, establish a symmetric layout-optimization “no-benefit zone.”

## 2. Decisive all-rank readiness scan

The exact N=4096, `r=1`, `fixed7_one_double` route was frozen. Ranks 0–7 were
independently delayed at `0,4,8,12,16,24,32,48,64,96 µs` in a strict scan and
in a separately executed original-path scan. Strict-derived `H_i` values were
frozen before original-path evaluation.

| rank | strict H (µs) | 95% block interval (µs) | original fitted H (µs) | frozen-curve RMSE (µs) | first identified positive tested delay (µs) |
|---:|---:|---:|---:|---:|---:|
| 0 | 30.202 | [22.928, 39.655] | 13.882 | 6.526 | 48 |
| 1 | 26.980 | [18.513, 36.252] | 0.000 | 11.823 | 64 |
| 2 | 42.891 | [36.372, 63.475] | 7.959 | 9.245 | 96 |
| 3 | 26.876 | [18.671, 34.819] | 11.291 | 8.963 | 64 |
| 4 | 42.907 | [36.866, 51.675] | 0.000 | 12.027 | 48 |
| 5 | 38.978 | [34.487, 55.124] | 20.753 | 7.732 | 96 |
| 6 | 38.448 | [26.463, 52.993] | 36.012 | 6.011 | 96 |
| 7 | 38.520 | [33.268, 54.720] | 1.366 | 9.924 | 64 |

### Last-ready identity was not stable

The smallest point-estimate slack belonged to rank 3, but the five block
winners were **`[1,1,3,3,3]`**. The block-interval overlap candidate set was
**`[0,1,3,6]`**. The experiment therefore did not identify one stable singular
last-ready rank and does not force any fixed rank to have zero
distribution-level slack.

At the smallest tested delay (4 µs), every rank's original-path confidence
interval crossed zero. The “first identified positive” delays in the table are
only the first tested points whose five-block lower interval endpoint exceeded
zero. They are coarse, uncertainty-limited thresholds, not precise hinges.

### Rank-specific plateau prediction was not validated

Across ranks:

- strict-vs-original fitted-`H_i` Pearson correlation: **0.086**;
- strict-vs-original fitted-`H_i` RMSE: **27.425 µs**;
- frozen unit-slope all-point RMSE: **9.269 µs**;
- frozen unit-slope nonzero-point RMSE: **9.770 µs**.

The frozen curves capture part of the broad response at large delays. However,
the near-zero cross-rank correlation and the 27.425 µs slack error mean that
the strict estimator did **not** independently validate the ordering or width
of every rank's original-path plateau. Baseline ordering switched, and the
rank-specific `H_i` values were not precise enough for the stronger prediction.

## 3. Padding-aware real routes

The real-route generator keeps exactly seven destination ranks per token while
moving only the second expert assignment. The complete source→destination
unique-row matrix is unchanged. Requested ratios are
`r={1,1.02,1.05,1.10,1.20,1.40}`.

DeepGEMM aligns each expert independently to 128 rows. Therefore raw `r` is
route metadata, not a linear compute-time proxy. At N=4096 the hot rank's
aligned work changes in steps:

| requested r | raw r | hot aligned rows | aligned max/mean |
|---:|---:|---:|---:|
| 1.00 | 1.000000 | 4096 | 1.000000 |
| 1.02 | 1.020020 | 4224 | 1.027237 |
| 1.05 | 1.050049 | 4352 | 1.054264 |
| 1.10 | 1.100098 | 4608 | 1.107692 |
| 1.20 | 1.199951 | 4992 | 1.186312 |
| 1.40 | 1.399902 | 5760 | 1.338290 |

At N=256, all nonempty expert counts remain on the first 128-row plateau even
at raw `r≈1.398`; aligned max/mean remains 1.0. These manifests rule out a
universal conversion such as `δ=S(r-1)`.

## 4. Primary all-rank global-ready model

For a route condition, the relevant causal quantity is the change in the latest
ready time over all ranks:

\[
\Delta R_{\mathrm{global}}
=
\max_i R_i^{\mathrm{route}}-
\max_i R_i^{\mathrm{baseline}}.
\]

The direct strict-path measurement is

\[
\Delta\max_i(\text{layer\_ms}_i-
\text{combine\_after\_gate\_ms}_i).
\]

It already includes baseline readiness ordering and any last-ready rank switch.
Consequently, **`H` is not subtracted again**. A secondary reconstruction uses
`max_i(-H_i+ΔR_i)-max_i(-H_i)` and is expected to degrade if rank-specific
`H_i` is unstable or imprecise.

| requested r | raw r | aligned r | original layer Δ (µs) [95% interval] | direct global-ready Δ (µs) | reconstructed Δ (µs) |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000 |
| 1.02 | 1.020020 | 1.027237 | −4.848 [−10.480, 0.704] | 2.112 | 0.176 |
| 1.05 | 1.050049 | 1.054264 | 3.568 [−8.656, 4.816] | 1.184 | 1.584 |
| 1.10 | 1.100098 | 1.107692 | 52.432 [0.416, 70.368] | 59.904 | 77.393 |
| 1.20 | 1.199951 | 1.186312 | 51.664 [6.528, 66.416] | 58.656 | 78.833 |
| 1.40 | 1.399902 | 1.338290 | 206.960 [105.008, 226.544] | 217.600 | 244.593 |

The `r=1.02` and `r=1.05` original-layer intervals cross zero and are therefore
compatible with zero; their signs are not identified. The route points at
`r=1.10` and `r=1.20` have nearly identical layer penalties despite different
raw ratios, while `r=1.40` jumps to the next large response. This is consistent
with a padding/compute staircase rather than a linear raw-ratio mapping.

### Model comparison on the same route-ready scan

| model | RMSE (µs) | MAE (µs) | status |
|---|---:|---:|---|
| direct all-rank global-ready change | **6.734** | **5.741** | primary |
| legacy hot-expert-minus-`H` | 11.895 | 9.489 | diagnostic ablation |
| per-rank reconstruction with frozen `H_i` | 21.628 | 16.129 | secondary failed/limited test |

The direct all-rank description is best among the three. The older 3.754 µs
fit from a separate hot-rank-only scan is retained in its historical artifact,
but it is no longer treated as causal validation: it omits dispatch changes,
other ranks, and last-ready switching. On the new matched route-ready scan, the
same legacy form has 11.895 µs RMSE.

Direct global-ready intervals are themselves broad at several points—for
example `r=1.10: [−25.136,74.016] µs`, `r=1.20: [−4.608,70.448] µs`, and
`r=1.40: [96.976,226.976] µs`. The 6.734 µs RMSE is therefore a descriptive
point-estimate comparison, not high-precision proof at every route ratio.

## 5. Scientific conclusions

### Supported

1. Extra delay on a currently noncritical rank can be absorbed by natural
   readiness slack; useful early combine progress is not required.
2. Baseline last-ready identity can switch across blocks.
3. Real-route layer effects are better described by an all-rank latest-ready
   change than by one hot rank's expert interval.
4. Once a global latest-ready delta is formed, subtracting `H` again would
   double-count baseline readiness ordering.
5. Raw `r` alone does not determine compute delay. Per-expert 128-row padding
   must enter the route-to-time model.

### Not supported

1. One fixed, universal per-rank plateau width.
2. One stable singular last-ready rank for this baseline.
3. Accurate cross-rank prediction of original-path plateau widths from the
   strict-derived `H_i` values.
4. The claim that a noncritical-rank delay plateau proves an equally wide
   no-benefit region for speeding up the current slowest rank.
5. A universal controller stopping law `(δ-H)+` inferred solely from this
   one-sided delay intervention.

A workload-specific stopping threshold may still be useful as an empirical
controller heuristic, but it must be calibrated from directional swap effects
or a model of `Δ max_i R_i`, not presented as a symmetric consequence of the
delay experiment.

## Artifacts

- All-rank readiness report and data:
  `results/readiness_analysis_N4096/`
- Global-ready route report and data:
  `results/route_ready_analysis_N4096/`
- Earlier coarse/dense delay controls:
  `results/analysis_*`, `results/analysis_dense_*`, and
  `results/analysis_merged_*`
- Historical hot-rank-only route analyses (diagnostic only):
  `results/load_analysis_*`
- Static and Nsight timelines:
  `results/timeline_analysis_N4096/` and `traces/N4096/`
- Full validation: `results/validation.json`
- Commands and environment: `README.md`
