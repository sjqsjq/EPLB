# Per-rank readiness experiment (N=4096)

The strict scan estimates each rank's effective baseline readiness slack from same-rank gate-wait changes. The independently executed original scan is then predicted with those values frozen; no CUDA timestamp is subtracted across GPUs.

| rank | strict H (µs) | 95% block CI | original fitted H (µs) | first tested effect at 4 µs [95% CI] (µs) | first identified positive tested delay (µs) | frozen RMSE (µs) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.202 | [22.928, 39.655] | 13.882 | -2.176 [-11.120, 23.664] | 48 | 6.526 |
| 1 | 26.980 | [18.513, 36.252] | 0.000 | 11.344 [-32.928, 23.632] | 64 | 11.823 |
| 2 | 42.891 | [36.372, 63.475] | 7.959 | 9.968 [-14.432, 27.952] | 96 | 9.245 |
| 3 | 26.876 | [18.671, 34.819] | 11.291 | -4.704 [-29.136, 30.544] | 64 | 8.963 |
| 4 | 42.907 | [36.866, 51.675] | 0.000 | -4.272 [-15.712, 20.400] | 48 | 12.027 |
| 5 | 38.978 | [34.487, 55.124] | 20.753 | -9.776 [-24.752, 18.128] | 96 | 7.732 |
| 6 | 38.448 | [26.463, 52.993] | 36.012 | -8.016 [-9.856, 22.672] | 96 | 6.011 |
| 7 | 38.520 | [33.268, 54.720] | 1.366 | 0.512 [-25.152, 32.704] | 64 | 9.924 |

## Last-ready identity

**No stable singular last-ready rank was identified.** The point-estimate minimum is rank 3 (26.876 µs), but the block winners are **[1, 1, 3, 3, 3]** (ranks 1 and 3 switch) and the block-interval candidate set is **[0, 1, 3, 6]**. Therefore no fixed rank is assigned zero distribution-level slack. At the first 4 µs tested delay, every rank's original-path interval spans zero; no fixed rank shows an identified response from the smallest intervention.

## Independent plateau prediction

Strict-vs-original fitted-H Pearson correlation is **0.086** and H RMSE is **27.425 µs**. The frozen unit-slope curve has **9.269 µs** RMSE over all points (including the fixed zero anchor for every rank) and **9.770 µs** over nonzero-delay points. The curves capture part of the aggregate rise at large delay, but the near-zero correlation and large H error mean that rank-specific plateau widths were **not independently validated**.

The one-sided intervention has an asymmetric interpretation: absorbing extra delay on a noncritical rank does not show that reducing computation on the current slowest rank has no benefit over the same interval.

Negative baseline-subtracted estimates are retained unchanged. Whenever an interval spans zero, the estimate is compatible with a zero effect and the experiment does not identify its sign.
