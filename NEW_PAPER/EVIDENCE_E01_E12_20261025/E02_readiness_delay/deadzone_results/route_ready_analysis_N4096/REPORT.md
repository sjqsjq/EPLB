# Global-ready real-route analysis (N=4096)

The primary delta is the all-rank latest-ready change. It already includes baseline readiness slack and rank switching, so **H is not subtracted again**.

| requested r | raw r | aligned r | direct global ready Δ (µs) | reconstructed global ready Δ (µs) | original layer Δ (µs) |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000 |
| 1.02 | 1.020020 | 1.027237 | 2.112 | 0.176 | -4.848 |
| 1.05 | 1.050049 | 1.054264 | 1.184 | 1.584 | 3.568 |
| 1.10 | 1.100098 | 1.107692 | 59.904 | 77.393 | 52.432 |
| 1.20 | 1.199951 | 1.186312 | 58.656 | 78.833 | 51.664 |
| 1.40 | 1.399902 | 1.338290 | 217.600 | 244.593 | 206.960 |

Direct global-ready → original-layer RMSE: **6.734 µs** (MAE 5.741 µs).
Per-rank reconstruction → original-layer RMSE: **21.628 µs** (MAE 16.129 µs).

Raw r is not converted to time. The route manifest's per-expert 128-row alignment and measured per-rank ready changes are used, preserving the DeepGEMM staircase.

Negative baseline-subtracted estimates are retained unchanged. Whenever an interval spans zero, the estimate is compatible with a zero effect and the experiment does not identify its sign.

Legacy hot-expert-minus-H ablation (original-rank0, H=16.596 µs) has RMSE **11.895 µs**. It is retained only as a labeled ablation because it omits dispatch changes, other ranks, and last-ready switching.
