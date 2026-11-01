# Legacy hot-rank-only route ablation (N=4096)

**Status: diagnostic ablation only.** This analysis uses only the hot rank's expert interval and omits dispatch changes, non-hot ranks, baseline readiness ordering, and last-ready switching. The all-rank global-ready analysis is the primary route model.

Delay-derived hinge (original-rank0): **H=16.596 µs**, slope=0.960.
Unit-slope hinge RMSE: **3.754 µs**; MAE: **2.874 µs**.
Delay-fit slope sensitivity RMSE: **3.945 µs**; MAE: **3.280 µs**.

| requested r | raw r | aligned r | δ hot expert (µs) | layer Δ (µs) | predicted (µs) | residual (µs) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.02 | 1.020020 | 1.027237 | -0.112 | -6.000 | 0.000 | -6.000 |
| 1.05 | 1.050049 | 1.054264 | 2.592 | -1.280 | 0.000 | -1.280 |
| 1.10 | 1.100098 | 1.107692 | 85.632 | 69.744 | 69.036 | 0.708 |
| 1.20 | 1.199951 | 1.186312 | 88.496 | 65.936 | 71.900 | -5.964 |
| 1.40 | 1.399902 | 1.338290 | 245.440 | 225.552 | 228.844 | -3.292 |

Every token keeps exactly seven destination ranks at every r; only the second expert assignment moves. The source→destination unique-row matrix is therefore fixed.

Negative baseline-subtracted measurements are retained. Values whose five-block cluster interval spans zero are compatible with a zero effect; their sign is not identified by this experiment.
