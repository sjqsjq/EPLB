# Legacy hot-rank-only route ablation (N=4096)

**Status: diagnostic ablation only.** This analysis uses only the hot rank's expert interval and omits dispatch changes, non-hot ranks, baseline readiness ordering, and last-ready switching. The all-rank global-ready analysis is the primary route model.

Delay-derived hinge (original-rank0): **H=8.985 µs**, slope=0.946.
Unit-slope hinge RMSE: **8.046 µs**; MAE: **6.444 µs**.
Delay-fit slope sensitivity RMSE: **4.738 µs**; MAE: **3.535 µs**.

| requested r | raw r | aligned r | δ hot expert (µs) | layer Δ (µs) | predicted (µs) | residual (µs) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.02 | 1.020020 | 1.027237 | -0.112 | -6.000 | 0.000 | -6.000 |
| 1.05 | 1.050049 | 1.054264 | 2.592 | -1.280 | 0.000 | -1.280 |
| 1.10 | 1.100098 | 1.107692 | 85.632 | 69.744 | 76.647 | -6.903 |
| 1.20 | 1.199951 | 1.186312 | 88.496 | 65.936 | 79.511 | -13.575 |
| 1.40 | 1.399902 | 1.338290 | 245.440 | 225.552 | 236.455 | -10.903 |

Every token keeps exactly seven destination ranks at every r; only the second expert assignment moves. The source→destination unique-row matrix is therefore fixed.

Negative baseline-subtracted measurements are retained. Values whose five-block cluster interval spans zero are compatible with a zero effect; their sign is not identified by this experiment.
