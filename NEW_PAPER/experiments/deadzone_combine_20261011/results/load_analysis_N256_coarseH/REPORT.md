# Legacy hot-rank-only route ablation (N=256)

**Status: diagnostic ablation only.** This analysis uses only the hot rank's expert interval and omits dispatch changes, non-hot ranks, baseline readiness ordering, and last-ready switching. The all-rank global-ready analysis is the primary route model.

Delay-derived hinge (original-rank0): **H=22.697 µs**, slope=0.918.
Unit-slope hinge RMSE: **2.412 µs**; MAE: **2.003 µs**.
Delay-fit slope sensitivity RMSE: **2.412 µs**; MAE: **2.003 µs**.

| requested r | raw r | aligned r | δ hot expert (µs) | layer Δ (µs) | predicted (µs) | residual (µs) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.02 | 1.019531 | 1.000000 | 2.000 | 0.912 | 0.000 | 0.912 |
| 1.05 | 1.050781 | 1.000000 | 1.264 | 2.768 | 0.000 | 2.768 |
| 1.10 | 1.101562 | 1.000000 | 1.152 | -2.176 | 0.000 | -2.176 |
| 1.20 | 1.199219 | 1.000000 | -0.160 | -1.920 | 0.000 | -1.920 |
| 1.40 | 1.398438 | 1.000000 | 0.720 | -4.240 | 0.000 | -4.240 |

Every token keeps exactly seven destination ranks at every r; only the second expert assignment moves. The source→destination unique-row matrix is therefore fixed.

Negative baseline-subtracted measurements are retained. Values whose five-block cluster interval spans zero are compatible with a zero effect; their sign is not identified by this experiment.
