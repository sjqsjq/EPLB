# Nsight Systems trace validation (N=4096)

All **48** per-rank `.nsys-rep` files and all **48** SQLite exports are nonempty. Every capture contains the named layer/dispatch/expert/delay/combine NVTX ranges, at least two DeepGEMM kernels, DeepEP dispatch and combine kernels, and exactly one `deadzone_gpu_delay_ns` kernel. Strict captures also contain `strict_gpu_ready_allreduce`.

| condition | target differential (µs) | rank-3 delay kernel (µs) | other-rank median (µs) | measured differential (µs) | error (µs) |
|---|---:|---:|---:|---:|---:|
| timeline-original-zero | 0.000 | 21.024 | 21.024 | 0.000 | +0.000 |
| timeline-original-inside | 10.580 | 31.616 | 21.056 | 10.560 | -0.020 |
| timeline-original-outside | 84.640 | 105.824 | 20.992 | 84.832 | +0.192 |
| timeline-barrier-zero | 0.000 | 20.960 | 20.992 | -0.032 | -0.032 |
| timeline-barrier-inside | 10.580 | 31.648 | 21.024 | 10.624 | +0.044 |
| timeline-barrier-outside | 84.640 | 105.695 | 21.120 | 84.575 | -0.065 |

These are separate one-sample profiler runs. Their overall layer latencies are perturbed by profiling and are not used in the timing fits or the legacy hot-rank-only `(δ−H)+` diagnostic. The delay-kernel duration itself confirms the requested differential at the injection boundary.
