#!/bin/bash
# Full §5.3.1 baseline reproduction on A100. Fresh server per (method, table), doc-faithful.
A100=/workspace/EPLB/OEPLB/baselines/a100
for M in identity datafore moetuner eplb_static eplb_dyn; do
  for T in 0914 freq6; do
    echo "==================== $M / $T ===================="
    bash "$A100/driver_baselines_a100.sh" "$M" "$T" 3 || echo "!!!! FAILED: $M $T"
    sleep 15
  done
done
echo "ALL_BASELINES_DONE"
