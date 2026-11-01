#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 NCCL_DEBUG=WARN
for J in j4 j5; do
  echo "=== $J start $(date -u +%FT%TZ)" >> gpu_harness.log
  SCEN=$J timeout 1800 torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
    --master_port=2961$([ $J = j4 ] && echo 8 || echo 9) \
    test_swap_race_gpu.py >> gpu_harness.log 2>&1
  echo "=== $J EXIT=$? $(date -u +%FT%TZ)" >> gpu_harness.log
done
echo ALL_DONE >> gpu_harness.log
