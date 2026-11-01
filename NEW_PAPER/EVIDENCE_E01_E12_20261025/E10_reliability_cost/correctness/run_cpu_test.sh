#!/bin/bash
# Fixed per-job timeout: 600s hard. gloo on loopback. 8 procs. Full log.
cd "$(dirname "$0")"
export GLOO_SOCKET_IFNAME=lo TP_SOCKET_IFNAME=lo OMP_NUM_THREADS=1
timeout 600 torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29617 \
  test_swap_race_cpu.py > cpu_test_full.log 2>&1
echo "EXIT=$?" >> cpu_test_full.log
