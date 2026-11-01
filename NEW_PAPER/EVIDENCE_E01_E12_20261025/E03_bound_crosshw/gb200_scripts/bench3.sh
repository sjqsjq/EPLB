#!/bin/bash
cd /workspace/EPLB/OEPLB/scripts
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
DS=/workspace/EPLB/OEPLB/benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
for r in r3 r4 r5; do
  python3 run_grid_bench.py _repro5_bl_$r $DS 256 > /workspace/logs/bench_bl_$r.log 2>&1
done
echo BENCH3_DONE > /workspace/logs/bench3.done
