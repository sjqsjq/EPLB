#!/bin/bash
# Serial GPU correctness queue (followup phase 2, budget <=6 jobs: J4 J5 J1 J2 J3 J6).
# Halt on first FAIL (fix-round budget governance).
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
L=$G/correctness/gpu_jobs/LEDGER.tsv
echo -e "job\tstart_utc\tend_utc\trc\tverdict" > $L
echo "# deployed async_swapper sha: $(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/async_swapper.py | cut -c1-16)" >> $L
echo "# deployed controller   sha: $(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/controller.py | cut -c1-16)" >> $L

run_harness() { # $1=j4|j5
  local J=$1 S=$(date -u +%FT%TZ) RC V
  (cd $G/correctness/gpu_harness && export OMP_NUM_THREADS=2 NCCL_DEBUG=WARN && \
   SCEN=$J timeout 1800 torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
     --master_port=2962$([ $J = j4 ] && echo 1 || echo 2) \
     test_swap_race_gpu.py > gpu_${J}.log 2>&1)
  RC=$?
  grep -q "GPU_$(echo $J | tr a-z A-Z)_PASS" $G/correctness/gpu_harness/gpu_${J}.log && [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$J\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  [ $V = PASS ]
}
run_server() { # JOB ARM PARAMS STREAM NREQ CONC BT MIN_SWAPS
  local S=$(date -u +%FT%TZ) RC V
  bash $G/correctness/gpu_jobs/run_server_job.sh "$@"
  RC=$?
  [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$1\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  [ $V = PASS ]
}

run_harness j4 || { echo HALTED_AT=j4 >> $L; exit 1; }
run_harness j5 || { echo HALTED_AT=j5 >> $L; exit 1; }
run_server J1_E3_F F "1 16 16" e3_multidomain_switch.jsonl 8192 256 3600 4 || { echo HALTED_AT=J1 >> $L; exit 1; }
run_server J2_E3_P P ""        e3_multidomain_switch.jsonl 8192 256 3600 4 || { echo HALTED_AT=J2 >> $L; exit 1; }
run_server J3_E4_F F "1 16 16" e4_fullduplex.jsonl         2048 256 3600 6 || { echo HALTED_AT=J3 >> $L; exit 1; }
run_server J6_E4_P P ""        e4_fullduplex.jsonl         2048 256 3600 6 || { echo HALTED_AT=J6 >> $L; exit 1; }
echo -e "ALL\t-\t$(date -u +%FT%TZ)\t0\tQUEUE_PASS" >> $L
