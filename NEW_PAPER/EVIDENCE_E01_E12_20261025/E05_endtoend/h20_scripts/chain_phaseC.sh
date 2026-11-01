#!/bin/bash
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
for i in $(seq 1 240); do
  grep -q "PHASE_B_DONE" $EXP/results/step6/phaseB.log && break
  sleep 15
done
grep -q "PHASE_B_DONE" $EXP/results/step6/phaseB.log || { echo "CHAIN TIMEOUT" >> $EXP/results/step6/phaseC.log; exit 1; }
cd $EXP/scripts && exec bash run_step6_phaseC.sh
