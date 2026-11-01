#!/bin/bash
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang::" 2>/dev/null
sleep 3
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [ "$used" -lt 2000 ] && break
  sleep 3
done
echo "killed; gpu_used_total=${used}MiB"
