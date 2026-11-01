#!/bin/bash
exec nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --delay=115 --duration=20 --kill=none --force-overwrite=true \
  -o /workspace/logs/prof_eplb \
  bash /workspace/logs/launch_eplb.sh
