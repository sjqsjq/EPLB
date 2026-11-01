#!/bin/bash
exec nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --delay=105 --duration=20 --kill=none --force-overwrite=true \
  -o /workspace/logs/prof_oe \
  bash /workspace/logs/launch_oeplb.sh
