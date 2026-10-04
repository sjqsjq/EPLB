#!/bin/bash
# 用 nsys 包裹 identity server; --delay 跳过建模+warmup(~90s), --duration 采稳态负载窗口.
# --kill=none: 采集结束后不杀 server. 采集期间需有 bench 负载(另起).
exec nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --delay=105 --duration=20 --kill=none \
  --force-overwrite=true -o /workspace/logs/prof_id \
  bash /workspace/logs/launch_identity.sh
