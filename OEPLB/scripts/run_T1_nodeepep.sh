#!/bin/bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
exec python3 -m sglang.launch_server --model-path /workspace/Qwen3-30B-A3B-FP8 --tp 4 --ep-size 4 --attention-backend flashinfer --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code
