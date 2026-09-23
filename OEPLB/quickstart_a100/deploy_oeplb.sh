#!/bin/sh
# Deploy OEPLB into the freshly-installed SGLang 0.5.6.post2 (A100 box).
# Copies OEPLB core + the 3 reference-patched SGLang files (server_args, model_runner, topk).
set -e
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
REF=/workspace/EPLB/benchmark/sglang_src/python/sglang
OEPLB=/workspace/EPLB/OEPLB
echo "SGLANG_PATH=$SGLANG_PATH"

# 1. OEPLB core code
mkdir -p "$SGLANG_PATH/srt/managers/pb_oeplb"
cp "$OEPLB"/src/*.py "$SGLANG_PATH/srt/managers/pb_oeplb/"
echo "copied OEPLB src -> pb_oeplb/"

# 2. Diff the 3 patched files vs the fresh install (sanity: additions must be OEPLB-only)
for f in srt/server_args.py srt/model_executor/model_runner.py srt/layers/moe/topk.py; do
  echo "===== DIFF $f ====="
  diff "$SGLANG_PATH/$f" "$REF/$f" | head -80 || true
done
