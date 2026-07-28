#!/usr/bin/env python3
"""Build datasets for the comprehensive input×output grid.
Reuse existing Prover-V1 prompts, adjust max_tokens and request count.
"""
import json, os

SRC_DIR = "/workspace/EPLB/OEPLB/benchmarks/workload_grid"
OUT_DIR = "/workspace/EPLB/OEPLB/benchmarks/comprehensive_grid"
os.makedirs(OUT_DIR, exist_ok=True)

# Design:
# Input lengths: 256, 512, 1024, 2048, 4096
# Output lengths: 1, 64, 256, 1024
# Requests: 1024 for most, 512 for L=4096 with O>=256 (those are very slow)
LENGTHS = [256, 512, 1024, 2048, 4096]
OUTPUTS = [1, 64, 256, 1024]

for L in LENGTHS:
    src = f"{SRC_DIR}/tok{L}_out1.jsonl"
    with open(src) as f:
        all_reqs = [json.loads(line) for line in f]
    for O in OUTPUTS:
        if L >= 4096 and O >= 256:
            n = 512
        else:
            n = 1024
        reqs = all_reqs[:n]
        outpath = f"{OUT_DIR}/L{L}_O{O}.jsonl"
        with open(outpath, "w") as f:
            for i, req in enumerate(reqs):
                r = dict(req)
                r["id"] = f"L{L}_O{O}_{i}"
                r["max_tokens"] = O
                r["ignore_eos"] = True if O > 1 else False
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"L{L}_O{O}: {n} reqs")
print("DONE")
