#!/usr/bin/env python3
"""Build datasets for the full input×output grid experiment.
Reuse existing tok{L}_out1.jsonl prompts, just change max_tokens.
For out>=256, use only 512 requests (not 2048) to keep run times manageable.
"""
import json, os

SRC_DIR = "/workspace/EPLB/OEPLB/benchmarks/workload_grid"
OUT_DIR = "/workspace/EPLB/OEPLB/benchmarks/full_grid"
os.makedirs(OUT_DIR, exist_ok=True)

LENGTHS = [256, 1024, 4096]
OUTPUTS = [1, 64, 256, 1024]

for L in LENGTHS:
    src = f"{SRC_DIR}/tok{L}_out1.jsonl"
    with open(src) as f:
        all_reqs = [json.loads(line) for line in f]
    
    for OUT in OUTPUTS:
        n = 512 if OUT >= 256 else 2048
        reqs = all_reqs[:n]
        outpath = f"{OUT_DIR}/tok{L}_out{OUT}.jsonl"
        with open(outpath, "w") as f:
            for i, req in enumerate(reqs):
                req_copy = dict(req)
                req_copy["id"] = f"tok{L}_out{OUT}_{i}"
                req_copy["max_tokens"] = OUT
                req_copy["ignore_eos"] = True if OUT > 1 else False
                f.write(json.dumps(req_copy, ensure_ascii=False) + "\n")
        print(f"tok{L}_out{OUT}: {n} reqs -> {outpath}")

print("DONE")
