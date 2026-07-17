#!/usr/bin/env python3
"""Generate frozen request sequence: 500 requests, fixed prompt + fixed max_tokens + ignore_eos."""
import json, random

PROMPT_FILE = "/workspace/EPLB/OEPLB/benchmarks/prompts/real_long_unique.jsonl"
OUTPUT_FILE = "/workspace/EPLB/OEPLB/benchmarks/frozen_requests.jsonl"
NUM_REQUESTS = 500
MAX_TOKENS = 1024  # fixed output length for all requests
SEED = 2026

with open(PROMPT_FILE) as f:
    prompts = [json.loads(l)["prompt"] for l in f if l.strip()]

rng = random.Random(SEED)
rng.shuffle(prompts)

requests = []
for i in range(NUM_REQUESTS):
    requests.append({
        "id": i,
        "prompt": prompts[i % len(prompts)],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,        # greedy for reproducibility
        "ignore_eos": True,       # force exact max_tokens output
    })

with open(OUTPUT_FILE, "w") as f:
    for r in requests:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Frozen {len(requests)} requests → {OUTPUT_FILE}")
print(f"  max_tokens={MAX_TOKENS}, temperature=0, ignore_eos=True")
print(f"  {len(prompts)} unique prompts, cycled to {NUM_REQUESTS}")
