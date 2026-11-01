"""
Send requests in controlled phases, dump per-window expert load distribution
to analyze swap effectiveness and hot-spot patterns.
"""
import requests, json, time, sys
import numpy as np

URL = "http://localhost:30000/v1/completions"
MODEL = "/workspace/Qwen3-30B-A3B-FP8"

# Load different domain prompts for phase transitions
with open("/data/minghua/sjq/longbench_tokenized.jsonl") as f:
    all_samples = [json.loads(l) for l in f]

# Group by type
from collections import defaultdict
by_type = defaultdict(list)
for s in all_samples:
    by_type[s['type']].append(s['text'])

# Select 3 distinct domains
code_prompts = by_type['lcc'][:50]
qa_prompts = by_type['hotpotqa'][:50]
chinese_prompts = by_type['dureader'][:50]

def send_batch(prompts, label, n=10, max_tokens=32):
    """Send n requests from a domain, measure time."""
    t0 = time.time()
    for i in range(min(n, len(prompts))):
        p = prompts[i][:2000]  # truncate to keep fast
        r = requests.post(URL, json={
            "model": MODEL,
            "prompt": p,
            "max_tokens": max_tokens,
            "temperature": 0,
        }, timeout=60)
    elapsed = time.time() - t0
    print(f"  [{label}] sent {n} requests in {elapsed:.1f}s")

# Phase 1: Code domain (establish initial routing pattern)
print("Phase 1: CODE domain (10 requests)")
send_batch(code_prompts, "code", n=10)

# Phase 2: QA domain (different routing pattern → should trigger swap)  
print("Phase 2: QA domain (10 requests)")
send_batch(qa_prompts, "qa", n=10)

# Phase 3: Chinese domain (another routing shift)
print("Phase 3: CHINESE domain (10 requests)")
send_batch(chinese_prompts, "chinese", n=10)

# Phase 4: Back to Code (does previous swap help or hurt?)
print("Phase 4: CODE again (10 requests)")
send_batch(code_prompts[10:], "code2", n=10)

print("\nDone. Check server logs for PB-OEPLB-DIAG entries to see swap trajectory.")
