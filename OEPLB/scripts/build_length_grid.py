#!/usr/bin/env python3
"""Build tok{512,1024,4096}_out1.jsonl for the sync_window x input_length grid experiment.
Reuses Prover-V1 source and concat/truncate method from /tmp/build_workload_grid.py.
"""
import json, os, random
from transformers import AutoTokenizer

MODEL_PATH = "/root/.cache/modelscope/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master"
PROVER_DATA = "/tmp/prover_data/dataset.jsonl"
OUTDIR = "/workspace/EPLB/OEPLB/benchmarks/workload_grid"
N = 2048

SYS = "You are a helpful math assistant skilled in Lean 4 theorem proving."

def build_prompt(row):
    body = f"{row['header']}\n\n{row['formal_statement']}\n\nGoal: {row['goal']}\n\nPlease provide a proof."
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"

def concat_truncate(tok, prompts, target_len):
    all_ids = []
    for p in prompts:
        all_ids.extend(tok.encode(p))
        if len(all_ids) >= target_len:
            break
    return tok.decode(all_ids[:target_len])

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading source data...")
rows = [json.loads(l) for l in open(PROVER_DATA)]
all_prompts = [build_prompt(r) for r in rows]
all_encoded = [(p, tok.encode(p)) for p in all_prompts]
print(f"Source: {len(all_encoded)} prompts, token range: "
      f"{min(len(e) for _,e in all_encoded)}-{max(len(e) for _,e in all_encoded)}, "
      f"avg={sum(len(e) for _,e in all_encoded)/len(all_encoded):.0f}")

# For 512/1024: try filter first, fall back to truncation if not enough
# For 4096: always concatenate
TARGETS = [
    (512,  (450, 600)),
    (1024, (900, 1150)),
    (4096, None),
]

os.makedirs(OUTDIR, exist_ok=True)

for target_len, filt_range in TARGETS:
    outpath = f"{OUTDIR}/tok{target_len}_out1.jsonl"
    picked = []

    if filt_range is not None:
        lo, hi = filt_range
        for prompt, ids in all_encoded:
            if lo <= len(ids) <= hi:
                picked.append(prompt)
            if len(picked) >= N:
                break
        if len(picked) < N:
            print(f"  tok{target_len}: only {len(picked)} in [{lo},{hi}], "
                  f"filling with truncated longer prompts")
            for prompt, ids in all_encoded:
                if len(ids) > target_len:
                    picked.append(tok.decode(ids[:target_len]))
                if len(picked) >= N:
                    break
        if len(picked) < N:
            print(f"  tok{target_len}: still only {len(picked)}, "
                  f"filling with concatenated prompts")
            rng = random.Random(2026 + target_len)
            shuffled = list(all_prompts)
            rng.shuffle(shuffled)
            while len(picked) < N:
                idx = len(picked)
                offset = (idx * 7) % len(shuffled)
                chunk = shuffled[offset:] + shuffled[:offset]
                picked.append(concat_truncate(tok, chunk, target_len))
    else:
        rng = random.Random(2026)
        shuffled = list(all_prompts)
        rng.shuffle(shuffled)
        for i in range(N):
            offset = (i * 7) % len(shuffled)
            chunk = shuffled[offset:] + shuffled[:offset]
            picked.append(concat_truncate(tok, chunk, target_len))

    picked = picked[:N]
    lens = [len(tok.encode(p)) for p in picked]
    print(f"tok{target_len}: n={len(picked)} "
          f"actual min={min(lens)} avg={sum(lens)/len(lens):.1f} max={max(lens)}")

    with open(outpath, "w") as f:
        for i, p in enumerate(picked):
            f.write(json.dumps({
                "id": f"tok{target_len}_{i}",
                "prompt": p,
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": False,
            }, ensure_ascii=False) + "\n")
    print(f"  -> {outpath}")

print("DONE")
