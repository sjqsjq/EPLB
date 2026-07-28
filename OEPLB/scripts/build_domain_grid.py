#!/usr/bin/env python3
"""Build tok{256,4096}_out1.jsonl for BookCorpus / HellaSwag / HumanEvalPlus,
mirroring build_length_grid.py's concat/truncate methodology, for the
domain-diversity spot-check (does optimal-window-vs-length hold outside Prover-V1?).
"""
import json, os, glob, random

MODEL_PATH = "/data/models/Qwen3-235B-A22B-FP8"
OUTDIR = "/workspace/EPLB/OEPLB/benchmarks/domain_grid"
N = 2048
TARGETS = [256, 4096]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
os.makedirs(OUTDIR, exist_ok=True)


def concat_truncate(prompts, target_len):
    all_ids = []
    for p in prompts:
        all_ids.extend(tok.encode(p))
        if len(all_ids) >= target_len:
            break
    return tok.decode(all_ids[:target_len])


def write_dataset(domain, prompts):
    rng = random.Random(2026)
    shuffled = list(prompts)
    rng.shuffle(shuffled)
    for target_len in TARGETS:
        outpath = f"{OUTDIR}/{domain}_tok{target_len}_out1.jsonl"
        picked = []
        for i in range(N):
            offset = (i * 7) % len(shuffled)
            chunk = shuffled[offset:] + shuffled[:offset]
            picked.append(concat_truncate(chunk, target_len))
        lens = [len(tok.encode(p)) for p in picked[:50]]
        print(f"{domain} tok{target_len}: n={len(picked)} sample_min={min(lens)} "
              f"sample_avg={sum(lens)/len(lens):.1f} sample_max={max(lens)}")
        with open(outpath, "w") as f:
            for i, p in enumerate(picked):
                f.write(json.dumps({
                    "id": f"{domain}_tok{target_len}_{i}",
                    "prompt": p,
                    "max_tokens": 1,
                    "temperature": 0,
                    "ignore_eos": False,
                }, ensure_ascii=False) + "\n")
        print(f"  -> {outpath}")


# ---------------- BookCorpus ----------------
print("Loading BookCorpus...")
book_files = glob.glob("/workspace/EPLB/OEPLB/benchmarks/raw_bookcorpus/books1/epubtxt/*.txt")
rng = random.Random(1)
rng.shuffle(book_files)
book_prompts = []
for fp in book_files[:300]:
    try:
        with open(fp, errors="ignore") as f:
            text = f.read(6000)
        text = text.strip()
        if len(text) > 200:
            book_prompts.append(text)
    except Exception:
        continue
    if len(book_prompts) >= 300:
        break
print(f"  loaded {len(book_prompts)} book excerpts")
write_dataset("bookcorpus", book_prompts)

# ---------------- HellaSwag ----------------
print("Loading HellaSwag...")
hs_rows = [json.loads(l) for l in open("/workspace/EPLB/OEPLB/benchmarks/raw_hellaswag_val.jsonl")]
def hs_prompt(row):
    opts = "\n".join(f"{i}. {e}" for i, e in enumerate(row["endings"]))
    return (f"Context: {row['ctx']}\n\nWhich of the following is the most plausible "
            f"continuation?\n{opts}\n\nAnswer:")
hs_prompts = [hs_prompt(r) for r in hs_rows]
print(f"  loaded {len(hs_prompts)} hellaswag prompts")
write_dataset("hellaswag", hs_prompts)

# ---------------- HumanEvalPlus ----------------
print("Loading HumanEvalPlus...")
he_rows = [json.loads(l) for l in open(
    "/root/.cache/modelscope/datasets/evalscope--humanevalplus/snapshots/master/test.jsonl")]
he_prompts = [r["prompt"] for r in he_rows]
print(f"  loaded {len(he_prompts)} humanevalplus prompts")
write_dataset("humanevalplus", he_prompts)

print("DONE")
