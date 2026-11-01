#!/usr/bin/env python3
"""Task B step 1: read-only inventory of all real data pools.
Per file: rows, unique-content (sha256 of stripped prompt), domain histogram,
token-length percentiles per domain (Qwen3 tokenizer, add_special_tokens=False).
Writes compact per-row index jsonl.gz for later manifest building.
NO performance data is used for selection (governance)."""
import json, gzip, hashlib, os, sys, re
from multiprocessing import Pool

TOK_PATH = "/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8"
D = "/data/minghua/sjq/OEPLBdata/datasets/multi_domain"
W = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/inputs/fullservice_v2/workload"
OUT = os.environ["FIN"] + "/B_data"
FILES = []
for f in sorted(os.listdir(D)):
    if f.endswith(".jsonl"):
        FILES.append(("pool", os.path.join(D, f)))
for f in sorted(os.listdir(W)):
    if f.endswith(".jsonl"):
        FILES.append(("workload", os.path.join(W, f)))

_tok = None
def tok():
    global _tok
    if _tok is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    return _tok

def domain_of(o):
    if o.get("domain"):
        return str(o["domain"])
    m = re.match(r"(?:seg\d+_)?([a-z0-9_]+?)_\d+$", str(o.get("id", "")))
    return m.group(1) if m else "unknown"

def work(args):
    tag, path = args
    t = tok()
    name = os.path.basename(path).replace(".jsonl", "")
    idx_path = os.path.join(OUT, "index", name + ".jsonl.gz")
    stats = {"file": path, "tag": tag, "rows": 0, "bad_json": 0, "domains": {}, "uniq_content": {}}
    lengths = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fin, gzip.open(idx_path, "wt", encoding="utf-8") as fout:
        for ln, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            p = o.get("prompt", "")
            h = hashlib.sha256(p.strip().encode("utf-8")).hexdigest()[:16]
            dom = domain_of(o)
            n = len(t(p, add_special_tokens=False).input_ids)
            stats["rows"] += 1
            stats["domains"][dom] = stats["domains"].get(dom, 0) + 1
            uc = stats["uniq_content"].setdefault(dom, {})
            uc[h] = uc.get(h, 0) + 1
            lengths.setdefault(dom, []).append(n)
            fout.write(json.dumps({"l": ln, "id": o.get("id"), "d": dom, "h": h, "n": n,
                                   "mt": o.get("max_tokens"), "ig": o.get("ignore_eos", False)}, ensure_ascii=False) + "\n")
    stats["uniq_content"] = {k: len(v) for k, v in stats["uniq_content"].items()}
    stats["len_pct"] = {}
    for dom, arr in lengths.items():
        arr.sort()
        n = len(arr)
        pct = lambda q: arr[min(n - 1, int(q * n))]
        stats["len_pct"][dom] = {"n": n, "min": arr[0], "p10": pct(.10), "p25": pct(.25), "p50": pct(.50),
                                  "p75": pct(.75), "p90": pct(.90), "max": arr[-1]}
    return stats

if __name__ == "__main__":
    with Pool(16) as pool:
        results = pool.map(work, FILES)
    with open(os.path.join(OUT, "pool_inventory.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("INVENTORY_DONE", len(results), "files")
