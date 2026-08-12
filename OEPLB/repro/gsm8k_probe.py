"""GSM8K exact-match probe against a running SGLang server.

Used pre- and post-swap on the SAME server instance, so the only difference
between the two probes is that PB-OEPLB has physically moved expert weights
between GPUs in between.  Reports both the task metric (accuracy) and the much
sharper signal: how many of the N greedy decodes are byte-identical.
"""
import json, re, sys, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = sys.argv[1]
MODEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
PARQ = "/data/minghua/keyi/gsm8k/test.parquet"
API = "http://127.0.0.1:30000/v1/chat/completions"
WORKERS = 32

import pandas as pd
df = pd.read_parquet(PARQ).head(N)   # schema: prompt=[{role,content}], reward_model.ground_truth

def norm(x):
    return str(x).replace(",", "").replace("$", "").strip().rstrip(".")

def pred(t):
    """The dataset prompt already asks for the answer after '####'."""
    m = re.findall(r"####\s*\$?(-?[\d,]+\.?\d*)", t)
    if m:
        return norm(m[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t.replace("$", ""))
    return norm(nums[-1]) if nums else None

def one(q):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": q}],
                       "max_tokens": 512, "temperature": 0.0, "top_p": 1.0,
                       "seed": 0}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as ex:
            err = str(ex)
    return "<<ERROR>> " + err

qs = [str(p[0]["content"]) for p in df["prompt"].tolist()]
gs = [norm(r["ground_truth"]) for r in df["reward_model"].tolist()]
with ThreadPoolExecutor(WORKERS) as ex:
    outs = list(ex.map(one, qs))

ok = [pred(o) is not None and pred(o) == g for o, g in zip(outs, gs)]
acc = sum(ok) / len(ok)
json.dump({"n": len(ok), "acc": acc, "correct": ok, "outputs": outs,
           "md5": [hashlib.md5(o.encode()).hexdigest() for o in outs]},
          open(OUT, "w"))
print(f"saved {OUT}  n={len(ok)}  acc={100*acc:.2f}%  "
      f"errors={sum(1 for o in outs if o.startswith('<<ERROR>>'))}")
