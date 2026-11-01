#!/usr/bin/env python3
"""P1 stream builder — final_plan_20261019 Task P1 (plan §10).

PRE-REGISTERED construction (frozen BEFORE any P1 timing; reuses sealed
build_streams.py rules R1-R4 verbatim — same doc universe, same doc-level
split, same hash-ordered spliced corpora, same window/dedup rule):

P1-R1 Pools: pool(dom, test, L=2048, s=256) for dom in sharegpt/prover/book/code.
      code pool must be bit-identical to the sealed e1 pool (asserted against
      e1_code_stable.jsonl first 512 rows: prompt text equality).
P1-R2 Quota: first 512 windows of each pool (pool order = corpus offset order,
      deterministic prefix selection, no RNG). N = 4*512 = 2048.
      (N=2048 tier of plan §10.3 chosen: all four pools >= 512 windows,
      capacity verified BEFORE construction; frozen here.)
P1-R3 Sampling: max_tokens=128, temperature=0, ignore_eos=true for EVERY row
      (fixed-length performance workload per §5.2/§10.3; identical across arms).
P1-R4 Order: round-robin interleave, j=0..511, domain order
      (sharegpt, prover, book, code) — same domain order as sealed E3 blocks.
      Interleaved (not blocked) so routing composition is stationary over the
      run: P1 measures KV/redundancy pressure, NOT domain switching.
P1-R5 Determinism: no RNG anywhere; rerun is bit-identical.
"""
import json, os, sys, hashlib, collections, time
sys.path.insert(0, "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019/scripts")
import build_streams as bs

F = bs.F
L, S, QUOTA = 2048, 256, 512
DOMS = ("sharegpt", "prover", "book", "code")
OUT = f"{F}/B_data/streams/p1_longio.jsonl"
REG = f"{F}/B_data/registry_p1.json"

def log(*a):
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}]", *a, flush=True)

t0 = time.time()
log("STEP1 docs")
byfile = bs.collect_docs()
docs = bs.read_sources(byfile)
tok = bs.tokenize_docs(docs)
log(f"  {len(docs)} docs, tokcache {len(tok)}")

log("STEP2 test corpora (hash-ordered concat, R3 rule)")
per = collections.defaultdict(list)
for h, d in docs.items():
    if bs.split_of(h) == "test" and d["dom"] in DOMS:
        per[d["dom"]].append((h, tok[h]))
corpus = {}
for dom in DOMS:
    lst = sorted(per[dom], key=lambda x: x[0])
    cat = []
    for _, ids in lst:
        cat.extend(ids)
    corpus[dom] = cat
    log(f"  {dom}/test: {len(lst)} docs, {len(cat)} tok")

log("STEP3 pools L2048 s256")
pools = {}
for dom in DOMS:
    p = bs.make_pool(corpus[dom], L, S)
    pools[dom] = p
    log(f"  pool {dom}: {len(p)} windows")
    assert len(p) >= QUOTA, f"{dom} pool {len(p)} < {QUOTA}"

log("STEP4 integrity: code pool vs sealed e1 stream (first 512)")
e1 = [json.loads(l) for l in open(f"{F}/B_data/streams/e1_code_stable.jsonl")][:QUOTA]
for j in range(QUOTA):
    off, w = pools["code"][j]
    assert f"code|test|L{L}|s{S}@{off}" == e1[j]["src_window"], f"code window {j} offset mismatch"
log("  offsets match; prompt text equality checked during decode step")

log("STEP5 decode + assemble (round-robin)")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer
dec = AutoTokenizer.from_pretrained(bs.TOK_PATH, trust_remote_code=True)

rows = []
for j in range(QUOTA):
    for dom in DOMS:
        off, w = pools[dom][j]
        prompt = dec.decode(w)
        if dom == "code":
            assert prompt == e1[j]["prompt"], f"code row {j} prompt != sealed e1 row"
        rows.append({"id": f"p1_{dom}_{j:05d}", "prompt": prompt,
                     "max_tokens": 128, "temperature": 0, "ignore_eos": True,
                     "domain": dom, "src_window": f"{dom}|test|L{L}|s{S}@{off}"})
log(f"  {len(rows)} rows; code-pool bit-identity vs e1 CONFIRMED")

hs = {hashlib.sha256(r["prompt"].strip().encode()).hexdigest()[:16] for r in rows}
assert len(hs) == len(rows), "duplicate content in P1 stream!"

tmp = OUT + ".tmp"
with open(tmp, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(tmp, OUT)
sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
log(f"  wrote {OUT} sha={sha}")

log("STEP6 retokenized length check (sample 100, sealed rng seed 20261019)")
import random
rng = random.Random(20261019)
samp = rng.sample(rows, 100)
lens = sorted(len(dec.encode(r["prompt"], add_special_tokens=False)) for r in samp)
p = lambda q: lens[min(int(len(lens) * q), len(lens) - 1)]
retok = [p(.05), p(.5), p(.95)]
log(f"  retok len p5/p50/p95 = {retok}")

reg = {
    "built_utc": time.strftime("%FT%T+0000", time.gmtime()),
    "purpose": "P1 (plan §10): redundancy-vs-KV-pressure. Real long-in/long-out controlled stream.",
    "rules": __doc__,
    "stream": {
        "name": "p1_longio", "path": OUT, "n": len(rows), "uniq_text_hash": len(hs),
        "sha256": sha, "quota_per_domain": QUOTA, "domains_order": list(DOMS),
        "interleave": "round-robin j-major (stationary composition)",
        "max_tokens": 128, "temperature": 0, "ignore_eos": True,
        "retok_len_p5_p50_p95": retok,
        "pool_sizes_test_L2048_s256": {d: len(pools[d]) for d in DOMS},
        "code_pool_identity": "first 512 windows bit-identical to sealed e1_code_stable (asserted)",
    },
    "n_choice": "N=2048 tier (§10.3 'N=2048 or 4096 by CPU inventory'): all four test-split L2048 pools >= 512; frozen before construction; 4096 would need 1024/domain — also feasible but 2048 keeps P1 bench wall bounded; recorded here as the frozen choice.",
    "kv_capacity_evidence": {
        "source": "server.log 'KV Cache is allocated. #tokens:' of EXISTING boots (capacity numbers only, not performance endpoints)",
        "worst_arm_E_R32_tokens": 190307, "E_R16_tokens": 208750, "E_R0_tokens": 227168,
        "P_I_F_tokens": {"P": 227292, "I": 227269, "F_1_16_16": 227258},
        "per_request_budget_tokens": 2176,
        "inflight_capacity_worst_arm": 87,
    },
}
json.dump(reg, open(REG, "w"), indent=1, ensure_ascii=False)
log(f"DONE in {time.time()-t0:.0f}s — {REG} written")
