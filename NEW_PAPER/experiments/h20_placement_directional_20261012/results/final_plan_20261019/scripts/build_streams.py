#!/usr/bin/env python3
"""Task B stream builder — final_plan_20261019.

PRE-REGISTERED CONSTRUCTION RULES (frozen BEFORE any timing run; selection
uses source/length/domain/content-hash ONLY, never performance):

R1 Doc universe: per-domain unique contents from B_data/index (global
   content-hash dedup). code := {code, repobench-p_e, lcc_e, repobench-p,
   lcc}; prover/sharegpt/book := same-named domains. raw_hellaswag_val
   (empty prompts) excluded. One source occurrence per doc = first by
   (file, lineno); hash re-verified on extraction (abort on mismatch).
R2 Split: bucket = int(h,16) % 100 (h = sha256(stripped prompt)[:16] from
   the frozen inventory). 0-11 -> calib_screen docs, 12-24 -> calib_confirm
   docs, 25-99 -> test docs. DOC-LEVEL: calib and test share no source doc.
R3 Corpus: per (domain, split), docs sorted by h, token ids concatenated
   (spliced real-text corpus; sealed W1/W2 precedent). Tokenizer: Qwen3
   (model dir), add_special_tokens=False.
R4 Windows: pool(dom, split, L, s) = corpus[i:i+L] for i in
   range(0, len-L+1, s), deduped by sha256(token ids). Prompt text =
   tokenizer.decode(window). Labeled "真实内容构造的受控请求流" (controlled
   stream constructed from real-content token windows), NOT natural
   production distribution. No synthetic routing.
R5 Pools (test split): code L2048 s256 (E1); code L512 s256 (E3/E4);
   prover L512 s256 (E2/E3/E4); sharegpt L512 s512; book L512 s512 (E3/E4).
   Calib splits: all four domains L512 s256 + code L2048 s256.
R6 Streams:
   e1_code_stable     = full code L2048 test pool, O1 (max_tokens=1, t=0,
                        ignore_eos=false). N ADJUSTED from target 4096 per
                        plan §5.1 (unique-content capacity; registered).
   e2_prover_stable   = full prover L512 test pool, O1. N ADJUSTED from
                        target 8192 per §5.1 (registered).
   e3_multidomain_switch = blocks sharegpt[0:2048] prover[0:2048]
                        book[0:2048] code[0:2048] (pool order), O1.
                        File-level block switching per plan §5.2.
   e4_fullduplex      = E3 block prefixes: each domain's first 512 windows,
                        max_tokens=128, temperature=0, ignore_eos=true
                        (identical across arms; fixed-length performance
                        workload per §5.2, disclosed).
   calib_screen       = screen-split pools, segments (mixed phase/length,
                        stable+changing, short+long output per §4.2):
                        S1 code L2048[0:256] O1 | S2 prover[0:256] O1 |
                        S3 switch sharegpt[0:128] prover[256:384]
                        book[0:128] code_L512[0:128] O1 |
                        S4 longout sharegpt[128:256] book[128:256] O128
                        (t=0, ignore_eos=true). Segments use DISJOINT
                        window ranges. 1280 requests.
   calib_confirm      = identical composition from confirm-split pools.
R7 Determinism: no RNG anywhere; hash-ordered docs, prefix selection.
   Reruns are bit-identical. Registry + splits + manifest persisted.
"""
import json, gzip, glob, os, sys, hashlib, collections, pickle, time
from array import array
from multiprocessing import Pool

F = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019"
TOK_PATH = "/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8"
DOMGROUP = {"code": {"code", "repobench-p_e", "lcc_e", "repobench-p", "lcc"},
            "prover": {"prover"}, "sharegpt": {"sharegpt"}, "book": {"book"}}
DOMOF = {d: g for g, ds in DOMGROUP.items() for d in ds}

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------- R1: doc universe from index ----------
def collect_docs():
    need = {}          # (dom, h) -> (file, lineno, n)
    for p in sorted(glob.glob(f"{F}/B_data/index/*.jsonl.gz")):
        name = p.split("/")[-1].replace(".jsonl.gz", "")
        if name == "raw_hellaswag_val":
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                g = DOMOF.get(r.get("d"))
                if not g:
                    continue
                key = (g, r["h"])
                cand = (name, r["l"], r["n"])
                if key not in need or cand[:2] < need[key][:2]:
                    need[key] = cand
    byfile = collections.defaultdict(dict)   # file -> lineno -> (dom,h,n)
    for (dom, h), (fname, l, n) in need.items():
        byfile[fname][l] = (dom, h, n)
    return byfile

INV = {e["file"].split("/")[-1].replace(".jsonl", ""): e["file"]
       for e in json.load(open(f"{F}/B_data/pool_inventory.json"))
       if isinstance(e, dict) and "file" in e}

def read_sources(byfile):
    docs = {}    # h -> dict(dom, prompt, n_index, src, lineno)
    for fname, lines_needed in sorted(byfile.items()):
        path = INV[fname]
        got = 0
        with open(path) as f:
            for i, line in enumerate(f):
                want = lines_needed.get(i)
                if not want:
                    continue
                dom, h, n = want
                if h in docs:
                    continue
                row = json.loads(line)
                prompt = row["prompt"]
                hh = hashlib.sha256(prompt.strip().encode()).hexdigest()[:16]
                if hh != h:
                    raise SystemExit(
                        f"INTEGRITY FAIL: {fname}:{i} hash {hh} != index {h}")
                docs[h] = {"dom": dom, "prompt": prompt, "n_index": n,
                           "src": fname, "lineno": i}
                got += 1
        log(f"  read {fname}: +{got} docs (total {len(docs)})")
    return docs

# ---------- R3: tokenize ----------
_tok = None
def _init_tok():
    global _tok
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

def _tok_one(item):
    h, prompt = item
    ids = _tok.encode(prompt, add_special_tokens=False)
    return h, ids

def tokenize_docs(docs):
    cache_p = f"{F}/B_data/tokcache.pkl"
    cache = {}
    if os.path.exists(cache_p):
        with open(cache_p, "rb") as f:
            cache = pickle.load(f)
        log(f"  tokcache loaded: {len(cache)}")
    todo = [(h, d["prompt"]) for h, d in docs.items() if h not in cache]
    if todo:
        log(f"  tokenizing {len(todo)} docs on 16 workers ...")
        t0 = time.time()
        with Pool(16, initializer=_init_tok) as pool:
            for k, (h, ids) in enumerate(pool.imap_unordered(_tok_one, todo, chunksize=16)):
                cache[h] = ids
                if (k + 1) % 2000 == 0:
                    log(f"    {k+1}/{len(todo)} ({time.time()-t0:.0f}s)")
        with open(cache_p, "wb") as f:
            pickle.dump(cache, f, protocol=4)
    return cache

# ---------- R2: split ----------
def split_of(h):
    b = int(h, 16) % 100
    return "screen" if b < 12 else ("confirm" if b < 25 else "test")

# ---------- R4: windows ----------
def thash(ids):
    return hashlib.sha256(array("q", ids).tobytes()).hexdigest()[:16]

def make_pool(corpus, L, s):
    seen, out = set(), []
    if len(corpus) < L:
        return out
    for i in range(0, len(corpus) - L + 1, s):
        w = corpus[i:i + L]
        th = thash(w)
        if th in seen:
            continue
        seen.add(th)
        out.append((i, w))
    return out

def main():
    t00 = time.time()
    log("STEP1 doc universe")
    byfile = collect_docs()
    log(f"  needed: {sum(len(v) for v in byfile.values())} docs across {len(byfile)} files")
    docs = read_sources(byfile)
    log(f"STEP2 tokenize ({len(docs)} docs)")
    tok = tokenize_docs(docs)
    log("STEP3 corpora + splits")
    per = collections.defaultdict(list)      # (dom, split) -> [(h, ids)]
    manifest = {}
    for h, d in docs.items():
        sp = split_of(h)
        ids = tok[h]
        per[(d["dom"], sp)].append((h, ids))
        manifest[h] = {"dom": d["dom"], "n_tok": len(ids), "n_index": d["n_index"],
                       "src": d["src"], "lineno": d["lineno"], "split": sp}
    corpus = {}
    caps = {}
    for key, lst in sorted(per.items()):
        lst.sort(key=lambda x: x[0])          # deterministic: hash order
        cat = []
        for _, ids in lst:
            cat.extend(ids)
        corpus[key] = cat
        caps[f"{key[0]}|{key[1]}"] = {"docs": len(lst), "tok": len(cat)}
        log(f"  corpus {key[0]}/{key[1]}: {len(lst)} docs, {len(cat)} tok")
    log("STEP4 pools")
    POOLS = {}
    def pool(dom, sp, L, s):
        k = (dom, sp, L, s)
        if k not in POOLS:
            POOLS[k] = make_pool(corpus[(dom, sp)], L, s)
            log(f"  pool {dom}/{sp} L{L} s{s}: {len(POOLS[k])} windows")
        return POOLS[k]

    dec_tok = None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    dec_tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

    def rows(dom, sp, L, s, sl, mt, te, ig, idpfx):
        p = pool(dom, sp, L, s)[sl]
        out = []
        for j, (off, w) in enumerate(p):
            out.append({"id": f"{idpfx}_{j:05d}",
                        "prompt": dec_tok.decode(w),
                        "max_tokens": mt, "temperature": te, "ignore_eos": ig,
                        "domain": dom,
                        "src_window": f"{dom}|{sp}|L{L}|s{s}@{off}"})
        return out

    O1 = dict(mt=1, te=0, ig=False)
    O128 = dict(mt=128, te=0, ig=True)
    log("STEP5 streams")
    streams = {}
    streams["e1_code_stable"] = rows("code", "test", 2048, 256, slice(None), **O1, idpfx="e1_code")
    streams["e2_prover_stable"] = rows("prover", "test", 512, 256, slice(None), **O1, idpfx="e2_prover")
    e3 = []
    for dom in ("sharegpt", "prover", "book", "code"):
        s_ = 512 if dom in ("sharegpt", "book") else 256
        blk = rows(dom, "test", 512, s_, slice(0, 2048), **O1, idpfx=f"e3_{dom}")
        assert len(blk) == 2048, f"E3 block {dom}: only {len(blk)} windows < 2048"
        e3.extend(blk)
    streams["e3_multidomain_switch"] = e3
    e4 = []
    for dom in ("sharegpt", "prover", "book", "code"):
        s_ = 512 if dom in ("sharegpt", "book") else 256
        blk = rows(dom, "test", 512, s_, slice(0, 512), **O128, idpfx=f"e4_{dom}")
        assert len(blk) == 512, f"E4 block {dom}: only {len(blk)}"
        e4.extend(blk)
    streams["e4_fullduplex"] = e4

    def calib(sp, tag):
        segs = []
        segs += [("S1_code_stable_O1", rows("code", sp, 2048, 256, slice(0, 256), **O1, idpfx=f"{tag}s1_code"))]
        segs += [("S2_prover_stable_O1", rows("prover", sp, 512, 256, slice(0, 256), **O1, idpfx=f"{tag}s2_prover"))]
        sw = []
        sw += rows("sharegpt", sp, 512, 256, slice(0, 128), **O1, idpfx=f"{tag}s3_sharegpt")
        sw += rows("prover", sp, 512, 256, slice(256, 384), **O1, idpfx=f"{tag}s3_prover")
        sw += rows("book", sp, 512, 256, slice(0, 128), **O1, idpfx=f"{tag}s3_book")
        sw += rows("code", sp, 512, 256, slice(0, 128), **O1, idpfx=f"{tag}s3_code")
        segs += [("S3_switch_O1", sw)]
        lo = []
        lo += rows("sharegpt", sp, 512, 256, slice(128, 256), **O128, idpfx=f"{tag}s4_sharegpt")
        lo += rows("book", sp, 512, 256, slice(128, 256), **O128, idpfx=f"{tag}s4_book")
        segs += [("S4_longout_O128", lo)]
        out = []
        for name, rr in segs:
            out.extend(rr)
        return out, [(name, len(rr)) for name, rr in segs]

    streams["calib_screen"], seg_s = calib("screen", "calibS")
    streams["calib_confirm"], seg_c = calib("confirm", "calibC")

    log("STEP6 write + registry")
    os.makedirs(f"{F}/B_data/streams", exist_ok=True)
    reg = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%S+0000", time.gmtime()),
           "rules": "see scripts/build_streams.py docstring R1-R7 (pre-registered)",
           "corpus_caps": caps,
           "pool_sizes": {f"{d}|{sp}|L{L}|s{s}": len(v) for (d, sp, L, s), v in POOLS.items()},
           "calib_segments": {"screen": seg_s, "confirm": seg_c},
           "N_adjustments": {}, "streams": {}}
    e1_target, e2_target = 4096, 8192
    n1 = len(streams["e1_code_stable"]); n2 = len(streams["e2_prover_stable"])
    reg["N_adjustments"]["E1"] = {
        "plan_target": e1_target, "actual": n1,
        "reason": "code unique-content capacity (333 docs / 1.108M tok; doc-level calib/test split; L2048 s256 windows). Adjusted per plan §5.1 BEFORE freeze; never 32x repeats as independent."}
    reg["N_adjustments"]["E2"] = {
        "plan_target": e2_target, "actual": n2,
        "reason": "prover unique-content capacity (4200 docs / 1.410M tok, p50=285; doc-level split; L512 s256 spliced windows). Adjusted per plan §5.1 BEFORE freeze."}
    for name, rr in streams.items():
        path = f"{F}/B_data/streams/{name}.jsonl"
        with open(path, "w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # uniqueness (content level) + length stats on the FINAL prompts
        hs = set(); import re
        for r in rr:
            hs.add(hashlib.sha256(r["prompt"].strip().encode()).hexdigest()[:16])
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        reg["streams"][name] = {"n": len(rr), "uniq_text_hash": len(hs),
                                "sha256": sha, "path": path}
        log(f"  {name}: n={len(rr)} uniq={len(hs)} sha={sha[:12]}")
        assert len(hs) == len(rr), f"{name}: duplicate content!"
    # token-length round-trip check (sample 100/stream)
    import random
    rng = random.Random(20261019)
    for name, rr in streams.items():
        samp = rng.sample(rr, min(100, len(rr)))
        lens = sorted(len(dec_tok.encode(r["prompt"], add_special_tokens=False)) for r in samp)
        p = lambda q: lens[min(int(len(lens) * q), len(lens) - 1)]
        reg["streams"][name]["retok_len_p5_p50_p95"] = [p(.05), p(.5), p(.95)]
        log(f"  {name}: retokenized len p5/p50/p95 = {p(.05)}/{p(.5)}/{p(.95)}")
    json.dump(manifest, open(f"{F}/B_data/doc_manifest.json", "w"))
    json.dump({h: split_of(h) for h in docs}, open(f"{F}/B_data/splits.json", "w"))
    json.dump(reg, open(f"{F}/B_data/registry.json", "w"), indent=1, ensure_ascii=False)
    log(f"DONE in {time.time()-t00:.0f}s — registry.json written")

if __name__ == "__main__":
    _init_tok if False else None
    main()
