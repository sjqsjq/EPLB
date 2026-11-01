#!/usr/bin/env python3
"""bench_client_v2 — final_plan_20261019 shared measurement client.

STRICT SUPERSET of OEPLB/scripts/run_grid_bench.py (v1, sealed): identical
API endpoint, payload, /health connection warmup (zero OEPLB-state impact),
semaphore concurrency, SSE parsing and aggregate schema. Additions (v2):
  1. per-request persistence: {label}_perreq.jsonl — one JSON object per
     request: idx, id, t_send_wall_s, ttft_s, t_last_wall_s, elapsed_s,
     comp_tokens, prompt_tokens, error (absent when ok).
  2. aggregate additions: *_p95_ms, total_prompt_tokens, client_version,
     perreq_file. ALL v1 aggregate fields kept verbatim (superset).
  3. output dir: OEPLB_BENCH_OUT env (default = v1 RESULT_DIR).
Shared by ALL arms (I/E/P/F/A1-A4/P1); sha-locked in
C_configs/bench_client_v2.sha256 before any formal timing run.

Usage: python3 bench_client_v2.py <label> <dataset.jsonl> [concurrency]
"""
import asyncio, aiohttp, json, sys, time, statistics, os

API = "http://127.0.0.1:30000/v1/completions"
MODEL = os.environ.get("OEPLB_MODEL", "/data/models/Qwen3-30B-A3B-FP8")
RESULT_DIR = os.environ.get("OEPLB_BENCH_OUT",
                            "/workspace/EPLB/OEPLB/benchmarks/results")
CLIENT_VERSION = "bench_client_v2/final_plan_20261019"


async def send(session, req, idx):
    t0 = time.perf_counter()
    t_send_wall = time.time()
    ttft = None
    comp_tokens = 0
    prompt_tokens = None
    try:
        payload = {
            "model": MODEL, "prompt": req["prompt"],
            "max_tokens": req.get("max_tokens", 1),
            "temperature": req.get("temperature", 0),
            "ignore_eos": req.get("ignore_eos", False),
            "stream": True, "stream_options": {"include_usage": True},
        }
        async with session.post(API, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
            async for line in r.content:
                if not line.startswith(b"data:"):
                    continue
                chunk = line[len(b"data:"):].strip()
                if chunk == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage:
                    comp_tokens = usage.get("completion_tokens", comp_tokens)
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = usage.get("prompt_tokens")
                else:
                    choices = obj.get("choices", [])
                    if choices and choices[0].get("text"):
                        comp_tokens += 1
            total = time.perf_counter() - t0
            return {"idx": idx, "id": req["id"], "t_send_wall_s": t_send_wall,
                    "ttft_s": ttft, "t_last_wall_s": time.time(),
                    "elapsed_s": total, "comp_tokens": comp_tokens,
                    "prompt_tokens": prompt_tokens}
    except Exception as ex:
        return {"idx": idx, "id": req["id"], "t_send_wall_s": t_send_wall,
                "ttft_s": ttft, "t_last_wall_s": time.time(),
                "elapsed_s": time.perf_counter() - t0,
                "comp_tokens": comp_tokens, "prompt_tokens": prompt_tokens,
                "error": str(ex)}


async def run(label, dataset, conc):
    with open(dataset) as f:
        all_reqs = [json.loads(l) for l in f]
    N = len(all_reqs)
    print(f"[{label}] {N} reqs from {dataset}, conc={conc}, streaming TTFT/TPOT ({CLIENT_VERSION})", flush=True)

    connector = aiohttp.TCPConnector(limit=conc + 10, limit_per_host=conc + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup via /health ONLY (v1 rationale kept verbatim): opens `conc`
        # TCP connections with ZERO model-forward / OEPLB-state impact.
        health_url = f"{API.rsplit('/v1', 1)[0]}/health"

        async def _warm_one():
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    await r.read()
            except Exception:
                pass

        await asyncio.gather(*[_warm_one() for _ in range(conc)])
        await asyncio.sleep(1)

        sem = asyncio.Semaphore(conc)

        async def bound_send(i, req):
            async with sem:
                return await send(session, req, i)

        t_start = time.perf_counter()
        mono_start_ns = time.monotonic_ns()
        wall_start_s = time.time()
        results = await asyncio.gather(*[bound_send(i, r) for i, r in enumerate(all_reqs)])
        wall = time.perf_counter() - t_start
        mono_end_ns = time.monotonic_ns()
        wall_end_s = time.time()

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    comp_list = [r["comp_tokens"] for r in ok]
    comp = sum(comp_list)
    ptoks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]
    lats = [r["elapsed_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpots = []
    for r in ok:
        if r["comp_tokens"] and r["comp_tokens"] > 1 and r["ttft_s"] is not None:
            tpots.append((r["elapsed_s"] - r["ttft_s"]) / (r["comp_tokens"] - 1))

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        idx = min(int(len(s) * p), len(s) - 1)
        return round(s[idx] * 1000, 2)

    m = {
        "label": label, "dataset": dataset, "num_requests": N, "concurrency": conc,
        "ok": len(ok), "errors": len(errors),
        "total_time_s": round(wall, 2),
        "total_comp_tokens": comp,
        "tps": round(comp / wall, 1) if wall > 0 else 0,
        "avg_output_tokens": round(statistics.mean(comp_list), 1) if comp_list else 0,
        "lat_mean_ms": round(statistics.mean(lats) * 1000, 1) if lats else None,
        "lat_p50_ms": pct(lats, 0.50),
        "lat_p95_ms": pct(lats, 0.95),
        "lat_p99_ms": pct(lats, 0.99),
        "ttft_mean_ms": round(statistics.mean(ttfts) * 1000, 2) if ttfts else None,
        "ttft_p50_ms": pct(ttfts, 0.50),
        "ttft_p95_ms": pct(ttfts, 0.95),
        "ttft_p99_ms": pct(ttfts, 0.99),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 2) if tpots else None,
        "tpot_p50_ms": pct(tpots, 0.50),
        "tpot_p95_ms": pct(tpots, 0.95),
        "tpot_p99_ms": pct(tpots, 0.99),
        "mono_start_ns": mono_start_ns,
        "mono_end_ns": mono_end_ns,
        "wall_start_s": round(wall_start_s, 6),
        "wall_end_s": round(wall_end_s, 6),
        # --- v2 additions ---
        "client_version": CLIENT_VERSION,
        "total_prompt_tokens": sum(ptoks) if ptoks else None,
        "perreq_file": None,  # filled below
    }
    print(f"\n  Results: {json.dumps(m, indent=2)}")
    if errors[:3]:
        print(f"  Sample errors: {errors[:3]}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    outfile = f"{RESULT_DIR}/{label}.json"
    perreq = f"{RESULT_DIR}/{label}_perreq.jsonl"
    with open(perreq, "w") as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    m["perreq_file"] = perreq
    with open(outfile, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  -> {outfile}\n  -> {perreq} ({len(results)} rows)")


if __name__ == "__main__":
    label = sys.argv[1]
    dataset = sys.argv[2]
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    asyncio.run(run(label, dataset, conc))
