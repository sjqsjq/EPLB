#!/usr/bin/env python3
"""Grid benchmark client: streaming, measures TTFT/TPOT, for the
sync_window x input_length (x domain) grid experiment.

Usage: python3 run_grid_bench.py <label> <dataset.jsonl> [concurrency]

Output schema matches OEPLB/benchmarks/results/gridL*.json (label, dataset,
num_requests, concurrency, ok, errors, total_time_s, total_comp_tokens, tps,
avg_output_tokens, lat_*_ms, ttft_*_ms, tpot_*_ms).
"""
import asyncio, aiohttp, json, sys, time, statistics, os

API = "http://127.0.0.1:30000/v1/completions"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results"


async def send(session, req):
    t0 = time.perf_counter()
    ttft = None
    comp_tokens = 0
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
                else:
                    choices = obj.get("choices", [])
                    if choices and choices[0].get("text"):
                        comp_tokens += 1
            total = time.perf_counter() - t0
            return {"id": req["id"], "elapsed": total, "ttft": ttft, "comp": comp_tokens}
    except Exception as ex:
        return {"id": req["id"], "elapsed": time.perf_counter() - t0, "ttft": ttft,
                "comp": comp_tokens, "error": str(ex)}


async def run(label, dataset, conc):
    with open(dataset) as f:
        all_reqs = [json.loads(l) for l in f]
    N = len(all_reqs)
    print(f"[{label}] {N} reqs from {dataset}, conc={conc}, streaming TTFT/TPOT", flush=True)

    connector = aiohttp.TCPConnector(limit=conc + 10, limit_per_host=conc + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup: open exactly `conc` concurrent connections BEFORE starting the
        # timer, via /health -- a lightweight endpoint that does NOT invoke the
        # model at all. Earlier versions sent real /v1/completions warmup
        # requests (first a slice of the real dataset -- caused N-dependent
        # double-counting when conc>=N; then a fixed dummy prompt sent `conc`
        # times), but BOTH of those go through the actual model forward pass,
        # which for OEPLB means: they get recorded by record_next_layer() and
        # advance controller._steps_since_last_check. With conc=1024 requests
        # arriving in one burst, that alone can be enough forward passes to
        # cross a small sync_window and trigger a REAL swap decision based on
        # the dummy/warmup prompt's routing distribution -- BEFORE the timed
        # run (and its real dataset) even starts. That swap is not just wasted
        # work, it can actively mis-correct the placement for the real traffic
        # that follows. /health never reaches the model, so it warms the TCP
        # connection pool with zero effect on OEPLB's internal state.
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

        async def bound_send(req):
            async with sem:
                return await send(session, req)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[bound_send(r) for r in all_reqs])
        wall = time.perf_counter() - t_start

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    comp_list = [r["comp"] for r in ok]
    comp = sum(comp_list)
    lats = [r["elapsed"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = []
    for r in ok:
        if r["comp"] and r["comp"] > 1 and r["ttft"] is not None:
            tpots.append((r["elapsed"] - r["ttft"]) / (r["comp"] - 1))

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
        "lat_p99_ms": pct(lats, 0.99),
        "ttft_mean_ms": round(statistics.mean(ttfts) * 1000, 2) if ttfts else None,
        "ttft_p50_ms": pct(ttfts, 0.50),
        "ttft_p99_ms": pct(ttfts, 0.99),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 2) if tpots else None,
        "tpot_p50_ms": pct(tpots, 0.50),
        "tpot_p99_ms": pct(tpots, 0.99),
    }
    print(f"\n  Results: {json.dumps(m, indent=2)}")
    if errors[:3]:
        print(f"  Sample errors: {errors[:3]}")
    outfile = f"{RESULT_DIR}/{label}.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  -> {outfile}")


if __name__ == "__main__":
    label = sys.argv[1]
    dataset = sys.argv[2]
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    asyncio.run(run(label, dataset, conc))
