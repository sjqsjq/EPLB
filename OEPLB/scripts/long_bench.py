#!/usr/bin/env python3
"""Long-duration concurrent benchmark with streaming TTFT/TPOT measurement.
Same client shape as run_bench.py but parameterized, and reports periodic
progress so it can be watched while running in the background.

TTFT = time from request send to first streamed chunk.
TPOT = (total_elapsed - TTFT) / (completion_tokens - 1), i.e. avg per-token
decode time after the first token (undefined/None when completion_tokens<=1).
"""
import asyncio, aiohttp, json, time, sys, statistics

API = "http://localhost:30000/v1/completions"
MODEL = "/workspace/Qwen3-30B-A3B-FP8"
CONC = 1024

async def send(session, req):
    t0 = time.perf_counter()
    ttft = None
    comp_tokens = 0
    try:
        payload = {"model": MODEL, "prompt": req["prompt"], "max_tokens": req["max_tokens"],
                    "temperature": req["temperature"], "ignore_eos": req["ignore_eos"],
                    "stream": True, "stream_options": {"include_usage": True}}
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
        return {"id": req["id"], "elapsed": time.perf_counter()-t0, "ttft": ttft, "comp": comp_tokens, "error": str(ex)}

async def run(label, dataset):
    with open(dataset) as f:
        all_reqs = [json.loads(l) for l in f]
    N = len(all_reqs)
    print(f"[{label}] {N} reqs from {dataset}, conc={CONC}, streaming TTFT/TPOT", flush=True)

    conn = aiohttp.TCPConnector(limit=CONC+10)
    async with aiohttp.ClientSession(connector=conn) as session:
        for i in range(min(5, N)):
            await send(session, {"id":-1,"prompt":all_reqs[i]["prompt"],"max_tokens":32,"temperature":0,"ignore_eos":False})
        await asyncio.sleep(2)

        results = []
        t_start = time.time()
        active = set()
        idx = 0
        while idx < N or active:
            while len(active) < CONC and idx < N:
                active.add(asyncio.create_task(send(session, all_reqs[idx])))
                idx += 1
            if not active: break
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for t in done: results.append(t.result())
            if len(results) % 25 == 0:
                ok = [r for r in results if "error" not in r]
                comp = sum(r["comp"] for r in ok)
                elapsed = time.time() - t_start
                print(f"[{label}] [{elapsed:.0f}s] done={len(results)}/{N}, tok={comp}, tps={comp/elapsed:.0f}", flush=True)
        wall = time.time() - t_start

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    comp_list = [r["comp"] for r in ok]
    comp = sum(comp_list)
    lats = [r["elapsed"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [(r["elapsed"]-r["ttft"])/(r["comp"]-1) for r in ok if r["ttft"] is not None and r["comp"] > 1]

    def pctl(vals, p):
        if not vals: return None
        return round(sorted(vals)[int(len(vals)*p)]*1000, 2)

    m = {"label": label, "dataset": dataset, "num_requests": N, "concurrency": CONC,
         "ok": len(ok), "errors": len(errors), "total_time_s": round(wall, 2),
         "total_comp_tokens": comp, "tps": round(comp/wall, 1),
         "avg_output_tokens": round(statistics.mean(comp_list), 1) if comp_list else 0,
         "lat_mean_ms": round(statistics.mean(lats)*1000, 1),
         "lat_p50_ms": round(statistics.median(lats)*1000, 1),
         "lat_p99_ms": pctl(lats, 0.99),
         "ttft_mean_ms": round(statistics.mean(ttfts)*1000, 2) if ttfts else None,
         "ttft_p50_ms": pctl(ttfts, 0.50),
         "ttft_p99_ms": pctl(ttfts, 0.99),
         "tpot_mean_ms": round(statistics.mean(tpots)*1000, 2) if tpots else None,
         "tpot_p50_ms": pctl(tpots, 0.50),
         "tpot_p99_ms": pctl(tpots, 0.99)}
    print(f"[{label}] DONE: {m}", flush=True)
    with open(f"/workspace/EPLB/OEPLB/benchmarks/results/{label}.json", "w") as f:
        json.dump(m, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], sys.argv[2]))
