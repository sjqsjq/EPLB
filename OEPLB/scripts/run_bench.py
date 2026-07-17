#!/usr/bin/env python3
"""Deterministic benchmark: send frozen requests, non-streaming, get precise token counts."""
import asyncio, aiohttp, json, time, sys, statistics

API = "http://localhost:30000/v1/completions"
MODEL = "/workspace/Qwen3-30B-A3B-FP8"
FROZEN = "/workspace/EPLB/OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl"
CONC = 128

async def send(session, req):
    t0 = time.perf_counter()
    try:
        payload = {
            "model": MODEL,
            "prompt": req["prompt"],
            "max_tokens": req["max_tokens"],
            "temperature": req["temperature"],
            "ignore_eos": req["ignore_eos"],
        }
        async with session.post(API, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
            res = await r.json()
            total = time.perf_counter() - t0
            usage = res.get("usage", {})
            comp = usage.get("completion_tokens", 0)
            prompt_tok = usage.get("prompt_tokens", 0)
            return {"id": req["id"], "elapsed": total, "comp": comp, "prompt_tok": prompt_tok}
    except Exception as ex:
        return {"id": req["id"], "elapsed": time.perf_counter()-t0, "error": str(ex)}

async def run(label):
    with open(FROZEN) as f:
        all_reqs = [json.loads(l) for l in f]
    N = len(all_reqs)

    print(f"\n{'='*70}")
    print(f"  {label} | {N} frozen requests | conc={CONC} | non-streaming")
    print(f"{'='*70}")

    conn = aiohttp.TCPConnector(limit=CONC+10)
    async with aiohttp.ClientSession(connector=conn) as session:
        print("  [Warmup]...")
        for i in range(min(5, N)):
            await send(session, {"id":-1,"prompt":all_reqs[i]["prompt"],"max_tokens":32,"temperature":0,"ignore_eos":False})
        await asyncio.sleep(3)

        print(f"  [Sending {N} requests, conc={CONC}]...")
        results = []
        t_start = time.time()
        active = set()
        idx = 0

        while idx < N or active:
            while len(active) < CONC and idx < N:
                task = asyncio.create_task(send(session, all_reqs[idx]))
                active.add(task)
                idx += 1
            if not active: break
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for t in done: results.append(t.result())
            if len(results) % 50 == 0:
                ok = [r for r in results if "error" not in r]
                comp = sum(r["comp"] for r in ok)
                elapsed = time.time() - t_start
                print(f"  [{elapsed:.0f}s] done={len(results)}/{N}, tok={comp}, tps={comp/elapsed:.0f}")

        wall = time.time() - t_start

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    comp_list = [r["comp"] for r in ok]
    comp = sum(comp_list)
    lats = [r["elapsed"] for r in ok]
    # TPOT = (E2E - estimated_TTFT) / comp_tokens per request
    # Without streaming we estimate TTFT from prompt_tokens
    tpots = []
    for r in ok:
        if r["comp"] > 1:
            tpots.append(r["elapsed"] / r["comp"])  # approximate per-token time

    m = {
        "label": label, "num_requests": N, "concurrency": CONC,
        "ok": len(ok), "errors": len(errors),
        "total_time_s": round(wall, 2),
        "total_comp_tokens": comp,
        "tps": round(comp / wall, 1),
        "avg_output_tokens": round(statistics.mean(comp_list), 1) if comp_list else 0,
        "expected_output_tokens": all_reqs[0]["max_tokens"],
        "lat_mean_ms": round(statistics.mean(lats)*1000, 1),
        "lat_p50_ms": round(statistics.median(lats)*1000, 1),
        "lat_p99_ms": round(sorted(lats)[int(len(lats)*0.99)]*1000, 1),
        "tpot_approx_mean_ms": round(statistics.mean(tpots)*1000, 2) if tpots else None,
        "tpot_approx_p50_ms": round(statistics.median(tpots)*1000, 2) if tpots else None,
        "tpot_approx_p99_ms": round(sorted(tpots)[int(len(tpots)*0.99)]*1000, 2) if tpots else None,
    }
    print(f"\n  Results:")
    for k,v in m.items(): print(f"    {k}: {v}")
    outfile = f"/workspace/EPLB/OEPLB/benchmarks/results/{label}.json"
    with open(outfile, "w") as f: json.dump(m, f, indent=2)
    print(f"  → {outfile}")

if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
