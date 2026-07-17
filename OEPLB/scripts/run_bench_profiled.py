#!/usr/bin/env python3
"""Combined benchmark + in-load profiler capture.
Fixes two issues found in the earlier methodology:
 1. The old profiling script sent 80 requests SEQUENTIALLY (conc=1), a totally
    different load regime than the conc=128 benchmark whose TPS numbers we
    report -> tiny, unrepresentative, noisy samples.
 2. Its manual /start_profile+/stop_profile pair let the sampled window length
    (and hence "step count") drift with request-completion timing (batching
    dynamics), producing 5 vs 6 "steps" between runs for no controlled reason.
This script starts the profiler with num_steps=N (SGLang's native forward_ct
based auto-stop, see scheduler_profiler_mixin.py: profiler_target_forward_ct
= forward_ct + num_steps) DURING the real conc=128 request stream, so every
run captures exactly N forward passes (prefill+decode both increment
forward_ct even though decode's per-layer python annotations are skipped
under CUDA-graph replay) under the same load regime as the reported TPS.
"""
import asyncio, aiohttp, json, time, sys, statistics, os, requests

API = "http://localhost:30000/v1/completions"
MODEL = "/workspace/Qwen3-30B-A3B-FP8"
FROZEN = "/workspace/EPLB/OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl"
CONC = 128
NUM_STEPS = 40
PROFILE_START_DELAY_S = 0.0  # let concurrency ramp to steady-state before starting

async def send(session, req):
    t0 = time.perf_counter()
    try:
        payload = {"model": MODEL, "prompt": req["prompt"], "max_tokens": req["max_tokens"],
                    "temperature": req["temperature"], "ignore_eos": req["ignore_eos"]}
        async with session.post(API, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
            res = await r.json()
            total = time.perf_counter() - t0
            usage = res.get("usage", {})
            return {"id": req["id"], "elapsed": total, "comp": usage.get("completion_tokens", 0),
                    "prompt_tok": usage.get("prompt_tokens", 0)}
    except Exception as ex:
        return {"id": req["id"], "elapsed": time.perf_counter()-t0, "error": str(ex)}

async def run(label, trace_dir):
    with open(FROZEN) as f:
        all_reqs = [json.loads(l) for l in f]
    N = len(all_reqs)
    print(f"\n{'='*70}\n  {label} | {N} frozen requests | conc={CONC} | profile num_steps={NUM_STEPS}\n{'='*70}")

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
        profiled = False

        while idx < N or active:
            while len(active) < CONC and idx < N:
                task = asyncio.create_task(send(session, all_reqs[idx]))
                active.add(task)
                idx += 1
            if not active: break
            if not profiled and (time.time() - t_start) >= PROFILE_START_DELAY_S:
                os.makedirs(trace_dir, exist_ok=True)
                r = requests.post("http://localhost:30000/start_profile", json={
                    "output_dir": trace_dir, "num_steps": NUM_STEPS, "activities": ["CPU", "GPU"],
                    "with_stack": False, "record_shapes": False})
                print(f"  [profile] start_profile -> {r.status_code} {r.text[:200]} at t={time.time()-t_start:.1f}s")
                profiled = True
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
    m = {
        "label": label, "num_requests": N, "concurrency": CONC,
        "ok": len(ok), "errors": len(errors),
        "total_time_s": round(wall, 2), "total_comp_tokens": comp,
        "tps": round(comp / wall, 1),
        "avg_output_tokens": round(statistics.mean(comp_list), 1) if comp_list else 0,
        "lat_mean_ms": round(statistics.mean(lats)*1000, 1),
        "lat_p50_ms": round(statistics.median(lats)*1000, 1),
        "lat_p99_ms": round(sorted(lats)[int(len(lats)*0.99)]*1000, 1),
        "profile_num_steps_requested": NUM_STEPS,
        "trace_dir": trace_dir,
    }
    print(f"\n  Results:")
    for k,v in m.items(): print(f"    {k}: {v}")
    outfile = f"/workspace/EPLB/OEPLB/benchmarks/results/{label}.json"
    with open(outfile, "w") as f: json.dump(m, f, indent=2)
    print(f"  -> {outfile}")

if __name__ == "__main__":
    label = sys.argv[1]
    trace_dir = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/trace_profiled_{label}"
    asyncio.run(run(label, trace_dir))
