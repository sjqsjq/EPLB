#!/usr/bin/env python3
"""Dedicated profiling pass with a deterministic, config-independent forward-step
count. Root cause fixed (see investigation): SGLang's /start_profile takes
num_steps and auto-stops purely on scheduler.forward_ct (see
scheduler_profiler_mixin.py: profiler_target_forward_ct = forward_ct+num_steps,
checked every batch in _profile_batch_predicate) -- forward_ct increments once
per scheduler tick for BOTH prefill and decode (even idle DP ranks tick in
lockstep), so this auto-stop is exact and config-independent BY CONSTRUCTION.
The earlier script's variable 5-vs-6 step counts came from calling manual
/stop_profile after a client-timing-dependent loop (racing the auto-stop),
not from num_steps itself. Fix: never call stop_profile -- let num_steps do it.
Also: with_stack must be False. Its default (True) caused a 71s trace capture
to grow to 56GB/rank and hang the whole server (see incident this session).
"""
import requests, json, time, sys, os

MODEL = "/workspace/Qwen3-30B-A3B-FP8"
DATASET = "/workspace/EPLB/OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl"
NUM_STEPS = 600   # total forward_ct ticks (prefill+decode+idle), NOT prefill-window count
N_REQUESTS = 40   # sequential single-flight requests; ~33 fwd ticks/req => ~1300 ticks available, comfortably > NUM_STEPS

def main():
    label = sys.argv[1]
    trace_dir = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/trace_fixed_{label}"
    os.makedirs(trace_dir, exist_ok=True)

    with open(DATASET) as f:
        reqs = [json.loads(l) for l in f]

    print(f"[{label}] warmup (5 reqs)...")
    for req in reqs[:5]:
        requests.post("http://localhost:30000/v1/completions", json={
            "model": MODEL, "prompt": req["prompt"], "max_tokens": 32, "temperature": 0}, timeout=60)

    r = requests.post("http://localhost:30000/start_profile", json={
        "output_dir": trace_dir, "num_steps": NUM_STEPS, "activities": ["CPU", "GPU"],
        "with_stack": False, "record_shapes": False})
    print(f"[{label}] start_profile -> {r.status_code} {r.text[:150]}")
    t0 = time.time()

    sent = 0
    for req in reqs[5:5+N_REQUESTS]:
        requests.post("http://localhost:30000/v1/completions", json={
            "model": MODEL, "prompt": req["prompt"], "max_tokens": 32, "temperature": 0}, timeout=60)
        sent += 1
        # poll: has the server auto-exported the trace yet?
        n_files = len([f for f in os.listdir(trace_dir) if f.endswith('.gz')])
        if n_files >= 4:
            print(f"[{label}] auto-stopped after {sent} sequential reqs, {time.time()-t0:.1f}s")
            break
    else:
        print(f"[{label}] WARNING: sent all {N_REQUESTS} reqs, still not auto-stopped after {time.time()-t0:.1f}s -- waiting up to 60s more")
        for _ in range(60):
            time.sleep(1)
            n_files = len([f for f in os.listdir(trace_dir) if f.endswith('.gz')])
            if n_files >= 4:
                break

    n_files = len([f for f in os.listdir(trace_dir) if f.endswith('.gz')])
    print(f"[{label}] done: {n_files} trace files in {trace_dir}, elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
