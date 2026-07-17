import time, requests, json, sys

url = "http://localhost:30000/v1/completions"
prompt = "Hello, tell me about"
max_tokens = 128

# Single request - measure TTFT and TPOT via streaming
stream_url = "http://localhost:30000/v1/completions"

# Non-streaming for total time
for trial in range(3):
    t0 = time.perf_counter()
    r = requests.post(url, json={
        "model": "/workspace/Qwen3-30B-A3B-FP8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }, timeout=60)
    t1 = time.perf_counter()
    d = r.json()
    out_tokens = d["usage"]["completion_tokens"]
    total_ms = (t1 - t0) * 1000
    tpot = total_ms / out_tokens if out_tokens > 0 else 0
    # Approximate TTFT: first token time ≈ total_time - (out_tokens - 1) * tpot
    # But for non-streaming we can't measure TTFT directly
    print(f"Trial {trial+1}: {out_tokens} tokens, total={total_ms:.1f}ms, tpot_approx={tpot:.2f}ms/tok")

# Streaming for TTFT
print("\n--- Streaming (TTFT) ---")
for trial in range(3):
    t0 = time.perf_counter()
    first_token_t = None
    token_count = 0
    r = requests.post(url, json={
        "model": "/workspace/Qwen3-30B-A3B-FP8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }, stream=True, timeout=60)
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: ") and line != "data: [DONE]":
                token_count += 1
                if first_token_t is None:
                    first_token_t = time.perf_counter()
    t1 = time.perf_counter()
    ttft = (first_token_t - t0) * 1000 if first_token_t else 0
    total_ms = (t1 - t0) * 1000
    print(f"Trial {trial+1}: TTFT={ttft:.1f}ms, total={total_ms:.1f}ms, chunks={token_count}")

