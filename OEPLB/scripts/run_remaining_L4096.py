#!/usr/bin/env python3
"""Run remaining 9 runs (8-16) of L=4096 manual verification, conc=16,
interleaved BL/OEPLB order, checking json result file for completion."""
import os, subprocess, sys, time, signal, json

DATASET = "/workspace/EPLB/OEPLB/benchmarks/workload_grid/tok4096_out1.jsonl"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/verify_L4096_manual"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/verify_L4096_manual"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
CONC = 16

ENV_EXTRA = {
    "NVSHMEM_HOME": "/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem",
    "NVSHMEM_REMOTE_TRANSPORT": "none", "NVSHMEM_IB_ENABLE_IBGDA": "0",
    "NVSHMEM_HCA_LIST": "", "NVSHMEM_BOOTSTRAP": "UID", "NVSHMEM_DISABLE_P2P": "0",
    "NCCL_IB_DISABLE": "1", "NCCL_P2P_LEVEL": "NVL",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}

BASE_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "8", "--dp", "8", "--ep-size", "8",
    "--enable-dp-attention", "--moe-a2a-backend", "deepep", "--deepep-mode", "auto",
    "--moe-runner-backend", "deep_gemm", "--quantization", "fp8",
    "--mem-fraction-static", "0.8", "--cuda-graph-max-bs", "128",
    "--port", str(PORT), "--host", "0.0.0.0", "--trust-remote-code", "--disable-radix-cache",
]

OEPLB_TMPL = lambda sw: [
    "--enable-pb-oeplb", "--pb-oeplb-threshold-ratio", "1.02",
    "--pb-oeplb-min-prefill-tokens", "256", "--pb-oeplb-sync-window", str(sw),
    "--pb-oeplb-cooldown-steps", "5", "--pb-oeplb-max-total-swap-layers", "94",
    "--pb-oeplb-max-swaps-per-layer", "64",
]

REMAINING = [
    {"label": "run08_sw32_b",      "window": 32},
    {"label": "run09_bl_sw64_a",   "window": None},
    {"label": "run10_sw64_a",      "window": 64},
    {"label": "run11_bl_sw64_b",   "window": None},
    {"label": "run12_sw64_b",      "window": 64},
    {"label": "run13_bl_sw128_a",  "window": None},
    {"label": "run14_sw128_a",     "window": 128},
    {"label": "run15_bl_sw128_b",  "window": None},
    {"label": "run16_sw128_b",     "window": 128},
]

def wait_health(proc, timeout=900):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None: return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200: return True
        except: pass
        time.sleep(5)
    return False

def shutdown(proc, log_f):
    try: proc.send_signal(signal.SIGTERM)
    except: pass
    try: proc.wait(timeout=60)
    except: pass
    log_f.close()
    time.sleep(8)

for i, cfg in enumerate(REMAINING):
    label = cfg["label"]
    result_path = f"{RESULT_DIR}/{label}.json"
    if os.path.exists(result_path):
        print(f"[{i+8}/16] SKIP {label} (already done)", flush=True)
        continue
    args = list(BASE_ARGS)
    if cfg["window"] is not None:
        args += OEPLB_TMPL(cfg["window"])
    env = os.environ.copy()
    env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"] + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    log_f = open(f"{LOG_DIR}/{label}.log", "w")
    print(f"\n[{i+8}/16] {label} — starting server...", flush=True)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    if not wait_health(proc):
        print(f"  FAIL: server didn't become healthy"); shutdown(proc, log_f); continue
    print(f"  server healthy, benchmarking (conc={CONC})...", flush=True)
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"verify_L4096_manual/{label}", DATASET, str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts", capture_output=True, text=True, timeout=1200)
    shutdown(proc, log_f)
    if os.path.exists(result_path):
        tps = json.load(open(result_path))["tps"]
        print(f"  DONE: tps={tps}", flush=True)
    else:
        print(f"  FAIL: no result file", flush=True)

print("\nALL REMAINING DONE", flush=True)
