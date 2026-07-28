#!/usr/bin/env python3
"""Re-test each (L,O) combo with:
1. Its oracle-best static window (from final sweep)
2. Adaptive with base=that oracle window
Compare to see if adaptive can match or beat the oracle.
"""
import os, subprocess, sys, time, signal, json

GRID_DIR = "/workspace/EPLB/OEPLB/benchmarks/final_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/adaptive_opt"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/adaptive_opt"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
CONC = 256

# Oracle best static window per (L,O) from final sweep
ORACLE = {
    (256,1): 8, (256,64): 16, (256,256): 32, (256,1024): 64,
    (512,1): 8, (512,64): 32, (512,256): 16, (512,1024): 64,
    (1024,1): 16, (1024,64): 8, (1024,256): 64, (1024,1024): 32,
    (2048,1): 16, (2048,64): 8, (2048,256): 16, (2048,1024): 32,
    (4096,1): 16, (4096,64): 8, (4096,256): 8, (4096,1024): 32,
}

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

def oeplb_args(sw, adaptive=False):
    args = [
        "--enable-pb-oeplb", "--pb-oeplb-threshold-ratio", "1.02",
        "--pb-oeplb-min-prefill-tokens", "256", "--pb-oeplb-sync-window", str(sw),
        "--pb-oeplb-cooldown-steps", "5", "--pb-oeplb-max-total-swap-layers", "94",
        "--pb-oeplb-max-swaps-per-layer", "64", "--pb-oeplb-min-swap-ops", "8",
    ]
    if adaptive:
        args += ["--pb-oeplb-adaptive-window", "--pb-oeplb-window-floor", str(sw)]
    return args

def gpu_mem_clear(threshold_mib=500, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            if vals and max(vals) < threshold_mib: return True
        except: pass
        time.sleep(2)
    return False

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
    except:
        try: proc.kill(); proc.wait(timeout=10)
        except: pass
    log_f.close()
    gpu_mem_clear()
    time.sleep(3)

def run_one(label, L, O, args_extra):
    dataset = f"{GRID_DIR}/L{L}_O{O}.jsonl"
    result_path = f"{RESULT_DIR}/{label}.json"
    if os.path.exists(result_path):
        r = json.load(open(result_path))
        print(f"  SKIP (exists, rps={r['ok']/r['total_time_s']:.1f})", flush=True)
        return True
    args = list(BASE_ARGS) + args_extra
    env = os.environ.copy()
    env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"] + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_f = open(f"{LOG_DIR}/{label}.log", "w")
    print(f"  starting server...", flush=True)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    if not wait_health(proc):
        print(f"  FAIL: not healthy", flush=True)
        shutdown(proc, log_f); return False
    print(f"  healthy, bench conc={CONC}...", flush=True)
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"adaptive_opt/{label}", dataset, str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts", capture_output=True, text=True, timeout=7200)
    shutdown(proc, log_f)
    if os.path.exists(result_path):
        r = json.load(open(result_path))
        print(f"  DONE: rps={r['ok']/r['total_time_s']:.1f}", flush=True)
        return True
    print(f"  FAIL: no result", flush=True)
    return False

configs = []
for (L,O), sw in sorted(ORACLE.items()):
    # baseline (reuse from final sweep if exists)
    configs.append({"label": f"ao_L{L}_O{O}_bl", "L": L, "O": O, "args": []})
    # best static window
    configs.append({"label": f"ao_L{L}_O{O}_sw{sw}", "L": L, "O": O, "args": oeplb_args(sw)})
    # adaptive with base=best static
    configs.append({"label": f"ao_L{L}_O{O}_adapt{sw}", "L": L, "O": O, "args": oeplb_args(sw, adaptive=True)})

print(f"=== Adaptive-Optimal sweep: {len(configs)} configs ===", flush=True)
n_ok, n_fail = 0, 0
for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] {cfg['label']}", flush=True)
    try:
        # Copy baseline result from final sweep if available
        if cfg['label'].endswith('_bl'):
            final_label = cfg['label'].replace('ao_', 'f_')
            final_path = f"/workspace/EPLB/OEPLB/benchmarks/results/final/{final_label}.json"
            result_path = f"{RESULT_DIR}/{cfg['label']}.json"
            if os.path.exists(final_path) and not os.path.exists(result_path):
                os.makedirs(RESULT_DIR, exist_ok=True)
                import shutil
                shutil.copy(final_path, result_path)
                r = json.load(open(result_path))
                print(f"  COPIED from final (rps={r['ok']/r['total_time_s']:.1f})", flush=True)
                n_ok += 1
                continue
        success = run_one(cfg['label'], cfg['L'], cfg['O'], cfg['args'])
    except Exception as e:
        print(f"  EXCEPTION: {e}", flush=True)
        success = False
    if success: n_ok += 1
    else: n_fail += 1
    print(f"Progress: {n_ok} ok, {n_fail} fail, {len(configs)-i-1} remaining", flush=True)

print(f"\n=== ALL DONE: {n_ok} ok, {n_fail} fail ===", flush=True)
