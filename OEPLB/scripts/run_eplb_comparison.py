#!/usr/bin/env python3
"""OEPLB vs SGLang EPLB fair comparison.
Both use deepep-mode=normal. EPLB uses 8 redundant experts.
Test: L=256 O=1, L=256 O=64, L=1024 O=1, L=1024 O=64.
"""
import os, subprocess, time, signal, json

GRID_DIR = "/workspace/EPLB/OEPLB/benchmarks/final_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/eplb_comparison"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/eplb_comparison"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000; CONC = 256

ENV_EXTRA = {
    "NVSHMEM_HOME": "/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem",
    "NVSHMEM_REMOTE_TRANSPORT": "none", "NVSHMEM_IB_ENABLE_IBGDA": "0",
    "NVSHMEM_HCA_LIST": "", "NVSHMEM_BOOTSTRAP": "UID", "NVSHMEM_DISABLE_P2P": "0",
    "NCCL_IB_DISABLE": "1", "NCCL_P2P_LEVEL": "NVL",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}

# All use deepep-mode=normal for fairness (EPLB doesn't support auto)
BASE = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "8", "--dp", "8", "--ep-size", "8",
    "--enable-dp-attention", "--moe-a2a-backend", "deepep", "--deepep-mode", "normal",
    "--moe-runner-backend", "deep_gemm", "--quantization", "fp8",
    "--mem-fraction-static", "0.8", "--cuda-graph-max-bs", "128",
    "--port", str(PORT), "--host", "0.0.0.0", "--trust-remote-code",
    "--disable-radix-cache", "--watchdog-timeout", "600",
]

# EPLB config: 8 redundant experts, iter=64
EPLB_ARGS = [
    "--ep-num-redundant-experts", "8",
    "--ep-dispatch-algorithm", "dynamic",
    "--enable-eplb", "--eplb-algorithm", "auto",
    "--eplb-rebalance-num-iterations", "64",
    "--expert-distribution-recorder-mode", "stat",
]

# OEPLB config: no redundant experts, adaptive window
OEPLB_ARGS = [
    "--enable-pb-oeplb",
    "--pb-oeplb-threshold-ratio", "1.02",
    "--pb-oeplb-min-prefill-tokens", "256",
    "--pb-oeplb-sync-window", "8",
    "--pb-oeplb-cooldown-steps", "5",
    "--pb-oeplb-max-total-swap-layers", "94",
    "--pb-oeplb-max-swaps-per-layer", "64",
    "--pb-oeplb-min-swap-ops", "8",
    "--pb-oeplb-adaptive-window",
    "--pb-oeplb-window-floor", "8",
]

CASES = [
    ("L256_O1", f"{GRID_DIR}/L256_O1.jsonl"),
    ("L256_O64", f"{GRID_DIR}/L256_O64.jsonl"),
    ("L1024_O1", f"{GRID_DIR}/L1024_O1.jsonl"),
    ("L1024_O64", f"{GRID_DIR}/L1024_O64.jsonl"),
]

def gpu_clear():
    for _ in range(30):
        try:
            out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                capture_output=True,text=True,timeout=10).stdout
            if max(int(x.strip()) for x in out.strip().splitlines()) < 500: return
        except: pass
        time.sleep(2)

def wait_health(proc):
    import urllib.request
    for _ in range(180):
        if proc.poll() is not None: return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200: return True
        except: pass
        time.sleep(5)
    return False

def run_one(label, dataset, extra_args):
    rp = f"{RESULT_DIR}/{label}.json"
    if os.path.exists(rp):
        d = json.load(open(rp))
        if d.get('errors',0) <= d['ok']*0.1:
            print(f"  SKIP (exists, rps={d['ok']/d['total_time_s']:.1f})", flush=True)
            return True
        else:
            os.remove(rp)  # bad result, redo
    args = BASE + extra_args
    env = dict(os.environ); env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"]+"/lib:"+env.get("LD_LIBRARY_PATH","")
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(RESULT_DIR, exist_ok=True)
    lf = open(f"{LOG_DIR}/{label}.log", "w")
    print(f"  starting server...", flush=True)
    proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env)
    if not wait_health(proc):
        print(f"  FAIL health", flush=True); proc.kill(); lf.close(); gpu_clear(); return False
    print(f"  healthy, bench conc={CONC}...", flush=True)
    subprocess.run(["python3","run_grid_bench.py",f"eplb_comparison/{label}",dataset,str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts",capture_output=True,text=True,timeout=3600)
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(60)
    except: proc.kill()
    lf.close(); gpu_clear(); time.sleep(3)
    if os.path.exists(rp):
        d = json.load(open(rp))
        print(f"  DONE rps={d['ok']/d['total_time_s']:.1f}", flush=True)
        return True
    print(f"  FAIL", flush=True); return False

# Build config list: for each case, run baseline → EPLB → OEPLB
configs = []
for case_name, dataset in CASES:
    configs.append((f"cmp_{case_name}_bl", dataset, []))
    configs.append((f"cmp_{case_name}_eplb", dataset, EPLB_ARGS))
    configs.append((f"cmp_{case_name}_oeplb", dataset, OEPLB_ARGS))

print(f"=== EPLB vs OEPLB Comparison: {len(configs)} runs ===", flush=True)
print(f"Mode: deepep-mode=normal for all, EPLB has 8 redundant experts", flush=True)
n_ok, n_fail = 0, 0
for i, (label, dataset, extra) in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] {label}", flush=True)
    try:
        if run_one(label, dataset, extra): n_ok += 1
        else: n_fail += 1
    except Exception as e:
        print(f"  EXCEPTION: {e}", flush=True); n_fail += 1
    print(f"Progress: {n_ok} ok, {n_fail} fail, {len(configs)-i-1} remaining", flush=True)

# Print results
print(f"\n\n{'='*60}", flush=True)
print("RESULTS: OEPLB vs EPLB (both deepep-mode=normal)", flush=True)
print(f"{'='*60}", flush=True)
print(f"| Case | BL(rps) | EPLB(rps) | EPLB% | OEPLB(rps) | OEPLB% | Winner |", flush=True)
print(f"|------|---------|-----------|-------|------------|--------|--------|", flush=True)
for case_name, _ in CASES:
    bl_f = f"{RESULT_DIR}/cmp_{case_name}_bl.json"
    ep_f = f"{RESULT_DIR}/cmp_{case_name}_eplb.json"
    oe_f = f"{RESULT_DIR}/cmp_{case_name}_oeplb.json"
    bl = json.load(open(bl_f))['ok']/json.load(open(bl_f))['total_time_s'] if os.path.exists(bl_f) else 0
    ep = json.load(open(ep_f))['ok']/json.load(open(ep_f))['total_time_s'] if os.path.exists(ep_f) else 0
    oe = json.load(open(oe_f))['ok']/json.load(open(oe_f))['total_time_s'] if os.path.exists(oe_f) else 0
    ep_p = f"{(ep-bl)/bl*100:+.1f}%" if bl else "-"
    oe_p = f"{(oe-bl)/bl*100:+.1f}%" if bl else "-"
    w = "OEPLB" if oe>ep else "EPLB"
    print(f"| {case_name} | {bl:.1f} | {ep:.1f} | {ep_p} | {oe:.1f} | {oe_p} | {w} |", flush=True)

print(f"\nALL DONE: {n_ok} ok, {n_fail} fail", flush=True)
