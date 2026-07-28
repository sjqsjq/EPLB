#!/usr/bin/env python3
"""Comprehensive sync_window experiment: input_length × output_length × window.

Design decisions:
- 5 input lengths: {256, 512, 1024, 2048, 4096}
- 4 output lengths: {1, 64, 256, 1024}
- 5 windows: {8, 16, 32, 64, 128}
- Concurrency matched to input length to ensure GPU saturation without queuing:
    L=256:  conc=256 (short prefill, need many concurrent to saturate 8 DP ranks)
    L=512:  conc=128
    L=1024: conc=64
    L=2048: conc=32
    L=4096: conc=16
- Each (input,output) combo: 1 baseline + 5 OEPLB windows = 6 runs
- Total: 5 × 4 × 6 = 120 runs
- Resumable: skips configs whose result json already exists
"""
import os, subprocess, sys, time, signal, json

GRID_DIR = "/workspace/EPLB/OEPLB/benchmarks/comprehensive_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/comprehensive"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/comprehensive"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000

LENGTHS = [256, 512, 1024, 2048, 4096]
OUTPUTS = [1, 64, 256, 1024]
WINDOWS = [8, 16, 32, 64, 128]

# Concurrency matched to input length
CONC_MAP = {256: 256, 512: 128, 1024: 64, 2048: 32, 4096: 16}

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


def build_configs():
    """For each (L,O): baseline first, then 5 windows."""
    configs = []
    for L in LENGTHS:
        for O in OUTPUTS:
            conc = CONC_MAP[L]
            configs.append({"label": f"c_L{L}_O{O}_bl", "L": L, "O": O, "window": None, "conc": conc})
            for W in WINDOWS:
                configs.append({"label": f"c_L{L}_O{O}_sw{W}", "L": L, "O": O, "window": W, "conc": conc})
    return configs


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


def gpu_mem_clear(threshold_mib=500, timeout=60):
    """Poll nvidia-smi until all GPUs drop below threshold_mib used memory.
    A fixed sleep() after SIGTERM is not reliable -- a straggler process can
    hold GPU memory for a variable amount of time, contaminating the next
    run's measurement (observed 21-331MiB residuals in earlier runs)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            if vals and max(vals) < threshold_mib:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def shutdown(proc, log_f):
    try: proc.send_signal(signal.SIGTERM)
    except: pass
    try: proc.wait(timeout=60)
    except:
        try: proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
        except:
            try: proc.kill(); proc.wait(timeout=10)
            except: pass
    log_f.close()
    if not gpu_mem_clear():
        print("  [warn] GPU memory did not clear within timeout, proceeding anyway", flush=True)
    time.sleep(3)


def run_one(cfg):
    label = cfg["label"]
    dataset = f"{GRID_DIR}/L{cfg['L']}_O{cfg['O']}.jsonl"
    result_path = f"{RESULT_DIR}/{label}.json"
    conc = cfg["conc"]

    if os.path.exists(result_path):
        tps = json.load(open(result_path)).get("tps", "?")
        print(f"  SKIP (exists, tps={tps})", flush=True)
        return True

    args = list(BASE_ARGS)
    if cfg["window"] is not None:
        args += OEPLB_TMPL(cfg["window"])

    env = os.environ.copy()
    env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"] + "/lib:" + env.get("LD_LIBRARY_PATH", "")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_f = open(f"{LOG_DIR}/{label}.log", "w")

    print(f"  starting server...", flush=True)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    if not wait_health(proc):
        print(f"  FAIL: server didn't become healthy", flush=True)
        shutdown(proc, log_f); return False

    print(f"  healthy, bench conc={conc}...", flush=True)
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"comprehensive/{label}", dataset, str(conc)],
        cwd="/workspace/EPLB/OEPLB/scripts", capture_output=True, text=True, timeout=3600)
    shutdown(proc, log_f)

    if os.path.exists(result_path):
        res = json.load(open(result_path))
        print(f"  DONE: tps={res['tps']}", flush=True)
        return True
    else:
        print(f"  FAIL: no result", flush=True)
        return False


def main():
    configs = build_configs()
    print(f"=== Comprehensive Sweep: {len(configs)} configs ===", flush=True)
    print(f"Layout: {len(LENGTHS)} lengths × {len(OUTPUTS)} outputs × (1 BL + {len(WINDOWS)} windows) = {len(configs)}", flush=True)
    print(f"Concurrency map: {CONC_MAP}", flush=True)
    print(flush=True)

    n_ok, n_fail, n_skip = 0, 0, 0
    for i, cfg in enumerate(configs):
        w_str = f"sw={cfg['window']}" if cfg['window'] else "baseline"
        print(f"\n[{i+1}/{len(configs)}] L={cfg['L']} O={cfg['O']} {w_str} conc={cfg['conc']}", flush=True)
        try:
            if os.path.exists(f"{RESULT_DIR}/{cfg['label']}.json"):
                tps = json.load(open(f"{RESULT_DIR}/{cfg['label']}.json")).get("tps", "?")
                print(f"  SKIP (exists, tps={tps})", flush=True)
                n_skip += 1
                continue
            success = run_one(cfg)
        except Exception as e:
            print(f"  EXCEPTION: {e}", flush=True)
            success = False
        if success: n_ok += 1
        else: n_fail += 1
        remain = len(configs) - i - 1
        print(f"Progress: {n_ok} ok, {n_fail} fail, {n_skip} skip, {remain} remaining", flush=True)

    print(f"\n=== ALL DONE: {n_ok} ok, {n_fail} fail, {n_skip} skip out of {len(configs)} ===", flush=True)


if __name__ == "__main__":
    main()
