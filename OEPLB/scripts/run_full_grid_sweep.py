#!/usr/bin/env python3
"""Full input×output×window grid sweep.
3 input lengths × 4 output lengths × 5 windows + 12 baselines = 72 runs.
conc=16, auto mode, each run: start server -> health -> bench -> save -> shutdown.
"""
import os, subprocess, sys, time, signal, json

GRID_DIR = "/workspace/EPLB/OEPLB/benchmarks/full_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/full_grid"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/full_grid"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
CONC = 16

LENGTHS = [256, 1024, 4096]
OUTPUTS = [1, 64, 256, 1024]
WINDOWS = [8, 16, 32, 64, 128]

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
    configs = []
    for L in LENGTHS:
        for O in OUTPUTS:
            configs.append({"label": f"fg_L{L}_O{O}_bl", "length": L, "output": O, "window": None})
            for W in WINDOWS:
                configs.append({"label": f"fg_L{L}_O{O}_sw{W}", "length": L, "output": O, "window": W})
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


def shutdown(proc, log_f):
    try: proc.send_signal(signal.SIGTERM)
    except: pass
    try: proc.wait(timeout=60)
    except:
        try: proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
        except: pass
    log_f.close()
    time.sleep(8)


def run_one(cfg):
    label = cfg["label"]
    dataset = f"{GRID_DIR}/tok{cfg['length']}_out{cfg['output']}.jsonl"
    result_path = f"{RESULT_DIR}/{label}.json"

    if os.path.exists(result_path):
        tps = json.load(open(result_path)).get("tps", "?")
        print(f"  SKIP (already done, tps={tps})", flush=True)
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

    print(f"  server healthy, benchmarking (conc={CONC})...", flush=True)
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"full_grid/{label}", dataset, str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts", capture_output=True, text=True, timeout=3600)
    shutdown(proc, log_f)

    if os.path.exists(result_path):
        tps = json.load(open(result_path))["tps"]
        print(f"  DONE: tps={tps}", flush=True)
        return True
    else:
        print(f"  FAIL: no result file", flush=True)
        return False


def main():
    configs = build_configs()
    print(f"Total configs: {len(configs)}", flush=True)
    n_ok, n_fail = 0, 0
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['label']}", flush=True)
        try:
            success = run_one(cfg)
        except Exception as e:
            print(f"  EXCEPTION: {e}", flush=True)
            success = False
        if success: n_ok += 1
        else: n_fail += 1
        print(f"Progress: {n_ok} ok, {n_fail} fail, {len(configs)-i-1} remaining", flush=True)
    print(f"\nALL DONE: {n_ok} ok, {n_fail} fail out of {len(configs)}", flush=True)


if __name__ == "__main__":
    main()
