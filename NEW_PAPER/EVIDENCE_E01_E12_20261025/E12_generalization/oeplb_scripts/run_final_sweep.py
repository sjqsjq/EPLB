#!/usr/bin/env python3
"""Final comprehensive sweep with real datasets.
5 input lengths × 4 output lengths × (1 baseline + 5 OEPLB windows + 1 adaptive) = 140 runs.
Windows: {8,16,32,64} (no 128 per user request) + adaptive(base=8).
All use min_swap_ops=8, conc=256 fixed.
"""
import os, subprocess, sys, time, signal, json

GRID_DIR = "/workspace/EPLB/OEPLB/benchmarks/final_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/final"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/final"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
CONC = 256

LENGTHS = [256, 512, 1024, 2048, 4096]
OUTPUTS = [1, 64, 256, 1024]
WINDOWS = [8, 16, 32, 64]  # no 128

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


def build_configs():
    configs = []
    for L in LENGTHS:
        for O in OUTPUTS:
            # baseline
            configs.append({"label": f"f_L{L}_O{O}_bl", "L": L, "O": O, "args_extra": []})
            if O >= 1024:
                # O=1024: only adaptive (static windows too slow, marginal benefit)
                configs.append({"label": f"f_L{L}_O{O}_adaptive8", "L": L, "O": O,
                                "args_extra": oeplb_args(8, adaptive=True)})
            else:
                # static windows
                for W in WINDOWS:
                    configs.append({"label": f"f_L{L}_O{O}_sw{W}", "L": L, "O": O,
                                    "args_extra": oeplb_args(W)})
                # adaptive (base=8)
                configs.append({"label": f"f_L{L}_O{O}_adaptive8", "L": L, "O": O,
                                "args_extra": oeplb_args(8, adaptive=True)})
    return configs


def gpu_mem_clear(threshold_mib=500, timeout=60):
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
        except:
            try: proc.kill(); proc.wait(timeout=10)
            except: pass
    log_f.close()
    gpu_mem_clear()
    time.sleep(3)


def run_one(cfg):
    label = cfg["label"]
    dataset = f"{GRID_DIR}/L{cfg['L']}_O{cfg['O']}.jsonl"
    result_path = f"{RESULT_DIR}/{label}.json"

    if os.path.exists(result_path):
        tps = json.load(open(result_path)).get("tps", "?")
        print(f"  SKIP (exists, tps={tps})", flush=True)
        return True

    if not os.path.exists(dataset):
        print(f"  SKIP (dataset not found: {dataset})", flush=True)
        return False

    args = list(BASE_ARGS) + cfg["args_extra"]
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

    print(f"  healthy, bench conc={CONC}...", flush=True)
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"final/{label}", dataset, str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts", capture_output=True, text=True, timeout=7200)
    shutdown(proc, log_f)

    if os.path.exists(result_path):
        res = json.load(open(result_path))
        print(f"  DONE: tps={res['tps']} ok={res['ok']} errors={res['errors']}", flush=True)
        return True
    else:
        print(f"  FAIL: no result", flush=True)
        return False


def main():
    configs = build_configs()
    print(f"=== Final Sweep: {len(configs)} configs, conc={CONC} ===", flush=True)
    n_ok, n_fail, n_skip = 0, 0, 0
    for i, cfg in enumerate(configs):
        tag = cfg['label'].split('_', 2)[2] if '_' in cfg['label'] else cfg['label']
        print(f"\n[{i+1}/{len(configs)}] {cfg['label']}", flush=True)
        try:
            result_path = f"{RESULT_DIR}/{cfg['label']}.json"
            if os.path.exists(result_path):
                tps = json.load(open(result_path)).get("tps", "?")
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
