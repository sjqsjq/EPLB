#!/usr/bin/env python3
"""Manual verification run for L=256: user suspects high variance / mismatch
vs their own earlier results. Re-run with concurrency=64 (not 1024), and for
each window alternate baseline/OEPLB twice (BL,OE,BL,OE) to cancel out any
time-drift confound instead of blocking all baselines then all OEPLB runs.
"""
import os, subprocess, sys, time, signal

DATASET = "/workspace/EPLB/OEPLB/benchmarks/workload_grid/tok256_out1.jsonl"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results/verify_L256"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs/verify_L256"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"
CONC = 64
WINDOWS = [16, 32, 64, 128]

ENV_EXTRA = {
    "NVSHMEM_HOME": "/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem",
    "NVSHMEM_REMOTE_TRANSPORT": "none",
    "NVSHMEM_IB_ENABLE_IBGDA": "0",
    "NVSHMEM_HCA_LIST": "",
    "NVSHMEM_BOOTSTRAP": "UID",
    "NVSHMEM_DISABLE_P2P": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

BASE_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL,
    "--tp", "8", "--dp", "8", "--ep-size", "8", "--enable-dp-attention",
    "--moe-a2a-backend", "deepep", "--deepep-mode", "auto",
    "--moe-runner-backend", "deep_gemm",
    "--quantization", "fp8", "--mem-fraction-static", "0.8",
    "--cuda-graph-max-bs", "128",
    "--port", str(PORT), "--host", "0.0.0.0", "--trust-remote-code",
    "--disable-radix-cache",
]

OEPLB_ARGS_TMPL = lambda sw: [
    "--enable-pb-oeplb",
    "--pb-oeplb-threshold-ratio", "1.02",
    "--pb-oeplb-min-prefill-tokens", "256",
    "--pb-oeplb-sync-window", str(sw),
    "--pb-oeplb-cooldown-steps", "5",
    "--pb-oeplb-max-total-swap-layers", "94",
    "--pb-oeplb-max-swaps-per-layer", "64",
]


def build_sequence():
    """For each window: BL_a, OE_a, BL_b, OE_b (interleaved, not blocked)."""
    seq = []
    for W in WINDOWS:
        seq.append({"label": f"verifyL256_bl_forSW{W}_a", "window": None, "sw_group": W})
        seq.append({"label": f"verifyL256_sw{W}_a", "window": W, "sw_group": W})
        seq.append({"label": f"verifyL256_bl_forSW{W}_b", "window": None, "sw_group": W})
        seq.append({"label": f"verifyL256_sw{W}_b", "window": W, "sw_group": W})
    return seq


def wait_health(proc, timeout=900):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, "server process died before becoming healthy"
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
                if r.status == 200:
                    return True, None
        except Exception:
            pass
        time.sleep(5)
    return False, "health check timeout"


def shutdown_server(proc, log_f):
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            pass
    log_f.close()
    time.sleep(5)


def run_one(cfg):
    label = cfg["label"]
    result_path = f"{RESULT_DIR}/{label}.json"

    args = list(BASE_ARGS)
    if cfg["window"] is not None:
        args += OEPLB_ARGS_TMPL(cfg["window"])

    env = os.environ.copy()
    env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"] + "/lib:" + env.get("LD_LIBRARY_PATH", "")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{label}.log"
    log_f = open(log_path, "w")

    print(f"\n=== [{time.strftime('%H:%M:%S')}] Starting {label} ===", flush=True)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    ok, err = wait_health(proc)
    if not ok:
        print(f"  [FAIL] {label}: {err}")
        shutdown_server(proc, log_f)
        return False

    print(f"  server healthy, running benchmark (conc={CONC})...")
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", f"verify_L256/{label}", DATASET, str(CONC)],
        cwd="/workspace/EPLB/OEPLB/scripts",
        capture_output=True, text=True, timeout=1200,
    )
    print(bench.stdout[-1500:])
    if bench.returncode != 0:
        print(f"  [FAIL] bench script error: {bench.stderr[-1500:]}")

    shutdown_server(proc, log_f)

    ok_result = os.path.exists(result_path)
    print(f"=== [{time.strftime('%H:%M:%S')}] Finished {label}, result_saved={ok_result} ===", flush=True)
    return ok_result


def main():
    seq = build_sequence()
    print(f"Total runs: {len(seq)} (conc={CONC}, dataset={DATASET})")
    n_ok, n_fail = 0, 0
    for i, cfg in enumerate(seq):
        print(f"\n[{i+1}/{len(seq)}] {cfg['label']}")
        try:
            success = run_one(cfg)
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            success = False
        if success:
            n_ok += 1
        else:
            n_fail += 1
        print(f"Progress: {n_ok} ok, {n_fail} fail, {len(seq) - i - 1} remaining")
    print(f"\nALL DONE: {n_ok} ok, {n_fail} fail out of {len(seq)}")


if __name__ == "__main__":
    main()
