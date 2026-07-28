#!/usr/bin/env python3
"""Orchestrate the full Prover-V1 sync_window x input_length grid rerun.

For each config: launch 235B server (auto mode) -> wait health -> run
run_grid_bench.py -> save persistent server log -> SIGTERM -> next config.

Resumable: skips configs whose result json + DONE marker already exist in
this run's result dir (controlled by --resume). Default is to redo everything
fresh per user's request ("全部重测").
"""
import argparse, json, os, subprocess, sys, time, signal

WORKLOAD_DIR = "/workspace/EPLB/OEPLB/benchmarks/workload_grid"
RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results"
LOG_DIR = "/workspace/EPLB/OEPLB/benchmarks/logs"
MODEL = "/data/models/Qwen3-235B-A22B-FP8"
PORT = 30000
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"

LENGTHS = [256, 512, 1024, 2048, 4096]
WINDOWS = [16, 32, 64, 128]
REPS = [1, 2]

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


def build_configs():
    configs = []
    for L in LENGTHS:
        for r in REPS:
            configs.append({"label": f"gridL{L}_bl_r{r}", "length": L, "window": None, "rep": r})
    for L in LENGTHS:
        for W in WINDOWS:
            for r in REPS:
                configs.append({"label": f"gridL{L}_sw{W}_r{r}", "length": L, "window": W, "rep": r})
    return configs


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
        print("  [warn] server did not exit within 60s of SIGTERM, sending again")
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            pass
    log_f.close()
    time.sleep(5)


def run_one(cfg):
    label = cfg["label"]
    dataset = f"{WORKLOAD_DIR}/tok{cfg['length']}_out1.jsonl"
    result_path = f"{RESULT_DIR}/{label}.json"

    args = list(BASE_ARGS)
    if cfg["window"] is not None:
        args += OEPLB_ARGS_TMPL(cfg["window"])

    env = os.environ.copy()
    env.update(ENV_EXTRA)
    env["LD_LIBRARY_PATH"] = env["NVSHMEM_HOME"] + "/lib:" + env.get("LD_LIBRARY_PATH", "")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{label}.log"
    log_f = open(log_path, "w")

    print(f"\n=== [{time.strftime('%H:%M:%S')}] Starting {label} (dataset={dataset}) ===", flush=True)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    ok, err = wait_health(proc)
    if not ok:
        print(f"  [FAIL] {label}: {err}")
        shutdown_server(proc, log_f)
        return False

    print(f"  server healthy, running benchmark...")
    bench = subprocess.run(
        ["python3", "run_grid_bench.py", label, dataset, "1024"],
        cwd="/workspace/EPLB/OEPLB/scripts",
        capture_output=True, text=True, timeout=1200,
    )
    print(bench.stdout[-2000:])
    if bench.returncode != 0:
        print(f"  [FAIL] bench script error: {bench.stderr[-2000:]}")

    shutdown_server(proc, log_f)

    ok_result = os.path.exists(result_path)
    print(f"=== [{time.strftime('%H:%M:%S')}] Finished {label}, result_saved={ok_result} ===", flush=True)
    return ok_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                     help="Skip configs whose result json already exists in this run")
    ap.add_argument("--only", default=None, help="Comma-separated label substrings to filter (for testing)")
    args = ap.parse_args()

    configs = build_configs()
    if args.only:
        filters = args.only.split(",")
        configs = [c for c in configs if any(f in c["label"] for f in filters)]

    print(f"Total configs to run: {len(configs)}")
    n_ok, n_fail = 0, 0
    for i, cfg in enumerate(configs):
        if args.resume and os.path.exists(f"{RESULT_DIR}/{cfg['label']}.json"):
            print(f"[{i+1}/{len(configs)}] SKIP {cfg['label']} (already done)")
            n_ok += 1
            continue
        print(f"\n[{i+1}/{len(configs)}] {cfg['label']}")
        try:
            success = run_one(cfg)
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            success = False
        if success:
            n_ok += 1
        else:
            n_fail += 1
        print(f"Progress: {n_ok} ok, {n_fail} fail, {len(configs) - i - 1} remaining")

    print(f"\nALL DONE: {n_ok} ok, {n_fail} fail out of {len(configs)}")


if __name__ == "__main__":
    main()
