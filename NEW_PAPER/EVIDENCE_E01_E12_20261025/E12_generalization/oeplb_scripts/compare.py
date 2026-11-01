#!/usr/bin/env python3
"""Compare benchmark results with anomaly detection."""
import json, sys, os

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def load(name):
    path = f"/workspace/EPLB/OEPLB/benchmarks/results/{name}.json"
    with open(path) as f:
        return json.load(f)

def d(base, val):
    return (val - base) / base * 100 if base != 0 else 0

def fmt_delta(val, good_if_negative=True):
    s = f"{val:+.1f}%"
    if good_if_negative:
        return f"{GREEN}{s}{RESET}" if val < 0 else (f"{RED}{s}{RESET}" if val > 5 else s)
    else:
        return f"{GREEN}{s}{RESET}" if val > 0 else (f"{RED}{s}{RESET}" if val < -5 else s)

labels = sys.argv[1:] if len(sys.argv) > 1 else ["T1_baseline", "T2_oeplb_sparse", "T3_oeplb_always", "T4_eplb"]

data = {}
for label in labels:
    try:
        data[label] = load(label)
    except FileNotFoundError:
        print(f"WARNING: {label}.json not found, skipping")

if not data:
    print("No data found"); sys.exit(1)

bl_key = labels[0]
bl = data[bl_key]

print("=" * 110)
print(f"  Comparison: {len(data)} configs | {bl['num_requests']} frozen requests | conc={bl['concurrency']} | ignore_eos=True")
print("=" * 110)

# Header
header = f"{'Metric':<20}"
for label in labels:
    if label in data:
        short = label.replace("_", " ")
        header += f"{short:>18}"
        if label != bl_key:
            header += f"{'Δ':>8}"
print(header)
print("-" * 110)

# Rows
metrics = [
    ("requests_ok", "ok", False),
    ("total_time_s", "total_time_s", True),
    ("throughput tok/s", "tps", False),
    ("avg_output_tok", "avg_output_tokens", False),
    ("TTFT mean ms", "ttft_mean_ms", True),
    ("TTFT P50 ms", "ttft_p50_ms", True),
    ("TTFT P99 ms", "ttft_p99_ms", True),
    ("TPOT mean ms", "tpot_approx_mean_ms", True),
    ("TPOT P50 ms", "tpot_approx_p50_ms", True),
    ("TPOT P99 ms", "tpot_approx_p99_ms", True),
    ("E2E mean ms", "lat_mean_ms", True),
    ("E2E P99 ms", "lat_p99_ms", True),
]

for name, key, good_if_neg in metrics:
    row = f"{name:<20}"
    for label in labels:
        if label not in data: continue
        val = data[label].get(key, "N/A")
        if val is None: val = "N/A"
        row += f"{val:>18}"
        if label != bl_key and isinstance(val, (int, float)) and isinstance(bl.get(key), (int, float)):
            delta = d(bl[key], val)
            row += f"  {fmt_delta(delta, good_if_neg)}"
    print(row)

# Anomaly detection
print("\n" + "=" * 110)
print("  Anomaly Detection")
print("=" * 110)
expected = bl.get("expected_output_tokens", 1024)
for label in labels:
    if label not in data: continue
    avg = data[label].get("avg_output_tokens", 0)
    if abs(avg - expected) / expected > 0.05:
        print(f"  {RED}WARNING{RESET}: {label} avg_output_tokens={avg} (expected ~{expected}, deviation >{5}%)")
    elif abs(avg - expected) / expected > 0.02:
        print(f"  NOTE: {label} avg_output_tokens={avg} (expected ~{expected}, deviation >{2}%)")
    else:
        print(f"  OK: {label} avg_output_tokens={avg} ≈ {expected}")

    errs = data[label].get("errors", 0)
    if errs > 0:
        print(f"  {RED}WARNING{RESET}: {label} has {errs} errors!")

