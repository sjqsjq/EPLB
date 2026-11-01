#!/usr/bin/env python3
"""ADW v2.1 r3 -- overhead accounting decomposition (single H20 GPU).

Scope: the window-protocol timing from the 8xH20 nccl fault tests (--timing)
covers ONLY det clone + det SUM all_reduce + local step()/decision_hash +
confirm MIN/MAX all_reduce + cuda sync, per rank (ranks run in PARALLEL;
per-rank times must never be summed across the 8 cards). The test-harness
all_gather_object is evidence collection, NOT protocol, and is measured
nowhere.

This bench measures the two pieces the protocol timing omits:
  (A) record-path addition, per forward: the gated second scatter_add (F1
      detection accumulator) in record_next_layer -- once per MoE layer per
      forward (94 layers). Measured with the EXACT incremental expression
      (flat/mask already computed for the planner scatter_add are reused;
      the v2.1 addition is masked_fill + mask.long() + scatter_add only).
      Two calipers: GPU event time (per-call and batched K=50) and CPU
      enqueue wall time (comparable to the live _prof_record_ns self-
      measurement in formal-run server logs, which is a CPU-side caliper).
  (B) window-boundary signal bundle the timing stand-in omitted:
      logical_load, row_distribution, change_score, ref-EWMA update,
      z_from_signals (incl. .item() sync), on real [94,128] shapes, with a
      per-piece breakdown. Exposed once per W forwards (scheduler thread).
Cross-check: live v2 formal runs self-measured avg_record_us (ONE
scatter_add, CPU caliper): 6 starts, median-of-medians ~88.9us
(range of per-start medians 84.4-91.9).
"""
import argparse
import json
import os
import sys
import time

import torch

SRC = "/workspace/EPLB/OEPLB/src"
sys.path.insert(0, SRC)
import v2p1_signals as sig  # noqa: E402

L, P, EP = 94, 128, 8
TOKENS, TOPK = 8192, 8
N_IDX = TOKENS * TOPK


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _p90(xs):
    xs = sorted(xs)
    return xs[int(0.9 * (len(xs) - 1))]


def _stats(xs):
    return {"median_us": round(_median(xs), 3), "p90_us": round(_p90(xs), 3),
            "mean_us": round(sum(xs) / len(xs), 3), "iters": len(xs)}


def bench_events_percall(fn, dev, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    xs = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize(dev)
        xs.append(s.elapsed_time(e) * 1000.0)
    return _stats(xs)


def bench_events_batched(fn, dev, warmup=10, rounds=20, K=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    xs = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(K):
            fn()
        e.record()
        torch.cuda.synchronize(dev)
        xs.append(s.elapsed_time(e) * 1000.0 / K)
    return _stats(xs)


def bench_cpu(fn, dev, warmup=50, iters=200):
    """CPU-side wall time of the call incl. enqueue (no per-iter sync) --
    same caliper class as the live _prof_record_ns self-measurement."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        xs.append((time.perf_counter_ns() - t0) / 1000.0)
    torch.cuda.synchronize(dev)
    return _stats(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        "/workspace/EPLB/NEW_PAPER/experiments/"
        "h20_placement_directional_20261012/results/adw_v2p1/r3",
        "overhead_decomposition_gpu.json"))
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.init()
    out = {"device": torch.cuda.get_device_name(0),
           "shapes": {"L": L, "P": P, "EP": EP, "tokens_per_forward": TOKENS,
                      "topk": TOPK, "flat_index_per_layer": N_IDX}}

    # ---- (A) record-path addition: EXACT incremental expression -----------
    det_layer = torch.zeros(P, dtype=torch.int64, device=dev)
    load_layer = torch.zeros(P, dtype=torch.int64, device=dev)
    flat = torch.randint(0, P, (N_IDX,), dtype=torch.int64, device=dev)
    mask = flat != -1           # computed ONCE for the planner scatter_add
    idx_prepped = flat.masked_fill(~mask, 0).long()

    def extra_v21():            # v2.1 addition per record call
        det_layer.scatter_add_(dim=0,
                               index=flat.masked_fill(~mask, 0).long(),
                               src=mask.long())

    def planner_v2():           # v2 baseline cost (context only; already
        load_layer.scatter_add_(dim=0, index=idx_prepped, src=mask.long())
                                # present in the frozen formal campaign)

    out["A_record_addition_per_layer"] = {
        "gpu_event_percall": bench_events_percall(extra_v21, dev),
        "gpu_event_batched_K50": bench_events_batched(extra_v21, dev),
        "cpu_enqueue": bench_cpu(extra_v21, dev),
    }
    out["A_context_v2_planner_scatter_per_layer"] = {
        "gpu_event_percall": bench_events_percall(planner_v2, dev),
        "cpu_enqueue": bench_cpu(planner_v2, dev),
    }
    a_gpu = out["A_record_addition_per_layer"]["gpu_event_batched_K50"]["median_us"]
    a_cpu = out["A_record_addition_per_layer"]["cpu_enqueue"]["median_us"]
    out["A_per_forward_94_layers"] = {
        "gpu_time_us": round(a_gpu * L, 1),
        "cpu_enqueue_us": round(a_cpu * L, 1),
        "note": ("GPU time = added kernel time on the compute stream "
                 "(serialized inside the forward). CPU enqueue = scheduler-"
                 "side call time, overlaps with GPU execution; the live "
                 "v2 _prof_record self-measurement is this caliper class."),
    }
    out["A_live_crosscheck"] = {
        "source": "formal phaseB V2 server.log [PB-OEPLB-PROF] avg_record_us",
        "starts": 6, "per_start_medians_us": [91.95, 84.65, 88.94, 84.38,
                                               91.84, 86.15],
        "median_of_medians_us": 88.94,
        "caliper": "CPU wall around ONE planner scatter_add call (v2 frozen)",
    }

    # ---- (B) window-boundary signal bundle + breakdown --------------------
    det = torch.randint(1, 1000, (L, P), dtype=torch.int64, device=dev)
    p2l = torch.stack([torch.randperm(P) for _ in range(L)]).to(dev)
    ref = torch.rand(L, P, device=dev)
    n_fresh = N_IDX * L * 8

    logical = sig.logical_load(det, p2l)
    p_dist, valid = sig.row_distribution(logical)

    def f_logical():
        sig.logical_load(det, p2l)

    def f_rowdist():
        sig.row_distribution(logical)

    def f_change():
        sig.change_score(p_dist, ref, valid)

    def f_z():
        sig.z_from_signals(0.01, n_fresh, L, 2806.9)

    def f_ewma():
        ref.mul_(0.5).add_(p_dist, alpha=0.5)

    def f_bundle():
        lg = sig.logical_load(det, p2l)
        pp, vv = sig.row_distribution(lg)
        s, _nv = sig.change_score(pp, ref, vv)
        sig.z_from_signals(s, n_fresh, L, 2806.9)
        ref.mul_(0.5).add_(pp, alpha=0.5)

    out["B_signals_per_window"] = {
        "bundle": bench_events_percall(f_bundle, dev),
        "breakdown": {"logical_load": bench_events_percall(f_logical, dev),
                      "row_distribution": bench_events_percall(f_rowdist, dev),
                      "change_score": bench_events_percall(f_change, dev),
                      "z_from_signals": bench_events_percall(f_z, dev),
                      "ref_ewma": bench_events_percall(f_ewma, dev)},
        "note": ("headroom + clone + det.sum already inside the protocol "
                 "timing; excluded here to avoid double counting. Exposed "
                 "once per W forwards on the scheduler thread (blocking)."),
    }
    b = out["B_signals_per_window"]["bundle"]["median_us"]

    # ---- assembled decomposition -------------------------------------------
    proto = {"r2_run_median_us": 569.5, "r3_rerun_median_us": 411.3,
             "note": ("window protocol (3 collectives + step/hash + sync), "
                      "per rank, 8xH20 nccl fault-test --timing; two "
                      "independent runs reported, none cherry-picked")}
    out["C_protocol_timing_from_fault_tests"] = proto
    rows = []
    for W in (8, 16, 32, 64):
        rows.append({
            "W": W,
            "record_path_per_forward_gpu_us": round(a_gpu * L, 1),
            "window_protocol_per_forward_r2_us": round(proto["r2_run_median_us"] / W, 2),
            "window_protocol_per_forward_r3_us": round(proto["r3_rerun_median_us"] / W, 2),
            "signals_per_forward_us": round(b / W, 2),
            "total_added_per_forward_r2_us": round(
                a_gpu * L + proto["r2_run_median_us"] / W + b / W, 1),
            "total_added_per_forward_r3_us": round(
                a_gpu * L + proto["r3_rerun_median_us"] / W + b / W, 1),
        })
    out["D_amortization_per_forward"] = rows
    out["E_relative_context"] = {
        "live_forward_wall_ms_approx": 350,
        "derivation": ("W0 formal start: ~1044 forwards over ~365s serving "
                       "wall -> ~350ms/forward all-in"),
        "record_path_share_percent_approx": round(100 * a_gpu * L / 1000 / 350, 2),
        "caveat": ("order-of-magnitude only; ~350ms includes inter-request "
                   "gaps and decode; GPU-added kernel time is what "
                   "serializes into the forward"),
    }
    out["F_critical_path_exposure"] = {
        "record_path": ("per forward, inside the forward hook; added kernels "
                        "on the compute stream -> exposed EVERY forward"),
        "window_protocol": ("once per W forwards, scheduler thread, blocking "
                            "collectives -> exposed at window boundaries"),
        "harness_all_gather_object": ("fault-test evidence collection ONLY; "
                                      "not protocol; in no timing anywhere"),
        "rank_summing": ("ranks run the protocol in PARALLEL; per-rank "
                         "medians must NOT be summed across the 8 cards"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:3000])
    print("BENCH_WRITTEN", args.out)


if __name__ == "__main__":
    main()
