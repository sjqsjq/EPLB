#!/usr/bin/env python3
"""PB-OEPLB Gain Predictor — estimate the throughput ceiling before running.

Usage:
    # Level 0: architecture only (no measurement needed)
    python3 predict_gain.py --model-config config.json --ep 8

    # Level 1: with recorded routing counts (one 6-min recording)
    python3 predict_gain.py --model-config config.json --ep 8 --counts counts.json

    # Level 2: with measured (beta, r_k) from a T(r) sweep
    python3 predict_gain.py --model-config config.json --ep 8 --counts counts.json \
                            --beta 0.285 --rk 1.099

Outputs a table showing:
    - r_before (dataset imbalance, from counts or estimated)
    - r_k (dead-zone knee, from power law or given)
    - beta (sensitivity slope, from FLOP fraction or given)
    - Delta_ceiling (throughput upper bound under perfect balancing)
    - Deployment recommendation (enable / skip / measure more)

The model:  Delta_ceiling = beta * max(0, r_before - r_k)
    - beta and r_k are configuration constants (model + GPU count)
    - r_before is a workload property (dataset + routing distribution)
"""
import argparse
import json
import math
import os
import sys


def compute_flop_fraction(config: dict, ep: int) -> dict:
    """Estimate routed-expert FLOP fraction from model config."""
    E = config.get("num_experts", config.get("num_local_experts", 0))
    K = config.get("num_experts_per_tok", config.get("top_k",
        config.get("num_selected_experts", 0)))
    hidden = config.get("hidden_size", 0)
    inter = config.get("intermediate_size", 0)
    moe_inter = config.get("moe_intermediate_size", inter)
    shared_inter = config.get("shared_expert_intermediate_size", 0)
    num_heads = config.get("num_attention_heads", 0)
    head_dim = config.get("head_dim", hidden // max(num_heads, 1))
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    # Per-token FLOPs (multiply-add = 2 ops per weight element)
    # Routed experts: K experts × (gate_up + down) = K × 2 × (hidden × moe_inter) × 2
    routed_flop = K * 2 * (2 * hidden * moe_inter) if moe_inter else 0
    # Shared expert (if any)
    shared_flop = 2 * (2 * hidden * shared_inter) if shared_inter else 0
    # Attention projections: QKV + output
    attn_flop = 2 * hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim + num_heads * head_dim)

    total = routed_flop + shared_flop + attn_flop
    frac = routed_flop / total if total > 0 else 0

    return {
        "num_experts": E, "top_k": K, "experts_per_gpu": E // max(ep, 1),
        "routed_flop": routed_flop, "shared_flop": shared_flop,
        "attn_flop": attn_flop, "total_flop": total,
        "routed_flop_fraction": frac,
    }


def compute_r_before(counts_path: str, ep: int) -> dict:
    """Compute per-layer average imbalance ratio from recorded routing counts."""
    d = json.load(open(counts_path))
    C = d["counts"]  # [num_layers][num_experts]
    L, E = d["num_layers"], d["num_experts"]
    per = E // ep

    # Identity placement: expert e goes to GPU e // per
    ratios = []
    for l in range(L):
        loads = [0.0] * ep
        for e in range(E):
            loads[e // per] += C[l][e]
        tot = sum(loads)
        if tot > 0:
            ratios.append(max(loads) / (tot / ep))
        else:
            ratios.append(1.0)
    r_avg = sum(ratios) / len(ratios)

    # LPT floor (optimal placement, no redundancy)
    lpt_ratios = []
    for l in range(L):
        c = C[l]
        load = [0.0] * ep
        cnt = [0] * ep
        for e in sorted(range(E), key=lambda e: -c[e]):
            g = min((i for i in range(ep) if cnt[i] < per), key=lambda i: load[i])
            load[g] += c[e]
            cnt[g] += 1
        tot = sum(load)
        lpt_ratios.append(max(load) / (tot / ep) if tot > 0 else 1.0)
    r_lpt = sum(lpt_ratios) / len(lpt_ratios)

    return {"r_before": r_avg, "r_lpt": r_lpt, "num_layers": L, "num_experts": E}


def predict_rk(ep: int) -> float:
    """EP power law for r_k, calibrated on 4 configurations (cross-model error 3.8%)."""
    return 1.0 + 0.00408 * ep ** 1.52


def predict(ep: int, r_before: float, r_lpt: float, beta: float, rk: float,
            flop_frac: float, experts_per_gpu: int) -> dict:
    """Core prediction."""
    x_eff = max(0.0, r_before - rk) / r_before if r_before > 1 else 0.0
    ceiling = beta * max(0.0, r_before - rk)
    # Placement vs routing ceiling gap
    placement_gap = beta * max(0.0, r_lpt - rk)

    # Recommendation
    if beta < 0:
        rec = "SKIP: f_sens likely negative (dispatch-dominated model), balancer will hurt"
    elif ceiling < 0.005:
        rec = "SKIP: ceiling < 0.5%, not worth the overhead"
    elif ceiling < 0.015:
        rec = "MARGINAL: ceiling 0.5-1.5%, might break even after swap overhead"
    elif r_lpt > rk:
        rec = f"ENABLE+REDUNDANCY: r_LPT={r_lpt:.4f} > r_k={rk:.4f}, placement alone leaves {100*placement_gap:.1f}% on the table; consider redundant experts"
    else:
        rec = f"ENABLE: ceiling {100*ceiling:.1f}%, placement is provably sufficient (r_LPT={r_lpt:.4f} <= r_k={rk:.4f})"

    return {
        "r_before": r_before,
        "r_lpt": r_lpt,
        "r_k": rk,
        "beta": beta,
        "x_eff": x_eff,
        "ceiling_pct": 100 * ceiling,
        "placement_gap_pct": 100 * placement_gap,
        "recommended_threshold": max(1.02, rk),
        "recommendation": rec,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True,
                        help="Path to model's config.json")
    parser.add_argument("--ep", type=int, required=True,
                        help="Expert parallelism degree (number of GPUs for EP)")
    parser.add_argument("--counts", default=None,
                        help="Path to recorded routing counts JSON (from dump_counts.py)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Measured beta (from T(r) sweep). If omitted, estimated from FLOP fraction / 1.6")
    parser.add_argument("--rk", type=float, default=None,
                        help="Measured r_k (from T(r) sweep). If omitted, uses EP power law")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable table")
    args = parser.parse_args()

    # Load model config
    cfg = json.load(open(args.model_config))
    # Some models nest config under "model" or similar
    if "model_type" not in cfg and "model" in cfg:
        cfg = cfg["model"]

    ep = args.ep
    arch = compute_flop_fraction(cfg, ep)

    # r_k
    rk = args.rk if args.rk else predict_rk(ep)
    rk_source = "measured" if args.rk else f"EP power law (error ~4%)"

    # beta
    if args.beta:
        beta = args.beta
        beta_source = "measured"
    else:
        beta = arch["routed_flop_fraction"] / 1.6
        beta_source = f"FLOP fraction ({arch['routed_flop_fraction']:.3f}) / 1.6 (uncertainty ±30%)"

    # r_before
    if args.counts:
        r_info = compute_r_before(args.counts, ep)
        r_before = r_info["r_before"]
        r_lpt = r_info["r_lpt"]
        r_source = "measured (offline from routing counts)"
    else:
        # Cannot compute without counts; give a range based on typical values
        r_before = None
        r_lpt = 1.0 + 0.01 * (64 / (arch["num_experts"] or 64))
        r_source = "NOT AVAILABLE — provide --counts for a real prediction"

    if r_before is None:
        print("="*70)
        print("WARNING: No routing counts provided. Cannot predict r_before.")
        print("Run the model with --expert-distribution-recorder-mode stat,")
        print("then use dump_counts.py to extract counts, and pass --counts.")
        print("="*70)
        print(f"\nModel: {arch['num_experts']} experts, top-{arch['top_k']}, "
              f"{arch['experts_per_gpu']} per GPU at EP={ep}")
        print(f"Routed-expert FLOP fraction: {arch['routed_flop_fraction']:.3f}")
        print(f"Estimated beta: {beta:.4f} ({beta_source})")
        print(f"Predicted r_k: {rk:.4f} ({rk_source})")
        print(f"Recommended threshold: {max(1.02, rk):.3f}")
        print(f"\nTo get the ceiling, record routing counts (6 min) and re-run with --counts.")
        return

    result = predict(ep, r_before, r_lpt, beta, rk, arch["routed_flop_fraction"],
                     arch["experts_per_gpu"])

    if args.json:
        out = {**result, "arch": arch, "rk_source": rk_source, "beta_source": beta_source,
               "r_source": r_source}
        print(json.dumps(out, indent=2))
        return

    print("="*70)
    print("  PB-OEPLB Gain Prediction")
    print("="*70)
    print(f"\n  Model:     {arch['num_experts']} experts, top-{arch['top_k']}, "
          f"{arch['experts_per_gpu']}/GPU at EP={ep}")
    print(f"  FLOP frac: {arch['routed_flop_fraction']:.3f}")
    print(f"\n  {'Parameter':<20s} {'Value':>10s}   Source")
    print(f"  {'-'*20} {'-'*10}   {'-'*40}")
    print(f"  {'r_before':<20s} {r_before:>10.4f}   {r_source}")
    print(f"  {'r_lpt (floor)':<20s} {r_lpt:>10.4f}   LPT packing of recorded counts")
    print(f"  {'r_k (knee)':<20s} {rk:>10.4f}   {rk_source}")
    print(f"  {'beta (slope)':<20s} {beta:>10.4f}   {beta_source}")
    print(f"\n  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Throughput ceiling':<30s} {result['ceiling_pct']:>9.2f}%")
    print(f"  {'x_eff (effective headroom)':<30s} {result['x_eff']:>10.4f}")
    if result['placement_gap_pct'] > 0.01:
        print(f"  {'Redundancy extra':<30s} {result['placement_gap_pct']:>9.2f}%")
    print(f"  {'Recommended threshold':<30s} {result['recommended_threshold']:>10.3f}")
    print(f"\n  >>> {result['recommendation']}")
    print()


if __name__ == "__main__":
    main()
