#!/usr/bin/env python3
"""Cost-benefit comparison for the near-equilibrium scan study.

Hash-bound to configs/followup_v1/scan_cost_benefit_spec.json (frozen before any
comparison). Compares low-gain placement actions against actually measured
migration costs from the frozen stopping campaign:

  * per-prefix break-even horizons (batch estimator and sequential upper bound),
  * the marginal final greedy step versus one measured single-swap transaction,
  * horizon-indexed expected net benefit N*Q_cum - C_batch and optimal prefixes.

No new timing, no refit, no subset selection; signed values preserved; a CI
crossing zero is compatible-with-zero, never a finite horizon and never proof of
no effect. Migration costs are compared at the decision level only; nothing is
subtracted inside any measured execution timing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prediction_common import sha256_file, write_json_atomic  # noqa: E402

T_QUANTILE_95_DF3 = 3.1824463052837078
EXPERTS_PER_RANK = 16
DEFAULT_SPEC = "configs/followup_v1/scan_cost_benefit_spec.json"
OUTPUT_REL = "results/nearscan_v1/cost_benefit.json"


def fieller_break_even(c: float, se_c: float, q: float, se_q: float) -> dict[str, float] | None:
    """Ratio interval for N = C/Q from two independent means (df=3 each).

    Identical method to the frozen stopping analyzer (analyze_stopping.py).
    """
    t2 = T_QUANTILE_95_DF3**2
    denom = q * q - t2 * se_q * se_q
    if denom <= 0.0:
        return None  # unreachable when CI(Q) excludes zero; guard anyway
    center = c * q
    spread = (
        t2 * (c * c * se_q * se_q + q * q * se_c * se_c - t2 * se_c * se_c * se_q * se_q)
    ) ** 0.5
    return {
        "point_estimate_forwards": c / q,
        "ci95_low_forwards": (center - spread) / denom,
        "ci95_high_forwards": (center + spread) / denom,
    }


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def se_from_ci(row: dict[str, Any]) -> float:
    return (row["ci95_high_us"] - row["ci95_low_us"]) / (2 * T_QUANTILE_95_DF3)


def moves_from_l2p(l2p: list[int]) -> int:
    return sum(1 for logical, physical in enumerate(l2p)
               if physical // EXPERTS_PER_RANK != logical // EXPERTS_PER_RANK)


def verify_inputs(spec: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    loaded = {}
    for name, record in spec["inputs"].items():
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = root / path
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"input hash differs for {name}: {path}")
        loaded[name] = load_json(path)
    return loaded


def breakeven_row(q_mean: float, q_ci: tuple[float, float], verdict: str,
                  c_mean: float, c_se: float, cost_label: str) -> dict[str, Any]:
    se_q = (q_ci[1] - q_ci[0]) / (2 * T_QUANTILE_95_DF3)
    row: dict[str, Any] = {
        "cost_basis": cost_label,
        "migration_cost_mean_us": c_mean,
        "migration_cost_se_us": c_se,
        "q_mean_us": q_mean,
    }
    if verdict == "excludes_zero_positive":
        interval = fieller_break_even(c_mean, c_se, q_mean, se_q)
        if interval is None:
            raise ValueError(f"Fieller interval unbounded despite positive CI: {row}")
        row.update(status="finite_break_even", **interval)
    elif verdict == "compatible_with_zero":
        row.update(
            status="indeterminate_ci_crosses_zero",
            rule="a CI crossing zero is compatible with zero, not proof of no effect; no finite break-even is claimed",
        )
    else:
        row.update(
            status="no_finite_beneficial_breakeven",
            rule="Q <= 0 measured; migration can never pay for itself per forward",
        )
    return row


def classify(row: dict[str, Any]) -> str:
    return row["status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    spec_path = pathlib.Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = load_json(spec_path)

    if spec.get("status") != "frozen_before_cost_benefit_comparison":
        raise ValueError(f"spec status differs: {spec.get('status')}")
    if spec.get("study_id") != "h20-placement-nearscan-v1":
        raise ValueError("spec study differs")
    self_hash = sha256_file(pathlib.Path(__file__).resolve())
    bound = spec["analyzer_implementation"]["sha256"]
    if self_hash != bound:
        raise ValueError(f"analyzer hash differs from frozen binding: {self_hash}")

    inputs = verify_inputs(spec, root)
    scan = inputs["scan_analysis"]
    stop = inputs["stopping_analysis"]
    layout_index = inputs["scan_layout_index"]
    mig_protocol = inputs["stopping_migration_protocol"]

    if scan.get("study_id") != "h20-placement-nearscan-v1":
        raise ValueError("scan analysis study differs")
    if scan.get("analysis") != "near_equilibrium_scan_directional_effects":
        print(f"note: scan analysis label = {scan.get('analysis')!r}")

    # ------------------------------------------------------------------ costs
    migration_costs = stop["migration_costs"]
    cost_inputs: dict[str, Any] = {}
    for capture_id in sorted(migration_costs):
        txs = migration_costs[capture_id]
        singles = [txs[k]["combined"] for k in sorted(txs) if k != "multi-batch"]
        if len(singles) != 12:
            raise ValueError(f"{capture_id}: expected 12 single-swap transactions")
        single_mean = sum(s["mean_us"] for s in singles) / 12.0
        single_se = (sum(se_from_ci(s) ** 2 for s in singles) ** 0.5) / 12.0
        multi = txs["multi-batch"]["combined"]
        cost_inputs[capture_id] = {
            "single_swap_combined_mean_us": single_mean,
            "single_swap_combined_se_us": single_se,
            "multi_batch_combined_mean_us": multi["mean_us"],
            "multi_batch_combined_se_us": se_from_ci(multi),
        }

    # ------------------------------------------------------------------ moves
    manifests: dict[str, dict[str, Any]] = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {path}")
        manifest = load_json(path)
        if manifest.get("capture_id") != record["capture_id"]:
            raise ValueError("manifest capture id differs from index")
        manifests[record["capture_id"]] = manifest

    moves_by_capture: dict[str, dict[str, dict[str, Any]]] = {}
    for capture_id, manifest in sorted(manifests.items()):
        per_layout: dict[str, Any] = {}
        for layout in manifest["layouts"]:
            moves = moves_from_l2p(layout["l2p"])
            if layout["role"] == "anchor_identity" and moves != 0:
                raise ValueError(f"{capture_id}: identity has nonzero moves")
            per_layout[layout["layout_id"]] = {
                "moves_vs_identity": moves,
                "step_index": layout["step_index"],
                "role": layout["role"],
                "map_sha256": layout["map_sha256"],
                "r_raw": layout["r_raw"],
            }
        base = per_layout["base"]
        proto_txs = mig_protocol["transactions"][capture_id]
        proto_multi = [t for t in proto_txs if t["kind"] == "multi_swap_batch"]
        if len(proto_multi) != 1:
            raise ValueError(f"{capture_id}: expected one multi-swap batch transaction")
        if base["moves_vs_identity"] != proto_multi[0]["displaced_expert_count"]:
            raise ValueError(
                f"{capture_id}: scan base moves {base['moves_vs_identity']} != "
                f"stopping protocol displaced_expert_count {proto_multi[0]['displaced_expert_count']}"
            )
        cost_inputs[capture_id]["moves_base"] = base["moves_vs_identity"]
        cost_inputs[capture_id]["displaced_expert_count_stopping_protocol"] = (
            proto_multi[0]["displaced_expert_count"]
        )
        moves_by_capture[capture_id] = per_layout

    # ------------------------------------------------------- cumulative rows
    cumulative_rows: list[dict[str, Any]] = []
    for contrast in scan["identity_cumulative_contrasts"]:
        capture_id = contrast["capture_id"]
        layout_id = contrast["layout_b"]
        meta = moves_by_capture[capture_id][layout_id]
        costs = cost_inputs[capture_id]
        moves_base = costs["moves_base"]
        moves_k = meta["moves_vs_identity"]
        if moves_base <= 0:
            raise ValueError(f"{capture_id}: nonpositive base moves")
        frac = moves_k / moves_base
        c_batch = costs["multi_batch_combined_mean_us"] * frac
        se_batch = costs["multi_batch_combined_se_us"] * frac
        k = meta["step_index"]
        c_seq = k * costs["single_swap_combined_mean_us"]
        se_seq = k * costs["single_swap_combined_se_us"]
        q = contrast["mean_us"]
        q_ci = (contrast["ci95_low_us"], contrast["ci95_high_us"])
        verdict = contrast["verdict"]
        cumulative_rows.append({
            "capture_id": capture_id,
            "layout_id": layout_id,
            "target_region": contrast["target_region"],
            "step_index": k,
            "moves_vs_identity": moves_k,
            "moves_fraction_of_base": frac,
            "q_mean_us": q,
            "q_ci95_low_us": q_ci[0],
            "q_ci95_high_us": q_ci[1],
            "q_verdict": verdict,
            "practically_equivalent_within_margin": contrast["practically_equivalent_within_margin"],
            "c_batch_mean_us": c_batch,
            "c_batch_se_us": se_batch,
            "c_batch_estimator": "measured_multi_batch_endpoint_x_moves_fraction",
            "breakeven_batch": breakeven_row(q, q_ci, verdict, c_batch, se_batch,
                                             "batch estimator (linear-in-moves, endpoint-anchored)"),
            "c_seq_mean_us": c_seq,
            "c_seq_se_us": se_seq,
            "c_seq_estimator": "sequential single-swap upper bound k*C_single",
            "breakeven_seq": breakeven_row(q, q_ci, verdict, c_seq, se_seq,
                                           "sequential single-swap upper bound"),
        })
    if len(cumulative_rows) != 83:
        raise ValueError(f"expected 83 cumulative rows, got {len(cumulative_rows)}")

    # --------------------------------------------- marginal final step rows
    marginal_rows: list[dict[str, Any]] = []
    for contrast in scan["adjacent_contrasts"]:
        if contrast["layout_b"] != "base":
            continue
        capture_id = contrast["capture_id"]
        costs = cost_inputs[capture_id]
        q = contrast["mean_us"]
        q_ci = (contrast["ci95_low_us"], contrast["ci95_high_us"])
        verdict = contrast["verdict"]
        row = {
            "capture_id": capture_id,
            "contrast": contrast["contrast"],
            "entering_region": contrast["entering_region"],
            "delta_r": contrast["delta_r"],
            "q_mean_us": q,
            "q_ci95_low_us": q_ci[0],
            "q_ci95_high_us": q_ci[1],
            "q_verdict": verdict,
            "cost_model": "one measured single-swap transaction (per-capture mean of 12)",
            "breakeven_single_swap": breakeven_row(
                q, q_ci, verdict,
                costs["single_swap_combined_mean_us"],
                costs["single_swap_combined_se_us"],
                "single measured swap transaction",
            ),
        }
        marginal_rows.append(row)
    if len(marginal_rows) != 12:
        raise ValueError(f"expected 12 marginal final-step rows, got {len(marginal_rows)}")

    # ------------------------------------------------- horizon net benefit
    horizons = spec["horizon_net_benefit"]["horizons_forwards"]
    verdict_by_key = {(r["capture_id"], r["layout_id"]): r["q_verdict"] for r in cumulative_rows}
    q_by_key = {(r["capture_id"], r["layout_id"]): r["q_mean_us"] for r in cumulative_rows}
    c_batch_by_key = {(r["capture_id"], r["layout_id"]): r["c_batch_mean_us"] for r in cumulative_rows}
    per_capture_horizon: dict[str, Any] = {}
    summary_by_horizon: dict[str, Any] = {str(n): {"optimal_at_base": 0, "optimal_at_intermediate": 0,
                                                   "optimal_at_identity": 0,
                                                   "optimal_net_us_values": []} for n in horizons}
    for capture_id in sorted(moves_by_capture):
        layouts = moves_by_capture[capture_id]
        ordered = sorted(layouts.items(), key=lambda kv: kv[1]["step_index"])
        entry: dict[str, Any] = {"steps": [], "by_horizon": {}}
        for layout_id, meta in ordered:
            if layout_id == "identity":
                entry["steps"].append({"layout_id": layout_id, "step_index": 0,
                                       "q_mean_us": 0.0, "c_batch_mean_us": 0.0,
                                       "q_verdict": "anchor_identity"})
                continue
            entry["steps"].append({
                "layout_id": layout_id,
                "step_index": meta["step_index"],
                "q_mean_us": q_by_key[(capture_id, layout_id)],
                "c_batch_mean_us": c_batch_by_key[(capture_id, layout_id)],
                "q_verdict": verdict_by_key[(capture_id, layout_id)],
            })
        for n in horizons:
            best = None
            for step in entry["steps"]:
                net = n * step["q_mean_us"] - step["c_batch_mean_us"]
                cand = (net, -step["step_index"], step)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            net_at = {s["layout_id"]: n * s["q_mean_us"] - s["c_batch_mean_us"] for s in entry["steps"]}
            optimal = best[2]
            entry["by_horizon"][str(n)] = {
                "optimal_layout_id": optimal["layout_id"],
                "optimal_step_index": optimal["step_index"],
                "optimal_net_us": best[0],
                "optimal_q_verdict": optimal["q_verdict"],
                "base_net_us": net_at["base"],
                "identity_net_us": 0.0,
            }
            bucket = ("optimal_at_identity" if optimal["layout_id"] == "identity"
                      else "optimal_at_base" if optimal["layout_id"] == "base"
                      else "optimal_at_intermediate")
            summary_by_horizon[str(n)][bucket] += 1
            summary_by_horizon[str(n)]["optimal_net_us_values"].append(best[0])
        per_capture_horizon[capture_id] = entry
    for n in horizons:
        vals = summary_by_horizon[str(n)]["optimal_net_us_values"]
        summary_by_horizon[str(n)]["mean_optimal_net_us"] = sum(vals) / len(vals)
        summary_by_horizon[str(n)]["min_optimal_net_us"] = min(vals)

    # ------------------------------------------------------- cross-study ctx
    stop_be = stop["break_even"]
    stop_counts: dict[str, int] = {}
    for row in stop_be:
        stop_counts[row["status"]] = stop_counts.get(row["status"], 0) + 1
    finite_stop = [r for r in stop_be if r["status"] == "finite_break_even"]
    all_single = [cost_inputs[c]["single_swap_combined_mean_us"] for c in sorted(cost_inputs)]
    all_multi = [cost_inputs[c]["multi_batch_combined_mean_us"] for c in sorted(cost_inputs)]

    def status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            s = classify(row[key])
            out[s] = out.get(s, 0) + 1
        return out

    finite_batch = [r for r in cumulative_rows if r["breakeven_batch"]["status"] == "finite_break_even"]
    finite_points = sorted(r["breakeven_batch"]["point_estimate_forwards"] for r in finite_batch)

    summary = {
        "cumulative_rows": len(cumulative_rows),
        "marginal_final_step_rows": len(marginal_rows),
        "breakeven_batch_status_counts": status_counts(cumulative_rows, "breakeven_batch"),
        "breakeven_seq_status_counts": status_counts(cumulative_rows, "breakeven_seq"),
        "marginal_single_swap_status_counts": status_counts(marginal_rows, "breakeven_single_swap"),
        "finite_batch_breakeven_rows": len(finite_batch),
        "finite_batch_breakeven_forwards": {
            "min": finite_points[0] if finite_points else None,
            "median": finite_points[len(finite_points) // 2] if finite_points else None,
            "max": finite_points[-1] if finite_points else None,
        },
        "single_swap_cost_per_capture_mean_us_range": [min(all_single), max(all_single)],
        "multi_batch_cost_per_capture_mean_us_range": [min(all_multi), max(all_multi)],
        "stopping_study_break_even_counts": stop_counts,
        "stopping_study_finite_rows": finite_stop,
    }

    output = {
        "schema_version": 1,
        "analysis": "scan_low_gain_actions_versus_measured_migration_costs",
        "study_id": spec["study_id"],
        "spec_id": spec["spec_id"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analyzer_implementation": {"path": "scripts/analyze_scan_cost_benefit.py", "sha256": self_hash},
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "inputs": {name: {"path": rec["path"], "sha256": rec["sha256"]}
                   for name, rec in spec["inputs"].items()},
        "cost_model_inputs_by_capture": cost_inputs,
        "se_approximation_note": (
            "single-swap per-capture SE = sqrt(sum of squared per-transaction SEs)/12; the 12 "
            "transactions share the same 4 starts within a capture, so cross-transaction "
            "independence is an approximation; it affects only Fieller interval widths, never "
            "the status trichotomy, which is driven by the frozen scan verdicts"
        ),
        "cumulative_break_even_rows": cumulative_rows,
        "marginal_final_step_rows": marginal_rows,
        "horizon_net_benefit": {
            "horizons_forwards": horizons,
            "net_definition": spec["horizon_net_benefit"]["definition"],
            "per_capture": per_capture_horizon,
            "summary_by_horizon": summary_by_horizon,
        },
        "cross_study_context": {
            "stopping_break_even_counts": stop_counts,
            "stopping_finite_row": finite_stop,
            "note": (
                "the stopping campaign's residual single swaps around near-balanced bases were "
                "overwhelmingly indeterminate or harmful; this scan extends the same comparison "
                "to full greedy-prefix adoption from identity"
            ),
        },
        "summary": summary,
        "interpretation_rules": spec["interpretation_rules"],
    }

    out_path = root / OUTPUT_REL
    if out_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {out_path}")
    write_json_atomic(out_path, output)
    print(json.dumps({"status": "cost_benefit_comparison_complete",
                      "output": str(out_path),
                      "sha256": sha256_file(out_path),
                      "summary": {k: v for k, v in summary.items() if k != "stopping_study_finite_rows"}},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
