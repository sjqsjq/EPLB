#!/usr/bin/env python3
"""Analysis of the frozen trigger x map crosstab campaign (h20-placement-crosstab-v1).

Frozen BEFORE any crosstab timing exists; this file's hash is locked into the
crosstab execution protocol. Implements exactly the pre-registered analysis:

  cells      T(capture, column) = mean over 4 independent starts of the
             5-block mean completion_fenced_us_per_forward (primary metric,
             no fence subtraction, no overhead subtraction anywhere).
  contrasts  within-capture, start-level means of within-block paired
             differences; Student-t df=3, T=3.1824463052837078.
  verdicts   per layer: tracking_room_supported / common_static_sufficient /
             mixed, by the rules frozen in the protocol analysis_spec.
  extras     scan-era drift cross-check (reference only, never in verdicts)
             and cost linkage via the frozen batch estimator
             C_batch(k) = C_multi(capture) * moves_k / moves_base.

Interpretation discipline (inherited verbatim): an interval crossing zero means
"compatible with zero", never proof of no effect and never equivalence;
equivalence is claimed only when the interval lies wholly inside
[-epsilon, +epsilon]; negative effects are preserved exactly; no best-start,
repeat, prefix or subset selection anywhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from typing import Any

from prediction_common import sha256_file, write_json_atomic

CROSSTAB_STUDY_ID = "h20-placement-crosstab-v1"
SCAN_STUDY_ID = "h20-placement-nearscan-v1"
BLOCKS_PER_START = 5
STARTS_PER_CAPTURE = 4
T_QUANTILE_95_DF3 = 3.1824463052837078
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
LAYERS = (24, 28, 44, 71)
BATCHES = ("evaluation-01", "evaluation-02", "evaluation-03")
COLUMNS = ("identity", "base-of-evaluation-01", "base-of-evaluation-02", "base-of-evaluation-03")
OWN_COLUMN = {batch: f"base-of-{batch}" for batch in BATCHES}

try:  # guard: hard-coded constant must equal the frozen scipy value
    from scipy import stats as _stats

    assert abs(float(_stats.t.ppf(0.975, 3)) - T_QUANTILE_95_DF3) < 1e-12
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--execution-index", type=pathlib.Path, required=True)
    parser.add_argument("--scan-execution-index", type=pathlib.Path)
    parser.add_argument("--cost-benefit", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    if len(values) != STARTS_PER_CAPTURE:
        raise ValueError(f"start-level sample size differs: {len(values)}")
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    se = var**0.5 / len(values) ** 0.5
    half = T_QUANTILE_95_DF3 * se
    return mean, mean - half, mean + half


def classify(mean: float, low: float, high: float, eps: float) -> dict[str, Any]:
    inside_margin = low >= -eps and high <= eps
    if low > 0.0:
        verdict = "excludes_zero_positive"
    elif high < 0.0:
        verdict = "excludes_zero_negative"
    else:
        verdict = "compatible_with_zero"
    return {
        "mean_us": mean,
        "ci95_low_us": low,
        "ci95_high_us": high,
        "verdict": verdict,
        "practically_equivalent_within_margin": inside_margin,
    }


def welch_interval(a: list[float], b: list[float]) -> dict[str, Any]:
    """Reference-only drift cross-check: Welch CI for mean(a) - mean(b)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((v - ma) ** 2 for v in a) / (na - 1)
    vb = sum((v - mb) ** 2 for v in b) / (nb - 1)
    se = (va / na + vb / nb) ** 0.5
    if se == 0.0:
        return {"mean_us": ma - mb, "ci95_low_us": ma - mb, "ci95_high_us": ma - mb,
                "verdict": "zero_variance_reference_only"}
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    from scipy import stats

    t = float(stats.t.ppf(0.975, df))
    return {"mean_us": ma - mb, "ci95_low_us": ma - mb - t * se,
            "ci95_high_us": ma - mb + t * se, "welch_df": df,
            "verdict": "reference_only_drift_cross_check"}


def load_tables(protocol: dict[str, Any], root: pathlib.Path,
                index: dict[str, Any]) -> dict[str, list[dict[int, dict[str, float]]]]:
    by_start = {record["start_id"]: record for record in index["results"]}
    starts_by_capture: dict[str, list[dict[str, Any]]] = {}
    for start in protocol["starts"]:
        starts_by_capture.setdefault(start["capture_id"], []).append(start)
    tables_by_capture: dict[str, list[dict[int, dict[str, float]]]] = {}
    for capture_id in sorted(starts_by_capture):
        entries = sorted(starts_by_capture[capture_id], key=lambda s: s["start_id"])
        if len(entries) != STARTS_PER_CAPTURE:
            raise ValueError(f"{capture_id}: start count differs")
        tables = []
        for start in entries:
            record = by_start.get(start["start_id"])
            if record is None:
                raise ValueError(f"missing indexed start {start['start_id']}")
            path = pathlib.Path(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"raw result hash differs: {start['start_id']}")
            raw = json.loads(path.read_text())
            if raw.get("study_id") != CROSSTAB_STUDY_ID:
                raise ValueError(f"raw study differs: {start['start_id']}")
            if raw.get("capture_id") != capture_id:
                raise ValueError(f"raw capture differs: {start['start_id']}")
            if raw.get("protocol", {}).get("sha256") != sha256_file(
                pathlib.Path(protocol["_protocol_path"])
            ):
                raise ValueError(f"raw protocol hash differs: {start['start_id']}")
            records = raw["primary_timing_records"]
            if len(records) != len(COLUMNS) * BLOCKS_PER_START:
                raise ValueError(f"{start['start_id']}: primary cardinality differs")
            table: dict[int, dict[str, float]] = {}
            for rec in records:
                if rec["pass"] != "primary":
                    raise ValueError(f"{start['start_id']}: unexpected pass {rec['pass']}")
                table.setdefault(int(rec["block"]), {})[rec["layout_id"]] = float(
                    rec["completion_fenced_us_per_forward"]
                )
            if len(table) != BLOCKS_PER_START:
                raise ValueError(f"{start['start_id']}: block count differs")
            for block, completions in table.items():
                if set(completions) != set(COLUMNS):
                    raise ValueError(f"{start['start_id']}: block {block} layout set differs")
            tables.append(table)
        tables_by_capture[capture_id] = tables
    return tables_by_capture


def paired_effect(tables: list[dict[int, dict[str, float]]], a: str, b: str) -> list[float]:
    start_effects = []
    for table in tables:
        block_effects = [table[blk][a] - table[blk][b] for blk in sorted(table)]
        start_effects.append(sum(block_effects) / len(block_effects))
    return start_effects


def cell_means(tables: list[dict[int, dict[str, float]]], column: str) -> list[float]:
    return [
        sum(table[blk][column] for blk in sorted(table)) / len(table)
        for table in tables
    ]


def hamming(p2l_a: list[int], p2l_b: list[int]) -> int:
    return sum(1 for x, y in zip(p2l_a, p2l_b) if x != y)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol["_protocol_path"] = str(protocol_path)
    if protocol.get("study_id") != CROSSTAB_STUDY_ID:
        raise ValueError("protocol belongs to another study")
    if protocol.get("status") != "frozen_before_crosstab_timing":
        raise ValueError("protocol is not frozen")
    if protocol.get("split") != "crosstab":
        raise ValueError("protocol split differs")
    spec = protocol.get("analysis_spec", {})
    if float(spec.get("practical_equivalence_margin_us", -1)) != PRACTICAL_EQUIVALENCE_MARGIN_US:
        raise ValueError("protocol margin differs from analyzer constant")
    if float(spec.get("t_quantile_95_df3", -1)) != T_QUANTILE_95_DF3:
        raise ValueError("protocol t quantile differs from analyzer constant")

    index_path = args.execution_index.resolve()
    index = json.loads(index_path.read_text())
    if index.get("study_id") != CROSSTAB_STUDY_ID:
        raise ValueError("execution index belongs to another study")
    if index.get("status") != "all_crosstab_timing_results_validated_and_hash_indexed":
        raise ValueError("execution index status differs")
    if index.get("counts", {}).get("starts") != len(LAYERS) * len(BATCHES) * STARTS_PER_CAPTURE:
        raise ValueError("execution index start count differs")

    manifests = {}
    for record in protocol["layout_manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {record['capture_id']}")
        manifests[record["capture_id"]] = json.loads(path.read_text())

    tables_by_capture = load_tables(protocol, root, index)
    eps = PRACTICAL_EQUIVALENCE_MARGIN_US

    # cost model inputs (frozen measured migration costs from the scan study)
    cost_inputs = {}
    cost_path = args.cost_benefit or root / "results/nearscan_v1/cost_benefit.json"
    if cost_path.is_file():
        cb = json.loads(cost_path.read_text())
        cost_inputs = cb.get("cost_model_inputs_by_capture", {})

    layers_out: dict[str, Any] = {}
    pooled_own_vs_identity = []
    pooled_own_vs_best_other = []
    drift_rows = []

    # optional scan drift cross-check data
    scan_cells: dict[tuple[str, str], list[float]] = {}
    scan_index_path = args.scan_execution_index
    if scan_index_path is None:
        candidate = root / "results/nearscan_v1/execution/index.json"
        scan_index_path = candidate if candidate.is_file() else None
    if scan_index_path is not None:
        scan_index = json.loads(pathlib.Path(scan_index_path).read_text())
        if scan_index.get("study_id") == SCAN_STUDY_ID:
            for record in scan_index["results"]:
                raw = json.loads(pathlib.Path(record["path"]).read_text())
                table: dict[int, dict[str, float]] = {}
                for rec in raw["primary_timing_records"]:
                    table.setdefault(int(rec["block"]), {})[rec["layout_id"]] = float(
                        rec["completion_fenced_us_per_forward"]
                    )
                start_means = {
                    layout: sum(table[b][layout] for b in sorted(table)) / len(table)
                    for layout in table[sorted(table)[0]]
                }
                key_capture = raw["capture_id"]
                for layout, value in start_means.items():
                    if layout in ("identity", "base"):
                        scan_cells.setdefault((key_capture, layout), []).append(value)

    for layer in LAYERS:
        layer_records = {}
        rows_tracking = {}
        column_checks = {column: [] for column in COLUMNS[1:]}
        for batch in BATCHES:
            capture_id = f"pred-v1-{batch}-layer-{layer:03d}"
            manifest = manifests[capture_id]
            tables = tables_by_capture[capture_id]
            own = OWN_COLUMN[batch]
            meta = {rec["layout_id"]: rec for rec in manifest["layouts"]}

            cells = {}
            for column in COLUMNS:
                start_means = cell_means(tables, column)
                mean, low, high = mean_interval(start_means)
                cells[column] = {
                    **classify(mean, low, high, eps),
                    "map_sha256": meta[column]["map_sha256"],
                    "r_raw_on_this_capture": meta[column]["r_raw"],
                    "padded_max_over_mean": meta[column]["padded_max_over_mean"],
                    "start_means_us": start_means,
                }

            contrasts = {}
            own_vs_identity = paired_effect(tables, own, "identity")
            mean, low, high = mean_interval(own_vs_identity)
            contrasts["own_base_vs_identity"] = classify(mean, low, high, eps)
            pooled_own_vs_identity.append(
                {"capture_id": capture_id, **classify(mean, low, high, eps)}
            )
            others_vs_own = {}
            for other in (b for b in BATCHES if b != batch):
                col = OWN_COLUMN[other]
                effects = paired_effect(tables, col, own)
                mean, low, high = mean_interval(effects)
                others_vs_own[col] = classify(mean, low, high, eps)
            contrasts["other_base_vs_own_base"] = others_vs_own
            row_tracks = all(
                entry["verdict"] == "excludes_zero_positive"
                for entry in others_vs_own.values()
            )
            rows_tracking[batch] = row_tracks

            best_other_col = min(others_vs_own, key=lambda c: others_vs_own[c]["mean_us"])
            pooled_own_vs_best_other.append(
                {"capture_id": capture_id, "column": best_other_col,
                 **classify(others_vs_own[best_other_col]["mean_us"],
                            others_vs_own[best_other_col]["ci95_low_us"],
                            others_vs_own[best_other_col]["ci95_high_us"], eps)}
            )

            for column in COLUMNS[1:]:
                if column == own:
                    column_checks[column].append(True)  # own row: delta = 0 by construction
                    continue
                effects = paired_effect(tables, column, own)
                mean, low, high = mean_interval(effects)
                column_checks[column].append(low <= eps)

            cost_linkage = []
            cap_costs = cost_inputs.get(capture_id)
            if cap_costs is not None:
                moves_base = int(cap_costs["moves_base"])
                c_multi = float(cap_costs["multi_batch_combined_mean_us"])
                for column in COLUMNS[1:]:
                    if column == own:
                        continue
                    moves_k = hamming(meta[column]["p2l"], meta[own]["p2l"])
                    est_cost = c_multi * moves_k / moves_base if moves_base else None
                    gain_entry = others_vs_own[column]
                    gain = gain_entry["mean_us"]
                    if est_cost is None or not math.isfinite(gain) or gain <= 0.0:
                        n_star = None
                        status = "no_finite_break_even"
                    else:
                        n_star = est_cost / gain
                        status = "finite"
                    cost_linkage.append({
                        "switch_from_column": column,
                        "switch_to_column": own,
                        "moves_k_experts": moves_k,
                        "estimated_migration_cost_us": est_cost,
                        "measured_per_forward_gain_us": gain,
                        "gain_verdict": gain_entry["verdict"],
                        "break_even_forwards_n_star": n_star,
                        "status": status,
                        "estimator": "C_batch(k)=C_multi(capture)*moves_k/moves_base (frozen scan cost-benefit)",
                    })

            layer_records[batch] = {
                "capture_id": capture_id,
                "cells": cells,
                "contrasts": contrasts,
                "row_tracks_own_base": row_tracks,
                "cost_linkage": cost_linkage,
            }

            if (capture_id, "identity") in scan_cells and len(scan_cells[(capture_id, "identity")]) == STARTS_PER_CAPTURE:
                drift_rows.append({
                    "capture_id": capture_id, "column": "identity",
                    "crosstab_mean_us": cells["identity"]["mean_us"],
                    "scan_mean_us": sum(scan_cells[(capture_id, "identity")]) / STARTS_PER_CAPTURE,
                    **welch_interval(cells["identity"]["start_means_us"],
                                     scan_cells[(capture_id, "identity")]),
                })
            if (capture_id, "base") in scan_cells and len(scan_cells[(capture_id, "base")]) == STARTS_PER_CAPTURE:
                drift_rows.append({
                    "capture_id": capture_id, "column": own,
                    "crosstab_mean_us": cells[own]["mean_us"],
                    "scan_mean_us": sum(scan_cells[(capture_id, "base")]) / STARTS_PER_CAPTURE,
                    **welch_interval(cells[own]["start_means_us"],
                                     scan_cells[(capture_id, "base")]),
                })

        tracking = [rows_tracking[b] for b in BATCHES]
        tracking_room = sum(tracking) >= 2
        sufficient_columns = [c for c in COLUMNS[1:] if all(column_checks[c])]
        common_static = len(sufficient_columns) > 0
        if tracking_room:
            verdict = "tracking_room_supported"
        elif common_static:
            verdict = "common_static_sufficient"
        else:
            verdict = "mixed"
        layers_out[f"layer-{layer:03d}"] = {
            "rows": layer_records,
            "rows_tracking_own_base": {b: rows_tracking[b] for b in BATCHES},
            "tracking_room_supported": tracking_room,
            "common_static_sufficient_columns": sufficient_columns,
            "common_static_sufficient": common_static,
            "verdict": verdict,
        }

    output = {
        "schema_version": 1,
        "study_id": CROSSTAB_STUDY_ID,
        "analysis": "trigger_x_map_crosstab_frozen_analysis",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analyzer_implementation": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "execution_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "statistics": {
            "t_quantile_95_df3": T_QUANTILE_95_DF3,
            "starts_per_capture": STARTS_PER_CAPTURE,
            "blocks_per_start": BLOCKS_PER_START,
            "practical_equivalence_margin_us": eps,
            "paired_within_block": True,
            "overhead_subtracted_from_contrasts": False,
        },
        "verdict_rules": {
            "row_tracks_own_base": "both other-base-minus-own-base paired CIs exclude zero positively",
            "tracking_room_supported": ">=2 of 3 rows track their own base",
            "common_static_sufficient": (
                "some base column's delta-vs-own CI lower bound is <= epsilon in every row "
                "(not significantly worse beyond the practical margin anywhere)"
            ),
            "precedence": "tracking_room_supported > common_static_sufficient > mixed",
        },
        "layers": layers_out,
        "pooled": {
            "own_base_vs_identity": pooled_own_vs_identity,
            "own_base_vs_best_other_base": pooled_own_vs_best_other,
        },
        "drift_cross_check_reference_only": {
            "note": (
                "scan-era identity/base cells vs crosstab cells for the same capture and map; "
                "Welch reference only, never used in any verdict"
            ),
            "rows": drift_rows,
        },
        "interpretation_rules": [
            "an interval crossing zero means compatible with zero, not proof of no effect and not equivalence",
            "equivalence is claimed only when the interval lies wholly inside [-epsilon,+epsilon]",
            "negative effects are preserved exactly as measured",
            "no best-start, repeat, prefix or subset selection was applied",
            "crosstab numbers are layer-replay completion times, not server-side latencies",
        ],
    }
    write_json_atomic(args.output.resolve(), output)

    for layer_key, layer_out in layers_out.items():
        print(f"== {layer_key}: verdict={layer_out['verdict']} "
              f"tracking_rows={list(layer_out['rows_tracking_own_base'].values())} "
              f"sufficient_columns={layer_out['common_static_sufficient_columns']}")
        header = f"{'row':<16}" + "".join(f"{c.replace('base-of-',''):>22}" for c in COLUMNS)
        print(header)
        for batch in BATCHES:
            row = layer_out["rows"][batch]["cells"]
            line = f"{batch:<16}"
            for column in COLUMNS:
                line += f"{row[column]['mean_us']:>14.2f}±{(row[column]['ci95_high_us']-row[column]['ci95_low_us'])/2:>6.2f}"
            print(line)
    print(f"wrote {args.output} sha256={sha256_file(args.output.resolve())}")


if __name__ == "__main__":
    main()
