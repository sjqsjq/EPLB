#!/usr/bin/env python3
"""Frozen analysis of the CD-X0 cross-domain replay pilot (h20-placement-cdx0-v1).

Written and hash-locked into the execution protocol BEFORE any CD-X0 timing.
Implements exactly the pre-registered analysis (cdx0_pilot_protocol.json +
cdx0_amendment_v1.json, which governs where they conflict):

  cells      T(row, column) = mean over valid starts (4 planned, >=3 required)
             of the 5-block mean completion_fenced_us_per_forward (primary
             metric; no fence subtraction, no overhead subtraction anywhere).
  primary    24 paired contrasts = 2 layers x 4 rows x 3 own-vs-other-base
             (T(x_d, L_other) - T(x_d, L_own)); start-level means of
             within-block paired differences; Student-t df = valid_starts - 1.
             ALL primary labels are EXPLORATORY screening (amendment 3): no
             family-wise error rate control is claimed; Holm-Bonferroni
             adjusted p-values are computed WITHIN each layer (family = 12)
             as descriptive sensitivity only and never alter the verdict.
  verdicts   row_tracks_own_base = all three own-vs-other CIs exclude zero
             positively (epsilon plays NO role in verdicts, amendment 2);
             clear_crossover_advantage = >=2 of 4 rows track; layer
             indeterminate if >=2 rows have <3 valid starts; overall
             detected / not_detected / indeterminate, every verdict string
             carries "exploratory" and speaks ONLY about the chosen layers
             and candidates at replay caliber - never whole-service benefit
             or harm in either direction.
  mandatory  complete per-layer 4x5 matrix (cell estimates, per-start values,
  reporting  all within-row paired CIs - every one of the 10 column pairs per
             row, a superset of the amendment-4 floor of 20 per layer), per-row
             best column, descriptive oracle gaps (row-equal K and K-tilde with
             the mandatory assumption note incl. the row-size clause), pairwise
             displacement counts and frozen C_batch migration-cost context.

Interpretation discipline (inherited verbatim): an interval crossing zero
means "compatible with zero", never proof of no effect and never equivalence;
equivalence is claimed only when the interval lies wholly inside
[-epsilon, +epsilon]; negative effects are preserved exactly; no best-start,
repeat, prefix or subset selection anywhere; failed starts stay retired and
are reported, never retried, never silently dropped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cdx0_common import sha256_file, write_json_atomic  # noqa: E402

CDX0_STUDY_ID = "h20-placement-cdx0-v1"
LAYERS = (24, 71)
DOMAINS = ("sharegpt", "prover", "book", "code")
COLUMNS = ("identity",) + tuple(f"base-of-{d}" for d in DOMAINS)
OWN_COLUMN = {d: f"base-of-{d}" for d in DOMAINS}
BLOCKS_PER_START = 5
PLANNED_STARTS = 4
MIN_VALID_STARTS = 3
T_QUANTILE_95 = {3: 3.1824463052837078, 2: 4.302652729911275}
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
K_TILDE = {"sharegpt": 0.157, "prover": 0.12, "book": 0.2372, "code": 0.4857}
K_TILDE_NOTE = (
    "K_tilde = evaluation token shares, an EXOGENOUS approximation of each "
    "domain's service relevance; CD-X0 rows are NOT equal-token (book ~2x, "
    "code ~4x the sharegpt/prover row sizes), so K_tilde weighting is an "
    "assumption-flagged approximation, not a replay-measured quantity; no "
    "service-share multiplication may be applied to these numbers and no "
    "full-service benefit or harm may be inferred in either direction"
)

try:  # guard: hard-coded constants versus scipy (see audit note below for df=2)
    from scipy import stats as _stats

    # df=3: the frozen pre-registered constant equals scipy exactly.
    assert float(_stats.t.ppf(0.975, 3)) == T_QUANTILE_95[3]
    # df=2: DOCUMENTED frozen-constant discrepancy, ruled in
    # results/cdx0_v1/cdx0_t_quantile_audit_note.json: the pre-registered
    # constant 4.302652729911275 differs from the exact closed-form value
    # sqrt(2*q^2/(1-q^2)), q=0.95 (== scipy, 4.302652729749464) by 1.6e-10
    # absolute (3.8e-11 relative) - a design-phase transcription error
    # discovered AFTER protocol freeze. The frozen pre-registered constant
    # governs (it is bound in the frozen protocol analysis_spec and changing
    # it post-freeze is prohibited); its effect on any CI half-width is
    # < 1e-7 microseconds. The tolerance below admits exactly this documented
    # case and would still fail on any genuine constant error (>= 1e-9).
    assert abs(float(_stats.t.ppf(0.975, 2)) - T_QUANTILE_95[2]) < 1e-9
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--execution-index", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    return parser.parse_args()


def mean_interval(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n < 2:
        raise ValueError(f"interval needs >=2 starts, got {n}")
    # T_QUANTILE_95 is keyed by df = n - 1 (pre-registered df rule).
    if n - 1 not in T_QUANTILE_95:
        raise ValueError(f"unexpected valid-start count: {n}")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = var**0.5 / n**0.5
    half = T_QUANTILE_95[n - 1] * se
    return {"mean_us": mean, "se_us": se, "df": n - 1,
            "ci95_low_us": mean - half, "ci95_high_us": mean + half}


def classify(stats: dict[str, float], eps: float) -> dict[str, Any]:
    low, high = stats["ci95_low_us"], stats["ci95_high_us"]
    if low > 0.0:
        verdict = "excludes_zero_positive"
    elif high < 0.0:
        verdict = "excludes_zero_negative"
    else:
        verdict = "compatible_with_zero"
    return {
        **stats,
        "verdict": verdict,
        "practically_equivalent_within_margin": low >= -eps and high <= eps,
    }


def two_sided_p(stats: dict[str, float]) -> float:
    from scipy import stats as sps

    if stats["se_us"] <= 0.0:
        return 0.0 if stats["mean_us"] != 0.0 else 1.0
    t = abs(stats["mean_us"]) / stats["se_us"]
    return float(2.0 * sps.t.sf(t, stats["df"]))


def paired_effect(tables: list[dict[int, dict[str, float]]], a: str, b: str) -> list[float]:
    return [
        sum(table[blk][a] - table[blk][b] for blk in sorted(table)) / len(table)
        for table in tables
    ]


def cell_block_values(tables: list[dict[int, dict[str, float]]], column: str) -> list[list[float]]:
    return [[table[blk][column] for blk in sorted(table)] for table in tables]


def cell_start_means(tables: list[dict[int, dict[str, float]]], column: str) -> list[float]:
    return [sum(blocks) / len(blocks) for blocks in cell_block_values(tables, column)]


def hamming(p2l_a: list[int], p2l_b: list[int]) -> int:
    return sum(1 for x, y in zip(p2l_a, p2l_b) if x != y)


def load_tables(protocol: dict[str, Any], root: pathlib.Path,
                index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per capture: valid-start timing tables + retired-start records."""
    by_start = {record["start_id"]: record for record in index["results"]}
    retired_by_start = {record["start_id"]: record for record in index.get("retired_starts", [])}
    starts_by_capture: dict[str, list[dict[str, Any]]] = {}
    for start in protocol["starts"]:
        starts_by_capture.setdefault(start["capture_id"], []).append(start)
    out: dict[str, dict[str, Any]] = {}
    protocol_sha = sha256_file(pathlib.Path(protocol["_protocol_path"]))
    for capture_id in sorted(starts_by_capture):
        entries = sorted(starts_by_capture[capture_id], key=lambda s: s["start_id"])
        if len(entries) != PLANNED_STARTS:
            raise ValueError(f"{capture_id}: planned start count differs")
        tables: list[dict[int, dict[str, float]]] = []
        valid_ids: list[str] = []
        retired_ids: list[str] = []
        for start in entries:
            sid = start["start_id"]
            if sid in retired_by_start and sid in by_start:
                raise ValueError(f"{sid} is both valid and retired")
            if sid in retired_by_start:
                retired_ids.append(sid)
                continue
            record = by_start.get(sid)
            if record is None:
                raise ValueError(f"missing indexed start {sid}")
            path = pathlib.Path(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"raw result hash differs: {sid}")
            raw = json.loads(path.read_text())
            if raw.get("study_id") != CDX0_STUDY_ID:
                raise ValueError(f"raw study differs: {sid}")
            if raw.get("capture_id") != capture_id:
                raise ValueError(f"raw capture differs: {sid}")
            if raw.get("protocol", {}).get("sha256") != protocol_sha:
                raise ValueError(f"raw protocol hash differs: {sid}")
            records = raw["primary_timing_records"]
            if len(records) != len(COLUMNS) * BLOCKS_PER_START:
                raise ValueError(f"{sid}: primary cardinality differs")
            table: dict[int, dict[str, float]] = {}
            for rec in records:
                if rec["pass"] != "primary":
                    raise ValueError(f"{sid}: unexpected pass {rec['pass']}")
                table.setdefault(int(rec["block"]), {})[rec["layout_id"]] = float(
                    rec["completion_fenced_us_per_forward"]
                )
            if len(table) != BLOCKS_PER_START:
                raise ValueError(f"{sid}: block count differs")
            for block, completions in table.items():
                if set(completions) != set(COLUMNS):
                    raise ValueError(f"{sid}: block {block} layout set differs")
            tables.append(table)
            valid_ids.append(sid)
        out[capture_id] = {
            "tables": tables,
            "valid_start_ids": valid_ids,
            "retired_start_ids": retired_ids,
            "n_valid": len(tables),
        }
    return out


def oracle_gap(cells_by_row: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, Any]:
    static_by_column = {
        column: sum(weights[d] * cells_by_row[d][column] for d in DOMAINS)
        for column in COLUMNS
    }
    best_column = min(static_by_column, key=lambda c: static_by_column[c])
    t_static = static_by_column[best_column]
    t_switch = sum(weights[d] * min(cells_by_row[d].values()) for d in DOMAINS)
    gap = t_static - t_switch
    return {
        "weights": weights,
        "t_static_us_per_forward": t_static,
        "t_static_best_column": best_column,
        "t_static_all_columns": static_by_column,
        "t_switch_us_per_forward": t_switch,
        "oracle_gap_us_per_forward": gap,
        "oracle_gap_pct_of_static": 100.0 * gap / t_static if t_static else None,
        "within_margin_of_inherited_epsilon": gap < PRACTICAL_EQUIVALENCE_MARGIN_US,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol["_protocol_path"] = str(protocol_path)
    if protocol.get("study_id") != CDX0_STUDY_ID:
        raise ValueError("protocol belongs to another study")
    if protocol.get("status") != "frozen_before_cdx0_timing":
        raise ValueError("protocol is not frozen")
    if protocol.get("split") != "cdx0":
        raise ValueError("protocol split differs")
    spec = protocol.get("analysis_spec", {})
    if float(spec.get("practical_equivalence_margin_us", -1)) != PRACTICAL_EQUIVALENCE_MARGIN_US:
        raise ValueError("protocol margin differs from analyzer constant")
    if float(spec.get("t_quantile_95_df3", -1)) != T_QUANTILE_95[3]:
        raise ValueError("protocol t quantile (df3) differs from analyzer constant")
    if float(spec.get("t_quantile_95_df2", -1)) != T_QUANTILE_95[2]:
        raise ValueError("protocol t quantile (df2) differs from analyzer constant")
    # Frozen protocol nests these under oracle_gap_descriptive /
    # migration_context_descriptive (alignment fix before any analyzer output).
    oracle_spec = spec.get("oracle_gap_descriptive", {})
    if oracle_spec.get("k_tilde") != K_TILDE:
        raise ValueError("protocol K_tilde differs from analyzer constant")
    if oracle_spec.get("k_tilde_assumption_note_mandatory") != K_TILDE_NOTE:
        raise ValueError("protocol K_tilde assumption note differs from analyzer constant")
    ref_cost = float(
        spec["migration_context_descriptive"]["reference_cost_per_displaced_expert_us"]
    )

    index_path = args.execution_index.resolve()
    index = json.loads(index_path.read_text())
    if index.get("study_id") != CDX0_STUDY_ID:
        raise ValueError("execution index belongs to another study")
    if index.get("status") != "cdx0_timing_results_validated_and_hash_indexed":
        raise ValueError("execution index status differs")
    if index.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("execution index references another protocol")

    manifests: dict[str, dict[str, Any]] = {}
    for record in protocol["layout_manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {record['capture_id']}")
        manifests[record["capture_id"]] = json.loads(path.read_text())

    tables_by_capture = load_tables(protocol, root, index)
    eps = PRACTICAL_EQUIVALENCE_MARGIN_US

    layers_out: dict[str, Any] = {}
    layer_results: dict[str, str] = {}
    for layer in LAYERS:
        rows_out: dict[str, Any] = {}
        row_tracks: dict[str, Any] = {}
        primary_layer: list[dict[str, Any]] = []
        cells_point: dict[str, dict[str, float]] = {}
        for domain in DOMAINS:
            capture_id = f"cdx0-evaluation-{domain}-layer-{layer:03d}"
            manifest = manifests[capture_id]
            meta = {rec["layout_id"]: rec for rec in manifest["layouts"]}
            entry = tables_by_capture[capture_id]
            tables = entry["tables"]
            n_valid = entry["n_valid"]
            own = OWN_COLUMN[domain]

            row: dict[str, Any] = {
                "capture_id": capture_id,
                "n_valid_starts": n_valid,
                "valid_start_ids": entry["valid_start_ids"],
                "retired_start_ids": entry["retired_start_ids"],
                "row_determinate": n_valid >= MIN_VALID_STARTS,
            }
            cells: dict[str, Any] = {}
            point: dict[str, float] = {}
            for column in COLUMNS:
                start_means = cell_start_means(tables, column) if tables else []
                cell: dict[str, Any] = {
                    "map_sha256": meta[column]["map_sha256"],
                    "r_raw_on_this_capture": meta[column]["r_raw"],
                    "padded_max_over_mean": meta[column]["padded_max_over_mean"],
                    "start_means_us": start_means,
                    "per_start_block_values_us": cell_block_values(tables, column) if tables else [],
                }
                if start_means:
                    point[column] = sum(start_means) / len(start_means)
                    if n_valid >= MIN_VALID_STARTS:
                        cell.update(classify(mean_interval(start_means), eps))
                    else:
                        cell.update({"mean_us": point[column],
                                     "verdict": "insufficient_valid_starts_for_ci"})
                else:
                    cell.update({"mean_us": None, "verdict": "no_valid_starts"})
                cells[column] = cell
            row["cells"] = cells
            cells_point[domain] = point

            if n_valid >= MIN_VALID_STARTS:
                # all 10 within-row column pairs (superset of the amendment-4
                # 20-pair reporting floor); primaries flagged separately
                pair_contrasts = []
                for x, y in itertools.combinations(COLUMNS, 2):
                    if y == own and x != "identity":
                        a, b, kind = x, y, "primary_own_vs_other"
                    elif x == own and y != "identity":
                        a, b, kind = y, x, "primary_own_vs_other"
                    elif {x, y} == {"identity", own}:
                        a, b, kind = own, "identity", "own_base_vs_identity"
                    else:
                        a, b, kind = x, y, "secondary_pair"
                    effects = paired_effect(tables, a, b)
                    stats = classify(mean_interval(effects), eps)
                    pair_contrasts.append({
                        "a": a, "b": b,
                        "contrast": f"T({a}) - T({b})",
                        "kind": kind,
                        "per_start_effects_us": effects,
                        **stats,
                    })
                row["paired_contrasts_all_10_pairs"] = pair_contrasts
                primaries = [c for c in pair_contrasts if c["kind"] == "primary_own_vs_other"]
                if len(primaries) != 3:
                    raise ValueError(f"{capture_id}: primary contrast count differs")
                row_tracks[domain] = all(
                    c["verdict"] == "excludes_zero_positive" for c in primaries
                )
                for c in primaries:
                    primary_layer.append({"capture_id": capture_id, "layer_id": layer,
                                          "row_domain": domain, **c})
                best_column = min(point, key=lambda c: point[c])
                row["best_column"] = {
                    "column": best_column,
                    "delta_vs_identity_us": point[best_column] - point["identity"],
                    "delta_vs_own_base_us": point[best_column] - point[own],
                }
            else:
                row_tracks[domain] = None
                row["paired_contrasts_all_10_pairs"] = None
                row["best_column"] = None
                row["indeterminate_reason"] = (
                    f"fewer than {MIN_VALID_STARTS} valid starts; retired, never retried "
                    "under this frozen campaign; row excluded from tracking counts"
                )
            rows_out[domain] = row

        # Holm-Bonferroni within layer (family = 12 primaries), descriptive only
        for entry in primary_layer:
            entry["p_two_sided"] = two_sided_p(entry)
        ordered = sorted(range(len(primary_layer)), key=lambda i: primary_layer[i]["p_two_sided"])
        m = len(primary_layer)
        running = 0.0
        for rank_pos, i in enumerate(ordered):
            adjusted = min(1.0, (m - rank_pos) * primary_layer[i]["p_two_sided"])
            running = max(running, adjusted)
            primary_layer[i]["holm_adjusted_p_within_layer_descriptive"] = running
            primary_layer[i]["holm_rejected_alpha005_descriptive"] = running < 0.05
        primary_sorted = sorted(
            primary_layer, key=lambda e: (e["row_domain"], e["a"])
        )

        indeterminate_rows = [d for d in DOMAINS if row_tracks[d] is None]
        layer_indeterminate = len(indeterminate_rows) >= 2
        tracking_count = sum(1 for d in DOMAINS if row_tracks[d] is True)
        if layer_indeterminate:
            layer_result = "indeterminate"
        elif tracking_count >= 2:
            layer_result = "clear_crossover_advantage"
        else:
            layer_result = "no_clear_crossover"
        layer_results[f"layer-{layer:03d}"] = layer_result

        # descriptive oracle gaps (point estimates; need every row to have cells)
        gaps: dict[str, Any] = {}
        if all(cells_point[d] for d in DOMAINS):
            k_equal = {d: 0.25 for d in DOMAINS}
            gaps["row_equal_K"] = oracle_gap(cells_point, k_equal)
            gaps["k_tilde_weighted"] = {
                **oracle_gap(cells_point, K_TILDE),
                "assumption_note_mandatory": K_TILDE_NOTE,
            }
        else:
            gaps["note"] = (
                "at least one row has no valid start; oracle gaps are not computable "
                "for this layer and are reported as null (never imputed)"
            )
            gaps["row_equal_K"] = None
            gaps["k_tilde_weighted"] = None

        # descriptive migration context: displacement + frozen C_batch estimator
        migration = []
        base_manifest = manifests[f"cdx0-evaluation-{DOMAINS[0]}-layer-{layer:03d}"]
        p2l_by_column = {rec["layout_id"]: rec["p2l"] for rec in base_manifest["layouts"]}
        for a, b in itertools.combinations(COLUMNS, 2):
            moves = hamming(p2l_by_column[a], p2l_by_column[b])
            migration.append({
                "column_a": a, "column_b": b,
                "displaced_experts_moves_k": moves,
                "estimated_batch_migration_cost_us": ref_cost * moves,
                "estimator": (
                    "C_batch(k) = reference_cost_per_displaced_expert_us * moves_k; "
                    "reference frozen from the nearscan cost-benefit study "
                    "(multi_batch_combined_mean_us / moves_base, mean over its captures); "
                    "CROSS-CAPTURE EXTRAPOLATION - descriptive context only, never in verdicts"
                ),
            })

        layers_out[f"layer-{layer:03d}"] = {
            "rows": rows_out,
            "rows_tracking_own_base": row_tracks,
            "indeterminate_rows": indeterminate_rows,
            "tracking_row_count": tracking_count,
            "layer_result_exploratory": layer_result,
            "primary_contrasts_exploratory": primary_sorted,
            "oracle_gaps_descriptive": gaps,
            "migration_context_descriptive": migration,
        }

    clear_layers = [k for k, v in layer_results.items() if v == "clear_crossover_advantage"]
    indet_layers = [k for k, v in layer_results.items() if v == "indeterminate"]
    if clear_layers:
        overall = "detected_exploratory"
    elif not indet_layers:
        overall = "not_detected_exploratory"
    else:
        overall = "indeterminate_exploratory"

    output = {
        "schema_version": 1,
        "study_id": CDX0_STUDY_ID,
        "analysis": "cdx0_cross_domain_pilot_frozen_analysis",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analyzer_implementation": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "execution_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "statistics": {
            "t_quantile_95_by_df": {str(k): v for k, v in sorted(T_QUANTILE_95.items())},
            "df_rule": "valid_starts - 1 (df=3 with 4 valid starts; df=2 with 3)",
            "planned_starts_per_capture": PLANNED_STARTS,
            "min_valid_starts_per_row": MIN_VALID_STARTS,
            "blocks_per_start": BLOCKS_PER_START,
            "practical_equivalence_margin_us": eps,
            "paired_within_block": True,
            "overhead_subtracted_from_contrasts": False,
            "epsilon_in_verdicts": False,
            "contrast_orientation_note": (
                "primary contrasts oriented as T(other-base) - T(own-base): positive "
                "means own base faster; own_base_vs_identity oriented as T(own) - "
                "T(identity): negative means own base faster than identity"
            ),
            "primary_family_multiplicity_control": "none_claimed_exploratory_screening",
            "holm_secondary_within_layer_family_size": 12,
        },
        "verdict_rules": {
            "row_tracks_own_base": (
                "all three other-base-minus-own-base paired 95% CIs exclude zero "
                "positively (epsilon plays no role)"
            ),
            "clear_crossover_advantage_layer": ">=2 of 4 rows track own base",
            "layer_indeterminate": ">=2 rows with fewer than 3 valid starts",
            "detected": ">=1 of the 2 layers shows clear crossover advantage",
            "not_detected": "0 layers show it and no layer is indeterminate",
            "indeterminate": "any layer indeterminate and none clear",
            "exploratory_label": (
                "all 24 primary contrasts and every verdict derived from them are "
                "EXPLORATORY screening labels; no FWER control claimed; confirmatory "
                "use requires a new pre-registered protocol"
            ),
            "not_detected_meaning": (
                "the pre-registered screen did not fire on these two layers and "
                "candidates at replay caliber; NOT evidence of absence of exploitable "
                "cross-domain structure"
            ),
        },
        "overall_verdict_exploratory": overall,
        "layer_results_exploratory": layer_results,
        "layers": layers_out,
        "retired_starts": index.get("retired_starts", []),
        "interpretation_rules": [
            "an interval crossing zero means compatible with zero, not proof of no effect and not equivalence",
            "equivalence is claimed only when the interval lies wholly inside [-epsilon,+epsilon]",
            "negative effects are preserved exactly as measured",
            "no best-start, repeat, prefix or subset selection was applied",
            "failed starts are retired, reported, and never retried under this frozen campaign",
            "cdx0 numbers are layer-replay completion times, not server-side latencies",
            "results judge only the cross-domain execution-time response among the chosen captures, the two layers (024, 071) and the candidate layouts at replay caliber; NO full-service benefit or harm conversion in either direction",
            "own-base tracking is one pre-registered screening summary, NOT a necessary condition for cross-domain tracking value; the full matrix is reported regardless",
        ],
    }
    out_path = args.output.resolve()
    summary_path = args.summary.resolve()
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    write_json_atomic(out_path, output)

    lines = [
        "# CD-X0 cross-domain replay pilot - frozen analysis summary",
        "",
        f"- study: {CDX0_STUDY_ID}",
        f"- overall verdict (EXPLORATORY): **{overall}**",
        f"- layer results (EXPLORATORY): {json.dumps(layer_results)}",
        f"- valid starts: {index['counts']['starts_valid']}/{index['counts']['starts_expected']}"
        f" (retired: {index['counts']['starts_retired']})",
        f"- epsilon = {eps} us (labels only; NOT part of any verdict)",
        "",
        "Verdict scope: the chosen captures, layers 024/071 and the five candidate",
        "columns at replay caliber ONLY. No whole-service benefit or harm claim in",
        "either direction. 'not_detected' is not evidence of absence of structure.",
        "",
    ]
    for layer_key, layer_out in layers_out.items():
        lines.append(f"## {layer_key}: {layer_out['layer_result_exploratory']} (exploratory)")
        lines.append("")
        header = f"{'row':<10}" + "".join(f"{c.replace('base-of-',''):>20}" for c in COLUMNS)
        lines.append("```")
        lines.append(header)
        for domain in DOMAINS:
            row = layer_out["rows"][domain]
            line = f"{domain:<10}"
            for column in COLUMNS:
                cell = row["cells"][column]
                if cell["mean_us"] is None:
                    line += f"{'no_valid_start':>20}"
                elif "ci95_low_us" in cell:
                    half = (cell["ci95_high_us"] - cell["ci95_low_us"]) / 2
                    line += f"{cell['mean_us']:>13.2f}+-{half:>5.2f}"
                else:
                    line += f"{cell['mean_us']:>13.2f} (n<3)"
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append(f"rows tracking own base: {layer_out['rows_tracking_own_base']}")
        lines.append(f"indeterminate rows: {layer_out['indeterminate_rows']}")
        gaps = layer_out["oracle_gaps_descriptive"]
        if gaps.get("row_equal_K"):
            lines.append(
                f"oracle gap (row-equal K, descriptive): "
                f"{gaps['row_equal_K']['oracle_gap_pct_of_static']:.3f}% "
                f"({gaps['row_equal_K']['oracle_gap_us_per_forward']:.2f} us/forward)"
            )
        if gaps.get("k_tilde_weighted"):
            lines.append(
                f"oracle gap (K-tilde weighted, descriptive, assumption-flagged): "
                f"{gaps['k_tilde_weighted']['oracle_gap_pct_of_static']:.3f}% "
                f"({gaps['k_tilde_weighted']['oracle_gap_us_per_forward']:.2f} us/forward)"
            )
        lines.append("")
        lines.append("Primary contrasts (exploratory; + means own base faster):")
        lines.append("")
        for c in layer_out["primary_contrasts_exploratory"]:
            lines.append(
                f"- {c['row_domain']}: {c['contrast']} = {c['mean_us']:+.2f} us "
                f"[{c['ci95_low_us']:+.2f}, {c['ci95_high_us']:+.2f}] {c['verdict']}"
                + (" (practically_equivalent_within_margin)"
                   if c["practically_equivalent_within_margin"] else "")
                + f"; holm_adj_p={c['holm_adjusted_p_within_layer_descriptive']:.4f} (descriptive)"
            )
        lines.append("")
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"overall_verdict_exploratory={overall}")
    for layer_key, res in layer_results.items():
        print(f"{layer_key}: {res}")
    print(f"wrote {out_path} sha256={sha256_file(out_path)}")
    print(f"wrote {summary_path} sha256={sha256_file(summary_path)}")


if __name__ == "__main__":
    main()
