#!/usr/bin/env python3
"""Frozen analyzer for the low-r stopping study and migration-cost study.

The statistical design in configs/followup_v1/stopping_analysis_spec.json is
frozen BEFORE any stopping timing; this script refuses to run unless its own
file hash matches the spec and the spec's constants match the frozen protocols.
All signed effects (including negatives) are reported for every candidate; no
best-measured-prefix selection; zero crossing is compatible-with-zero, never a
deadzone; nothing is subtracted from an effect measured as a global completion
change; equivalence is supported only by intervals wholly inside [-eps,+eps].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any

from scipy import stats

from prediction_common import sha256_file, write_json_atomic

STOPPING_STUDY_ID = "h20-placement-stopping-v1"
BLOCKS_PER_START = 5
STARTS_PER_CAPTURE = 4
PAIRS_PER_TRANSACTION = 50
T_QUANTILE_95_DF3 = float(stats.t.ppf(0.975, STARTS_PER_CAPTURE - 1))
CANDIDATE_IDS = (
    "gain-01", "gain-02", "gain-03", "gain-04",
    "near-01", "near-02", "near-03", "near-04",
    "neg-01", "neg-02", "neg-03", "neg-04",
)
STRATA = ("gain", "near", "neg")


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--spec", type=pathlib.Path)
    parser.add_argument("--execution-index", type=pathlib.Path)
    parser.add_argument("--migration-index", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
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


def fieller_break_even(
    c: float, se_c: float, q: float, se_q: float
) -> dict[str, float] | None:
    """Ratio interval for N = C/Q from two independent means (df=3 each)."""
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


def verify_index(
    index_path: pathlib.Path,
    expected_status: str,
    expected_split: str,
    protocol_path: pathlib.Path,
    label: str,
) -> dict[str, Any]:
    index = load_json(index_path)
    if index.get("status") != expected_status:
        raise ValueError(f"{label}: index status differs: {index.get('status')}")
    if index.get("split") != expected_split:
        raise ValueError(f"{label}: index split differs")
    if index.get("study_id") != STOPPING_STUDY_ID:
        raise ValueError(f"{label}: index study differs")
    if sha256_file(protocol_path) != index["protocol"]["sha256"]:
        raise ValueError(f"{label}: frozen protocol hash differs from index")
    raws = {}
    for record in index["results"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{label}: raw file hash differs: {path}")
        raw = load_json(path)
        if raw.get("split") != expected_split:
            raise ValueError(f"{label}: raw split differs: {record['start_id']}")
        raws[record["start_id"]] = {"record": record, "raw": raw}
    return {"index": index, "raws": raws, "label": label}


def execution_effects(
    verified: dict[str, Any],
    protocol: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    eps: float,
) -> list[dict[str, Any]]:
    effects = []
    starts_by_capture: dict[str, list[str]] = {}
    for start in protocol["starts"]:
        starts_by_capture.setdefault(start["capture_id"], []).append(start["start_id"])
    for capture_id in sorted(starts_by_capture):
        start_ids = sorted(starts_by_capture[capture_id])
        if len(start_ids) != STARTS_PER_CAPTURE:
            raise ValueError(f"{capture_id}: start count differs")
        manifest = manifests[capture_id]
        layout_meta = {layout["layout_id"]: layout for layout in manifest["layouts"]}
        # per start, per block: completion table
        tables: list[dict[int, dict[str, float]]] = []
        for start_id in start_ids:
            entry = verified["raws"].get(start_id)
            if entry is None or entry["record"]["capture_id"] != capture_id:
                raise ValueError(f"missing execution start {start_id}")
            records = entry["raw"]["primary_timing_records"]
            if len(records) != len(CANDIDATE_IDS) * BLOCKS_PER_START + BLOCKS_PER_START:
                raise ValueError(f"{start_id}: primary cardinality differs")
            table: dict[int, dict[str, float]] = {}
            for record in records:
                if record["pass"] != "primary":
                    raise ValueError(f"{start_id}: unexpected pass {record['pass']}")
                table.setdefault(int(record["block"]), {})[record["layout_id"]] = float(
                    record["completion_fenced_us_per_forward"]
                )
            if len(table) != BLOCKS_PER_START:
                raise ValueError(f"{start_id}: block count differs")
            for block, completions in table.items():
                if set(completions) != set(CANDIDATE_IDS) | {"base"}:
                    raise ValueError(f"{start_id}: block {block} layout set differs")
            tables.append(table)
        for candidate_id in CANDIDATE_IDS:
            start_effects = []
            for table in tables:
                block_effects = [
                    table[block]["base"] - table[block][candidate_id]
                    for block in sorted(table)
                ]
                start_effects.append(sum(block_effects) / len(block_effects))
            mean, low, high = mean_interval(start_effects)
            meta = layout_meta[candidate_id]
            effects.append(
                {
                    "capture_id": capture_id,
                    "layout_id": candidate_id,
                    "stratum": meta["stratum"],
                    "stratum_group": candidate_id.split("-")[0],
                    "map_sha256": meta["map_sha256"],
                    "locked_m2_predicted_signed_gain_us": meta[
                        "locked_m2_predicted_signed_gain_us"
                    ],
                    "start_level_effects_us": start_effects,
                    **classify(mean, low, high, eps),
                }
            )
    if len(effects) != 144:
        raise ValueError(f"execution effect count differs: {len(effects)}")
    return effects


def migration_costs(
    verified: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    costs: dict[str, dict[str, dict[str, Any]]] = {}
    starts_by_capture: dict[str, list[str]] = {}
    for start in protocol["starts"]:
        starts_by_capture.setdefault(start["capture_id"], []).append(start["start_id"])
    for capture_id in sorted(starts_by_capture):
        start_ids = sorted(starts_by_capture[capture_id])
        if len(start_ids) != STARTS_PER_CAPTURE:
            raise ValueError(f"{capture_id}: migration start count differs")
        per_tx: dict[str, dict[str, list[float]]] = {}
        for start_id in start_ids:
            entry = verified["raws"].get(start_id)
            if entry is None or entry["record"]["capture_id"] != capture_id:
                raise ValueError(f"missing migration start {start_id}")
            transactions = entry["raw"]["transactions"]
            if len(transactions) != 13:
                raise ValueError(f"{start_id}: transaction count differs")
            for transaction in transactions:
                pairs = transaction["pairs"]
                if len(pairs) != PAIRS_PER_TRANSACTION:
                    raise ValueError(f"{start_id}: pair count differs")
                forward = [float(pair["forward_wall_us"]) for pair in pairs]
                reverse = [float(pair["reverse_wall_us"]) for pair in pairs]
                if any(pair.get("hashes_all_matched_canonical") is not True for pair in pairs):
                    raise ValueError(f"{start_id}: hash verification flag differs")
                bucket = per_tx.setdefault(
                    transaction["transaction_id"], {"forward": [], "reverse": [], "combined": []}
                )
                f_mean = sum(forward) / len(forward)
                r_mean = sum(reverse) / len(reverse)
                bucket["forward"].append(f_mean)
                bucket["reverse"].append(r_mean)
                bucket["combined"].append((f_mean + r_mean) / 2.0)
        summary = {}
        for transaction_id in sorted(per_tx):
            bucket = per_tx[transaction_id]
            if any(len(values) != STARTS_PER_CAPTURE for values in bucket.values()):
                raise ValueError(f"{capture_id}/{transaction_id}: start-level sample differs")
            summary[transaction_id] = {
                direction: dict(zip(
                    ("mean_us", "ci95_low_us", "ci95_high_us"), mean_interval(values)
                ))
                for direction, values in bucket.items()
            }
        if len(summary) != 13:
            raise ValueError(f"{capture_id}: transaction summary count differs")
        costs[capture_id] = summary
    return costs


def break_even(
    effects: list[dict[str, Any]], costs: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for effect in effects:
        capture_id = effect["capture_id"]
        transaction_id = f"swap-{effect['layout_id']}"
        combined = costs[capture_id][transaction_id]["combined"]
        c = combined["mean_us"]
        se_c = (combined["ci95_high_us"] - combined["ci95_low_us"]) / (2 * T_QUANTILE_95_DF3)
        q = effect["mean_us"]
        se_q = (effect["ci95_high_us"] - effect["ci95_low_us"]) / (2 * T_QUANTILE_95_DF3)
        row = {
            "capture_id": capture_id,
            "layout_id": effect["layout_id"],
            "transaction_id": transaction_id,
            "migration_cost_mean_us": c,
            "q_mean_us": q,
        }
        if effect["verdict"] == "excludes_zero_positive":
            interval = fieller_break_even(c, se_c, q, se_q)
            if interval is None:
                raise ValueError(f"Fieller interval unbounded despite positive CI: {row}")
            row.update(status="finite_break_even", **interval)
        elif effect["verdict"] == "compatible_with_zero":
            row.update(
                status="indeterminate_ci_crosses_zero",
                rule="a CI crossing zero is compatible with zero, not proof of no effect; no finite break-even is claimed",
            )
        else:
            row.update(
                status="no_finite_beneficial_breakeven",
                rule="Q <= 0 measured; migration can never pay for itself per forward",
            )
        rows.append(row)
    return rows


STRATUM_LABELS = {
    "gain": "four_largest_locked_m2_predicted_signed_gains",
    "near": "four_locked_m2_predictions_closest_to_zero_signs_retained",
    "neg": "four_most_negative_locked_m2_predictions",
}


def strata_summary(effects: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for stratum in STRATA:
        rows = [effect for effect in effects if effect["stratum_group"] == stratum]
        if any(effect["stratum"] != STRATUM_LABELS[stratum] for effect in rows):
            raise ValueError(f"stratum label differs from frozen manifest wording: {stratum}")
        means = [effect["mean_us"] for effect in rows]
        summary[stratum] = {
            "count": len(rows),
            "mean_observed_effect_us": sum(means) / len(means),
            "mean_predicted_gain_us": sum(
                effect["locked_m2_predicted_signed_gain_us"] for effect in rows
            )
            / len(rows),
            "excludes_zero_positive": sum(
                effect["verdict"] == "excludes_zero_positive" for effect in rows
            ),
            "excludes_zero_negative": sum(
                effect["verdict"] == "excludes_zero_negative" for effect in rows
            ),
            "compatible_with_zero": sum(
                effect["verdict"] == "compatible_with_zero" for effect in rows
            ),
            "practically_equivalent_within_margin": sum(
                effect["practically_equivalent_within_margin"] for effect in rows
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    spec_path = (
        args.spec or root / "configs/followup_v1/stopping_analysis_spec.json"
    ).resolve()
    execution_index_path = (
        args.execution_index or root / "results/stopping_v1/execution/index.json"
    ).resolve()
    migration_index_path = (
        args.migration_index or root / "results/stopping_v1/migration/index.json"
    ).resolve()
    output_path = (args.output or root / "results/stopping_v1/analysis.json").resolve()

    spec = load_json(spec_path)
    if sha256_file(pathlib.Path(__file__).resolve()) != spec["analyzer_implementation"]["sha256"]:
        raise ValueError("analyzer file differs from the frozen analysis spec")
    if "original_spec" in spec:
        original_path = pathlib.Path(spec["original_spec"]["path"])
        if sha256_file(original_path) != spec["original_spec"]["sha256"]:
            raise ValueError("original analysis spec differs from amendment record")
        original = load_json(original_path)
        if original.get("status") != "frozen_before_any_stopping_timing":
            raise ValueError("original spec was not frozen before timing")
        if spec["statistics"] != original["statistics"]:
            raise ValueError("amendment changed the frozen statistics")
        if spec["interpretation_rules"] != original["interpretation_rules"]:
            raise ValueError("amendment changed the frozen interpretation rules")
    if spec.get("study_id") != STOPPING_STUDY_ID:
        raise ValueError("analysis spec study differs")
    if spec["statistics"]["blocks_per_start"] != BLOCKS_PER_START:
        raise ValueError("spec block count differs")
    if spec["statistics"]["starts_per_capture"] != STARTS_PER_CAPTURE:
        raise ValueError("spec start count differs")
    if spec["statistics"]["pairs_per_transaction"] != PAIRS_PER_TRANSACTION:
        raise ValueError("spec pair count differs")
    if abs(spec["statistics"]["t_quantile_95_df3"] - T_QUANTILE_95_DF3) > 1e-12:
        raise ValueError("spec t quantile differs")

    execution_protocol_path = root / "configs/followup_v1/stopping_execution_protocol.json"
    migration_protocol_path = root / "configs/followup_v1/stopping_migration_protocol.json"
    execution_protocol = load_json(execution_protocol_path)
    migration_protocol = load_json(migration_protocol_path)
    eps = float(execution_protocol["design"]["practical_equivalence_margin_us"])
    if abs(eps - float(spec["statistics"]["practical_equivalence_margin_us"])) > 1e-9:
        raise ValueError("spec margin differs from frozen protocol")
    if execution_protocol["timing"]["blocks_per_start"] != BLOCKS_PER_START:
        raise ValueError("protocol block count differs")

    layout_index = load_json(root / "inputs/stopping_v1/layouts/index.json")
    if layout_index.get("status") != "all_stopping_layouts_frozen_before_timing":
        raise ValueError("layout index status differs")
    manifests = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {record['capture_id']}")
        manifests[record["capture_id"]] = load_json(path)
    if len(manifests) != 12:
        raise ValueError("manifest count differs")

    verified_execution = verify_index(
        execution_index_path,
        "all_stopping_timing_results_validated_and_hash_indexed",
        "stopping",
        execution_protocol_path,
        "execution",
    )
    verified_migration = verify_index(
        migration_index_path,
        "all_stopping_migration_timing_results_validated_and_hash_indexed",
        "stopping_migration",
        migration_protocol_path,
        "migration",
    )

    effects = execution_effects(verified_execution, execution_protocol, manifests, eps)
    costs = migration_costs(verified_migration, migration_protocol)
    rows = break_even(effects, costs)
    strata = strata_summary(effects)

    prediction_residuals = [
        effect["mean_us"] - effect["locked_m2_predicted_signed_gain_us"] for effect in effects
    ]
    signed_bias = sum(prediction_residuals) / len(prediction_residuals)
    mae = sum(abs(value) for value in prediction_residuals) / len(prediction_residuals)
    tau = stats.kendalltau(
        [effect["locked_m2_predicted_signed_gain_us"] for effect in effects],
        [effect["mean_us"] for effect in effects],
    )

    payload = {
        "schema_version": 1,
        "study_id": STOPPING_STUDY_ID,
        "analysis": "low_r_stopping_and_migration_break_even",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "analyzer_implementation": spec["analyzer_implementation"],
        "inputs": {
            "execution_index": {"path": str(execution_index_path), "sha256": sha256_file(execution_index_path)},
            "migration_index": {"path": str(migration_index_path), "sha256": sha256_file(migration_index_path)},
            "execution_protocol": {"path": str(execution_protocol_path), "sha256": sha256_file(execution_protocol_path)},
            "migration_protocol": {"path": str(migration_protocol_path), "sha256": sha256_file(migration_protocol_path)},
            "layout_index_sha256": sha256_file(root / "inputs/stopping_v1/layouts/index.json"),
        },
        "statistics": spec["statistics"],
        "interpretation_rules": spec["interpretation_rules"],
        "execution_effects": effects,
        "strata_summary": strata,
        "migration_costs": costs,
        "break_even": rows,
        "locked_m2_prediction_diagnostic_descriptive": {
            "note": "descriptive only; no refit; predictions were locked before any stopping timing",
            "signed_bias_us": signed_bias,
            "mae_us": mae,
            "kendall_tau": float(tau.statistic),
            "kendall_tau_p": float(tau.pvalue),
        },
        "counts": {
            "execution_effects": len(effects),
            "captures": len(costs),
            "transactions_per_capture": 13,
            "break_even_rows": len(rows),
        },
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output_path}")
    write_json_atomic(output_path, payload)
    positive = strata["gain"]["excludes_zero_positive"]
    print(f"wrote {output_path}")
    print(
        f"verdicts: gain stratum {positive}/48 positive-excluding-zero; "
        f"compatible_with_zero total {sum(s['compatible_with_zero'] for s in strata.values())}/144; "
        f"negative total {sum(s['excludes_zero_negative'] for s in strata.values())}/144"
    )
    print(f"locked-M2 residual diagnostic: bias {signed_bias:+.1f} us, MAE {mae:.1f} us, tau {float(tau.statistic):.3f}")


if __name__ == "__main__":
    main()
