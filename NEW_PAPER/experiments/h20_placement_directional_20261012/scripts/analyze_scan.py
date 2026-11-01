#!/usr/bin/env python3
"""Frozen analyzer for the near-equilibrium real-layout scan (deadzone study).

The statistical design in configs/followup_v1/scan_analysis_spec.json is frozen
BEFORE any scan timing; this script refuses to run unless its own file hash
matches the spec and the spec's constants match the frozen scan protocol. All
signed contrasts (including negatives) are reported for every adjacent layout
pair and every identity-cumulative pair; no best-measured-prefix or subset
selection; an interval crossing zero is compatible-with-zero and is never by
itself a plateau declaration; equivalence is supported only by intervals wholly
inside [-eps,+eps]; a supported plateau region additionally requires raw r to
strictly decrease across every contrast in the band; nothing is subtracted from
an effect measured as a global completion change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any

from scipy import stats

from prediction_common import sha256_file, write_json_atomic

SCAN_STUDY_ID = "h20-placement-nearscan-v1"
BLOCKS_PER_START = 5
STARTS_PER_CAPTURE = 4
T_QUANTILE_95_DF3 = float(stats.t.ppf(0.975, STARTS_PER_CAPTURE - 1))
SCAN_BIN_ORDER = (
    "bin-r-gt-1.80",
    "bin-r-1.40-1.80",
    "bin-r-1.20-1.40",
    "bin-r-1.10-1.20",
    "bin-r-1.05-1.10",
    "bin-r-1.00-1.05",
)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--spec", type=pathlib.Path)
    parser.add_argument("--execution-index", type=pathlib.Path)
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


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def region_label(layout_record: dict[str, Any]) -> str:
    """Mechanical label extraction (amendment 1): the frozen manifests store the
    bin as a record {bin_id, lo_exclusive, hi_inclusive}; region grouping keys
    must be the bin_id string (or the anchor label), never the whole record."""
    bin_field = layout_record.get("bin")
    if isinstance(bin_field, dict):
        return str(bin_field["bin_id"])
    if layout_record["layout_id"] == "identity":
        return "identity_anchor"
    if layout_record["layout_id"] == "base":
        return "base_anchor"
    raise ValueError(f"layout without bin label: {layout_record['layout_id']}")


def verify_index(
    index_path: pathlib.Path,
    protocol_path: pathlib.Path,
) -> dict[str, Any]:
    index = load_json(index_path)
    if index.get("status") != "all_nearscan_timing_results_validated_and_hash_indexed":
        raise ValueError(f"index status differs: {index.get('status')}")
    if index.get("split") != "nearscan":
        raise ValueError("index split differs")
    if index.get("study_id") != SCAN_STUDY_ID:
        raise ValueError("index study differs")
    if sha256_file(protocol_path) != index["protocol"]["sha256"]:
        raise ValueError("frozen protocol hash differs from index")
    raws = {}
    for record in index["results"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"raw file hash differs: {path}")
        raw = load_json(path)
        if raw.get("split") != "nearscan":
            raise ValueError(f"raw split differs: {record['start_id']}")
        raws[record["start_id"]] = {"record": record, "raw": raw}
    return {"index": index, "raws": raws}


def start_tables(
    verified: dict[str, Any],
    protocol: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, list[dict[int, dict[str, float]]]]:
    starts_by_capture: dict[str, list[str]] = {}
    for start in protocol["starts"]:
        starts_by_capture.setdefault(start["capture_id"], []).append(start["start_id"])
    tables_by_capture: dict[str, list[dict[int, dict[str, float]]]] = {}
    for capture_id in sorted(starts_by_capture):
        start_ids = sorted(starts_by_capture[capture_id])
        if len(start_ids) != STARTS_PER_CAPTURE:
            raise ValueError(f"{capture_id}: start count differs")
        order = [layout["layout_id"] for layout in manifests[capture_id]["layouts"]]
        tables: list[dict[int, dict[str, float]]] = []
        for start_id in start_ids:
            entry = verified["raws"].get(start_id)
            if entry is None or entry["record"]["capture_id"] != capture_id:
                raise ValueError(f"missing scan start {start_id}")
            records = entry["raw"]["primary_timing_records"]
            if len(records) != len(order) * BLOCKS_PER_START:
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
                if set(completions) != set(order):
                    raise ValueError(f"{start_id}: block {block} layout set differs")
            tables.append(table)
        tables_by_capture[capture_id] = tables
    return tables_by_capture


def paired_effect(
    tables: list[dict[int, dict[str, float]]], a: str, b: str
) -> list[float]:
    """Start-level mean of within-block paired differences T(a) - T(b)."""
    start_effects = []
    for table in tables:
        block_effects = [table[blk][a] - table[blk][b] for blk in sorted(table)]
        start_effects.append(sum(block_effects) / len(block_effects))
    return start_effects


def capture_analysis(
    capture_id: str,
    manifest: dict[str, Any],
    tables: list[dict[int, dict[str, float]]],
    eps: float,
) -> dict[str, Any]:
    layouts = manifest["layouts"]
    meta = {layout["layout_id"]: layout for layout in layouts}
    order = [layout["layout_id"] for layout in layouts]
    r_values = [float(meta[layout_id]["r_raw"]) for layout_id in order]
    if any(later > former for former, later in zip(r_values, r_values[1:])):
        raise ValueError(f"{capture_id}: manifest r_raw not non-increasing")
    if order[0] != "identity" or order[-1] != "base":
        raise ValueError(f"{capture_id}: anchor order differs")

    layout_times = []
    for layout_id in order:
        start_means = [
            sum(table[blk][layout_id] for blk in sorted(table)) / len(table)
            for table in tables
        ]
        mean, low, high = mean_interval(start_means)
        layout_times.append(
            {
                "capture_id": capture_id,
                "layout_id": layout_id,
                "role": meta[layout_id]["role"],
                "bin": region_label(meta[layout_id]),
                "step_index": meta[layout_id]["step_index"],
                "r_raw": float(meta[layout_id]["r_raw"]),
                "padded_max_over_mean": meta[layout_id]["padded_max_over_mean"],
                "map_sha256": meta[layout_id]["map_sha256"],
                "start_level_mean_completion_us_per_forward": start_means,
                "mean_us": mean,
                "ci95_low_us": low,
                "ci95_high_us": high,
            }
        )

    adjacent = []
    for a, b in zip(order, order[1:]):
        start_effects = paired_effect(tables, a, b)
        mean, low, high = mean_interval(start_effects)
        delta_r = float(meta[a]["r_raw"]) - float(meta[b]["r_raw"])
        adjacent.append(
            {
                "capture_id": capture_id,
                "contrast": f"{a}->{b}",
                "layout_a": a,
                "layout_b": b,
                "entering_region": region_label(meta[b]),
                "r_a": float(meta[a]["r_raw"]),
                "r_b": float(meta[b]["r_raw"]),
                "delta_r": delta_r,
                "strict_r_decrease": delta_r > 0.0,
                "start_level_effects_us": start_effects,
                **classify(mean, low, high, eps),
            }
        )

    cumulative = []
    for layout_id in order[1:]:
        start_effects = paired_effect(tables, "identity", layout_id)
        mean, low, high = mean_interval(start_effects)
        cumulative.append(
            {
                "capture_id": capture_id,
                "contrast": f"identity->{layout_id}",
                "layout_b": layout_id,
                "target_region": region_label(meta[layout_id]),
                "r_identity": float(meta["identity"]["r_raw"]),
                "r_b": float(meta[layout_id]["r_raw"]),
                "delta_r": float(meta["identity"]["r_raw"]) - float(meta[layout_id]["r_raw"]),
                "start_level_effects_us": start_effects,
                **classify(mean, low, high, eps),
            }
        )

    bands = []
    current: list[dict[str, Any]] = []
    for contrast in adjacent:
        qualifies = (
            contrast["strict_r_decrease"]
            and contrast["verdict"] == "compatible_with_zero"
            and contrast["practically_equivalent_within_margin"]
        )
        if qualifies:
            current.append(contrast)
        elif current:
            bands.append(current)
            current = []
    if current:
        bands.append(current)
    plateau_bands = [
        {
            "capture_id": capture_id,
            "contrasts": [contrast["contrast"] for contrast in band],
            "r_from": band[0]["r_a"],
            "r_to": band[-1]["r_b"],
            "adjacent_contrast_count": len(band),
            "kind": (
                "supported_plateau_region"
                if len(band) >= 2
                else "single_step_equivalence_not_a_region"
            ),
        }
        for band in bands
    ]
    return {
        "layout_times": layout_times,
        "adjacent_contrasts": adjacent,
        "identity_cumulative_contrasts": cumulative,
        "plateau_bands": plateau_bands,
    }


def pooled_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    regions = [region for region in SCAN_BIN_ORDER] + ["base_anchor"]
    for region in regions:
        group = [row for row in rows if row[key] == region]
        if not group:
            pooled[region] = {"contrast_count": 0}
            continue
        means = [row["mean_us"] for row in group]
        pooled[region] = {
            "contrast_count": len(group),
            "mean_observed_effect_us": sum(means) / len(means),
            "strict_r_decrease_count": sum(row["strict_r_decrease"] for row in group)
            if "strict_r_decrease" in group[0]
            else None,
            "excludes_zero_positive": sum(
                row["verdict"] == "excludes_zero_positive" for row in group
            ),
            "excludes_zero_negative": sum(
                row["verdict"] == "excludes_zero_negative" for row in group
            ),
            "compatible_with_zero": sum(
                row["verdict"] == "compatible_with_zero" for row in group
            ),
            "practically_equivalent_within_margin": sum(
                row["practically_equivalent_within_margin"] for row in group
            ),
        }
    return pooled


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    spec_path = (
        args.spec or root / "configs/followup_v1/scan_analysis_spec.json"
    ).resolve()
    execution_index_path = (
        args.execution_index or root / "results/nearscan_v1/execution/index.json"
    ).resolve()
    output_path = (args.output or root / "results/nearscan_v1/analysis.json").resolve()

    spec = load_json(spec_path)
    if sha256_file(pathlib.Path(__file__).resolve()) != spec["analyzer_implementation"]["sha256"]:
        raise ValueError("analyzer file differs from the frozen analysis spec")
    if "original_spec" in spec:
        original_path = pathlib.Path(spec["original_spec"]["path"])
        if sha256_file(original_path) != spec["original_spec"]["sha256"]:
            raise ValueError("original analysis spec differs from amendment record")
        original = load_json(original_path)
        if original.get("status") != "frozen_before_any_scan_timing":
            raise ValueError("original spec was not frozen before timing")
        if spec["statistics"] != original["statistics"]:
            raise ValueError("amendment changed the frozen statistics")
        if spec["interpretation_rules"] != original["interpretation_rules"]:
            raise ValueError("amendment changed the frozen interpretation rules")
    if spec.get("study_id") != SCAN_STUDY_ID:
        raise ValueError("analysis spec study differs")
    if spec["statistics"]["blocks_per_start"] != BLOCKS_PER_START:
        raise ValueError("spec block count differs")
    if spec["statistics"]["starts_per_capture"] != STARTS_PER_CAPTURE:
        raise ValueError("spec start count differs")
    if abs(spec["statistics"]["t_quantile_95_df3"] - T_QUANTILE_95_DF3) > 1e-12:
        raise ValueError("spec t quantile differs")

    execution_protocol_path = root / "configs/followup_v1/scan_execution_protocol.json"
    execution_protocol = load_json(execution_protocol_path)
    eps = float(execution_protocol["design"]["practical_equivalence_margin_us"])
    if abs(eps - float(spec["statistics"]["practical_equivalence_margin_us"])) > 1e-9:
        raise ValueError("spec margin differs from frozen protocol")
    if execution_protocol["timing"]["blocks_per_start"] != BLOCKS_PER_START:
        raise ValueError("protocol block count differs")
    if execution_protocol.get("study_id") != SCAN_STUDY_ID:
        raise ValueError("execution protocol belongs to another study")

    layout_index = load_json(root / "inputs/nearscan_v1/layouts/index.json")
    if layout_index.get("status") != "all_scan_layouts_frozen_before_timing":
        raise ValueError("layout index status differs")
    if abs(float(layout_index["design"]["practical_equivalence_margin_us"]) - eps) > 1e-9:
        raise ValueError("layout index margin differs from protocol")
    if layout_index["design"]["deadzone_rule"] != spec["statistics"]["deadzone_rule"]:
        raise ValueError("layout index deadzone rule differs from frozen spec")
    manifests = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {record['capture_id']}")
        manifests[record["capture_id"]] = load_json(path)
    if len(manifests) != 12:
        raise ValueError("manifest count differs")

    verified = verify_index(execution_index_path, execution_protocol_path)
    tables_by_capture = start_tables(verified, execution_protocol, manifests)

    layout_times: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    plateau_bands: list[dict[str, Any]] = []
    for capture_id in sorted(tables_by_capture):
        result = capture_analysis(
            capture_id, manifests[capture_id], tables_by_capture[capture_id], eps
        )
        layout_times.extend(result["layout_times"])
        adjacent.extend(result["adjacent_contrasts"])
        cumulative.extend(result["identity_cumulative_contrasts"])
        plateau_bands.extend(result["plateau_bands"])

    expected_counts = spec["statistics"]["expected_counts"]
    if len(layout_times) != int(expected_counts["layout_time_rows"]):
        raise ValueError(f"layout time row count differs: {len(layout_times)}")
    if len(adjacent) != int(expected_counts["adjacent_contrasts"]):
        raise ValueError(f"adjacent contrast count differs: {len(adjacent)}")
    if len(cumulative) != int(expected_counts["identity_cumulative_contrasts"]):
        raise ValueError(f"cumulative contrast count differs: {len(cumulative)}")
    if len(tables_by_capture) != int(expected_counts["captures"]):
        raise ValueError("capture count differs")

    deadzone_summary = {
        "adjacent_contrasts": len(adjacent),
        "strictly_r_decreasing": sum(row["strict_r_decrease"] for row in adjacent),
        "practically_equivalent_within_margin": sum(
            row["practically_equivalent_within_margin"] for row in adjacent
        ),
        "compatible_with_zero": sum(
            row["verdict"] == "compatible_with_zero" for row in adjacent
        ),
        "compatible_with_zero_but_not_equivalent": sum(
            row["verdict"] == "compatible_with_zero"
            and not row["practically_equivalent_within_margin"]
            for row in adjacent
        ),
        "excludes_zero_positive": sum(
            row["verdict"] == "excludes_zero_positive" for row in adjacent
        ),
        "excludes_zero_negative": sum(
            row["verdict"] == "excludes_zero_negative" for row in adjacent
        ),
        "supported_plateau_regions": sum(
            band["kind"] == "supported_plateau_region" for band in plateau_bands
        ),
        "single_step_equivalences_not_regions": sum(
            band["kind"] == "single_step_equivalence_not_a_region" for band in plateau_bands
        ),
        "regional_plateau_verdict": (
            "plateau_supported_in_listed_bands"
            if any(band["kind"] == "supported_plateau_region" for band in plateau_bands)
            else "no_regional_plateau_supported"
        ),
    }

    payload = {
        "schema_version": 1,
        "study_id": SCAN_STUDY_ID,
        "analysis": "near_equilibrium_real_layout_scan_deadzone",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "analyzer_implementation": spec["analyzer_implementation"],
        "inputs": {
            "execution_index": {
                "path": str(execution_index_path),
                "sha256": sha256_file(execution_index_path),
            },
            "execution_protocol": {
                "path": str(execution_protocol_path),
                "sha256": sha256_file(execution_protocol_path),
            },
            "layout_index_sha256": sha256_file(root / "inputs/nearscan_v1/layouts/index.json"),
        },
        "statistics": spec["statistics"],
        "interpretation_rules": spec["interpretation_rules"],
        "layout_times": layout_times,
        "adjacent_contrasts": adjacent,
        "identity_cumulative_contrasts": cumulative,
        "plateau_bands": plateau_bands,
        "pooled_adjacent_by_entering_region": pooled_summary(adjacent, "entering_region"),
        "pooled_cumulative_by_target_region": pooled_summary(cumulative, "target_region"),
        "deadzone_summary": deadzone_summary,
        "counts": {
            "captures": len(tables_by_capture),
            "layout_time_rows": len(layout_times),
            "adjacent_contrasts": len(adjacent),
            "identity_cumulative_contrasts": len(cumulative),
            "plateau_bands": len(plateau_bands),
        },
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output_path}")
    write_json_atomic(output_path, payload)
    print(f"wrote {output_path}")
    print(
        f"adjacent contrasts: {deadzone_summary['excludes_zero_positive']} positive / "
        f"{deadzone_summary['excludes_zero_negative']} negative / "
        f"{deadzone_summary['compatible_with_zero']} compatible-with-zero "
        f"({deadzone_summary['practically_equivalent_within_margin']} within margin); "
        f"plateau bands: {deadzone_summary['supported_plateau_regions']} regional, "
        f"{deadzone_summary['single_step_equivalences_not_regions']} single-step; "
        f"verdict: {deadzone_summary['regional_plateau_verdict']}"
    )


if __name__ == "__main__":
    main()
