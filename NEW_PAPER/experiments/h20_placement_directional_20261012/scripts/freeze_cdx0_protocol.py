#!/usr/bin/env python3
"""Freeze (or verify) the CD-X0 cross-domain replay execution protocol.

Study h20-placement-cdx0-v1. The protocol is frozen BEFORE any CD-X0 timing
(after captures, route-metadata extraction, layout generation and the
pre-freeze smoke replays). It binds: the approved pilot protocol and its
governing amendment v1, the tolerance-v2 two-tier correctness gate (inherited
verbatim, never relaxed), the frozen CD-X0 layout index (8 manifests, five
fixed columns each, base maps built ONLY from calibration-capture metadata by
the frozen greedy lineage), the capture index, the frozen migration-cost
reference constant (descriptive context only), and hash records for every
implementation file the driver, runner, indexer and analyzer will use.

Schedules are the pre-registered deterministic rotation of the frozen
five-column order by (start_index + block_index) mod 5 - balanced by
construction, no randomness, never derived from measurements. --verify-only
regenerates all deterministic content and compares without writing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cdx0_common import (  # noqa: E402
    STUDY_ID as CDX0_STUDY_ID,
    NUM_EXPERTS,
    RouteMetadata,
    layout_descriptors,
    placement_map_sha256,
    sha256_file,
    write_json_atomic,
)
from generate_stopping_layouts import greedy_base  # noqa: E402

PROTOCOL_ID = "h20-placement-cdx0-execution-v1"
GENERATOR_VERSION = "cdx0-execution-protocol-v1"
STATUS = "frozen_before_cdx0_timing"
LAYERS = (24, 71)
DOMAINS = ("sharegpt", "prover", "book", "code")
COLUMNS = ("identity",) + tuple(f"base-of-{d}" for d in DOMAINS)
STARTS_PER_CAPTURE = 4
BLOCKS_PER_START = 5
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
T_QUANTILE_95_DF3 = 3.1824463052837078
T_QUANTILE_95_DF2 = 4.302652729911275
K_TILDE = {"sharegpt": 0.157, "prover": 0.12, "book": 0.2372, "code": 0.4857}
K_TILDE_NOTE = (
    "K_tilde = evaluation token shares, an EXOGENOUS approximation of each "
    "domain's service relevance; CD-X0 rows are NOT equal-token (book ~2x, "
    "code ~4x the sharegpt/prover row sizes), so K_tilde weighting is an "
    "assumption-flagged approximation, not a replay-measured quantity; no "
    "service-share multiplication may be applied to these numbers and no "
    "full-service benefit or harm may be inferred in either direction"
)
AMENDMENT_ADDED_PROHIBITIONS = [
    "modify_prediction_common_or_any_frozen_prior_artifact",
    "relabel_exploratory_as_confirmatory",
    "capture_retry_beyond_one_r2_suffix",
    "band_or_rule_change_after_batch_freeze",
]
IMPLEMENTATION_FILES = {
    "cdx0_common_record": "cdx0_common.py",
    "replay_placement_record": "replay_placement.py",
    "cdx0_replay_record": "cdx0_replay.py",
    "index_cdx0_results_record": "index_cdx0_results.py",
    "analyze_cdx0_record": "analyze_cdx0.py",
    "generate_cdx0_layouts_record": "generate_cdx0_layouts.py",
    "extract_cdx0_route_metadata_record": "extract_cdx0_route_metadata.py",
    "generate_stopping_layouts_record": "generate_stopping_layouts.py",
    "run_cdx0_execution_record": "run_cdx0_execution.sh",
}
CDX0_SPEC: dict[str, Any] = {
    "rows_per_layer": 4,
    "columns_per_layer": list(COLUMNS),
    "timed_cells_total": 40,
    "starts_per_capture": STARTS_PER_CAPTURE,
    "replay_start_count_total": 32,
    "blocks_per_start": BLOCKS_PER_START,
    "initial_warmup": 10,
    "activation_warmup": 10,
    "timed_forwards": 50,
    "timing_endpoint": "completion_fenced_us_per_forward",
    "within_block_column_order": (
        "deterministic rotation of the frozen five-column order by "
        "(start_index + block_index) mod 5; start_index is the 0-based index "
        "of the start in this protocol's starts list"
    ),
    "identity_role": "timed_anchor_and_tier1_correctness_reference",
    "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def derived_seed(generator_version: str, *parts: str) -> int:
    text = "|".join((CDX0_STUDY_ID, generator_version, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def rotation_schedule(start_index: int, block: int) -> list[str]:
    k = (start_index + block) % len(COLUMNS)
    return list(COLUMNS[k:] + COLUMNS[:k])


def rebuild_metadata(record: dict[str, Any], *, batch_id: str, split: str,
                     layer_id: int, capture_id: str) -> RouteMetadata:
    return RouteMetadata(
        batch_id=batch_id,
        split=split,
        layer_id=layer_id,
        capture_id=capture_id,
        hidden_rows_by_source_rank=tuple(record["hidden_rows_by_source_rank"]),
        source_expert_counts=tuple(tuple(row) for row in record["source_expert_counts"]),
        expert_counts=tuple(record["expert_counts"]),
        expert_blocks=tuple(record["expert_blocks_128"]),
        total_hidden_rows=record["total_hidden_rows"],
        total_assignments=record["total_assignments"],
    )


def migration_reference(root: pathlib.Path) -> dict[str, Any]:
    """Frozen descriptive migration-cost reference from the nearscan study.

    reference_cost_per_displaced_expert_us = mean over the frozen nearscan
    cost-benefit captures of multi_batch_combined_mean_us / moves_base.
    Descriptive context only (CROSS-CAPTURE EXTRAPOLATION), never in verdicts.
    """
    cost_path = root / "results/nearscan_v1/cost_benefit.json"
    cost = json.loads(cost_path.read_text())
    inputs = cost["cost_model_inputs_by_capture"]
    ratios = {
        capture: float(values["multi_batch_combined_mean_us"]) / int(values["moves_base"])
        for capture, values in sorted(inputs.items())
        if int(values["moves_base"]) > 0
    }
    if not ratios:
        raise ValueError("nearscan cost-benefit study provides no usable ratios")
    return {
        "source_file": file_record(cost_path),
        "per_capture_ratio_us": ratios,
        "reference_cost_per_displaced_expert_us": sum(ratios.values()) / len(ratios),
        "estimator": (
            "C_batch(k) = reference_cost_per_displaced_expert_us * moves_k; "
            "CROSS-CAPTURE EXTRAPOLATION - descriptive context only, never in verdicts"
        ),
    }


def validate_inputs(root: pathlib.Path) -> dict[str, Any]:
    pilot_path = root / "configs/followup_v1/cdx0_pilot_protocol.json"
    pilot = json.loads(pilot_path.read_text())
    amend_path = root / "configs/followup_v1/cdx0_amendment_v1.json"
    amendment = json.loads(amend_path.read_text())
    if amendment.get("status") != "FROZEN_BEFORE_ANY_CDX0_TIMING":
        raise ValueError("cdx0 amendment is not frozen before timing")
    if sha256_file(pilot_path) != amendment["amends"]["sha256"]:
        raise ValueError("amendment does not reference the pilot protocol on disk")

    tolerance_path = root / "configs/followup_v1/prediction_tolerance_v2.json"
    tolerance = json.loads(tolerance_path.read_text())
    if tolerance.get("status") != "frozen_before_evaluation_attempt02_timing":
        raise ValueError("tolerance v2 status differs")
    gate = tolerance["correctness_gate"]
    if (float(gate["identity_versus_captured"]["atol"]) != 2.0**-8
            or float(gate["identity_versus_captured"]["rtol"]) != 0.0):
        raise ValueError("tier-1 strictness must equal the pilot tolerance")

    capture_index_path = root / "inputs/cdx0_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    if capture_index.get("study_id") != CDX0_STUDY_ID:
        raise ValueError("capture index belongs to another study")
    if capture_index.get("status") != "all_cdx0_capture_inputs_validated_and_hash_indexed":
        raise ValueError(f"capture index status differs: {capture_index.get('status')}")

    meta_dir = root / "inputs/cdx0_v1/route_metadata"
    meta_docs: dict[str, Any] = {}
    for path in sorted(meta_dir.glob("cdx0-*.json")):
        doc = json.loads(path.read_text())
        if doc.get("study_id") != CDX0_STUDY_ID:
            raise ValueError(f"route metadata study differs: {path}")
        meta_docs[doc["capture_id"]] = doc
    if len(meta_docs) != 16:
        raise ValueError(f"expected 16 route-metadata files, found {len(meta_docs)}")

    layout_index_path = root / "inputs/cdx0_v1/layouts/index.json"
    layout_index = json.loads(layout_index_path.read_text())
    if layout_index.get("study_id") != CDX0_STUDY_ID:
        raise ValueError("layout index belongs to another study")
    if layout_index.get("status") != "generated_before_protocol_freeze":
        raise ValueError("layout index status differs")
    if layout_index.get("selection_uses_performance_timing") is not False:
        raise ValueError("layout selection discipline differs")
    if sha256_file(capture_index_path) != layout_index["capture_index"]["sha256"]:
        raise ValueError("layout index references another capture index")
    if tuple(layout_index.get("columns", ())) != COLUMNS:
        raise ValueError("layout index columns differ from frozen design")

    identity_sha = placement_map_sha256(tuple(range(NUM_EXPERTS)))

    # deep check 1: recompute every greedy base from calibration metadata
    greedy_by_layer: dict[int, dict[str, Any]] = {}
    for layer in LAYERS:
        per_domain = {}
        for domain in DOMAINS:
            calib_id = f"cdx0-calibration-{domain}-layer-{layer:03d}"
            calib_doc = meta_docs[calib_id]
            calib_meta = rebuild_metadata(
                calib_doc["route_metadata"], batch_id=calib_doc["batch_id"],
                split="calibration", layer_id=layer, capture_id=calib_id,
            )
            result = greedy_base(calib_meta)
            per_domain[domain] = {
                "p2l": [int(v) for v in result["p2l"]],
                "map_sha256": placement_map_sha256(tuple(result["p2l"])),
                "step_count": int(result["step_count"]),
            }
        greedy_by_layer[layer] = per_domain

    manifests: dict[str, dict[str, Any]] = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"layout manifest hash differs: {record['capture_id']}")
        manifest = json.loads(path.read_text())
        capture_id = record["capture_id"]
        layer = int(manifest["layer_id"])
        domain = manifest["domain"]
        expected_header = {
            "schema_version": 1,
            "study_id": CDX0_STUDY_ID,
            "split": "cdx0",
            "capture_id": capture_id,
            "batch_id": f"cdx0-evaluation-{domain}",
            "domain": domain,
            "layer_id": layer,
            "selection_uses_performance_timing": False,
        }
        mismatches = {k: (manifest.get(k), v) for k, v in expected_header.items()
                      if manifest.get(k) != v}
        if mismatches:
            raise ValueError(f"manifest header mismatch: {capture_id}: {mismatches}")
        if (layer, domain) not in {(l, d) for l in LAYERS for d in DOMAINS}:
            raise ValueError(f"manifest outside frozen rows: {capture_id}")
        eval_doc = meta_docs[capture_id]
        if manifest["route_metadata"] != eval_doc["route_metadata"]:
            raise ValueError(f"manifest route metadata differs from extraction: {capture_id}")
        if tuple(layout["layout_id"] for layout in manifest["layouts"]) != COLUMNS:
            raise ValueError(f"manifest column order differs: {capture_id}")
        metadata = rebuild_metadata(
            manifest["route_metadata"], batch_id=manifest["batch_id"],
            split="evaluation", layer_id=layer, capture_id=capture_id,
        )
        base_shas = manifest["base_source_map_sha256"]
        if tuple(sorted(base_shas)) != tuple(sorted(DOMAINS)):
            raise ValueError(f"base source domains differ: {capture_id}")
        observed_sha_owner: dict[str, str] = {}
        observed_duplicates: set[tuple[str, str]] = set()
        for layout in manifest["layouts"]:
            lid = layout["layout_id"]
            p2l = tuple(int(v) for v in layout["p2l"])
            if sorted(p2l) != list(range(NUM_EXPERTS)):
                raise ValueError(f"layout not bijective: {capture_id} {lid}")
            l2p = tuple(int(v) for v in layout["l2p"])
            if any(l2p[logical] != physical for physical, logical in enumerate(p2l)):
                raise ValueError(f"layout inverse differs: {capture_id} {lid}")
            digest = placement_map_sha256(p2l)
            if digest != layout["map_sha256"]:
                raise ValueError(f"map sha differs: {capture_id} {lid}")
            if layout_descriptors(metadata, p2l) != layout["descriptors"]:
                raise ValueError(f"descriptors differ on recompute: {capture_id} {lid}")
            if lid == "identity":
                if digest != identity_sha or layout["role"] != "anchor_identity":
                    raise ValueError(f"identity anchor differs: {capture_id}")
            else:
                col_domain = lid.removeprefix("base-of-")
                if layout["role"] != "cdx0_base_column":
                    raise ValueError(f"column role differs: {capture_id} {lid}")
                greedy = greedy_by_layer[layer][col_domain]
                if list(p2l) != greedy["p2l"] or digest != greedy["map_sha256"]:
                    raise ValueError(
                        f"base column is NOT the recomputed calibration greedy base: "
                        f"{capture_id} {lid}"
                    )
                if layout["step_index"] != greedy["step_count"]:
                    raise ValueError(f"base step count differs: {capture_id} {lid}")
                if digest != base_shas[col_domain]:
                    raise ValueError(f"base sha record differs: {capture_id} {lid}")
                if layout["source"]["domain"] != col_domain or layout["source"]["split"] != "calibration":
                    raise ValueError(f"base source record differs: {capture_id} {lid}")
            if digest in observed_sha_owner:
                observed_duplicates.add(tuple(sorted((observed_sha_owner[digest], lid))))
            else:
                observed_sha_owner[digest] = lid
        flagged = {
            tuple(sorted((e["layout_a"], e["layout_b"])))
            for e in manifest.get("duplicate_base_maps", [])
        }
        if flagged != observed_duplicates:
            raise ValueError(f"duplicate flag differs from observation: {capture_id}")
        if any("identity" in pair for pair in flagged):
            raise ValueError(f"identity duplicated by a base column: {capture_id}")
        if record.get("layout_count") != 5:
            raise ValueError(f"manifest layout count differs: {capture_id}")
        manifests[capture_id] = {
            "capture_id": capture_id,
            "batch_id": manifest["batch_id"],
            "domain": domain,
            "layer_id": layer,
            "path": record["path"],
            "bytes": path.stat().st_size,
            "sha256": record["sha256"],
        }
    if len(manifests) != 8:
        raise ValueError("expected eight cdx0 layout manifests")

    return {
        "pilot_path": pilot_path,
        "pilot": pilot,
        "amend_path": amend_path,
        "amendment": amendment,
        "tolerance_path": tolerance_path,
        "gate": gate,
        "capture_index_path": capture_index_path,
        "layout_index_path": layout_index_path,
        "manifests": manifests,
        "migration_reference": migration_reference(root),
    }


def build_execution(context: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    pilot = context["pilot"]
    amendment = context["amendment"]
    manifests = context["manifests"]
    starts = []
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        record = manifests[capture_id]
        for start_number in range(1, STARTS_PER_CAPTURE + 1):
            start_id = f"cdx0-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(GENERATOR_VERSION, capture_id,
                                f"cdx0-execution-start-{start_number}")
            start_index = len(starts)
            starts.append(
                {
                    "start_id": start_id,
                    "start_index": start_index,
                    "capture_id": capture_id,
                    "batch_id": record["batch_id"],
                    "layer_id": record["layer_id"],
                    "domain": record["domain"],
                    "layout_manifest": {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    },
                    "random_seed": seed,
                    "primary_schedules": [
                        rotation_schedule(start_index, block)
                        for block in range(BLOCKS_PER_START)
                    ],
                    "output_relative_path": f"results/cdx0_v1/execution/raw/{start_id}.json",
                    "log_relative_path": f"results/cdx0_v1/execution/logs/{start_id}.log",
                    "failure_relative_path": f"results/cdx0_v1/execution/raw/{start_id}.failure.json",
                }
            )
    implementation = {
        name: file_record(SCRIPT_DIR / filename)
        for name, filename in IMPLEMENTATION_FILES.items()
    }
    prohibitions = list(pilot["prohibitions_frozen"]) + AMENDMENT_ADDED_PROHIBITIONS
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "study_id": CDX0_STUDY_ID,
        "status": STATUS,
        "split": "cdx0",
        "attempt": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
            "seed_derivation": "uint64 big-endian SHA256(cdx0_study|version|parts) mod 2^62",
            "schedule_rule": CDX0_SPEC["within_block_column_order"],
        },
        "scope": {
            "pilot_scope_statement": pilot["scope_statement"],
            "amendment_scope_reaffirmation": amendment["scope_reaffirmation"],
            "hardware_software_lock": pilot["hardware_software_lock"],
        },
        "pilot_protocol": file_record(context["pilot_path"]),
        "amendment": file_record(context["amend_path"]),
        "cdx0_study_spec": CDX0_SPEC,
        "capture_index": file_record(context["capture_index_path"]),
        "layout_index": file_record(context["layout_index_path"]),
        "layout_manifests": [manifests[c] for c in sorted(manifests)],
        "tolerance_file": file_record(context["tolerance_path"]),
        "correctness_gate": context["gate"],
        "implementation": implementation,
        "analysis_spec": {
            "estimand": (
                "T(row, column) = mean over valid starts (4 planned, >=3 required) "
                "of the 5-block mean completion_fenced_us_per_forward for that "
                "frozen column on that evaluation capture"
            ),
            "t_quantile_95_df3": T_QUANTILE_95_DF3,
            "t_quantile_95_df2": T_QUANTILE_95_DF2,
            "df_rule": "valid_starts - 1 (df=3 with 4 valid starts; df=2 with 3)",
            "min_valid_starts_per_row": 3,
            "row_with_fewer_valid_starts": "indeterminate row; retired starts never retried",
            "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
            "epsilon_in_verdicts": False,
            "epsilon_uses": [
                "practically_equivalent_within_margin labels (CI wholly inside +-epsilon)",
                "descriptive oracle-gap context",
            ],
            "paired_contrasts": (
                "within-capture start-level means of within-block paired differences; "
                "all 10 within-row column pairs reported per row (superset of the "
                "amendment-4 20-pair floor); primaries oriented as "
                "T(other-base) - T(own-base), positive = own base faster"
            ),
            "primary_family": (
                "24 primary paired contrasts = 2 layers x 4 rows x 3 own-vs-other-base; "
                "EXPLORATORY screening only; no family-wise error rate control claimed"
            ),
            "holm_secondary": (
                "Holm-Bonferroni adjusted p-values within each layer (family = 12 "
                "primary contrasts), descriptive sensitivity only; never alters the "
                "pre-registered verdict; confirmatory use requires a NEW protocol"
            ),
            "row_tracks_own_base": (
                "all three other-base-minus-own-base paired 95% CIs exclude zero positively"
            ),
            "clear_crossover_advantage_layer": ">=2 of 4 rows track own base",
            "layer_indeterminate": ">=2 rows with fewer than 3 valid starts",
            "verdict_rule": {
                "detected": ">=1 of the 2 layers shows clear crossover advantage",
                "not_detected": "0 layers show it and no layer is indeterminate",
                "indeterminate": "any layer indeterminate and none clear",
                "exploratory_label_required": True,
                "verdict_wording_constraint": (
                    "verdicts speak ONLY about the chosen layers and candidates at "
                    "replay caliber; never about whole-service benefit or harm in "
                    "either direction; not_detected is NOT evidence of absence of "
                    "exploitable cross-domain structure"
                ),
            },
            "oracle_gap_descriptive": {
                "t_static": "min over the 5 columns of sum_d K_d T(x_d, column)",
                "t_switch": "sum_d K_d min over columns of T(x_d, column)",
                "weightings": ["row_equal_K", "k_tilde_weighted"],
                "k_tilde": K_TILDE,
                "k_tilde_assumption_note_mandatory": K_TILDE_NOTE,
            },
            "migration_context_descriptive": context["migration_reference"],
            "mandatory_reporting": amendment["amendment_4_own_base_not_necessary_condition"][
                "mandatory_reporting"
            ],
        },
        "design": {
            "capture_units": 8,
            "starts_per_capture": STARTS_PER_CAPTURE,
            "fresh_start_semantics": (
                "one torchrun process group per start; every start reloads all "
                "immutable artifacts and rebuilds weights before warmup"
            ),
            "timed_layout_ids_by_capture": {c: list(COLUMNS) for c in sorted(manifests)},
            "identity_role": CDX0_SPEC["identity_role"],
            "matrix": (
                "rows = four evaluation-domain captures per layer; columns = "
                "identity + calibration-greedy base of each domain reused verbatim "
                "across the layer's rows; 2 layers x 4 rows x 5 columns = 40 timed "
                "cells in this one campaign"
            ),
            "response_definition": (
                "start-level contrast T(a) - T(b) for two columns of the same "
                "capture = mean of the five paired block effects; each block "
                "effect is completion_fenced_us_per_forward(a) - "
                "completion_fenced_us_per_forward(b) within the same rotated block"
            ),
            "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
            "report_all_signed_effects_including_negative": True,
            "no_best_measured_prefix_selection": True,
            "no_subset_selection": True,
            "zero_crossing_is_not_a_deadzone": True,
            "no_slack_subtracted_from_measured_q": True,
            "failed_start_handling": (
                "failure record written, start retired, campaign CONTINUES "
                "(amendment v1 execution discipline; implements the frozen "
                "<3-valid-starts row rule)"
            ),
            "layout_purpose": "cross_domain_execution_response_pilot",
        },
        "timing": {
            "initial_warmup_forwards_per_layout": CDX0_SPEC["initial_warmup"],
            "activation_warmup_forwards": CDX0_SPEC["activation_warmup"],
            "timed_forwards_per_activation": CDX0_SPEC["timed_forwards"],
            "blocks_per_start": BLOCKS_PER_START,
            "diagnostic_blocks_per_start": 0,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "schedule_algorithm": (
                "deterministic rotation of the frozen five-column order by "
                "(start_index + block_index) mod 5; every column is timed in "
                "every block and every position appears exactly once per start"
            ),
            "primary_metric": "completion_fenced_us_per_forward",
            "diagnostic_pass": False,
        },
        "result_index_relative_path": "results/cdx0_v1/execution/index.json",
        "prohibitions": prohibitions,
        "starts": starts,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = (
        args.protocol or root / "configs/followup_v1/cdx0_execution_protocol.json"
    ).resolve()
    context = validate_inputs(root)
    execution = build_execution(context, root)
    if args.verify_only:
        if not protocol_path.exists():
            raise FileNotFoundError("verify-only requires the frozen protocol")
        frozen = json.loads(protocol_path.read_text())
        if frozen.get("status") != STATUS or frozen.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("frozen protocol identity differs")
        for key in ("starts", "design", "prohibitions", "cdx0_study_spec", "scope",
                    "timing", "analysis_spec"):
            if frozen.get(key) != execution[key]:
                raise ValueError(f"frozen {key} differs from deterministic regeneration")
        if frozen.get("correctness_gate") != execution["correctness_gate"]:
            raise ValueError("frozen correctness gate differs")
        for name, record in frozen.get("implementation", {}).items():
            actual = sha256_file(pathlib.Path(record["path"]))
            if actual != record["sha256"]:
                raise ValueError(f"implementation hash differs on disk: {name}")
        generator = frozen.get("generator", {})
        if sha256_file(pathlib.Path(generator["path"])) != generator["sha256"]:
            raise ValueError("generator script changed since freeze")
        print(
            f"cdx0 execution protocol verified: {len(frozen['implementation'])} "
            f"implementation records, {len(frozen['starts'])} starts, "
            f"sha256={sha256_file(protocol_path)}"
        )
        return
    if protocol_path.exists():
        raise FileExistsError("refusing to overwrite the frozen cdx0 protocol")
    write_json_atomic(protocol_path, execution)
    print(f"frozen {protocol_path}")
    print(f"cdx0_execution_protocol_sha256={sha256_file(protocol_path)}")


if __name__ == "__main__":
    main()
