#!/usr/bin/env python3
"""Freeze (or verify) the trigger x map crosstab execution protocol.

Study h20-placement-crosstab-v1. The protocol is frozen BEFORE any crosstab
timing. It binds: the master follow-up protocol, the locked prediction models,
the tolerance-v2 two-tier correctness gate, the frozen crosstab layout index
(12 manifests, four fixed columns each, base maps reused VERBATIM from the
frozen nearscan manifests), the capture index, and hash records for every
implementation file the driver, runner, indexer and analyzer will use.

Seed derivation and the block-schedule algorithm are inherited unchanged from
freeze_scan_protocol.py with a new study/version prefix, so schedules are
deterministic and independent from the scan campaign. --verify-only
regenerates all deterministic content and compares without writing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
from typing import Any

from prediction_common import (
    STUDY_ID as SOURCE_STUDY_ID,
    RouteMetadata,
    layout_descriptors,
    sha256_file,
    write_json_atomic,
)

CROSSTAB_STUDY_ID = "h20-placement-crosstab-v1"
NEARSCAN_STUDY_ID = "h20-placement-nearscan-v1"
PROTOCOL_ID = "h20-placement-crosstab-execution-v1"
GENERATOR_VERSION = "crosstab-execution-protocol-v1"
STATUS = "frozen_before_crosstab_timing"
LAYERS = (24, 28, 44, 71)
BATCHES = ("evaluation-01", "evaluation-02", "evaluation-03")
COLUMNS = ("identity", "base-of-evaluation-01", "base-of-evaluation-02", "base-of-evaluation-03")
IDENTITY_MAP_SHA256 = "e374288e6fd77688ab0bf9c4f4da1c9310f37aa631ec818b593d9d22c3a25e4b"
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
T_QUANTILE_95_DF3 = 3.1824463052837078
PLANNED_TIMING = (
    "4 independent starts x 5 randomized blocks x 50 timed forwards per cell; "
    "paired within-block contrasts; Student-t df=3"
)
CROSSTAB_SPEC: dict[str, Any] = {
    "activation_warmup": 10,
    "blocks_per_start": 5,
    "columns": list(COLUMNS),
    "identity_role": "timed_anchor_and_tier1_correctness_reference",
    "initial_warmup": 10,
    "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
    "rows": "three locked evaluation triggers per layer (real captured routes)",
    "starts": 4,
    "timed_forwards": 50,
}
IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "prediction_models_record": "prediction_models.py",
    "replay_placement_record": "replay_placement.py",
    "crosstab_replay_record": "crosstab_replay.py",
    "index_crosstab_results_record": "index_crosstab_results.py",
    "analyze_crosstab_record": "analyze_crosstab.py",
    "generate_crosstab_layouts_record": "generate_crosstab_layouts.py",
    "run_crosstab_execution_record": "run_crosstab_execution.sh",
}
ANALYSIS_SPEC: dict[str, Any] = {
    "estimand": (
        "T(capture, column) = mean over 4 independent starts of the 5-block mean "
        "completion_fenced_us_per_forward for that frozen layout on that capture"
    ),
    "t_quantile_95_df3": T_QUANTILE_95_DF3,
    "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
    "paired_contrasts": (
        "within-capture start-level means of within-block paired differences: "
        "own-base-vs-identity and other-base-vs-own-base for both other columns"
    ),
    "row_tracks_own_base": (
        "both other-base-minus-own-base paired 95% CIs exclude zero positively"
    ),
    "tracking_room_supported": ">=2 of the layer's 3 rows track their own base",
    "common_static_sufficient": (
        "some base column c* has, in every row, paired delta T(c*)-T(own base) "
        "whose 95% CI lower bound is <= epsilon (not significantly worse beyond "
        "the practical margin in any row)"
    ),
    "verdict_precedence": "tracking_room_supported > common_static_sufficient > mixed",
    "equivalence_rule": (
        "only an interval wholly inside [-epsilon,+epsilon] supports practical "
        "equivalence; crossing zero alone does not"
    ),
    "ci_crossing_zero_wording": "compatible with zero; never proof of no effect",
    "report_all_signed_effects_including_negative": True,
    "cells_remeasured_in_one_campaign": (
        "all four columns are re-timed for every capture inside this single "
        "campaign so own-vs-other contrasts are internally consistent; the "
        "scan-era identity/base cells are reported only as a Welch drift "
        "cross-check and never enter any verdict"
    ),
    "cost_linkage": (
        "for significant row gains, break-even horizons are computed with the "
        "frozen batch estimator C_batch(k)=C_multi(capture)*moves_k/moves_base "
        "using measured migration costs from the frozen scan cost-benefit study; "
        "moves_k is the expert displacement between the two frozen maps"
    ),
    "scope_note": (
        "crosstab numbers are layer-replay completion times on one node with "
        "8xH20, EP=8, DeepEP normal, DeepGEMM FP8; they are not server-side "
        "latencies and cannot replace the full-service three-arm validation"
    ),
    "no_selection": (
        "no best-start, repeat, prefix or subset selection; failed starts are "
        "retired, never retried under this frozen campaign"
    ),
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def derived_seed(generator_version: str, *parts: str) -> int:
    text = "|".join((CROSSTAB_STUDY_ID, generator_version, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def schedule(seed: int, block: int, layout_ids: tuple[str, ...]) -> list[str]:
    order = list(range(len(layout_ids)))
    random.Random(seed + block).shuffle(order)
    return [layout_ids[index] for index in order]


def rebuild_metadata(record: dict[str, Any], *, batch_id: str, layer_id: int,
                     capture_id: str) -> RouteMetadata:
    return RouteMetadata(
        batch_id=batch_id,
        split="evaluation",
        layer_id=layer_id,
        capture_id=capture_id,
        hidden_rows_by_source_rank=tuple(record["hidden_rows_by_source_rank"]),
        source_expert_counts=tuple(tuple(row) for row in record["source_expert_counts"]),
        expert_counts=tuple(record["expert_counts"]),
        expert_blocks=tuple(record["expert_blocks_128"]),
        total_hidden_rows=record["total_hidden_rows"],
        total_assignments=record["total_assignments"],
    )


def validate_inputs(root: pathlib.Path) -> dict[str, Any]:
    followup_path = root / "configs/followup_v1/protocol.json"
    followup = json.loads(followup_path.read_text())

    lock_path = root / "configs/followup_v1/prediction_model_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock is not in locked status")
    for name, record in lock["locked_implementation"].items():
        actual = sha256_file(pathlib.Path(record["path"]))
        if actual != record["sha256"]:
            raise ValueError(f"locked implementation changed since model lock ({name})")

    tolerance_path = root / "configs/followup_v1/prediction_tolerance_v2.json"
    tolerance = json.loads(tolerance_path.read_text())
    if tolerance.get("status") != "frozen_before_evaluation_attempt02_timing":
        raise ValueError("tolerance v2 status differs")
    gate = tolerance["correctness_gate"]
    if float(gate["identity_versus_captured"]["atol"]) != 2.0**-8 or float(gate["identity_versus_captured"]["rtol"]) != 0.0:
        raise ValueError("tier-1 strictness must equal the pilot tolerance")

    capture_index_path = root / "inputs/prediction_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    if capture_index.get("status") != "all_prediction_capture_inputs_validated_and_hash_indexed":
        raise ValueError(f"capture index status differs: {capture_index.get('status')}")

    layout_index_path = root / "inputs/crosstab_v1/layouts/index.json"
    layout_index = json.loads(layout_index_path.read_text())
    if layout_index.get("study_id") != CROSSTAB_STUDY_ID:
        raise ValueError("crosstab layout index belongs to another study")
    if layout_index.get("status") != "generated_before_protocol_freeze":
        raise ValueError("crosstab layout index status differs")
    if layout_index.get("selection_uses_performance_timing") is not False:
        raise ValueError("crosstab selection discipline differs")
    source_index_record = layout_index.get("source_nearscan_layout_index", {})
    nearscan_index_path = pathlib.Path(source_index_record["path"])
    if sha256_file(nearscan_index_path) != source_index_record["sha256"]:
        raise ValueError("nearscan source layout index hash differs")
    nearscan_index = json.loads(nearscan_index_path.read_text())
    if nearscan_index.get("study_id") != NEARSCAN_STUDY_ID:
        raise ValueError("nearscan source index belongs to another study")
    if nearscan_index.get("capture_index_sha256") != sha256_file(capture_index_path):
        raise ValueError("source index references a different capture index")

    nearscan_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in nearscan_index["manifests"]:
        path = pathlib.Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"nearscan source manifest hash differs: {entry['capture_id']}")
        payload = json.loads(path.read_text())
        nearscan_by_key[(int(payload["layer_id"]), payload["batch_id"])] = payload

    manifests: dict[str, dict[str, Any]] = {}
    per_layer_columns: dict[int, dict[str, str]] = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"crosstab manifest hash differs: {record['capture_id']}")
        manifest = json.loads(path.read_text())
        capture_id = record["capture_id"]
        layer = int(manifest["layer_id"])
        batch = manifest["batch_id"]
        if manifest.get("study_id") != CROSSTAB_STUDY_ID or manifest.get("split") != "crosstab":
            raise ValueError(f"manifest header differs: {capture_id}")
        if manifest.get("selection_uses_performance_timing") is not False:
            raise ValueError(f"manifest selection discipline differs: {capture_id}")
        if (layer, batch) not in {(l, b) for l in LAYERS for b in BATCHES}:
            raise ValueError(f"manifest outside frozen rows: {capture_id}")
        if tuple(layout["layout_id"] for layout in manifest["layouts"]) != COLUMNS:
            raise ValueError(f"manifest column order differs: {capture_id}")
        if record.get("layout_count") != 4:
            raise ValueError(f"manifest layout count differs: {capture_id}")
        base_shas = manifest["base_source_map_sha256"]
        if tuple(sorted(base_shas)) != BATCHES or len(set(base_shas.values())) != 3:
            raise ValueError(f"base source map record differs: {capture_id}")
        metadata = rebuild_metadata(
            manifest["route_metadata"], batch_id=batch, layer_id=layer, capture_id=capture_id
        )
        src_payload = nearscan_by_key[(layer, batch)]
        if src_payload["capture_id"] != capture_id:
            raise ValueError(f"source capture differs: {capture_id}")
        if manifest["route_metadata"] != src_payload["route_metadata"]:
            raise ValueError(f"route metadata differs from frozen source: {capture_id}")
        src_bases = {rec["layout_id"]: rec for rec in src_payload["layouts"]}
        for layout in manifest["layouts"]:
            role_ok = (
                layout["role"] == "anchor_identity"
                if layout["layout_id"] == "identity"
                else layout["role"] == "crosstab_base_column"
            )
            if not role_ok:
                raise ValueError(f"column role differs: {capture_id} {layout['layout_id']}")
            if layout_descriptors(metadata, tuple(layout["p2l"])) != layout["descriptors"]:
                raise ValueError(f"descriptors differ on recompute: {capture_id} {layout['layout_id']}")
            digest = hashlib.sha256(
                ",".join(str(v) for v in layout["p2l"]).encode()
            ).hexdigest()
            if digest != layout["map_sha256"]:
                raise ValueError(f"map hash differs: {capture_id} {layout['layout_id']}")
            if layout["layout_id"] == "identity":
                if digest != IDENTITY_MAP_SHA256:
                    raise ValueError(f"identity map differs from frozen anchor: {capture_id}")
            else:
                col_batch = layout["source"]["batch_id"]
                if col_batch not in BATCHES:
                    raise ValueError(f"column source batch differs: {capture_id}")
                if digest != base_shas[col_batch]:
                    raise ValueError(f"column map differs from frozen source anchor: {capture_id}")
                src_rec = src_bases["base"] if col_batch == batch else None
                if src_rec is not None and list(src_rec["p2l"]) != layout["p2l"]:
                    raise ValueError(f"own column not verbatim from source manifest: {capture_id}")
                other_src = nearscan_by_key[(layer, col_batch)]
                other_base = next(r for r in other_src["layouts"] if r["layout_id"] == "base")
                if list(other_base["p2l"]) != layout["p2l"]:
                    raise ValueError(f"column not verbatim from its trigger's frozen base: {capture_id} {col_batch}")
                per_layer_columns.setdefault(layer, {})[col_batch] = digest
        manifests[capture_id] = {
            "capture_id": capture_id,
            "batch_id": batch,
            "layer_id": layer,
            "path": record["path"],
            "bytes": path.stat().st_size,
            "sha256": record["sha256"],
        }
    if len(manifests) != 12:
        raise ValueError("expected twelve crosstab layout manifests")
    for layer in LAYERS:
        columns = per_layer_columns.get(layer, {})
        if len(columns) != 3 or len(set(columns.values())) != 3:
            raise ValueError(f"layer {layer}: base columns are not three distinct frozen maps")

    return {
        "followup_path": followup_path,
        "followup": followup,
        "lock_path": lock_path,
        "tolerance_path": tolerance_path,
        "gate": gate,
        "layout_index_path": layout_index_path,
        "layout_index": layout_index,
        "manifests": manifests,
        "capture_index_path": capture_index_path,
    }


def build_execution(context: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    followup = context["followup"]
    manifests = context["manifests"]
    starts = []
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        record = manifests[capture_id]
        for start_number in range(1, int(CROSSTAB_SPEC["starts"]) + 1):
            start_id = f"crosstab-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(GENERATOR_VERSION, capture_id, f"crosstab-execution-start-{start_number}")
            starts.append(
                {
                    "start_id": start_id,
                    "capture_id": capture_id,
                    "batch_id": record["batch_id"],
                    "layer_id": record["layer_id"],
                    "layout_manifest": {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    },
                    "random_seed": seed,
                    "primary_schedules": [
                        schedule(seed, block, COLUMNS)
                        for block in range(int(CROSSTAB_SPEC["blocks_per_start"]))
                    ],
                    "output_relative_path": f"results/crosstab_v1/execution/raw/{start_id}.json",
                    "log_relative_path": f"results/crosstab_v1/execution/logs/{start_id}.log",
                    "failure_relative_path": f"results/crosstab_v1/execution/raw/{start_id}.failure.json",
                }
            )
    implementation = {
        name: file_record(root / "scripts" / filename)
        for name, filename in IMPLEMENTATION_FILES.items()
    }
    lock = json.loads(context["lock_path"].read_text())
    for name in sorted(lock["locked_implementation"]):
        if name in implementation and implementation[name]["sha256"] != lock["locked_implementation"][name]["sha256"]:
            raise ValueError(f"implementation hash differs from model lock: {name}")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "study_id": CROSSTAB_STUDY_ID,
        "source_study_id": SOURCE_STUDY_ID,
        "status": STATUS,
        "split": "crosstab",
        "attempt": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
            "seed_derivation": "uint64 big-endian SHA256(crosstab_study|version|parts) mod 2^62",
        },
        "scope": followup["scope"],
        "followup_protocol": file_record(context["followup_path"]),
        "crosstab_study_spec": CROSSTAB_SPEC,
        "capture_index": file_record(context["capture_index_path"]),
        "layout_index": file_record(context["layout_index_path"]),
        "layout_manifests": [manifests[capture_id] for capture_id in sorted(manifests)],
        "model_lock": file_record(context["lock_path"]),
        "tolerance_file": file_record(context["tolerance_path"]),
        "correctness_gate": context["gate"],
        "implementation": implementation,
        "analysis_spec": ANALYSIS_SPEC,
        "design": {
            "capture_units": 12,
            "starts_per_capture": int(CROSSTAB_SPEC["starts"]),
            "fresh_start_semantics": (
                "one torchrun process group per start; every start reloads all "
                "immutable artifacts and rebuilds weights before warmup"
            ),
            "timed_layout_ids_by_capture": {
                capture_id: list(COLUMNS) for capture_id in sorted(manifests)
            },
            "identity_role": CROSSTAB_SPEC["identity_role"],
            "matrix": (
                "rows = three locked evaluation triggers per layer; columns = "
                "identity + frozen greedy base of each trigger reused verbatim "
                "across the layer; every cell timed in this one campaign"
            ),
            "response_definition": (
                "start-level contrast T(a) - T(b) for two columns of the same "
                "capture = mean of the five paired block effects; each block "
                "effect is completion_fenced_us_per_forward(a) - "
                "completion_fenced_us_per_forward(b) within the same randomized block"
            ),
            "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
            "report_all_signed_effects_including_negative": True,
            "no_best_measured_prefix_selection": True,
            "no_subset_selection": True,
            "zero_crossing_is_not_a_deadzone": True,
            "no_slack_subtracted_from_measured_q": True,
            "layout_purpose": "trigger_x_map_crosstab_tracking_room_test",
        },
        "timing": {
            "initial_warmup_forwards_per_layout": int(CROSSTAB_SPEC["initial_warmup"]),
            "activation_warmup_forwards": int(CROSSTAB_SPEC["activation_warmup"]),
            "timed_forwards_per_activation": int(CROSSTAB_SPEC["timed_forwards"]),
            "blocks_per_start": int(CROSSTAB_SPEC["blocks_per_start"]),
            "diagnostic_blocks_per_start": 0,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "schedule_algorithm": (
                "random.Random(seed + block).shuffle over the frozen four-column "
                "order; every column is timed in every block"
            ),
            "primary_metric": "completion_fenced_us_per_forward",
            "diagnostic_pass": False,
        },
        "result_index_relative_path": "results/crosstab_v1/execution/index.json",
        "prohibitions": followup["prohibitions"],
        "starts": starts,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = (
        args.protocol or root / "configs/followup_v1/crosstab_execution_protocol.json"
    ).resolve()
    context = validate_inputs(root)
    execution = build_execution(context, root)
    if args.verify_only:
        if not protocol_path.exists():
            raise FileNotFoundError("verify-only requires the frozen protocol")
        frozen = json.loads(protocol_path.read_text())
        if frozen.get("status") != STATUS or frozen.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("frozen protocol identity differs")
        for key in ("starts", "design", "prohibitions", "crosstab_study_spec", "scope",
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
            f"crosstab execution protocol verified: {len(frozen['implementation'])} "
            f"implementation records, {len(frozen['starts'])} starts, "
            f"sha256={sha256_file(protocol_path)}"
        )
        return
    if protocol_path.exists():
        raise FileExistsError("refusing to overwrite the frozen crosstab protocol")
    write_json_atomic(protocol_path, execution)
    print(f"frozen {protocol_path}")
    print(f"crosstab_execution_protocol_sha256={sha256_file(protocol_path)}")


if __name__ == "__main__":
    main()
