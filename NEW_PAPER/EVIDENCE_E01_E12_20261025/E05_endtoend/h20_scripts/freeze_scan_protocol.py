#!/usr/bin/env python3
"""Freeze (or verify) the near-equilibrium scan execution protocol.

The protocol is frozen BEFORE any scan timing. The frozen scan layout index
inside inputs/nearscan_v1/layouts/index.json is asserted verbatim (status,
counts, design constants including the pre-registered deadzone rule and the
practical-equivalence margin); every input (model lock, scan layout manifests,
tolerance v2, capture index) is hash verified; implementation records cover
every script the launcher will run. --verify-only regenerates all deterministic
content and compares it against the frozen protocol file without writing
anything.
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
    sha256_file,
    write_json_atomic,
)

SCAN_STUDY_ID = "h20-placement-nearscan-v1"
PROTOCOL_ID = "h20-placement-nearscan-execution-v1"
GENERATOR_VERSION = "nearscan-execution-protocol-v1"
STATUS = "frozen_before_scan_timing"
SCAN_BIN_ORDER = (
    "bin-r-gt-1.80",
    "bin-r-1.40-1.80",
    "bin-r-1.20-1.40",
    "bin-r-1.10-1.20",
    "bin-r-1.05-1.10",
    "bin-r-1.00-1.05",
)
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
DEADZONE_RULE = (
    "a plateau is supported only where raw r strictly decreases while every "
    "adjacent-bin contrast interval lies wholly inside [-epsilon,+epsilon]; an "
    "interval crossing zero is compatible with zero and is never by itself a "
    "plateau declaration"
)
PLANNED_TIMING = (
    "4 independent starts x 5 randomized blocks x 50 timed forwards; paired "
    "within-block contrasts; Student-t df=3"
)
BIN_SELECTION_RULE = (
    "first greedy-path prefix whose raw r enters (lo, hi]; interior prefixes "
    "only; identity and base are anchors"
)
SCAN_SPEC: dict[str, Any] = {
    "activation_warmup": 10,
    "blocks_per_start": 5,
    "deadzone_rule": DEADZONE_RULE,
    "identity_role": "timed_anchor_and_tier1_correctness_reference",
    "initial_warmup": 10,
    "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
    "r_bins": list(SCAN_BIN_ORDER),
    "starts": 4,
    "timed_forwards": 50,
}
IMPLEMENTATION_FILES = {
    "prediction_common_record": "prediction_common.py",
    "prediction_models_record": "prediction_models.py",
    "replay_placement_record": "replay_placement.py",
    "scan_replay_record": "scan_replay.py",
    "index_scan_results_record": "index_scan_results.py",
    "generate_scan_layouts_record": "generate_scan_layouts.py",
    "run_scan_execution_record": "run_scan_execution.sh",
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def derived_seed(generator_version: str, *parts: str) -> int:
    text = "|".join((SCAN_STUDY_ID, generator_version, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def expected_scan_layout_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    present = tuple(
        entry["bin_id"]
        for entry in manifest["selection_log"]
        if entry["selected_step_index"] is not None
    )
    if any(bin_id not in SCAN_BIN_ORDER for bin_id in present):
        raise ValueError("scan manifest references an unknown bin")
    if present != tuple(bin_id for bin_id in SCAN_BIN_ORDER if bin_id in present):
        raise ValueError("scan manifest bin order differs from frozen design")
    return ("identity",) + present + ("base",)


def schedule(seed: int, block: int, layout_ids: tuple[str, ...]) -> list[str]:
    order = list(range(len(layout_ids)))
    random.Random(seed + block).shuffle(order)
    return [layout_ids[index] for index in order]


def validate_inputs(root: pathlib.Path) -> dict[str, Any]:
    followup_path = root / "configs/followup_v1/protocol.json"
    followup = json.loads(followup_path.read_text())

    lock_path = root / "configs/followup_v1/prediction_model_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock is not in locked status")
    lock_sha = sha256_file(lock_path)
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

    layout_index_path = root / "inputs/nearscan_v1/layouts/index.json"
    layout_index = json.loads(layout_index_path.read_text())
    if layout_index.get("study_id") != SCAN_STUDY_ID:
        raise ValueError("scan layout index belongs to another study")
    if layout_index.get("status") != "all_scan_layouts_frozen_before_timing":
        raise ValueError("scan layout index is not frozen before timing")
    if layout_index.get("capture_index_sha256") != sha256_file(root / "inputs/prediction_v1/capture_index.json"):
        raise ValueError("scan layout index references a different capture index")
    counts = layout_index.get("counts", {})
    if (
        counts.get("capture_units") != 12
        or counts.get("total_layouts") != 95
        or counts.get("unique_timed_map_hashes") != 73
        or counts.get("cross_capture_reused_maps") != 2
    ):
        raise ValueError(f"scan layout index counts differ: {counts}")
    design = layout_index.get("design", {})
    expected_design = {
        "bin_selection_rule": BIN_SELECTION_RULE,
        "capture_units": 12,
        "deadzone_rule": DEADZONE_RULE,
        "identity_is_timed_anchor": True,
        "no_subset_selection": True,
        "planned_timing": PLANNED_TIMING,
        "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
        "selection_uses_performance_timing": False,
    }
    drift = {key: (design.get(key), value) for key, value in expected_design.items() if design.get(key) != value}
    if drift:
        raise ValueError(f"frozen scan design drift: {drift}")

    manifests: dict[str, dict[str, Any]] = {}
    total_layouts = 0
    unique_hashes: dict[str, list[str]] = {}
    for record in layout_index["manifests"]:
        path = pathlib.Path(record["path"])
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"scan layout manifest hash differs: {record['capture_id']}")
        manifest = json.loads(path.read_text())
        capture_id = record["capture_id"]
        if manifest.get("capture_id") != capture_id or manifest.get("split") != "nearscan":
            raise ValueError(f"manifest header differs: {capture_id}")
        if manifest.get("study_id") != SCAN_STUDY_ID:
            raise ValueError(f"manifest belongs to another study: {capture_id}")
        if manifest.get("selection_uses_performance_timing") is not False:
            raise ValueError(f"manifest selection discipline differs: {capture_id}")
        layout_ids = expected_scan_layout_ids(manifest)
        if tuple(layout["layout_id"] for layout in manifest["layouts"]) != layout_ids:
            raise ValueError(f"manifest layout ids differ: {capture_id}")
        if record.get("layout_count") != len(layout_ids):
            raise ValueError(f"manifest layout count differs from index record: {capture_id}")
        if not 3 <= len(layout_ids) <= 8:
            raise ValueError(f"manifest layout count outside frozen bounds: {capture_id}")
        for layout in manifest["layouts"]:
            unique_hashes.setdefault(layout["map_sha256"], []).append(
                f"{capture_id}:{layout['layout_id']}"
            )
        total_layouts += len(layout_ids)
        manifests[capture_id] = record
    if len(manifests) != 12:
        raise ValueError("expected twelve scan layout manifests")
    if total_layouts != counts.get("total_layouts"):
        raise ValueError("manifest layout total differs from index counts")
    if len(unique_hashes) != counts.get("unique_timed_map_hashes"):
        raise ValueError("unique map hash count differs from index counts")
    reused = {digest: users for digest, users in unique_hashes.items() if len(users) > 1}
    if len(reused) != counts.get("cross_capture_reused_maps"):
        raise ValueError("cross-capture reuse count differs from index counts")
    if reused != layout_index.get("cross_capture_map_reuse"):
        raise ValueError("cross-capture reuse detail differs from index record")

    capture_index_path = root / "inputs/prediction_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    if capture_index.get("status") != "all_prediction_capture_inputs_validated_and_hash_indexed":
        raise ValueError(f"capture index status differs: {capture_index.get('status')}")

    return {
        "followup_path": followup_path,
        "followup": followup,
        "lock_path": lock_path,
        "lock_sha": lock_sha,
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
    layout_index = context["layout_index"]
    starts = []
    timed_ids_by_capture: dict[str, list[str]] = {}
    for capture_number, capture_id in enumerate(sorted(manifests), 1):
        record = manifests[capture_id]
        manifest = json.loads(pathlib.Path(record["path"]).read_text())
        layout_ids = expected_scan_layout_ids(manifest)
        timed_ids_by_capture[capture_id] = list(layout_ids)
        for start_number in range(1, int(SCAN_SPEC["starts"]) + 1):
            start_id = f"scan-{capture_number:02d}-start-{start_number}"
            seed = derived_seed(GENERATOR_VERSION, capture_id, f"scan-execution-start-{start_number}")
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
                        schedule(seed, block, layout_ids)
                        for block in range(int(SCAN_SPEC["blocks_per_start"]))
                    ],
                    "output_relative_path": f"results/nearscan_v1/execution/raw/{start_id}.json",
                    "log_relative_path": f"results/nearscan_v1/execution/logs/{start_id}.log",
                    "failure_relative_path": f"results/nearscan_v1/execution/raw/{start_id}.failure.json",
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
        "study_id": SCAN_STUDY_ID,
        "source_study_id": SOURCE_STUDY_ID,
        "status": STATUS,
        "split": "nearscan",
        "attempt": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
            "seed_derivation": "uint64 big-endian SHA256(nearscan_study|version|parts) mod 2^62",
        },
        "scope": followup["scope"],
        "followup_protocol": file_record(context["followup_path"]),
        "scan_study_spec": SCAN_SPEC,
        "capture_index": file_record(context["capture_index_path"]),
        "layout_index": file_record(context["layout_index_path"]),
        "layout_manifests": [manifests[capture_id] for capture_id in sorted(manifests)],
        "model_lock": file_record(context["lock_path"]),
        "tolerance_file": file_record(context["tolerance_path"]),
        "correctness_gate": context["gate"],
        "implementation": implementation,
        "design": {
            "capture_units": 12,
            "starts_per_capture": int(SCAN_SPEC["starts"]),
            "fresh_start_semantics": (
                "one torchrun process group per start; every start reloads all "
                "immutable artifacts and rebuilds weights before warmup"
            ),
            "timed_layout_ids_by_capture": timed_ids_by_capture,
            "identity_role": SCAN_SPEC["identity_role"],
            "response_definition": (
                "start-level contrast T(a) - T(b) for two layouts of the same "
                "capture = mean of the five paired block effects; each block "
                "effect is completion_fenced_us_per_forward(a) - "
                "completion_fenced_us_per_forward(b) within the same randomized "
                "block; adjacent-bin contrasts follow r-descending layout order; "
                "identity-vs-layout contrasts measure cumulative gain"
            ),
            "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
            "equivalence_rule": (
                "only an interval wholly inside [-epsilon,+epsilon] supports practical "
                "equivalence; crossing zero alone does not"
            ),
            "deadzone_rule": DEADZONE_RULE,
            "report_all_signed_effects_including_negative": True,
            "no_best_measured_prefix_selection": True,
            "no_subset_selection": True,
            "zero_crossing_is_not_a_deadzone": True,
            "no_slack_subtracted_from_measured_q": True,
            "no_refit_before_or_during_scan": True,
            "cross_capture_map_reuse": (
                "two maps (identity and one perfectly balanced layout) recur across "
                "captures by construction; the measurement unit is (capture, layout), "
                "so reuse is legal and is recorded in the frozen layout index"
            ),
            "layout_purpose": "near_equilibrium_real_layout_deadzone_scan",
        },
        "timing": {
            "initial_warmup_forwards_per_layout": int(SCAN_SPEC["initial_warmup"]),
            "activation_warmup_forwards": int(SCAN_SPEC["activation_warmup"]),
            "timed_forwards_per_activation": int(SCAN_SPEC["timed_forwards"]),
            "blocks_per_start": int(SCAN_SPEC["blocks_per_start"]),
            "diagnostic_blocks_per_start": 0,
            "empty_fence_warmup": 5,
            "empty_fence_repetitions": 20,
            "schedule_algorithm": (
                "random.Random(seed + block).shuffle over that capture's frozen "
                "layout order (identity anchor + first-entry r-bin layouts + base "
                "anchor; every layout in the manifest is timed)"
            ),
            "primary_metric": "completion_fenced_us_per_forward",
            "diagnostic_pass": False,
        },
        "result_index_relative_path": "results/nearscan_v1/execution/index.json",
        "prohibitions": followup["prohibitions"],
        "starts": starts,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = (
        args.protocol or root / "configs/followup_v1/scan_execution_protocol.json"
    ).resolve()
    context = validate_inputs(root)
    execution = build_execution(context, root)
    if args.verify_only:
        if not protocol_path.exists():
            raise FileNotFoundError("verify-only requires the frozen protocol")
        frozen = json.loads(protocol_path.read_text())
        if frozen.get("status") != STATUS or frozen.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("frozen protocol identity differs")
        for key in ("starts", "design", "prohibitions", "scan_study_spec", "scope", "timing"):
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
            f"scan execution protocol verified: {len(frozen['implementation'])} "
            f"implementation records, {len(frozen['starts'])} starts, "
            f"sha256={sha256_file(protocol_path)}"
        )
        return
    if protocol_path.exists():
        raise FileExistsError("refusing to overwrite the frozen scan protocol")
    write_json_atomic(protocol_path, execution)
    print(f"frozen {protocol_path}")
    print(f"scan_execution_protocol_sha256={sha256_file(protocol_path)}")


if __name__ == "__main__":
    main()
