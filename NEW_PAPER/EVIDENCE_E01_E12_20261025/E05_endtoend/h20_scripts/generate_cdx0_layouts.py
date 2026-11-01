#!/usr/bin/env python3
"""SETUP-only generator of the frozen CD-X0 layout manifests.

Study h20-placement-cdx0-v1. For each of the two locked layers (024, 071) and
each of the four evaluation-domain captures, the timed matrix has five frozen
columns:

    identity | base-of-sharegpt | base-of-prover | base-of-book | base-of-code

Every base column map is the deterministic first-improvement lexicographic
greedy (frozen generate_stopping_layouts.greedy_base, identical code and
parameters to the crosstab_v1 base-of-* lineage) computed from the REAL
assignment metadata of the CALIBRATION capture of that domain and layer
ONLY. Evaluation captures are timing rows only: they never enter layout
construction. No new optimization is performed and no performance timing is
used anywhere in this script (selection_uses_performance_timing: False).
Descriptors (rank loads, zero-aware 128 padding, comm multiset) are
RECOMPUTED per evaluation capture against that capture's own real route
metadata.

Pre-registered duplicate rule: if two domains' greedy bases produce the SAME
map for a layer, the pair is flagged in duplicate_base_maps and the columns
are kept (flag must match observations exactly at replay/index time); if any
base equals the identity map, generation aborts (design premise failure).
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

from cdx0_common import (  # noqa: E402
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    STUDY_ID,
    RouteMetadata,
    invert_p2l,
    layout_descriptors,
    placement_map_sha256,
    sha256_file,
    validate_p2l,
    write_json_atomic,
)
from generate_stopping_layouts import greedy_base  # noqa: E402

GENERATOR_VERSION = "cdx0-layouts-v1"
LAYERS = (24, 71)
DOMAINS = ("sharegpt", "prover", "book", "code")
COLUMN_DOMAINS = DOMAINS
COLUMNS = ("identity",) + tuple(f"base-of-{d}" for d in COLUMN_DOMAINS)
IDENTITY_P2L = tuple(range(NUM_EXPERTS))
MAP_SHA_CONVENTION = (
    "canonical_sha256({'mapping_semantics': 'p2l[physical]=logical', 'p2l': [...]})"
)


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def raw_r(p2l: tuple[int, ...], expert_counts: list[int]) -> float:
    loads = [0] * 8
    for slot, logical in enumerate(p2l):
        loads[slot // LOCAL_EXPERTS] += expert_counts[logical]
    mean = sum(loads) / len(loads)
    return max(loads) / mean


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


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    meta_dir = root / "inputs/cdx0_v1/route_metadata"
    out_dir = root / "inputs/cdx0_v1/layouts"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = root / "inputs/cdx0_v1/capture_index.json"
    capture_index = json.loads(index_path.read_text())
    if capture_index.get("study_id") != STUDY_ID:
        raise ValueError("capture index belongs to another study")
    capture_index_record = file_record(index_path)

    pilot_path = root / "configs/followup_v1/cdx0_pilot_protocol.json"
    amend_path = root / "configs/followup_v1/cdx0_amendment_v1.json"
    amendment = json.loads(amend_path.read_text())
    if amendment.get("status") != "FROZEN_BEFORE_ANY_CDX0_TIMING":
        raise ValueError("cdx0 amendment is not frozen before timing")
    if sha256_file(pilot_path) != amendment["amends"]["sha256"]:
        raise ValueError("amendment does not reference the pilot protocol on disk")
    requests_path = root / "configs/followup_v1/cdx0_requests.json"
    requests = json.loads(requests_path.read_text())
    if requests.get("status") != "frozen_before_cdx0_capture":
        raise ValueError("request batches are not frozen")

    # --- load the 16 extracted route-metadata records -----------------------
    meta_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    meta_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(meta_dir.glob("cdx0-*.json")):
        doc = json.loads(path.read_text())
        if doc.get("study_id") != STUDY_ID or doc.get("schema_version") != 1:
            raise ValueError(f"route-metadata header differs: {path}")
        if sha256_file(index_path) != doc["capture_index"]["sha256"]:
            raise ValueError(f"route metadata references another capture index: {path}")
        key = (doc["split"], doc["domain"], int(doc["layer_id"]))
        if key in meta_by_key:
            raise ValueError(f"duplicate route metadata for {key}")
        meta_by_key[key] = doc
        meta_records[key] = file_record(path)
    expected_keys = {(split, domain, layer)
                     for split in ("calibration", "evaluation")
                     for domain in DOMAINS for layer in LAYERS}
    if set(meta_by_key) != expected_keys:
        raise ValueError("route metadata does not cover exactly 2 splits x 4 domains x 2 layers")

    locked_implementation = {
        "generator": {
            **file_record(SCRIPT_DIR / pathlib.Path(__file__).name),
            "version": GENERATOR_VERSION,
        },
        "cdx0_common_record": file_record(SCRIPT_DIR / "cdx0_common.py"),
        "generate_stopping_layouts_record": file_record(SCRIPT_DIR / "generate_stopping_layouts.py"),
    }

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifests: list[dict[str, Any]] = []
    greedy_records: dict[str, Any] = {}

    for layer in LAYERS:
        # --- greedy bases from CALIBRATION metadata only ---------------------
        bases: dict[str, dict[str, Any]] = {}
        trajectories: dict[str, Any] = {}
        for domain in COLUMN_DOMAINS:
            calib_doc = meta_by_key[("calibration", domain, layer)]
            calib_meta = rebuild_metadata(
                calib_doc["route_metadata"],
                batch_id=calib_doc["batch_id"], split="calibration",
                layer_id=layer, capture_id=calib_doc["capture_id"],
            )
            result = greedy_base(calib_meta)
            p2l = validate_p2l(result["p2l"])
            digest = placement_map_sha256(p2l)
            if digest == placement_map_sha256(IDENTITY_P2L) and result["step_count"] > 0:
                raise SystemExit(f"layer {layer} domain {domain}: greedy returned identity via swaps")
            bases[domain] = {
                "p2l": p2l,
                "l2p": invert_p2l(p2l),
                "map_sha256": digest,
                "calibration_capture_id": calib_doc["capture_id"],
                "calibration_batch_id": calib_doc["batch_id"],
                "step_count": int(result["step_count"]),
                "scans": int(result["scans"]),
                "final_lexicographic_score_on_calibration": result["final_lexicographic_score"],
                "r_raw_on_own_calibration_capture": raw_r(p2l, list(calib_meta.expert_counts)),
                "route_metadata_file": meta_records[("calibration", domain, layer)],
            }
            trajectories[domain] = {
                "capture_id": calib_doc["capture_id"],
                "step_count": int(result["step_count"]),
                "scans": int(result["scans"]),
                "final_lexicographic_score": result["final_lexicographic_score"],
                "steps": result["steps"],
            }
        if result_identity_dup := any(
            b["map_sha256"] == placement_map_sha256(IDENTITY_P2L) for b in bases.values()
        ):
            raise SystemExit(
                f"layer {layer}: a calibration greedy base equals the identity map; "
                "CD-X0 design premise (base column distinct from identity) fails"
            )
        greedy_path = out_dir / f"greedy_layer-{layer:03d}.json"
        write_json_atomic(greedy_path, {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "layer_id": layer,
            "created_at_utc": created_at,
            "builder": "generate_stopping_layouts.greedy_base (frozen lineage)",
            "objective": "minimize lexicographic (max raw rank assignments, max padded "
                         "rank rows zero-aware 128, sorted-desc raw load vector, map sha256)",
            "inputs": "calibration-split capture route metadata only",
            "domains": trajectories,
        })
        greedy_records[f"layer-{layer:03d}"] = file_record(greedy_path)

        base_shas = {d: bases[d]["map_sha256"] for d in COLUMN_DOMAINS}
        duplicate_base_maps = [
            {"layout_a": f"base-of-{a}", "layout_b": f"base-of-{b}",
             "map_sha256": base_shas[a]}
            for i, a in enumerate(COLUMN_DOMAINS)
            for b in COLUMN_DOMAINS[i + 1:]
            if base_shas[a] == base_shas[b]
        ]

        # --- one manifest per evaluation row ---------------------------------
        for domain in DOMAINS:
            eval_doc = meta_by_key[("evaluation", domain, layer)]
            capture_id = eval_doc["capture_id"]
            eval_meta = rebuild_metadata(
                eval_doc["route_metadata"],
                batch_id=eval_doc["batch_id"], split="evaluation",
                layer_id=layer, capture_id=capture_id,
            )
            layouts: list[dict[str, Any]] = []
            ident_desc = layout_descriptors(eval_meta, IDENTITY_P2L)
            layouts.append({
                "layout_id": "identity",
                "role": "anchor_identity",
                "p2l": list(IDENTITY_P2L),
                "l2p": list(IDENTITY_P2L),
                "map_sha256": placement_map_sha256(IDENTITY_P2L),
                "descriptors": ident_desc,
                "r_raw": raw_r(IDENTITY_P2L, list(eval_meta.expert_counts)),
                "padded_max_over_mean": ident_desc["padded_max_over_mean"],
                "step_index": 0,
                "source": {
                    "domain": None, "split": None, "capture_id": None,
                    "note": "identity anchor (initial deployed service map semantics)",
                },
            })
            for col_domain in COLUMN_DOMAINS:
                base = bases[col_domain]
                p2l = base["p2l"]
                desc = layout_descriptors(eval_meta, p2l)
                layouts.append({
                    "layout_id": f"base-of-{col_domain}",
                    "role": "cdx0_base_column",
                    "p2l": list(p2l),
                    "l2p": list(base["l2p"]),
                    "map_sha256": base["map_sha256"],
                    "descriptors": desc,
                    "r_raw": raw_r(p2l, list(eval_meta.expert_counts)),
                    "padded_max_over_mean": desc["padded_max_over_mean"],
                    "step_index": base["step_count"],
                    "source": {
                        "domain": col_domain,
                        "split": "calibration",
                        "capture_id": base["calibration_capture_id"],
                        "batch_id": base["calibration_batch_id"],
                        "map_sha256": base["map_sha256"],
                        "greedy_step_count": base["step_count"],
                        "greedy_scans": base["scans"],
                        "final_lexicographic_score_on_calibration":
                            base["final_lexicographic_score_on_calibration"],
                        "r_raw_on_own_calibration_capture":
                            base["r_raw_on_own_calibration_capture"],
                        "route_metadata_file": base["route_metadata_file"],
                    },
                })
            assert [rec["layout_id"] for rec in layouts] == list(COLUMNS)
            manifest = {
                "schema_version": 1,
                "study_id": STUDY_ID,
                "split": "cdx0",
                "capture_id": capture_id,
                "batch_id": eval_doc["batch_id"],
                "domain": domain,
                "layer_id": layer,
                "selection_uses_performance_timing": False,
                "created_at_utc": created_at,
                "map_sha256_convention": MAP_SHA_CONVENTION,
                "provenance": {
                    "capture_index": capture_index_record,
                    "evaluation_route_metadata_file": meta_records[("evaluation", domain, layer)],
                    "pilot_protocol": file_record(pilot_path),
                    "amendment": file_record(amend_path),
                    "request_manifest": file_record(requests_path),
                    "greedy_trajectory": greedy_records[f"layer-{layer:03d}"],
                },
                "base_source_map_sha256": base_shas,
                "duplicate_base_maps": duplicate_base_maps,
                "route_metadata": eval_doc["route_metadata"],
                "trajectory": {
                    "greedy_path_steps": {d: bases[d]["step_count"] for d in COLUMN_DOMAINS},
                    "note": (
                        "per-domain greedy step counts on the layer's calibration "
                        "captures; full paths in the greedy_trajectory record; base "
                        "columns carry per-source provenance records"
                    ),
                },
                "generation": {
                    "generator_version": GENERATOR_VERSION,
                    "rule": (
                        "columns are the identity anchor and the deterministic "
                        "greedy bases of the layer's four CALIBRATION-domain "
                        "captures, reused verbatim across all evaluation rows of "
                        "the layer; descriptors recomputed per row; no "
                        "optimization; no timing"
                    ),
                },
                "locked_implementation": locked_implementation,
                "layouts": layouts,
            }
            out_path = out_dir / f"cdx0_{capture_id}.json"
            write_json_atomic(out_path, manifest)
            manifests.append({
                "capture_id": capture_id,
                "batch_id": eval_doc["batch_id"],
                "domain": domain,
                "layer_id": layer,
                "layout_count": len(layouts),
                "duplicate_base_maps": duplicate_base_maps,
                **file_record(out_path),
            })
            print(
                f"{capture_id}: bases steps="
                + ",".join(f"{d}:{bases[d]['step_count']}" for d in COLUMN_DOMAINS)
                + f" duplicates={len(duplicate_base_maps)}"
            )

    index_payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "generated_before_protocol_freeze",
        "selection_uses_performance_timing": False,
        "created_at_utc": created_at,
        "capture_index": capture_index_record,
        "pilot_protocol": file_record(pilot_path),
        "amendment": file_record(amend_path),
        "greedy_trajectories": greedy_records,
        "columns": list(COLUMNS),
        "map_sha256_convention": MAP_SHA_CONVENTION,
        "counts": {"manifests": len(manifests), "layouts_per_manifest": len(COLUMNS)},
        "manifests": manifests,
    }
    write_json_atomic(out_dir / "index.json", index_payload)
    print(f"wrote {len(manifests)} manifests; index sha256={sha256_file(out_dir / 'index.json')}")


if __name__ == "__main__":
    main()
