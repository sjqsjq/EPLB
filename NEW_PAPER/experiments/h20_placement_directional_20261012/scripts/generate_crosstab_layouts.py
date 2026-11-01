#!/usr/bin/env python3
"""Setup-only generator of the frozen trigger x map crosstab layout manifests.

Study h20-placement-crosstab-v1. For each of the four locked layers (24/28/44/71)
and each of the three locked evaluation triggers (evaluation-01/02/03), the timed
matrix has four frozen columns:

    identity | base-of-evaluation-01 | base-of-evaluation-02 | base-of-evaluation-03

Every base column map is taken VERBATIM (p2l) from the base anchor record of the
frozen nearscan layout manifest of that trigger for that layer. No new
optimization is performed and no performance timing is used anywhere in this
script (selection_uses_performance_timing: False). Descriptors (rank loads,
padding, comm multiset) are RECOMPUTED per capture against the capture's own
real route metadata, because the same map meets different routes on different
triggers - that recomputation is exactly what answers whether per-trigger map
choice has tracking room.

Route metadata is copied verbatim from the source nearscan manifest (which was
itself validated collectively against the eight real forward payloads at scan
execution time); the rebuild is additionally self-checked here by recomputing
the source manifest's own identity and base descriptors and requiring exact
equality.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    RouteMetadata,
    layout_descriptors,
    sha256_file,
    write_json_atomic,
)

CROSSTAB_STUDY_ID = "h20-placement-crosstab-v1"
SOURCE_STUDY_ID = "h20-placement-nearscan-v1"
GENERATOR_VERSION = "crosstab-layouts-v1"
LAYERS = (24, 28, 44, 71)
BATCHES = ("evaluation-01", "evaluation-02", "evaluation-03")
CROSSTAB_LAYOUT_IDS = (
    "identity",
    "base-of-evaluation-01",
    "base-of-evaluation-02",
    "base-of-evaluation-03",
)
LOCKED_RECORDS = {
    "prediction_common": "prediction_common.py",
    "generate_stopping_layouts": "generate_stopping_layouts.py",
    "generate_scan_layouts": "generate_scan_layouts.py",
    "crosstab_replay": "crosstab_replay.py",
}


def map_hash(p2l: list[int]) -> str:
    # EXACT convention of the frozen nearscan generator (generate_scan_layouts.py
    # map_hash), so crosstab map shas remain comparable with scan-era records.
    return hashlib.sha256(",".join(str(v) for v in p2l).encode()).hexdigest()


def raw_r(p2l: list[int], expert_counts: list[int]) -> float:
    loads = [0] * 8
    for slot, logical in enumerate(p2l):
        loads[slot // LOCAL_EXPERTS] += expert_counts[logical]
    mean = sum(loads) / len(loads)
    return max(loads) / mean


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


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--nearscan-layout-index", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    index_path = args.nearscan_layout_index or root / "inputs/nearscan_v1/layouts/index.json"
    out_dir = args.output_dir or root / "inputs/crosstab_v1/layouts"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_payload = json.loads(index_path.read_text())
    if index_payload.get("study_id") != SOURCE_STUDY_ID:
        raise ValueError("nearscan layout index study_id differs")
    index_record = {"path": str(index_path), "sha256": sha256_file(index_path)}

    # --- load and validate the twelve frozen nearscan manifests -------------
    sources: dict[tuple[int, str], dict[str, Any]] = {}
    source_records: dict[tuple[int, str], dict[str, str]] = {}
    for entry in index_payload["manifests"]:
        path = pathlib.Path(entry["path"])
        if not path.is_absolute():
            path = root / path
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"nearscan manifest hash differs: {path}")
        payload = json.loads(path.read_text())
        if payload["study_id"] != SOURCE_STUDY_ID or payload["split"] != "nearscan":
            raise ValueError(f"unexpected source manifest header: {path}")
        key = (int(payload["layer_id"]), payload["batch_id"])
        if key in sources:
            raise ValueError(f"duplicate source manifest for {key}")
        sources[key] = payload
        source_records[key] = {"path": str(path), "sha256": digest}
    expected_keys = {(layer, batch) for layer in LAYERS for batch in BATCHES}
    if set(sources) != expected_keys:
        raise ValueError("nearscan layout index does not cover exactly 4 layers x 3 triggers")

    # --- per layer: extract frozen identity + three distinct base maps ------
    layer_columns: dict[int, dict[str, Any]] = {}
    for layer in LAYERS:
        identity_records = []
        base_by_batch: dict[str, dict[str, Any]] = {}
        for batch in BATCHES:
            payload = sources[(layer, batch)]
            by_id = {record["layout_id"]: record for record in payload["layouts"]}
            ident = by_id["identity"]
            if ident["role"] != "anchor_identity":
                raise ValueError(f"identity role differs in {batch} layer {layer}")
            identity_records.append(ident)
            base = by_id["base"]
            if base["role"] != "anchor_base":
                raise ValueError(f"base role differs in {batch} layer {layer}")
            base_by_batch[batch] = base
        identity_shas = {rec["map_sha256"] for rec in identity_records}
        if len(identity_shas) != 1:
            raise ValueError(f"identity map differs across triggers of layer {layer}")
        base_shas = [base_by_batch[b]["map_sha256"] for b in BATCHES]
        if len(set(base_shas)) != 3:
            raise ValueError(
                f"layer {layer}: greedy base maps are NOT three distinct frozen maps; "
                "crosstab design premise fails on this layer"
            )
        layer_columns[layer] = {
            "identity": identity_records[0],
            "bases": base_by_batch,
            "base_source_map_sha256": dict(zip(BATCHES, base_shas)),
        }

    locked_implementation = {
        "generator": {
            "path": str(SCRIPT_DIR / pathlib.Path(__file__).name),
            "sha256": sha256_file(SCRIPT_DIR / pathlib.Path(__file__).name),
            "version": GENERATOR_VERSION,
        },
        **{
            key: {"path": str(SCRIPT_DIR / name), "sha256": sha256_file(SCRIPT_DIR / name)}
            for key, name in LOCKED_RECORDS.items()
        },
    }

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifests = []
    for layer in LAYERS:
        columns = layer_columns[layer]
        for batch in BATCHES:
            src_payload = sources[(layer, batch)]
            capture_id = src_payload["capture_id"]
            route_metadata = src_payload["route_metadata"]
            metadata = rebuild_metadata(
                route_metadata, batch_id=batch, layer_id=layer, capture_id=capture_id
            )
            # self-check: rebuild reproduces the source manifest's own descriptors
            for check_id in ("identity", "base"):
                src_rec = next(r for r in src_payload["layouts"] if r["layout_id"] == check_id)
                recomputed = layout_descriptors(metadata, tuple(src_rec["p2l"]))
                if recomputed != src_rec["descriptors"]:
                    raise ValueError(
                        f"metadata rebuild self-check failed on {capture_id} {check_id}"
                    )

            layouts: list[dict[str, Any]] = []
            ident = columns["identity"]
            ident_p2l = [int(v) for v in ident["p2l"]]
            ident_desc = layout_descriptors(metadata, tuple(ident_p2l))
            if map_hash(ident_p2l) != ident["map_sha256"]:
                raise ValueError(f"identity map sha mismatch on {capture_id}")
            layouts.append({
                "layout_id": "identity",
                "role": "anchor_identity",
                "p2l": ident_p2l,
                "l2p": [int(v) for v in ident["l2p"]],
                "map_sha256": ident["map_sha256"],
                "descriptors": ident_desc,
                "r_raw": raw_r(ident_p2l, list(metadata.expert_counts)),
                "padded_max_over_mean": ident_desc["padded_max_over_mean"],
                "step_index": 0,
                "source": {
                    "batch_id": None,
                    "note": "identity anchor; same frozen map as all nearscan identity anchors",
                },
            })
            for col_batch in BATCHES:
                src_base = columns["bases"][col_batch]
                p2l = [int(v) for v in src_base["p2l"]]
                l2p = [int(v) for v in src_base["l2p"]]
                digest = map_hash(p2l)
                if digest != src_base["map_sha256"]:
                    raise ValueError(
                        f"base map sha mismatch: layer {layer} column {col_batch}"
                    )
                desc = layout_descriptors(metadata, tuple(p2l))
                layouts.append({
                    "layout_id": f"base-of-{col_batch}",
                    "role": "crosstab_base_column",
                    "p2l": p2l,
                    "l2p": l2p,
                    "map_sha256": digest,
                    "descriptors": desc,
                    "r_raw": raw_r(p2l, list(metadata.expert_counts)),
                    "padded_max_over_mean": desc["padded_max_over_mean"],
                    "step_index": src_base["step_index"],
                    "source": {
                        "batch_id": col_batch,
                        "layer_id": layer,
                        "map_sha256": digest,
                        "source_manifest_path": source_records[(layer, col_batch)]["path"],
                        "source_manifest_sha256": source_records[(layer, col_batch)]["sha256"],
                        "source_step_index": src_base["step_index"],
                        "source_r_raw_on_own_capture": src_base["r_raw"],
                    },
                })
            assert [rec["layout_id"] for rec in layouts] == list(CROSSTAB_LAYOUT_IDS)

            manifest = {
                "schema_version": 1,
                "study_id": CROSSTAB_STUDY_ID,
                "source_study_id": SOURCE_STUDY_ID,
                "split": "crosstab",
                "capture_id": capture_id,
                "batch_id": batch,
                "layer_id": layer,
                "selection_uses_performance_timing": False,
                "created_at_utc": created_at,
                "source_nearscan_layout_index": index_record,
                "source_nearscan_manifests": {
                    b: source_records[(layer, b)] for b in BATCHES
                },
                "base_source_map_sha256": columns["base_source_map_sha256"],
                "route_metadata": route_metadata,
                "trajectory": {
                    "greedy_path_steps": src_payload["trajectory"]["greedy_path_steps"],
                    "note": (
                        "greedy_path_steps copied from this capture's source nearscan "
                        "manifest; base columns carry per-source provenance records"
                    ),
                },
                "generation": {
                    "generator_version": GENERATOR_VERSION,
                    "rule": (
                        "columns are the frozen identity anchor and the frozen greedy "
                        "base anchors of the layer's three evaluation triggers, reused "
                        "verbatim across all captures of the layer; descriptors "
                        "recomputed per capture; no optimization; no timing"
                    ),
                },
                "locked_implementation": locked_implementation,
                "layouts": layouts,
            }
            out_path = out_dir / f"crosstab_{capture_id}_layer-{layer:03d}.json"
            if out_path.exists():
                raise SystemExit(f"refusing to overwrite {out_path}")
            write_json_atomic(out_path, manifest)
            manifests.append({
                "capture_id": capture_id,
                "batch_id": batch,
                "layer_id": layer,
                "path": str(out_path),
                "sha256": sha256_file(out_path),
                "layout_count": len(layouts),
                "base_source_map_sha256": columns["base_source_map_sha256"],
                "r_identity": layouts[0]["r_raw"],
                "r_own_base": next(
                    rec["r_raw"] for rec in layouts
                    if rec["layout_id"] == f"base-of-{batch}"
                ),
            })
            print(f"{capture_id}: wrote {out_path.name}")

    index = {
        "schema_version": 1,
        "study_id": CROSSTAB_STUDY_ID,
        "status": "generated_before_protocol_freeze",
        "created_at_utc": created_at,
        "selection_uses_performance_timing": False,
        "source_nearscan_layout_index": index_record,
        "design": {
            "rows": "three locked evaluation triggers per layer (real captured routes)",
            "columns": list(CROSSTAB_LAYOUT_IDS),
            "column_rule": (
                "identity plus the frozen greedy base map of each trigger, reused "
                "verbatim across the layer's three captures"
            ),
            "layers": list(LAYERS),
            "batches": list(BATCHES),
            "cells_remeasured_in_one_campaign": True,
            "rationale": (
                "all 4 columns are timed for every capture inside a single campaign "
                "so own-vs-other-map contrasts are internally consistent; existing "
                "nearscan identity/base cells serve only as drift cross-checks"
            ),
        },
        "manifests": manifests,
        "locked_implementation": locked_implementation,
    }
    write_json_atomic(out_dir / "index.json", index)
    if len(manifests) != 12:
        raise AssertionError("expected 12 crosstab manifests")
    print(f"wrote {out_dir}/index.json with {len(manifests)} manifests")


if __name__ == "__main__":
    main()
