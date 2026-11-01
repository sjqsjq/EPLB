#!/usr/bin/env python3
"""Generate frozen full-service validation inputs (study h20-placement-fullservice-v1).

Per layer (024, 028, 044, 071):
  * three real evaluation-split triggers (captured routed-layer forwards),
  * one pooled offline-optimized map: the SAME frozen greedy descent
    (generate_stopping_layouts.greedy_base) applied to the pooled route
    metadata (elementwise-summed expert counts) of the three triggers;
    selection_uses_performance_timing = False (route metadata only),
  * the frozen fixed-workload schedule: 2 passes x 3 trigger blocks x 48
    service forwards, block order sorted by capture id, no shuffling,
  * identity and pooled-map descriptors for the pre-registered secondary
    model check (H_padding_linear prediction on the service workload).

Nothing here uses performance timing; all inputs are hash-verified frozen
artifacts. Outputs are new files under inputs/fullservice_v1/ (no clobber).
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
from prediction_common import (  # noqa: E402
    IDENTITY_P2L,
    NUM_EXPERTS,
    RouteMetadata,
    layout_descriptors,
    placement_map_sha256,
    sha256_file,
    write_json_atomic,
    zero_aware_blocks,
)
from generate_stopping_layouts import greedy_base  # noqa: E402

FULLSERVICE_STUDY_ID = "h20-placement-fullservice-v1"
SPLIT = "fullservice"
LAYERS = (24, 28, 44, 71)
PASSES = 2
FORWARDS_PER_BLOCK = 48
ROOT_DEFAULT = SCRIPT_DIR.parent


def route_metadata_from_record(record: dict[str, Any], *, batch_id: str, split: str,
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


def pooled_metadata(per_trigger: list[RouteMetadata], layer_id: int) -> RouteMetadata:
    counts = [sum(m.expert_counts[e] for m in per_trigger) for e in range(NUM_EXPERTS)]
    rows = tuple(sum(m.hidden_rows_by_source_rank[r] for m in per_trigger) for r in range(8))
    source = tuple(
        tuple(sum(m.source_expert_counts[r][e] for m in per_trigger) for e in range(NUM_EXPERTS))
        for r in range(8)
    )
    meta = RouteMetadata(
        batch_id=f"pooled-fullservice-layer-{layer_id:03d}",
        split=SPLIT,
        layer_id=layer_id,
        capture_id=f"fullservice-pooled-layer-{layer_id:03d}",
        hidden_rows_by_source_rank=rows,
        source_expert_counts=source,
        expert_counts=tuple(counts),
        expert_blocks=tuple(zero_aware_blocks(c) for c in counts),
        total_hidden_rows=sum(m.total_hidden_rows for m in per_trigger),
        total_assignments=sum(m.total_assignments for m in per_trigger),
    )
    if sum(meta.expert_counts) != meta.total_assignments:
        raise ValueError(f"layer {layer_id}: pooled assignment sum mismatch")
    return meta


def validate_p2l(p2l: Any) -> tuple[int, ...]:
    values = tuple(int(v) for v in p2l)
    if len(values) != NUM_EXPERTS or sorted(values) != list(range(NUM_EXPERTS)):
        raise ValueError("p2l is not a permutation of logical experts")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()

    scan_index_path = root / "inputs/nearscan_v1/layouts/index.json"
    scan_index = json.loads(scan_index_path.read_text())
    if scan_index.get("study_id") != "h20-placement-nearscan-v1":
        raise ValueError("nearscan layout index study differs")
    scan_index_sha = sha256_file(scan_index_path)

    capture_index_path = root / "inputs/prediction_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    capture_index_sha = sha256_file(capture_index_path)

    # hash-verified scan manifests keyed by capture id
    manifests: dict[str, dict[str, Any]] = {}
    manifest_shas: dict[str, str] = {}
    for record in scan_index["manifests"]:
        path = pathlib.Path(record["path"])
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"scan manifest hash differs: {path}")
        payload = json.loads(path.read_text())
        manifests[record["capture_id"]] = payload
        manifest_shas[record["capture_id"]] = digest

    out_dir = root / "inputs/fullservice_v1/service_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise ValueError(f"refusing to write into non-empty directory: {out_dir}")

    index_records = []
    for layer_id in LAYERS:
        trigger_ids = sorted(
            cid for cid in manifests
            if manifests[cid]["layer_id"] == layer_id
            and manifests[cid]["split"] == "evaluation"
        )
        if len(trigger_ids) != 3:
            raise ValueError(f"layer {layer_id}: expected 3 evaluation triggers, got {trigger_ids}")
        per_trigger = []
        trigger_records = {}
        for cid in trigger_ids:
            payload = manifests[cid]
            meta = route_metadata_from_record(
                payload["route_metadata"],
                batch_id=payload["batch_id"],
                split=payload["split"],
                layer_id=layer_id,
                capture_id=cid,
            )
            if sum(meta.expert_counts) != meta.total_assignments:
                raise ValueError(f"{cid}: assignment sum mismatch")
            per_trigger.append(meta)
            base_layout = next(l for l in payload["layouts"] if l["layout_id"] == "base")
            trigger_records[cid] = {
                "batch_id": payload["batch_id"],
                "total_hidden_rows": meta.total_hidden_rows,
                "total_assignments": meta.total_assignments,
                "r_identity": payload["trajectory"]["r_identity"],
                "r_base": payload["trajectory"]["r_base"],
                "per_trigger_base_map_sha256": base_layout["map_sha256"],
                "per_trigger_base_step_count": base_layout["step_index"],
                "scan_manifest_sha256": manifest_shas[cid],
            }
        pooled = pooled_metadata(per_trigger, layer_id)
        result = greedy_base(pooled)
        pooled_p2l = validate_p2l(result["p2l"])
        pooled_l2p = [0] * NUM_EXPERTS
        for physical, logical in enumerate(pooled_p2l):
            pooled_l2p[logical] = physical
        pooled_map_sha = placement_map_sha256(pooled_p2l)
        moves = sum(
            1 for logical, physical in enumerate(pooled_l2p)
            if physical // 16 != logical // 16
        )
        identity_desc = layout_descriptors(pooled, IDENTITY_P2L)
        pooled_desc = layout_descriptors(pooled, pooled_p2l)
        schedule_blocks = [
            {"pass_index": p, "trigger_index": t, "capture_id": trigger_ids[t],
             "forwards": FORWARDS_PER_BLOCK}
            for p in range(PASSES) for t in range(len(trigger_ids))
        ]
        expanded = [b["capture_id"] for b in schedule_blocks
                    for _ in range(b["forwards"])]
        manifest = {
            "schema_version": 1,
            "study_id": FULLSERVICE_STUDY_ID,
            "split": SPLIT,
            "layer_id": layer_id,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection_uses_performance_timing": False,
            "frozen_offline_map_rule": (
                "greedy_base (frozen lexicographic descent: max raw rank load, then max "
                "128-aligned padded load, then sorted raw profile, then map hash) applied "
                "to the pooled route metadata (elementwise-summed expert counts) of the "
                "three evaluation triggers of this layer; route metadata only, no timing"
            ),
            "trigger_capture_ids": trigger_ids,
            "triggers": trigger_records,
            "pooled_route_metadata": {
                "hidden_rows_by_source_rank": list(pooled.hidden_rows_by_source_rank),
                "total_hidden_rows": pooled.total_hidden_rows,
                "total_assignments": pooled.total_assignments,
                "expert_counts": list(pooled.expert_counts),
                "expert_blocks_128": list(pooled.expert_blocks),
                "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
            },
            "pooled_optimized_layout": {
                "layout_id": "pooled-optimized",
                "role": "frozen_offline_optimized_static_arm",
                "p2l": list(pooled_p2l),
                "l2p": list(pooled_l2p),
                "map_sha256": pooled_map_sha,
                "greedy_step_count": result["step_count"],
                "greedy_scans": result["scans"],
                "final_lexicographic_score": result["final_lexicographic_score"],
                "moves_vs_identity": moves,
                "descriptors": pooled_desc,
            },
            "identity_layout": {
                "layout_id": "identity",
                "role": "static_identity_arm",
                "p2l": list(IDENTITY_P2L),
                "l2p": list(range(NUM_EXPERTS)),
                "map_sha256": placement_map_sha256(IDENTITY_P2L),
                "descriptors": identity_desc,
            },
            "workload_schedule": {
                "passes": PASSES,
                "block_order": trigger_ids,
                "forwards_per_block": FORWARDS_PER_BLOCK,
                "total_forwards": len(expanded),
                "blocks": schedule_blocks,
                "expanded_trigger_by_service_step": expanded,
                "rule": "deterministic block order, no shuffling, identical across arms and starts",
            },
        }
        out_path = out_dir / f"fullservice_layer-{layer_id:03d}.json"
        write_json_atomic(out_path, manifest)
        digest = sha256_file(out_path)
        index_records.append({
            "layer_id": layer_id,
            "path": str(out_path),
            "sha256": digest,
            "bytes": out_path.stat().st_size,
            "trigger_capture_ids": trigger_ids,
            "pooled_map_sha256": pooled_map_sha,
            "pooled_greedy_step_count": result["step_count"],
            "pooled_moves_vs_identity": moves,
            "pooled_r_raw": pooled_desc["raw_max_over_mean"],
            "pooled_padded_max_over_mean": pooled_desc["padded_max_over_mean"],
            "total_forwards": len(expanded),
        })
        print(f"layer {layer_id:03d}: pooled map {pooled_map_sha[:12]} steps={result['step_count']} "
              f"moves={moves} r={pooled_desc['raw_max_over_mean']:.4f} "
              f"pad={pooled_desc['padded_max_over_mean']:.4f} forwards={len(expanded)}")

    index = {
        "schema_version": 1,
        "study_id": FULLSERVICE_STUDY_ID,
        "split": SPLIT,
        "status": "generated_before_fullservice_timing",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_uses_performance_timing": False,
        "counts": {
            "layers": len(index_records),
            "service_manifests": len(index_records),
            "total_service_forwards_per_run": PASSES * 3 * FORWARDS_PER_BLOCK,
        },
        "sources": {
            "nearscan_layout_index": {"path": str(scan_index_path), "sha256": scan_index_sha},
            "capture_index": {"path": str(capture_index_path), "sha256": capture_index_sha},
        },
        "locked_implementation": {
            "generate_fullservice_inputs": {
                "path": "scripts/generate_fullservice_inputs.py",
                "sha256": sha256_file(SCRIPT_DIR / "generate_fullservice_inputs.py"),
            },
            "greedy_base_source": {
                "path": "scripts/generate_stopping_layouts.py",
                "sha256": sha256_file(SCRIPT_DIR / "generate_stopping_layouts.py"),
            },
            "prediction_common": {
                "path": "scripts/prediction_common.py",
                "sha256": sha256_file(SCRIPT_DIR / "prediction_common.py"),
            },
        },
        "manifests": index_records,
    }
    index_path = out_dir / "index.json"
    write_json_atomic(index_path, index)
    print("index:", index_path, "sha256 =", sha256_file(index_path))


if __name__ == "__main__":
    main()
