#!/usr/bin/env python3
"""Freeze the near-equilibrium real-layout scan (deadzone curve) layouts.

Deterministic CPU generator. For every locked evaluation-split capture unit the
frozen low-r lexicographic greedy path (already recorded inside the stopping
layout manifests) is replayed from identity; the raw imbalance r = max-over-mean
rank assignments is non-increasing along that path, so path prefixes form a
dense deterministic trajectory from the captured imbalance down to r ~ 1.000.

Frozen selection rule (before any timing): six pre-specified r bins plus the
identity and base anchors; each bin takes the FIRST path prefix whose r enters
the bin (earliest step index); empty bins stay empty and are recorded as such.
No randomness, no performance timing, no subset selection of captures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
from typing import Any

from prediction_common import (
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    RouteMetadata,
    invert_p2l,
    layout_descriptors,
    sha256_file,
    write_json_atomic,
)

SCAN_STUDY_ID = "h20-placement-nearscan-v1"
GENERATOR_VERSION = "nearscan-layouts-v1"
# (layout_id, lo_exclusive, hi_inclusive) with r in (lo, hi]
R_BINS = (
    ("bin-r-gt-1.80", 1.8, math.inf),
    ("bin-r-1.40-1.80", 1.4, 1.8),
    ("bin-r-1.20-1.40", 1.2, 1.4),
    ("bin-r-1.10-1.20", 1.1, 1.2),
    ("bin-r-1.05-1.10", 1.05, 1.1),
    ("bin-r-1.00-1.05", 1.0, 1.05),
)
PRACTICAL_EQUIVALENCE_MARGIN_US = 22.948043518066406
LOCKED_RECORDS = {
    "prediction_common": "prediction_common.py",
    "generate_stopping_layouts": "generate_stopping_layouts.py",
}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--stopping-layouts", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    return parser.parse_args()


def raw_r(p2l: list[int], expert_counts: list[int]) -> float:
    loads = [0] * 8
    for slot, logical in enumerate(p2l):
        loads[slot // LOCAL_EXPERTS] += expert_counts[logical]
    mean = sum(loads) / len(loads)
    return max(loads) / mean


def map_hash(p2l: list[int]) -> str:
    return hashlib.sha256(",".join(str(v) for v in p2l).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stopping_dir = (args.stopping_layouts or root / "inputs/stopping_v1/layouts").resolve()
    out_dir = (args.output_dir or root / "inputs/nearscan_v1/layouts").resolve()

    stopping_index_path = stopping_dir / "index.json"
    stopping_index = json.loads(stopping_index_path.read_text())
    if stopping_index.get("status") != "all_stopping_layouts_frozen_before_timing":
        raise ValueError("stopping layout index is not frozen")
    stopping_index_sha = sha256_file(stopping_index_path)

    locked = {
        name: {
            "path": str((root / "scripts" / filename).resolve()),
            "sha256": sha256_file(root / "scripts" / filename),
        }
        for name, filename in LOCKED_RECORDS.items()
    }

    capture_index_path = root / "inputs/prediction_v1/capture_index.json"
    capture_index = json.loads(capture_index_path.read_text())
    capture_index_sha = sha256_file(capture_index_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    all_timed_hashes: dict[str, str] = {}
    for record in stopping_index["manifests"]:
        src_path = pathlib.Path(record["path"])
        if sha256_file(src_path) != record["sha256"]:
            raise ValueError(f"stopping manifest hash differs: {record['capture_id']}")
        src = json.loads(src_path.read_text())
        capture_id = record["capture_id"]
        expert_counts = [int(v) for v in src["route_metadata"]["expert_counts"]]
        base_layout = next(l for l in src["layouts"] if l["layout_id"] == "base")
        base_p2l = [int(v) for v in base_layout["p2l"]]
        steps = src["base_greedy_path"]["steps"]

        # replay the frozen greedy path from identity
        p2l = list(range(NUM_EXPERTS))
        trajectory = [{"step_index": 0, "r_raw": raw_r(p2l, expert_counts), "p2l": list(p2l)}]
        for index, step in enumerate(steps, 1):
            a, b = int(step["slot_a"]), int(step["slot_b"])
            if step["logical_a"] != p2l[a] or step["logical_b"] != p2l[b]:
                raise ValueError(f"greedy step {index} does not match replay for {capture_id}")
            p2l[a], p2l[b] = p2l[b], p2l[a]
            trajectory.append({"step_index": index, "r_raw": raw_r(p2l, expert_counts), "p2l": list(p2l)})
        if trajectory[-1]["p2l"] != base_p2l:
            raise ValueError(f"greedy replay does not end at frozen base for {capture_id}")
        rs = [point["r_raw"] for point in trajectory]
        if any(rs[i + 1] > rs[i] + 1e-12 for i in range(len(rs) - 1)):
            raise ValueError(f"raw r is not monotone non-increasing along path for {capture_id}")

        # frozen first-entry bin selection over interior prefixes (base excluded)
        chosen: dict[str, dict[str, Any]] = {}
        for point in trajectory[1:-1]:
            r = point["r_raw"]
            for bin_id, lo, hi in R_BINS:
                if bin_id in chosen:
                    continue
                if lo < r <= hi:
                    chosen[bin_id] = point
                    break
        layouts = []
        selections = []
        identity = trajectory[0]
        layouts.append(
            {
                "layout_id": "identity",
                "role": "anchor_identity",
                "bin": None,
                "step_index": 0,
                "r_raw": identity["r_raw"],
                "p2l": identity["p2l"],
            }
        )
        for bin_id, lo, hi in R_BINS:
            if bin_id in chosen:
                point = chosen[bin_id]
                layouts.append(
                    {
                        "layout_id": bin_id,
                        "role": "bin_first_entry",
                        "bin": {"bin_id": bin_id, "lo_exclusive": lo, "hi_inclusive": hi},
                        "step_index": point["step_index"],
                        "r_raw": point["r_raw"],
                        "p2l": point["p2l"],
                    }
                )
                selections.append(
                    {
                        "bin_id": bin_id,
                        "selected_step_index": point["step_index"],
                        "r_raw": point["r_raw"],
                        "rule": "first path prefix whose raw r enters (lo, hi]",
                    }
                )
            else:
                selections.append({"bin_id": bin_id, "selected_step_index": None, "r_raw": None, "rule": "empty bin: no interior prefix enters the bin"})
        layouts.append(
            {
                "layout_id": "base",
                "role": "anchor_base",
                "bin": None,
                "step_index": len(steps),
                "r_raw": trajectory[-1]["r_raw"],
                "p2l": base_p2l,
            }
        )

        metadata = RouteMetadata(
            batch_id=src["batch_id"],
            split="evaluation",
            layer_id=src["layer_id"],
            capture_id=capture_id,
            hidden_rows_by_source_rank=tuple(src["route_metadata"]["hidden_rows_by_source_rank"]),
            source_expert_counts=tuple(tuple(row) for row in src["route_metadata"]["source_expert_counts"]),
            expert_counts=tuple(expert_counts),
            expert_blocks=tuple(src["route_metadata"]["expert_blocks_128"]),
            total_hidden_rows=src["route_metadata"]["total_hidden_rows"],
            total_assignments=src["route_metadata"]["total_assignments"],
        )
        hashes = set()
        for layout in layouts:
            p2l_tuple = tuple(int(v) for v in layout["p2l"])
            if sorted(p2l_tuple) != list(range(NUM_EXPERTS)):
                raise ValueError(f"layout {layout['layout_id']} is not bijective ({capture_id})")
            l2p = invert_p2l(p2l_tuple)
            digest = map_hash(list(p2l_tuple))
            if digest in hashes:
                raise ValueError(f"duplicate map inside capture {capture_id}")
            # Cross-capture map reuse is legal: the measurement unit is
            # (capture, layout) and each capture replays its own real forward.
            # Reuse is recorded for transparency instead of prohibited.
            hashes.add(digest)
            all_timed_hashes.setdefault(digest, []).append(f"{capture_id}:{layout['layout_id']}")
            layout["l2p"] = list(l2p)
            layout["map_sha256"] = digest
            layout["padded_max_over_mean"] = layout_descriptors(metadata, p2l_tuple)["padded_max_over_mean"]
            layout["descriptors"] = layout_descriptors(metadata, p2l_tuple)
        manifest = {
            "schema_version": 1,
            "study_id": SCAN_STUDY_ID,
            "split": "nearscan",
            "batch_id": src["batch_id"],
            "layer_id": src["layer_id"],
            "capture_id": capture_id,
            "selection_uses_performance_timing": False,
            "designated_capture_rule": "all locked evaluation-split capture units; no subset selection",
            "source_stopping_manifest": {"path": str(src_path), "sha256": record["sha256"]},
            "route_metadata": {
                "hidden_rows_by_source_rank": list(metadata.hidden_rows_by_source_rank),
                "total_hidden_rows": metadata.total_hidden_rows,
                "total_assignments": metadata.total_assignments,
                "source_expert_counts": [list(row) for row in metadata.source_expert_counts],
                "expert_counts": list(metadata.expert_counts),
                "expert_blocks_128": list(metadata.expert_blocks),
                "zero_expert_padding_rule": "0 if n_e=0 else 128*ceil(n_e/128)",
            },
            "trajectory": {
                "greedy_path_steps": len(steps),
                "r_identity": trajectory[0]["r_raw"],
                "r_base": trajectory[-1]["r_raw"],
                "r_monotone_non_increasing": True,
            },
            "bin_definitions": [
                {"bin_id": bin_id, "lo_exclusive": lo, "hi_inclusive": hi}
                for bin_id, lo, hi in R_BINS
            ],
            "selection_log": selections,
            "layouts": layouts,
            "locked_implementation": locked,
            "stopping_layout_index_sha256": stopping_index_sha,
            "capture_index_sha256": capture_index_sha,
            "generation": {
                "generator": str(pathlib.Path(__file__).resolve()),
                "generator_sha256": sha256_file(pathlib.Path(__file__).resolve()),
                "version": GENERATOR_VERSION,
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        }
        out_path = out_dir / f"nearscan_{capture_id}.json"
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite {out_path}")
        staging = out_path.with_suffix(".json.tmp")
        write_json_atomic(staging, manifest)
        staging.replace(out_path)
        manifests.append(
            {
                "capture_id": capture_id,
                "batch_id": src["batch_id"],
                "layer_id": src["layer_id"],
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
                "layout_count": len(layouts),
                "r_identity": trajectory[0]["r_raw"],
                "r_base": trajectory[-1]["r_raw"],
                "empty_bins": [s["bin_id"] for s in selections if s["selected_step_index"] is None],
            }
        )
        print(
            f"{capture_id}: steps={len(steps)} r_id={trajectory[0]['r_raw']:.3f} "
            f"r_base={trajectory[-1]['r_raw']:.3f} layouts={len(layouts)} "
            f"empty_bins={manifests[-1]['empty_bins']}"
        )

    index = {
        "schema_version": 1,
        "study_id": SCAN_STUDY_ID,
        "status": "all_scan_layouts_frozen_before_timing",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "design": {
            "capture_units": 12,
            "bin_selection_rule": "first greedy-path prefix whose raw r enters (lo, hi]; interior prefixes only; identity and base are anchors",
            "identity_is_timed_anchor": True,
            "practical_equivalence_margin_us": PRACTICAL_EQUIVALENCE_MARGIN_US,
            "deadzone_rule": (
                "a plateau is supported only where raw r strictly decreases while every "
                "adjacent-bin contrast interval lies wholly inside [-epsilon,+epsilon]; "
                "an interval crossing zero is compatible with zero and is never by itself "
                "a plateau declaration"
            ),
            "planned_timing": "4 independent starts x 5 randomized blocks x 50 timed forwards; paired within-block contrasts; Student-t df=3",
            "no_subset_selection": True,
            "selection_uses_performance_timing": False,
        },
        "counts": {
            "capture_units": len(manifests),
            "total_layouts": sum(m["layout_count"] for m in manifests),
            "unique_timed_map_hashes": len(all_timed_hashes),
            "cross_capture_reused_maps": sum(1 for users in all_timed_hashes.values() if len(users) > 1),
        },
        "cross_capture_map_reuse": {
            digest: users
            for digest, users in sorted(all_timed_hashes.items())
            if len(users) > 1
        },
        "manifests": manifests,
        "stopping_layout_index_sha256": stopping_index_sha,
        "capture_index_sha256": capture_index_sha,
        "locked_implementation": locked,
        "generation": {
            "generator": str(pathlib.Path(__file__).resolve()),
            "generator_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
        },
    }
    index_path = out_dir / "index.json"
    if index_path.exists():
        raise FileExistsError("refusing to overwrite scan index")
    staging = index_path.with_suffix(".json.tmp")
    write_json_atomic(staging, index)
    staging.replace(index_path)
    print(f"\nscan index: {index_path}")
    print(f"index_sha256={sha256_file(index_path)}")
    print(f"total_layouts={index['counts']['total_layouts']} unique_timed_maps={index['counts']['unique_timed_map_hashes']}")


if __name__ == "__main__":
    main()
