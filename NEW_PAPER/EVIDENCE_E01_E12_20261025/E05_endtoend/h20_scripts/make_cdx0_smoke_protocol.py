#!/usr/bin/env python3
"""Build the CD-X0 pre-freeze mechanics smoke protocol (SETUP artifact).

Two smoke starts on real evaluation captures, chosen BEFORE any replay timing
and independent of any measurement: the smallest-row unit (prover layer 024)
and the largest-row unit (code layer 071, 8x1024 rows = worst case for DeepEP
dispatch and DeepGEMM shapes). Reduced timing parameters; outputs land in
results/cdx0_v1/smoke/ and are never indexed as campaign results, never used
for any selection, layout, or verdict decision. The real execution protocol is
frozen only after both smokes pass.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from cdx0_common import STUDY_ID, sha256_file, write_json_atomic  # noqa: E402

COLUMNS = ("identity", "base-of-sharegpt", "base-of-prover", "base-of-book", "base-of-code")
SMOKE_UNITS = (
    ("cdx0-smoke-prover-024", "cdx0-evaluation-prover-layer-024", 0),
    ("cdx0-smoke-code-071", "cdx0-evaluation-code-layer-071", 1),
)
TIMING = {
    "initial_warmup_forwards_per_layout": 2,
    "activation_warmup_forwards": 2,
    "timed_forwards_per_activation": 5,
    "blocks_per_start": 2,
    "diagnostic_blocks_per_start": 0,
    "empty_fence_warmup": 2,
    "empty_fence_repetitions": 5,
    "schedule_algorithm": "smoke rotation by (start_index + block_index) mod 5",
    "primary_metric": "completion_fenced_us_per_forward",
    "diagnostic_pass": False,
}


def file_record(path: pathlib.Path) -> dict:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def rotation(start_index: int, block: int) -> list[str]:
    k = (start_index + block) % len(COLUMNS)
    return list(COLUMNS[k:] + COLUMNS[:k])


def main() -> None:
    out = ROOT / "results/cdx0_v1/smoke/smoke_protocol.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite smoke protocol: {out}")
    capture_index = ROOT / "inputs/cdx0_v1/capture_index.json"
    index = json.loads(capture_index.read_text())
    units = {u["capture_id"]: u for u in index["capture_units"]}
    layout_index = json.loads((ROOT / "inputs/cdx0_v1/layouts/index.json").read_text())
    manifests = {m["capture_id"]: m for m in layout_index["manifests"]}
    tolerance = json.loads((ROOT / "configs/followup_v1/prediction_tolerance_v2.json").read_text())
    starts = []
    for start_id, capture_id, start_index in SMOKE_UNITS:
        unit = units[capture_id]
        manifest = manifests[capture_id]
        seed = int.from_bytes(
            hashlib.sha256(f"{STUDY_ID}|cdx0-smoke|{capture_id}".encode()).digest()[:8], "big"
        ) % 2**62
        starts.append(
            {
                "start_id": start_id,
                "start_index": start_index,
                "capture_id": capture_id,
                "batch_id": unit["batch_id"],
                "layer_id": unit["layer_id"],
                "layout_manifest": {
                    "path": manifest["path"],
                    "bytes": manifest["bytes"],
                    "sha256": manifest["sha256"],
                },
                "random_seed": seed,
                "primary_schedules": [rotation(start_index, b) for b in range(TIMING["blocks_per_start"])],
                "output_relative_path": f"results/cdx0_v1/smoke/raw/{start_id}.json",
                "log_relative_path": f"results/cdx0_v1/smoke/logs/{start_id}.log",
            }
        )
    protocol = {
        "schema_version": 1,
        "protocol_id": "h20-placement-cdx0-smoke-v1",
        "study_id": STUDY_ID,
        "status": "frozen_before_cdx0_timing",
        "split": "cdx0",
        "purpose": (
            "pre-freeze mechanics smoke: harness load, tier-1/tier-2 correctness gates, "
            "DeepEP dispatch and DeepGEMM grouped-GEMM shapes at the real evaluation row "
            "counts (worst case 8x1024), timing machinery, output schema"
        ),
        "selection_uses_performance_timing": False,
        "values_not_used_for_any_decision": True,
        "outputs_never_indexed_as_campaign_results": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capture_index": file_record(capture_index),
        "implementation": {
            "cdx0_replay_record": file_record(SCRIPT_DIR / "cdx0_replay.py"),
            "cdx0_common_record": file_record(SCRIPT_DIR / "cdx0_common.py"),
            "replay_placement_record": file_record(SCRIPT_DIR / "replay_placement.py"),
        },
        "correctness_gate": tolerance["correctness_gate"],
        "timing": TIMING,
        "starts": starts,
    }
    write_json_atomic(out, protocol)
    print(f"wrote {out}")
    print(f"smoke_protocol_sha256={sha256_file(out)}")


if __name__ == "__main__":
    main()
