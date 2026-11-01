#!/usr/bin/env python3
"""Setup-only deterministic calibration/evaluation split of the frozen workload.

Rule (frozen, content-based, uses no performance or timing information):
  for line index i (0-based) with raw line bytes L (excluding the trailing
  newline), h = SHA256(b"fullservice-v2-split-v1|" + str(i) + b"|" + L);
  the request is CALIBRATION iff int.from_bytes(h[:8], "big") < 2**63,
  otherwise EVALUATION. Expected 50/50; the realized counts are recorded.

The two halves are disjoint by construction (every source line lands in exactly
one half) and together exhaust the source file. Line order within each half is
the source order. The source file is only read, never modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib

from prediction_common import sha256_file, write_json_atomic

SPLIT_TAG = b"fullservice-v2-split-v1"
SOURCE = pathlib.Path(
    "/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl"
)
EXPECTED_SOURCE_SHA256 = (
    "e2a30dee2901f7af53e39eb61abd70461a3bc8a0cea0d900f056c57f0feac343"
)
EXPECTED_SOURCE_LINES = 16000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(SOURCE)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"source workload hash differs: {source_sha}")

    calib_path = out / "calibration.jsonl"
    eval_path = out / "evaluation.jsonl"
    n_calib = n_eval = 0
    with SOURCE.open("rb") as src, calib_path.open("wb") as calib, eval_path.open("wb") as ev:
        index = 0
        for raw in src:
            line = raw.rstrip(b"\n")
            if not line:
                raise SystemExit(f"empty line at index {index}")
            h = hashlib.sha256(SPLIT_TAG + b"|" + str(index).encode() + b"|" + line).digest()
            if int.from_bytes(h[:8], "big") < 2**63:
                calib.write(raw if raw.endswith(b"\n") else raw + b"\n")
                n_calib += 1
            else:
                ev.write(raw if raw.endswith(b"\n") else raw + b"\n")
                n_eval += 1
            index += 1
    if index != EXPECTED_SOURCE_LINES:
        raise SystemExit(f"source line count differs: {index}")
    if n_calib + n_eval != index:
        raise SystemExit("split does not exhaust the source")

    manifest = {
        "schema_version": 1,
        "study_id": "h20-placement-fullservice-v2",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_uses_performance_timing": False,
        "rule": "sha256(fullservice-v2-split-v1|index|line)[:8] big-endian < 2**63 -> calibration, else evaluation",
        "source": {
            "path": str(SOURCE),
            "bytes": SOURCE.stat().st_size,
            "sha256": source_sha,
            "lines": index,
        },
        "calibration": {
            "path": str(calib_path),
            "bytes": calib_path.stat().st_size,
            "sha256": sha256_file(calib_path),
            "requests": n_calib,
        },
        "evaluation": {
            "path": str(eval_path),
            "bytes": eval_path.stat().st_size,
            "sha256": sha256_file(eval_path),
            "requests": n_eval,
        },
        "disjoint_and_exhaustive": True,
        "generator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
    }
    write_json_atomic(out / "split_manifest.json", manifest)
    print(f"calibration: {n_calib} requests, sha256={manifest['calibration']['sha256']}")
    print(f"evaluation:  {n_eval} requests, sha256={manifest['evaluation']['sha256']}")


if __name__ == "__main__":
    main()
