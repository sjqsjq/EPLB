#!/usr/bin/env python3
"""Atomically arm one CD-X0 frozen-batch snapshot capture (setup only)."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, pathlib
from typing import Any

STUDY_ID = "h20-placement-cdx0-v1"
ALLOWED_LAYERS = (24, 71)

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-id", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--request-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--trigger", type=pathlib.Path, required=True)
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--weight-output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--identity-map", type=pathlib.Path, required=True)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--min-hidden-rows", type=int, default=400)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    if args.target_layer not in ALLOWED_LAYERS:
        raise ValueError(f"CD-X0 layers are {ALLOWED_LAYERS}, got {args.target_layer}")
    if args.min_hidden_rows <= 0:
        raise ValueError("min hidden rows must be positive")
    manifest = json.loads(args.request_manifest.read_text())
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError("request manifest has the wrong study ID")
    if manifest.get("status") != "frozen_before_cdx0_capture":
        raise ValueError("request manifest is not frozen")
    matches = [b for b in manifest["batches"] if b["batch_id"] == args.batch_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate batch {args.batch_id!r}")
    batch = matches[0]
    request_file = pathlib.Path(batch["request_file"]["path"])
    if file_record(request_file) != batch["request_file"]:
        raise ValueError("request file differs from the frozen manifest")
    split = batch["split"]
    if split not in ("calibration", "evaluation"):
        raise ValueError(f"unexpected split {split!r}")
    derived = f"cdx0-{split}-{batch['domain']}-layer-{args.target_layer:03d}"
    if args.capture_id not in (derived, derived + "-r2"):
        raise ValueError(f"capture id {args.capture_id!r} does not match frozen naming (base or single -r2 setup retry)")
    capture_context = {
        "cdx0_request_manifest": file_record(args.request_manifest),
        "amendment": manifest["amendment"],
        "batch_id": batch["batch_id"],
        "split": split,
        "domain": batch["domain"],
        "request_file": batch["request_file"],
        "request_source_lines_zero_based": [r["source_line_index_zero_based"] for r in batch["requests"]],
        "request_token_counts": [r["token_count"] for r in batch["requests"]],
        "total_input_tokens": batch["total_input_tokens"],
        "segment_zero_based_half_open": batch["segment_zero_based_half_open"],
        "dataset": manifest["datasets"][split],
        "model": manifest["model"],
        "selection_rule": manifest["selection_rule"],
        "domain_purity": "verifiable_by_construction: every request line lies inside the named domain segment",
    }
    payload = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "mode": "snapshot_split",
        "capture_id": args.capture_id,
        "output_dir": str(args.output_dir.resolve()),
        "weight_output_dir": str(args.weight_output_dir.resolve()),
        "identity_map": file_record(args.identity_map),
        "identity_map_sha256": sha256(args.identity_map),
        "identity_map_required": True,
        "expected_layers": 94,
        "num_experts": 128,
        "min_hidden_rows": args.min_hidden_rows,
        "target_layer": args.target_layer,
        "armed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capture_context": capture_context,
    }
    expected = [args.output_dir / f"forward_rank{r}.pt" for r in range(8)]
    existing = [str(p) for p in expected if p.exists()]
    if existing:
        raise FileExistsError(f"capture outputs already exist: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.weight_output_dir.mkdir(parents=True, exist_ok=True)
    args.trigger.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.trigger.with_suffix(args.trigger.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"stale trigger temporary exists: {tmp}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.trigger)
    print(json.dumps({k: payload[k] for k in ("capture_id", "target_layer", "min_hidden_rows", "output_dir")}, sort_keys=True))

if __name__ == "__main__":
    main()
