#!/usr/bin/env python3
"""Freeze disjoint screening, calibration, and evaluation request batches."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

from transformers import AutoTokenizer


BATCH_SPECS = (
    ("screening-01", "screening"),
    ("calibration-01", "calibration"),
    ("calibration-02", "calibration"),
    ("calibration-03", "calibration"),
    ("evaluation-01", "evaluation"),
    ("evaluation-02", "evaluation"),
    ("evaluation-03", "evaluation"),
)
MODEL_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require_absent(paths: list[pathlib.Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    temporary = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    stale = [str(path) for path in temporary if path.exists()]
    if stale:
        raise FileExistsError(f"stale temporary outputs exist: {stale}")


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--e1-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--requests-per-batch", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests_per_batch <= 0:
        raise ValueError("requests per batch must be positive")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("invalid token-length interval")

    e1_manifest = json.loads(args.e1_manifest.read_text())
    if e1_manifest.get("schema_version") != 1:
        raise ValueError("unsupported E1 request-manifest schema")
    if sha256(args.dataset) != e1_manifest["dataset"]["sha256"]:
        raise ValueError("dataset hash differs from the E1 frozen dataset")
    e1_line_ids = {
        int(item["source_line_index_zero_based"])
        for item in e1_manifest["requests"]
    }
    if len(e1_line_ids) != len(e1_manifest["requests"]):
        raise ValueError("E1 request manifest contains duplicate source lines")

    output_paths = [args.output_dir / f"{name}.jsonl" for name, _ in BATCH_SPECS]
    require_absent([*output_paths, args.manifest_output])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    wanted = len(BATCH_SPECS) * args.requests_per_batch
    selected: list[tuple[int, dict[str, Any], int]] = []
    with args.dataset.open() as handle:
        for line_index, line in enumerate(handle):
            if line_index in e1_line_ids:
                continue
            request = json.loads(line)
            token_count = len(
                tokenizer.encode(request["prompt"], add_special_tokens=True)
            )
            if args.min_tokens <= token_count <= args.max_tokens:
                selected.append((line_index, request, token_count))
                if len(selected) == wanted:
                    break
    if len(selected) != wanted:
        raise ValueError(
            f"found only {len(selected)} of {wanted} disjoint requests in interval"
        )

    batches: list[dict[str, Any]] = []
    all_new_lines: set[int] = set()
    for batch_index, ((batch_id, split), output_path) in enumerate(
        zip(BATCH_SPECS, output_paths, strict=True)
    ):
        begin = batch_index * args.requests_per_batch
        items = selected[begin : begin + args.requests_per_batch]
        request_lines = []
        request_records = []
        for within_batch_index, (line_index, request, token_count) in enumerate(items):
            if line_index in all_new_lines:
                raise ValueError(f"duplicate selected source line {line_index}")
            all_new_lines.add(line_index)
            frozen = {
                "id": request["id"],
                "prompt": request["prompt"],
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": bool(request.get("ignore_eos", False)),
            }
            request_lines.append(json.dumps(frozen, ensure_ascii=False))
            request_records.append(
                {
                    "within_batch_index": within_batch_index,
                    "source_line_index_zero_based": line_index,
                    "id": request["id"],
                    "token_count": token_count,
                }
            )
        write_text_atomic(output_path, "\n".join(request_lines) + "\n")
        batches.append(
            {
                "batch_index": batch_index,
                "batch_id": batch_id,
                "split": split,
                "requests": request_records,
                "request_count": len(request_records),
                "total_input_tokens": sum(item[2] for item in items),
                "request_file": file_record(output_path),
            }
        )

    if all_new_lines & e1_line_ids:
        raise AssertionError("new request split overlaps E1")
    split_lines = {
        split: {
            request["source_line_index_zero_based"]
            for batch in batches
            if batch["split"] == split
            for request in batch["requests"]
        }
        for split in ("screening", "calibration", "evaluation")
    }
    if any(
        split_lines[left] & split_lines[right]
        for index, left in enumerate(split_lines)
        for right in list(split_lines)[index + 1 :]
    ):
        raise AssertionError("prediction-study splits overlap")

    model_files = {
        name: file_record(args.model / name)
        for name in MODEL_FILES
        if (args.model / name).exists()
    }
    manifest = {
        "schema_version": 1,
        "study_id": "h20-placement-prediction-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_before_screening_route_profile",
        "selection_rule": {
            "description": (
                "scan the hash-identified RealProver JSONL in source-line order; "
                "exclude every E1 source line; retain the first 56 remaining "
                "requests whose local tokenizer length with special tokens is "
                "500 through 512 inclusive; assign consecutive groups of eight "
                "to the predeclared screening, calibration, and locked-evaluation "
                "batch order"
            ),
            "route_or_performance_information_used": False,
            "batch_order": [name for name, _ in BATCH_SPECS],
            "requests_per_batch": args.requests_per_batch,
            "min_tokens_inclusive": args.min_tokens,
            "max_tokens_inclusive": args.max_tokens,
        },
        "dataset": file_record(args.dataset),
        "model": {
            "path": str(args.model.resolve()),
            "files": model_files,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "add_special_tokens": True,
        },
        "e1_exclusion": {
            "request_manifest": file_record(args.e1_manifest),
            "source_line_indices_zero_based": sorted(e1_line_ids),
        },
        "batches": batches,
        "split_counts": {
            split: {
                "batches": sum(batch["split"] == split for batch in batches),
                "requests": len(lines),
            }
            for split, lines in split_lines.items()
        },
        "disjointness": {
            "all_56_new_source_lines_unique": len(all_new_lines) == wanted,
            "new_source_lines_disjoint_from_e1": not (all_new_lines & e1_line_ids),
            "screening_calibration_evaluation_pairwise_disjoint": True,
        },
    }
    write_text_atomic(
        args.manifest_output,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"frozen {len(batches)} batches and {len(all_new_lines)} disjoint requests; "
        f"manifest_sha256={sha256(args.manifest_output)}"
    )


if __name__ == "__main__":
    main()
