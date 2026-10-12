#!/usr/bin/env python3
"""Freeze a deterministic real-request batch near the target token count."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

from transformers import AutoTokenizer


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("invalid token-length interval")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    selected: list[tuple[int, dict[str, Any], int]] = []
    with args.dataset.open() as handle:
        for line_index, line in enumerate(handle):
            request = json.loads(line)
            token_count = len(
                tokenizer.encode(request["prompt"], add_special_tokens=True)
            )
            if args.min_tokens <= token_count <= args.max_tokens:
                selected.append((line_index, request, token_count))
                if len(selected) == args.count:
                    break
    if len(selected) != args.count:
        raise ValueError(
            f"found only {len(selected)} requests in the frozen token interval"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for _, request, _ in selected:
            frozen = {
                "id": request["id"],
                "prompt": request["prompt"],
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": bool(request.get("ignore_eos", False)),
            }
            handle.write(json.dumps(frozen, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    model_files = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        path = args.model / name
        if path.exists():
            model_files[name] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "requests_frozen_route_not_yet_captured",
        "selection_rule": {
            "description": (
                "scan source JSONL in file order and take the first eight "
                "requests whose local tokenizer length with special tokens is "
                "between 500 and 512 inclusive"
            ),
            "count": args.count,
            "min_tokens_inclusive": args.min_tokens,
            "max_tokens_inclusive": args.max_tokens,
            "route_or_timing_information_used": False,
        },
        "dataset": {
            "path": str(args.dataset.resolve()),
            "bytes": args.dataset.stat().st_size,
            "sha256": sha256(args.dataset),
        },
        "model": {
            "path": str(args.model.resolve()),
            "files": model_files,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "add_special_tokens": True,
        },
        "requests": [
            {
                "batch_index": batch_index,
                "source_line_index_zero_based": line_index,
                "id": request["id"],
                "token_count": token_count,
            }
            for batch_index, (line_index, request, token_count) in enumerate(selected)
        ],
        "total_input_tokens": sum(item[2] for item in selected),
        "request_file": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
    }
    write_json_atomic(args.manifest_output, manifest)
    print(
        f"frozen {len(selected)} requests, total_input_tokens="
        f"{manifest['total_input_tokens']}"
    )


if __name__ == "__main__":
    main()
