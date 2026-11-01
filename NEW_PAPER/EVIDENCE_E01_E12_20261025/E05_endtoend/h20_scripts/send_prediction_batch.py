#!/usr/bin/env python3
"""Send one hash-bound prediction batch and retain complete server responses."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import pathlib
import time
from typing import Any

import aiohttp


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=pathlib.Path, required=True)
    parser.add_argument("--request-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--trigger", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:30000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


async def send_one(
    session: aiohttp.ClientSession,
    api: str,
    model: str,
    request: dict[str, Any],
    release: asyncio.Event,
) -> dict[str, Any]:
    await release.wait()
    started = time.perf_counter()
    payload = {
        "model": model,
        "prompt": request["prompt"],
        "max_tokens": int(request.get("max_tokens", 1)),
        "temperature": float(request.get("temperature", 0)),
        "ignore_eos": bool(request.get("ignore_eos", False)),
        "stream": False,
    }
    try:
        async with session.post(api, json=payload) as response:
            body = await response.text()
            try:
                parsed_body: Any = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = body
            return {
                "id": request["id"],
                "status": response.status,
                "elapsed_s": time.perf_counter() - started,
                "response": parsed_body,
            }
    except Exception as exc:
        return {
            "id": request["id"],
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }


async def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite response artifact: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale response temporary exists: {temporary}")
    manifest = json.loads(args.request_manifest.read_text())
    if manifest.get("study_id") != "h20-placement-prediction-v1":
        raise ValueError("request manifest has the wrong study ID")
    batches = [
        batch for batch in manifest["batches"]
        if batch["batch_id"] == args.batch_id
    ]
    if len(batches) != 1:
        raise ValueError(f"unknown or duplicate batch ID {args.batch_id!r}")
    batch = batches[0]
    if pathlib.Path(batch["request_file"]["path"]).resolve() != args.requests.resolve():
        raise ValueError("request path differs from the frozen batch")
    if sha256(args.requests) != batch["request_file"]["sha256"]:
        raise ValueError("request hash differs from the frozen batch")
    trigger = json.loads(args.trigger.read_text())
    if trigger.get("study_id") != "h20-placement-prediction-v1":
        raise ValueError("capture trigger has the wrong study ID")
    if trigger["capture_context"]["batch_id"] != args.batch_id:
        raise ValueError("capture trigger is armed for another batch")
    if trigger["capture_context"]["request_file"]["sha256"] != sha256(args.requests):
        raise ValueError("capture trigger has another request-file hash")
    requests = [
        json.loads(line)
        for line in args.requests.read_text().splitlines()
        if line.strip()
    ]
    if len(requests) != batch["request_count"]:
        raise ValueError("request count differs from the frozen batch")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(
        limit=len(requests) + 4,
        limit_per_host=len(requests) + 4,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        health = args.api.rsplit("/v1", 1)[0] + "/health"
        async with session.get(health) as response:
            await response.read()
            if response.status != 200:
                raise RuntimeError(f"health endpoint returned {response.status}")
        release = asyncio.Event()
        tasks = [
            asyncio.create_task(
                send_one(session, args.api, args.model, request, release)
            )
            for request in requests
        ]
        batch_started = time.perf_counter()
        release.set()
        responses = await asyncio.gather(*tasks)
        batch_elapsed = time.perf_counter() - batch_started

    failed = [
        item
        for item in responses
        if item.get("status") != 200 or "error" in item
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api": args.api,
        "model": str(pathlib.Path(args.model).resolve()),
        "study_id": "h20-placement-prediction-v1",
        "capture_id": trigger["capture_id"],
        "batch_id": args.batch_id,
        "request_manifest": {
            "path": str(args.request_manifest.resolve()),
            "bytes": args.request_manifest.stat().st_size,
            "sha256": sha256(args.request_manifest),
        },
        "request_file": {
            "path": str(args.requests.resolve()),
            "bytes": args.requests.stat().st_size,
            "sha256": sha256(args.requests),
        },
        "capture_trigger": {
            "path": str(args.trigger.resolve()),
            "bytes": args.trigger.stat().st_size,
            "sha256": sha256(args.trigger),
            "mode": trigger["mode"],
            "target_layer": trigger["target_layer"],
        },
        "concurrency": len(requests),
        "batch_elapsed_s": batch_elapsed,
        "responses": responses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(responses)} requests failed")
    print(f"captured {len(responses)} responses in {batch_elapsed:.3f} s")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
