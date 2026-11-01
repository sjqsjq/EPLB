#!/usr/bin/env python3
"""Validate and hash-index the 16 CD-X0 capture units (setup; refuses overwrite)."""
from __future__ import annotations
import argparse, datetime as dt, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from cdx0_common import (STUDY_ID, TOP_K, WORLD_SIZE, NUM_EXPERTS, ALIGNMENT,
                         sha256_file, write_json_atomic)

LAYERS = (24, 71)

def file_record(path: pathlib.Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}

def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())

def parse_chunked_prefill_size(log_path: pathlib.Path) -> int:
    """Read the server's own chunked_prefill_size from its log; refuse ambiguity."""
    found = set()
    with log_path.open("r", errors="replace") as handle:
        for line in handle:
            found.update(int(m) for m in re.findall(r"chunked_prefill_size=(\d+)", line))
    if len(found) != 1:
        raise ValueError(f"server log does not record exactly one chunked_prefill_size: {sorted(found)}")
    return found.pop()

def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", type=pathlib.Path, default=root / "inputs/cdx0_v1/captures")
    ap.add_argument("--weight-root", type=pathlib.Path, default=root / "inputs/cdx0_v1/expert_weights")
    ap.add_argument("--response-root", type=pathlib.Path, default=root / "results/cdx0_v1/capture_responses")
    ap.add_argument("--trigger-root", type=pathlib.Path, default=root / "inputs/cdx0_v1/triggers")
    ap.add_argument("--request-manifest", type=pathlib.Path, default=root / "configs/followup_v1/cdx0_requests.json")
    ap.add_argument("--identity-map-sha256", required=True)
    ap.add_argument("--server-log", type=pathlib.Path, required=True,
                    help="capture server log; chunked_prefill_size is parsed from it, never trusted")
    ap.add_argument("--output", type=pathlib.Path, default=root / "inputs/cdx0_v1/capture_index.json")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    manifest = load_json(args.request_manifest)
    if manifest.get("study_id") != STUDY_ID or manifest.get("status") != "frozen_before_cdx0_capture":
        raise ValueError("request manifest is not the frozen CD-X0 manifest")
    chunk = parse_chunked_prefill_size(args.server_log)
    unit_records, n_forward = [], 0
    for batch in manifest["batches"]:
        batch_id, split, domain = batch["batch_id"], batch["split"], batch["domain"]
        # Documented capture mechanism: the server ran with its default chunked
        # prefill under the frozen launcher flags, so the captured forward at the
        # target layer is the first prefill chunk of exactly one frozen request
        # per dp rank: rows = min(token_count, chunked_prefill_size). The check
        # remains exact; a second-chunk or mixed capture would fail it.
        expected_multiset = sorted(min(int(r["token_count"]), chunk) for r in batch["requests"])
        expected_total_rows = sum(expected_multiset)
        for layer_id in LAYERS:
            base_id = f"cdx0-{split}-{domain}-layer-{layer_id:03d}"
            # setup-retry resolution (amendment_5): prefer complete base attempt, else single -r2
            capture_id, attempt = None, None
            for cand, att in ((base_id, 1), (base_id + "-r2", 2)):
                t = args.trigger_root / f"{cand}.json"
                r = args.response_root / f"{cand}.json"
                d = args.capture_root / batch_id / (f"layer-{layer_id:03d}" + ("" if att == 1 else "-r2"))
                if t.is_file() and r.is_file() and all((d / f"forward_rank{k}.pt").is_file() for k in range(WORLD_SIZE)):
                    capture_id, attempt = cand, att
            if capture_id is None:
                capture_id, attempt = base_id, 1
            unit_dir = args.capture_root / batch_id / (f"layer-{layer_id:03d}" + ("" if attempt == 1 else "-r2"))
            trigger_path = args.trigger_root / f"{capture_id}.json"
            response_path = args.response_root / f"{capture_id}.json"
            trigger, response = load_json(trigger_path), load_json(response_path)
            if trigger.get("capture_id") != capture_id or trigger.get("study_id") != STUDY_ID:
                raise ValueError(f"trigger mismatch for {capture_id}")
            if response.get("capture_id") != capture_id or response.get("study_id") != STUDY_ID:
                raise ValueError(f"response mismatch for {capture_id}")
            if response["capture_trigger"]["sha256"] != sha256_file(trigger_path):
                raise ValueError(f"response trigger hash mismatch for {capture_id}")
            responses = response.get("responses", [])
            if len(responses) != len(batch["requests"]) or any(r.get("status") != 200 or "error" in r for r in responses):
                raise ValueError(f"incomplete server responses for {capture_id}")
            forward_records, hidden_rows = [], []
            for rank in range(WORLD_SIZE):
                path = unit_dir / f"forward_rank{rank}.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                header = {"schema_version": 2, "study_id": STUDY_ID, "capture_id": capture_id,
                          "mode": "snapshot_split", "rank": rank, "layer_id": layer_id,
                          "identity_map_required": True,
                          "identity_map_sha256": args.identity_map_sha256,
                          "weight_block_size": [ALIGNMENT, ALIGNMENT]}
                mism = {k: (payload.get(k), v) for k, v in header.items() if payload.get(k) != v}
                if mism:
                    raise ValueError(f"forward header mismatch at {path}: {mism}")
                ctx = payload.get("capture_context", {})
                if (ctx.get("split"), ctx.get("batch_id"), ctx.get("domain")) != (split, batch_id, domain):
                    raise ValueError(f"forward context mismatch at {path}")
                if ctx.get("total_input_tokens") != batch["total_input_tokens"]:
                    raise ValueError(f"forward context token total mismatch at {path}")
                rows = int(payload["hidden_rows"])
                for name, dt_expect in (("hidden_states", torch.bfloat16), ("logical_topk_ids", torch.int32),
                                          ("logical_topk_weights", torch.float32), ("routed_output", torch.bfloat16)):
                    t = payload.get(name)
                    if not isinstance(t, torch.Tensor):
                        raise TypeError(f"missing tensor {name} at {path}")
                    exp_shape = {"hidden_states": (rows, 4096), "logical_topk_ids": (rows, TOP_K),
                                 "logical_topk_weights": (rows, TOP_K), "routed_output": (rows, 4096)}[name]
                    if tuple(t.shape) != exp_shape or not t.is_contiguous() or t.dtype != dt_expect:
                        raise ValueError(f"tensor {name} shape/dtype/contiguity mismatch at {path}")
                    tf = t.to(torch.float32) if t.dtype != torch.int32 else None
                    if tf is not None and not bool(torch.isfinite(tf).all()):
                        raise ValueError(f"non-finite {name} at {path}")
                ids = payload["logical_topk_ids"].to(torch.int64)
                if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS):
                    raise ValueError(f"route id range violation at {path}")
                sids = ids.sort(dim=1).values
                if ids.numel() and bool((sids[:, 1:] == sids[:, :-1]).any()):
                    raise ValueError(f"duplicate expert in a top-k row at {path}")
                hidden_rows.append(rows)
                forward_records.append(file_record(path))
                n_forward += 1
            if sorted(hidden_rows) != expected_multiset:
                raise ValueError(f"{capture_id}: hidden-row multiset {sorted(hidden_rows)} != request tokens {expected_multiset}")
            if sum(hidden_rows) != expected_total_rows:
                raise ValueError(f"{capture_id}: row sum {sum(hidden_rows)} != chunk-adjusted total {expected_total_rows}")
            unit_records.append({
                "capture_id": capture_id, "setup_attempt": attempt,
                "batch_id": batch_id, "split": split, "domain": domain,
                "layer_id": layer_id,
                "hidden_rows_by_source_rank": hidden_rows,
                "total_hidden_rows": sum(hidden_rows),
                "total_assignments": sum(hidden_rows) * TOP_K,
                "total_input_tokens_expected": batch["total_input_tokens"],
                "domain_purity": ("verified_by_construction (all source lines inside the named "
                                    "domain segment; row multiset equals min(request token_count, "
                                    "chunked_prefill_size) per request - captured forward is the "
                                    "first prefill chunk of exactly one frozen request per rank)"),
                "chunked_prefill_size": chunk,
                "expected_rows_per_request": expected_multiset,
                "forward_files": forward_records,
                "trigger": file_record(trigger_path),
                "response": file_record(response_path),
                "capture_context": {k: batch[k] for k in ("segment_zero_based_half_open",)} | {
                    "request_source_lines_zero_based": [r["source_line_index_zero_based"] for r in batch["requests"]],
                    "request_token_counts": [r["token_count"] for r in batch["requests"]],
                },
            })
            print(f"validated {capture_id}: rows={hidden_rows} sum={sum(hidden_rows)}", flush=True)
    expert_records = []
    for layer_id in LAYERS:
        for rank in range(WORLD_SIZE):
            path = args.weight_root / f"layer-{layer_id:03d}" / f"expert_rank{rank}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            header = {"schema_version": 2, "study_id": STUDY_ID,
                      "artifact_type": "immutable_layer_expert_weights", "rank": rank, "layer_id": layer_id,
                      "identity_map_required": True, "identity_map_sha256": args.identity_map_sha256,
                      "logical_expert_ids": list(range(rank * 16, (rank + 1) * 16)),
                      "weight_block_size": [ALIGNMENT, ALIGNMENT]}
            mism = {k: (payload.get(k), v) for k, v in header.items() if payload.get(k) != v}
            if mism:
                raise ValueError(f"expert header mismatch at {path}: {mism}")
            expert_records.append({"layer_id": layer_id, "rank": rank, **file_record(path)})
    if len(unit_records) != 16 or n_forward != 128 or len(expert_records) != 16:
        raise ValueError(f"count mismatch: units={len(unit_records)} forwards={n_forward} experts={len(expert_records)}")
    index = {
        "schema_version": 1, "study_id": STUDY_ID,
        "status": "all_cdx0_capture_inputs_validated_and_hash_indexed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_manifest": file_record(args.request_manifest),
        "capture_mechanism": {
            "chunked_prefill_size": chunk,
            "chunked_prefill_size_source": file_record(args.server_log),
            "row_semantics": ("each captured forward is the first prefill chunk of exactly one "
                               "frozen request at the target layer; rows per rank = "
                               "min(token_count, chunked_prefill_size); this is the sglang server "
                               "default under the frozen launcher flags, identical to all prior "
                               "runs in this lineage including the five-arm service campaign"),
            "band_and_selection_unchanged": True,
            "frozen_request_totals_preserved_in": "capture_context.request_token_counts and total_input_tokens_expected",
        },
        "counts": {"capture_units": 16, "forward_rank_files": n_forward, "expert_rank_files": len(expert_records)},
        "capture_units": unit_records,
        "expert_files": expert_records,
    }
    write_json_atomic(args.output, index)
    print(f"validated {len(unit_records)} units, {n_forward} forward files, {len(expert_records)} expert files; sha={sha256_file(args.output)}")

if __name__ == "__main__":
    main()
