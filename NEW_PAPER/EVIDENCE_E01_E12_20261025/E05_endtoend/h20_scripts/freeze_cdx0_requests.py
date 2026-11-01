#!/usr/bin/env python3
"""Freeze the CD-X0 capture request batches (SETUP; no routing/performance info).

Rule (pre-registered in configs/followup_v1/cdx0_amendment_v1.json):
per (split, domain) scan the segment FORWARD in source-line order from the
segment midpoint (inclusive); retain requests whose Qwen3-tokenized prompt
length (add_special_tokens=True) lies in [400, 2048]; if fewer than 8 by the
segment end, continue BACKWARD from midpoint-1 to segment start; take the
first 8 in scan order. Identical rule for all domains and both splits.
Refuses to overwrite anything.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WL = ROOT / "inputs/fullservice_v2/workload"
MODEL = "/root/models/ms_cache/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master"
OUT_MANIFEST = ROOT / "configs/followup_v1/cdx0_requests.json"
OUT_BATCHES = ROOT / "inputs/cdx0_v1/batches"
BAND = (400, 2048)
N_PER_BATCH = 8
SEG = {
    "evaluation": {"sharegpt": (0, 1925), "prover": (1925, 3948), "book": (3948, 5932), "code": (5932, 7963)},
    "calibration": {"sharegpt": (0, 2075), "prover": (2075, 4052), "book": (4052, 6068), "code": (6068, 8037)},
}
DOMAIN_ORDER = ("sharegpt", "prover", "book", "code")

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_record(path: pathlib.Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}

def main() -> None:
    if OUT_MANIFEST.exists() or OUT_BATCHES.exists():
        raise SystemExit(f"refusing to overwrite: {OUT_MANIFEST} / {OUT_BATCHES}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    batches = []
    for split in ("calibration", "evaluation"):
        src = WL / f"{split}.jsonl"
        with src.open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        lens = [len(x) for x in tok([r["prompt"] for r in rows], add_special_tokens=True)["input_ids"]]
        for domain in DOMAIN_ORDER:
            a, b = SEG[split][domain]
            assert b - a == len(rows[a:b])
            for i in range(a, b):
                assert rows[i]["id"].rsplit("_", 1)[0] == domain, f"domain/segment mismatch at {split}:{i}"
            mid = (a + b) // 2
            inband_fwd = [i for i in range(mid, b) if BAND[0] <= lens[i] <= BAND[1]]
            inband_bwd = [i for i in range(mid - 1, a - 1, -1) if BAND[0] <= lens[i] <= BAND[1]]
            sel = (inband_fwd + inband_bwd)[:N_PER_BATCH]
            if len(sel) != N_PER_BATCH:
                raise SystemExit(f"{split}/{domain}: only {len(sel)} in-band requests")
            batch_id = f"cdx0-{split}-{domain}"
            req_path = OUT_BATCHES / f"{batch_id}.jsonl"
            req_path.parent.mkdir(parents=True, exist_ok=True)
            if req_path.exists():
                raise SystemExit(f"refusing to overwrite {req_path}")
            with req_path.open("w") as f:
                for i in sel:
                    f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
            batches.append({
                "batch_id": batch_id, "split": split, "domain": domain,
                "segment_zero_based_half_open": [a, b], "segment_midpoint": mid,
                "band_inclusive": list(BAND),
                "request_file": file_record(req_path),
                "total_input_tokens": sum(lens[i] for i in sel),
                "requests": [
                    {"id": rows[i]["id"], "source_line_index_zero_based": i,
                     "token_count": lens[i], "within_batch_index": k}
                    for k, i in enumerate(sel)],
            })
            print(f"{batch_id}: lines={sel} tokens={[lens[i] for i in sel]} sum={sum(lens[i] for i in sel)}", flush=True)
    manifest = {
        "schema_version": 1,
        "study_id": "h20-placement-cdx0-v1",
        "status": "frozen_before_cdx0_capture",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "amendment": file_record(ROOT / "configs/followup_v1/cdx0_amendment_v1.json"),
        "datasets": {s: file_record(WL / f"{s}.jsonl") for s in ("calibration", "evaluation")},
        "model": {"path": MODEL},
        "tokenizer": {"class": type(tok).__name__, "add_special_tokens": True},
        "selection_rule": {
            "description": "forward scan from segment midpoint (inclusive) for tokenized length in [400,2048] (add_special_tokens=True), then backward from midpoint-1 if needed; first 8 in scan order; identical rule for all domains/splits; no routing or performance information used",
            "requests_per_batch": N_PER_BATCH, "band_inclusive": list(BAND),
            "route_or_performance_information_used": False,
        },
        "segment_boundaries_zero_based_half_open": SEG,
        "disjointness": {
            "calibration_evaluation_files_disjoint": True,
            "domain_segments_pairwise_disjoint_within_split": True,
            "no_prior_frozen_capture_batch_reuse": "prior prediction_v1 batches came from a different dataset (L512_O1_realprover_n8192.jsonl)",
        },
        "batches": batches,
    }
    tmp = OUT_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT_MANIFEST)
    print("wrote", OUT_MANIFEST)

if __name__ == "__main__":
    main()
