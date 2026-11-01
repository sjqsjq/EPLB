"""Freeze SLO reference from C boots (must run before A/B evaluation)."""
import json, os
G = os.path.dirname(os.path.abspath(__file__))
out = {"frozen_utc": None, "source": {}, "reference": {}}
import datetime
for stream, label in [("E3", "prep3_C1_I_E3"), ("E4", "prep3_C2_I_E4")]:
    prq = f"{G}/boots/{label}/{label}_perreq.jsonl"
    rows = [json.loads(l) for l in open(prq)]
    ok = [r for r in rows if not r.get("error") and r.get("ttft_s") is not None]
    ttfts = sorted(r["ttft_s"] for r in ok)
    p95 = lambda xs: xs[int(0.95 * (len(xs) - 1))]
    ref = {"n_ok": len(ok), "ref_p95_TTFT_s": round(p95(ttfts), 4)}
    if stream == "E4":
        tpots = sorted((r["elapsed_s"] - r["ttft_s"]) / (r["comp_tokens"] - 1)
                       for r in ok if r.get("comp_tokens", 1) > 1)
        ref["ref_p95_TPOT_s"] = round(p95(tpots), 5)
        ref["n_tpot"] = len(tpots)
    out["reference"][stream] = ref
    out["source"][stream] = label
out["frozen_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
out["goodput_def"] = {
    "good_request": "error==null AND TTFT<=k*ref_p95_TTFT AND (E4: TPOT<=k*ref_p95_TPOT)",
    "tpot": "(elapsed_s - ttft_s)/(comp_tokens-1), comp_tokens>1 only",
    "E3_note": "TTFT-only (comp_tokens==1 by stream design, pre-declared)",
    "k_grid": [0.8, 1.0, 1.2], "primary_k": 1.0,
    "goodput_req_s": "GOOD_count / bench_wall_time_s",
    "goodness_rate": "GOOD_count / total_requests"}
json.dump(out, open(f"{G}/SLO_REFERENCE_FROZEN.json", "w"), indent=1)
print("SLO reference frozen:", json.dumps(out["reference"]))
