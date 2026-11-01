"""Pooled sampling-eligibility adjudication over the 4 v2 B boots (locked rule, resume variant).

ADDENDUM (round-2 resume): prep3_B1_P_E3 ran under v1 code and FAILED (NCCL
budget-split deadlock, window#17, kept as failure evidence, never re-run over).
Per governance (failed v1 boot kept, excluded from pooling; new/old not mixed),
the eligibility pool is the four boots executed under v2:
prep3_B1r_P_E3, prep3_B2_P_E3, prep3_B3_P_E4, prep3_B4_P_E4.
Rule itself is UNCHANGED from extract_eligibility.py (frozen protocol):
ELIGIBLE iff pooled S>1 share >= 0.10 AND pooled recording_ratio < 0.039062.
"""
import json, os
G = os.path.dirname(os.path.abspath(__file__))
A3_REF = 0.039062  # frozen from 8 old A3 boots (see PROTOCOL.md)
per_boot, pool = {}, {"steps": 0, "steps_s_gt1": 0, "rec": 0, "true": 0}
for label in ["prep3_B1r_P_E3", "prep3_B2_P_E3", "prep3_B3_P_E4", "prep3_B4_P_E4"]:
    vp = f"{G}/boots/{label}/v2p1_rank0.jsonl"
    if not os.path.exists(vp):
        per_boot[label] = {"error": "v2p1 missing"}; continue
    rows = [json.loads(l) for l in open(vp) if l.strip()]
    steps = sum(r.get("steps_actual", 0) or 0 for r in rows)
    s1 = sum(r.get("steps_actual", 0) or 0 for r in rows if (r.get("sp_s") or 0) > 1)
    rec = sum(r.get("sp_recorded_steps", 0) or 0 for r in rows)
    tru = sum(r.get("true_steps_total", 0) or 0 for r in rows)
    per_boot[label] = {"windows": len(rows), "S>1_share": round(s1 / steps, 4) if steps else None,
                       "recording_ratio": round(rec / tru, 6) if tru else None,
                       "sp_s_values": sorted({r.get("sp_s") for r in rows})}
    pool["steps"] += steps; pool["steps_s_gt1"] += s1; pool["rec"] += rec; pool["true"] += tru
missing = [l for l, v in per_boot.items() if "error" in v]
if missing:
    print(f"INCOMPLETE: {missing} -> refusing to adjudicate (4/4 v2 boots required)")
    out = {"rule": "ELIGIBLE iff pooled S>1 share >=0.10 AND pooled recording_ratio < 0.039062 (A3 frozen ref)",
           "pool_labels": ["prep3_B1r_P_E3", "prep3_B2_P_E3", "prep3_B3_P_E4", "prep3_B4_P_E4"],
           "per_boot": per_boot, "INCOMPLETE": missing, "ELIGIBLE": None}
    json.dump(out, open(f"{G}/SAMPLING_ELIGIBILITY.json", "w"), indent=1)
    raise SystemExit(1)
p_share = pool["steps_s_gt1"] / pool["steps"] if pool["steps"] else 0.0
p_ratio = pool["rec"] / pool["true"] if pool["true"] else None
eligible = (p_share >= 0.10) and (p_ratio is not None and p_ratio < A3_REF)
out = {"rule": "ELIGIBLE iff pooled S>1 share >=0.10 AND pooled recording_ratio < 0.039062 (A3 frozen ref)",
       "pool_labels": ["prep3_B1r_P_E3", "prep3_B2_P_E3", "prep3_B3_P_E4", "prep3_B4_P_E4"],
       "note": "v2 resume pool; prep3_B1_P_E3 (v1, FAIL) kept as failure evidence and excluded (see PROTOCOL.md addendum A1)",
       "per_boot": per_boot,
       "pooled": {"S>1_share": round(p_share, 4), "recording_ratio": round(p_ratio, 6) if p_ratio is not None else None,
                  "steps": pool["steps"], "steps_s_gt1": pool["steps_s_gt1"], "rec": pool["rec"], "true": pool["true"]},
       "ELIGIBLE": bool(eligible)}
json.dump(out, open(f"{G}/SAMPLING_ELIGIBILITY.json", "w"), indent=1)
print(json.dumps(out["pooled"]), "-> ELIGIBLE" if eligible else "-> NOT_ELIGIBLE (phase-6 STOP)")
