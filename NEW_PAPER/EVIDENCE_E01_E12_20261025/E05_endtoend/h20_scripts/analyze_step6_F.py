#!/usr/bin/env python3
"""Deliverable F analysis — Step-6 formal comparison (protocol: step6_formal_protocol_E.md §8).
Pre-declared tables T1-T5. Reads only artifacts; never re-runs anything; no selection."""
import json, os, collections, glob
S6 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/step6"
NN = {"W0": 8037, "W1": 4000, "W2": 8000}
CONDS = ["W0", "W1", "W2"]

def bench(cell):
    p = f"{S6}/phaseB/{cell}/bench.json"
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return d if d["ok"] == NN[cell.split("_")[0]] and d["errors"] == 0 else None

out = {"protocol": "step6_formal_protocol_E.md", "tables": {}}

# T1 raw service times per cell (ALL starts shown)
t1 = {}
for c in CONDS:
    t1[c] = {}
    for arm in ["V2", "FG", "FL"]:
        vals = []
        for s in (1, 2):
            d = bench(f"{c}_{arm}__s{s}")
            vals.append({"start": s, "total_time_s": d["total_time_s"], "tps": d["tps"],
                         "ok": d["ok"], "errors": d["errors"]} if d else {"start": s, "RETIRED": True})
        t1[c][arm] = vals
out["tables"]["T1_raw_service_times"] = t1

# T2 per-condition deltas (pre-declared comparisons; per-start-pair; no pooling; no significance)
t2 = {}
for c in CONDS:
    d = {}
    for base in ["FG", "FL"]:
        pairs = []
        for s in (1, 2):
            v2, fx = bench(f"{c}_V2__s{s}"), bench(f"{c}_{base}__s{s}")
            if v2 and fx:
                delta = v2["total_time_s"] - fx["total_time_s"]
                pairs.append({"start": s, "V2_minus_base_s": round(delta, 2),
                              "pct": round(delta / fx["total_time_s"] * 100, 2)})
        spread_v2 = None
        vv = [bench(f"{c}_V2__s{s}") for s in (1, 2)]
        if all(vv): spread_v2 = abs(vv[0]["total_time_s"] - vv[1]["total_time_s"])
        d[f"V2_minus_{base}"] = {"pairs": pairs, "V2_start_spread_s": round(spread_v2, 2) if spread_v2 is not None else None}
    t2[c] = d
out["tables"]["T2_deltas"] = t2

# T3 behavior chains for V2 cells
t3 = {}
for cell_dir in sorted(glob.glob(f"{S6}/phaseB/*_V2__s*")):
    cell = os.path.basename(cell_dir)
    jp = f"{cell_dir}/v2_rank0.jsonl"
    if not os.path.exists(jp): continue
    rows = [json.loads(l) for l in open(jp)]
    changes = [{"w": i, "W": f"{r['W_before']}->{r['W_after']}", "event": r["event"],
                "z": round(r["z"], 1) if r["z"] is not None else None,
                "h_w": int(r["h_w_assignments"]), "reasons": r["reasons"]}
               for i, r in enumerate(rows) if r["W_after"] != r["W_before"]]
    steps = collections.Counter(); wins = collections.Counter()
    for r in rows:
        wins[r["W_after"]] += 1; steps[r["W_after"]] += r["steps_actual"]
    tot = sum(steps.values())
    cw = sorted(r["c_window_ms"] for r in rows[1:])
    t3[cell] = {
        "windows": len(rows),
        "events": dict(collections.Counter(r["event"] for r in rows)),
        "changes": changes,
        "dwell_windows": dict(wins), "dwell_steps": dict(steps),
        "dwell_step_frac": {w: round(s / tot, 3) for w, s in steps.items()},
        "desync_any": any(r["desync"] for r in rows),
        "c_window_ms_median": round(cw[len(cw)//2], 3) if cw else None,
        "c_window_ms_max": round(cw[-1], 3) if cw else None,
    }
out["tables"]["T3_behavior"] = t3

# T4 ablations vs Full (Phase-C cells vs Phase-B V2 s1/s2 same condition)
t4 = {}
for cell_dir in sorted(glob.glob(f"{S6}/phaseC/*")):
    cell = os.path.basename(cell_dir).split("__")[0]  # e.g. W0_M2off
    d = bench.__call__  # noqa
    p = f"{cell_dir}/bench.json"
    if not os.path.exists(p): continue
    b = json.load(open(p))
    cond = cell.split("_")[0]; abl = cell.split("_", 1)[1]
    v2s = [bench(f"{cond}_V2__s{s}") for s in (1, 2)]
    v2s = [v["total_time_s"] for v in v2s if v]
    nchg = None; evs = None
    jp = f"{cell_dir}/v2_rank0.jsonl"
    if os.path.exists(jp):
        rows = [json.loads(l) for l in open(jp)]
        nchg = sum(1 for r in rows if r["W_after"] != r["W_before"])
        evs = dict(collections.Counter(r["event"] for r in rows))
    t4[cell] = {"total_time_s": b["total_time_s"], "ok": b["ok"], "errors": b["errors"],
                "V2_full_times_same_cond": v2s, "W_changes": nchg, "events": evs}
out["tables"]["T4_ablations"] = t4

json.dump(out, open(f"{S6}/deliverable_F_analysis.json", "w"), indent=1)

# print compact report
print("== T1 raw service times (s) ==")
for c in CONDS:
    for arm in ["V2", "FG", "FL"]:
        vs = t1[c][arm]
        print(f"  {c} {arm}: " + " / ".join(str(v.get("total_time_s", "RETIRED")) for v in vs))
print("== T2 deltas (V2 - base, s; negative = V2 faster) ==")
for c in CONDS:
    for k, v in t2[c].items():
        pr = ", ".join(f"s{p['start']}: {p['V2_minus_base_s']:+.2f} ({p['pct']:+.2f}%)" for p in v["pairs"])
        print(f"  {c} {k}: {pr}  [V2 start spread {v['V2_start_spread_s']}s]")
print("== T3 behavior (V2 cells) ==")
for cell, v in t3.items():
    print(f"  {cell}: windows={v['windows']} changes={len(v['changes'])} desync={v['desync_any']} c_med={v['c_window_ms_median']}ms dwell_steps={v['dwell_steps']}")
print("== T4 ablations ==")
for cell, v in t4.items():
    print(f"  {cell}: t={v['total_time_s']}s ok={v['ok']} err={v['errors']} W_changes={v['W_changes']} V2full={v['V2_full_times_same_cond']}")
