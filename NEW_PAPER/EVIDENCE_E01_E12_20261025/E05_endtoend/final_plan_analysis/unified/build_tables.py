#!/usr/bin/env python3
"""build_tables.py — unified analysis step 1: assemble per-cell tables from
frozen artifacts (MAIN 168 + P1 32). Read-only on sealed artifacts.
T = bench total_time_s (client monotonic; == wall == mono verified, §9.6)."""
import csv, json, os, re, glob

F = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019"
OUT = f"{F}/analysis/unified"

def load_summary(phase_dir, name):
    m = json.load(open(f"{F}/{phase_dir}/{name}"))
    return m["verdicts"]

def bench_metrics(path):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    mono = (m["mono_end_ns"] - m["mono_start_ns"]) / 1e9
    return {
        "T_s": m["total_time_s"],
        "T_mono_s": round(mono, 3),
        "tps_out": m["tps"],
        "ok": m["ok"], "errors": m["errors"], "num_requests": m["num_requests"],
        "total_comp_tokens": m["total_comp_tokens"],
        "total_prompt_tokens": m["total_prompt_tokens"],
        "avg_output_tokens": m.get("avg_output_tokens"),
        "concurrency": m.get("concurrency"),
        "ttft_p50_ms": m["ttft_p50_ms"], "ttft_p95_ms": m["ttft_p95_ms"], "ttft_p99_ms": m["ttft_p99_ms"],
        "lat_p50_ms": m["lat_p50_ms"], "lat_p95_ms": m["lat_p95_ms"], "lat_p99_ms": m["lat_p99_ms"],
        "tpot_p50_ms": m.get("tpot_p50_ms"), "tpot_p95_ms": m.get("tpot_p95_ms"), "tpot_p99_ms": m.get("tpot_p99_ms"),
    }

def verdict_t(art):
    p = f"{art}/verdict.txt"
    if os.path.exists(p):
        m = re.search(r"VERDICT:(\w+) t=([\d.]+)", open(p).read())
        if m: return m.group(1), float(m.group(2))
    return None, None

def kv_min(serverlog):
    if not os.path.exists(serverlog): return None, 0
    vals = [int(x) for x in re.findall(r"KV Cache is allocated\. #tokens: (\d+)", open(serverlog, errors="replace").read())]
    return (min(vals) if vals else None), len(vals)

def sp_behavior(art):
    p = f"{art}/sp_run_end.json"
    if not os.path.exists(p): return {}
    try: d = json.load(open(p))
    except Exception: return {}
    keys = ["S_final","H_final","W_final","mode","state_final","transitions","h_cycles",
            "sp_fwd_total","swaps_started_total","recorded_steps","sample_opportunities",
            "seals","residency_committed_h","nocommit_planned_ops_total","decode_record_batches","prof_record_calls","seed"]
    out = {}
    for k in keys:
        v = d.get(k)
        if isinstance(v, (list, dict)): v = len(v) if isinstance(v, list) else json.dumps(v, sort_keys=True)[:120]
        out[k] = v
    return out

# ---------- MAIN ----------
verdicts = load_summary("main", "summary_main.json")
rows = list(csv.DictReader(open(f"{F}/D_matrix/run_list_main.csv")))
assert len(rows) == 168
main_tbl = []
nreq_by_cond = {}
for r in rows:
    art = f"{F}/{r['artifact_dir']}"
    label = r["bench_label"]
    verd = verdicts.get(label, "MISSING")
    bm = bench_metrics(f"{art}/bench.json")
    vt, tval = verdict_t(art)
    if bm and bm["num_requests"]:
        nreq_by_cond.setdefault(r["cond"], set()).add(bm["num_requests"])
    rec = {"label": label, "cond": r["cond"], "block": int(r["block"]), "arm": r["arm"],
           "params": r["params"], "verdict": verd, "verdict_t": tval,
           "has_bench": int(bm is not None)}
    if bm:
        rec.update(bm)
        rec["completion_rate"] = round(bm["ok"] / bm["num_requests"], 6) if bm["num_requests"] else None
        assert abs(bm["T_s"] - tval) < 0.5, f"T mismatch {label}: {bm['T_s']} vs {tval}"
    else:
        rec.update({"T_s": None, "ok": None, "errors": None, "completion_rate": None})
    rec.update({f"behav_{k}": v for k, v in sp_behavior(art).items()})
    main_tbl.append(rec)

nreq_final = {c: sorted(s)[-1] for c, s in nreq_by_cond.items()}
for rec in main_tbl:
    rec["planned_nreq"] = nreq_final.get(rec["cond"])

# ---------- P1 ----------
verd1 = load_summary("p1", "summary_p1.json")
rows1 = list(csv.DictReader(open(f"{F}/D_matrix/run_list_p1.csv")))
assert len(rows1) == 32
p1_tbl = []
for r in rows1:
    art = f"{F}/{r['artifact_dir']}"
    label = r["bench_label"]
    verd = verd1.get(label, "MISSING")
    bm = bench_metrics(f"{art}/bench.json")
    vt, tval = verdict_t(art)
    kvm, kvn = kv_min(f"{art}/server.log")
    rec = {"label": label, "cond": r["cond"], "block": int(r["block"]), "arm": r["arm"],
           "params": r["params"], "verdict": verd, "verdict_t": tval,
           "kv_tokens_min": kvm, "kv_rank_lines": kvn,
           "pressure": "PH" if r["cond"] == "PH" else "PL"}
    if bm:
        rec.update(bm)
        rec["completion_rate"] = round(bm["ok"] / bm["num_requests"], 6)
        assert abs(bm["T_s"] - tval) < 0.5, f"T mismatch {label}"
    rec.update({f"behav_{k}": v for k, v in sp_behavior(art).items()})
    p1_tbl.append(rec)

def dump(name, tbl):
    if not tbl: return
    cols = sorted({k for r in tbl for k in r})
    order = ["label","cond","block","arm","params","pressure","verdict","has_bench","T_s","tps_out","ok","errors","num_requests","planned_nreq","completion_rate","kv_tokens_min"]
    cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in tbl: w.writerow(r)
    json.dump(tbl, open(f"{OUT}/{name}.json", "w"), indent=1, sort_keys=True)

dump("main_table", main_tbl)
dump("p1_table", p1_tbl)

nv = sum(1 for r in main_tbl if r["verdict"] == "VALID")
nr = sum(1 for r in main_tbl if r["verdict"] == "RESULT_RETIRE")
p1v = sum(1 for r in p1_tbl if r["verdict"] == "VALID")
retired = [(r["label"], r["has_bench"], r["ok"], r["errors"], r["planned_nreq"]) for r in main_tbl if r["verdict"] != "VALID"]
print(f"MAIN: {nv} VALID + {nr} RESULT_RETIRE = {len(main_tbl)}; P1: {p1v} VALID = {len(p1_tbl)}")
print("nreq_by_cond:", nreq_final)
print("retired cells (label, has_bench, ok, err, planned):")
for x in retired: print(" ", x)
kv = {}
for r in p1_tbl:
    kv.setdefault((r["arm"], r["params"]), []).append(r["kv_tokens_min"])
print("P1 KV min by (arm,params):", {k: (min(v), max(v)) for k, v in kv.items()})
