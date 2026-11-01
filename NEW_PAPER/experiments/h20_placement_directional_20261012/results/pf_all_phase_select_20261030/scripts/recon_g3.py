#!/usr/bin/env python3
# recon_g3.py - formal-run reconciliation for pf_all_phase_select_20261030.
# Usage: python3 recon_g3.py            (run after queue completion)
# Reads boots/*/g3_facts.json + boots/*/LABEL.json (bench scalars) for the
# 16 formal labels; builds analysis/recon_g3_facts.json; prints the primary
# 16-row table, paired-block deltas, per-task means, record-volume and
# migration-ledger aggregates. Descriptive only; n=4 per task x arm cell;
# NO non-inferiority/superiority/online-algorithm/verified-gain language.
import json, os, glob

G3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ("ARC", "PROV")
ARMS = ("PF", "ALL")
BLOCKS = (1, 2, 3, 4)
rows = {}
missing = []
for t in TASKS:
    for a in ARMS:
        for b in BLOCKS:
            lab = "G3_%s_%s_b%d" % (t, a, b)
            fp = os.path.join(G3, "boots", lab, "g3_facts.json")
            bp = os.path.join(G3, "boots", lab, lab + ".json")
            if not os.path.exists(fp):
                missing.append(lab + ":g3_facts")
                continue
            f = json.load(open(fp))
            bench = {}
            if os.path.exists(bp):
                bench = json.load(open(bp))
            else:
                missing.append(lab + ":bench")
            sre = f.get("sp_run_end", {})
            sl = f.get("server_log", {})
            seg0 = (f.get("segments") or [{}])[0]
            v = seg0.get("v2p1", {})
            rows[lab] = {
                "task": t, "arm": a, "block": b,
                "total_time_s": bench.get("total_time_s"),
                "bench_scalars": {k: bench.get(k) for k in sorted(bench) if isinstance(bench.get(k), (int, float))},
                "all_pass": f.get("all_pass"),
                "fails": f.get("fail_names", []),
                "decode_record_batches": sre.get("decode_record_batches"),
                "sp_fwd_total": sre.get("sp_fwd_total"),
                "sample_opportunities": sre.get("sample_opportunities"),
                "recorded_steps": sre.get("recorded_steps"),
                "swaps_started_total": sre.get("swaps_started_total"),
                "window_count": sre.get("window_count"),
                "true_steps_total": sre.get("true_steps_total"),
                "issued_ops_total": sl.get("issued_ops_total"),
                "planhash_count": sl.get("planhash_count"),
                "diag_count": sl.get("diag_count"),
                "moved_mb_estimate": sl.get("moved_mb_estimate"),
                "armflags_rap": (sl.get("armflags_first") or {}).get("record_all_phase"),
                "traceback": sl.get("traceback_count"), "oom": sl.get("cuda_oom_count"),
                "ready": sl.get("ready_to_roll_count"),
                "v2p1_sum_valid_rows": v.get("sum_sp_valid_rows"),
                "v2p1_sum_recorded_steps": v.get("sum_sp_recorded_steps"),
                "v2p1_committed_windows": v.get("count_committed_true"),
                "c_window_ms_mean": v.get("c_window_ms_mean"),
                "c_window_ms_max": v.get("c_window_ms_max"),
            }
paired = []
for t in TASKS:
    for b in BLOCKS:
        pf = rows.get("G3_%s_PF_b%d" % (t, b), {}).get("total_time_s")
        al = rows.get("G3_%s_ALL_b%d" % (t, b), {}).get("total_time_s")
        d = None
        if isinstance(pf, (int, float)) and isinstance(al, (int, float)):
            d = {"task": t, "block": b, "pf_s": pf, "all_s": al,
                 "delta_all_minus_pf_s": round(al - pf, 3),
                 "pct": round(100.0 * (al - pf) / pf, 3)}
        paired.append(d)
means = {}
for t in TASKS:
    for a in ARMS:
        vals = [rows["G3_%s_%s_b%d" % (t, a, b)]["total_time_s"] for b in BLOCKS
                if "G3_%s_%s_b%d" % (t, a, b) in rows
                and isinstance(rows["G3_%s_%s_b%d" % (t, a, b)]["total_time_s"], (int, float))]
        means[t + "_" + a] = {"n": len(vals), "mean_s": round(sum(vals) / len(vals), 3) if vals else None,
                              "values": vals}
recvol = {}
for t in TASKS:
    for a in ARMS:
        acc = {"decode_record_batches": [], "recorded_steps": [], "sample_opportunities": [],
               "v2p1_sum_valid_rows": [], "swaps_started_total": [], "issued_ops_total": []}
        for b in BLOCKS:
            r = rows.get("G3_%s_%s_b%d" % (t, a, b))
            if r:
                for k in acc:
                    acc[k].append(r.get(k))
        recvol[t + "_" + a] = acc
out = {"generated_utc": None, "n_rows": len(rows), "missing": missing,
       "primary_rows": rows, "paired_blocks": paired, "means": means,
       "record_volumes": recvol,
       "guardrails": "n=4 per task x arm cell; DESCRIPTIVE only; no non-inferiority/superiority claims; no online-algorithm paper claim; wall-clock measurements on frozen stack; moved_mb is an ESTIMATE (ops x 37.74MB)."}
import subprocess
out["generated_utc"] = subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip()
os.makedirs(os.path.join(G3, "analysis"), exist_ok=True)
with open(os.path.join(G3, "analysis", "recon_g3_facts.json"), "w") as f2:
    json.dump(out, f2, indent=1, sort_keys=True)
print("RECON_G3 rows=%d missing=%s" % (len(rows), missing))
print("label | arm | task | blk | total_time_s | drb | rec_steps | issued | all_pass")
for lab in sorted(rows):
    r = rows[lab]
    print("%s | %s | %s | %d | %s | %s | %s | %s | %s" % (lab, r["arm"], r["task"], r["block"],
          r["total_time_s"], r["decode_record_batches"], r["recorded_steps"], r["issued_ops_total"], r["all_pass"]))
print("--- paired deltas (ALL - PF) ---")
for p in paired:
    print(p)
print("--- means ---")
for k in sorted(means):
    print(k, means[k])
