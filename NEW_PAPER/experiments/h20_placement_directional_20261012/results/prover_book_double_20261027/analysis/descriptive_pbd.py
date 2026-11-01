#!/usr/bin/env python3
"""pbd campaign formal descriptive analysis (doubled stream n=5344, switch 3344).
Cloned from descriptive_pb.py with deltas: split 2048->3344, hi 3048->5344, bsplit 2560->4344
(book block 2000 rows, half at 1000); comparison campaign = pb (arm-ordering replication)
instead of pc; lat_mean_ms/ttft_mean_ms endpoints added per PROTOCOL_pbd.md sec.6.
n=2 rounds: descriptive deltas only, NON-INFERENTIAL. Written BEFORE results inspection (sec.7)."""
import json, os, glob, re
import numpy as np

H = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
PBD = H + "/results/prover_book_double_20261027"
PB = H + "/results/prover_book_switch_20261027"
OUT = PBD + "/analysis"
R = {}
NAN = float("nan")

def g(s, *keys):
    for k in keys:
        v = s.get(k)
        if v is not None: return v
    return NAN

def phases(camp, lab, split, hi, bsplit):
    pf = camp + "/boots/" + lab + "/" + lab + "_perreq.jsonl"
    sf = camp + "/boots/" + lab + "/" + lab + ".json"
    if not os.path.exists(pf) or not os.path.exists(sf):
        return None
    s = json.load(open(sf))
    rows = [json.loads(l) for l in open(pf)]
    def ph(lo, h):
        sub = [r for r in rows if lo <= r["idx"] < h]
        if not sub: return None
        t0 = min(r["t_send_wall_s"] for r in sub)
        t1 = max(r["t_last_wall_s"] for r in sub)
        return len(sub) / (t1 - t0)
    return {"total": g(s, "tps"), "ok": s.get("ok"), "err": s.get("errors"),
            "lat_mean_ms": g(s, "lat_mean_ms", "lat_mean"),
            "ttft_mean_ms": g(s, "ttft_mean_ms", "ttft_mean"),
            "stage1": ph(0, split), "stage2": ph(split, hi),
            "s2_early": ph(split, bsplit), "s2_late": ph(bsplit, hi)}

# ---- 1. pbd 12-boot table (split 3344, book halves at 4344)
ARMS = ["I", "ES", "ED", "MT", "DF", "P0"]
pbd = {}
for a in ARMS:
    for rd in (1, 2):
        lab = "%s_r%d" % (a, rd)
        pbd[lab] = phases(PBD, lab, 3344, 5344, 4344)
R["pbd_boots"] = pbd

# ---- 2. pb comparison (same arms, original 3048 stream) for ordering replication
pb = {}
for a in ARMS:
    for rd in (1, 2):
        lab = "%s_r%d" % (a, rd)
        pb[lab] = phases(PB, lab, 2048, 3048, 2560)
R["pb_boots"] = pb

# ---- 3. P0 boundary/swap timing both campaigns
def p0_track(camp, lab):
    f = camp + "/boots/" + lab + "/v2p1_rank0.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f)]
    per = {}
    for r in rows:
        w = r.get("window_id")
        if w is not None and w not in per: per[w] = r
    ws = sorted(per)
    bnd = None
    for w in ws:
        z = per[w].get("z")
        if isinstance(z, float) and z > 5:
            bnd = w; break
    swap_windows = [w for w in ws if (per[w].get("sp_swaps_started") or 0) > 0]
    def avg(lo, hi, key):
        vals = [per[w].get(key) for w in ws if lo <= w <= hi and isinstance(per[w].get(key), float)]
        return (sum(vals) / len(vals)) if vals else None
    ev = {}
    for w in ws: ev[per[w].get("event")] = ev.get(per[w].get("event"), 0) + 1
    se = {}
    sep = camp + "/boots/" + lab + "/sp_run_end.json"
    if os.path.exists(sep): se = json.load(open(sep))
    return {"n_windows": len(ws), "boundary_w": bnd,
            "h_rate_pre": avg(2, (bnd or 3) - 1, "h_rate"),
            "h_rate_spike": per[bnd].get("h_rate") if bnd else None,
            "z_spike": per[bnd].get("z") if bnd else None,
            "h_rate_plateau_post": avg((bnd or 0) + 5, 10 ** 6, "h_rate"),
            "z_plateau_post": avg((bnd or 0) + 5, 10 ** 6, "z"),
            "swap_windows": swap_windows, "n_swap_windows": len(swap_windows),
            "swaps_total": se.get("swaps_started_total"), "p2l_version_final": se.get("p2l_version"),
            "events": ev, "true_steps": se.get("true_steps_total"), "state_final": se.get("state_final")}
R["p0"] = {}
for camp, tag in ((PBD, "pbd"), (PB, "pb")):
    for lab in sorted(os.path.basename(p) for p in glob.glob(camp + "/boots/P0_*")):
        t = p0_track(camp, lab)
        if t: R["p0"][tag + "_" + lab] = t

# ---- 4. ED rebalance event timeline
def ed_events(camp, lab):
    f = camp + "/boots/" + lab + "/server.log"
    if not os.path.exists(f): return None
    n_reb = 0; stamps = []; sample = None
    pat = re.compile(r"rebalanc", re.I)
    tpat = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    for line in open(f, errors="replace"):
        if pat.search(line) and "EPLB" in line.upper():
            n_reb += 1
            m = tpat.search(line)
            if m: stamps.append(m.group(1))
            if sample is None: sample = line.strip()[:200]
    return {"n_lines": n_reb, "first": stamps[0] if stamps else None,
            "last": stamps[-1] if stamps else None, "sample": sample}
R["ed"] = {}
for camp, tag in ((PBD, "pbd"), (PB, "pb")):
    for lab in ("ED_r1", "ED_r2"):
        v = ed_events(camp, lab)
        if v: R["ed"][tag + "_" + lab] = v

# ---- 5. static arms 0-periodic-rebalance evidence (pbd)
R["static_rebalance_scan"] = {}
pat = re.compile(r"rebalanc", re.I)
for a in ("I", "ES", "MT", "DF"):
    for rd in (1, 2):
        lab = "%s_r%d" % (a, rd)
        f = PBD + "/boots/" + lab + "/server.log"
        if not os.path.exists(f): continue
        n = 0; sample = None
        for line in open(f, errors="replace"):
            if pat.search(line):
                n += 1
                if sample is None: sample = line.strip()[:180]
        R["static_rebalance_scan"]["pbd_" + lab] = {"n_rebalance_lines": n, "sample": sample}

with open(OUT + "/descriptive_pbd.json", "w") as f:
    json.dump(R, f, indent=1)

# ---- readable
L = []
L.append("# pbd campaign descriptive analysis (doubled stream n=5344, n=2 rounds, NON-INFERENTIAL)")
L.append("")
L.append("## 1. pbd 12-boot throughput (req/s). stage1=prover[0:3344), stage2=book[3344:5344), early[3344:4344), late[4344:5344)")
L.append("| boot | total | prover | book | b_early | b_late | lat_ms | ttft_ms | dbook_vs_I(round) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for rd in (1, 2):
    base = pbd.get("I_r%d" % rd)
    bi = base["stage2"] if base else None
    for a in ARMS:
        lab = "%s_r%d" % (a, rd)
        v = pbd.get(lab)
        if not v: continue
        d = ""
        if bi and a != "I":
            d = "%+.1f%%" % (100.0 * (v["stage2"] - bi) / bi)
        L.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f | %.1f | %s |" % (lab, v["total"], v["stage1"], v["stage2"], v["s2_early"], v["s2_late"], v["lat_mean_ms"], v["ttft_mean_ms"], d))
L.append("")
L.append("## 2. round means (2 reps, descriptive)")
L.append("| arm | prover r1/r2 | book r1/r2 | book mean | b_late r1/r2 | prover mean |")
L.append("|---|---|---|---|---|---|")
for a in ARMS:
    v1 = pbd.get(a + "_r1"); v2 = pbd.get(a + "_r2")
    if v1 and v2:
        L.append("| %s | %.2f / %.2f | %.2f / %.2f | %.2f | %.2f / %.2f | %.2f |" % (a, v1["stage1"], v2["stage1"], v1["stage2"], v2["stage2"], (v1["stage2"] + v2["stage2"]) / 2, v1["s2_late"], v2["s2_late"], (v1["stage1"] + v2["stage1"]) / 2))
L.append("")
L.append("## 3. pb (original 3048 stream) same arms, ordering-replication reference")
L.append("| boot | total | prover | book | b_early | b_late |")
L.append("|---|---|---|---|---|---|")
for rd in (1, 2):
    for a in ARMS:
        lab = "%s_r%d" % (a, rd)
        v = pb.get(lab)
        if not v: continue
        L.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (lab, v["total"], v["stage1"], v["stage2"], v["s2_early"], v["s2_late"]))
L.append("")
L.append("## 3b. arm ordering comparison (book-mean and prover-mean, descending)")
def mean2(dd, a, key):
    v1 = dd.get(a + "_r1"); v2 = dd.get(a + "_r2")
    if v1 and v2 and v1.get(key) is not None and v2.get(key) is not None:
        return (v1[key] + v2[key]) / 2
    return None
for keyname, key in (("book", "stage2"), ("prover", "stage1")):
    o_pbd = sorted([(mean2(pbd, a, key), a) for a in ARMS if mean2(pbd, a, key)], reverse=True)
    o_pb = sorted([(mean2(pb, a, key), a) for a in ARMS if mean2(pb, a, key)], reverse=True)
    L.append("- pbd %s: %s" % (keyname, " > ".join("%s %.2f" % (a, v) for v, a in o_pbd)))
    L.append("- pb  %s: %s" % (keyname, " > ".join("%s %.2f" % (a, v) for v, a in o_pb)))
L.append("")
L.append("## 4. P0 boundary + swap timing (pbd vs pb)")
for k in sorted(R["p0"].keys()):
    t = R["p0"][k]
    L.append("- %s: windows=%s boundary_w=%s h_rate pre=%.5f spike=%.5f plateau_post=%.5f z_spike=%.1f z_plateau=%.2f swaps_total=%s swap_windows=%s p2l_v=%s steps=%s state=%s" % (
        k, t["n_windows"], t["boundary_w"], t["h_rate_pre"] or -1, t["h_rate_spike"] or -1, t["h_rate_plateau_post"] or -1,
        t["z_spike"] or -1, t["z_plateau_post"] or -1, t["swaps_total"], t["n_swap_windows"], t["p2l_version_final"], t["true_steps"], t["state_final"]))
    L.append("    events: %s" % json.dumps(t["events"]))
L.append("")
L.append("## 5. ED rebalance lines (events x 8 DP ranks expected)")
for k in sorted(R["ed"].keys()):
    v = R["ed"][k]
    L.append("- %s: lines=%s first=%s last=%s" % (k, v["n_lines"], v["first"], v["last"]))
L.append("")
L.append("## 6. static arms periodic-rebalance scan (expect init-only, no periodic events)")
for k in sorted(R["static_rebalance_scan"].keys()):
    v = R["static_rebalance_scan"][k]
    L.append("- %s: rebalance-lines=%s sample=%s" % (k, v["n_rebalance_lines"], (v["sample"] or "")[:140]))
with open(OUT + "/descriptive_pbd.md", "w") as f:
    f.write("\n".join(L) + "\n")
print("WROTE descriptive_pbd.json/.md")
