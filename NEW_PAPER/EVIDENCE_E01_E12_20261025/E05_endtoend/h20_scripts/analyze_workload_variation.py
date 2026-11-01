#!/usr/bin/env python3
"""Offline workload-variation analysis for the five-arm campaign (post-hoc, read-only).

Builds, for each of the 25 campaign starts, an aligned timeline of:
  - route-distribution change probes  ([PB-OEPLB-STABILITY] cos_sim, raw logical load,
    placement-independent, per-layer cosine averaged; logged every 4th window by DP0)
  - adaptive-window events            ([PB-OEPLB-ADW] ratio jumps / shrink / grow)
  - domain-shift resets               ([PB-OEPLB-RESET] converged/prev/current ratio)
  - window doubling/shift             ([PB-OEPLB-WINDOW])
  - swap issuance + walls             ([PB-OEPLB] window#N: issued / swap(s) done)
  - controller profiling              ([PB-OEPLB-PROF])
and aligns them with the KNOWN domain-segment structure of evaluation.jsonl
(4 contiguous blocks sharegpt->prover->book->code; exact cumulative token fractions
computed offline with the model tokenizer).

No raw campaign artifact is modified. Outputs are written to a new directory only.
"""
import json, re, os, sys, collections, statistics
from datetime import datetime, timezone

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
LOGS = f"{EXP}/results/ea_v1/fivearm_campaign/logs"
CAMP = f"{EXP}/results/ea_v1/fivearm_campaign/campaign.log"
OUTDIR = f"{EXP}/results/ea_v1/workload_variation"
TOK = "/tmp/eval_tokens.json"

os.makedirs(OUTDIR, exist_ok=True)

tk = json.load(open(TOK))
per_dom = tk["per_domain"]
dom_order = ["sharegpt", "prover", "book", "code"]
tot = tk["total_tokens"]
cum = 0.0
bounds = []
for d in dom_order[:-1]:
    cum += per_dom[d]["tokens"] / tot
    bounds.append(round(cum, 4))

ZONE = 0.03
STARTUP = 0.05

def zone_of(frac):
    if frac < STARTUP:
        return "startup"
    for i, b in enumerate(bounds):
        if abs(frac - b) <= ZONE:
            return f"boundary{i+1}"
    if frac < bounds[0]:
        return "seg_sharegpt"
    if frac < bounds[1]:
        return "seg_prover"
    if frac < bounds[2]:
        return "seg_book"
    return "seg_code"

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\+00:00\] BENCH_DONE (\S+) rc=0 bench=\{[^}]*\"total_time_s\": ([0-9.]+)")
bench = {}
for ln in open(CAMP):
    m = TS_RE.match(ln)
    if m:
        t1 = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        dur = float(m.group(3))
        bench[m.group(2)] = (t1.timestamp() - dur, t1.timestamp(), dur)
assert len(bench) == 25, f"expected 25 BENCH_DONE, got {len(bench)}"

LOG_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d)")

def parse_log(run):
    t0, t1, dur = bench[run]
    ev = collections.defaultdict(list)
    swaps_done = collections.defaultdict(list)
    resets = collections.defaultdict(list)
    for ln in open(f"{LOGS}/{run}_server.log", errors="replace"):
        m = LOG_TS.match(ln)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        dp = int(m.group(2))
        frac = (ts - t0) / dur
        if "PB-OEPLB-STABILITY" in ln and dp == 0:
            mm = re.search(r"w#(\d+) cos_sim=([0-9.]+) recent_avg=([0-9.]+) all_avg=([0-9.]+)", ln)
            ev["stability"].append({"ts": ts, "frac": round(frac, 4), "w": int(mm.group(1)),
                                    "cos": float(mm.group(2)), "recent": float(mm.group(3)),
                                    "all": float(mm.group(4)), "zone": zone_of(frac)})
        elif "PB-OEPLB-ADW" in ln and dp == 0:
            mm = re.search(r"ratio jump ([0-9.]+) -> (.+)$", ln.strip())
            kind = "shrink" if "shrink" in ln else ("grow" if "grow" in ln else "other")
            ev["adw"].append({"ts": ts, "frac": round(frac, 4), "jump": float(mm.group(1)) if mm else None,
                              "kind": kind, "detail": (mm.group(2)[:80] if mm else ln.strip()[-80:]), "zone": zone_of(frac)})
        elif "PB-OEPLB-RESET" in ln:
            mm = re.search(r"converged=([0-9.]+) prev=([0-9.]+) current=([0-9.]+)", ln)
            resets[round(ts)].append({"dp": dp, "converged": float(mm.group(1)),
                                      "prev": float(mm.group(2)), "current": float(mm.group(3))})
        elif "PB-OEPLB-WINDOW" in ln and dp == 0:
            kind = "stable_double" if "doubling" in ln else ("shift" if "shift" in ln else "other")
            mm = re.search(r"(\d+) -> (\d+)", ln)
            ev["window"].append({"ts": ts, "frac": round(frac, 4), "kind": kind,
                                 "from": int(mm.group(1)) if mm else None,
                                 "to": int(mm.group(2)) if mm else None, "zone": zone_of(frac)})
        elif dp == 0 and " issued " in ln and "swap(s)" in ln:
            mm = re.search(r"window#(\d+): issued (\d+) swap\(s\) \((\d+) tok global, dp=(\d+)\)", ln)
            ev["issued"].append({"ts": ts, "frac": round(frac, 4), "w": int(mm.group(1)),
                                 "swaps": int(mm.group(2)), "tok_global": int(mm.group(3)),
                                 "zone": zone_of(frac)})
        elif "swap(s) done" in ln:
            mm = re.search(r"(\d+) swap\(s\) done \(([0-9.]+)ms\)", ln)
            swaps_done[round(ts, 1)].append(float(mm.group(2)))
        elif "PB-OEPLB-PROF" in ln and dp == 0:
            mm = re.search(r"window#(\d+) calls=(\d+) record=([0-9.]+)ms allreduce=([0-9.]+)ms planbuild=([0-9.]+)ms finalize=([0-9.]+)ms total=([0-9.]+)ms", ln)
            if mm:
                ev["prof"].append({"ts": ts, "frac": round(frac, 4), "w": int(mm.group(1)),
                                   "calls": int(mm.group(2)), "record_ms": float(mm.group(3)),
                                   "allreduce_ms": float(mm.group(4)), "planbuild_ms": float(mm.group(5)),
                                   "finalize_ms": float(mm.group(6)), "total_ms": float(mm.group(7)),
                                   "zone": zone_of(frac)})
    for rts, rl in sorted(resets.items()):
        frac = (rts - t0) / dur
        ev["reset"].append({"ts": rts, "frac": round(frac, 4), "n_ranks": len(rl),
                            "converged": rl[0]["converged"], "prev": rl[0]["prev"],
                            "current": rl[0]["current"], "zone": zone_of(frac)})
    walls = [{"ts": k, "frac": round((k - t0) / dur, 4), "max_ms": max(v), "n_ranks": len(v),
              "zone": zone_of((k - t0) / dur)} for k, v in sorted(swaps_done.items())]
    max_w = max([e["w"] for e in ev["issued"] + ev["prof"] + ev["stability"]] or [0])
    prof = sorted(ev["prof"], key=lambda e: e["w"])
    rate_note = None
    if len(prof) >= 5 and max_w:
        xs = [e["w"] for e in prof]; ys = [e["frac"] for e in prof]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); sxx = sum((x-mx)**2 for x in xs)
        syy = sum((y-my)**2 for y in ys)
        r2 = (sxy*sxy)/(sxx*syy) if sxx and syy else None
        rate_note = {"r2_window_vs_wallfrac": round(r2, 4) if r2 else None,
                     "mean_s_per_window": round(dur/max_w, 2)}
    return {"run": run, "t0_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
            "dur_s": dur, "max_window": max_w, "events": ev, "walls": walls,
            "window_rate_check": rate_note}

ARMS = ["A00", "A10", "A01", "A11", "P11"]
runs = [f"eacR{r}_{a}" for r in range(1, 6) for a in ARMS]
runs.sort(key=lambda x: bench[x][0])

out = {"boundaries_cum_token_frac": dict(zip(["sharegpt|prover", "prover|book", "book|code"], bounds)),
       "zone_halfwidth": ZONE, "startup_frac": STARTUP, "starts": {}}
for run in runs:
    out["starts"][run] = parse_log(run)
    s = out["starts"][run]
    print(f"{run}: dur={s['dur_s']:.1f}s max_w={s['max_window']} "
          f"stab={len(s['events']['stability'])} adw={len(s['events']['adw'])} "
          f"reset={len(s['events']['reset'])} win={len(s['events']['window'])} "
          f"issued={len(s['events']['issued'])} walls={len(s['walls'])} "
          f"r2={s['window_rate_check']['r2_window_vs_wallfrac'] if s['window_rate_check'] else None}")

agg = {}
for arm in ARMS:
    arm_runs = [f"eacR{r}_{arm}" for r in range(1, 6)]
    zone_counts = collections.Counter()
    zone_cos_low = collections.Counter()
    zone_cos_vlow = collections.Counter()
    cos_by_zone = collections.defaultdict(list)
    jumps_by_zone = collections.defaultdict(list)
    reset_deltas = []
    wall_by_zone = collections.defaultdict(list)
    for run in arm_runs:
        ev = out["starts"][run]["events"]
        for e in ev["stability"]:
            zone_counts[("stab", e["zone"])] += 1
            cos_by_zone[e["zone"]].append(e["cos"])
            if e["cos"] < 0.95: zone_cos_low[e["zone"]] += 1
            if e["cos"] < 0.90: zone_cos_vlow[e["zone"]] += 1
        for e in ev["adw"]:
            zone_counts[("adw_"+e["kind"], e["zone"])] += 1
            if e["jump"] is not None:
                jumps_by_zone[e["zone"]].append(e["jump"])
        for e in ev["reset"]:
            zone_counts[("reset", e["zone"])] += 1
            reset_deltas.append({"frac": e["frac"], "zone": e["zone"],
                                 "delta": round(e["current"]-e["prev"], 3)})
        for e in ev["window"]:
            zone_counts[("win_"+e["kind"], e["zone"])] += 1
        for w in out["starts"][run]["walls"]:
            zone_counts[("wall", w["zone"])] += 1
            wall_by_zone[w["zone"]].append(w["max_ms"])
    agg[arm] = {
        "zone_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(zone_counts.items())},
        "cos_low_lt095_by_zone": dict(zone_cos_low), "cos_vlow_lt090_by_zone": dict(zone_cos_vlow),
        "cos_stats_by_zone": {z: {"n": len(v), "mean": round(statistics.mean(v), 4),
                                   "min": round(min(v), 4)} for z, v in cos_by_zone.items() if v},
        "adw_jump_by_zone": {z: {"n": len(v), "mean": round(statistics.mean(v), 3),
                                  "max": round(max(v), 3)} for z, v in jumps_by_zone.items() if v},
        "reset_deltas": reset_deltas,
        "wall_ms_by_zone": {z: {"n": len(v), "mean_ms": round(statistics.mean(v), 1),
                                 "max_ms": round(max(v), 1)} for z, v in wall_by_zone.items() if v},
    }
out["aggregation"] = agg
json.dump(out, open(f"{OUTDIR}/timeline_events.json", "w"), indent=1)
print("\n=== per-arm zone aggregation (events over 5 starts) ===")
for arm in ARMS:
    a = agg[arm]
    print(f"\n[{arm}] cos_by_zone:", json.dumps(a["cos_stats_by_zone"]))
    print(f"[{arm}] cos<0.95 by zone:", a["cos_low_lt095_by_zone"], " cos<0.90:", a["cos_vlow_lt090_by_zone"])
    zc = collections.Counter()
    for k, v in a["zone_counts"].items():
        typ, z = k.split("|")
        if typ in ("adw_shrink", "adw_grow", "adw_other", "reset", "win_shift", "win_stable_double"):
            zc[z] += v
    print(f"[{arm}] adaptive-mechanism events by zone:", dict(zc))
    print(f"[{arm}] adw jump size by zone:", json.dumps(a["adw_jump_by_zone"]))
    if arm in ("A01", "A11", "P11"):
        print(f"[{arm}] reset deltas:", json.dumps(a["reset_deltas"]))
print("\nWROTE", f"{OUTDIR}/timeline_events.json")
