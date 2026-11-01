#!/usr/bin/env python3
"""POST-HOC supplement to the frozen v3 analysis (documented, never overwrites).

The frozen analyzer's overhead-decomposition regexes were written against an
older controller log format and did not match the deployed PB-OEPLB v0.11
lines:
  actual swap-done : "[PB-OEPLB] 241 swap(s) done (1413.0ms) | total=241"
  analyzer expected: "... done (1413.0 ms) | total=..."  (space before ms)
  actual prof      : "window#3 calls=120 record=1.2ms allreduce=0.9ms planbuild=2.1ms finalize=0.4ms total=4.6ms avg_record_us=38.3"
  analyzer expected: "record/.../.../.../..." slash format.
Consequence in analysis.json (sha 86785b86...): swap_events=0 and
cumulative_swap_wall_s=0 UNDERSTATE the online arm's real migration activity.
The three arm CONTRASTS are unaffected: they come from bench total_time_s
direct measurements, and the decomposition is reported ALONGSIDE and NEVER
subtracted (frozen protocol rule). This supplement extracts the true numbers
with corrected regexes. Raw logs untouched.
"""
import json, re, hashlib, os, sys
from collections import defaultdict
from datetime import datetime, timezone

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUT = os.path.join(EXP, "results", "fullservice_v3", "overhead_supplement.json")
LOGDIR = os.path.join(EXP, "results", "fullservice_v3", "logs")

SWAP_DONE = re.compile(r"\[PB-OEPLB\] (\d+) swap\(s\) done \(([\d.]+)ms\) \| total=(\d+)")
PROF = re.compile(r"\[PB-OEPLB-PROF\] window#(\d+) calls=(\d+) record=([\d.]+)ms allreduce=([\d.]+)ms planbuild=([\d.]+)ms finalize=([\d.]+)ms total=([\d.]+)ms avg_record_us=([\d.]+)")
ISSUED = re.compile(r"\[PB-OEPLB\] window#(\d+): issued (\d+) swap\(s\) \((\d+) tok global, dp=(\d+)\)")
RANKP = re.compile(r"\[20[0-9-]+ [0-9:]+ DP(\d+) TP\d+ EP\d+\]")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()

def parse(path):
    per_win_ms = defaultdict(dict)   # window_key -> rank -> ms  (done lines carry no window#; order them by occurrence per rank)
    done_seq = defaultdict(list)     # rank -> [ms,...] in order
    totals = {}                      # rank -> final cumulative total
    issued = {}                      # window# -> swaps issued (rank-independent, identical across ranks)
    prof = defaultdict(dict)         # rank -> window# -> fields
    for line in open(path, errors="replace"):
        rm = RANKP.search(line)
        rank = int(rm.group(1)) if rm else -1
        m = SWAP_DONE.search(line)
        if m:
            done_seq[rank].append(float(m.group(2)))
            totals[rank] = int(m.group(3))
            continue
        m = ISSUED.search(line)
        if m:
            issued[int(m.group(1))] = int(m.group(2))
            continue
        m = PROF.search(line)
        if m:
            prof[rank][int(m.group(1))] = {
                "calls": int(m.group(2)), "record_ms": float(m.group(3)),
                "allreduce_ms": float(m.group(4)), "planbuild_ms": float(m.group(5)),
                "finalize_ms": float(m.group(6)), "total_ms": float(m.group(7)),
                "avg_record_us": float(m.group(8)),
            }
    # swap wall: per commit step, max across ranks (commit is synchronized); steps align by sequence index
    nsteps = max((len(v) for v in done_seq.values()), default=0)
    step_wall = []
    for i in range(nsteps):
        vals = [done_seq[r][i] for r in done_seq if i < len(done_seq[r])]
        step_wall.append(max(vals) if vals else 0.0)
    prof_last = {}
    for r, wins in prof.items():
        if wins:
            w = max(wins)
            prof_last[r] = {"window": w, **wins[w]}
    prof_rec_sum = {r: sum(v["record_ms"] for v in wins.values()) for r, wins in prof.items()}
    return {
        "commit_steps": nsteps,
        "swap_wall_ms_sum_over_steps_maxrank": round(sum(step_wall), 1),
        "swap_wall_ms_per_rank_sum": {str(r): round(sum(v), 1) for r, v in sorted(done_seq.items())},
        "final_cumulative_swap_total_per_rank": {str(r): t for r, t in sorted(totals.items())},
        "windows_issued": {str(k): v for k, v in sorted(issued.items())},
        "total_swaps_issued_rank0_view": sum(issued.values()),
        "prof_windows_seen": {str(r): len(w) for r, w in sorted(prof.items())},
        "prof_record_ms_sum_per_rank": {str(r): round(v, 1) for r, v in sorted(prof_rec_sum.items())},
        "prof_last_window_per_rank": {str(r): v for r, v in sorted(prof_last.items())},
    }

def main():
    starts = {}
    for lab in ["fs300v3_on_s1", "fs300v3_on_s2", "fs300v3_on_s3"]:
        p = os.path.join(LOGDIR, lab + "_server.log")
        starts[lab] = parse(p)
        starts[lab]["server_log_sha256"] = sha(p)
        bench = json.load(open(os.path.join(EXP, "results", "fullservice_v3", "bench", lab + ".json")))
        starts[lab]["bench_total_time_s"] = bench["total_time_s"]
        starts[lab]["swap_wall_share_of_bench_upper_bound"] = round(
            starts[lab]["swap_wall_ms_sum_over_steps_maxrank"] / 1000.0 / bench["total_time_s"], 5)
    doc = {
        "study_id": "h20-placement-fullservice-v3-overhead-supplement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "frozen analyzer regex/format mismatch (see script docstring); contrasts in analysis.json (sha 86785b86cd6da65bf46c18fe62dd2b0480c76ae39f0780c38c09bcd0e6825663) are direct bench measurements and remain the sole basis for all reported effects",
        "frozen_analysis_untouched": True,
        "caveat": "async swapper overlaps migration with serving; wall share is an UPPER-BOUND reported ALONGSIDE and NEVER subtracted from measured contrasts",
        "by_start": starts,
    }
    allw = [s["swap_wall_ms_sum_over_steps_maxrank"] for s in starts.values()]
    allt = [s["bench_total_time_s"] for s in starts.values()]
    doc["summary"] = {
        "mean_swap_wall_s_per_start": round(sum(allw) / len(allw) / 1000.0, 3),
        "mean_bench_total_s": round(sum(allt) / len(allt), 2),
        "mean_upper_bound_wall_share": round(sum(w / 1000.0 / t for w, t in zip(allw, allt)) / len(allt), 5),
        "total_swaps_issued_per_start_rank0_view": [s["total_swaps_issued_rank0_view"] for s in starts.values()],
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    print("wrote", OUT, "sha256=" + sha(OUT))
    print(json.dumps(doc["summary"], indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
