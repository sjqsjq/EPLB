#!/usr/bin/env python3
"""Pre-registered analysis of the h20-placement-fullservice-v3 three-arm campaign.

Implements exactly the frozen protocol's statistics section:
  primary   total_time_s per valid start (evaluation workload, concurrency 256)
  validity  boot ok, no Traceback in server log, bench rc=0, ok==N_EVAL, errors==0
  contrasts frozen-identity, online-identity, online-frozen; Welch two-sample t,
            95% CI; frozen trichotomy wording (excludes zero / compatible with
            zero; never 'equivalence'); negative point estimates preserved.
  arms with <2 valid starts -> contrasts reported blocked (never imputed).
  online net overhead decomposition parsed from PB-OEPLB controller log lines,
  reported ALONGSIDE contrasts and NEVER subtracted from them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
from typing import Any

from prediction_common import sha256_file, write_json_atomic

SPEC_ID = "h20-placement-fullservice-v3"
ARMS = ("identity", "frozen", "online")
LABELS = {
    "identity": ["fs300v3_id_s1", "fs300v3_id_s2", "fs300v3_id_s3"],
    "frozen": ["fs300v3_fx_s1", "fs300v3_fx_s2", "fs300v3_fx_s3"],
    "online": ["fs300v3_on_s1", "fs300v3_on_s2", "fs300v3_on_s3"],
}
CONTRASTS = (("frozen", "identity"), ("online", "identity"), ("online", "frozen"))


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def welch_ci(a: list[float], b: list[float]) -> dict[str, Any]:
    """95% Welch CI for mean(a) - mean(b)."""
    from scipy import stats

    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((v - ma) ** 2 for v in a) / (na - 1) if na > 1 else 0.0
    vb = sum((v - mb) ** 2 for v in b) / (nb - 1) if nb > 1 else 0.0
    se2 = va / na + vb / nb
    diff = ma - mb
    if se2 <= 0.0:
        return {"mean_diff_s": diff, "ci95_low_s": diff, "ci95_high_s": diff,
                "df": None, "verdict": "zero_variance_degenerate"}
    df = se2**2 / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1))
    t = float(stats.t.ppf(0.975, df))
    se = se2**0.5
    low, high = diff - t * se, diff + t * se
    if low > 0.0:
        verdict = "excludes_zero_positive"
    elif high < 0.0:
        verdict = "excludes_zero_negative"
    else:
        verdict = "compatible_with_zero"
    return {"mean_diff_s": diff, "ci95_low_s": low, "ci95_high_s": high,
            "welch_df": df, "verdict": verdict}


_SWAP_DONE = re.compile(r"\[PB-OEPLB\] (\d+) swap\(s\) done \(([\d.]+) ms\) \| total=(\d+)")
_PROF = re.compile(
    r"\[PB-OEPLB-PROF\] window#(\d+) calls=\d+ record/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms total=([\d.]+) avg_record_us=([\d.]+)"
)
_BUDGET = re.compile(r"\[PB-OEPLB\] swap budget exhausted: spent ([\d.]+)s of ([\d.]+)s \(([\d.]+)%\)")
_DECISION = re.compile(r"\[PB-OEPLB\] window#(\d+): (\w+)")


def parse_online_overhead(server_log: pathlib.Path) -> dict[str, Any]:
    text = server_log.read_text(errors="replace")
    swap_ms_total = 0.0
    swap_events = 0
    swap_count_final = 0
    for m in _SWAP_DONE.finditer(text):
        swap_events += 1
        swap_ms_total += float(m.group(2))
        swap_count_final = int(m.group(3))
    prof_last = None
    for m in _PROF.finditer(text):
        prof_last = {
            "window": int(m.group(1)),
            "record_ms": float(m.group(2)),
            "allreduce_ms": float(m.group(3)),
            "planbuild_ms": float(m.group(4)),
            "finalize_ms": float(m.group(5)),
            "cum_total_ms": float(m.group(6)),
            "avg_record_us": float(m.group(7)),
        }
    budget = None
    for m in _BUDGET.finditer(text):
        budget = {"spent_s": float(m.group(1)), "of_s": float(m.group(2)),
                  "pct": float(m.group(3))}
    decisions: dict[str, int] = {}
    for m in _DECISION.finditer(text):
        decisions[m.group(2)] = decisions.get(m.group(2), 0) + 1
    timing_lines = sum(1 for line in text.splitlines() if "[PB-OEPLB-TIMING]" in line)
    return {
        "swap_events": swap_events,
        "cumulative_swap_wall_ms": swap_ms_total,
        "final_cumulative_swap_count": swap_count_final,
        "prof_last_window_cumulative": prof_last,
        "budget_exhausted": budget,
        "window_decisions": decisions,
        "async_swapper_timing_lines": timing_lines,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("spec_id") != SPEC_ID or protocol.get("status") != "frozen_before_execution":
        raise ValueError("protocol is not the frozen fullservice v2 protocol")
    out_dir = root / "results/fullservice_v3"
    n_eval = protocol["workload"]["evaluation"]["requests"]

    state = json.loads((out_dir / "campaign_state.json").read_text())
    measure_entries = {e["run"]: e for e in state if e.get("phase") == "measure"}

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        starts = []
        for label in LABELS[arm]:
            entry = measure_entries.get(label)
            bench_path = out_dir / "bench" / f"{label}.json"
            server_log = out_dir / "logs" / f"{label}_server.log"
            record: dict[str, Any] = {"label": label, "state": entry}
            valid = False
            reasons = []
            if entry is None:
                reasons.append("no campaign state entry (run missing)")
            elif entry.get("status") == "boot_failed":
                reasons.append(f"boot failed rc={entry.get('boot_rc')} (retained, never retried)")
            elif entry.get("bench_rc") != 0:
                reasons.append(f"bench rc={entry.get('bench_rc')}")
            elif not bench_path.is_file():
                reasons.append("bench json missing")
            else:
                bench = json.loads(bench_path.read_text())
                record["bench_sha256"] = sha256_file(bench_path)
                record["total_time_s"] = bench["total_time_s"]
                record["tps"] = bench.get("tps")
                record["ttft_mean_ms"] = bench.get("ttft_mean_ms")
                record["ttft_p99_ms"] = bench.get("ttft_p99_ms")
                record["tpot_mean_ms"] = bench.get("tpot_mean_ms")
                record["lat_p99_ms"] = bench.get("lat_p99_ms")
                record["ok"] = bench.get("ok")
                record["errors"] = bench.get("errors")
                if bench.get("ok") != n_eval or bench.get("errors") != 0:
                    reasons.append(f"validity gate: ok={bench.get('ok')}!={n_eval} or errors={bench.get('errors')}!=0")
                tb = 0
                if server_log.is_file():
                    tb = server_log.read_text(errors="replace").count("Traceback")
                if tb:
                    reasons.append(f"{tb} Traceback(s) in server log")
                if not reasons:
                    valid = True
            record["valid"] = valid
            record["invalid_reasons"] = reasons
            starts.append(record)
        arms[arm] = {
            "starts": starts,
            "valid_starts": sum(s["valid"] for s in starts),
            "valid_total_times_s": [s["total_time_s"] for s in starts if s["valid"]],
        }

    contrasts = []
    for arm_a, arm_b in CONTRASTS:
        a = arms[arm_a]["valid_total_times_s"]
        b = arms[arm_b]["valid_total_times_s"]
        entry: dict[str, Any] = {"contrast": f"{arm_a}-{arm_b}", "n_a": len(a), "n_b": len(b)}
        if len(a) < 2 or len(b) < 2:
            entry["status"] = "blocked_insufficient_valid_starts"
        else:
            entry["status"] = "computed"
            entry.update(welch_ci(a, b))
            entry["mean_a_s"] = sum(a) / len(a)
            entry["mean_b_s"] = sum(b) / len(b)
        contrasts.append(entry)

    overhead_by_start = {}
    for label in LABELS["online"]:
        server_log = out_dir / "logs" / f"{label}_server.log"
        if server_log.is_file():
            overhead_by_start[label] = parse_online_overhead(server_log)
    online_times = arms["online"]["valid_total_times_s"]
    overhead_summary = None
    if online_times:
        total_swap_s = sum(
            o["cumulative_swap_wall_ms"] for o in overhead_by_start.values()
        ) / 1000.0
        overhead_summary = {
            "cumulative_swap_wall_s_all_online_starts": total_swap_s,
            "mean_online_total_time_s": sum(online_times) / len(online_times),
            "overhead_fraction_upper_bound_wall_share": total_swap_s
            / max(len(overhead_by_start), 1)
            / (sum(online_times) / len(online_times)),
            "caveat": (
                "async swapper overlaps migration with serving; fraction is an "
                "upper-bound wall-share reported ALONGSIDE (never subtracted from) "
                "the measured arm contrasts"
            ),
        }

    output = {
        "schema_version": 1,
        "study_id": SPEC_ID,
        "analysis": "three_arm_fullservice_frozen_analysis",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analyzer_implementation": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "evaluation_requests": n_eval,
        "arms": arms,
        "contrasts": contrasts,
        "online_overhead_decomposition": {
            "by_start": overhead_by_start,
            "summary": overhead_summary,
        },
        "interpretation_rules": protocol["interpretation_rules"],
        "prohibitions_respected": [
            "no best-start or repeat selection: every start entered its arm",
            "failed starts retained with reasons, never retried",
            "overhead decomposition never subtracted from measured contrasts",
            "CI crossing zero reported as compatible with zero, never equivalence",
            "negative point estimates preserved exactly",
        ],
    }
    write_json_atomic(args.output.resolve(), output)

    for arm in ARMS:
        info = arms[arm]
        print(f"arm {arm}: valid {info['valid_starts']}/3 "
              f"times={[round(t,2) for t in info['valid_total_times_s']]}")
        for s in info["starts"]:
            if not s["valid"]:
                print(f"  INVALID {s['label']}: {s['invalid_reasons']}")
    for c in contrasts:
        if c["status"] == "computed":
            print(f"contrast {c['contrast']}: {c['mean_diff_s']:+.2f}s "
                  f"CI[{c['ci95_low_s']:+.2f},{c['ci95_high_s']:+.2f}] {c['verdict']}")
        else:
            print(f"contrast {c['contrast']}: {c['status']} (n_a={c['n_a']}, n_b={c['n_b']})")
    print(f"wrote {args.output} sha256={sha256_file(args.output.resolve())}")


if __name__ == "__main__":
    main()
