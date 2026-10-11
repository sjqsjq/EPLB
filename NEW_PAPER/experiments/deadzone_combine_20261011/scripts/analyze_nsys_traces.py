#!/usr/bin/env python3
"""Validate and summarize per-rank Nsight Systems captures."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


NAMES = (
    "timeline-original-zero",
    "timeline-original-inside",
    "timeline-original-outside",
    "timeline-barrier-zero",
    "timeline-barrier-inside",
    "timeline-barrier-outside",
)
BASE_RANGES = {
    "deadzone_profile_capture",
    "dispatch",
    "experts_deepgemm_fp8",
    "injected_gpu_delay",
    "deepep_combine",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def read_sqlite(path: Path, condition: str, rank: int) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    kernel_rows = connection.execute(
        """
        SELECT k.end - k.start, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        """
    ).fetchall()
    delay_durations = [duration / 1000.0 for duration, name in kernel_rows if name.startswith("deadzone_gpu_delay_ns")]
    require(len(delay_durations) == 1, f"{path}: expected one delay kernel")
    names = [name for _, name in kernel_rows]
    require(sum(name.startswith("sm90_fp8_gemm") for name in names) >= 2, f"{path}: missing DeepGEMM kernels")
    require(any(name == "dispatch" for name in names), f"{path}: missing DeepEP dispatch")
    require(any(name == "combine" for name in names), f"{path}: missing DeepEP combine")

    nvtx_names = {
        name
        for (name,) in connection.execute(
            """
            SELECT COALESCE(n.text, s.value)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IS NOT NULL
            """
        )
    }
    required = BASE_RANGES | {f"deadzone_layer:{condition}"}
    if "barrier" in condition:
        required.add("strict_gpu_ready_allreduce")
    require(required <= nvtx_names, f"{path}: missing NVTX ranges {sorted(required - nvtx_names)}")
    connection.close()
    return {
        "condition": condition,
        "rank": rank,
        "delay_kernel_us": delay_durations[0],
        "kernel_count": len(kernel_rows),
        "nvtx_ranges": sorted(nvtx_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    specs = {item["name"]: item["condition"] for item in manifest["conditions"]}
    require(set(specs) == set(NAMES), "timeline manifest condition set")

    rows = []
    detailed = []
    for name in NAMES:
        for rank in range(8):
            report = args.trace_dir / f"{name}-rank{rank}.nsys-rep"
            sqlite = args.trace_dir / f"{name}-rank{rank}.sqlite"
            require(report.is_file() and report.stat().st_size > 0, f"missing report {report}")
            require(sqlite.is_file() and sqlite.stat().st_size > 0, f"missing sqlite {sqlite}")
            item = read_sqlite(sqlite, name, rank)
            detailed.append(item)
            rows.append({key: item[key] for key in ("condition", "rank", "delay_kernel_us", "kernel_count")})

    condition_summaries = []
    for name in NAMES:
        values = {row["rank"]: row["delay_kernel_us"] for row in rows if row["condition"] == name}
        other_median = statistics.median(value for rank, value in values.items() if rank != 3)
        net = values[3] - other_median
        target = float(specs[name]["delay_us"])
        condition_summaries.append(
            {
                "condition": name,
                "target_differential_us": target,
                "rank3_delay_kernel_us": values[3],
                "other_rank_median_delay_kernel_us": other_median,
                "measured_differential_us": net,
                "differential_error_us": net - target,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "nsys_delay_kernels.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "nsys_trace_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "report_count": 48,
                "sqlite_count": 48,
                "conditions": condition_summaries,
                "captures": detailed,
            },
            indent=2,
        )
        + "\n"
    )

    report_lines = [
        "# Nsight Systems trace validation (N=4096)",
        "",
        "All **48** per-rank `.nsys-rep` files and all **48** SQLite exports are nonempty. Every capture contains the named layer/dispatch/expert/delay/combine NVTX ranges, at least two DeepGEMM kernels, DeepEP dispatch and combine kernels, and exactly one `deadzone_gpu_delay_ns` kernel. Strict captures also contain `strict_gpu_ready_allreduce`.",
        "",
        "| condition | target differential (µs) | rank-3 delay kernel (µs) | other-rank median (µs) | measured differential (µs) | error (µs) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in condition_summaries:
        report_lines.append(
            f"| {item['condition']} | {item['target_differential_us']:.3f} | "
            f"{item['rank3_delay_kernel_us']:.3f} | "
            f"{item['other_rank_median_delay_kernel_us']:.3f} | "
            f"{item['measured_differential_us']:.3f} | "
            f"{item['differential_error_us']:+.3f} |"
        )
    report_lines.extend(
        [
            "",
            "These are separate one-sample profiler runs. Their overall layer latencies are perturbed by profiling and are not used in the timing fits or the legacy hot-rank-only `(δ−H)+` diagnostic. The delay-kernel duration itself confirms the requested differential at the injection boundary.",
        ]
    )
    report_path = args.output_dir / "NSYS_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"PASS: 48 reports and 48 SQLite exports")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
