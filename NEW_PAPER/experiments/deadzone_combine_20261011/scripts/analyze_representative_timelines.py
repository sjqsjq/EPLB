#!/usr/bin/env python3
"""Plot representative per-rank stage timelines from the main timing scan."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CONDITIONS = (
    ("original-zero", "Original: zero"),
    ("original-rank3-x0.05", "Original: inside (0.05 C0)"),
    ("original-rank3-x0.4", "Original: outside (0.40 C0)"),
    ("barrier-zero", "Strict: zero"),
    ("barrier-rank3-x0.05", "Strict: inside (0.05 C0)"),
    ("barrier-rank3-x0.4", "Strict: outside (0.40 C0)"),
)
STAGES = (
    ("dispatch_ms", "Dispatch", "#4c78a8"),
    ("expert_ms", "Experts", "#f58518"),
    ("injected_delay_ms", "Delay kernel", "#e45756"),
    ("strict_gate_ms", "Ready gate", "#72b7b2"),
    ("combine_after_gate_ms", "Combine / wait", "#54a24b"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    records_by_condition = {
        name: [record for record in payload["records"] if record["condition"] == name]
        for name, _ in CONDITIONS
    }
    for name, records in records_by_condition.items():
        if len(records) != 500:
            raise ValueError(f"{name}: expected 500 records, got {len(records)}")

    rows = []
    for name, label in CONDITIONS:
        records = records_by_condition[name]
        for rank in range(8):
            row = {"condition": name, "label": label, "rank": rank}
            for key, _, _ in STAGES:
                row[key.replace("_ms", "_us")] = 1000.0 * statistics.median(
                    record["ranks"][rank][key] for record in records
                )
            row["layer_us"] = 1000.0 * statistics.median(
                record["ranks"][rank]["layer_ms"] for record in records
            )
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "representative_timeline_stages.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), sharex=True, sharey=False)
    for axis, (name, label) in zip(axes.flat, CONDITIONS, strict=True):
        condition_rows = [row for row in rows if row["condition"] == name]
        left = [0.0] * 8
        for key, stage_label, color in STAGES:
            values = [row[key.replace("_ms", "_us")] for row in condition_rows]
            axis.barh(
                range(8),
                values,
                left=left,
                height=0.72,
                color=color,
                label=stage_label,
            )
            left = [start + width for start, width in zip(left, values, strict=True)]
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
        axis.set_yticks(range(8), [f"rank {rank}" for rank in range(8)])
        axis.get_yticklabels()[3].set_color("#c62828")
        axis.get_yticklabels()[3].set_fontweight("bold")
    for axis in axes[:, 0]:
        axis.set_ylabel("EP rank (injected rank 3 in red)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Median same-device CUDA-event time from layer start (µs)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(STAGES), frameon=False)
    fig.suptitle(
        "N=4096 representative stage timelines (5 blocks × 100; profiling excluded)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = args.output_dir / "representative_timelines.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    def median(condition: str, rank: int, metric: str) -> float:
        return 1000.0 * statistics.median(
            record["ranks"][rank][metric]
            for record in records_by_condition[condition]
        )

    report = [
        "# Representative timelines (N=4096)",
        "",
        "The static figure is reconstructed from same-device CUDA-event intervals in the main 5×100 timing run. Nsight is collected in separate one-sample runs and is not used for numerical latency estimates.",
        "",
        "| path | delay | rank-3 delay (µs) | rank-3 ready gate (µs) | median combine after gate, all ranks (µs) | max layer time (µs) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for prefix, path_label in (("original", "original"), ("barrier", "strict")):
        for region, suffix in (("zero", "zero"), ("inside", "rank3-x0.05"), ("outside", "rank3-x0.4")):
            condition = f"{prefix}-{suffix}"
            combine = statistics.median(
                median(condition, rank, "combine_after_gate_ms") for rank in range(8)
            )
            max_layer = max(median(condition, rank, "layer_ms") for rank in range(8))
            report.append(
                f"| {path_label} | {region} | "
                f"{median(condition, 3, 'injected_delay_ms'):.3f} | "
                f"{median(condition, 3, 'strict_gate_ms'):.3f} | "
                f"{combine:.3f} | {max_layer:.3f} |"
            )
    report.extend(
        [
            "",
            "In the strict path, the post-gate combine duration remains approximately constant while the ready-gate wait shifts across ranks. Thus removing all possible pre-all-ready combine progress does not remove the plateau.",
            "",
            "Negative baseline-subtracted values elsewhere in the experiment remain unmodified; intervals spanning zero are compatible with a zero effect, and their sign is not identified.",
        ]
    )
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
