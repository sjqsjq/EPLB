#!/usr/bin/env python3
"""Combine independently baseline-corrected coarse and dense delay scans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_delay_scan import fit_rank_curves


TEXT_FIELDS = {"condition", "mode", "rank", "delay_ranks", "source_scan"}


def load_rows(path: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {"source_scan": source}
            for key, value in raw.items():
                if key in TEXT_FIELDS:
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    coarse_analysis = json.loads(
        (args.coarse_dir / "delay_scan_analysis.json").read_text()
    )
    dense_analysis = json.loads(
        (args.dense_dir / "delay_scan_analysis.json").read_text()
    )
    if coarse_analysis["total_tokens"] != dense_analysis["total_tokens"]:
        raise ValueError("coarse and dense analyses use different token counts")
    c0_us = float(coarse_analysis["c0_us"])
    rows = load_rows(args.coarse_dir / "delay_scan_summary.csv", "coarse")
    rows += load_rows(args.dense_dir / "delay_scan_summary.csv", "dense")

    layer_fits = fit_rank_curves(rows, "layer_effect_us")
    mechanism_fits = fit_rank_curves(rows, "post_expert_to_done_effect_us")
    all_rank = coarse_analysis["fits"]["original_all_ranks_layer_linear"]
    analysis = {
        "source": [str(args.coarse_dir), str(args.dense_dir)],
        "total_tokens": coarse_analysis["total_tokens"],
        "c0_us": c0_us,
        "estimator": (
            "Each source scan first computes same-block baseline-subtracted medians. "
            "Hinge fits combine those already corrected coarse and dense points; duplicate "
            "ratios are retained as independent scan observations."
        ),
        "negative_effect_policy": coarse_analysis["negative_effect_policy"],
        "fits": {
            "layer_time": layer_fits,
            "post_expert_to_done": mechanism_fits,
            "original_all_ranks_layer_linear": all_rank,
        },
        "source_fits": {
            "coarse": coarse_analysis["fits"],
            "dense": dense_analysis["fits"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "delay_scan_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    csv_path = args.output_dir / "delay_scan_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    colors = {0: "#1f77b4", 3: "#ff7f0e", 7: "#2ca02c"}
    for axis, mode, title in (
        (axes[0], "original", "Original path"),
        (axes[1], "strict_barrier", "Strict GPU-ready barrier"),
    ):
        for rank in (0, 3, 7):
            points = [
                row
                for row in rows
                if row["mode"] == mode
                and row["rank"] == str(rank)
                and row["target_ratio"] > 0
            ]
            axis.scatter(
                [row["actual_delay_us"] for row in points],
                [row["post_expert_to_done_effect_us"] for row in points],
                color=colors[rank],
                s=24,
                alpha=0.70,
                label=f"late rank {rank}",
            )
        fit = mechanism_fits[f"{mode}-pooled"]
        maximum = max(
            row["actual_delay_us"]
            for row in rows
            if row["mode"] == mode and row["rank"] in {"0", "3", "7"}
        )
        xx = np.linspace(0.0, maximum, 300)
        yy = fit["slope"] * np.maximum(0.0, xx - fit["hinge_us"])
        axis.plot(
            xx,
            yy,
            "k--",
            label=f"pooled H={fit['hinge_us']:.1f} µs, slope={fit['slope']:.3f}",
        )
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set(
            title=title,
            xlabel="actual selected-rank delay (µs)",
            ylabel="post-expert-to-done increase (µs)",
        )
        axis.legend(fontsize=8)
    figure_path = args.output_dir / "delay_scan_mechanism_merged.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    original = mechanism_fits["original-pooled"]
    strict = mechanism_fits["strict_barrier-pooled"]
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Merged delay scan (N={analysis['total_tokens']})",
                "",
                f"- Original post-expert hinge: **H={original['hinge_us']:.3f} µs**, "
                f"slope={original['slope']:.3f}",
                f"- Strict-barrier post-expert hinge: **H={strict['hinge_us']:.3f} µs**, "
                f"slope={strict['slope']:.3f}",
                f"- H difference (original − strict): "
                f"**{original['hinge_us'] - strict['hinge_us']:.3f} µs**",
                "",
                "The dense run is interpreted jointly with the full-range run. Fitting the "
                "dense near-hinge window alone is underidentified because it contains too little "
                "post-hinge range to estimate both H and slope reliably.",
                "",
            ]
        )
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")
    print(
        f"N={analysis['total_tokens']} post H original/strict="
        f"{original['hinge_us']:.3f}/{strict['hinge_us']:.3f} us "
        f"slopes={original['slope']:.3f}/{strict['slope']:.3f}"
    )


if __name__ == "__main__":
    main()
