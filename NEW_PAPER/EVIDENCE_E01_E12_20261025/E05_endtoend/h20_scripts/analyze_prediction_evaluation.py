#!/usr/bin/env python3
"""Pre-specified out-of-sample analysis of the frozen evaluation campaign.

Analysis specification (fixed before any evaluation result was read):

Primary dataset: one unit u = (capture_id, layout_id) for each of the 96
nonidentity evaluation cells. Observed Q_obs(u) = mean over the capture's four
fresh starts of the start-level Q, where the start-level Q is the mean of four
paired block effects (identity - candidate) exactly as defined in the frozen
protocol response_definition. Predicted Q_hat_m(u) comes from the locked models
only (no refit, no evaluation data influence).

Primary metrics per locked model m (unit level, n=96): MAE, RMSE, signed bias
(observed - predicted), Spearman rho, Kendall tau, sign agreement (exact
np.sign match, sign(0)=0). Negative observed effects are preserved exactly;
nothing is floored, clipped, or shifted.

Primary inference: paired loss differences d_{m,m'}(u) = |err_m(u)| - |err_m'(u)|
and per-model losses, with 95% percentile cluster bootstrap CIs resampling the
12 capture units (B=20000, frozen derived seed). Secondary conservative CIs
resample the 3 request batches. Selection regret per capture:
regret_m(c) = max_l Q_obs(c,l) - Q_obs(c, argmax_l Q_hat_m(c,l)) over the eight
nonidentity layouts; frozen tie-break: exact-equal predictions resolve to the
smallest layout index in the frozen order (raw-01, raw-02, padding-01,
padding-02, m2-01, m2-02, random-01, random-02). Oracle reference: mean over
captures of max_l Q_obs(c,l).

Strata (descriptive plus the same paired-loss CIs): request batch (3), layer
(4), layer class {calibration layers 24/44, held-out layers 28/71}. Sensitivity:
start-level units (n=384) with capture-clustered bootstrap.

Secondary diagnostic: locked M2 runner-stage predictions versus same-device
runner.run diagnostic means per (capture, layout, rank) cell (n=864), never
substituted for the primary completion metric.

Interpretation rules (fixed): a CI crossing zero means "compatible with zero",
NOT proof of no effect and NOT equivalence. Superiority of one model over
another is claimed only if the locked out-of-sample paired-loss CI excludes
zero. No H or fence/slack term is ever subtracted from an effect measured as a
global completion-time change. Zero-crossings are never reinterpreted as a
symmetric deadzone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from prediction_common import STUDY_ID, sha256_file, write_json_atomic
from prediction_models import (
    MODEL_IDS,
    direct_effect_features,
    m2_effect_features,
    m2_rank_feature_matrix,
    predict_ridge,
)

LAYOUT_ORDER = (
    "raw-01",
    "raw-02",
    "padding-01",
    "padding-02",
    "m2-01",
    "m2-02",
    "random-01",
    "random-02",
)
CALIBRATION_LAYERS = (24, 44)
HELD_OUT_LAYERS = (28, 71)
BOOTSTRAP_REPLICATES = 20000
CI_LEVEL = 0.95


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=root)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--results-index", type=pathlib.Path, required=True)
    parser.add_argument("--model-lock", type=pathlib.Path, required=True)
    parser.add_argument("--analysis-spec", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def derived_seed(*parts: str) -> int:
    text = "|".join((STUDY_ID, "analysis-bootstrap-v1", *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**62


def check_record(record: dict[str, Any], label: str) -> None:
    path = pathlib.Path(record["path"]).resolve()
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"size mismatch for {label}: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {label}: {path}")


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias_observed_minus_predicted": float(np.mean(residual)),
        "sign_agreement": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "spearman_rho": float(spearmanr(observed, predicted).statistic),
        "kendall_tau": float(kendalltau(observed, predicted).statistic),
        "n": int(observed.size),
    }


def cluster_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    transform: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Percentile CI for a cluster-resampled statistic.

    transform 'mean': statistic is the pooled mean of per-observation values.
    transform 'rmse': statistic is sqrt of the pooled mean of squared values
    (values must already be errors).
    """
    clusters = sorted(set(cluster_ids.tolist()))
    per_cluster = [values[cluster_ids == cluster] for cluster in clusters]
    point_values = np.concatenate(per_cluster)
    if transform == "mean":
        point = float(np.mean(point_values))
    elif transform == "rmse":
        point = float(np.sqrt(np.mean(point_values**2)))
    else:
        raise ValueError(transform)
    cluster_means = np.stack(
        [
            np.mean(group) if transform == "mean" else np.mean(group**2)
            for group in per_cluster
        ]
    )
    draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    stats = cluster_means[draws].mean(axis=1)
    if transform == "rmse":
        stats = np.sqrt(np.maximum(stats, 0.0))
    lo, hi = np.percentile(stats, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi)}


def metrics_with_ci(
    observed: np.ndarray,
    predicted: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    metrics = error_metrics(observed, predicted)
    residual = observed - predicted
    metrics["mae_ci"] = cluster_bootstrap_ci(np.abs(residual), cluster_ids, rng, transform="mean")
    metrics["rmse_ci"] = cluster_bootstrap_ci(residual, cluster_ids, rng, transform="rmse")
    metrics["bias_ci"] = cluster_bootstrap_ci(residual, cluster_ids, rng, transform="mean")
    metrics["sign_agreement_ci"] = cluster_bootstrap_ci(
        (np.sign(observed) == np.sign(predicted)).astype(np.float64),
        cluster_ids,
        rng,
        transform="mean",
    )
    return metrics


def interpretation(ci: dict[str, float]) -> str:
    if ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0:
        return "cluster_bootstrap_CI_excludes_zero"
    return "compatible_with_zero_not_proof_of_no_effect_and_not_equivalence"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    analyzer_path = pathlib.Path(__file__).resolve()

    spec = json.loads(args.analysis_spec.resolve().read_text())
    if spec.get("status") != "analysis_specification_frozen_before_evaluation_results":
        raise ValueError("analysis spec is not in pre-results frozen status")
    if sha256_file(analyzer_path) != spec["analyzer"]["sha256"]:
        raise ValueError("analyzer differs from the pre-frozen analysis specification")

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("study_id") != STUDY_ID or protocol.get("split") != "evaluation":
        raise ValueError("analysis accepts the evaluation protocol only")
    if protocol.get("status") != "frozen_before_evaluation_timing":
        raise ValueError("evaluation protocol is not frozen")
    for name in (
        "prediction_common_record",
        "prediction_models_record",
        "prediction_replay_record",
        "replay_placement_record",
    ):
        check_record(protocol["implementation"][name], name)

    lock_path = args.model_lock.resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "models_locked_before_evaluation_layout_generation_and_timing":
        raise ValueError("model lock is not in locked status")
    if protocol.get("model_lock", {}).get("sha256") != sha256_file(lock_path):
        raise ValueError("protocol references a different model lock")
    for name, record in lock["locked_implementation"].items():
        check_record(record, f"lock:{name}")

    index_path = args.results_index.resolve()
    index = json.loads(index_path.read_text())
    if index.get("study_id") != STUDY_ID or index.get("split") != "evaluation":
        raise ValueError("analysis accepts the evaluation timing index only")
    if index.get("status") != "all_evaluation_timing_results_validated_and_hash_indexed":
        raise ValueError("evaluation timing index is not validated")
    if index.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("timing index references a different evaluation protocol")
    if len(index.get("results", [])) != 48:
        raise ValueError("evaluation index start count differs")

    # Layout manifests, hash-verified through the protocol start records.
    manifests: dict[str, dict[str, Any]] = {}
    for start in protocol["starts"]:
        record = start["layout_manifest"]
        capture_id = start["capture_id"]
        if capture_id in manifests:
            continue
        check_record(record, f"layout_manifest:{capture_id}")
        manifest = json.loads(pathlib.Path(record["path"]).read_text())
        if manifest.get("capture_id") != capture_id or manifest.get("split") != "evaluation":
            raise ValueError(f"manifest header differs: {capture_id}")
        manifests[capture_id] = manifest

    # Locked predictions per (capture, layout), computed once.
    predictions: dict[str, dict[str, dict[str, float]]] = {}
    for capture_id, manifest in manifests.items():
        layout_by_id = {layout["layout_id"]: layout for layout in manifest["layouts"]}
        identity = layout_by_id["identity"]["descriptors"]
        per_layout: dict[str, dict[str, float]] = {}
        for layout_id in LAYOUT_ORDER:
            candidate = layout_by_id[layout_id]["descriptors"]
            values: dict[str, float] = {}
            for model_id in MODEL_IDS:
                if model_id == "M2_grouped_gemm":
                    names, features = m2_effect_features(
                        identity_descriptors=identity,
                        candidate_descriptors=candidate,
                        runner_model=lock["models"][model_id]["runner_compute_model"],
                    )
                else:
                    names, features = direct_effect_features(
                        model_id,
                        identity_descriptors=identity,
                        candidate_descriptors=candidate,
                    )
                completion = lock["models"][model_id]["completion_model"]
                values[model_id] = float(predict_ridge(completion, features, names))
            per_layout[layout_id] = values
        predictions[capture_id] = per_layout

    # Observed start-level Q values from hash-verified raw results.
    start_q: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    runner_observed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    start_specs = {start["start_id"]: start for start in protocol["starts"]}
    blocks_per_start = int(protocol["timing"]["blocks_per_start"])
    for record in index["results"]:
        check_record(record, f"raw:{record['start_id']}")
        raw = json.loads(pathlib.Path(record["path"]).read_text())
        start = start_specs[raw["start_id"]]
        if raw["capture_id"] != start["capture_id"] or raw["split"] != "evaluation":
            raise ValueError(f"raw metadata differs: {raw['start_id']}")
        primary_by_block: dict[int, dict[str, float]] = defaultdict(dict)
        for timing in raw["primary_timing_records"]:
            primary_by_block[int(timing["block"])][timing["layout_id"]] = float(
                timing["completion_fenced_us_per_forward"]
            )
        if sorted(primary_by_block) != list(range(blocks_per_start)):
            raise ValueError(f"primary blocks differ: {raw['start_id']}")
        for layout_id in LAYOUT_ORDER:
            effects = [
                values["identity"] - values[layout_id]
                for _, values in sorted(primary_by_block.items())
            ]
            start_q[(raw["capture_id"], layout_id)][raw["start_id"]] = float(
                np.mean(effects)
            )
        for timing in raw["diagnostic_timing_records"]:
            layout_id = timing["layout_id"]
            for rank, rank_record in enumerate(timing["ranks"]):
                runner_observed[(raw["capture_id"], layout_id, rank)].extend(
                    float(value) for value in rank_record["runner_run_iteration_us"]
                )
    for key, starts in start_q.items():
        if len(starts) != 4:
            raise ValueError(f"start cardinality differs for {key}")

    capture_ids = sorted(manifests)
    capture_meta = {
        capture_id: (manifests[capture_id]["batch_id"], int(manifests[capture_id]["layer_id"]))
        for capture_id in capture_ids
    }

    # Unit-level arrays in frozen order: captures x layouts.
    units = [(capture_id, layout_id) for capture_id in capture_ids for layout_id in LAYOUT_ORDER]
    q_obs_unit = np.asarray(
        [np.mean(list(start_q[unit].values())) for unit in units], dtype=np.float64
    )
    q_obs_start_level = np.asarray(
        [start_q[unit][start_id] for unit in units for start_id in sorted(start_q[unit])],
        dtype=np.float64,
    )
    start_clusters = np.asarray(
        [unit[0] for unit in units for _ in range(4)], dtype=object
    )
    unit_clusters = np.asarray([unit[0] for unit in units], dtype=object)
    batch_clusters = np.asarray([capture_meta[unit[0]][0] for unit in units], dtype=object)
    pred_unit = {
        model_id: np.asarray(
            [predictions[unit[0]][unit[1]][model_id] for unit in units], dtype=np.float64
        )
        for model_id in MODEL_IDS
    }
    pred_start_level = {
        model_id: np.repeat(pred_unit[model_id], 4) for model_id in MODEL_IDS
    }

    rng = np.random.default_rng(derived_seed("evaluation-unit"))
    per_model: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        per_model[model_id] = {
            "unit_level": metrics_with_ci(
                q_obs_unit, pred_unit[model_id], unit_clusters, rng
            ),
            "start_level_sensitivity": metrics_with_ci(
                q_obs_start_level, pred_start_level[model_id], start_clusters, rng
            ),
        }

    pairwise: dict[str, Any] = {}
    rng_pair = np.random.default_rng(derived_seed("evaluation-pairwise"))
    for i, first in enumerate(MODEL_IDS):
        for second in MODEL_IDS[i + 1 :]:
            loss_first = np.abs(q_obs_unit - pred_unit[first])
            loss_second = np.abs(q_obs_unit - pred_unit[second])
            delta = loss_first - loss_second
            ci_captures = cluster_bootstrap_ci(delta, unit_clusters, rng_pair, transform="mean")
            rng_batch = np.random.default_rng(derived_seed("evaluation-pairwise-batch"))
            ci_batches = cluster_bootstrap_ci(delta, batch_clusters, rng_batch, transform="mean")
            start_delta = np.repeat(delta, 4)
            rng_start = np.random.default_rng(derived_seed("evaluation-pairwise-start"))
            ci_starts = cluster_bootstrap_ci(start_delta, start_clusters, rng_start, transform="mean")
            pairwise[f"{first}_minus_{second}"] = {
                "definition": "|error(first)| - |error(second)| at unit level; negative favors first",
                "mean_delta_us": float(np.mean(delta)),
                "capture_cluster_ci": ci_captures,
                "batch_cluster_ci_secondary": ci_batches,
                "start_level_ci_sensitivity": ci_starts,
                "interpretation": interpretation(ci_captures),
            }

    # Regret with the frozen tie-break rule.
    regret: dict[str, Any] = {}
    q_matrix = q_obs_unit.reshape(len(capture_ids), len(LAYOUT_ORDER))
    oracle = q_matrix.max(axis=1)
    for model_id in MODEL_IDS:
        pred_matrix = pred_unit[model_id].reshape(len(capture_ids), len(LAYOUT_ORDER))
        selected_index = np.argmax(pred_matrix, axis=1)  # first maximum = frozen order tie-break
        selected_regret = oracle - q_matrix[np.arange(len(capture_ids)), selected_index]
        rng_regret = np.random.default_rng(derived_seed(f"evaluation-regret-{model_id}"))
        regret[model_id] = {
            "per_capture_selected_layout": [
                LAYOUT_ORDER[index] for index in selected_index
            ],
            "per_capture_regret_us": [float(value) for value in selected_regret],
            "mean_regret_us": cluster_bootstrap_ci(
                selected_regret, np.asarray(capture_ids, dtype=object), rng_regret, transform="mean"
            ),
        }
    oracle_summary = {
        "mean_best_observed_q_us": cluster_bootstrap_ci(
            oracle, np.asarray(capture_ids, dtype=object),
            np.random.default_rng(derived_seed("evaluation-oracle")), transform="mean"
        ),
        "identity_reference_q_us": 0.0,
    }

    # Strata: descriptive metrics plus capture-clustered paired-loss CIs.
    strata: dict[str, Any] = {}
    layer_class = {
        capture_id: (
            "calibration_layers_24_44"
            if capture_meta[capture_id][1] in CALIBRATION_LAYERS
            else "held_out_layers_28_71"
        )
        for capture_id in capture_ids
    }
    stratum_definitions: dict[str, dict[str, Any]] = {}
    for batch_id in sorted(set(capture_meta[c][0] for c in capture_ids)):
        stratum_definitions[f"batch:{batch_id}"] = {
            capture_id: capture_meta[capture_id][0] == batch_id for capture_id in capture_ids
        }
    for layer in sorted(capture_meta[c][1] for c in capture_ids):
        stratum_definitions[f"layer:{layer}"] = {
            capture_id: capture_meta[capture_id][1] == layer for capture_id in capture_ids
        }
    for class_name in ("calibration_layers_24_44", "held_out_layers_28_71"):
        stratum_definitions[f"layer_class:{class_name}"] = {
            capture_id: layer_class[capture_id] == class_name for capture_id in capture_ids
        }
    for name, membership in stratum_definitions.items():
        mask = np.asarray([membership[unit[0]] for unit in units], dtype=bool)
        entry: dict[str, Any] = {}
        rng_stratum = np.random.default_rng(derived_seed(f"evaluation-stratum-{name}"))
        for model_id in MODEL_IDS:
            entry[model_id] = error_metrics(q_obs_unit[mask], pred_unit[model_id][mask])
        for i, first in enumerate(MODEL_IDS):
            for second in MODEL_IDS[i + 1 :]:
                delta = (
                    np.abs(q_obs_unit - pred_unit[first])
                    - np.abs(q_obs_unit - pred_unit[second])
                )[mask]
                clusters = unit_clusters[mask]
                ci = cluster_bootstrap_ci(delta, clusters, rng_stratum, transform="mean")
                entry[f"{first}_minus_{second}"] = {
                    "mean_delta_us": float(np.mean(delta)),
                    "capture_cluster_ci": ci,
                    "interpretation": interpretation(ci),
                }
        entry["n_captures"] = int(sum(membership.values()))
        strata[name] = entry

    # Secondary: locked M2 runner-stage diagnostic (never substituted for primary).
    runner_model = lock["models"]["M2_grouped_gemm"]["runner_compute_model"]
    runner_rows: list[tuple[str, str, int, float, float]] = []
    for capture_id in capture_ids:
        layout_by_id = {
            layout["layout_id"]: layout
            for layout in manifests[capture_id]["layouts"]
        }
        for layout_id in ("identity",) + LAYOUT_ORDER:
            names, matrix = m2_rank_feature_matrix(layout_by_id[layout_id]["descriptors"])
            predicted = predict_ridge(runner_model, matrix, names)
            for rank in range(8):
                values = runner_observed[(capture_id, layout_id, rank)]
                if len(values) != 4 * int(protocol["timing"]["diagnostic_blocks_per_start"]) * int(
                    protocol["timing"]["timed_forwards_per_activation"]
                ):
                    raise ValueError(f"runner diagnostic cardinality differs: {capture_id}/{layout_id}/{rank}")
                runner_rows.append(
                    (capture_id, layout_id, rank, float(np.mean(values)), float(predicted[rank]))
                )
    runner_observed_mean = np.asarray([row[3] for row in runner_rows])
    runner_predicted_mean = np.asarray([row[4] for row in runner_rows])
    runner_clusters = np.asarray([row[0] for row in runner_rows], dtype=object)
    rng_runner = np.random.default_rng(derived_seed("evaluation-runner-diagnostic"))
    runner_diagnostic = {
        "metric": "absolute runner.run microseconds per forward, same-device CUDA events, diagnostic only",
        "n_cells": len(runner_rows),
        **metrics_with_ci(
            runner_observed_mean, runner_predicted_mean, runner_clusters, rng_runner
        ),
    }

    # Descriptive: realized Q of the two locked-M2-selected layouts per capture.
    m2_selections = {
        capture_id: {
            layout_id: {
                "locked_m2_predicted_q_us": float(
                    predictions[capture_id][layout_id]["M2_grouped_gemm"]
                ),
                "observed_q_us_mean_over_starts": float(
                    np.mean(list(start_q[(capture_id, layout_id)].values()))
                ),
                "per_start_q_us": {
                    start_id: start_q[(capture_id, layout_id)][start_id]
                    for start_id in sorted(start_q[(capture_id, layout_id)])
                },
            }
            for layout_id in ("m2-01", "m2-02")
        }
        for capture_id in capture_ids
    }

    output = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "out_of_sample_evaluation_analysis_complete_prespecified_only",
        "analyzed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_spec": file_record(args.analysis_spec.resolve()),
        "analyzer": file_record(analyzer_path),
        "protocol": file_record(protocol_path),
        "results_index": file_record(index_path),
        "model_lock": file_record(lock_path),
        "units": {
            "unit_definition": "(capture_id, layout_id); Q_obs = mean over 4 starts of mean over 4 paired block effects",
            "n_units": len(units),
            "n_start_level_observations": int(q_obs_start_level.size),
            "captures": capture_ids,
            "capture_metadata": {
                capture_id: {
                    "batch_id": capture_meta[capture_id][0],
                    "layer_id": capture_meta[capture_id][1],
                    "layer_class": layer_class[capture_id],
                }
                for capture_id in capture_ids
            },
        },
        "observed_q_unit_us": {
            f"{unit[0]}|{unit[1]}": float(value) for unit, value in zip(units, q_obs_unit)
        },
        "predicted_q_unit_us": {
            model_id: {
                f"{unit[0]}|{unit[1]}": float(value)
                for unit, value in zip(units, pred_unit[model_id])
            }
            for model_id in MODEL_IDS
        },
        "per_model": per_model,
        "pairwise_paired_loss": pairwise,
        "selection_regret": regret,
        "oracle_reference": oracle_summary,
        "strata": strata,
        "m2_runner_diagnostic_secondary": runner_diagnostic,
        "locked_m2_selected_layouts_descriptive": m2_selections,
        "interpretation_rules": {
            "ci_crossing_zero": (
                "compatible with zero; not proof of no effect and not equivalence"
            ),
            "superiority_claim_requires": (
                "locked out-of-sample paired-loss cluster-bootstrap CI excluding zero"
            ),
            "negative_effects": "preserved exactly; never floored or clipped",
            "no_H_or_fence_subtraction": True,
            "zero_crossing_not_a_deadzone": True,
            "diagnostic_never_substitutes_primary": True,
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "ci_level": CI_LEVEL,
            "primary_clusters": "12 capture units",
            "secondary_clusters": "3 request batches; 48 start-level sensitivity",
            "seed_derivation": "uint64 big-endian SHA256(study|analysis-bootstrap-v1|purpose) mod 2^62",
        },
    }
    write_json_atomic(args.output.resolve(), output)
    print(f"wrote {args.output.resolve()}")
    print(f"analysis_sha256={sha256_file(args.output.resolve())}")
    for model_id in MODEL_IDS:
        summary = per_model[model_id]["unit_level"]
        print(
            f"{model_id}: MAE={summary['mae']:.1f} "
            f"[{summary['mae_ci']['ci_low']:.1f},{summary['mae_ci']['ci_high']:.1f}] "
            f"bias={summary['bias_observed_minus_predicted']:.1f} "
            f"rho={summary['spearman_rho']:.3f} tau={summary['kendall_tau']:.3f} "
            f"sign={summary['sign_agreement']:.3f}"
        )
    for name, entry in pairwise.items():
        ci = entry["capture_cluster_ci"]
        print(
            f"{name}: mean_delta={entry['mean_delta_us']:.1f} "
            f"CI=[{ci['ci_low']:.1f},{ci['ci_high']:.1f}] -> {entry['interpretation']}"
        )
    for model_id in MODEL_IDS:
        entry = regret[model_id]["mean_regret_us"]
        print(
            f"regret[{model_id}]: {entry['point']:.1f} "
            f"CI=[{entry['ci_low']:.1f},{entry['ci_high']:.1f}]"
        )


if __name__ == "__main__":
    main()
