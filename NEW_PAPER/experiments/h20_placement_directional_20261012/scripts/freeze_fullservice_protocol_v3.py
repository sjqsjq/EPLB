#!/usr/bin/env python3
"""Freeze the h20-placement-fullservice-v3 protocol BEFORE any v3 campaign run.

v3 supersedes v2 (configs/followup_v1/fullservice_protocol_v2.json, sha256
0e982cd11fe8fd476ec94e4a66bbc89b8a68cadc4571bb8469de3c6d745163b7). v2 produced
valid identity (3/3) and online (3/3) data but its frozen arm failed 3/3 boots
plus 4/4 diagnostic boots with a deterministic async CUDA 'invalid configuration
argument' crash on the first real-routing forward under the greedy_base map, in
every launcher configuration tested. The historical MoETuner-ILP map and a
rotate map boot and serve fine on the identical stack, isolating the trigger to
the CONTENT of the greedy-descent map (see
results/fullservice_v2/diagnostics/facts.json). No frozen-arm measured data
exists, so re-freezing before measurement preserves the pre-registration
(v1->v2 precedent).

v3 changes vs v2 (all pre-execution):
  1. static reference map regenerated from the SAME sha-locked calibration
     counts (counts_calibration.json from the v2 setup phase) using the
     MoETuner ILP pipeline (solve_ilp.py, G=8, K=16, time-limit 30, mip-gap
     0.005 - the proven-bootable generator lineage of placement_pinned_fair);
  2. the map is boot-validated with a full-size calibration-subset load BEFORE
     freeze (setup validation, no evaluation requests, no measured data);
  3. fresh round-robin 3x3 campaign (labels fs300v3_*): v2 identity/online
     data is retained as reference but never mixed into v3 contrasts.
Everything else (workload split, launchers, bench client, concurrency,
statistics, validity gates, prohibitions, overhead decomposition) is inherited
verbatim from v2.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF = os.path.join(EXP, "configs", "followup_v1")
LAUNCH_DIR = os.path.join(CONF, "fullservice_launchers_v3")
PROTO_PATH = os.path.join(CONF, "fullservice_protocol_v3.json")
SCRIPTS = os.path.join(EXP, "scripts")
SITE_PB = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
MODEL_DIR = "/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8"
WORKLOAD = "/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl"
SPLIT_MANIFEST = os.path.join(EXP, "inputs", "fullservice_v2", "workload", "split_manifest.json")
V2_PROTOCOL = os.path.join(CONF, "fullservice_protocol_v2.json")
V2_PROTOCOL_SHA256 = "0e982cd11fe8fd476ec94e4a66bbc89b8a68cadc4571bb8469de3c6d745163b7"
SETUP = os.path.join(EXP, "results", "fullservice_v2", "setup")
DIAG = os.path.join(EXP, "results", "fullservice_v2", "diagnostics")
SOLVE_ILP = "/workspace/EPLB/OEPLB/baselines/moetuner/src/solve_ilp.py"

FIXED_LAUNCHER = """#!/bin/bash
# h20-placement-fullservice-v3 arm B launcher (SETUP-DERIVED, hash-frozen).
# Verbatim copy of /workspace/logs/launch235b_identity.sh with exactly one
# added flag: --init-expert-location "$1" (the calibration-derived ILP map).
# $1 = json with {"physical_to_logical_map": 94x128}
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \\
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \\
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \\
  --moe-a2a-backend deepep --deepep-mode auto \\
  --moe-runner-backend deep_gemm \\
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \\
  --cuda-graph-max-bs 128 \\
  --port 30000 --host 0.0.0.0 --trust-remote-code \\
  --disable-radix-cache --watchdog-timeout 600 \\
  --init-expert-location "$1"
"""


def sha256_file(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main():
    if os.path.exists(PROTO_PATH):
        print("refusing to overwrite", PROTO_PATH)
        return 3
    if sha256_file(V2_PROTOCOL)[0] != V2_PROTOCOL_SHA256:
        print("v2 protocol hash differs from the recorded superseded hash")
        return 3
    os.makedirs(LAUNCH_DIR, exist_ok=True)
    fixed_p = os.path.join(LAUNCH_DIR, "launch235b_fixedh20.sh")
    if os.path.exists(fixed_p):
        # pre-created for boot validation: must match the frozen content exactly
        with open(fixed_p) as f:
            if f.read() != FIXED_LAUNCHER:
                print("existing v3 frozen launcher differs from FIXED_LAUNCHER")
                return 3
    else:
        with open(fixed_p, "w") as f:
            f.write(FIXED_LAUNCHER)
        os.chmod(fixed_p, 0o755)

    wl_sha, wl_bytes = sha256_file(WORKLOAD)
    split = json.load(open(SPLIT_MANIFEST))
    if split["source"]["sha256"] != wl_sha:
        print("split manifest source hash differs")
        return 3

    v2 = json.load(open(V2_PROTOCOL))

    # ---- setup artifacts (v2 profiling lineage + ILP solve), all sha-locked ----
    setup_files = {
        "counts_calibration": os.path.join(SETUP, "counts_calibration.json"),
        "counts_npz": os.path.join(SETUP, "counts_calibration.npz"),
        "frozen_map": os.path.join(SETUP, "frozen_map_ilp.json"),
        "frozen_map_meta": os.path.join(SETUP, "frozen_map_ilp.meta.json"),
        "ilp_solve_log": os.path.join(SETUP, "ilp_solve_calib.log"),
        "ilp_attempt1_crash_log": os.path.join(SETUP, "ilp_solve_calib_attempt1_crash.log"),
        "diagnostics_facts": os.path.join(DIAG, "facts.json"),
        "boot_validation_log": os.path.join(DIAG, "bootval_ilpmap_server.log"),
        "boot_validation_bench": os.path.join(DIAG, "bootval_calibsub500_bench.json"),
    }
    setup_artifacts = {}
    for key, p in setup_files.items():
        if not os.path.exists(p):
            print(f"missing setup artifact {key}: {p}")
            return 3
        setup_artifacts[key] = {"path": p, "sha256": sha256_file(p)[0],
                                "bytes": os.path.getsize(p)}
    # map must equal the ILP output recorded in the meta sidecar
    meta = json.load(open(setup_files["frozen_map_meta"]))
    fmap_sha = sha256_file(setup_files["frozen_map"])[0]

    deployed = {}
    for name in sorted(os.listdir(SITE_PB)):
        if name.endswith(".py"):
            deployed[name] = sha256_file(os.path.join(SITE_PB, name))[0]
    repo = {}
    repo_src = "/workspace/EPLB/OEPLB/src"
    for name in sorted(os.listdir(repo_src)):
        if name.endswith(".py"):
            repo[name] = sha256_file(os.path.join(repo_src, name))[0]

    def h(p):
        return sha256_file(p)[0]

    model_index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    locked = {
        "identity_launcher": {"path": "/workspace/logs/launch235b_identity.sh", "sha256": h("/workspace/logs/launch235b_identity.sh")},
        "online_launcher": {"path": "/workspace/logs/launch235b_oeplb.sh", "sha256": h("/workspace/logs/launch235b_oeplb.sh")},
        "frozen_arm_launcher": {"path": fixed_p, "sha256": h(fixed_p)},
        "env_file": {"path": "/workspace/logs/env_235b.sh", "sha256": h("/workspace/logs/env_235b.sh")},
        "bench_client": {"path": "/workspace/EPLB/OEPLB/scripts/run_grid_bench.py", "sha256": h("/workspace/EPLB/OEPLB/scripts/run_grid_bench.py")},
        "workload_splitter": {"path": os.path.join(SCRIPTS, "split_fullservice_workload.py"), "sha256": h(os.path.join(SCRIPTS, "split_fullservice_workload.py"))},
        "split_manifest": {"path": SPLIT_MANIFEST, "sha256": h(SPLIT_MANIFEST)},
        "map_generator_ilp": {"path": SOLVE_ILP, "sha256": h(SOLVE_ILP),
                              "invocation": "solve_ilp.py --profile counts_calibration.npz -G 8 -K 16 --time-limit 30 --mip-gap 0.005 --out <abs>/results/fullservice_v2/setup/frozen_map_ilp.json (absolute path required: relative --out crashes in the writer; attempt-1 crash log retained as setup artifact)"},
        "campaign_driver": {"path": os.path.join(SCRIPTS, "run_fullservice_campaign_v3.sh"), "sha256": h(os.path.join(SCRIPTS, "run_fullservice_campaign_v3.sh"))},
        "analyzer": {"path": os.path.join(SCRIPTS, "analyze_fullservice_v3.py"), "sha256": h(os.path.join(SCRIPTS, "analyze_fullservice_v3.py"))},
        "boot_validation_script": {"path": os.path.join(SCRIPTS, "bootval_ilp_map.sh"), "sha256": h(os.path.join(SCRIPTS, "bootval_ilp_map.sh"))},
        "model": {
            "path": MODEL_DIR,
            "config_json_sha256": h(os.path.join(MODEL_DIR, "config.json")),
            "safetensors_index_sha256": h(model_index) if os.path.exists(model_index) else None,
        },
        "deployed_pb_oeplb_sha256": deployed,
        "repo_src_pb_oeplb_sha256": repo,
    }

    n_eval = split["evaluation"]["requests"]
    proto = {
        "spec_id": "h20-placement-fullservice-v3",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_execution",
        "parent_campaign": "h20_placement_directional_20261012",
        "supersedes": {
            "protocol": V2_PROTOCOL,
            "sha256": V2_PROTOCOL_SHA256,
            "reason": (
                "v2 frozen arm: 0/3 valid campaign starts plus 0/4 diagnostic "
                "boots under the greedy_base calibration map - deterministic "
                "async CUDA 'invalid configuration argument' on the first "
                "real-routing forward in every launcher configuration, while "
                "the MoETuner-ILP historical map and a rotate map boot and "
                "serve on the identical stack; trigger isolated to the greedy "
                "map content, root kernel not isolated (async path outside "
                "CUDA_LAUNCH_BLOCKING coverage). v2 identity (3/3) and online "
                "(3/3) data are valid and retained as reference only. No "
                "frozen-arm measured data existed, so v3 re-freeze preserves "
                "pre-registration. All v2 failures and diagnostics retained, "
                "never retried under v2."
            ),
            "v2_analysis_reference": {
                "path": os.path.join(EXP, "results", "fullservice_v2", "analysis.json"),
                "sha256": h(os.path.join(EXP, "results", "fullservice_v2", "analysis.json")),
                "note": "online-identity -29.34s CI[-36.84,-21.83] excludes_zero_negative; frozen contrasts blocked",
            },
        },
        "objective": (
            "Experiment 3 (v3): full-service three-arm validation on one node, "
            "8xH20, EP=8, real Qwen3-235B-A22B-FP8 weights, real SGLang serving "
            "stack, identical serving configuration across arms. Arms: identity "
            "placement vs frozen static placement derived from CALIBRATION "
            "requests only (MoETuner ILP pipeline over the sha-locked v2 "
            "calibration counts) vs original online PB-OEPLB (deployed "
            "controller unchanged). ALL measured benches run on the DISJOINT "
            "EVALUATION requests. All real online record/plan/migration/commit "
            "costs stay inside the online arm's measured wall time; the "
            "controller-log overhead decomposition is reported alongside and "
            "never subtracted. Every independent start is retained."
        ),
        "hardware": {"node": 1, "gpus": 8, "gpu": "NVIDIA H20", "ep_size": 8, "tp": 8, "dp": 8},
        "workload": {
            "source": {
                "path": WORKLOAD,
                "sha256": wl_sha,
                "bytes": wl_bytes,
                "num_requests": split["source"]["lines"],
                "lineage": "identical workload and frozen split as v2",
            },
            "split_rule": split["rule"],
            "split_properties": "deterministic content-hash split; disjoint and exhaustive; uses no performance or timing information",
            "calibration": split["calibration"],
            "evaluation": split["evaluation"],
            "concurrency": 256,
            "usage": "calibration requests: untimed profiling pass (v2 lineage) and untimed pre-freeze boot validation only; evaluation requests: all nine measured arm benches",
        },
        "arms": {
            "identity": {
                "launcher": "/workspace/logs/launch235b_identity.sh",
                "labels": ["fs300v3_id_s1", "fs300v3_id_s2", "fs300v3_id_s3"],
                "init_expert_location": "trivial (default)",
            },
            "frozen": {
                "launcher": fixed_p,
                "labels": ["fs300v3_fx_s1", "fs300v3_fx_s2", "fs300v3_fx_s3"],
                "init_expert_location": "results/fullservice_v2/setup/frozen_map_ilp.json - MoETuner ILP (solve_ilp.py, per-layer balanced clustering + greedy cluster->GPU) over the sha-locked CALIBRATION-aggregate logical counts; boot-validated pre-freeze",
            },
            "online": {
                "launcher": "/workspace/logs/launch235b_oeplb.sh",
                "labels": ["fs300v3_on_s1", "fs300v3_on_s2", "fs300v3_on_s3"],
                "controller": "deployed site-packages pb_oeplb, UNCHANGED (see controller_version_decision, inherited from v2)",
                "flags_note": v2["arms"]["online"]["flags_note"],
            },
        },
        "run_order": {
            "scheme": "round-robin over starts to decorrelate temporal drift",
            "sequence": ["id_s1", "fx_s1", "on_s1", "id_s2", "fx_s2", "on_s2", "id_s3", "fx_s3", "on_s3"],
            "each_run": "fresh server boot (pkill -> gpu-clear-wait -> setsid boot -> poll 'ready to roll'/'Traceback' 600x4s -> full evaluation bench -> collect logs+json -> pkill -> gpu-clear-wait)",
        },
        "setup_phase_untimed": {
            "counts_lineage": "counts_calibration.json produced by the v2 setup phase (profiling boot under --deepep-mode normal + CALIBRATION-request pass + dump_counts.py); sha-locked here and re-verified by the v3 driver before phase 1",
            "npz_conversion": "counts_calibration.json {counts: 94x128} -> counts_calibration.npz {P: int64 94x128} (pure format conversion, deterministic)",
            "map_generation": "solve_ilp.py (MoETuner lineage, sha-locked) per layer: ILP balanced clustering (G=8 clusters x K=16 experts, minimize sum |T_c - T_bar|, time-limit 30s, mip-gap 0.005) + greedy cluster->GPU assignment + per-cluster descending-load slot fill -> frozen_map_ilp.json + meta sidecar + solve log",
            "boot_validation": "pre-freeze validation on CALIBRATION data only: full boot with the v3 frozen-arm launcher (warmup + cuda graph enabled, i.e. the exact measured configuration) followed by a 500-request calibration-subset bench at concurrency 256; logs and bench json sha-locked in setup_artifacts; no evaluation request served, no measured statistic produced",
            "no_timing_use": "setup uses no performance measurement to select anything; the map is a deterministic function of the calibration counts given the frozen solver settings",
            "no_evaluation_leakage": "the evaluation requests are never served before or during setup; the static map cannot encode evaluation-specific hindsight",
            "attempt1_note": "the first solve run completed all 94 layers (mean_imbalance 1.000, 2822.0s) but crashed in the output writer on a relative --out path and was lost; the identical settings were re-run with an absolute path (setup tooling failure, not a measured start; crash log retained and sha-locked)",
        },
        "setup_artifacts": setup_artifacts,
        "metrics": {
            "primary": f"total_time_s per start (fixed {n_eval}-request evaluation workload at concurrency 256 -> full-service wall time; lower is better)",
            "secondary": ["tps", "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms", "tpot_mean_ms", "lat_p99_ms", "ok", "errors"],
            "source": "run_grid_bench.py output json (single-machine client perf_counter; no cross-device timestamp subtraction)",
        },
        "statistics": v2["statistics"],
        "online_net_overhead": v2["online_net_overhead"],
        "controller_version_decision": {
            "deployed_sha256": deployed.get("controller.py"),
            "repo_sha256": repo.get("controller.py"),
            "decision": v2["controller_version_decision"]["decision"],
            "diff_scope": v2["controller_version_decision"]["diff_scope"],
            "equivalence_argument": v2["controller_version_decision"]["equivalence_argument"],
        },
        "scope_note": v2["scope_note"],
        "prohibitions": v2["prohibitions"],
        "interpretation_rules": v2["interpretation_rules"] + [
            "the v3 frozen arm validates a calibration-derived ILP static reference; the greedy-descent map that failed to boot under v2 is an infrastructure-incompatibility casualty, not evidence about its serving performance",
        ],
        "estimated_wall_time_h": [1.0, 1.5],
        "locked_implementation": locked,
        "freezer_sha256": None,
    }
    proto["freezer_sha256"] = h(os.path.abspath(__file__))
    with open(PROTO_PATH, "w") as f:
        json.dump(proto, f, indent=1)
    print(json.dumps({
        "proto": PROTO_PATH,
        "proto_sha256": sha256_file(PROTO_PATH)[0],
        "frozen_map_sha256": fmap_sha,
        "meta_num_layers": meta.get("num_layers"),
        "eval_requests": n_eval,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
