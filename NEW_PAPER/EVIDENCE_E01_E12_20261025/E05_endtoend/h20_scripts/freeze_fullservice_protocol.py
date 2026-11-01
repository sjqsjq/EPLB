#!/usr/bin/env python3
"""Freeze the h20-placement-fullservice-v1 protocol BEFORE any campaign run.

Creates the two derived launcher scripts (arm-B fixed-map launcher and the
setup-only profiling launcher) from verbatim identity-launcher content, hashes
every frozen input, and writes configs/followup_v1/fullservice_protocol.json.
Refuses to overwrite anything that already exists.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF = os.path.join(EXP, "configs", "followup_v1")
LAUNCH_DIR = os.path.join(CONF, "fullservice_launchers")
PROTO_PATH = os.path.join(CONF, "fullservice_protocol.json")
SCRIPTS = os.path.join(EXP, "scripts")
SITE_PB = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
MODEL_DIR = "/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8"
WORKLOAD = "/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl"

FIXED_LAUNCHER = """#!/bin/bash
# h20-placement-fullservice-v1 arm B launcher (SETUP-DERIVED, hash-frozen).
# Verbatim copy of /workspace/logs/launch235b_identity.sh with exactly one
# added flag: --init-expert-location "$1" (the frozen optimized map JSON).
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

PROFILE_LAUNCHER = """#!/bin/bash
# h20-placement-fullservice-v1 SETUP-ONLY profiling launcher (untimed).
# Verbatim identity launcher + expert-distribution recorder (stat mode).
# Recorded logical routing counts are placement-independent (router gate
# weights are frozen); identity placement is used so no swaps occur.
. /workspace/logs/env_235b.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/logs/recdump
exec python3 -m sglang.launch_server \\
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \\
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \\
  --moe-a2a-backend deepep --deepep-mode auto \\
  --moe-runner-backend deep_gemm \\
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \\
  --cuda-graph-max-bs 128 \\
  --port 30000 --host 0.0.0.0 --trust-remote-code \\
  --disable-radix-cache --watchdog-timeout 600 \\
  --expert-distribution-recorder-mode stat
"""


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest(), n


def main():
    if os.path.exists(PROTO_PATH):
        print("refusing to overwrite", PROTO_PATH)
        return 3
    os.makedirs(LAUNCH_DIR, exist_ok=True)
    fixed_p = os.path.join(LAUNCH_DIR, "launch235b_fixedh20.sh")
    prof_p = os.path.join(LAUNCH_DIR, "launch235b_profileh20.sh")
    for p, content in ((fixed_p, FIXED_LAUNCHER), (prof_p, PROFILE_LAUNCHER)):
        if os.path.exists(p):
            print("refusing to overwrite", p)
            return 3
        with open(p, "w") as f:
            f.write(content)
        os.chmod(p, 0o755)

    wl_sha, wl_bytes = sha256_file(WORKLOAD)

    deployed = {}
    for name in sorted(os.listdir(SITE_PB)):
        if name.endswith(".py"):
            sha, _ = sha256_file(os.path.join(SITE_PB, name))
            deployed[name] = sha
    repo = {}
    repo_src = "/workspace/EPLB/OEPLB/src"
    for name in sorted(os.listdir(repo_src)):
        if name.endswith(".py"):
            sha, _ = sha256_file(os.path.join(repo_src, name))
            repo[name] = sha

    def h(p):
        sha, _ = sha256_file(p)
        return sha

    model_index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    locked = {
        "identity_launcher": {"path": "/workspace/logs/launch235b_identity.sh", "sha256": h("/workspace/logs/launch235b_identity.sh")},
        "online_launcher": {"path": "/workspace/logs/launch235b_oeplb.sh", "sha256": h("/workspace/logs/launch235b_oeplb.sh")},
        "frozen_arm_launcher": {"path": fixed_p, "sha256": h(fixed_p)},
        "profiling_launcher": {"path": prof_p, "sha256": h(prof_p)},
        "env_file": {"path": "/workspace/logs/env_235b.sh", "sha256": h("/workspace/logs/env_235b.sh")},
        "bench_client": {"path": "/workspace/EPLB/OEPLB/scripts/run_grid_bench.py", "sha256": h("/workspace/EPLB/OEPLB/scripts/run_grid_bench.py")},
        "recorder_tool": {"path": "/workspace/EPLB/OEPLB/repro/dump_counts.py", "sha256": h("/workspace/EPLB/OEPLB/repro/dump_counts.py")},
        "map_generator": {"path": os.path.join(SCRIPTS, "gen_frozen_map.py"), "sha256": h(os.path.join(SCRIPTS, "gen_frozen_map.py"))},
        "campaign_driver": {"path": os.path.join(SCRIPTS, "run_fullservice_campaign.sh"), "sha256": h(os.path.join(SCRIPTS, "run_fullservice_campaign.sh"))},
        "optimizer_lineage": {
            "generate_stopping_layouts.py": h(os.path.join(SCRIPTS, "generate_stopping_layouts.py")),
            "prediction_common.py": h(os.path.join(SCRIPTS, "prediction_common.py")),
        },
        "model": {
            "path": MODEL_DIR,
            "config_json_sha256": h(os.path.join(MODEL_DIR, "config.json")),
            "safetensors_index_sha256": h(model_index) if os.path.exists(model_index) else None,
        },
        "deployed_pb_oeplb_sha256": deployed,
        "repo_src_pb_oeplb_sha256": repo,
    }

    proto = {
        "spec_id": "h20-placement-fullservice-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_execution",
        "parent_campaign": "h20_placement_directional_20261012",
        "objective": (
            "Experiment 3: fixed-workload full-service validation on one node, "
            "8xH20, EP=8, real Qwen3-235B-A22B-FP8 weights, real SGLang serving "
            "stack. Three arms on the SAME frozen request workload: identity "
            "placement vs frozen offline-optimized placement (campaign optimizer "
            "lineage) vs online PB-OEPLB. Multiple independent server starts per "
            "arm. Online net overhead reported explicitly from controller logs, "
            "never subtracted from measured arm contrasts."
        ),
        "hardware": {"node": 1, "gpus": 8, "gpu": "NVIDIA H20", "ep_size": 8, "tp": 8, "dp": 8},
        "workload": {
            "path": WORKLOAD,
            "sha256": wl_sha,
            "bytes": wl_bytes,
            "num_requests": 16000,
            "concurrency": 256,
            "lineage": "identical workload/concurrency as prior driver35/driver39 H20 runs",
        },
        "arms": {
            "identity": {
                "launcher": "/workspace/logs/launch235b_identity.sh",
                "labels": ["fs300_id_s1", "fs300_id_s2", "fs300_id_s3"],
                "init_expert_location": "trivial (default)",
            },
            "frozen": {
                "launcher": fixed_p,
                "labels": ["fs300_fx_s1", "fs300_fx_s2", "fs300_fx_s3"],
                "init_expert_location": "results/fullservice_v1/setup/frozen_map.json generated in setup phase by gen_frozen_map.py (greedy_base lineage over workload-aggregate logical counts)",
            },
            "online": {
                "launcher": "/workspace/logs/launch235b_oeplb.sh",
                "labels": ["fs300_on_s1", "fs300_on_s2", "fs300_on_s3"],
                "controller": "deployed site-packages pb_oeplb, UNCHANGED (see controller_version_decision)",
                "flags_note": "verbatim H20-validated production flags: threshold 1.02, sync-window 16, adaptive-window, adaptive-decay, window-floor 8, window-shift-confirm 2, min-prefill-tokens 256, min-swap-ops 8, max-total-ops 300, max-total-swap-layers 94, max-swaps-per-layer 64, decay 0.5",
            },
        },
        "run_order": {
            "scheme": "round-robin over starts to decorrelate temporal drift",
            "sequence": ["id_s1", "fx_s1", "on_s1", "id_s2", "fx_s2", "on_s2", "id_s3", "fx_s3", "on_s3"],
            "each_run": "fresh server boot (pkill -> gpu-clear-wait -> setsid boot -> poll 'ready to roll'/'Traceback' 600x4s -> full bench -> collect logs+json -> pkill -> gpu-clear-wait)",
        },
        "setup_phase_untimed": {
            "profiling": "boot profiling launcher; POST /start_expert_distribution_record; full 16000-request pass via run_grid_bench.py fs300_profile (result retained as provenance only, excluded from analysis); POST /stop+/dump; dump_counts.py -> results/fullservice_v1/setup/counts_fullservice.json",
            "map_generation": "gen_frozen_map.py: per layer greedy_base (frozen lexicographic-descent optimizer from experiments 1/2) over the 94x128 workload-aggregate logical counts -> frozen_map.json + frozen_map_report.json (hashes, per-layer r_identity/r_greedy/moves)",
            "placement_independence_note": "logical routing counts are produced by frozen router gate weights and do not depend on physical placement; profiling on identity is canonical",
            "no_timing_use": "setup uses no performance measurement to select anything; map is a deterministic function of counts",
        },
        "metrics": {
            "primary": "total_time_s per start (fixed 16000-request workload at concurrency 256 -> full-service wall time; lower is better)",
            "secondary": ["tps", "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms", "tpot_mean_ms", "lat_p99_ms", "ok", "errors"],
            "source": "run_grid_bench.py output json (single-machine client perf_counter; no cross-device timestamp subtraction)",
        },
        "statistics": {
            "design": "3 independent server starts per arm; every start enters its arm mean; no best-start or repeat selection",
            "contrasts": ["frozen-identity", "online-identity", "online-frozen"],
            "test": "Welch two-sample t on per-start total_time_s (unequal variance), 95% CI, scipy.stats",
            "verdict_language": "frozen trichotomy: CI excludes zero -> directional effect; CI crosses zero -> 'compatible with zero' (never 'equivalence', never 'no effect proven'); negative point estimates preserved exactly as measured",
            "validity_gate": "a start is valid iff boot ok, no Traceback in server log, bench rc=0, ok==16000, errors==0; invalid starts retained with status, never retried under this frozen campaign; if an arm has <2 valid starts its contrasts are reported blocked",
        },
        "online_net_overhead": {
            "extraction": "from arm-online server logs only: (a) sum of '(X ms)' over all '[PB-OEPLB] K swap(s) done' lines and final cumulative 'total=N' swap count; (b) last '[PB-OEPLB-PROF] window#' line cumulative record/allreduce/planbuild/finalize ms and avg_record_us; (c) '[PB-OEPLB] swap budget exhausted' line spent/elapsed if present; (d) '[PB-OEPLB-TIMING]' async_swapper lines if present; (e) window decision counts (executed/skipped/deferred/below_min_prefill) from '[PB-OEPLB] window#' lines",
            "reporting": "overhead_fraction = cumulative_swap_wall_s / mean arm-online total_time_s; reported as a mechanism decomposition alongside (NEVER subtracted from) the measured online-identity and online-frozen contrasts, because arm contrasts are already global completion-time measurements",
            "caveat_frozen": "async swapper overlaps migration with serving, so swap wall time is not purely additive latency; the fraction is an upper-bound wall-share, stated as such",
        },
        "controller_version_decision": {
            "deployed_sha256": deployed.get("controller.py"),
            "repo_sha256": repo.get("controller.py"),
            "decision": "run the deployed site-packages version UNCHANGED: preserves lineage with prior H20 driver35/38/39 results and avoids mutating the environment inside a frozen campaign",
            "diff_scope": [
                "deployed-only: [PB-OEPLB-HOOKDIAG] diagnostic counters/logging (no decision-path effect)",
                "repo-only: adaptive-window shrink clamp new_window=min(W,max(floor,W//2)) vs deployed max(floor,W//2)",
            ],
            "equivalence_argument": "Under the frozen online flags (sync_window=16, window_floor=8) the invariant W>=floor holds at all times (init 16>=8; shrink maps W->max(floor,W//2)>=floor; grow maps W->min(128,2W)). For any W>=floor: max(floor,W//2) <= max(W,W//2) = W, hence the deployed shrink is non-expanding and min(W,max(floor,W//2)) = max(floor,W//2). The two controller versions are therefore functionally identical on this campaign's flag set; the repo clamp differs only when floor>W (the GB200 repro case), unreachable here. The ratio-jump shrink path (max(8,W//2)) is byte-identical in both versions.",
        },
        "prohibitions": [
            "push_without_separate_instruction",
            "best_start_or_repeat_selection",
            "retry_failed_start_same_campaign",
            "overwrite_existing_raw_result",
            "post_freeze_flag_workload_or_launcher_change",
            "subtract_overhead_from_measured_contrast",
            "artificial_delay_floor",
            "differential_delay",
            "sleep_or_spin_as_intervention",
            "mid_layer_collective_or_gate",
            "cross_device_timestamp_subtraction",
        ],
        "interpretation_rules": [
            "arm contrasts are whole-service effects under one fixed workload; they do not isolate MoE-layer placement effects",
            "the frozen arm uses workload-aggregate counts (hindsight static optimum of the campaign optimizer lineage); it is an upper reference for static placement, not a deployable online policy",
            "the online arm carries decision+swap overhead that the frozen arm does not; the overhead decomposition explains mechanism, it never rewrites the measured contrast",
            "with n=3 starts per arm, CIs are wide by construction; a CI crossing zero means 'compatible with zero' at this sample size",
            "negative effects are preserved exactly; no sign is flipped or discarded",
        ],
        "estimated_wall_time_h": [4.5, 6.0],
        "locked_implementation": locked,
        "freezer_sha256": None,
    }
    proto["freezer_sha256"] = h(os.path.abspath(__file__))
    with open(PROTO_PATH, "w") as f:
        json.dump(proto, f, indent=1)
    print(json.dumps({
        "proto": PROTO_PATH,
        "proto_sha256": sha256_file(PROTO_PATH)[0],
        "workload_sha256": wl_sha,
        "controller_deployed": deployed.get("controller.py"),
        "controller_repo": repo.get("controller.py"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
