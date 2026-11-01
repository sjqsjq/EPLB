#!/usr/bin/env python3
"""ADW v2.1 r2 -- REAL multiprocess fault-injection test of the confirm-
then-commit protocol (no server, no model, no performance campaign).

Runs `world` real torch.distributed processes (gloo/CPU or nccl/GPU). Each
window mirrors the controller r2 phase structure exactly:
  phase 1  det SUM all_reduce          -- UNCONDITIONAL on every rank
  phase 2  local recoverable scope     -- tripwire / signals / step() / hash
                                          (NO collective inside; injected
                                          faults land here)
  phase 3  confirm MIN + MAX all_reduce-- UNCONDITIONAL, fixed order
  phase 4  symmetric verdict + commit/rollback (streak>=2 -> freeze w_init)

Injected schedule (same design for world=4 and world=8):
  w1 init commit | w2 normal commit | w3 rank-A LOCAL ERROR -> joint hold
  w4 normal commit (streak reset) | w5 rank-B VERSION MISMATCH -> joint hold
  (r1 would have let rank-B skip the det collective; r2 must stay aligned)
  w6 normal commit | w7 rank-C divergent proposal -> non-unanimous hold
  w8 rank-D divergent proposal -> hold, streak=2 -> FREEZE at w_init (all)
  w9 frozen: identical desync records -> commit | w10 ALL ranks error ->
  unanimous sentinel but v_min=0 -> joint hold | w11 frozen commit.

Evidence per window: full per-rank collective sequences (op:size), verdicts,
committed flags, full controller state tuples -- asserted IDENTICAL across
ranks; rank0 dumps JSON. Any deadlock trips the init timeout -> FAIL.
Exit 0 + ALL_FAULT_TESTS_PASS on success. --timing adds per-phase collective
cost accounting (GPU device verification only).
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = "/workspace/EPLB/OEPLB/src"
sys.path.insert(0, SRC)
from window_v2p1 import (WindowControllerV2P1, decision_hash,          # noqa
                         error_sentinel, confirm_commit)
import v2p1_signals as sig                                             # noqa

KNOBS = dict(kappa=1.0, z_lo=10.0, z_hi=20.0, z_urgent=1000.0,
             m_shrink=2, m_grow=3, n_min=0, u_max=4,
             theta_h_low=0.10, theta_h_act=0.15,
             dwell_shrink=2, dwell_grow=2,
             w_uncertain_target=16, urgent_episode_cap=3)
N_WINDOWS = 11
EXPECTED_COMMIT = {1: True, 2: True, 3: False, 4: True, 5: False, 6: True,
                   7: False, 8: False, 9: True, 10: False, 11: True}


def _state_tuple(ctrl, eff, committed_hist, event_hist):
    return (ctrl.W, ctrl.alert_k, ctrl.stable_s, ctrl.dwell, ctrl.dwell_dir,
            ctrl.uncertain_run, ctrl.carry_windows, ctrl.episode_shrinks,
            ctrl.window_seq, ctrl.confirm_fail_streak, ctrl.frozen, eff,
            tuple(committed_hist), tuple(event_hist))


def _injections(rank, wid, world):
    a, b, c, d = 1 % world, (world - 1), (1 % world), (world - 2)
    if wid == 3 and rank == a:
        return {"local_error": True}
    if wid == 5 and rank == b:
        return {"version_mismatch": True}
    if wid == 7 and rank == c:
        return {"h_rate_delta": 0.37}
    if wid == 8 and rank == d:
        return {"h_rate_delta": 0.37}
    if wid == 10:
        return {"local_error": True}
    return {}


def run(rank, world, backend, device, port, out_path, do_timing):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world,
                            timeout=timedelta(seconds=180))
    dev = torch.device(device)
    if device == "cuda":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    if device == "cuda":
        L, P, EP = 94, 128, 8          # real serving shapes
    else:
        L, P, EP = 4, 8, 2
    det_base = ((torch.arange(L * P, dtype=torch.int64).reshape(L, P) % 7) + 1)
    det_base = det_base.to(dev)

    ctrl = WindowControllerV2P1(16, dict(KNOBS), beta=0.5)
    seq = []                            # protocol collective sequence
    committed_hist, event_hist = [], []
    eff = [16]
    evidence = []
    ok_all = True

    for wid in range(1, N_WINDOWS + 1):
        inject = _injections(rank, wid, world)
        snap = ctrl.snapshot()
        wseq0 = len(seq)
        # ---- phase 1: det collective, UNCONDITIONAL ----
        det = det_base.clone()
        seq.append(f"{wid}:det_sum:{det.numel()}")
        dist.all_reduce(det, op=dist.ReduceOp.SUM)
        prop_valid = 0
        rec = None
        h = error_sentinel(wid)
        proposed, event = "error", "confirm_hold"
        # ---- phase 2: local recoverable scope, NO collective inside ----
        try:
            if inject.get("version_mismatch"):
                raise RuntimeError(
                    "mapping version mismatch (INJECTED test fault)")
            if inject.get("local_error"):
                raise RuntimeError("local recoverable error (INJECTED)")
            n_fresh = int(det.sum().item())
            h_w = sig.headroom(det, EP)
            h_rate = sig.headroom_rate(h_w, n_fresh) + inject.get("h_rate_delta", 0.0)
            z = None if wid == 1 else 5.0
            rec = ctrl.step(z, h_rate, n_fresh)
            h = decision_hash(wid, rec)
            prop_valid = 1
        except Exception:
            ctrl.restore(snap)
            rec = None
            prop_valid = 0
            h = error_sentinel(wid)
        # ---- phase 3: confirm collectives, UNCONDITIONAL, MIN then MAX ----
        def reduce_fn(pair, op):
            t = torch.tensor([pair[0], pair[1]], dtype=torch.int64, device=dev)
            seq.append(f"{wid}:c{op}:2")
            dist.all_reduce(
                t, op=(dist.ReduceOp.MIN if op == "min"
                       else dist.ReduceOp.MAX))
            pair[0] = int(t[0].item())
            pair[1] = int(t[1].item())
        commit, h_min, h_max, v_min = confirm_commit(h, prop_valid, reduce_fn)
        # ---- phase 4: symmetric commit ----
        committed = False
        w_after = snap["W"]
        if commit and rec is not None and prop_valid == 1:
            committed = True
            proposed = event = rec["event"]
            w_after = rec["W_after"]
            eff[0] = w_after
            ctrl.confirm_fail_streak = 0
        else:
            ctrl.restore(snap)
            proposed = rec["event"] if rec is not None else "error"
            ctrl.confirm_fail_streak += 1
            w_after = snap["W"]
            if ctrl.confirm_fail_streak >= 2:
                ctrl.on_desync()
                eff[0] = ctrl.W
                event = "confirm_fail_freeze"
            else:
                event = "confirm_hold"
        committed_hist.append(committed)
        event_hist.append(event)
        # ---- harness-level cross-rank evidence gather (not protocol) ----
        mine = {"wid": wid, "rank": rank, "inject": sorted(inject),
                "seq": seq[wseq0:], "verdict": bool(commit),
                "committed": committed, "event": event,
                "proposed": proposed, "W": ctrl.W, "eff": eff[0],
                "streak": ctrl.confirm_fail_streak,
                "v_min": v_min, "h_min_eq_h_max": h_min == h_max,
                "state": _state_tuple(ctrl, eff[0], committed_hist,
                                      event_hist)}
        objs = [None] * world
        dist.all_gather_object(objs, mine)
        if rank == 0:
            same_seq = all(o["seq"] == objs[0]["seq"] for o in objs)
            same_state = all(o["state"] == objs[0]["state"] for o in objs)
            same_verdict = all(o["verdict"] == objs[0]["verdict"] for o in objs)
            same_commit = all(o["committed"] == objs[0]["committed"] for o in objs)
            exp = EXPECTED_COMMIT[wid]
            ok = (same_seq and same_state and same_verdict and same_commit
                  and objs[0]["committed"] == exp)
            ok_all &= ok
            evidence.append({
                "wid": wid, "injected_ranks": {
                    str(o["rank"]): o["inject"] for o in objs if o["inject"]},
                "protocol_collectives": objs[0]["seq"],
                "seq_identical_all_ranks": same_seq,
                "state_identical_all_ranks": same_state,
                "verdict_identical_all_ranks": same_verdict,
                "verdict": objs[0]["verdict"],
                "committed": objs[0]["committed"],
                "committed_expected": exp,
                "event": objs[0]["event"],
                "proposed_event_rank0": objs[0]["proposed"],
                "W": objs[0]["W"], "eff_W": objs[0]["eff"],
                "streak": objs[0]["streak"], "v_min": objs[0]["v_min"],
                "h_unanimous": objs[0]["h_min_eq_h_max"],
                "window_ok": ok,
                "per_rank_states": [list(map(str, o["state"][:12]))
                                    for o in objs],
            })
    # final global assertions (rank0)
    final = None
    if rank == 0:
        last = evidence[-1]
        final_ok = (ok_all
                    and all(e["window_ok"] for e in evidence)
                    and all(e["seq_identical_all_ranks"] for e in evidence)
                    and all(e["state_identical_all_ranks"] for e in evidence)
                    and last["W"] == 16 and last["eff_W"] == 16
                    and all(len(e["protocol_collectives"]) == 3
                            for e in evidence))
        final = {"all_windows_ok": bool(final_ok),
                 "n_windows": N_WINDOWS,
                 "commits_matched_design": all(
                     e["committed"] == e["committed_expected"]
                     for e in evidence),
                 "exactly_3_protocol_collectives_per_window_every_window":
                     all(len(e["protocol_collectives"]) == 3 for e in evidence),
                 "final_W": last["W"], "final_eff_W": last["eff_W"],
                 "frozen_after_double_failure": last["W"] == 16}
    timing = None
    if do_timing:
        # protocol cost accounting on real device shapes (NOT a service or
        # performance measurement): 30 warmup + 50 timed clean windows
        def timed_window():
            t0 = time.perf_counter_ns()
            det = det_base.clone()
            dist.all_reduce(det, op=dist.ReduceOp.SUM)
            n = int(det.sum().item())
            hw = sig.headroom(det, EP)
            hr = sig.headroom_rate(hw, n)
            r = ctrl.step(5.0, hr, n)
            hh = decision_hash(999, r)
            def rfn(pair, op):
                t = torch.tensor([pair[0], pair[1]], dtype=torch.int64,
                                 device=dev)
                dist.all_reduce(t, op=(dist.ReduceOp.MIN if op == "min"
                                       else dist.ReduceOp.MAX))
                pair[0] = int(t[0].item())
                pair[1] = int(t[1].item())
            confirm_commit(hh, 1, rfn)
            if device == "cuda":
                torch.cuda.synchronize(dev)
            return (time.perf_counter_ns() - t0) / 1e3   # us
        for _ in range(30):
            timed_window()
        samples = sorted(timed_window() for _ in range(50))
        # phase breakdown
        def t_det():
            t0 = time.perf_counter_ns()
            det = det_base.clone()
            dist.all_reduce(det, op=dist.ReduceOp.SUM)
            if device == "cuda":
                torch.cuda.synchronize(dev)
            return (time.perf_counter_ns() - t0) / 1e3
        def t_confirm():
            t0 = time.perf_counter_ns()
            def rfn(pair, op):
                t = torch.tensor([pair[0], pair[1]], dtype=torch.int64,
                                 device=dev)
                dist.all_reduce(t, op=(dist.ReduceOp.MIN if op == "min"
                                       else dist.ReduceOp.MAX))
                pair[0] = int(t[0].item())
                pair[1] = int(t[1].item())
            confirm_commit(123, 1, rfn)
            if device == "cuda":
                torch.cuda.synchronize(dev)
            return (time.perf_counter_ns() - t0) / 1e3
        for _ in range(20):
            t_det(); t_confirm()
        d_samples = sorted(t_det() for _ in range(50))
        c_samples = sorted(t_confirm() for _ in range(50))
        local = {
            "window_protocol_us": {"median": samples[25],
                                   "mean": sum(samples) / len(samples),
                                   "p90": samples[45], "max": samples[-1]},
            "det_sum_allreduce_us_median": d_samples[25],
            "confirm_minmax_allreduce_us_median": c_samples[25],
            "det_shape": list(det_base.shape), "backend": backend,
            "device": (torch.cuda.get_device_name(rank)
                       if device == "cuda" else "cpu"),
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        if rank == 0:
            timing = {"per_rank": gathered,
                      "note": ("protocol collective + local compute cost per "
                               "window; accounting only, NOT a service "
                               "throughput measurement")}
    if rank == 0:
        out = {"meta": {"backend": backend, "world": world,
                        "device": device, "torch": torch.__version__,
                        "protocol_rev": "r2",
                        "date": "2026-10-17",
                        "purpose": ("real multiprocess fault-injection "
                                    "acceptance of confirm-then-commit; NOT "
                                    "a performance run")},
               "windows": evidence, "final": final, "timing": timing}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        print("FINAL:", json.dumps(final))
        if timing:
            r0 = timing["per_rank"][0]
            print("TIMING rank0:", json.dumps(r0))
        print("ALL_FAULT_TESTS_PASS" if final["all_windows_ok"]
              else "FAULT_TESTS_FAILED")
    dist.destroy_process_group()
    sys.exit(0 if (rank != 0 or final["all_windows_ok"]) else 1)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    port = free_port()
    mp.spawn(run, args=(a.world, a.backend, a.device, port, a.out, a.timing),
             nprocs=a.world, join=True)
