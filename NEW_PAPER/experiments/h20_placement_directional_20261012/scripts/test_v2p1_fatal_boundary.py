#!/usr/bin/env python3
"""ADW v2.1 r3 -- communicator-fatal boundary acceptance test (REAL
multiprocess, gloo/CPU, world=2).

Boundary under test (user r2-review item 3):
  RECOVERABLE   = phase-2 local computation exception on a HEALTHY
                  communicator -> joint HOLD, serving continues
                  (covered by the 11-window fault-injection test).
  NON-RECOVERABLE = an exception from the collectives themselves
                  (phase-1 det SUM / phase-3 confirm MIN/MAX) -> FATAL:
                  log + re-raise -> the scheduler process dies -> the run
                  terminates (framework fault propagation / job restart).
                  NO ordinary confirm continuation, NO commit on a suspect
                  communicator (per NCCL fatal-error guidance: abort
                  operations; recovery only by rebuilding the communicator).

Scenario:
  w1: both ranks normal -> joint COMMIT (baseline: protocol healthy).
  w2: rank 1's det SUM all_reduce raises (simulated communicator fatal,
      NCCL async-error class). Faithful outcomes:
        rank1: except -> fatal evidence record -> re-raise -> process dies.
        rank0: its real collective loses the peer -> gloo error/timeout ->
               its OWN phase-1 except -> fatal record -> re-raise -> dies.
      Neither rank may: run the ordinary confirm MIN/MAX at w2, commit at
      w2, or reach w3.
  Evidence: per-rank incremental JSON files (written before each raise),
  driver-side assertions, summary JSON. PASS prints
  ALL_FATAL_BOUNDARY_TESTS_PASS and exits 0.
"""
import argparse
import json
import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = "/workspace/EPLB/OEPLB/src"
sys.path.insert(0, SRC)
from window_v2p1 import (WindowControllerV2P1, decision_hash,  # noqa: E402
                         error_sentinel, confirm_commit)
import v2p1_signals as sig                                     # noqa: E402

KNOBS = dict(kappa=1.0, z_lo=10.0, z_hi=20.0, z_urgent=1000.0,
             theta_h_low=0.10, theta_h_act=0.15, n_min=0,
             m_shrink=2, m_grow=3, dwell_shrink=2, dwell_grow=2,
             u_max=4, w_uncertain_target=16, urgent_episode_cap=3)
L, P, EP = 4, 8, 2
N_WINDOWS = 3
FATAL_RANK, FATAL_WID = 1, 2


def _reducer_factory():
    def reduce_fn(pair, op):
        t = torch.tensor(pair, dtype=torch.int64)
        dist.all_reduce(t, op=(dist.ReduceOp.MIN if op == "min"
                               else dist.ReduceOp.MAX))
        pair[0], pair[1] = int(t[0]), int(t[1])
    return reduce_fn


def worker(rank, world, out_dir):
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            timeout=timedelta(seconds=25))
    epath = os.path.join(out_dir, f"fatal_rank{rank}.json")
    records = []

    def flush():
        with open(epath, "w") as f:
            json.dump(records, f, indent=1)

    def reducer(pair, op):
        t = torch.tensor(pair, dtype=torch.int64)
        dist.all_reduce(t, op=(dist.ReduceOp.MIN if op == "min"
                               else dist.ReduceOp.MAX))
        pair[0], pair[1] = int(t[0]), int(t[1])

    ctrl = WindowControllerV2P1(w_init=16, knobs=KNOBS, beta=0.5)
    det_base = ((torch.arange(L * P).reshape(L, P) % 7) + 1).to(torch.int64)
    reduce_fn = _reducer_factory()
    try:
        for wid in range(1, N_WINDOWS + 1):
            entry = {"wid": wid, "seq": []}
            h = error_sentinel(wid)
            prop_valid = 0
            rec = None
            # -- phase 1: det SUM, unconditional; collective failure = FATAL
            try:
                det = det_base.clone()
                if rank == FATAL_RANK and wid == FATAL_WID:
                    raise RuntimeError(
                        "simulated communicator fatal error on det SUM "
                        "(NCCL async-error class)")
                dist.all_reduce(det)
                entry["seq"].append(f"{wid}:det_sum:{det.numel()}")
            except Exception as e:
                entry["fatal"] = "phase1_det_collective"
                entry["err"] = (type(e).__name__ + ": " + str(e))[:200]
                entry["committed"] = False
                records.append(entry)
                flush()
                raise  # r3: terminate; NO ordinary confirm continuation
            # -- phase 2: local recoverable scope (no collective inside)
            snap = ctrl.snapshot()
            try:
                n_fresh = int(det.sum().item())
                h_w = sig.headroom(det, EP)
                h_rate = sig.headroom_rate(h_w, n_fresh)
                rec = ctrl.step(0.0, h_rate, n_fresh)
                h = decision_hash(wid, rec)
                prop_valid = 1
            except Exception as e:
                ctrl.restore(snap)
                rec = None
                prop_valid = 0
                h = error_sentinel(wid)
                entry["phase2_err"] = str(e)[:120]
            # -- phase 3: confirm MIN/MAX, unconditional; failure = FATAL
            try:
                commit, h_min, h_max, v_min = confirm_commit(
                    h, prop_valid, reduce_fn)
                entry["seq"].append(f"{wid}:cmin:2")
                entry["seq"].append(f"{wid}:cmax:2")
            except Exception as e:
                entry["fatal"] = "phase3_confirm_collective"
                entry["err"] = (type(e).__name__ + ": " + str(e))[:200]
                entry["committed"] = False
                records.append(entry)
                flush()
                raise
            # -- phase 4: symmetric commit
            if commit and rec is not None and prop_valid == 1:
                entry["committed"] = True
                ctrl.confirm_fail_streak = 0
            else:
                ctrl.restore(snap)
                entry["committed"] = False
                ctrl.confirm_fail_streak += 1
                if ctrl.confirm_fail_streak >= 2:
                    ctrl.on_desync()
            entry["verdict"] = {"commit": bool(commit), "h_min": h_min,
                                "h_max": h_max, "v_min": v_min}
            records.append(entry)
            flush()
        entry = {"done": True}
        records.append(entry)
        flush()
    except Exception:
        records.append({"terminated_with": traceback.format_exc()[-600:]})
        flush()
        raise  # faithful to the real path: the fatal exception propagates
               # out of on_forward_pass_end -> scheduler process dies
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        "/workspace/EPLB/NEW_PAPER/experiments/"
        "h20_placement_directional_20261012/results/adw_v2p1/r3"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for r in (0, 1):
        p = os.path.join(args.out, f"fatal_rank{r}.json")
        if os.path.exists(p):
            os.remove(p)

    spawn_err, spawned_raised = None, False
    try:
        mp.spawn(worker, args=(2, args.out), nprocs=2, join=True)
    except Exception as e:                      # expected: run terminated
        spawned_raised = True
        spawn_err = (type(e).__name__ + ": " + str(e))[:300]

    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        tag = "PASS" if cond else "FAIL"
        print(f"[{tag}] {name}" + (f" :: {detail}" if detail else ""))

    r = {}
    for rank in (0, 1):
        p = os.path.join(args.out, f"fatal_rank{rank}.json")
        r[rank] = json.load(open(p)) if os.path.exists(p) else None
    check("evidence files written by BOTH ranks before termination",
          r[0] is not None and r[1] is not None)
    if r[0] is None or r[1] is None:
        _finish(results, args.out, r, spawn_err, spawned_raised)
        return

    w1_0 = next((e for e in r[0] if e.get("wid") == 1), None)
    w1_1 = next((e for e in r[1] if e.get("wid") == 1), None)
    check("w1 healthy: joint COMMIT on both ranks (protocol baseline)",
          w1_0 and w1_1 and w1_0.get("committed") and w1_1.get("committed")
          and w1_0["verdict"]["commit"] and w1_1["verdict"]["commit"])

    w2_0 = next((e for e in r[0] if e.get("wid") == 2), None)
    w2_1 = next((e for e in r[1] if e.get("wid") == 2), None)
    check("w2 injected rank: det-collective fatal recorded, re-raised, "
          "NO ordinary confirm attempted (seq has no cmin/cmax), NO commit",
          w2_1 is not None and w2_1.get("fatal") == "phase1_det_collective"
          and w2_1.get("committed") is False
          and not any("cmin" in s or "cmax" in s for s in w2_1.get("seq", []))
          and "simulated communicator fatal" in w2_1.get("err", ""),
          detail=(w2_1 or {}).get("err", "")[:100])
    check("w2 peer rank: its OWN real collective errored/timed out -> "
          "phase-1 fatal recorded, re-raised, NO confirm, NO commit",
          w2_0 is not None and w2_0.get("fatal") == "phase1_det_collective"
          and w2_0.get("committed") is False
          and not any("cmin" in s or "cmax" in s for s in w2_0.get("seq", [])),
          detail=(w2_0 or {}).get("err", "")[:100])
    check("NO rank produced ANY window-3 record (run terminated at the "
          "fatal window; no continuation of the normal schedule)",
          not any(e.get("wid") == 3 for e in r[0] + r[1]))
    check("NO rank reached the clean-exit marker after w2",
          not any(e.get("done") for e in r[0] + r[1]))
    check("both processes died with the fatal exception propagated "
          "(terminated_with recorded; mp.spawn raised)",
          spawned_raised
          and any("terminated_with" in e for e in r[0])
          and any("terminated_with" in e for e in r[1]),
          detail=(spawn_err or "spawn returned cleanly")[:140])
    _finish(results, args.out, r, spawn_err, spawned_raised)


def _finish(results, out, r, spawn_err, spawned_raised):
    ok = all(c for _, c in results)
    summary = {
        "test": "v2p1_fatal_boundary", "protocol_rev": "r3",
        "world": 2, "backend": "gloo", "pass": ok,
        "spawn_raised": spawned_raised, "spawn_err": spawn_err,
        "checks": [{"name": n, "pass": c} for n, c in results],
        "rank0_records": r.get(0), "rank1_records": r.get(1),
        "policy": ("healthy-communicator local error -> joint HOLD (11-window"
                   " test); communicator-level collective failure -> FATAL "
                   "log + re-raise -> run terminates via framework fault "
                   "propagation; no ordinary confirm/commit on a suspect "
                   "communicator"),
    }
    with open(os.path.join(out, "fatal_boundary_evidence.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print()
    if ok:
        print("ALL_FATAL_BOUNDARY_TESTS_PASS")
    else:
        print("FATAL_BOUNDARY_TESTS_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
