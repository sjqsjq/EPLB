#!/usr/bin/env python3
"""sp_coupling_v1 -- controller-level integration logic tests (CPU/gloo).

Builds a REAL PBOEPLBController (SRC shadowed into the sglang package path)
with a stub swap executor and a fake _decide_and_begin_swap that replicates
the real collective order (plannerSUM -> window-control), then drives
deterministic forward sequences through on_forward_pass_start /
on_forward_pass_end and asserts the MAPPING invariants:

  A: TRACK->QUIET->RECOVER->TRACK schedule, collective counts per boundary
     type (H-only=3, H+W=4), PD-1 ledgers (opportunities/recorded/valid_rows
     == n_fresh), PD-7 seals + counter restarts + cover steps, PD-8
     (rec W_after never applied, ctrl.W pinned), PD-6 commit-only transitions.
  B: accepted swap breaks the QUIET streak (entry delayed by one cycle).
  C: all-idle QUIET cycle -> coverage failure -> RECOVER (idle ranks
     contribute zero rows, never diverge the schedule).
  OFF: SP_MODE=off keeps the legacy single-counter path: no sp fields, no
     pre-forward gate, rec["W_after"] APPLIED (_effective_sync_window moves).

--world 2 runs scenario A on two gloo ranks and asserts identical committed
timelines (lockstep audit). PASS: ALL_SP_CONTROLLER_TESTS_PASS, exit 0.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import torch
import torch.distributed as dist

SRC = "/workspace/EPLB/OEPLB/src"

def shadow_import():
    import sglang.srt.managers.pb_oeplb as pkg
    if SRC not in pkg.__path__:
        pkg.__path__.insert(0, SRC)
    for m in list(sys.modules):
        if m.startswith("sglang.srt.managers.pb_oeplb."):
            del sys.modules[m]
    import importlib
    import sglang.srt.managers.pb_oeplb  # re-import package __init__
    C = importlib.import_module("sglang.srt.managers.pb_oeplb.controller")
    CFG = importlib.import_module("sglang.srt.managers.pb_oeplb.config")
    return C, CFG

N_PASS = 0
def ok(name, cond, detail=""):
    global N_PASS
    if not cond:
        print("FAIL %s %s" % (name, detail))
        raise SystemExit(1)
    N_PASS += 1
    print("PASS %s" % name)


class FM:
    def __init__(self, idle): self._idle = idle
    def is_idle(self): return self._idle
    def is_extend(self): return not self._idle

class FB:
    def __init__(self, idle=False): self.forward_mode = FM(idle)

class StubExec:
    def __init__(self): self.pending = None; self.begun = 0
    def try_finish(self, force_wait=False): return None
    def begin(self, plan): self.begun += 1


L, P, EP, TOK, TOPK = 4, 16, 4, 8, 2

def topk_uniform():
    rows = []
    for i in range(TOK):
        rows.append([(2 * i) % P, (2 * i + 1) % P])
    return torch.tensor(rows, dtype=torch.int64)

def topk_spike():
    return torch.zeros(TOK, TOPK, dtype=torch.int64)  # all to experts 0,1


def build(C, CFG, sp_mode, logdir, rank, seed=20261018):
    cfg = CFG.PBOEPLBConfig(
        sp_mode=sp_mode, window_controller_version=21, decay_factor=0.0,
        sync_window=16, min_record_tokens=2, always_record=False,
        sample_interval=1, probe_mode="off",
        v21_kappa=1.0, v21_z_lo=0.5, v21_z_hi=2.0, v21_z_urgent=6.0,
        v21_m_shrink=2, v21_m_grow=3, v21_n_min=512, v21_u_max=4,
        v21_theta_r_low=0.0101, v21_theta_r_act=0.05,
        v21_dwell_shrink=2, v21_dwell_grow=2, v21_w_uncertain_target=16,
        v21_urgent_episode_cap=3, v21_beta=0.5,
        v21_log_path=os.path.join(logdir, "w_rank%d.jsonl" % rank),
        sp_seed=seed)
    cfg.validate_sp_coupling()
    c = C.PBOEPLBController.__new__(C.PBOEPLBController)
    c.cfg = cfg
    c.num_layers = L; c.num_logical_experts = P; c.num_physical_experts = P
    c.ep_size = EP; c.num_local = P // EP
    c.load = torch.zeros(L, P, dtype=torch.int64)
    c.total_tokens = 0
    c._layer_offset = 0; c._layer_counter = 0
    c._cached_p2l = torch.arange(P, dtype=torch.int64).unsqueeze(0).repeat(L, 1)
    c.window_count = 0
    c._prefill_batch_counter = 0
    c._sample_interval = cfg.sample_interval
    c._should_record_this_batch = False
    c.async_executor = StubExec()
    c._serving_t0 = time.perf_counter(); c._cum_swap_s = 0.0
    c._budget_warned = False; c._prewarmed = True
    c._forward_id = 0; c._ready = True; c._warmup_forwards = 10
    c._steps_since_last_check = 0; c._decay_factor = cfg.decay_factor
    c._effective_sync_window = cfg.sync_window
    c._calib_done = True
    c._wctrl_version = 21
    c._v2_ctrl = None; c._v2_ref = None; c._v2_carry = False
    c._v2_hash_count = 0; c._v2_log_fh = None; c._last_window_steps = None
    # P1 sp defaults (mirroring controller __init__)
    c._sp = None; c._sp_active = False
    c._sp_fwd = 0; c._sp_h_cycle = 0
    c._steps_since_h = 0; c._steps_since_w = 0
    c._sp_opps_h = 0; c._sp_recorded_h = 0
    c._sp_opps_total = 0; c._sp_recorded_total = 0; c._sp_swaps_total = 0
    c._sp_swap_started_h = 0; c._sp_plan_opp_h = False
    c._sp_recorded_flag = False; c._sp_cover_w = 0
    c._sp_seal_log = []; c._sp_last_commit = None
    c._sp_n_min_full = 0; c._sp_dumped = False; c._sp_valid_rows_h = None
    c._prof_sp_gate_ns = 0; c._v2p1_fatal_flag = False
    c._prof_record_calls = 0; c._prof_record_ns = 0; c._prof_allreduce_ns = 0
    c._prof_planbuild_ns = 0; c._prof_finalize_ns = 0
    c._prof_last_report_forward = 0
    c._true_steps_total = 0; c._prof_probe_ns = 0
    c._prev_boundary_mono_ns = None; c._v22_last_bcost_ns = 0
    c._probe_dumped = False
    c._rank = rank
    c._init_window_control_v2p1(cfg)
    return c


COUNT = {"n": 0}
def install_ar_counter():
    real = dist.all_reduce
    def counted(*a, **k):
        COUNT["n"] += 1
        return real(*a, **k)
    dist.all_reduce = counted
    return real


def drive(c, n, route="uniform", idle=False, collectives=None):
    """Drive n forwards; record per-forward all_reduce deltas."""
    tk = topk_uniform() if route == "uniform" else topk_spike()
    for _ in range(n):
        batch = FB(idle=idle)
        before = COUNT["n"]
        c.on_forward_pass_start(batch)
        if c._should_record_this_batch:
            for _l in range(L):
                c.record_next_layer(tk)
        c.on_forward_pass_end(batch)
        if collectives is not None:
            collectives.append(COUNT["n"] - before)


def install_fake_decide(c, accept_at=()):
    def fake():
        gl = c.load.clone()
        dist.all_reduce(gl, op=dist.ReduceOp.SUM)
        c.window_count += 1
        c._window_control_v2p1(gl, int(gl.sum().item()))
        if c._sp_h_cycle in accept_at or c.window_count in accept_at and not c._sp_active:
            c.async_executor.begin([("stub",)])
            if c._sp_active:
                c._sp_swap_started_h += 1
                c._sp_swaps_total += 1
    c._decide_and_begin_swap = fake


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def scenario_A(C, CFG, logdir, rank=0, world=1, quiet=True):
    c = build(C, CFG, "adaptive", logdir, rank)
    install_fake_decide(c)
    coll = []
    timeline = []
    def snap_tl():
        lc = c._sp_last_commit or {}
        timeline.append((c._sp_h_cycle, c._sp.state, c._sp.S, c._sp.H,
                         c._sp.W, bool(lc.get("transition")),
                         c.window_count, c._true_steps_total))
    # -- TRACK: 4 H16 cycles (cycle1 init, cycles 2-4 quiet -> QUIET at 4)
    drive(c, 64, collectives=coll)
    snap_tl()
    ok("A1 TRACK joint boundaries every 16 (4 collectives each)",
       coll == ([0] * 15 + [4]) * 4, str(coll))
    ok("A2 QUIET entered at h_cycle 4", c._sp.state == "QUIET"
       and c._sp_h_cycle == 4 and (c._sp.S, c._sp.H, c._sp.W) == (4, 16, 64))
    ok("A3 PD-7 seal on S change (planner zeroed, cover=16)",
       len(c._sp_seal_log) == 1
       and c._sp_seal_log[0]["reason"] == "sp_seal_S_change"
       and c._sp_seal_log[0]["cover_steps"] == 16
       and int(c.load.sum()) == 0 and c.total_tokens == 0)
    ok("A4 PD-8 _effective_sync_window pinned 16, ctrl.W pinned 16",
       c._effective_sync_window == 16 and c._v2p1_ctrl.W == 16)
    if rank == 0:
        lines = read_jsonl(c._v2p1_log_path)
        ok("A5 jsonl: 4 lines, TRACK ledgers S1 (opps=16, rec=16, rows=n_fresh)",
           len(lines) == 4
           and all(l["sp_mode"] == "adaptive" for l in lines)
           and all(l["sp_sample_opportunities"] == 16 for l in lines)
           and all(l["sp_recorded_steps"] == 16 for l in lines)
           and all(l["n_fresh_assignments"] == world * l["sp_valid_rows"]
                   for l in lines)
           and lines[0]["sp_valid_rows"] == 1024)
        ok("A6 jsonl: init invalid, quiet cycles 2-4, transition line 4",
           lines[0]["sp_prop_valid_cycle"] is False
           and all(l["sp_prop_quiet_cond"] for l in lines[1:4])
           and lines[3]["sp_transition"] is True
           and lines[3]["sp_state_before"] == "TRACK"
           and lines[3]["sp_state_after"] == "QUIET"
           and lines[3]["sp_s"] == 4 and lines[3]["sp_w"] == 64)
        ok("A7 lockstep audit fields true",
           all(l["lockstep_ok"] for l in lines))
    # -- QUIET: cycles 5-7 H-only (3 collectives), cycle 8 joint (4)
    coll2 = []
    drive(c, 64, collectives=coll2)
    snap_tl()
    ok("A8 QUIET schedule H16/W64: collective pattern 0*15+3 ... joint=4",
       coll2 == [0] * 15 + [3] + [0] * 15 + [3] + [0] * 15 + [3]
              + [0] * 15 + [4], str(coll2))
    ok("A9 QUIET window_count only at joint boundary", c.window_count == 5,
       "wc=%d" % c.window_count)
    lines = read_jsonl(c._v2p1_log_path) if rank == 0 else []
    q = lines[4:8]
    # PD-9: S4 gate slots are 4-forward blocks; an H16 cycle spans 4 blocks
    # plus a possible edge block, so per-cycle opportunities are 3..5 with an
    # exact 25% long-run rate (totals asserted below). All-prefill -> rec==opps.
    ok("A10 QUIET ledgers S4: opps in 3..5, rec==opps, rows==n_fresh==64*opps, "
       "sum==16, plan_opp only joint",
       rank != 0 or (
       all(3 <= l["sp_sample_opportunities"] <= 5 for l in q)
       and all(l["sp_recorded_steps"] == l["sp_sample_opportunities"] for l in q)
       and all(l["sp_valid_rows"] == 64 * l["sp_sample_opportunities"]
               and l["n_fresh_assignments"]
               == world * 64 * l["sp_sample_opportunities"] for l in q)
       and sum(l["sp_sample_opportunities"] for l in q) == 16
       and [l["sp_plan_opportunity"] for l in q] == [False, False, False, True]
       and all(l["sp_n_min_eff"] == 128 for l in q)))
    ok("A11 incremental sp_run_end.json written at transition",
       rank != 0 or os.path.exists(os.path.join(logdir, "sp_run_end.json")))
    # -- shift during QUIET cycle 9 (forwards 129-144) -> RECOVER
    coll3 = []
    drive(c, 16, route="spike", collectives=coll3)
    snap_tl()
    ok("A12 first alert exits QUIET -> RECOVER (S1,H8,W8)",
       c._sp.state == "RECOVER" and (c._sp.S, c._sp.H, c._sp.W) == (1, 8, 8)
       and coll3 == [0] * 15 + [3])
    ok("A13 seal #2 with interrupted-W cover steps=16",
       len(c._sp_seal_log) == 2
       and c._sp_seal_log[1]["cover_steps"] == 16
       and c._sp_seal_log[1]["S_before"] == 4)
    lines = read_jsonl(c._v2p1_log_path) if rank == 0 else []
    ok("A14 exit line: z>=z_hi reason logged, sp_transition true",
       rank != 0 or (lines[8]["sp_transition"] is True
       and lines[8]["sp_state_after"] == "RECOVER"
       and any("first alert" in r for r in lines[8]["sp_reasons"])
       and lines[8]["z"] >= 2.0))
    ok("A15 PD-8 held through alerts (ctrl W proposals never applied)",
       c._effective_sync_window == 16 and c._v2p1_ctrl.W == 16)
    # -- RECOVER: uniform again; H8 joint boundaries; z decays below z_lo,
    #    then >=2 clean cycles -> TRACK (verified trace: cycles 10..16).
    coll4 = []
    drive(c, 96, collectives=coll4)
    snap_tl()
    nz4 = [(i + 1, v) for i, v in enumerate(coll4) if v]
    ok("A16 RECOVER H8 joint boundaries (7x at +8..+56), then TRACK H16 "
       "joints at +72,+88 -- all 4 collectives",
       nz4 == [(8, 4), (16, 4), (24, 4), (32, 4), (40, 4), (48, 4), (56, 4),
               (72, 4), (88, 4)], str(nz4))
    ok("A17 RECOVER->TRACK after >=2 clean cycles (no timeout)",
       c._sp.state == "TRACK" and (c._sp.S, c._sp.H, c._sp.W) == (1, 16, 16)
       and c._sp.transitions[-1]["from"] == "RECOVER")
    # -- TRACK re-entry: 3 quiet cycles -> QUIET again (joint at +104), then
    #    an H-only boundary at +120 (3 collectives): full re-arming check.
    coll5 = []
    drive(c, 32, collectives=coll5)
    snap_tl()
    nz5 = [(i + 1, v) for i, v in enumerate(coll5) if v]
    ok("A16b second QUIET entry: joint at +104 (4), H-only at +120 (3)",
       nz5 == [(8, 4), (24, 3)] and c._sp.state == "QUIET", str(nz5))
    tr = c._sp.transitions
    ok("A18 transition ledger TRACK->QUIET->RECOVER->TRACK->QUIET",
       [(t["from"], t["to"]) for t in tr]
       == [("TRACK", "QUIET"), ("QUIET", "RECOVER"), ("RECOVER", "TRACK"),
           ("TRACK", "QUIET")])
    ok("A19 residency ledger sums to committed H cycles",
       sum(c._sp.residency.values()) == c._sp_h_cycle,
       "%s vs %d" % (c._sp.residency, c._sp_h_cycle))
    if rank == 0:
        lines = read_jsonl(c._v2p1_log_path)
        ok("A20 jsonl lines == H cycles; window_count == joint boundaries",
           len(lines) == c._sp_h_cycle
           and c.window_count
           == sum(1 for l in lines if l["sp_plan_opportunity"]))
        ok("A21 PD-13 totals == logged sums + in-flight partial cycle",
           c._sp_recorded_total == sum(l["sp_recorded_steps"] for l in lines)
           + c._sp_recorded_h
           and c._sp_opps_total
           == sum(l["sp_sample_opportunities"] for l in lines) + c._sp_opps_h
           and c._sp_recorded_total <= c._sp_opps_total <= c._sp_fwd)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, timeline)
        if rank == 0:
            ok("A22 world=%d committed timelines identical" % world,
               all(g == gathered[0] for g in gathered),
               json.dumps(gathered)[:400])
    return c


def scenario_B(C, CFG, logdir, rank=0):
    c = build(C, CFG, "adaptive", logdir, rank)
    install_fake_decide(c, accept_at={2})   # accept a swap at joint cycle 2
    drive(c, 16 * 7)
    ok("B1 accepted swap breaks streak: QUIET at cycle 6 not 4",
       c._sp.state == "QUIET"
       and [ (t["from"], t["to"], t["h_id"]) for t in c._sp.transitions ]
       == [("TRACK", "QUIET", 6)],
       json.dumps(c._sp.transitions))
    lines = read_jsonl(c._v2p1_log_path)
    ok("B2 cycle-3 line shows swaps_started=1, quiet_cond False",
       lines[2]["sp_swaps_started"] == 1
       and lines[2]["sp_prop_quiet_cond"] is False
       and lines[1]["sp_swaps_started"] == 0)


def scenario_C(C, CFG, logdir, rank=0):
    c = build(C, CFG, "adaptive", logdir, rank)
    install_fake_decide(c)
    drive(c, 64)                      # -> QUIET at cycle 4
    assert c._sp.state == "QUIET"
    drive(c, 16, idle=True)           # all-idle H cycle
    ok("C1 idle cycle: opportunities gated, zero rows, coverage exit",
       c._sp.state == "RECOVER")
    lines = read_jsonl(c._v2p1_log_path)
    l = lines[4]
    ok("C2 idle line: opps gated rec=0 rows=0 n_fresh=0 coverage reason",
       3 <= l["sp_sample_opportunities"] <= 5 and l["sp_recorded_steps"] == 0
       and l["sp_valid_rows"] == 0 and l["n_fresh_assignments"] == 0
       and l["sp_transition"] is True
       and any("coverage failure" in r for r in l["sp_reasons"]))


def scenario_OFF(C, CFG, logdir, rank=0):
    c = build(C, CFG, "off", logdir, rank)
    install_fake_decide(c)
    coll = []
    drive(c, 64, collectives=coll)
    ok("OFF1 no pre-forward gate activity", c._sp_fwd == 0 and c._sp_active is False)
    ok("OFF2 legacy single-counter boundaries (4 collectives each, every 16)",
       coll == ([0] * 15 + [4]) * 4, str(coll))
    lines = read_jsonl(c._v2p1_log_path)
    ok("OFF3 jsonl carries no sp_* fields",
       all(not any(k.startswith("sp_") for k in l) for l in lines))
    ok("OFF4 rec W_after APPLIED (window grows under stable z)",
       c._effective_sync_window > 16 or lines[-1]["W_after"] > 16,
       "esw=%d W_after=%s" % (c._effective_sync_window, lines[-1]["W_after"]))
    ok("OFF5 legacy record decision used prefill counter",
       c._prefill_batch_counter >= 0 and c._sp_opps_total == 0)
    # drive further: boundaries now follow the grown window
    n0 = len(lines)
    drive(c, c._effective_sync_window)
    lines2 = read_jsonl(c._v2p1_log_path)
    ok("OFF6 next boundary after grown window", len(lines2) == n0 + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    C, CFG = shadow_import()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    dist.init_process_group(backend=args.backend, rank=0, world_size=1) \
        if args.world == 1 else None
    logroot = tempfile.mkdtemp(prefix="sp_logic_w%d_" % args.world)
    real_ar = install_ar_counter()
    try:
        if args.world == 1:
            for name, fn in (("A", scenario_A), ("B", scenario_B),
                             ("C", scenario_C), ("OFF", scenario_OFF)):
                d = os.path.join(logroot, name)
                os.makedirs(d, exist_ok=True)
                print("--- scenario %s (%s) ---" % (name, d))
                fn(C, CFG, d)
        else:
            _mp_main(args, C, CFG)
    finally:
        dist.all_reduce = real_ar
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"world": args.world, "assertions": N_PASS}, f)
    print("ALL_SP_CONTROLLER_TESTS_PASS (world=%d, %d assertions)"
          % (args.world, N_PASS))


def _mp_main(args, C, CFG):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    port = str(29600 + os.getpid() % 500)
    procs = []
    outs = []
    for r in range(args.world):
        out = tempfile.mktemp(prefix="sp_rank%d_" % r, suffix=".json")
        outs.append(out)
        p = ctx.Process(target=_mp_rank, args=(r, args.world, args.backend,
                                               port, out))
        p.start(); procs.append(p)
    fail = False
    for p in procs:
        p.join(600)
        if p.exitcode != 0:
            fail = True
    if fail:
        print("FAIL: a rank exited nonzero")
        raise SystemExit(1)
    global N_PASS
    tot = 0
    for o in outs:
        with open(o) as f:
            tot += json.load(f)["assertions"]
    N_PASS = tot


def _mp_rank(rank, world, backend, port, out):
    global N_PASS
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    C, CFG = shadow_import()
    logdir = tempfile.mkdtemp(prefix="sp_A_rank%d_" % rank)
    install_ar_counter()
    scenario_A(C, CFG, logdir, rank=rank, world=world)
    with open(out, "w") as f:
        json.dump({"rank": rank, "assertions": N_PASS}, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
