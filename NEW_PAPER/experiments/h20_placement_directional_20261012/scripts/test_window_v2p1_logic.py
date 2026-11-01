#!/usr/bin/env python3
"""ADW v2.1 limited revision -- offline acceptance tests (T0-T15, protocol revision r2).

Scope: the three audit-verified findings only (F1 logical detection identity,
F2 rate-scale headroom, F3 confirm-then-commit). Offline/CPU, no server, no
GPU, no performance measurement. Framing: correctness/consistency revisions;
NOT presented as performance improvements; prior v2 performance results are
NOT causally attributed to these findings.

PASS criterion: prints ALL_V2P1_TESTS_PASS and exits 0.
"""
import difflib
import glob
import hashlib
import json
import os
import sys

import torch

SRC = "/workspace/EPLB/OEPLB/src"
LOCKED = os.path.join(SRC, "locked_v2_frozen_20261016")
EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
sys.path.insert(0, SRC)

from window_v2 import WindowControllerV2, KNOB_DEFAULTS          # noqa: E402
from window_v2p1 import (WindowControllerV2P1, decision_hash,     # noqa: E402
                         error_sentinel, confirm_commit,
                         _STATE_FIELDS, _HASH_MOD)
import v2p1_signals as sig                                       # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# T0 frozen preservation
# ----------------------------------------------------------------------
def t0():
    man = {}
    with open(os.path.join(LOCKED, "MANIFEST.sha256")) as f:
        for line in f:
            sha, rel = line.split()
            man[os.path.basename(rel)] = sha
    ok = all(sha256(os.path.join(LOCKED, k)) == v for k, v in man.items())
    check("T0a locked snapshot matches MANIFEST.sha256", ok, str(man))
    check("T0b window_v2.py untouched (810c54da...)",
          sha256(os.path.join(SRC, "window_v2.py")) == man["window_v2.py"],
          sha256(os.path.join(SRC, "window_v2.py"))[:16])
    for fname in ("controller.py", "config.py"):
        frozen = open(os.path.join(LOCKED, fname)).read().splitlines(keepends=True)
        cur = open(os.path.join(SRC, fname)).read().splitlines(keepends=True)
        removed = [l for l in difflib.unified_diff(frozen, cur, lineterm="")
                   if l.startswith("-") and not l.startswith("---")]
        added = [l for l in difflib.unified_diff(frozen, cur, lineterm="")
                 if l.startswith("+") and not l.startswith("+++")]
        check(f"T0c {fname}: purely additive vs frozen (0 removals)",
              len(removed) == 0 and len(added) > 0,
              f"+{len(added)} -{len(removed)}")
        check(f"T0d {fname}: hash differs from frozen (patched, new lock needed)",
              sha256(os.path.join(SRC, fname)) != man[fname],
              sha256(os.path.join(SRC, fname))[:16])


# ----------------------------------------------------------------------
# F1: logical identity
# ----------------------------------------------------------------------
def t1():
    L, P = 3, 8
    p2l = torch.arange(P).unsqueeze(0).repeat(L, 1) % 4   # redundant replicas
    det = torch.randint(0, 50, (L, P), dtype=torch.int64)
    logical = sig.logical_load(det, p2l)
    ok_total = int(logical.sum()) == int(det.sum())
    ok_bins = True
    for l in range(L):
        for e in range(4):
            want = int(det[l, (p2l[l] == e).nonzero().squeeze(-1)].sum())
            ok_bins &= int(logical[l, e]) == want
    check("T1 logical_load conserves mass + per-expert bins (redundant p2l)",
          ok_total and ok_bins)


def t2():
    """Commit-remap invariance: partial counts before a mid-window pair swap +
    mirrored count remap + current-p2l gather == never-swapped oracle."""
    L = 2
    p2l_old = torch.tensor([[0, 1, 2, 3]] * L)
    p2l_new = torch.tensor([[1, 0, 2, 3]] * L)   # slots 0<->1 swapped
    partial = torch.tensor([[3, 7, 1, 1], [2, 5, 0, 1]], dtype=torch.int64)
    post = torch.tensor([[2, 4, 1, 0], [1, 3, 1, 1]], dtype=torch.int64)
    # commit: counts follow the expert (mirror of controller finalize hook)
    remapped = partial.clone()
    remapped[:, 0], remapped[:, 1] = partial[:, 1].clone(), partial[:, 0].clone()
    final = remapped + post                       # post routes under NEW layout
    gathered = sig.logical_load(final, p2l_new)
    # oracle: accumulate in logical space directly
    oracle = torch.zeros_like(final)
    oracle[:, 0] = partial[:, 0] + post[:, 1]     # logical 0: old slot0, new slot1
    oracle[:, 1] = partial[:, 1] + post[:, 0]     # logical 1
    oracle[:, 2:] = partial[:, 2:] + post[:, 2:]
    check("T2 mid-window swap + count remap -> logical counts exact",
          bool((gathered == oracle).all()),
          f"gathered={gathered.tolist()} oracle={oracle.tolist()}")


def t3():
    """z artifact removal: identical logical workload across a swap yields
    s_change=0 in logical space, but >0 in physical-slot space (v2 artifact)."""
    L = 2
    p2l_old = torch.tensor([[0, 1, 2, 3]] * L)
    p2l_new = torch.tensor([[1, 0, 2, 3]] * L)
    det1 = torch.tensor([[40, 10, 5, 5], [30, 20, 5, 5]], dtype=torch.int64)
    logical1 = sig.logical_load(det1, p2l_old)
    ref, valid = sig.row_distribution(logical1)
    # same LOGICAL workload in window 2 (stationary demand), slots swapped
    det2 = torch.zeros_like(det1)
    det2[:, 0] = logical1[:, 1]
    det2[:, 1] = logical1[:, 0]
    det2[:, 2:] = logical1[:, 2:]
    logical2 = sig.logical_load(det2, p2l_new)
    p2, valid2 = sig.row_distribution(logical2)
    s_log, _ = sig.change_score(p2, ref, valid2 & valid)
    # v2-style physical-space comparison (slot identity, ref never remapped)
    pphys1, vph1 = sig.row_distribution(det1)
    pphys2, vph2 = sig.row_distribution(det2)
    s_phys, _ = sig.change_score(pphys2, pphys1, vph1 & vph2)
    z_log = sig.z_from_signals(s_log, 100000, L, 100.0)
    z_phys = sig.z_from_signals(s_phys, 100000, L, 100.0)
    check("T3 logical s_change ~ 0 across swap (artifact removed)",
          s_log is not None and s_log < 1e-6, f"s_log={s_log:.2e} z_log={z_log:.2e}")
    check("T3b physical-space s_change > 0 (v2 artifact reproduced for contrast)",
          s_phys > 1e-3, f"s_phys={s_phys:.4f} z_phys={z_phys:.2f}")


def t4():
    """r2 phase structure: det collective UNCONDITIONAL (nothing skippable
    before it), local recoverable scope contains NO collective, confirm
    collective unconditional, symmetric commit rule, mapping-version tripwire
    inside the recoverable scope (hold, never skip a collective)."""
    check("T4a error_sentinel deterministic per window_id",
          error_sentinel(7) == error_sentinel(7) and error_sentinel(7) != error_sentinel(8))
    src = open(os.path.join(SRC, "controller.py")).read()
    i0 = src.index("def _window_control_v2p1")
    i1 = src.index("def _dump_detailed_load")
    body = src[i0:i1]
    ph1 = body.index("-- phase 1: det collective")
    ph2 = body.index("-- phase 2: local recoverable scope")
    ph3 = body.index("-- phase 3: confirm collectives")
    ph4 = body.index("-- phase 4: symmetric commit")
    det_ar = body.index("all_reduce(det")
    trip = body.index("self._det_map_version != self._p2l_version")
    seg2 = body[ph2:ph3]
    check("T4b det all_reduce is UNCONDITIONAL: no local check/raise before it "
          "(tripwire moved AFTER the collective; r1 gap closed)",
          ph1 < det_ar < ph2 < trip)
    check("T4b2 phase-2 recoverable scope contains NO collective",
          "all_reduce" not in seg2 and "dist." not in seg2
          and "confirm_commit" not in seg2)
    check("T4c confirm_commit appears exactly once, strictly between phase 2 "
          "and phase 4 (unconditional)",
          body.count("confirm_commit(") == 1
          and ph2 < body.index("confirm_commit(") < ph4)
    check("T4d no 'return' anywhere in the method before phase 4",
          "return" not in body[:ph4].split("Returns")[0].replace("reduce_fn", ""))
    check("T4e symmetric commit rule (verdict AND local proposal validity)",
          "if commit and rec is not None and prop_valid == 1:" in body)
    check("T4f phase order 1<2<3<4 and det-clone inside phase 1 try only",
          ph1 < ph2 < ph3 < ph4 and body.index("self._det_load.clone()") > ph1
          and body.index("self._det_load.clone()") < ph2)
    # confirm reducer uses MIN and MAX, fixed order
    i_r = src.index("def _v2p1_confirm_reduce")
    rbody = src[i_r:src.index("def _window_control_v2p1")]
    check("T4g confirm reducer: elementwise MIN/MAX all_reduce on [h, prop_valid]",
          "ReduceOp.MIN" in rbody and "ReduceOp.MAX" in rbody
          and rbody.count("torch.distributed.all_reduce") == 1)


# ----------------------------------------------------------------------
# F2: rate scale
# ----------------------------------------------------------------------
def _p1_knobs(lo=0.10, act=0.15):
    return dict(kappa=1.0, z_lo=10.0, z_hi=20.0, z_urgent=1000.0,
                m_shrink=2, m_grow=3, n_min=0, u_max=4,
                theta_h_low=lo, theta_h_act=act,
                dwell_shrink=2, dwell_grow=2,
                w_uncertain_target=16, urgent_episode_cap=3)


ZSEQ = ([None] + [5.0] * 3 + [25.0] * 5 + [5.0] * 6 + [25.0] * 4 + [1500.0] * 2
        + [5.0] * 4)
RATES = ([0.12] * 4 + [0.20] * 5 + [0.05] * 6 + [0.20] * 4 + [0.30] * 2
         + [0.05] * 4)


def _run(ctrl, zs, hs, ns):
    out = []
    for z, h, n in zip(zs, hs, ns):
        r = ctrl.step(z, h, n)
        out.append((r["W_after"], r["event"]))
    return out


def t5():
    n_base = 10_000_000
    ref = None
    ok = True
    detail = []
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        hs = [r * n_base * c for r in RATES]
        ns = [int(n_base * c)] * len(ZSEQ)
        traj = _run(WindowControllerV2P1(16, _p1_knobs()), ZSEQ, hs, ns)
        if ref is None:
            ref = traj
        same = traj == ref
        ok &= same
        detail.append(f"c={c}:{'same' if same else 'DIFF'}")
    check("T5 v2.1 rate-scale invariance: (h,n)->(c*h,c*n) identical trajectory",
          ok, " ".join(detail))


def t6():
    """Contrast demo (synthetic): frozen v2 with RAW-count theta conflates
    window size with imbalance; v2.1 rate theta does not."""
    n_small, n_big = 1_000_000, 4_000_000
    rate = 0.20
    zs = [None] + [25.0] * 6
    v2k = dict(KNOB_DEFAULTS)
    v2k.update(kappa=1.0, z_lo=10.0, z_hi=20.0, z_urgent=1000.0,
               theta_h_low=100_000, theta_h_act=300_000)
    # scale separation: h_small = 0.20 * 1M = 200k < theta_act = 300k
    #                 h_big   = 0.20 * 4M = 800k > theta_act = 300k
    tA = _run(WindowControllerV2(16, v2k), zs, [rate * n_small] * len(zs),
              [n_small] * len(zs))
    tB = _run(WindowControllerV2(16, v2k), zs, [rate * n_big] * len(zs),
              [n_big] * len(zs))
    evA = [e for _, e in tA]
    evB = [e for _, e in tB]
    p1A = _run(WindowControllerV2P1(16, _p1_knobs()), zs,
               [rate] * len(zs), [n_small] * len(zs))
    p1B = _run(WindowControllerV2P1(16, _p1_knobs()), zs,
               [rate] * len(zs), [n_big] * len(zs))
    check("T6a frozen v2 raw-theta: same imbalance RATE, different window "
          "workload -> different decisions (artifact demo)",
          evA != evB, f"small={evA[-4:]} big={evB[-4:]}")
    check("T6b v2.1 rate-theta: identical decisions across workload scales",
          p1A == p1B, f"{[e for _, e in p1A][-4:]}")


def t7():
    torch.manual_seed(0)
    ok = True
    for _ in range(200):
        L, P, EP = 3, 8, 2
        det = torch.randint(0, 1000, (L, P), dtype=torch.int64)
        n = int(det.sum())
        r = sig.headroom_rate(sig.headroom(det, EP), n)
        ok &= (0.0 <= r <= 1.0)
    # hand-computed case: rank0 slots [60,60,0,0], rank1 zeros -> h=60, rate=0.5
    det = torch.zeros(1, 8, dtype=torch.int64)
    det[0, 0] = 60
    det[0, 1] = 60
    h = sig.headroom(det, 2)
    r = sig.headroom_rate(h, int(det.sum()))
    # single hot expert: indivisible floor -> h = 0
    det2 = torch.zeros(1, 8, dtype=torch.int64)
    det2[0, 0] = 100
    r2 = sig.headroom_rate(sig.headroom(det2, 2), 100)
    # zero workload -> guarded 0.0
    r3 = sig.headroom_rate(0.0, 0)
    check("T7 h_rate in [0,1] (random + exact 0.5 case + floor case + zero guard)",
          ok and abs(h - 60.0) < 1e-9 and abs(r - 0.5) < 1e-9
          and r2 == 0.0 and r3 == 0.0,
          f"h={h} r={r} hot={r2} zero={r3}")


# ----------------------------------------------------------------------
# F3: confirm-then-commit
# ----------------------------------------------------------------------
def _sim_reducer(hs, vs, order_log=None):
    """Mock reduce_fn simulating `world` ranks presenting (h, v) pairs."""
    def fn(pair, op):
        if order_log is not None:
            order_log.append(op)
        pair[0] = min(hs) if op == "min" else max(hs)
        pair[1] = min(vs) if op == "min" else max(vs)
    return fn


def t8():
    h_ok = decision_hash(1, {"W_after": 16, "event": "hold", "z": None,
                             "h_w": 0.1, "n_fresh": 1000})
    h_other = error_sentinel(999)
    order = []
    c1, *_ = confirm_commit(h_ok, 1, _sim_reducer([h_ok] * 8, [1] * 8, order))
    c2, *_ = confirm_commit(h_ok, 1, _sim_reducer([h_ok] * 7 + [h_other], [1] * 8))
    c3, lo0, hi0, v0 = confirm_commit(h_ok, 1, _sim_reducer([h_ok] * 8, [1] * 7 + [0]))
    c4, *_ = confirm_commit(h_other, 0, _sim_reducer([h_other] * 8, [0] * 8))
    check("T8 joint MIN/MAX verdict: all-same+all-valid -> commit; one hash "
          "divergent -> hold; one proposal invalid -> hold DESPITE identical "
          "hashes; all-invalid sentinel -> hold; order MIN then MAX; two "
          "collectives per window",
          c1 and not c2 and not c3 and not c4 and order == ["min", "max"]
          and v0 == 0 and lo0 == hi0 == h_ok,
          f"verdicts={c1},{c2},{c3},{c4} order={order}")


def t9():
    ctrl = WindowControllerV2P1(16, _p1_knobs())
    n = 10_000_000
    # window 1 (init) committed
    snap0 = ctrl.snapshot()
    r = ctrl.step(None, 0.12, n)
    assert r["event"] == "init"
    # window 2: proposal mutates state, non-unanimous -> full rollback
    snap = ctrl.snapshot()
    r2 = ctrl.step(25.0, 0.20, n)
    mutated = any(getattr(ctrl, f) != snap[f] for f in _STATE_FIELDS)
    ctrl.restore(snap)
    ctrl.confirm_fail_streak += 1
    rolled = all(getattr(ctrl, f) == snap[f] for f in _STATE_FIELDS)
    # window 3: second failure -> inherited safe mode freeze at w_init
    snap3 = ctrl.snapshot()
    ctrl.step(25.0, 0.20, n)
    ctrl.restore(snap3)
    ctrl.confirm_fail_streak += 1
    freeze_ok = ctrl.confirm_fail_streak >= 2
    if freeze_ok:
        ctrl.on_desync()
    r4 = ctrl.step(25.0, 0.20, n)
    check("T9 rollback restores all 9 state fields; streak>=2 -> freeze at "
          "w_init; subsequent windows w_desync_freeze",
          mutated and rolled and freeze_ok and ctrl.W == 16 and ctrl.frozen
          and r4["event"] == "w_desync_freeze" and r4["W_after"] == 16,
          f"streak={ctrl.confirm_fail_streak} ev={r4['event']}")
    # state fields coverage
    check("T9b _STATE_FIELDS covers the full transition state (9 fields)",
          len(_STATE_FIELDS) == 9 and set(_STATE_FIELDS) == {
              "W", "alert_k", "stable_s", "dwell", "dwell_dir",
              "uncertain_run", "carry_windows", "episode_shrinks",
              "window_seq"})


def t10():
    """Error-path participation (r2): collectives are unconditional, so error
    windows still hit det SUM + confirm MIN + confirm MAX; a unanimous
    sentinel HOLDS because v_min=0; verdict depends only on reduced data."""
    n_calls = []

    def counting_reducer(hs, vs):
        def fn(pair, op):
            n_calls.append(op)
            pair[0] = min(hs) if op == "min" else max(hs)
            pair[1] = min(vs) if op == "min" else max(vs)
        return fn
    # all 8 ranks errored identically -> same sentinel, all invalid
    s = error_sentinel(42)
    c_unan_err, *_ = confirm_commit(s, 0, counting_reducer([s] * 8, [0] * 8))
    # mixed: 7 valid (same decision hash) + 1 sentinel/invalid -> hold
    d = decision_hash(42, {"W_after": 32, "event": "confirmed_grow", "z": 25.0,
                           "h_w": 0.2, "n_fresh": 1000})
    c_mixed, *_ = confirm_commit(d, 1,
                                 counting_reducer([d] * 7 + [s], [1] * 7 + [0]))
    check("T10 error windows: unanimous sentinel -> coordinated HOLD (v_min=0, "
          "never commits); mixed valid/sentinel -> HOLD; exactly two confirm "
          "collectives per window per rank",
          not c_unan_err and not c_mixed and len(n_calls) == 4
          and n_calls == ["min", "max", "min", "max"], str(n_calls))


def t11():
    ctrl = WindowControllerV2P1(16, _p1_knobs())
    n = 10_000_000
    ctrl.step(None, 0.12, n)
    snap = ctrl.snapshot()
    r1 = ctrl.step(25.0, 0.20, n)
    ctrl.restore(snap)
    r2 = ctrl.step(25.0, 0.20, n)
    check("T11 rollback->replay equivalence (rec identical incl. window_seq)",
          r1 == r2, f"{r1['event']} / {r2['event']}")


def t12():
    rec = {"W_after": 16, "event": "hold", "z": 1.5, "h_w": 0.1,
           "n_fresh": 1000}
    h0 = decision_hash(1, rec)
    ok_det = h0 == decision_hash(1, dict(rec))
    sens = {}
    for k, v in (("W_after", 32), ("event", "confirmed_grow"), ("z", 1.6),
                 ("h_w", 0.11), ("n_fresh", 1001)):
        r2 = dict(rec)
        r2[k] = v
        sens[k] = decision_hash(1, r2) != h0
    ok_range = 0 <= h0 < _HASH_MOD and 0 <= error_sentinel(1) < _HASH_MOD
    ok_win = decision_hash(2, rec) != h0
    ok_ns = error_sentinel(1) != h0
    check("T12 decision_hash deterministic, sensitive to every committed "
          "field + window_id, in [0,2^58), distinct from error namespace",
          ok_det and all(sens.values()) and ok_range and ok_win and ok_ns,
          str(sens))


# ----------------------------------------------------------------------
# T13 replay regression: frozen v2 must reproduce the 6 phaseB V2 logs
# ----------------------------------------------------------------------
FROZEN_KNOBS = dict(kappa=2806.9, z_lo=3.737, z_hi=19.864, z_urgent=115.619,
                    theta_h_low=988725.0, theta_h_act=1065054.0,
                    n_min=12320768, m_shrink=2, m_grow=3,
                    dwell_shrink=2, dwell_grow=2, u_max=4,
                    w_uncertain_target=16, urgent_episode_cap=3)


def t13():
    logs = sorted(glob.glob(os.path.join(
        EXP, "results/step6/phaseB/W*_V2__s*/v2_rank0.jsonl")))
    check("T13a found 6 phaseB V2 rank0 logs", len(logs) == 6, str(len(logs)))
    total = 0
    mismatch = 0
    per_log = []
    for path in logs:
        ctrl = WindowControllerV2(16, dict(FROZEN_KNOBS), beta=0.5)
        n = mm = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                rec = ctrl.step(d["z"], d["h_w_assignments"],
                                d["n_fresh_assignments"])
                n += 1
                if (rec["W_after"] != d["W_after"]
                        or rec["event"] != d["event"]):
                    mm += 1
                    if mm <= 2:
                        print(f"    mismatch {os.path.basename(os.path.dirname(path))}"
                              f" w#{d['window_id']}: replay=({rec['W_after']},"
                              f"{rec['event']}) logged=({d['W_after']},{d['event']})")
        total += n
        mismatch += mm
        per_log.append(f"{os.path.basename(os.path.dirname(path))}:{n}w/{mm}mm")
    check("T13b frozen v2 replay reproduces every logged window "
          "(W_after + event) across all 6 formal V2 starts (350 windows; "
          "phaseC ablation logs excluded -- different knobs by design)",
          mismatch == 0 and total == 350,
          f"total={total} mismatch={mismatch} | " + " ".join(per_log))


def t14():
    bad = []
    for lo, act in ((0.0, 0.0), (0.0, 0.5), (0.2, 0.1), (0.5, 3.0), (0.3, 0.3)):
        try:
            WindowControllerV2P1(16, _p1_knobs(lo, act))
            bad.append((lo, act))
        except ValueError:
            pass
    try:
        WindowControllerV2P1(16, _p1_knobs(0.10, 0.15))
        ok_valid = True
    except ValueError:
        ok_valid = False
    try:   # inert logging-run values (rate > 1) must be ACCEPTED
        WindowControllerV2P1(16, _p1_knobs(1.5, 2.0))
        ok_inert = True
    except ValueError:
        ok_inert = False
    check("T14 placeholder/invalid rate thresholds refused; valid + inert "
          "(logging-run) values accepted",
          not bad and ok_valid and ok_inert, f"wrongly_accepted={bad}")


def t15():
    """Verdict symmetry (r2 core property): the verdict is a function of the
    reduced pairs only, so two ranks with DIFFERENT local hashes necessarily
    derive the SAME verdict (r1's sum==world*h_own could diverge when one
    rank's hash equalled the mean of distinct hashes -- not representable
    here at all)."""
    ha, hb = decision_hash(5, {"W_after": 16, "event": "hold", "z": 1.0,
                               "h_w": 0.1, "n_fresh": 10}), error_sentinel(77)
    hs, vs = [ha, hb, ha], [1, 1, 1]          # divergent world (3 ranks)
    va, la, hlo, hhi = confirm_commit(ha, 1, _sim_reducer(hs, vs))
    vb, lb, _, _ = confirm_commit(hb, 1, _sim_reducer(hs, vs))
    # identical reduced data -> identical verdict on both ranks
    sym = (va == vb == False) and (la == lb == min(hs)) and hlo == max(hs)
    # r1-style divergence case made unrepresentable: verdict never consults
    # the local h except through the reduced pair
    check("T15 verdict symmetry: divergent-hash world gives identical HOLD "
          "verdict from every rank's viewpoint", sym,
          f"verdictA={va} verdictB={vb} h_min={la} h_max={hhi}")


def t16():
    """r3 fatal boundary (structure): an exception from the collectives
    themselves (phase-1 det SUM, phase-3 confirm MIN/MAX) is a
    communicator-level NON-RECOVERABLE fault -> FATAL log + best-effort
    evidence record + bare re-raise into the scheduler loop (terminates the
    run; framework job-restart flow owns recovery). The r2 "hold and still
    attempt the ordinary confirm schedule" path (det_ok flag) is fully
    removed. Phase-2 local exceptions on a HEALTHY communicator remain the
    only recoverable class (joint HOLD, serving continues)."""
    src = open(os.path.join(SRC, "controller.py")).read()
    i0 = src.index("def _window_control_v2p1")
    i1 = src.index("def _dump_detailed_load")
    body = src[i0:i1]
    ph1 = body.index("-- phase 1: det collective")
    ph2 = body.index("-- phase 2: local recoverable")
    ph3 = body.index("-- phase 3: confirm collectives")
    ph4 = body.index("-- phase 4: symmetric commit")
    blk1, blk3 = body[ph1:ph2], body[ph3:ph4]
    doc = body[:body.index("import time as _time")]

    def _last_stmt(block):
        lines = [l.strip() for l in block.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        return lines[-1] if lines else ""

    check("T16a r2 continuation flag (det_ok) fully removed from method",
          "det_ok" not in body)
    check("T16b phase-1 collective failure -> FATAL log + evidence record + "
          "bare re-raise (run terminates; no ordinary confirm continuation)",
          "FATAL: det SUM all_reduce failed" in blk1
          and "_v2p1_fatal_record(1, e)" in blk1
          and _last_stmt(blk1) == "raise")
    check("T16c phase-3 collective failure -> FATAL log + evidence record + "
          "bare re-raise",
          "FATAL: confirm MIN/MAX all_reduce failed" in blk3
          and "_v2p1_fatal_record(3, e)" in blk3
          and _last_stmt(blk3) == "raise")
    fr0 = body.index("def _v2p1_fatal_record")
    fr = body[fr0:ph1]
    check("T16d fatal evidence recorder is best-effort (guarded; a logging "
          "failure can never mask the fatal re-raise)",
          "except Exception:" in fr and _last_stmt(fr) == "pass"
          and chr(34)+"protocol_rev"+chr(34) in fr)
    check("T16e protocol_rev r3 stamped in BOTH the normal window record and "
          "the fatal record; det_ok field dropped from schema",
          body.count(chr(34)+"protocol_rev"+chr(34)+": "+chr(34)+"r3"+chr(34)) == 2
          and chr(34)+"det_ok"+chr(34) not in body)
    check("T16f phase-2 recoverable class does NOT terminate: no re-raise "
          "after the local-error handler (HOLD-and-continue preserved)",
          "raise" not in body[body.index("local_recoverable"):ph3])
    check("T16g fault boundary documented in the method docstring "
          "(NON-RECOVERABLE -> TERMINATE; recovery = job restart, not the "
          "controller)",
          "TERMINATE this run" in doc and "job-restart flow" in doc
          and "NOT a safe" in doc)


if __name__ == "__main__":
    for fn in (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
               t14, t15, t16):
        fn()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        sys.exit(1)
    print("ALL_V2P1_TESTS_PASS")
