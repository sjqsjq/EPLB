#!/usr/bin/env python3
"""rev1 offline acceptance suite (author directive 三; CPU-only, no serving).

Covers the REV1_quiet_exit_revision_20261018.md §7 scenario matrix:
  A  threshold-vicinity fluctuation      -> stays QUIET, streak 1->0 oscillates
  B  sustained margin rise (2 cycles)    -> TRACK (T7/P3)
  B' long in-band residency              -> QUIET throughout (hysteresis band)
  C  real change (z_hi/event/swap)       -> RECOVER single cycle (T2/T4/T5)
  D  insufficient samples                -> TRACK (T6/P2), NOT RECOVER
  E  simultaneous triggers               -> P1 > P2 > P3 priority
  F  rev0 equivalence regression         -> sealed 4 A-arm logs replayed
       through rev0 reproduce every committed transition/streak/hash
       (hash bit-identical to the original v1 payload construction)
  G  rev1 per-boundary oracle on the 32 sealed QUIET exits (rule-behavior
       classification ONLY -- counterfactual next-cycle signals are
       unobservable; NO throughput/routing prediction is made or implied)
  G2 rev1 free-run smoke over full logs (crash-free, divergence point noted)
  H  snapshot/restore symmetry incl. exit_streak

PASS: ALL_SP_REV1_TESTS_PASS, exit 0.
"""
import hashlib
import json
import os
import sys

SRC = "/workspace/EPLB/OEPLB/src"
sys.path.insert(0, SRC)
from sp_state import (SpController, SP_TABLE, SP_CHANGE_EVENTS,
                      SP_GUARDRAIL_EVENTS, SP_QUIET_EXIT_CONFIRM)

DEV = ("/workspace/EPLB/NEW_PAPER/experiments/"
       "h20_placement_directional_20261012/results/sp_coupling_v1/dev")
OUT = ("/workspace/EPLB/NEW_PAPER/experiments/"
       "h20_placement_directional_20261012/results/sp_coupling_v1/"
       "REV1_replay_classification_20261018.md")

# frozen calibration knobs (sealed; identical for rev0/rev1)
Z_LO = 22.643858727592757
Z_HI = 314.14010272527486
TH_LO = 0.010095552687949321   # v21_theta_r_low  = theta_in
TH_ACT = 0.011395643112507271  # v21_theta_r_act  = theta_out (rev1)
N_MIN = 12320768
N_PASS = [0]


def ok(cond, msg):
    assert cond, "FAIL: " + msg
    N_PASS[0] += 1


def ctl(rev):
    return SpController(mode="adaptive", seed=20261018, quiet_streak=3,
                        recover_min_cycles=2, z_lo=Z_LO, z_hi=Z_HI,
                        theta_h_low=TH_LO, n_min=N_MIN,
                        theta_h_act=TH_ACT, rev=rev)


def sig(z=1.0, h_rate=0.005, n_fresh=50_000_000, event="hold",
        swap_pending=False, swaps_started=0, plan_opp=True):
    return {"z": z, "h_rate": h_rate, "n_fresh": n_fresh, "event": event,
            "swap_pending": swap_pending, "swaps_started": swaps_started,
            "plan_opp": plan_opp}


def enter_quiet(c, rev):
    """Drive TRACK -> QUIET via 3 quiet cycles; apply commits."""
    for _ in range(3):
        p = c.propose(sig())
        c.apply(p, h_id=None)
    assert c.state == "QUIET", c.state
    ok(True, "entered QUIET (rev%d)" % rev)


def cycle(c, s):
    p = c.propose(s)
    r = c.apply(p, h_id=None)
    return p, r


# ---- invariant: frozen table untouched ----------------------------------
ok(SP_TABLE == {"TRACK": (1, 16, 16), "QUIET": (4, 16, 64),
                "RECOVER": (1, 8, 8)}, "SP_TABLE gears unchanged (R4)")
ok(SP_QUIET_EXIT_CONFIRM == 2, "m_exit frozen at 2")
ok(TH_ACT > TH_LO > 0, "calibration pair ordered")

# ---- scenario A: threshold-vicinity fluctuation --------------------------
c = ctl(1); enter_quiet(c, 1)
seq = [0.0105, 0.0118, 0.0103, 0.0120, 0.0101]
exp_streak = [0, 1, 0, 1, 0]
for hr, es in zip(seq, exp_streak):
    p, r = cycle(c, sig(h_rate=hr))
    ok(not p["transition"], "A: h_rate=%.4f must NOT transition" % hr)
    ok(c.state == "QUIET", "A: stays QUIET at h_rate=%.4f" % hr)
    ok(c.exit_streak == es, "A: exit_streak=%d expected %d at %.4f"
       % (c.exit_streak, es, hr))
# v1 comparison: the same first value alone exits v1 QUIET->RECOVER
c0 = ctl(0); enter_quiet(c0, 0)
p0, _ = cycle(c0, sig(h_rate=0.0105))
ok(p0["transition"] and p0["state_after"] == "RECOVER",
   "A: v1 exits on 0.0105 (documents the flapping rev1 removes)")

# ---- scenario B: sustained margin rise -> TRACK --------------------------
c = ctl(1); enter_quiet(c, 1)
p, _ = cycle(c, sig(h_rate=0.0115))
ok(not p["transition"] and c.exit_streak == 1, "B: first cycle streak=1")
p, _ = cycle(c, sig(h_rate=0.0115))
ok(p["transition"] and p["state_after"] == "TRACK",
   "B: second sustained cycle -> TRACK (T7/P3)")
ok(any("P3" in x for x in p["reasons"]), "B: reason tagged P3")
ok(c.S == 1 and c.H == 16 and c.W == 16, "B: TRACK gears (S1/H16/W16)")
ok(c.exit_streak == 0, "B: streak reset on transition (PD-5 ext)")

# ---- scenario B': long in-band residency ---------------------------------
c = ctl(1); enter_quiet(c, 1)
for i in range(10):
    p, _ = cycle(c, sig(h_rate=0.0105))
    ok(not p["transition"] and c.state == "QUIET",
       "B': in-band cycle %d stays QUIET" % i)

# ---- scenario C: real change -> RECOVER (single cycle) -------------------
for tag, s in [("C1 z>=z_hi", sig(z=400.0)),
               ("C2 confirmed_grow", sig(event="confirmed_grow")),
               ("C3 urgent_shrink", sig(event="urgent_shrink")),
               ("C4 swaps_started", sig(swaps_started=1)),
               ("C5 swap_pending", sig(swap_pending=True)),
               ("C6 hold_alert", sig(event="hold_alert")),
               ("C7 confirmed_shrink", sig(event="confirmed_shrink"))]:
    c = ctl(1); enter_quiet(c, 1)
    p, _ = cycle(c, s)
    ok(p["transition"] and p["state_after"] == "RECOVER",
       "%s -> RECOVER single cycle" % tag)
    ok(c.S == 1 and c.H == 8, "%s: RECOVER gears S1/H8" % tag)

# ---- scenario D: insufficient samples -> TRACK (NOT RECOVER) -------------
c = ctl(1); enter_quiet(c, 1)
p, _ = cycle(c, sig(n_fresh=100))
ok(p["transition"] and p["state_after"] == "TRACK",
   "D: coverage failure -> TRACK (T6/P2, R1)")
ok(any("P2" in x for x in p["reasons"]), "D: reason tagged P2")
c0 = ctl(0); enter_quiet(c0, 0)
p0, _ = cycle(c0, sig(n_fresh=100))
ok(p0["state_after"] == "RECOVER", "D: v1 went to RECOVER (contrast)")

# ---- scenario E: simultaneous triggers, priority P1 > P2 > P3 ------------
c = ctl(1); enter_quiet(c, 1)
p, _ = cycle(c, sig(z=400.0, n_fresh=100))
ok(p["state_after"] == "RECOVER" and p["transition"],
   "E1: z_hi + coverage fail -> RECOVER (P1a wins)")
ok(p["reasons"][0].startswith("P1a"), "E1: P1a first in reasons")

c = ctl(1); enter_quiet(c, 1)
cycle(c, sig(h_rate=0.0115))                     # streak=1
p, _ = cycle(c, sig(h_rate=0.0115, event="w_desync_freeze"))
ok(p["state_after"] == "RECOVER", "E2: guardrail + streak full -> RECOVER")
ok(p["reasons"][0].startswith("P1b"), "E2: P1b first in reasons")

c = ctl(1); enter_quiet(c, 1)
cycle(c, sig(h_rate=0.0115))                     # streak=1
p, _ = cycle(c, sig(h_rate=0.0115, n_fresh=100))
ok(p["state_after"] == "TRACK" and p["reasons"][0].startswith("P2"),
   "E3: coverage fail + streak full -> TRACK, P2 recorded before P3")

c = ctl(1); enter_quiet(c, 1)
p, _ = cycle(c, sig(z=400.0, event="confirmed_grow", swaps_started=1,
                    n_fresh=100, h_rate=0.05))
ok(p["state_after"] == "RECOVER" and p["reasons"][0].startswith("P1a"),
   "E4: all conditions at once -> RECOVER via P1a")

# ---- scenario H: snapshot/restore symmetry -------------------------------
c = ctl(1); enter_quiet(c, 1)
cycle(c, sig(h_rate=0.0115))                     # exit_streak=1
snap = c.snapshot()
ok(snap["exit_streak"] == 1, "H: snapshot carries exit_streak")
cycle(c, sig(h_rate=0.0115))                     # would transition
c.restore(snap)
ok(c.exit_streak == 1 and c.state == "QUIET", "H: restore rolls back")

# ---- rev1 init validation -------------------------------------------------
try:
    SpController(mode="adaptive", z_lo=Z_LO, z_hi=Z_HI, theta_h_low=TH_LO,
                 n_min=N_MIN, theta_h_act=0.0, rev=1)
    ok(False, "rev1 with theta_act=0 must raise")
except ValueError:
    ok(True, "rev1 rejects unordered/zero calibration pair")
try:
    SpController(mode="adaptive", z_lo=Z_LO, z_hi=Z_HI, theta_h_low=TH_LO,
                 n_min=N_MIN, theta_h_act=TH_ACT, rev=7)
    ok(False, "rev=7 must raise")
except ValueError:
    ok(True, "rev must be 0|1")


# ================= log replay (F / G / G2) =================================
def rows_of(run):
    out = []
    with open(os.path.join(DEV, run, "v2p1_rank0.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "sp_h_id" in r:
                out.append(r)
    out.sort(key=lambda r: r["sp_h_id"])
    return out


def row_sig(r):
    return {"z": r["z"], "h_rate": r["h_rate"],
            "n_fresh": r["n_fresh_assignments"], "event": r["event"],
            "swap_pending": r["sp_swap_pending"],
            "swaps_started": r["sp_swaps_started"],
            "plan_opp": r["sp_plan_opportunity"]}


def orig_hash(c, prop):
    """Bit-exact original v1 payload construction (frozen reference)."""
    payload = ["sp1", c.mode, c.state, c.streak, c.recover_cycles,
               bool(prop.get("transition")), prop.get("state_after", c.state),
               int(prop.get("streak_after", 0)),
               int(prop.get("recover_cycles_after", 0))]
    d = hashlib.sha256(json.dumps(payload).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") % (1 << 58)


RUNS = ["sp_dev_W1_b1_A", "sp_dev_W1_b2_A", "sp_dev_W2_b1_A", "sp_dev_W2_b2_A"]

# ---- F: rev0 equivalence regression --------------------------------------
for run in RUNS:
    rows = rows_of(run)
    c = ctl(0)
    n_tr = 0
    for r in rows:
        s = row_sig(r)
        p = c.propose(s)
        ok(c.proposal_hash(p) == orig_hash(c, p),
           "%s h%s: rev0 hash bit-identical to v1" % (run, r["sp_h_id"]))
        ok(bool(p["transition"]) == bool(r["sp_transition"])
           and p["state_after"] == r["sp_state_after"]
           and p["quiet_cond"] == r["sp_prop_quiet_cond"]
           and p["valid_cycle"] == r["sp_prop_valid_cycle"]
           and p["clean_cycle"] == r["sp_prop_clean_cycle"],
           "%s h%s: rev0 proposal matches committed row" % (run, r["sp_h_id"]))
        if r.get("committed", True):
            c.apply(p, h_id=r["sp_h_id"])
            n_tr += int(bool(r["sp_transition"]))
            ok(c.state == r["sp_state_after"]
               and c.streak == r["sp_streak"]
               and c.recover_cycles == r["sp_recover_cycles"]
               and c.n_min_eff() == r["sp_n_min_eff"],
               "%s h%s: rev0 committed state matches row" % (run, r["sp_h_id"]))
    print("F: %s replayed %d rows, %d transitions, 100%% equivalence"
          % (run, len(rows), n_tr))

# ---- G: rev1 per-boundary oracle on the 32 sealed exits ------------------
cls_rows = []
counts = {"P1_RECOVER": 0, "P2_TRACK": 0, "P3_TRACK": 0, "STAY_streak1": 0,
          "STAY_band": 0}
for run in RUNS:
    rows = rows_of(run)
    for i, r in enumerate(rows):
        if not (r["sp_state_before"] == "QUIET" and r["sp_transition"]
                and r["sp_state_after"] == "RECOVER"):
            continue
        prev = rows[i - 1] if i > 0 else None
        streak_in = int(prev is not None
                        and prev["sp_state_before"] == "QUIET"
                        and float(prev["h_rate"] or 0.0) >= TH_ACT)
        c = ctl(1)
        c.state = "QUIET"; c.exit_streak = streak_in; c.streak = 0
        p = c.propose(row_sig(r))
        if p["transition"] and p["state_after"] == "RECOVER":
            cls = "P1_RECOVER"
        elif p["transition"] and p["state_after"] == "TRACK":
            cls = "P2_TRACK" if p["reasons"][0].startswith("P2") else "P3_TRACK"
        elif p["exit_streak_after"] == 1:
            cls = "STAY_streak1"
        else:
            cls = "STAY_band"
        counts[cls] += 1
        cls_rows.append((run, r["sp_h_id"], r["h_rate"], r["z"], r["event"],
                         r["sp_swaps_started"], bool(r["sp_swap_pending"]),
                         streak_in, cls,
                         p["reasons"][0][:60] if p["reasons"] else ""))
tot = sum(counts.values())
ok(tot == 32, "G: all 32 sealed exits classified (got %d)" % tot)

# ---- G2: rev1 free-run smoke ---------------------------------------------
for run in RUNS:
    rows = rows_of(run)
    c = ctl(1)
    div = None
    for r in rows:
        p = c.propose(row_sig(r))
        if c.state != r["sp_state_before"] and div is None:
            div = r["sp_h_id"]
        if r.get("committed", True):
            c.apply(p, h_id=r["sp_h_id"])
        ok(0 <= c.exit_streak <= SP_QUIET_EXIT_CONFIRM,
           "%s h%s: exit_streak bounded" % (run, r["sp_h_id"]))
    print("G2: %s free-run clean; first divergence from sealed v1 "
          "trajectory at h_id=%s (expected; counterfactual beyond it)"
          % (run, div))

with open(OUT, "w") as f:
    f.write("# REV1 重放分类（规则行为检查；不预测吞吐/路由）\n\n"
            "免责（REV1 §8）：v1 退出后即离开 S4，下一 QUIET 周期信号为不可观测"
            "反事实。本表仅回答：给定该边界已记录的信号与 rev1 语义下该边界"
            "之前的 exit_streak，rev1 在该边界会作出什么决定。\n\n"
            "| run | h_id | h_rate | h_rate/θ_out | z | event | swaps | "
            "pending | streak_in | rev1 决定 | reason 首条 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    for (run, h, hr, z, ev, sw, pd_, si, cls, rsn) in cls_rows:
        f.write("| %s | %s | %.6f | %.3f | %.3f | %s | %d | %s | %d | "
                "**%s** | %s |\n" % (run.replace("sp_dev_", ""), h, hr,
                                     hr / TH_ACT, z, ev, sw, pd_, si, cls,
                                     rsn.replace("|", "/")))
    f.write("\n## 汇总\n\n| rev1 决定 | 次数 |\n|---|---|\n")
    for k in sorted(counts):
        f.write("| %s | %d |\n" % (k, counts[k]))
    f.write("| 合计 | %d |\n" % tot)
print("G: classification written to", OUT)
print("counts:", counts)

print("ALL_SP_REV1_TESTS_PASS (%d assertions)" % N_PASS[0])
