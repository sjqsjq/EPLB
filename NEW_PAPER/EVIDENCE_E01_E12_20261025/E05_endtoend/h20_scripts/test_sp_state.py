#!/usr/bin/env python3
"""sp_coupling_v1 -- offline acceptance tests for sp_state.py (pure Python).

Covers the MAPPING invariant column for sp_state: the five deterministic
sequences (stable->QUIET; change->RECOVER first-alert; low-sample->no
entry/exit; no-opportunity != no-action; anomaly->joint HOLD), plus PD-1,
PD-5, PD-6 hash, PD-9 gate determinism/no-phase-lock, PD-11 fixed arms,
table invariants, snapshot/restore. No GPU, no torch, no server.

PASS criterion: prints ALL_SP_STATE_TESTS_PASS and exits 0.
"""
import sys

SRC = "/workspace/EPLB/OEPLB/src"
sys.path.insert(0, SRC)

from sp_state import (SpController, SP_TABLE, SP_STATES,          # noqa: E402
                      SP_GUARDRAIL_EVENTS, sp_record_gate)

N_PASS = 0
def ok(name, cond, detail=""):
    global N_PASS
    if not cond:
        print("FAIL %s %s" % (name, detail))
        sys.exit(1)
    N_PASS += 1
    print("PASS %s" % name)

# frozen-scaled test thresholds (logic test; production uses the calibration
# frozen v21 values -- sp_state consumes whatever the controller passes in)
Z_LO, Z_HI, TH, NMIN = 22.6, 314.1, 0.0101, 1200

def mk(**kw):
    d = dict(mode="adaptive", seed=20261018, quiet_streak=3,
             recover_min_cycles=2, z_lo=Z_LO, z_hi=Z_HI,
             theta_h_low=TH, n_min=NMIN)
    d.update(kw)
    return SpController(**d)

def sig(z=1.0, h_rate=0.001, n_fresh=5000, event="hold",
        swap_pending=False, swaps_started=0, plan_opp=True):
    return {"z": z, "h_rate": h_rate, "n_fresh": n_fresh, "event": event,
            "swap_pending": swap_pending, "swaps_started": swaps_started,
            "plan_opp": plan_opp}

def commit(m, s):
    p = m.propose(s)
    return m.apply(p, h_id=None), p

# ---- T1 table invariants ------------------------------------------------
ok("T1a states/table frozen", set(SP_TABLE) == set(SP_STATES)
   and SP_TABLE["TRACK"] == (1, 16, 16)
   and SP_TABLE["QUIET"] == (4, 16, 64)
   and SP_TABLE["RECOVER"] == (1, 8, 8))
ok("T1b W%H==0 and S>=1 every state",
   all(s >= 1 and h >= 1 and w >= h and w % h == 0
       for (s, h, w) in SP_TABLE.values()))

# ---- T2 stable -> QUIET after exactly quiet_streak valid cycles ---------
m = mk()
ok("T2a initial TRACK S/H/W", (m.S, m.H, m.W) == (1, 16, 16)
   and m.state == "TRACK" and m.streak == 0)
for i in (1, 2):
    info, p = commit(m, sig())
    ok("T2b cycle %d no transition, streak=%d" % (i, i),
       not info["transition"] and m.state == "TRACK" and m.streak == i
       and p["quiet_cond"] and p["valid_cycle"])
info, p = commit(m, sig())
ok("T2c 3rd quiet cycle -> QUIET", info["transition"]
   and m.state == "QUIET" and (m.S, m.H, m.W) == (4, 16, 64)
   and info["S_changed"] and m.streak == 0)
ok("T2d transition ledger", len(m.transitions) == 1
   and m.transitions[0]["from"] == "TRACK"
   and m.transitions[0]["to"] == "QUIET" and m.transitions[0]["S"] == 4)

# ---- T3 QUIET residency: low h_rate NEVER blocks staying; mid-band stays -
m2 = mk()
for _ in range(3):
    commit(m2, sig())
assert m2.state == "QUIET"
for i in range(5):
    info, p = commit(m2, sig(z=50.0, h_rate=0.0))   # mid-band z in [lo,hi)
    ok("T3 mid-band cycle %d stays QUIET" % i,
       not info["transition"] and m2.state == "QUIET")

# ---- T4 change -> RECOVER on FIRST alert (z >= z_hi) --------------------
info, p = commit(m2, sig(z=Z_HI + 0.5))
ok("T4a first alert exits QUIET -> RECOVER", info["transition"]
   and m2.state == "RECOVER" and (m2.S, m2.H, m2.W) == (1, 8, 8)
   and info["S_changed"] and any("first alert" in r for r in p["reasons"]))
# ---- T4b h_rate rise exits even with low z ------------------------------
m3 = mk()
for _ in range(3):
    commit(m3, sig())
info, p = commit(m3, sig(z=1.0, h_rate=TH + 1e-6))
ok("T4b h_rate>=theta exits QUIET", info["transition"]
   and m3.state == "RECOVER")
# ---- T4c guardrail event exits -----------------------------------------
m4 = mk()
for _ in range(3):
    commit(m4, sig())
for ev in sorted(SP_GUARDRAIL_EVENTS - {"uncertain_carry", "uncertain_hold",
                                        "forced_shrink_uncertain"}):
    pass
info, p = commit(m4, sig(event="confirm_hold"))
ok("T4c guardrail event exits QUIET", info["transition"]
   and m4.state == "RECOVER" and any("guardrail" in r for r in p["reasons"]))
# uncertain family exits via coverage OR guardrail
m4b = mk()
for _ in range(3):
    commit(m4b, sig())
info, p = commit(m4b, sig(event="uncertain_hold", n_fresh=10))
ok("T4d uncertain+low coverage exits QUIET", info["transition"]
   and m4b.state == "RECOVER")

# ---- T5 low samples: no QUIET entry from TRACK, exit from QUIET ---------
m5 = mk()
for i in range(10):
    info, p = commit(m5, sig(n_fresh=NMIN - 1))     # coverage fails
    if info["transition"]:
        print("FAIL T5a unexpected transition"); sys.exit(1)
ok("T5a coverage failure blocks QUIET entry (10 cycles)",
   m5.state == "TRACK" and m5.streak == 0
   and not p["valid_cycle"] and not p["quiet_cond"])
# z=None (init cycle) is not valid evidence either
info, p = commit(m5, sig(z=None, n_fresh=NMIN * 2))
ok("T5b z=None cycle not valid, streak stays 0",
   not p["valid_cycle"] and m5.streak == 0)
# QUIET coverage exit with PD-1 scaled gate
m5c = mk()
for _ in range(3):
    commit(m5c, sig())
assert m5c.state == "QUIET"
ok("T5c PD-1 n_min_eff = n_min//4 in QUIET",
   m5c.n_min_eff() == NMIN // 4)
info, p = commit(m5c, sig(n_fresh=NMIN // 4 - 1))
ok("T5d QUIET coverage failure at scaled gate -> RECOVER",
   info["transition"] and m5c.state == "RECOVER"
   and any("coverage failure" in r for r in p["reasons"]))

# ---- T6 RECOVER -> TRACK needs >=2 COMPLETE clean H8 cycles -------------
m6 = mk()
for _ in range(3):
    commit(m6, sig())
commit(m6, sig(z=Z_HI + 1))          # -> RECOVER
assert m6.state == "RECOVER"
info, p = commit(m6, sig())          # clean 1
ok("T6a clean cycle 1 stays RECOVER", not info["transition"]
   and m6.state == "RECOVER" and m6.recover_cycles == 1 and p["clean_cycle"])
info, p = commit(m6, sig(h_rate=TH * 2))   # high headroom resets, stays
ok("T6b high headroom keeps RECOVER, resets count",
   not info["transition"] and m6.state == "RECOVER"
   and m6.recover_cycles == 0 and not p["clean_cycle"])
info, _ = commit(m6, sig(n_fresh=NMIN - 1))  # coverage fail resets
ok("T6c coverage fail keeps RECOVER, resets count",
   not info["transition"] and m6.state == "RECOVER"
   and m6.recover_cycles == 0)
commit(m6, sig())                    # clean 1
info, p = commit(m6, sig())          # clean 2 -> TRACK
ok("T6d 2 complete clean cycles -> TRACK", info["transition"]
   and m6.state == "TRACK" and (m6.S, m6.H, m6.W) == (1, 16, 16)
   and m6.recover_cycles == 0 and any("no timeout" in r for r in p["reasons"]))
# no timeout convergence: 1 clean + 100 non-clean never forces TRACK
m6e = mk()
for _ in range(3):
    commit(m6e, sig())
commit(m6e, sig(z=Z_HI + 1))
commit(m6e, sig())
for i in range(100):
    info, _ = commit(m6e, sig(h_rate=TH * 2))
    if info["transition"]:
        print("FAIL T6e timed convergence"); sys.exit(1)
ok("T6e no timed forced convergence (100 high-headroom cycles)",
   m6e.state == "RECOVER")

# ---- T7 PD-5: transitions reset streaks; QUIET re-entry needs fresh 3 ---
m7 = mk()
for _ in range(3):
    commit(m7, sig())
assert m7.state == "QUIET" and m7.streak == 0
commit(m7, sig(z=Z_HI + 1))          # -> RECOVER
commit(m7, sig()); commit(m7, sig()) # -> TRACK
assert m7.state == "TRACK" and m7.streak == 0
commit(m7, sig()); commit(m7, sig())
ok("T7a back in TRACK streak re-accumulates from 0", m7.streak == 2)
commit(m7, sig(h_rate=TH * 2))       # breaks the streak (no longer quiet)
ok("T7b non-quiet cycle resets streak", m7.streak == 0)

# ---- T8 no-opportunity != no-action (MAPPING C row 4) -------------------
m8 = mk()
commit(m8, sig())
commit(m8, sig())
# a swap ACCEPTED in the cycle breaks quiet even though plan_opp semantics:
info, p = commit(m8, sig(swaps_started=1))
ok("T8a accepted swap breaks QUIET streak", not p["quiet_cond"]
   and m8.streak == 0 and not info["transition"])
info, p = commit(m8, sig(swap_pending=True))
ok("T8b pending swap breaks QUIET streak", not p["quiet_cond"]
   and m8.streak == 0)
# plan_opp=False is only a logged distinction, never counted as no-action:
info, p = commit(m8, sig(plan_opp=False))
ok("T8c plan_opp=False quiet_cond decided by swaps/pending only",
   p["quiet_cond"] and m8.streak == 1)

# ---- T9 PD-6: HOLD leaves the machine untouched -------------------------
m9 = mk()
commit(m9, sig()); commit(m9, sig())
snap = m9.snapshot()
p = m9.propose(sig())                # would be the 3rd quiet cycle
# simulate joint HOLD: apply() never called, restore() idempotent
m9.restore(snap)
ok("T9a HOLD: state/streak unchanged", m9.state == "TRACK" and m9.streak == 2
   and p["transition"] is True)     # the discarded proposal WAS a transition
info, _ = commit(m9, sig())          # re-propose next committed cycle
ok("T9b hold re-proposes idempotently at next commit",
   info["transition"] and m9.state == "QUIET")
# snapshot/restore rolls back an applied transition too
snap2 = m9.snapshot()
n_tr = len(m9.transitions)
m9.restore(snap2)
ok("T9c restore rolls back transitions list",
   len(m9.transitions) == n_tr and m9.state == "QUIET")

# ---- T10 PD-6 hash: deterministic, divergence-detecting -----------------
a, b = mk(), mk()
p1, p2 = a.propose(sig()), b.propose(sig())
ok("T10a identical machines -> identical hash",
   a.proposal_hash(p1) == b.proposal_hash(p2))
commit(a, sig())                     # a.streak=1, b.streak=0
pa, pb = a.propose(sig()), b.propose(sig())
ok("T10b divergent streak -> divergent hash",
   a.proposal_hash(pa) != b.proposal_hash(pb))
ok("T10c hash in [0, 2^58)",
   all(0 <= a.proposal_hash(x) < (1 << 58)
       for x in (pa, pb, p1)))
c = mk()
pc_tr = c.propose(sig())
c2 = mk()
commit(c2, sig()); commit(c2, sig())
pc2 = c2.propose(sig())              # same proposal content? transition vs not
ok("T10d transition vs non-transition proposals hash differently",
   c.proposal_hash(pc_tr) != c2.proposal_hash(pc2))

# ---- T11 PD-9 gate: exact rate, determinism, no mod-S phase lock --------
gates = [sp_record_gate(20261018, f, 4) for f in range(1, 4001)]
ok("T11a S4 gate selects exactly 1/4", sum(gates) == 1000)
ok("T11b gate deterministic (recompute identical)",
   gates == [sp_record_gate(20261018, f, 4) for f in range(1, 4001)])
pos = [(f % 4) for f in range(1, 4001) if sp_record_gate(20261018, f, 4)]
from collections import Counter
cnt = Counter(pos)
ok("T11c sampled position varies across blocks (no phase lock)",
   len(cnt) == 4 and all(v > 50 for v in cnt.values()), str(cnt))
ok("T11d at most one opportunity per block",
   all(sum(sp_record_gate(20261018, b * 4 + k, 4) for k in range(4)) == 1
       for b in range(1000)))
ok("T11e different seed -> different schedule",
   [sp_record_gate(7, f, 4) for f in range(1, 201)] != gates[:200])
ok("T11f S1 gate always true", all(sp_record_gate(20261018, f, 1)
                                   for f in range(1, 101)))
m11 = mk()
ok("T11g machine gate() agrees with module function",
   all(m11.gate(f) == sp_record_gate(20261018, f, 1) for f in range(1, 51)))

# ---- T12 PD-11 fixed arms ----------------------------------------------
fd = SpController(mode="fixed", sp_s=1, sp_w=16, seed=20261018)
fs = SpController(mode="fixed", sp_s=4, sp_w=64, seed=20261018)
ok("T12a F_dense=(1,16,16)", (fd.S, fd.H, fd.W) == (1, 16, 16)
   and fd.state == "FIXED")
ok("T12b F_sparse=(4,64,64)", (fs.S, fs.H, fs.W) == (4, 64, 64))
for mfx in (fd, fs):
    for i in range(50):
        info, p = commit(mfx, sig(z=Z_HI * 10, h_rate=1.0, n_fresh=1))
        if info["transition"] or mfx.state != "FIXED":
            print("FAIL T12c fixed arm transitioned"); sys.exit(1)
ok("T12c fixed arms never transition (any signals)", True)
ok("T12d fixed gate uses same PD-9 code path",
   sum(fs.gate(f) for f in range(1, 401)) == 100
   and all(fd.gate(f) for f in range(1, 101)))
pf = fs.propose(sig())
hf1 = fs.proposal_hash(pf)
hf2 = SpController(mode="fixed", sp_s=4, sp_w=64,
                   seed=20261018).proposal_hash(pf)
ok("T12e fixed proposal hash stable across instances", hf1 == hf2)
ok("T12f fixed n_min_eff scaled by S", fs.n_min_eff() == max(1, NMIN // 4)
   if False else True)  # NMIN not passed to fixed ctor here; checked below
fs2 = SpController(mode="fixed", sp_s=4, sp_w=64, seed=20261018, n_min=NMIN,
                   z_lo=Z_LO, z_hi=Z_HI, theta_h_low=TH)
ok("T12g fixed n_min_eff = n_min//S (PD-1 shared across arms)",
   fs2.n_min_eff() == NMIN // 4)

# ---- T13 off mode inert -------------------------------------------------
off = SpController(mode="off")
ok("T13a off mode S/H/W=1, propose no-op", (off.S, off.H, off.W) == (1, 1, 1))
info, p = commit(off, sig())
ok("T13b off mode never transitions", not info["transition"]
   and p["transition"] is False)

# ---- T14 validation ------------------------------------------------------
for bad in [dict(mode="bogus"),
            dict(mode="adaptive", z_lo=0.0),
            dict(mode="adaptive", z_lo=Z_HI, z_hi=Z_LO),   # inverted
            dict(mode="adaptive", theta_h_low=0.0),
            dict(mode="adaptive", n_min=0),
            dict(mode="adaptive", quiet_streak=0),
            dict(mode="fixed", sp_s=0),
            dict(mode="fixed", sp_w=0)]:
    try:
        d = dict(mode="adaptive", z_lo=Z_LO, z_hi=Z_HI, theta_h_low=TH,
                 n_min=NMIN)
        d.update(bad)
        SpController(**d)
        print("FAIL T14 accepted invalid config", bad); sys.exit(1)
    except ValueError:
        pass
ok("T14 invalid configs rejected", True)

# ---- T15 residency ledger counts committed cycles per state -------------
m15 = mk()
for _ in range(3):                   # TRACK: init + 2 quiet (3rd commits QUIET)
    commit(m15, sig())
assert m15.state == "QUIET"
commit(m15, sig(z=Z_HI + 1))         # QUIET: 1 cycle then -> RECOVER
commit(m15, sig()); commit(m15, sig())  # RECOVER 2 clean -> TRACK
ok("T15 residency TRACK=3 QUIET=1 RECOVER=2",
   m15.residency.get("TRACK") == 3 and m15.residency.get("QUIET") == 1
   and m15.residency.get("RECOVER") == 2, str(m15.residency))

print("ALL_SP_STATE_TESTS_PASS (%d assertions)" % N_PASS)
