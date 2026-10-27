#!/usr/bin/env python3
"""AERO-RG offline branch tests + byte-identity sweep.

Compares three controller flavors on identical signal sequences:
  orig  = pre-patch frozen sp_state.py (ba572f93, code_lock backup)
  new0  = patched sp_state.py with OEPLB_SP_RG=0 (AERO-base; must be
          byte-identical to orig in decisions, reasons, hashes, records)
  new1  = patched sp_state.py with OEPLB_SP_RG=1 (AERO-RG)
Facts only; rc=0 iff every assertion passes.
"""
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
import random

CAM = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027"
SRC_NEW = "/workspace/EPLB/OEPLB/src/sp_state.py"
SRC_ORIG = CAM + "/code_lock/sp_state.py.orig_ba572f93"
NEW_SHA = "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466"
ORIG_SHA = "ba572f93e2d2c2558cfe2dd8cf0a750ed790e7f544f383cd6c5ebf3768b16329"
assert hashlib.sha256(open(SRC_NEW, "rb").read()).hexdigest() == NEW_SHA
assert hashlib.sha256(open(SRC_ORIG, "rb").read()).hexdigest() == ORIG_SHA

def load(path, name):
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m

M_ORIG = load(SRC_ORIG, "sp_orig")
M_NEW = load(SRC_NEW, "sp_new")

KW = dict(mode="adaptive", sp_s=1, sp_w=16, seed=20261018,
          quiet_streak=3, recover_min_cycles=2,
          z_lo=22.643858727592757, z_hi=314.14010272527486,
          theta_h_low=0.010095552687949321, n_min=12320768,
          theta_h_act=0.011395643112507271, rev=1)

def mk(mod, rg, **over):
    os.environ["OEPLB_SP_RG"] = "1" if rg else "0"
    kw = dict(KW)
    kw.update(over)
    return mod.SpController(**kw)

QUIET_SIG = {"z": 1.0, "h_rate": 0.005, "n_fresh": 98566144,
             "event": "hold", "swap_pending": False, "swaps_started": 0}
GROW_SIG = {"z": 1.0, "h_rate": 0.005, "n_fresh": 24641536,
            "event": "confirmed_grow", "swap_pending": False,
            "swaps_started": 0}

def drive_to_quiet(c):
    for i in range(3):
        p = c.propose(dict(QUIET_SIG))
        c.apply(p, h_id=i + 1)
    assert c.state == "QUIET", c.state

def js(d):
    return json.dumps(d, sort_keys=True)

results = []
def check(name, cond, detail=""):
    results.append({"name": name, "ok": bool(cond),
                    "detail": str(detail)[:200]})
    assert cond, "FAIL " + name + " " + str(detail)[:200]

# T1 lone confirmed_grow in QUIET: orig RECOVER(P1c) vs RG stays QUIET
a = mk(M_ORIG, 0); drive_to_quiet(a)
b0 = mk(M_NEW, 0); drive_to_quiet(b0)
b1 = mk(M_NEW, 1); drive_to_quiet(b1)
pa = a.propose(dict(GROW_SIG))
pb0 = b0.propose(dict(GROW_SIG))
pb1 = b1.propose(dict(GROW_SIG))
check("T1a_orig_recover_p1c",
      pa["transition"] and pa["state_after"] == "RECOVER"
      and pa["reasons"][0].startswith("P1c/P1d"), pa["reasons"][:1])
check("T1b_new0_identical_to_orig", js(pa) == js(pb0))
check("T1c_new1_stays_quiet",
      (not pb1["transition"]) and pb1["state_after"] == "QUIET",
      pb1["state_after"])
check("T1d_new1_annotation",
      any(r.startswith("[AERO-RG] lone confirmed_grow")
          for r in pb1["reasons"]), pb1["reasons"])

# T2 grow + swap_pending -> RECOVER under RG (not lone)
c2 = mk(M_NEW, 1); drive_to_quiet(c2)
s2 = dict(GROW_SIG); s2["swap_pending"] = True
p2 = c2.propose(s2)
check("T2_grow_plus_pending_recover",
      p2["transition"] and p2["state_after"] == "RECOVER"
      and p2["reasons"][0].startswith("P1c/P1d"), p2["reasons"][:1])
check("T2b_no_annotation_when_not_lone",
      not any(r.startswith("[AERO-RG] lone") for r in p2["reasons"]))

# T3 grow + swaps_started>0 -> RECOVER under RG (not lone)
c3 = mk(M_NEW, 1); drive_to_quiet(c3)
s3 = dict(GROW_SIG); s3["swaps_started"] = 2
p3 = c3.propose(s3)
check("T3_grow_plus_started_recover",
      p3["transition"] and p3["state_after"] == "RECOVER")

# T4 lone grow + z>=z_hi -> P1a priority preserved under RG
c4 = mk(M_NEW, 1); drive_to_quiet(c4)
s4 = dict(GROW_SIG); s4["z"] = 400.0
p4 = c4.propose(s4)
check("T4_p1a_priority",
      p4["transition"] and p4["state_after"] == "RECOVER"
      and any(r.startswith("P1a") for r in p4["reasons"]), p4["reasons"])

# T5 guardrail unchanged under RG
c5 = mk(M_NEW, 1); drive_to_quiet(c5)
s5 = dict(QUIET_SIG); s5["event"] = "uncertain_carry"
p5 = c5.propose(s5)
check("T5_guardrail_p1b",
      p5["transition"] and p5["state_after"] == "RECOVER"
      and p5["reasons"][0].startswith("P1b"), p5["reasons"][:1])

# T6/T7 all other change events unchanged under RG
for evname, tag in [("confirmed_shrink", "T6"), ("hold_alert", "T7a"),
                    ("urgent_shrink", "T7b"),
                    ("shrink_rejected_low_headroom", "T7c")]:
    c = mk(M_NEW, 1); drive_to_quiet(c)
    s = dict(QUIET_SIG); s["event"] = evname
    p = c.propose(s)
    check(tag + "_" + evname + "_recover",
          p["transition"] and p["state_after"] == "RECOVER"
          and p["reasons"][0].startswith("P1c/P1d"))

# T8 after ignoring lone grow, P2 coverage failure still exits to TRACK
c8 = mk(M_NEW, 1); drive_to_quiet(c8)
s8 = dict(GROW_SIG); s8["n_fresh"] = 1000
p8 = c8.propose(s8)
check("T8_p2_still_fires",
      p8["transition"] and p8["state_after"] == "TRACK"
      and any(r.startswith("P2") for r in p8["reasons"])
      and any(r.startswith("[AERO-RG] lone") for r in p8["reasons"]),
      p8["reasons"])

# T9 P3 sustained margin-rise exit unchanged under RG
c9 = mk(M_NEW, 1); drive_to_quiet(c9)
s9 = dict(GROW_SIG); s9["h_rate"] = 0.012
p9a = c9.propose(s9); c9.apply(p9a, h_id=10)
check("T9a_first_above_theta_stays",
      (not p9a["transition"]) and p9a["state_after"] == "QUIET"
      and p9a["exit_streak_after"] == 1, p9a["exit_streak_after"])
p9b = c9.propose(s9)
check("T9b_second_confirms_track",
      p9b["transition"] and p9b["state_after"] == "TRACK"
      and any(r.startswith("P3") for r in p9b["reasons"]), p9b["reasons"])
c9c = mk(M_NEW, 1); drive_to_quiet(c9c)
s9c = dict(QUIET_SIG); s9c["h_rate"] = 0.0105
p9c = c9c.propose(s9c)
check("T9c_band_resets_exit_streak",
      (not p9c["transition"]) and p9c["exit_streak_after"] == 0)

# T10 TRACK-phase props/hashes identical orig vs RG (h_rate kept
# >= theta_h_low so the pair never enters QUIET; QUIET-phase policy
# divergence is the mechanism itself, covered by T1/T8/T9)
t_o = mk(M_ORIG, 0); t_r = mk(M_NEW, 1)
rnd = random.Random(7)
same = True
hash_div = True
for i in range(12):
    s = dict(QUIET_SIG)
    s["event"] = rnd.choice(["hold", "confirmed_grow", "confirmed_shrink"])
    s["h_rate"] = rnd.choice([0.02, 0.05, 0.0105])
    po = t_o.propose(dict(s)); pr = t_r.propose(dict(s))
    if js(po) != js(pr):
        same = False
    if t_o.proposal_hash(po) == t_r.proposal_hash(pr):
        hash_div = False
    t_o.apply(po, h_id=i); t_r.apply(pr, h_id=i)
check("T10_track_identical",
      same and hash_div and t_o.state == "TRACK"
      and t_r.state == "TRACK", (t_o.state, t_r.state))

# T11 RECOVER-phase props/hashes identical orig vs RG
r_o = mk(M_ORIG, 0); drive_to_quiet(r_o)
r_r = mk(M_NEW, 1); drive_to_quiet(r_r)
salert = dict(QUIET_SIG); salert["z"] = 400.0
po = r_o.propose(salert); r_o.apply(po, h_id=5)
pr = r_r.propose(salert); r_r.apply(pr, h_id=5)
check("T11a_both_recover", r_o.state == "RECOVER" and r_r.state == "RECOVER")
same = True
hash_div2 = True
for i in range(3):
    po = r_o.propose(dict(QUIET_SIG)); pr = r_r.propose(dict(QUIET_SIG))
    if js(po) != js(pr):
        same = False
    if r_o.proposal_hash(po) == r_r.proposal_hash(pr):
        hash_div2 = False
    r_o.apply(po, h_id=6 + i); r_r.apply(pr, h_id=6 + i)
check("T11b_recover_identical_and_exit",
      same and hash_div2 and r_o.state == "TRACK"
      and r_r.state == "TRACK")

# T12 hash: base byte-identical to orig; RG tag diverges (mixed -> HOLD)
h_o = mk(M_ORIG, 0); drive_to_quiet(h_o)
h_0 = mk(M_NEW, 0); drive_to_quiet(h_0)
h_1 = mk(M_NEW, 1); drive_to_quiet(h_1)
so = h_o.propose(dict(QUIET_SIG))
s0 = h_0.propose(dict(QUIET_SIG))
s1 = h_1.propose(dict(QUIET_SIG))
check("T12a_base_hash_identical",
      h_o.proposal_hash(so) == h_0.proposal_hash(s0))
check("T12b_rg_hash_diverges",
      h_o.proposal_hash(so) != h_1.proposal_hash(s1))

# T14/T13 sweeps
EVENTS = ["hold", "init", "confirmed_grow", "confirmed_shrink",
          "hold_alert", "urgent_shrink", "urgent_shrink_capped",
          "shrink_blocked_dwell", "shrink_rejected_low_headroom",
          "uncertain_carry", "uncertain_hold", "forced_shrink_uncertain",
          "w_desync_freeze", "confirm_hold", "confirm_fail_freeze",
          "grow_blocked_dwell", ""]

def sweep(mod_a, rg_a, mod_b, rg_b, seed, steps=500):
    ca = mk(mod_a, rg_a); cb = mk(mod_b, rg_b)
    rnd = random.Random(seed)
    for st in range(steps):
        s = {"z": rnd.choice([None, 0.5, 25.0, 400.0, 1000.0]),
             "h_rate": rnd.choice([0.0, 0.005, 0.0105, 0.012, 0.5]),
             "n_fresh": rnd.choice([0, 1000, 3080192, 24641536, 98566144]),
             "event": rnd.choice(EVENTS),
             "swap_pending": rnd.choice([True, False]),
             "swaps_started": rnd.choice([0, 0, 1, 3])}
        p_a = ca.propose(dict(s)); p_b = cb.propose(dict(s))
        assert js(p_a) == js(p_b), "sweep seed %d step %d prop" % (seed, st)
        assert ca.proposal_hash(p_a) == cb.proposal_hash(p_b), \
            "sweep seed %d step %d hash" % (seed, st)
        ca.apply(p_a, h_id=st); cb.apply(p_b, h_id=st)
    return ca, cb

for seed in (11, 22, 33):
    ca, cb = sweep(M_ORIG, 0, M_NEW, 0, seed)
    assert js(ca.transitions) == js(cb.transitions), \
        "sweep seed %d transitions" % seed
    assert ca.residency == cb.residency, "sweep seed %d residency" % seed
check("T14_sweep_orig_vs_new0_3x500", True)
ca, cb = sweep(M_NEW, 1, M_NEW, 1, 44)
assert js(ca.transitions) == js(cb.transitions)
check("T13_sweep_rg1_dual_agreement_500", True)

# T15 fixed mode identical + RG guards raise
f_o = mk(M_ORIG, 0, mode="fixed", rev=0)
f_n = mk(M_NEW, 0, mode="fixed", rev=0)
pf_o = f_o.propose(dict(QUIET_SIG)); pf_n = f_n.propose(dict(QUIET_SIG))
check("T15a_fixed_identical",
      js(pf_o) == js(pf_n)
      and f_o.proposal_hash(pf_o) == f_n.proposal_hash(pf_n))
ok1 = False
try:
    mk(M_NEW, 1, mode="fixed", rev=0)
except ValueError as e:
    ok1 = "AERO-RG" in str(e)
ok2 = False
try:
    mk(M_NEW, 1, rev=0)
except ValueError as e:
    ok2 = "AERO-RG" in str(e)
check("T15b_guard_fixed_raises", ok1)
check("T15c_guard_rev0_raises", ok2)

# T16 transition record policy tagging
g1 = mk(M_NEW, 1); drive_to_quiet(g1)
s16 = dict(GROW_SIG); s16["n_fresh"] = 1000
p16 = g1.propose(s16); g1.apply(p16, h_id=9)
tr = g1.transitions[-1]
check("T16a_rg_record_tagged",
      tr["reasons"][0] == "[AERO-RG] policy=rg rev=1"
      and tr["from"] == "QUIET" and tr["to"] == "TRACK"
      and tr["h_id"] == 9, tr["reasons"][:2])
g0 = mk(M_NEW, 0); drive_to_quiet(g0)
p16b = g0.propose(s16); g0.apply(p16b, h_id=9)
tr0 = g0.transitions[-1]
check("T16b_base_record_untagged_p1c",
      tr0["reasons"][0].startswith("P1c/P1d") and tr0["to"] == "RECOVER"
      and not any("[AERO-RG]" in r for r in tr0["reasons"]),
      tr0["reasons"][:1])

# T17 snapshot/restore symmetry under RG
sn = mk(M_NEW, 1); drive_to_quiet(sn)
snap = sn.snapshot()
for i in range(2):
    p = sn.propose(dict(GROW_SIG)); sn.apply(p, h_id=30 + i)
sn.restore(snap)
check("T17_snapshot_restore",
      sn.state == snap["state"] and sn.streak == snap["streak"]
      and len(sn.transitions) == snap["n_transitions"]
      and sn.residency == snap["residency"] and bool(sn.rg))

out = {"all_pass": all(r["ok"] for r in results),
       "n_checks": len(results), "checks": results}
open(CAM + "/tests_rgv/branch_results.json", "w",
     encoding="utf-8").write(json.dumps(out, indent=1))
print("BRANCH_TESTS_PASS n_checks=%d" % len(results))
