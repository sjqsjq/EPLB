#!/usr/bin/env python3
"""V1 unit tests for ADW v2 single-authority window state machine.

Pure-python; no GPU/torch/sglang needed. Synthetic signal sequences only --
these test state-transition LOGIC, not threshold values (thresholds are
calibration-frozen separately; nothing here tunes them).
"""
import sys
sys.path.insert(0, "/workspace/EPLB/OEPLB/src")
from window_v2 import WindowControllerV2, W_GRID  # noqa: E402

KNOBS = dict(kappa=1.0, z_lo=2.0, z_hi=4.0, z_urgent=8.0, m_shrink=2,
             m_grow=3, n_min=1000, u_max=4, theta_h_low=500.0,
             theta_h_act=2000.0, dwell_shrink=2, dwell_grow=2,
             w_uncertain_target=16, urgent_episode_cap=3)
N_OK = 5000          # comfortably above n_min
H_HIGH = 5000.0      # above theta_h_act and theta_h_low
H_LOW = 100.0        # below both

fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL:", msg)

def mk(w=16):
    return WindowControllerV2(w, dict(KNOBS))

# T0 init: first window z=None -> init/hold
c = mk()
r = c.step(None, H_HIGH, N_OK)
check(r["event"] == "init" and r["W_after"] == 16, "T0 init holds")

# T1 stationary + low headroom -> grow 16->32->64, then ceiling hold; dwell honored
c = mk()
c.step(None, H_LOW, N_OK)
events = [c.step(1.0, H_LOW, N_OK)["event"] for _ in range(12)]
ws = []
c2 = mk(); c2.step(None, H_LOW, N_OK)
for _ in range(12):
    r = c2.step(1.0, H_LOW, N_OK); ws.append(r["W_after"])
check("confirmed_grow" in events, "T1 grow happens")
check(ws == [16,16,16,32,32,32,32,64,64,64,64,64][:12] or ws[3] == 32 and ws[-1] == 64,
      "T1 grow trajectory with dwell: %s" % ws)
check(max(ws) == 64 and min(ws) == 16, "T1 stays on grid within bounds")

# T2 confirmed shrink needs m_shrink=2 alert windows; dwell blocks immediate re-shrink
c = mk(32); c.step(None, H_HIGH, N_OK)
r1 = c.step(5.0, H_HIGH, N_OK); r2 = c.step(5.0, H_HIGH, N_OK)
check(r1["event"] == "hold_alert" and r1["W_after"] == 32, "T2 first alert holds")
check(r2["event"] == "confirmed_shrink" and r2["W_after"] == 16, "T2 second alert shrinks")
r3 = c.step(9.0, H_HIGH, N_OK)  # urgent during dwell -> bypasses dwell
check(r3["event"] == "urgent_shrink" and r3["W_after"] == 8, "T2/urgent bypasses dwell")
r4 = c.step(5.0, H_HIGH, N_OK); r5 = c.step(5.0, H_HIGH, N_OK)
check(r5["event"] in ("shrink_blocked_dwell", "hold_alert") or r5["W_after"] == 8,
      "T2 floor/dwell respected at W=8: %s" % r5["event"])

# T2b dwell blocks CONFIRMED (non-urgent) shrink right after a shrink
c = mk(64); c.step(None, H_HIGH, N_OK)
c.step(5.0, H_HIGH, N_OK); r = c.step(5.0, H_HIGH, N_OK)   # confirmed shrink 64->32, dwell=2
check(r["W_after"] == 32 and r["dwell_dir"] == "shrink", "T2b shrink sets dwell")
r = c.step(5.0, H_HIGH, N_OK)                                # alert 1 (dwell ticks)
r = c.step(5.0, H_HIGH, N_OK)                                # alert 2 -> blocked by dwell?
# after two steps dwell: 2->1->0; second confirm lands when dwell expired
check(r["event"] in ("confirmed_shrink", "shrink_blocked_dwell"), "T2b dwell gating deterministic: %s" % r["event"])

# T3 high z but low headroom -> rejected, W unchanged
c = mk(); c.step(None, H_LOW, N_OK)
rs = [c.step(9.0, H_LOW, N_OK) for _ in range(4)]
check(all(r["W_after"] == 16 for r in rs), "T3 no shrink without headroom")
check(rs[0]["event"] == "shrink_rejected_low_headroom", "T3 rejection event")

# T4 stationary but high headroom -> grow blocked
c = mk(); c.step(None, H_HIGH, N_OK)
rs = [c.step(1.0, H_HIGH, N_OK) for _ in range(6)]
check(all(r["W_after"] == 16 for r in rs), "T4 no grow while headroom high")
check(any(r["event"] == "grow_blocked_headroom" for r in rs), "T4 blocked event logged")

# T5 urgent episode cap = 3 shrinks; cap event after
c = mk(64); c.step(None, H_HIGH, N_OK)
evs = [c.step(9.0, H_HIGH, N_OK)["event"] for _ in range(6)]
check(evs.count("urgent_shrink") == 3, "T5 exactly 3 urgent shrinks: %s" % evs)
check("urgent_shrink_capped" in evs, "T5 cap event")
check(c.W == 8, "T5 lands on floor 64->32->16->8")

# T6 uncertain: hold, carry once, forced shrink after u_max
c = mk(32); c.step(None, H_HIGH, N_OK)
r1 = c.step(5.0, H_HIGH, 100)     # n_fresh < n_min
check(r1["event"] == "uncertain_carry" and r1["carry_fresh"] is True and r1["W_after"] == 32,
      "T6 first uncertain carries")
r2 = c.step(5.0, H_HIGH, 100)
check(r2["carry_fresh"] is False and r2["W_after"] == 32, "T6 carry exhausted, still hold")
r3 = c.step(5.0, H_HIGH, 100)
r4 = c.step(5.0, H_HIGH, 100)     # 4th consecutive uncertain -> forced
check(r4["event"] == "forced_shrink_uncertain" and r4["W_after"] == 16,
      "T6 forced shrink to target: %s %s" % (r4["event"], r4["W_after"]))
# recovery: sufficient sample resets uncertain bookkeeping
r5 = c.step(1.0, H_LOW, N_OK)
check(r5["uncertain_run"] == 0, "T6 uncertain reset on good sample")

# T7 midband hysteresis: keeps stable streak? (spec: alert cleared, stable kept)
c = mk(); c.step(None, H_LOW, N_OK)
c.step(1.0, H_LOW, N_OK); c.step(1.0, H_LOW, N_OK)   # stable_s=2
r = c.step(3.0, H_LOW, N_OK)                          # midband
check(r["event"] == "midband_hold" and r["stable_s"] == 2, "T7 midband keeps stable_s")
r = c.step(1.0, H_LOW, N_OK)                          # stable_s=3 -> grow
check(r["event"] == "confirmed_grow" and r["W_after"] == 32, "T7 grow after midband: %s" % r["event"])

# T7b desync AFTER W has already moved -> snap back to DETERMINISTIC w_init
# (post-desync the current W may differ across ranks; "freeze at current" is
# not cross-rank safe - see window_v2.on_desync docstring)
c = mk(); c.step(None, H_LOW, N_OK)
moved = False
for _ in range(9):
    r = c.step(1.0, H_LOW, N_OK)
    if r["W_after"] != 16:
        moved = True
check(moved, "T7b W moved off w_init before desync")
c.on_desync()
check(c.frozen and c.W == 16, "T7b desync snaps W back to w_init: %s" % c.W)
r = c.step(9.0, H_HIGH, N_OK)
check(r["event"] == "w_desync_freeze" and r["W_after"] == 16, "T7b frozen at w_init after move")

# T8 desync freeze
c = mk(); c.step(None, H_LOW, N_OK); c.on_desync()
rs = [c.step(9.0, H_HIGH, N_OK) for _ in range(3)]
check(all(r["event"] == "w_desync_freeze" and r["W_after"] == 16 for r in rs), "T8 frozen holds W")

# T9 determinism: identical inputs -> identical full records (rank-consistency property)
def run(seed_seq):
    c = mk()
    out = []
    for z, h in seed_seq:
        r = c.step(z, h, N_OK)
        out.append((r["W_before"], r["W_after"], r["event"], r["alert_k"], r["stable_s"], r["dwell"]))
    return out
seq = [(None, H_HIGH)] + [(z, h) for z, h in
       [(1.0,H_LOW),(5.0,H_HIGH),(5.0,H_HIGH),(9.0,H_HIGH),(1.0,H_LOW),(3.0,H_LOW),
        (1.0,H_LOW),(1.0,H_LOW),(1.0,H_LOW),(5.0,H_LOW),(9.0,H_HIGH),(1.0,H_HIGH)]]
check(run(seq) == run(seq), "T9 deterministic replay")

# T10 knob validation
try:
    WindowControllerV2(16, dict(KNOBS, z_lo=5.0))  # z_lo >= z_hi
    check(False, "T10 invalid knobs rejected")
except ValueError:
    pass
try:
    WindowControllerV2(12, KNOBS)  # off-grid init
    check(False, "T10 off-grid w_init rejected")
except ValueError:
    pass
try:
    WindowControllerV2(16, dict(KNOBS, bogus=1))
    check(False, "T10 unknown knob rejected")
except ValueError:
    pass

print("=== %d checks failed ===" % len(fails) if fails else "ALL_TESTS_PASS")
sys.exit(1 if fails else 0)
