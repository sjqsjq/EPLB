import importlib.util, os, sys, json
SRC = "/workspace/EPLB/OEPLB/src"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

old_sp = load("old_sp_state", "/tmp/old_sp_state.py")
new_sp = load("new_sp_state", f"{SRC}/sp_state.py")
old_rb = load("old_rebalancer", "/tmp/old_rebalancer.py")
new_rb = load("new_rebalancer", f"{SRC}/rebalancer.py")
os.environ.pop("OEPLB_PAIR_RULE", None)

FAILS = []
def chk(cond, msg):
    if cond: print(f"  PASS {msg}")
    else:
        FAILS.append(msg); print(f"  FAIL {msg}")

# ---------- 1. sp_state equivalence ----------
PAR = dict(z_lo=1.0, z_hi=5.0, theta_h_low=0.1, n_min=10, theta_h_act=0.2)
SIGSEQ = [
    dict(z=0.5, h_rate=0.05, n_fresh=100, event="", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=0.5, h_rate=0.05, n_fresh=100, event="", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=0.5, h_rate=0.05, n_fresh=100, event="", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=3.0, h_rate=0.5, n_fresh=100, event="", swap_pending=False, swaps_started=1, plan_opp=True),
    dict(z=8.0, h_rate=0.5, n_fresh=100, event="", swap_pending=True, swaps_started=2, plan_opp=True),
    dict(z=None, h_rate=0.0, n_fresh=1, event="", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=2.0, h_rate=0.2, n_fresh=100, event="uncertain_hold", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=2.0, h_rate=0.2, n_fresh=100, event="w_desync_freeze", swap_pending=False, swaps_started=0, plan_opp=False),
    dict(z=0.2, h_rate=0.01, n_fresh=50, event="", swap_pending=False, swaps_started=0, plan_opp=True),
    dict(z=4.9, h_rate=0.9, n_fresh=200, event="", swap_pending=False, swaps_started=3, plan_opp=True),
]
def drive(ctl, steps=120):
    tr = []
    for i in range(steps):
        g = ctl.gate(i)
        prop = ctl.propose(SIGSEQ[i % len(SIGSEQ)])
        ctl.apply(prop, h_id=i)
        snap = ctl.snapshot()
        tr.append((g, json.dumps(prop, sort_keys=True, default=str),
                   json.dumps(snap, sort_keys=True, default=str),
                   (ctl.S, ctl.H, ctl.W, ctl.n_min_eff())))
    return tr

print("== 1. sp_state old-vs-new equivalence (defaults off) ==")
for mode, kw in [
    ("adaptive rev1", dict(mode="adaptive", sp_s=1, sp_w=16, rev=1, quiet_streak=3, recover_min_cycles=2, **PAR)),
    ("adaptive rev0", dict(mode="adaptive", sp_s=1, sp_w=16, rev=0, quiet_streak=3, recover_min_cycles=2, **PAR)),
    ("fixed 1,16,16", dict(mode="fixed", sp_s=1, sp_w=16, rev=0, **PAR)),
]:
    to = drive(old_sp.SpController(seed=20261018, **kw))
    tn = drive(new_sp.SpController(seed=20261018, **kw))
    states = sorted({json.loads(t[2]).get("state", "?") for t in tn})
    chk(to == tn, f"{mode}: 120-step trace identical (states: {states})")

# ---------- 2. fixed H ----------
print("== 2. fixed (S,H,W) via OEPLB_SP_H ==")
c = new_sp.SpController(mode="fixed", sp_s=2, sp_w=32, sp_h=16, **PAR)
chk((c.S, c.H, c.W) == (2, 16, 32), f"fixed 2,16,32 -> {(c.S,c.H,c.W)}")
c = new_sp.SpController(mode="fixed", sp_s=1, sp_w=16, **PAR)
chk((c.S, c.H, c.W) == (1, 16, 16), "sp_h=0 keeps H=W (F_dense compat)")
c = new_sp.SpController(mode="fixed", sp_s=4, sp_w=64, sp_h=16, **PAR)
chk((c.S, c.H, c.W) == (4, 16, 64), "fixed 4,16,64 (F calib candidate)")
for bad in [dict(sp_w=16, sp_h=32), dict(sp_w=16, sp_h=5), dict(sp_w=16, sp_h=-1)]:
    try:
        new_sp.SpController(mode="fixed", sp_s=1, **bad, **PAR); chk(False, f"reject {bad}")
    except ValueError: chk(True, f"reject {bad}")

# ---------- 3. force_s ----------
print("== 3. force_s adaptive (A3) ==")
kw = dict(mode="adaptive", sp_s=1, sp_w=16, rev=1, quiet_streak=3, recover_min_cycles=2, seed=20261018, **PAR)
n_f, n_p = new_sp.SpController(force_s=1, **kw), new_sp.SpController(**kw)
saw = False
for i in range(120):
    sig = SIGSEQ[i % len(SIGSEQ)]
    for c in (n_f, n_p): c.apply(c.propose(sig), h_id=i)
    if n_p.state == "QUIET":
        saw = True
        chk(n_f.state == "QUIET", "state machine unchanged by force_s")
        chk(n_f.S == 1 and n_p.S == 4, f"QUIET S: force=1 plain={n_p.S}")
        chk((n_f.H, n_f.W) == (n_p.H, n_p.W) == (16, 64), "QUIET H/W untouched")
        break
chk(saw, "QUIET reached")
c = new_sp.SpController(mode="fixed", sp_s=2, sp_w=16, force_s=1, **PAR)
chk(c.S == 2, "force_s inert in fixed mode")

# ---------- 4. rebalancer ----------
print("== 4. rebalancer old-vs-new + PAIR_RULE ==")
import torch
L, R, NL = 3, 8, 16
lc = torch.zeros(L, R*NL, dtype=torch.int64)
# layer0 overshoot: rank0 slot0=300 slot1=100 rest 1 (H=414); rank1 C=120
# (slot16=10, slot17=12, slots18-31=7); ranks2-7 = 160 each
lc[0, 0] = 300; lc[0, 1] = 100; lc[0, 2:16] = 1
lc[0, 16] = 10; lc[0, 17] = 12; lc[0, 18:32] = 7
lc[0, 32:128] = 10
# layer1 diffuse (identical under both rules): rank0 all 40, others 5
lc[1, 0:16] = 40; lc[1, 16:128] = 5
# layer2 balanced
lc[2, :] = 10
p2l = torch.arange(R*NL, dtype=torch.int64).unsqueeze(0).repeat(L, 1)
ARGS = dict(num_ranks=R, num_local=NL, threshold_ratio=1.02,
            max_swaps_per_layer=64, max_total_swap_layers=94, max_total_ops=80)
os.environ.pop("OEPLB_PAIR_RULE", None)
po = old_rb.try_build_swap_plan(lc, p2l, **ARGS)
pn = new_rb.try_build_swap_plan(lc, p2l, **ARGS)
chk([tuple(x) for x in po] == [tuple(x) for x in pn], f"default gap: plans identical ({len(po)} ops)")
os.environ["OEPLB_PAIR_RULE"] = "gap"
chk([tuple(x) for x in new_rb.try_build_swap_plan(lc, p2l, **ARGS)] == [tuple(x) for x in pn], "explicit gap == default")
os.environ["OEPLB_PAIR_RULE"] = "maxdelta"
pm = new_rb.try_build_swap_plan(lc, p2l, **ARGS)
os.environ.pop("OEPLB_PAIR_RULE", None)
chk([tuple(x) for x in pm] != [tuple(x) for x in pn], f"maxdelta plan differs ({len(pm)} vs {len(pn)} ops)")
g0 = [op for op in pn if op.layer_id == 0]; m0 = [op for op in pm if op.layer_id == 0]
chk(bool(g0) and g0[0].phys_slot_a == 1, f"gap layer0 first op = slot1(100), gap/2 branch (got {g0[0].phys_slot_a if g0 else None})")
chk(bool(m0) and m0[0].phys_slot_a == 0, f"maxdelta layer0 first op = slot0(300) (got {m0[0].phys_slot_a if m0 else None})")
chk(bool(g0) and bool(m0) and g0[0].phys_slot_b == m0[0].phys_slot_b == 18, "both pair with coldest-rank min slot 18")
l1g = [op for op in pn if op.layer_id == 1]; l1m = [op for op in pm if op.layer_id == 1]
chk([tuple(x) for x in l1g] == [tuple(x) for x in l1m], f"diffuse layer1 identical under both rules ({len(l1g)} ops)")
chk(all(op.rank_a == 0 for op in l1g), "layer1 ops from hot rank0")

# ---------- 5. config ----------
print("== 5. config env parse/validation ==")
new_cfg = load("new_config", f"{SRC}/config.py")
class DummySA:
    enable_pb_oeplb = True
    pb_oeplb_decay_factor = 0.0
    pb_oeplb_sync_window = 16
def mk(env):
    keys = set(env) | {"OEPLB_PAIR_RULE","OEPLB_SP_H","OEPLB_SP_FORCE_S","OEPLB_RECORD_ALL_PHASE","OEPLB_NO_COMMIT","OEPLB_WINDOW_CONTROLLER"}
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys: os.environ.pop(k, None)
    os.environ.update(env)
    try:
        return new_cfg.PBOEPLBConfig.from_server_args(DummySA())
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
BASE = {"OEPLB_SP_MODE": "adaptive", "OEPLB_SP_S": "1", "OEPLB_SP_W": "16",
        "OEPLB_SP_REV": "1", "OEPLB_SP_SEED": "20261018",
        "OEPLB_SP_QUIET_STREAK": "3", "OEPLB_SP_RECOVER_MIN_CYCLES": "2",
        "OEPLB_WINDOW_CONTROLLER": "21"}
c = mk(BASE)
chk((c.sp_h, c.sp_force_s, c.record_all_phase, c.no_commit, c.pair_rule) == (0,0,0,0,'gap'), "P_final defaults: all five flags OFF")
c = mk({**BASE, "OEPLB_SP_H": "16", "OEPLB_SP_FORCE_S": "1", "OEPLB_RECORD_ALL_PHASE": "1", "OEPLB_NO_COMMIT": "1", "OEPLB_PAIR_RULE": "maxdelta"})
chk((c.sp_h, c.sp_force_s, c.record_all_phase, c.no_commit, c.pair_rule) == (16,1,1,1,'maxdelta'), "env parse all five flags")
for env, expect in [
    ({**BASE, "OEPLB_PAIR_RULE": "bogus"}, "OEPLB_PAIR_RULE"),
    ({"OEPLB_SP_MODE": "fixed", "OEPLB_SP_S": "1", "OEPLB_SP_W": "16", "OEPLB_SP_REV": "0", "OEPLB_SP_FORCE_S": "1", "OEPLB_WINDOW_CONTROLLER": "21"}, "OEPLB_SP_FORCE_S"),
    ({"OEPLB_SP_MODE": "off", "OEPLB_SP_REV": "0", "OEPLB_RECORD_ALL_PHASE": "1"}, "OEPLB_RECORD_ALL_PHASE"),
    ({"OEPLB_SP_MODE": "fixed", "OEPLB_SP_S": "1", "OEPLB_SP_W": "16", "OEPLB_SP_REV": "0", "OEPLB_SP_H": "-1", "OEPLB_WINDOW_CONTROLLER": "21"}, "OEPLB_SP_H"),
]:
    try:
        mk(env).validate_sp_coupling(); chk(False, f"reject {expect}")
    except (ValueError, RuntimeError) as e:
        chk(expect in str(e), f"reject {expect}")
# off-mode early return must NOT skip the new validations
try:
    mk({"OEPLB_SP_MODE": "off", "OEPLB_SP_REV": "0", "OEPLB_PAIR_RULE": "bogus"}).validate_sp_coupling()
    chk(False, "pair_rule validated even in off mode")
except ValueError:
    chk(True, "pair_rule validated even in off mode")

print("\n" + ("ALL_TESTS_PASS" if not FAILS else f"TEST_FAILURES={FAILS}"))
sys.exit(0 if not FAILS else 1)
