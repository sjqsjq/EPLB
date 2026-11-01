#!/usr/bin/env python3
"""Replay apsd2_P_b1 recorded v2p1 rows through orig/new0/new1 SpController.

strict (orig, new0): every row must reproduce recorded sp_state_before/after,
sp_transition and sp_reasons byte-identically; the 7 recorded transitions
must reappear with identical (h_id, from, to); proposal hashes must match
between orig and new0 row-for-row.
counterfactual (new1, AERO-RG): rows are replayed as recorded signals; the
trajectory legitimately diverges (recorded n_fresh/z come from the base-run
sampling schedule). Divergence is recorded, not judged. Full-system RG
behavior is measured by the GPU validation boot (PROTOCOL sec.4).
"""
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os

CAM = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027"
B = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/abl_prover_sd_20261026/boots_p2/apsd2_P_b1"
SRC_NEW = "/workspace/EPLB/OEPLB/src/sp_state.py"
SRC_ORIG = CAM + "/code_lock/sp_state.py.orig_ba572f93"

def load(path, name):
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m

M_ORIG = load(SRC_ORIG, "sp_orig_r")
M_NEW = load(SRC_NEW, "sp_new_r")

KW = dict(mode="adaptive", sp_s=1, sp_w=16, seed=20261018,
          quiet_streak=3, recover_min_cycles=2,
          z_lo=22.643858727592757, z_hi=314.14010272527486,
          theta_h_low=0.010095552687949321, n_min=12320768,
          theta_h_act=0.011395643112507271, rev=1)

EXPECTED = [(6, "TRACK", "QUIET"), (6, "QUIET", "RECOVER"),
            (8, "RECOVER", "TRACK"), (11, "TRACK", "QUIET"),
            (11, "QUIET", "RECOVER"), (13, "RECOVER", "TRACK"),
            (16, "TRACK", "QUIET")]

rows = [json.loads(l) for l in
        open(B + "/v2p1_rank0.jsonl", encoding="utf-8") if l.strip()]
assert len(rows) == 18, len(rows)

def run(mod, rg, strict):
    os.environ["OEPLB_SP_RG"] = "1" if rg else "0"
    c = mod.SpController(**KW)
    traj = []
    mism = []
    for i, r in enumerate(rows):
        sig = {"z": r.get("z"),
               "h_rate": float(r.get("h_rate") or 0.0),
               "n_fresh": int(r.get("n_fresh_assignments") or 0),
               "event": r.get("event") or "",
               "swap_pending": bool(r.get("sp_swap_pending")),
               "swaps_started": int(r.get("sp_swaps_started") or 0)}
        p = c.propose(sig)
        h = c.proposal_hash(p)
        rec = {"row": i, "window_id": r.get("window_id"),
               "before": c.state, "after": p["state_after"],
               "transition": bool(p["transition"]), "hash": h,
               "reasons": list(p["reasons"]),
               "recorded_after": r.get("sp_state_after"),
               "recorded_transition": bool(r.get("sp_transition"))}
        traj.append(rec)
        if strict:
            ok = (c.state == r.get("sp_state_before")
                  and p["state_after"] == r.get("sp_state_after")
                  and bool(p["transition"]) == bool(r.get("sp_transition"))
                  and list(p["reasons"]) == list(r.get("sp_reasons") or []))
            if not ok:
                mism.append({"row": i, "rec": rec,
                             "row_before": r.get("sp_state_before"),
                             "row_reasons": r.get("sp_reasons")})
        if r.get("committed"):
            c.apply(p, h_id=r.get("window_id"))
    return c, traj, mism

c_o, traj_o, mism_o = run(M_ORIG, 0, True)
assert mism_o == [], json.dumps(mism_o, indent=1)[:2000]
got = [(t["h_id"], t["from"], t["to"]) for t in c_o.transitions]
assert got == EXPECTED, got
print("REPLAY_ORIG_EXACT OK transitions=%d rows=%d" % (len(got), len(rows)))

c_n, traj_n, mism_n = run(M_NEW, 0, True)
assert mism_n == [], json.dumps(mism_n, indent=1)[:2000]
got_n = [(t["h_id"], t["from"], t["to"]) for t in c_n.transitions]
assert got_n == EXPECTED, got_n
assert [t["hash"] for t in traj_o] == [t["hash"] for t in traj_n]
assert (json.dumps(c_o.transitions, sort_keys=True)
        == json.dumps(c_n.transitions, sort_keys=True))
assert json.dumps(c_o.residency, sort_keys=True) == json.dumps(
    c_n.residency, sort_keys=True)
print("REPLAY_NEW0_IDENTICAL OK (states, reasons, hashes, records)")

c_r, traj_r, _ = run(M_NEW, 1, False)
got_r = [(t["h_id"], t["from"], t["to"]) for t in c_r.transitions]
n_ann = sum(1 for t in traj_r
            if any(str(x).startswith("[AERO-RG] lone") for x in t["reasons"]))
print("REPLAY_RG1_COUNTERFACTUAL transitions=%s lone_grow_annotations=%d"
      % (got_r, n_ann))
for t in traj_r:
    ann = any(str(x).startswith("[AERO-RG] lone") for x in t["reasons"])
    print("  w%-4s %-7s -> %-7s tr=%-5s rg_annot=%s rec_after=%s"
          % (t["window_id"], t["before"], t["after"], t["transition"],
             ann, t["recorded_after"]))

out = {"source_boot": "apsd2_P_b1",
       "orig_transitions": [list(x) for x in got],
       "new0_identical_to_orig": True,
       "rg1_transitions": [list(x) for x in got_r],
       "rg1_lone_grow_annotations": n_ann,
       "rg1_trajectory": traj_r,
       "note": ("rg1 replay is a policy-layer counterfactual on recorded "
                "base-run signals; full-system RG behavior is measured by "
                "the GPU validation boot")}
open(CAM + "/tests_rgv/replay_results.json", "w",
     encoding="utf-8").write(json.dumps(out, indent=1))
print("REPLAY_ALL_PASS")
