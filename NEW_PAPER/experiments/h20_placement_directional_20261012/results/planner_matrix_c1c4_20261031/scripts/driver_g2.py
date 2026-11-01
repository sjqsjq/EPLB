#!/usr/bin/env python3
# driver_g2.py - planner_matrix_c1c4_20261031 CPU matrix driver.
# Usage: python3 driver_g2.py SELFTEST|FULL
# Discipline (author fix3): CPU arithmetic only; NO verified-real-GPU-gain
# claims. The executed planner is the FROZEN BYTE COPY (code_lock/
# rebalancer_frozen_c250026b.py, sha256-verified against the freeze master)
# observed via sys.settrace line anchors - ZERO source modification, so
# byte-equivalence holds by construction; additionally every cell runs a
# no-tracer CONTROL pass and the two plans are asserted identical op-by-op.
# Matrix: C1-C4 x pair_rule {gap,maxdelta} x budgets {32,128,300} x thr
# {1.0962402945 production aperture PRIMARY, 1.02 rows labeled algorithm-
# diagnostic} + BI control cells (C1-C4, thr prod, budget 300).
# Protocol per T1 restoration: num_ranks=8 num_local=16 identity p2l
# max_swaps_per_layer=64 (param unused in frozen body, passed for fidelity)
# max_total_swap_layers=94; padded reporting rule w_e=128*ceil(n_e/128).
import hashlib, importlib.util, inspect, json, math, os, shutil, sys, time
import numpy as np
import torch

G2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP = os.path.dirname(G2)
FROZEN = os.path.join(G2, "code_lock", "rebalancer_frozen_c250026b.py")
MASTER = os.path.join(HP, "final_followup_20261020", "freeze", "code_fixed", "oeplb_src", "rebalancer.py")
THR_PROD = 1.0962402945
THR_DIAG = 1.02
DEAD_GATE_SCORE = 0.0096240295
MIN_SWAP_OPS = 8
MOVED_MB_PER_OP = 37.74  # 2 x 18.87 MB FP8 expert copy, ESTIMATE label

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_module():
    assert sha256(FROZEN) == sha256(MASTER), "frozen copy sha mismatch vs master"
    spec = importlib.util.spec_from_file_location("rebalancer_frozen_mod", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_snapshots():
    c = {}
    z = np.load(os.path.join(HP, "fullservice_v3", "setup", "counts_calibration.npz"))
    c["C1"] = np.asarray(z["P"], dtype=np.int64)
    z2 = np.load(os.path.join(HP, "cross_domain_e2e_20261027", "prep", "profile_build", "P_e3.npz"))
    c["C2"] = np.asarray(z2["P"], dtype=np.int64)
    for key, fn in (("C3", "book_profile_placement.json"), ("C4", "datafore_prover_placement.json")):
        p = os.path.join(HP, "book_prover_static_20261030", "placements", fn)
        j = json.load(open(p))
        c[key] = np.array(j["logical_count"], dtype=np.int64)
    for k, v in c.items():
        assert v.shape == (94, 128), (k, v.shape)
    return c

# ---------- anchors (content-based, occurrence within function line range) ----------
ANCHORS_MAIN = [
    ("round_start", lambda s: s == "ratio = best_ratio"),
    ("earlybreak", lambda s: s == "if best_layer is None or best_ratio < threshold_ratio:"),
    ("gapcalc", lambda s: s == "gap = hot_rank_sum - cold_rank_sum"),
    ("modebranch", lambda s: s.startswith("if ") and "OEPLB_PAIR_RULE" in s),
    ("maxdelta_assign", lambda s: s == "phys_a = hot_cands[0]"),
    ("target", lambda s: s == "target = gap / 2.0"),
    ("chosen", lambda s: s == "phys_a = best_a"),
    ("newratio", lambda s: s == "new_ratio = compute_ratio(best_layer)"),
    ("accept_if", lambda s: s == "if new_ratio < ratio - 0.0005:"),
    ("append", lambda s: s == "candidates.append(SwapOp("),
    ("exhausted", lambda s: s == "exhausted_layers.add(best_layer)"),
]
ANCHORS_BI = [
    ("round_start", lambda s: s == "ratio = best_ratio"),
    ("earlybreak", lambda s: s == "if best_layer is None or best_ratio < threshold_ratio:"),
    ("accept_if", lambda s: s == "if new_ratio < ratio - MARGIN:"),
    ("append", lambda s: s == "candidates.append(SwapOp("),
    ("exhausted", lambda s: s == "exhausted_layers.add(best_layer)"),
]

def build_anchor_map(src_lines, fname, anchors):
    di = None
    for i, l in enumerate(src_lines, 1):
        if l.startswith("def " + fname + "("):
            di = i
            break
    assert di is not None, fname
    de = len(src_lines)
    for i in range(di + 1, len(src_lines) + 1):
        if src_lines[i - 1].startswith("def ") and i > di + 1:
            de = i - 1
            break
    amap = {}
    for i in range(di, de + 1):
        s = src_lines[i - 1].strip()
        for tag, pred in anchors:
            if pred(s):
                amap.setdefault(i, tag)
    return amap, di, de

WATCH = ("best_layer", "best_ratio", "ratio", "gap", "new_ratio", "rank_hot",
         "rank_cold", "phys_a", "phys_b", "target", "best_score", "_delta",
         "_score", "_ha", "found_swap", "threshold_ratio", "hot_rank_sum",
         "cold_rank_sum", "best_a")
LOADS_TAGS = ("round_start", "accept_if")

def snapval(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "shape"):
        if getattr(v, "shape", None) == () or (hasattr(v, "numel") and v.numel() == 1):
            try:
                return v.item()
            except Exception:
                return None
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) <= 8:
        return [snapval(x) for x in v]
    return None

def make_tracer(fname, amap, events, max_events=400000):
    counter = [0]
    def local_trace(frame, event, arg):
        if event == "line":
            tag = amap.get(frame.f_lineno)
            if tag is not None and len(events) < max_events:
                L = frame.f_locals
                d = {"i": counter[0], "tag": tag, "line": frame.f_lineno}
                counter[0] += 1
                for k in WATCH:
                    if k in L:
                        sv = snapval(L[k])
                        if sv is not None:
                            d[k] = sv
                lc = L.get("lc")
                for src, dst in (("hot_cands", "hot_cands_head"), ("cold_cands", "cold_cands_head")):
                    hc = L.get(src)
                    if hc is not None and lc is not None:
                        try:
                            head = [int(x[0]) if isinstance(x, tuple) else int(x) for x in list(hc)[:4]]
                            d[dst] = head
                            if head:
                                d[dst[:4] + "0_load"] = int(lc[head[0]])
                        except Exception:
                            pass
                if "hot0_load" in d and "cold0_load" in d:
                    d["delta_max"] = d["hot0_load"] - d["cold0_load"]
                if lc is not None:
                    for vk, lk in (("besta_load", "best_a"), ("physa_load", "phys_a"), ("physb_load", "phys_b")):
                        if lk in L:
                            try:
                                d[vk] = int(lc[int(L[lk])])
                            except Exception:
                                pass
                if tag in LOADS_TAGS:
                    loads = L.get("loads")
                    if loads is not None:
                        try:
                            d["loads_list"] = [float(x) for x in loads]
                        except Exception:
                            pass
                if "candidates" in L:
                    try:
                        d["len_candidates"] = len(L["candidates"])
                    except Exception:
                        pass
                if "exhausted_layers" in L:
                    try:
                        d["len_exhausted"] = len(L["exhausted_layers"])
                    except Exception:
                        pass
                events.append(d)
        return local_trace
    def tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_name == fname:
            return local_trace
        return None
    return tracer

# ---------- ratio helpers (raw + T1 padded) ----------
def rank_loads_np(lc_layer, num_ranks=8, num_local=16, pad=False):
    v = lc_layer
    if pad:
        v = np.where(v == 0, 0, 128 * np.ceil(v / 128.0)).astype(np.int64)
    return np.array([v[r * num_local:(r + 1) * num_local].sum() for r in range(num_ranks)], dtype=np.float64)

def ratio_stats(counts, pad=False):
    per_layer = []
    for l in range(counts.shape[0]):
        rl = rank_loads_np(counts[l], pad=pad)
        avg = max(rl.sum() / 8.0, 1.0)
        per_layer.append(float(rl.max() / avg))
    return {"max": max(per_layer), "avg": sum(per_layer) / len(per_layer),
            "n_above_1_02": sum(1 for x in per_layer if x > 1.02)}

def apply_plan(counts, plan):
    c = counts.copy()
    for op in plan:
        t = op if isinstance(op, tuple) else tuple(op)
        layer, a, b = t[0], t[1], t[2]
        c[layer, a], c[layer, b] = c[layer, b].copy(), c[layer, a].copy()
    return c

# ---------- trial reconstruction from events ----------
def reconstruct(events, rule):
    trials = []
    cur = None
    rounds = 0
    earlybreak_reason = None
    exhausted_events = 0
    for e in events:
        tag = e["tag"]
        if tag == "round_start":
            rounds += 1
            cur = None
        elif tag == "earlybreak":
            if e.get("best_layer") is None:
                earlybreak_reason = "all_layers_exhausted"
            else:
                earlybreak_reason = "ratio_below_threshold"
            cur = None
        elif tag == "gapcalc":
            cur = {"round": rounds, "rank_hot": e.get("rank_hot"), "rank_cold": e.get("rank_cold"),
                   "gap": e.get("gap"), "delta_max": e.get("delta_max"),
                   "hot0_load": e.get("hot0_load"), "cold0_load": e.get("cold0_load"),
                   "branch": None, "best_score": None, "chosen_delta": None,
                   "accepted": None, "new_ratio": None, "ratio": e.get("ratio")}
            trials.append(cur)
        elif tag == "modebranch" and cur is not None:
            cur["branch"] = "maxdelta_env" if rule == "maxdelta" else "pending"
        elif tag == "maxdelta_assign" and cur is not None:
            cur["branch"] = "maxdelta"
        elif tag == "target" and cur is not None:
            cur["branch"] = "gap"
        elif tag == "chosen" and cur is not None:
            cur["best_score"] = e.get("best_score")
            if e.get("besta_load") is not None and e.get("cold0_load") is not None:
                cur["chosen_delta"] = e["besta_load"] - e["cold0_load"]
        elif tag == "newratio" and cur is not None:
            if e.get("physa_load") is not None and e.get("physb_load") is not None:
                cur["chosen_delta"] = e["physa_load"] - e["physb_load"]
        elif tag == "accept_if" and cur is not None:
            nr, rr = e.get("new_ratio"), e.get("ratio")
            cur["new_ratio"] = nr
            if nr is not None and rr is not None:
                cur["accepted"] = bool(nr < rr - 0.0005)
        elif tag == "exhausted":
            exhausted_events += 1
            cur = None
    if rule == "maxdelta":
        for t in trials:
            if t["branch"] in (None, "pending", "maxdelta_env"):
                t["branch"] = "maxdelta"
    for t in trials:
        if t["branch"] == "pending":
            t["branch"] = "maxdelta"  # default-rule fast branch (delta_max <= gap)
    return {"rounds": rounds, "earlybreak_reason": earlybreak_reason,
            "exhausted_events": exhausted_events, "trials": trials}

def exhibit(recon, rule):
    tr = recon["trials"]
    acc = [t for t in tr if t["accepted"]]
    rej = [t for t in tr if t["accepted"] is False]
    md = [t for t in acc + rej if t["branch"] == "maxdelta"]
    gp = [t for t in acc + rej if t["branch"] == "gap"]
    flips = [t for t in md if t.get("delta_max") is not None and t.get("gap") is not None
             and t["delta_max"] > t["gap"] / 2.0]
    strong = [t for t in md if t.get("delta_max") is not None and t.get("gap") is not None
              and t["delta_max"] > t["gap"]]
    feasible = [t for t in gp if t.get("best_score") is not None and t["best_score"] < float("inf")]
    near_half = [t for t in feasible if t.get("gap") and t["best_score"] <= 0.25 * t["gap"]]
    maxdiff_examples = []
    for t in gp:
        if t.get("delta_max") is not None and t.get("gap") is not None and t["delta_max"] - t["gap"] / 2.0 > 0 and t in feasible:
            maxdiff_examples.append({"round": t["round"], "gap": t["gap"], "delta_max": t["delta_max"],
                                     "overshoot_vs_half": t["delta_max"] - t["gap"] / 2.0,
                                     "chosen_delta": t.get("chosen_delta"), "best_score": t.get("best_score"),
                                     "accepted": t.get("accepted")})
    maxdiff_examples.sort(key=lambda x: -x["overshoot_vs_half"])
    return {"trials_total": len(tr), "accepted": len(acc), "rejected": len(rej),
            "branch_maxdelta": len(md), "branch_gap": len(gp),
            "maxdelta_flip_gt_half_gap": len(flips),
            "maxdelta_strong_overshoot_gt_gap": len(strong),
            "gapbranch_feasible_half_candidates": len(feasible),
            "gapbranch_near_half_le_025gap": len(near_half),
            "maxdiff_overshoot_with_feasible_half": len(maxdiff_examples),
            "maxdiff_top5": maxdiff_examples[:5]}

# ---------- cell runner ----------
def run_cell(mod, src_lines, counts, rule, budget, thr, thr_label, fname, amap, cellid, write_events=True):
    os.environ["OEPLB_PAIR_RULE"] = rule
    fn = getattr(mod, fname)
    sig = inspect.signature(fn)
    expect = ["logical_count", "physical_to_logical_map", "num_ranks", "num_local",
              "threshold_ratio", "max_swaps_per_layer", "max_total_swap_layers", "max_total_ops"]
    assert list(sig.parameters.keys()) == expect, (cellid, list(sig.parameters.keys()))
    p2l_id = torch.arange(128, dtype=torch.int64).unsqueeze(0).repeat(94, 1)
    lc_t = torch.from_numpy(counts.astype(np.int64))
    def args():
        return dict(logical_count=lc_t.clone(), physical_to_logical_map=p2l_id.clone(),
                    num_ranks=8, num_local=16, threshold_ratio=thr,
                    max_swaps_per_layer=64, max_total_swap_layers=94, max_total_ops=budget)
    t0 = time.perf_counter()
    plan_ctrl = fn(**args())
    t1 = time.perf_counter()
    events = []
    tracer = make_tracer(fname, amap, events)
    sys.settrace(tracer)
    try:
        plan_tr = fn(**args())
    finally:
        sys.settrace(None)
    t2 = time.perf_counter()
    identical = [tuple(x) for x in plan_ctrl] == [tuple(x) for x in plan_tr]
    assert identical, "TRACE PERTURBATION DETECTED in " + cellid
    recon = reconstruct(events, rule if fname == "try_build_swap_plan" else "bi")
    ex = exhibit(recon, rule if fname == "try_build_swap_plan" else "bi")
    init_raw = ratio_stats(counts, pad=False)
    init_pad = ratio_stats(counts, pad=True)
    final_counts = apply_plan(counts, [tuple(x) for x in plan_ctrl])
    fin_raw = ratio_stats(final_counts, pad=False)
    fin_pad = ratio_stats(final_counts, pad=True)
    plan_len = len(plan_ctrl)
    res = {
        "cellid": cellid, "snapshot": cellid.split("_")[0], "pair_rule": rule,
        "budget": budget, "threshold": thr, "threshold_label": thr_label,
        "planner": fname,
        "frozen_sha256": sha256(FROZEN),
        "plans_identical_ctrl_vs_traced": identical,
        "cpu_ms_control": round((t1 - t0) * 1000, 2),
        "cpu_ms_traced": round((t2 - t1) * 1000, 2),
        "event_count": len(events),
        "events_truncated": len(events) >= 400000,
        "snapshot_sum": int(counts.sum()),
        "init_ratio_raw": init_raw, "init_ratio_padded": init_pad,
        "final_ratio_raw": fin_raw, "final_ratio_padded": fin_pad,
        "plan_len": plan_len,
        "rounds": recon["rounds"], "earlybreak_reason": recon["earlybreak_reason"],
        "exhausted_events": recon["exhausted_events"],
        "exhibit": ex,
        "execution_permission_chain": {
            "threshold_aperture": thr_label,
            "early_exit_pred_init_max_le_thr": bool(init_raw["max"] <= thr),
            "plan_empty": plan_len == 0,
            "meets_min_swap_ops_8": bool(plan_len >= MIN_SWAP_OPS),
            "defer_gate_score_init_max_minus_1": round(init_raw["max"] - 1.0, 10),
            "dead_gate_score_reference": DEAD_GATE_SCORE,
            "bias_correct_env_unset_default_false": True,
            "moved_mb_estimate": round(plan_len * MOVED_MB_PER_OP, 1),
            "note": "CPU arithmetic only; production deferred=0 observed across 136 logs (G1/PBS evidence); no GPU gain claim",
        },
        "plan_ops_first5": [list(tuple(x)) for x in plan_ctrl[:5]],
    }
    cells_dir = os.path.join(G2, "analysis", "cells")
    ev_dir = os.path.join(G2, "analysis", "events")
    plans_dir = os.path.join(G2, "analysis", "plans")
    for d in (cells_dir, ev_dir, plans_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(cells_dir, cellid + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(plans_dir, cellid + ".json"), "w") as f:
        json.dump([list(tuple(x)) for x in plan_ctrl], f)
    if write_events:
        with open(os.path.join(ev_dir, cellid + ".jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e, sort_keys=True) + "\n")
    return res

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "SELFTEST"
    mod = load_module()
    src_lines = open(FROZEN).read().splitlines()
    amap_main, _, _ = build_anchor_map(src_lines, "try_build_swap_plan", ANCHORS_MAIN)
    amap_bi, _, _ = build_anchor_map(src_lines, "try_build_swap_plan_best_improvement", ANCHORS_BI)
    print("anchor map main:", len(amap_main), "bi:", len(amap_bi))
    snaps = load_snapshots()
    for k in sorted(snaps):
        print(k, "sum:", int(snaps[k].sum()), "max_init_ratio:", round(ratio_stats(snaps[k])["max"], 6))
    cells = []
    if mode == "SELFTEST":
        cells = [("C4", "gap", 300, THR_PROD, "prod1096", "try_build_swap_plan"),
                 ("C1", "maxdelta", 32, THR_DIAG, "diag102", "try_build_swap_plan"),
                 ("C1", "gap", 32, THR_PROD, "prod1096", "try_build_swap_plan_best_improvement")]
    else:
        for c in ("C1", "C2", "C3", "C4"):
            for rule in ("gap", "maxdelta"):
                for budget in (32, 128, 300):
                    for thr, lab in ((THR_PROD, "prod1096"), (THR_DIAG, "diag102")):
                        cells.append((c, rule, budget, thr, lab, "try_build_swap_plan"))
        for c in ("C1", "C2", "C3", "C4"):
            cells.append((c, "gap", 300, THR_PROD, "prod1096", "try_build_swap_plan_best_improvement"))
    results = []
    t00 = time.time()
    for (c, rule, budget, thr, lab, fname) in cells:
        cellid = "%s_%s_%d_%s%s" % (c, rule, budget, lab, "_BI" if fname.endswith("best_improvement") else "")
        r = run_cell(mod, src_lines, snaps[c], rule, budget, thr, lab, fname,
                     amap_bi if fname.endswith("best_improvement") else amap_main, cellid)
        results.append(r)
        print("CELL %s plan=%d rounds=%d acc=%d rej=%d initmax=%.4f finmax=%.4f ev=%d cpu=%.0f/%.0fms elapsed=%.0fs"
              % (cellid, r["plan_len"], r["rounds"], r["exhibit"]["accepted"], r["exhibit"]["rejected"],
                 r["init_ratio_raw"]["max"], r["final_ratio_raw"]["max"], r["event_count"],
                 r["cpu_ms_control"], r["cpu_ms_traced"], time.time() - t00))
        sys.stdout.flush()
    out = {"mode": mode, "n_cells": len(results), "frozen_sha256": sha256(FROZEN),
           "thr_prod": THR_PROD, "thr_diag": THR_DIAG, "results": results}
    with open(os.path.join(G2, "analysis", "matrix_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("MATRIX_WRITE analysis/matrix_results.json cells=%d wall=%.1fs" % (len(results), time.time() - t00))

if __name__ == "__main__":
    main()
