#!/usr/bin/env python3
"""Generate frozen run lists: run_list_main.csv (168) + run_list_p1.csv (32).

Rotation rule (PRE-REGISTERED, generated BEFORE any MAIN/P1 timing):
- Cells: MAIN = 4 conditions x arms x 8 blocks; P1 = 4 methods x 2 pressures x 4 blocks.
- Order: seeded constrained random shuffle, rng = random.Random(SEED_BASE + attempt),
  attempt = 1,2,3,... first success kept. Constraints (checked on run order):
  MAIN: no two adjacent cells share arm string; no two adjacent share condition.
  P1:   no two adjacent share method (P/ER0/ER16/ER32); no two adjacent share pressure.
- Seed, attempt, full order and per-cell positions persisted to rotation_*.json.
- Same fixed request stream per condition for ALL arms and blocks (plan §8.2/§9.4).
- CSV: lineterminator='\n', '\r' asserted absent, sha256 printed for driver lock.
"""
import csv, json, random, hashlib, time

F = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019"
HDR = ["seq", "phase", "cond", "block", "arm", "params", "bench_label", "artifact_dir"]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HDR)
        w.writerows(rows)
    data = open(path, "rb").read()
    assert b"\r" not in data, f"CR contamination in {path}"
    sha = hashlib.sha256(data).hexdigest()
    print(f"{path}: {len(rows)} rows sha={sha}")
    return sha

# ---------------- MAIN ----------------
E_PARAMS = "auto 16 256"   # E* = e5 (FINAL_SELECTION.json)
F_PARAMS = "1 16 16"       # F* = f_1_16_16 (FINAL_SELECTION.json)
COND_ARMS = {"E1": ["I", "E", "P", "F", "A3"],
             "E2": ["I", "E", "P", "F", "A2"],
             "E3": ["I", "E", "P", "F", "A2", "A4"],
             "E4": ["I", "E", "P", "F", "A1"]}
PARAMS = {"E": E_PARAMS, "F": F_PARAMS}

cells = []
for cond, arms in COND_ARMS.items():
    for arm in arms:
        for blk in range(1, 9):
            label = f"m_{cond}_{arm}_b{blk}"
            cells.append({"cond": cond, "block": blk, "arm": arm,
                          "params": PARAMS.get(arm, ""), "label": label,
                          "artifact": f"main/{label}"})
assert len(cells) == 168, len(cells)

def constrained_order(items, keyfns, seed_base, tag):
    """Seeded constrained random construction (pre-registered):
    rng = Random(seed_base + attempt); repeatedly draw uniformly among remaining
    items that differ from the previous item on ALL keyfns; dead end -> next
    attempt. Deterministic; first completed attempt frozen."""
    for attempt in range(1, 100001):
        rng = random.Random(seed_base + attempt)
        remaining = items[:]
        out = []
        prev = None
        dead = False
        while remaining:
            cands = [c for c in remaining
                     if prev is None or all(k(c) != k(prev) for k in keyfns)]
            if not cands:
                dead = True
                break
            pick = cands[rng.randrange(len(cands))]
            out.append(pick)
            remaining.remove(pick)
            prev = pick
        if not dead:
            print(f"{tag} rotation constructed: seed={seed_base + attempt} attempt={attempt}")
            return out, attempt, seed_base + attempt
    raise SystemExit(f"no valid {tag} rotation in 100000 attempts")

SEED_BASE_MAIN = 20261019 * 10000
order, used_attempt, used_seed = constrained_order(
    cells, [lambda c: c["arm"], lambda c: c["cond"]], SEED_BASE_MAIN, "MAIN")

rows = [[i + 1, "MAIN", c["cond"], c["block"], c["arm"], c["params"], c["label"], c["artifact"]]
        for i, c in enumerate(order)]
sha_main = write_csv(f"{F}/D_matrix/run_list_main.csv", rows)
json.dump({"seed_base": SEED_BASE_MAIN, "seed": used_seed, "attempt": used_attempt,
           "rule": "seeded constrained random construction: rng=Random(seed_base+attempt); draw uniformly among remaining cells differing from previous on arm AND cond; dead end -> attempt+1; first completion frozen",
           "constraints": "no adjacent same-arm; no adjacent same-cond (run order)",
           "generated_utc": time.strftime("%FT%T+0000", time.gmtime()),
           "n_cells": 168, "run_order": [c["label"] for c in order],
           "positions": {c["label"]: i + 1 for i, c in enumerate(order)},
           "csv_sha256": sha_main},
          open(f"{F}/D_matrix/rotation_main.json", "w"), indent=1)
print(f"MAIN rotation: seed={used_seed} attempt={used_attempt}")

# ---------------- P1 ----------------
METHODS = [("P", "P", ""), ("ER0", "E", "auto 0 256"),
           ("ER16", "E", "auto 16 256"), ("ER32", "E", "auto 32 256")]
PRESS = [("L", "PL"), ("H", "PH")]
pcells = []
for mname, arm, params in METHODS:
    for ptag, cond in PRESS:
        for blk in range(1, 5):
            label = f"p1_{mname}_{ptag}_b{blk}"
            pcells.append({"method": mname, "press": ptag, "cond": cond, "block": blk,
                           "arm": arm, "params": params, "label": label,
                           "artifact": f"p1/{label}"})
assert len(pcells) == 32, len(pcells)

SEED_BASE_P1 = 20261019 * 20000
porder, pattempt, pseed = constrained_order(
    pcells, [lambda c: c["method"], lambda c: c["press"]], SEED_BASE_P1, "P1")

prows = [[i + 1, "P1", c["cond"], c["block"], c["arm"], c["params"], c["label"], c["artifact"]]
         for i, c in enumerate(porder)]
sha_p1 = write_csv(f"{F}/D_matrix/run_list_p1.csv", prows)
json.dump({"seed_base": SEED_BASE_P1, "seed": pseed, "attempt": pattempt,
           "rule": "seeded constrained random construction (same algorithm as MAIN)",
           "constraints": "no adjacent same-method; no adjacent same-pressure (run order)",
           "generated_utc": time.strftime("%FT%T+0000", time.gmtime()),
           "n_cells": 32, "run_order": [c["label"] for c in porder],
           "positions": {c["label"]: i + 1 for i, c in enumerate(porder)},
           "csv_sha256": sha_p1},
          open(f"{F}/D_matrix/rotation_p1.json", "w"), indent=1)
print(f"P1 rotation: seed={pseed} attempt={pattempt}")
