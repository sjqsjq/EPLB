#!/usr/bin/env python3
"""Deploy the P11 planner (repo -> site-packages) and write the campaign code lock.

MUST run only AFTER the E-A calibration grid completes (frozen ep_v1 deployment
plan: deployed site-packages are NOT touched while the grid runs).

Steps (each refuses on precondition failure):
 1. grid done (campaign.log contains GRID_DONE; driver process gone)
 2. deployed controller/rebalancer/config shas == ea_v1_code_lock post-fix shas
 3. copy repo rebalancer.py + config.py -> deployed (byte-identical pre-P11)
 4. patch deployed controller.py with the three P11 edits by exact
    single-occurrence string replacement (keeps deployed-only HOOKDIAG diag)
 5. py_compile all three deployed files
 6. assert deployed-vs-repo controller diff confined to known regions
 7. rerun static switch-independence verification (v2 copy, new output path)
 8. write configs/followup_v1/eacampaign_code_lock.json (refuses overwrite)
"""
import hashlib, json, os, shutil, subprocess, sys, datetime, difflib

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
SITE = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
REPO = "/workspace/EPLB/OEPLB/src"
LOCK_OLD = os.path.join(EXP, "configs/followup_v1/ea_v1_code_lock.json")
LOCK_NEW = os.path.join(EXP, "configs/followup_v1/eacampaign_code_lock.json")
GRIDLOG = os.path.join(EXP, "results/ea_v1/calibration_grid/campaign.log")
STATIC_V1 = os.path.join(EXP, "scripts/verify_switch_independence_static.py")
STATIC_V2 = os.path.join(EXP, "scripts/verify_switch_independence_static_postp11.py")
STATIC_OUT_V2 = os.path.join(EXP, "results/ea_v1/switch_independence_static_postp11.json")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fail(msg):
    print("DEPLOY_REFUSED:", msg); sys.exit(1)

# ---- 1. grid done ----
if not os.path.exists(GRIDLOG) or "GRID_DONE" not in open(GRIDLOG).read():
    fail("grid not done (no GRID_DONE in campaign.log)")
r = subprocess.run(["pgrep", "-f", "run_ea_calibration_grid.sh"], capture_output=True, text=True)
if r.stdout.strip():
    fail("grid driver still alive: " + r.stdout.strip())

# ---- 2. deployed-side shas match the frozen locks ----
# (repo files legitimately changed after grid start: P11 edits, commit 5d2346e;
#  only DEPLOYED code was frozen during the grid.)
old = json.load(open(LOCK_OLD))
gp = json.load(open(os.path.join(EXP, "configs/followup_v1/ea_v1_calibration_grid_protocol.json")))
la = gp["locked_artifacts"]
expect = {
    os.path.join(SITE, "controller.py"): la["deployed_controller"]["sha256"],
    os.path.join(SITE, "config.py"): la["deployed_config"]["sha256"],
    os.path.join(SITE, "rebalancer.py"): old["deployed_package_files"]["rebalancer.py"]["sha256"],
}
for path, want in expect.items():
    got = sha(path)
    if got != want:
        fail(f"deployed {os.path.basename(path)} sha {got[:16]} != frozen {want[:16]} (mid-campaign change?)")
print("PRECHECK_OK deployed controller/config/rebalancer match frozen locks")

# ---- 3. copy rebalancer + config ----
for fname in ("rebalancer.py", "config.py"):
    src, dst = os.path.join(REPO, fname), os.path.join(SITE, fname)
    if sha(src) == sha(dst):
        print(f"copy {fname}: already identical"); continue
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst), f"copy {fname} failed"
    print(f"copy {fname}: deployed <- repo OK")

# ---- 4. patch controller (exact single-occurrence replacements) ----
ctl = os.path.join(SITE, "controller.py")
txt = open(ctl).read()
EDITS = [
 ("from sglang.srt.managers.pb_oeplb.rebalancer import try_build_swap_plan\n",
  "from sglang.srt.managers.pb_oeplb.rebalancer import try_build_swap_plan, try_build_swap_plan_best_improvement\n"),
 ('''                    f"sync_window={cfg.sync_window}, always_record={cfg.always_record}")\n''',
  '''                    f"sync_window={cfg.sync_window}, always_record={cfg.always_record}")\n'''
  '''        logger.info("[PB-OEPLB-PLANNER] rule="\n'''
  '''                    + ("best_improvement_rank_pair (P11 control)"\n'''
  '''                       if getattr(cfg, 'planner_best_improvement', False)\n'''
  '''                       else "heuristic_first_improvement (default)"))\n'''),
 ("            plan = try_build_swap_plan(\n",
  "            _planner_fn = (try_build_swap_plan_best_improvement\n"
  "                           if getattr(self.cfg, 'planner_best_improvement', False)\n"
  "                           else try_build_swap_plan)\n"
  "            plan = _planner_fn(\n"),
]
for i, (o, n) in enumerate(EDITS):
    cnt = txt.count(o)
    if cnt != 1:
        fail(f"controller edit #{i+1}: pattern occurs {cnt} times (expected 1)")
    txt = txt.replace(o, n)
open(ctl, "w").write(txt)
print("CONTROLLER_PATCHED (3 P11 edits, HOOKDIAG preserved)")

# ---- 5. py_compile ----
for fname in ("controller.py", "rebalancer.py", "config.py"):
    subprocess.run([sys.executable, "-m", "py_compile", os.path.join(SITE, fname)], check=True)
print("PY_COMPILE_OK")

# ---- 6. deployed-vs-repo controller diff confined to known regions ----
d = open(ctl).read().splitlines()
r = open(os.path.join(REPO, "controller.py")).read().splitlines()
diff = list(difflib.unified_diff(r, d, "repo/controller.py", "deployed/controller.py", lineterm=""))
changed = [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
def allowed_plus(body):
    s = body.strip()
    if not s: return True
    if s.startswith("#"): return True                      # comment wording (C3)
    if "_diag" in s or "HOOKDIAG" in s: return True        # deployed-only diagnostics
    if s in (")", "(") : return True                       # logger call punctuation
    if "import logging as _l" in s or "_l.getLogger" in s: return True
    if "expected_hits_per_fpe" in s: return True
    if "target={self.num_layers" in s: return True
    return False
# difflib alignment artifact: identical bodies appearing as both - and +
# (e.g. bare `return` lines surrounding inserted HOOKDIAG counters) cancel out.
from collections import Counter
minus_bodies = Counter(l[1:] for l in changed if l[0] == "-")
plus_bodies = Counter(l[1:] for l in changed if l[0] == "+")
identical = set(minus_bodies) & set(plus_bodies)
unknown = []
for l in changed:
    body = l[1:]
    if body in identical:
        continue                                           # no-op realignment
    if l[0] == "-":
        if body.strip() and not body.strip().startswith("#"):
            unknown.append(l)                              # repo-only: C3 comment lines only
    else:
        if not allowed_plus(body):
            unknown.append(l)
if unknown:
    print("UNKNOWN_DIFF_LINES:")
    for l in unknown[:40]:
        print("  ", l)
    fail(f"{len(unknown)} diff lines outside known HOOKDIAG/comment regions")
print(f"DIFF_CONFINED_OK ({len(changed)} changed lines, all HOOKDIAG/comment)")

# ---- 7. static verification v2 (new output path; v1 artifact untouched) ----
if not os.path.exists(STATIC_V2):
    v1 = open(STATIC_V1).read()
    assert "switch_independence_static.json" in v1
    v2 = v1.replace('switch_independence_static.json"', 'switch_independence_static_postp11.json"')
    open(STATIC_V2, "w").write(v2)
    print("WROTE static verifier v2 (only OUT path changed)")
if os.path.exists(STATIC_OUT_V2):
    fail("static v2 output already exists (already deployed? refusing double-deploy)")
res = subprocess.run([sys.executable, STATIC_V2], capture_output=True, text=True)
print(res.stdout[-800:])
if res.returncode != 0:
    print(res.stderr[-800:])
    fail("static verification v2 did not pass")
sv = json.load(open(STATIC_OUT_V2))
allpass = all(c.get("status") == "PASS" for f in sv.get("files", {}).values() for c in f.get("checks", {}).values()) if isinstance(sv.get("files"), dict) else None
print("STATIC_V2_OK sha", sha(STATIC_OUT_V2)[:16])

# ---- 8. campaign code lock ----
if os.path.exists(LOCK_NEW):
    fail("campaign code lock already exists")
files = {}
for fname in sorted(os.listdir(SITE)):
    if fname.endswith(".py"):
        files["deployed/" + fname] = sha(os.path.join(SITE, fname))
for fname in sorted(os.listdir(REPO)):
    if fname.endswith(".py"):
        files["repo/" + fname] = sha(os.path.join(REPO, fname))
lock = {
 "artifact": "eacampaign_code_lock",
 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "supersedes": "configs/followup_v1/ea_v1_code_lock.json (adds P11 planner deployment; C1/C2/C3/C7 unchanged)",
 "p11_edits_applied_to_deployed": {
   "rebalancer.py": "copied from repo (adds try_build_swap_plan_best_improvement, rule best_improvement_rank_pair)",
   "config.py": "copied from repo (adds planner_best_improvement field + OEPLB_PLANNER_BEST_IMPROVEMENT env fallback)",
   "controller.py": "patched in place: import, [PB-OEPLB-PLANNER] init log, _planner_fn branch (deployed-only HOOKDIAG counters preserved; diff vs repo confined to HOOKDIAG/comment regions, asserted)"},
 "a11_equivalence_note": "With planner_best_improvement=False (A00/A10/A01/A11), _planner_fn is try_build_swap_plan and all branches evaluate identically to ea_v1_code_lock code; the only additions are one init log line and an env-gated config field. P11 arm sets the env flag to 1.",
 "activation": "env OEPLB_PLANNER_BEST_IMPROVEMENT=1 via launch235b_eacampaign.sh EA_PLANNER_BI=1 (no server_args patch)",
 "static_verification_postp11": {"path": STATIC_OUT_V2, "sha256": sha(STATIC_OUT_V2)},
 "deployed_before_any_evaluation_start": True,
 "controller_deployed_vs_repo_diff_lines": changed,
 "files_sha256": files,
}
tmp = LOCK_NEW + ".tmp"
json.dump(lock, open(tmp, "w"), indent=1, ensure_ascii=False)
os.rename(tmp, LOCK_NEW)
print("WROTE", LOCK_NEW)
print("sha256", sha(LOCK_NEW))
print("DEPLOY_DONE")
