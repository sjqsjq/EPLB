#!/usr/bin/env python3
"""rev1 deploy: SRC -> site-packages ON TOP OF the verified v2 lock state.

Preconditions (refuse on ANY violation, no silent repair):
  1. deployed_shas_sp_coupling_v2.txt verifies (sha256sum -c) -- we are on the
     known sealed P1 deployed state.
  2. deployed_shas_sp_coupling_rev1.txt does NOT exist yet.
  3. no live server/bench processes (zombies harmless), port 30000 free.
  4. SRC config/controller/sp_state are committed (git clean for these paths).
Action: copy the 3 rev1-touched files, py_compile, verify post-copy shas,
re-verify the other 12 locked paths unchanged, write the rev1 lock.
model_runner.py hook: UNCHANGED (already deployed under v2; rev1 does not
touch it).
"""
import hashlib
import os
import py_compile
import subprocess
import sys
import shutil
import time

SRC = "/workspace/EPLB/OEPLB/src"
DEP = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
MR = ("/opt/conda/lib/python3.11/site-packages/sglang/srt/model_executor/"
      "model_runner.py")
SP = ("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_"
      "20261012/results/sp_coupling_v1")
LOCK_V2 = os.path.join(SP, "deployed_shas_sp_coupling_v2.txt")
LOCK_R1 = os.path.join(SP, "deployed_shas_sp_coupling_rev1.txt")
COPY_FILES = ["config.py", "controller.py", "sp_state.py"]


def fail(msg):
    print("DEPLOY-REFUSED: %s" % msg)
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- preconditions ---------------------------------------------------------
if os.path.exists(LOCK_R1):
    fail("rev1 lock already exists: %s (deploy already done; refusing)" % LOCK_R1)
rc = subprocess.run(["sha256sum", "-c", "--quiet", LOCK_V2]).returncode
if rc != 0:
    fail("v2 lock does NOT verify (rc=%d) -- deployed state is not the sealed "
         "P1 state; investigate manually" % rc)
out = subprocess.run(["ps", "-eo", "pid,stat,cmd"], capture_output=True,
                     text=True).stdout
for line in out.splitlines():
    if ("sglang" in line or "run_grid_bench" in line
            or "launch_server" in line):
        parts = line.split()
        if len(parts) >= 2 and ("Z" in parts[1] or "defunct" in line):
            continue
        if "grep" in line or "deploy_sp_rev1" in line:
            continue
        fail("live server/bench process: %s" % line.strip())
ss = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
if ":30000 " in ss:
    fail("port 30000 occupied")
git = subprocess.run(["git", "-C", "/workspace/EPLB", "status", "--porcelain",
                      "--", "OEPLB/src/config.py", "OEPLB/src/controller.py",
                      "OEPLB/src/sp_state.py"],
                     capture_output=True, text=True).stdout.strip()
if git:
    fail("SRC rev1 files not committed:\n%s" % git)
for f in COPY_FILES:
    py_compile.compile(os.path.join(SRC, f), doraise=True)
print("preconditions OK (v2 lock verified, no live procs, SRC committed)")

# ---- deploy -----------------------------------------------------------------
for f in COPY_FILES:
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DEP, f))
    if sha256(os.path.join(SRC, f)) != sha256(os.path.join(DEP, f)):
        fail("post-copy sha mismatch %s" % f)
    py_compile.compile(os.path.join(DEP, f), doraise=True)
print("copied + compiled: %s" % ", ".join(COPY_FILES))

# ---- write rev1 lock (same path set as v2) ----------------------------------
paths = []
with open(LOCK_V2) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        paths.append(line.split()[1])
lines = ["# sp_coupling rev1 deployed sha lock (sha256sum -c format; "
         "author-directed limited revision REV1_quiet_exit_revision_20261018.md;"
         " rev-gated OEPLB_SP_REV, v1 frozen at rev=0) -- %s"
         % time.strftime("%Y-%m-%dT%H:%M:%S%z")]
for p in paths:
    if not os.path.exists(p):
        fail("locked path missing after deploy: %s" % p)
    lines.append("%s  %s" % (sha256(p), p))
with open(LOCK_R1, "w") as f:
    f.write("\n".join(lines) + "\n")
rc = subprocess.run(["sha256sum", "-c", "--quiet", LOCK_R1]).returncode
if rc != 0:
    fail("rev1 lock does not self-verify (rc=%d)" % rc)
print("rev1 lock written + self-verified: %s (%d paths)" % (LOCK_R1, len(paths)))
print("DEPLOY_REV1_OK")
