#!/usr/bin/env python3
"""sp_coupling_v1 P0e deploy: SRC -> site-packages + model_runner hook + lock.

Pattern cloned from deploy_p11_and_lock.py. Refuses on ANY precondition
violation (no silent repair). Writes a sha lock; refuses to overwrite it.

Deploys (copy):   config.py, controller.py, sp_state.py (new)
Patches (in-place exact-string): model_runner.py pre-forward gate hook
Asserts-only (must already equal SRC): other 9 pb_oeplb package files
"""
import hashlib
import os
import py_compile
import shutil
import subprocess
import sys
import time

SRC = "/workspace/EPLB/OEPLB/src"
DEP = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
MR = ("/opt/conda/lib/python3.11/site-packages/sglang/srt/model_executor/"
      "model_runner.py")
SP = ("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_"
      "20261012/results/sp_coupling_v1")
LOCK = os.path.join(SP, "deployed_shas_sp_coupling.txt")
BACKUP = os.path.join(SP, "backup_pre_sp_deploy")

# expected PRE-deploy state (v2.1 frozen deployed code)
EXPECT_DEP = {
    "config.py": "4ba8de2d09db",
    "controller.py": "62293ea9a451",
}
EXPECT_OTHER_SAME = [
    "__init__.py", "async_swapper.py", "fast_metadata.py",
    "per_request_tracer.py", "probe_admission.py", "rebalancer.py",
    "routing_tracer.py", "v2p1_signals.py", "window_v2.py", "window_v2p1.py",
]
EXPECT_MR = "751dd61c7a41"
COPY_FILES = ["config.py", "controller.py", "sp_state.py"]

MR_OLD = """        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
"""
MR_NEW = """        self.forward_pass_id += 1

        # sp_coupling_v1 (plan 2026-10-18, Sec.2): PRE-forward sampling gate.
        # Guarded: no-op unless the pb_oeplb controller is active AND provides
        # on_forward_pass_start (older deployed controllers keep working).
        if self._pb_oeplb_controller is not None and hasattr(
                self._pb_oeplb_controller, "on_forward_pass_start"):
            self._pb_oeplb_controller.on_forward_pass_start(forward_batch)

        with get_global_expert_distribution_recorder().with_forward_pass(
"""


def sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def fail(msg):
    print("DEPLOY-REFUSED: %s" % msg)
    sys.exit(1)


def main():
    # ---- preconditions ----------------------------------------------------
    if os.path.exists(LOCK):
        fail("lock already exists: %s (deploy already done; refusing)" % LOCK)
    out = subprocess.run(
        ["ps", "-eo", "pid,stat,cmd"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ("sglang" in line or "run_grid_bench" in line
                or "launch_server" in line):
            if " Z" in line.split()[1] or "defunct" in line:
                continue  # zombie: dead, harmless
            if "grep" in line or "deploy_sp_coupling" in line:
                continue
            fail("live server/bench process: %s" % line.strip())
    ss = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
    if ":30000 " in ss:
        fail("port 30000 occupied")
    for f, want in EXPECT_DEP.items():
        got = sha(os.path.join(DEP, f))
        if got != want:
            fail("deployed %s sha %s != expected pre-SP %s (unexpected "
                 "deployed state; investigate manually)" % (f, got, want))
    if os.path.exists(os.path.join(DEP, "sp_state.py")):
        fail("sp_state.py already deployed but lock missing -- partial state")
    for f in EXPECT_OTHER_SAME:
        s, d = sha(os.path.join(SRC, f)), sha(os.path.join(DEP, f))
        if s != d:
            fail("untouched file %s differs src=%s dep=%s" % (f, s, d))
    mr_sha_pre = sha(MR)
    if mr_sha_pre != EXPECT_MR:
        fail("model_runner sha %s != expected pre-SP %s (unexpected state; "
             "investigate manually)" % (mr_sha_pre, EXPECT_MR))
    for f in COPY_FILES:
        p = os.path.join(SRC, f)
        if not os.path.exists(p):
            fail("SRC missing %s" % f)
        py_compile.compile(p, doraise=True)
    print("preconditions OK (deployed v2.1 state verified, GPUs free)")

    # ---- backup ------------------------------------------------------------
    os.makedirs(BACKUP, exist_ok=True)
    for f in ["config.py", "controller.py"]:
        dst = os.path.join(BACKUP, f)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(DEP, f), dst)
    mr_bak = os.path.join(BACKUP, "model_runner.py")
    if not os.path.exists(mr_bak):
        shutil.copyfile(MR, mr_bak)
    print("backups written to %s" % BACKUP)

    # ---- deploy package files ----------------------------------------------
    for f in COPY_FILES:
        shutil.copyfile(os.path.join(SRC, f), os.path.join(DEP, f))
        if sha(os.path.join(SRC, f)) != sha(os.path.join(DEP, f)):
            fail("post-copy sha mismatch %s" % f)
        py_compile.compile(os.path.join(DEP, f), doraise=True)
    print("copied: %s" % ", ".join(COPY_FILES))

    # ---- patch model_runner --------------------------------------------------
    with open(MR) as f:
        text = f.read()
    if text.count(MR_OLD) != 1:
        fail("model_runner anchor count=%d (expected 1)" % text.count(MR_OLD))
    patched = text.replace(MR_OLD, MR_NEW)
    with open(MR, "w") as f:
        f.write(patched)
    py_compile.compile(MR, doraise=True)
    mr_sha_post = sha(MR)
    # diff confinement: exactly the 7 inserted lines, nothing else
    import difflib
    diff = list(difflib.unified_diff(text.splitlines(),
                                     patched.splitlines(), lineterm=""))
    adds = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
    rems = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    if len(rems) != 0 or len(adds) != 7:
        fail("model_runner diff not confined: +%d -%d" % (len(adds), len(rems)))
    print("model_runner patched: %s -> %s (+7/-0)" % (mr_sha_pre, mr_sha_post))

    # ---- import smoke ----------------------------------------------------------
    r = subprocess.run(
        [sys.executable, "-c",
         "from sglang.srt.managers.pb_oeplb import config, controller, "
         "sp_state;"
         "c = config.PBOEPLBConfig();"
         "c.validate_sp_coupling();"
         "print('IMPORT_OK', sp_state.SP_TABLE['TRACK'], "
         "hasattr(controller.PBOEPLBController, 'on_forward_pass_start'))"],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or "IMPORT_OK" not in r.stdout:
        fail("import smoke failed rc=%d out=%s err=%s"
             % (r.returncode, r.stdout[-300:], r.stderr[-500:]))
    print("import smoke: %s" % r.stdout.strip())

    # ---- lock --------------------------------------------------------------------
    lines = ["# sp_coupling_v1 deployed sha lock (LOCAL ONLY) -- %s"
             % time.strftime("%Y-%m-%d %H:%M:%S")]
    lines.append("# pre-deploy expected: config=%s controller=%s mr=%s"
                 % (EXPECT_DEP["config.py"], EXPECT_DEP["controller.py"],
                    EXPECT_MR))
    for f in sorted(os.listdir(DEP)):
        if f.endswith(".py"):
            lines.append("pb_oeplb/%s %s" % (f, sha(os.path.join(DEP, f))))
    lines.append("model_runner.py %s (pre %s)" % (mr_sha_post, mr_sha_pre))
    with open(LOCK, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("lock written: %s" % LOCK)
    print("DEPLOY_SP_COUPLING_OK")


if __name__ == "__main__":
    main()
