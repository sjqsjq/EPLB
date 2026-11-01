#!/bin/bash
# git_push.sh -- plumbing-only GitHub push (NEVER touches working tree/index/branch).
# Commit 1 = archive (PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/** + EXPERIMENT_MAP.md row).
# Commit 2 = code change ONLY (OEPLB/src/sp_state.py, AERO-RG patched 525e88f0).
# Stacked on origin/main, fast-forward push to main. Requires: EXPERIMENT_MAP.md already updated in place.
set -eu
cd /workspace/EPLB
REMOTE_HEAD=$(git ls-remote origin refs/heads/main | cut -f1)
LOCAL_KNOWN=$(git rev-parse origin/main)
if [ "$REMOTE_HEAD" != "$LOCAL_KNOWN" ]; then echo "ABORT: origin/main moved ($REMOTE_HEAD != $LOCAL_KNOWN); re-plan"; exit 3; fi
export GIT_INDEX_FILE=/tmp/rge2e_gitidx
rm -f /tmp/rge2e_gitidx
git read-tree "$REMOTE_HEAD"
GH=PAPER_EXPERIMENTS/recovery_admission_e2e_20261027
find "$GH" -type f | sort | while read -r f; do
  B=$(git hash-object -w "$f"); git update-index --add --cacheinfo 100644,"$B","$f"
done
B=$(git hash-object -w PAPER_EXPERIMENTS/EXPERIMENT_MAP.md); git update-index --add --cacheinfo 100644,"$B",PAPER_EXPERIMENTS/EXPERIMENT_MAP.md
TREE1=$(git write-tree)
C1=$(git commit-tree "$TREE1" -p "$REMOTE_HEAD" -m "experiments: recovery_admission_e2e_20261027 archive -- 7-arm e2e comparison (I/ES/ED/MT/DF/P0=AERO-base/P1=AERO-RG), 4 rounds x 7 arms = 28 formal boots + 3 prep, locked protocol/order/code-shas, round-paired stats with Holm; bulk per-boot data at /data/minghua/sjq/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027 (manifest inside); EXPERIMENT_MAP updated (5.3.1 multi-baseline e2e, 5.4 recovery admission old-vs-new)")
echo "COMMIT1_ARCHIVE=$C1"
B=$(git hash-object -w OEPLB/src/sp_state.py); git update-index --add --cacheinfo 100644,"$B",OEPLB/src/sp_state.py
TREE2=$(git write-tree)
C2=$(git commit-tree "$TREE2" -p "$C1" -m "code: OEPLB/src/sp_state.py AERO-RG recovery admission policy (env OEPLB_SP_RG, default 0 = byte-identical AERO-base). Sole change: in QUIET rev1, lone confirmed_grow no longer enters RECOVER; all other exits/priorities unchanged; hash tag sp1r1rg keeps mixed-version clusters at HOLD. NOTE: main-side blob was an empty placeholder; true original ba572f93 preserved in PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/code_lock/sp_state.py.orig_ba572f93 and the 79-line diff in code_lock/AERO_RG.patch.diff")
echo "COMMIT2_CODE=$C2"
git push origin "$C2:refs/heads/main"
git fetch origin refs/heads/main 2>/dev/null || true
echo "PUSHED main -> $C2"
echo "LINKS:"
echo "  https://github.com/sjqsjq/EPLB/commit/$C1"
echo "  https://github.com/sjqsjq/EPLB/commit/$C2"
echo "  https://github.com/sjqsjq/EPLB/tree/main/$GH"
rm -f /tmp/rge2e_gitidx
