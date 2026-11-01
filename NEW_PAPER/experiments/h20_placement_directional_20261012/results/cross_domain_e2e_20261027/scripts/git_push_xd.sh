#!/bin/bash
# git_push_xd.sh -- plumbing-only GitHub push for cross_domain_e2e_20261027 (NEVER touches working tree/index/branch).
# Single archive commit (no code change this campaign): PAPER_EXPERIMENTS/cross_domain_e2e_20261027/** + EXPERIMENT_MAP.md.
set -eu
cd /workspace/EPLB
REMOTE_HEAD=$(git ls-remote origin refs/heads/main | cut -f1)
LOCAL_KNOWN=$(git rev-parse origin/main)
if [ "$REMOTE_HEAD" != "$LOCAL_KNOWN" ]; then echo "ABORT: origin/main moved ($REMOTE_HEAD != $LOCAL_KNOWN); re-plan"; exit 3; fi
echo "GUARD_OK origin/main=$REMOTE_HEAD"
export GIT_INDEX_FILE=/tmp/xd_gitidx
export GIT_AUTHOR_NAME=sjq GIT_AUTHOR_EMAIL=sjq@eplb.dev GIT_COMMITTER_NAME=sjq GIT_COMMITTER_EMAIL=sjq@eplb.dev
rm -f /tmp/xd_gitidx
git read-tree "$REMOTE_HEAD"
GH=PAPER_EXPERIMENTS/cross_domain_e2e_20261027
find "$GH" -type f | sort | while read -r f; do
  B=$(git hash-object -w "$f"); git update-index --add --cacheinfo 100644,"$B","$f"
done
B=$(git hash-object -w PAPER_EXPERIMENTS/EXPERIMENT_MAP.md); git update-index --add --cacheinfo 100644,"$B",PAPER_EXPERIMENTS/EXPERIMENT_MAP.md
TREE1=$(git write-tree)
C1=$(git commit-tree "$TREE1" -p "$REMOTE_HEAD" -m "experiments: cross_domain_e2e_20261027 archive -- 6-arm cross-domain e2e (I/ES/ED/MT/DF/P0=AERO-base) x 4 rounds = 24 formal boots + 3 prep, frozen E3 4-phase stream (sharegpt->prover->book->code, in512/out1), locked protocol/order/code+placement shas, round-paired t(df=3)+Holm all ns (min raw p=0.0499 P0_vs_DF); sec5.5 timeline: P0 swaps at boundary+1 window all boots, ED corrected to 2 true rebalance events/boot; bulk per-boot data + 48MB recorder pt at /data/minghua/sjq/PAPER_EXPERIMENTS/cross_domain_e2e_20261027 (manifest inside); EXPERIMENT_MAP updated (5.3.1 cross-domain main comparison, 5.5 boundary timeline case)")
echo "COMMIT1_ARCHIVE=$C1"
git push origin "$C1:refs/heads/main"
git fetch origin refs/heads/main 2>/dev/null || true
echo "PUSHED main -> $C1"
echo "LINKS:"
echo "  https://github.com/sjqsjq/EPLB/commit/$C1"
echo "  https://github.com/sjqsjq/EPLB/tree/main/$GH"
rm -f /tmp/xd_gitidx
