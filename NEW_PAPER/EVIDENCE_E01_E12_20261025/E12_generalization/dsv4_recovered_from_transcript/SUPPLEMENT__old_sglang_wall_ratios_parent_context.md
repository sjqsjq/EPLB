# SUPPLEMENT — old-sglang (0.5.6.post2 tree) wall-ratio numbers from PARENT SESSION CONTEXT
Written: 2026-10-25 by parent session (evidence audit). Additive; does not modify any RECOVERED__* file.

## Provenance (honest labeling)
- Disk originals (EFFECT_REPORT_old_sglang.md, bench_old_{i,p}_{e2,e3,e4}.json) were READ BY THE PARENT
  SESSION BEFORE the 2026-10-25 author-ordered deletion, then deleted; they were committed to git only
  BEFORE these old-tree runs finished, so the recovery agent could not extract them from git objects
  (see UNRECOVERED__EFFECT_REPORT_old_sglang.md).
- The numbers below are quoted from the parent session context (originally read off the disk originals).
- Status: UNVERIFIED against disk originals at audit time (originals no longer exist anywhere on disk).
  They are registered here so the unfavorable old-tree results are not lost; they must be cited with
  this provenance caveat if used in the paper.

## Numbers (old tree = sglang 0.5.6.post2 + pb_oeplb v2, i.e. the CURRENT frozen deployment tree)
DSV4-Flash, P_fixab vs I, wall-clock ratio (s per fixed work unit), lower=better:
- e2: 4.3 -> 4.1  (-6.0%)
- e3: 4.4 -> 4.1  (-6.7%)
- e4: 229.6 -> 203.5  (-11.4%)
Direction: P still SLOWER than I on DSV4 in the old tree as well (negative result preserved).
Companion facts already git-recovered (verified): kappa=3875.1768798828125; E4 61/75 init windows =
53 empty + 8 ref-zeroing cascade; function-preservation gate PASS; 0.5.22-track medians
e2 40.9 (-3.8%) / e3 41.4 (-7.8%) / e4 2763.1 (-14.1%).

## Cross-references
- RECOVERED__fixab_comparison.md, RECOVERED__STATS_PATH_TRACE.md, KEY_NUMBERS.md (git-object-derived)
- UNRECOVERED__EFFECT_REPORT_old_sglang.md (recovery instructions for client-host transcript extraction)
