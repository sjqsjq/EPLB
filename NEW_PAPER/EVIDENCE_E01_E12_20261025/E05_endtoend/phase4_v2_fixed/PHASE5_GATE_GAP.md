# Phase-5 (PH256, conditional ≤16 boots) — precondition text UNREACHABLE; package NOT auto-launched

**Date**: 2026-10-21. **Status**: gap reported to author per directive ("如果预算、兼容性
或数据有实际阻塞，报告具体缺口；不得自行扩预算…").

## What happened
The author directive defining phase 5 was delivered in-conversation before a context
compaction; the compaction summary preserves only "五: conditional ≤16 PH256 boots" —
the gate/trigger condition text was lost. The session transcript on the runner host
(/home/admin/.claude/projects/…afd3510d….jsonl) is unreachable from every tool
available in this session (main and subagents execute on the GPU pod only; a
dedicated extraction attempt confirmed this — see subagent final report 2026-10-21).

## What pod-side artifacts DO establish
- Phase-5 is a CONDITIONAL package of ≤16 boots inside the total formal budget ≤104
  (64 phase-4 + ≤16 phase-5 + others; quoted in PHASE4_CODE_VERSION_ADDENDUM.md).
- The string "PH256" appears nowhere in pod artifacts or git history; name inference
  (flagged as inference): matches old-campaign P1 "PH @ C_hi=256" cells (P arm,
  p1_longio.jsonl, conc 256; FREEZE_final_plan.md §8) — plausibly a P-arm high-pressure
  replication. NOT verifiable; NOT acted upon.
- Governance that would bind phase-5 regardless of gate: v2 frozen code only;
  fix rounds 2/2 exhausted (any same-class fault ⇒ STOP + blocker report);
  unmet-precondition packages not run and savings NOT converted.

## Decision
Phase-5 will NOT be auto-launched after phase-4. Per the directive, a package whose
precondition cannot be verified is treated as unmet-precondition: not run, savings
not converted. The author is asked to either (a) restate the phase-5 gate text, or
(b) confirm phase-5 is skipped. Until then phase-4 (64 boots) proceeds independently —
it is unconditional — and phase-6 remains STOPPED (eligibility NOT_ELIGIBLE, commit bab38ca).
This gap is carried into the phase-8 仍未知 list.
