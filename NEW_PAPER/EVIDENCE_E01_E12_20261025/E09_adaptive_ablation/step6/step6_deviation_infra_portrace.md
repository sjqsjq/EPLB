# Step-6 deviation memo — infra-class boot failure (port race), driver fix, re-run rule

Timestamp: 2026-10-16 ~12:40 UTC. Written BEFORE the re-run start; approved classification below.

## Event

Phase-A cell W0_f32__s1 boot failed at 12:38:07 with
`torch.distributed.DistNetworkError: EADDRINUSE port 30234` (previous cell's scheduler still
held the dist port ~12s after kill; GPU memory had already cleared). The driver's
kill_server waited for GPU clear only. The start never served a single request:
**no bench data, no controller activity, no mechanism execution, zero observed results.**
Per protocol §6 it was logged RETIRED and the driver continued (W0_f64 booted OK at 12:39:30,
same short gap — the race is intermittent, est. ~1/3 per consecutive cell).

## Classification (pre-declared here, applies to all of step 6)

- **INFRA-class failure**: boot-stage environmental fault (port race, GPU not released by a
  dying process, node/driver fault) in which the mechanism never ran AND no result of any
  kind was observed. Because zero data exists, a re-run cannot cherry-pick outcomes:
  no hypothesis, arm, or condition is favored by repeating it. INFRA-class starts may be
  re-run ONCE with the fixed driver, under a new cell suffix (`r`), with this memo cited.
  The original artifacts are preserved untouched.
- **MECHANISM-class failure**: anything after the server begins serving (desync, v2 error
  fallback, bench incomplete, errors>0): RETIRED, NEVER retried (protocol §6, standing
  constraint 失败start退役不重试). No reclassification after the fact.

This classification is about failure STAGE, not about which arm failed; it was written before
any V2/ablation start exists and cannot be used to rescue a mechanism result.

## Fix (infrastructure only — no knob, dataset, protocol, or code-path change)

step6_common.sh kill_server: after pkill, `fuser -k -n tcp -9 30234` + wait until neither
30234 nor 30000 is in LISTEN (≤90s) + 5s settle. Frozen code hashes untouched (verify_hashes
still gates every start).

## Actions

1. W0_f32: original start retired (artifacts in phaseA/W0_f32__s1/); ONE re-run as
   phaseA/W0_f32__s1r at the END of the resume driver (declared now, before its result exists).
   F-local(W0) argmin uses the s1r measurement (the only valid measurement of that cell).
2. W0_f64__s1: bench was orphaned when the driver was killed mid-cell (driver kill was the
   infra intervention to stop further port-race retirements). Bench ran to completion under
   the identical frozen env; artifacts archived manually with a `PASS_manual_archive` log
   line. Env identity: EA_W=64, controller=1, α=0.0, dataset hash verified at driver start.
3. Phase-A resume: W1×{8,16,32,64}, W2×{8,16,32,64}, then W0_f32__s1r — same fixed order
   as protocol §4 for the remaining cells; no cell added or dropped.
4. Phases B/C source the patched step6_common.sh at launch (automatic).

## Budget impact

None beyond the wasted f32 boot (~2 min). Total starts remain ≤36 (+1 infra re-run = 37
process launches; measurement starts unchanged: 12 grid cells still measured once each).
