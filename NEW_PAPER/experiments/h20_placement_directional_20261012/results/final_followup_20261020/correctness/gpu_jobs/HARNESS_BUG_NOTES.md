# GPU harness bugs (test infrastructure only — NOT fix-code failures)

Attempt 1 (07:25Z): NameError — dead dict assignment left from CPU port. Removed.
Attempt 2 (07:28Z): G3_hand8 canary FAIL total=4, first=(l=3,wi=0,j=1,phys=81).
  Diagnosis: the 4 bad rows are exactly the slots moved by G2's same-rank op
  (phys 81/82 × W/S); G3's plan does not touch them. debug_g3.py rerun of the
  identical G3 plan on fresh state: BAD_ROWS=[] (PASS) with the same deployed
  executor. Root cause = replica reset to identity origins each scenario while
  live tensors accumulate permutations (same bug class as CPU-test S5, fixed
  there via global_orig; reintroduced in GPU formula-fill rewrite).
  Fix A: persistent M_STATE carried across scenarios, adopted only after
  canary PASS.
  Fix B: hand() now slot-disjoint (real planner invariant); previously
  G4/G5 at L=12 had 104/111 duplicate slot uses. Verified collision-free
  and rank-deterministic for n∈{8,298,299,300,301} at L=12 and L=94.
Deployed fix-code sha throughout: async_swapper 85a2dac882177667,
controller 4964cfc8e8af397a (unchanged; no fix-round budget consumed).

Attempt 3 (08:29Z): G1-G7 ALL PASS (empty/same-rank/8/298/299/300/301 +
PLANHASH identity + canary, slot-disjoint plans). G8_consec_churn_w1 failed:
RuntimeError "threads can only be started once" -- churn Thread joined after
w0 cannot restart. Fix: Churn class, fresh Thread+Event per scenario.
Planner dry-run: G9(L=12,seed7)=174 ops real legal plan; H4(L=94,seed71)=300
ops real legal cap plan. Fix-code sha unchanged (no fix-round consumed).

Attempt 4 / J1 (08:40Z): server job RAN FULLY CLEAN (0 watchdog, PLANHASH
identical across 8 ranks on all 14 plan windows incl. 3x 300-op cap plans at
windows #9/#17/#25 -- window#17 is the original hang window; bench ok=8192
err=0 in 191s matching old VALID E3 boots' 180.7s signature). Verdict FAIL
was an ASSERT-SCRIPT bug, not a job failure:
  (a) looked for bench.json but client writes {label}.json;
  (b) invariant p2l_version==sp_swaps_started is wrong -- validated on old
      VALID boots (m_E3_P_b1: p2l=12 issued=13; m_E3_F_b1: p2l=13 issued=14):
      correct invariant p2l_version in {issued-1, issued}; sp_swaps_started
      is the SP sampling counter (=1 even on VALID P boots).
Offline re-adjudication of the SAME on-disk artifacts: PASS (below).
Fix-code sha unchanged throughout (85a2dac8/4964cfc8); no fix-round consumed.
