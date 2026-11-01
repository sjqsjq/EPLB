# CORRECTION (append-only) - F3/F4 recompute from EXISTING events - 2026-10-31

No rerun, no GPU, no modification of frozen source/driver/matrix_results. This file and the two
CORRECTION_*.json companions are NEW artifacts derived by re-reading analysis/events/*.jsonl only.

## C-1. Recorded strict-flip ZERO was STRUCTURAL, not empirical (instrument defect)
- driver_g2.py :140-151 writes the hottest-available-slot load under key `hot_0_load`
  (because dst[:4] of "hot_cands_head" == "hot_"), then :150 tests `if "hot0_load" in d` ->
  guard never true -> `delta_max` was NEVER written to ANY event (verified: 0 delta_max keys,
  hot_0_load present in 294/294 and 413/413 gapcalc events of C1_*_300_prod1096).
- exhibit() flips/strong/maxdiff filters require t["delta_max"] is not None -> empty in ALL 52 cells
  regardless of data. F4's "ZERO in all 52 cells" is therefore a measurement gap, NOT a data fact.
- F4's population figure "4617 maxdelta-branch trials" is not reproducible from matrix_results
  fields (recorded branch_maxdelta sums: 13141 all cells / 9903 env=maxdelta cells). Flagged as-is.

## C-2. Event `gap` field is STALE (previous trial); fresh gap recovered in-event
- Anchor line :150 fires BEFORE execution -> locals["gap"] = previous trial's value.
  Verified on C1_gap_300_prod1096: 293/293 events' gap == previous gapcalc's (hot_rank_sum-cold_rank_sum);
  0/293 == same-event value. Same-event hot_rank_sum/cold_rank_sum (:148-149, WATCHed) ARE fresh.
- Validation of the fresh-gap reconstruction: restricted to gap-rule cells, branch==gap <=>
  hot0 > (hot_rank_sum-cold_rank_sum) agrees 3663/3663 trials, 0 disagreements
  (== the frozen code criterion lc[hot_cands[0]] <= gap at :156). env=maxdelta cells excluded
  (short-circuit at :156 makes the criterion unevaluated there by design).
- Impact on recorded numbers: feasible counts and branch classification UNAFFECTED (anchor/best_score based).
  near_half used stale gap: all-cells recorded 346 -> recomputed with fresh gap 332.
  F3 subset (prod1096, budget 300, gap cells): gp=183, near_half recorded(stale)=146, recomputed(fresh)=140.

## C-3. Recomputed strict-flip / strong-overshoot counts (from existing events, fresh gap)
- delta_max_rec = hot_0_load - cold0_load (both captured pre-swap at gapcalc anchor).
- Population: adjudicated trials (accepted is not None) with branch==maxdelta:
  all adjudicated trials of env=maxdelta cells + fast-branch trials of gap cells; both apertures.
- TOTALS: md_adjudicated=13141, strict flips (delta_max > gap/2) = 9298 (70.8%),
  strong overshoot (delta_max > gap) = 5981. Per-cell table: CORRECTION_counters_recomputed_20261031.json.
- Structural consistency: strong=0 in every gap-rule cell (fast branch requires hot0<=gap and
  delta_max<=hot0) - matches code semantics exactly.

## C-4. First candidate divergence, gap vs maxdelta, same snapshot/threshold/budget
- All 8 pairs with identical plan lengths (every budget-32 pair; C1/C2 budget-128): ZERO divergence
  across all trials - consistent with identical plans.
- All 12 pairs with differing plans diverge at the FIRST trial where hot0 > gap (gap-branch fires);
  state identical at divergence in 12/12 (layer/rank_hot/rank_cold/gap/hot0/cold0 equal both sides);
  delta_max > gap/2 at ALL 12 divergence points (in fact delta_max > gap in all 12).
- Divergence points (full detail in CORRECTION_first_divergence_20261031.json):
  * C1_300 (both apertures) trial 179, layer 86, hot rank 5, cold rank 0: gap=952179, gap/2=476089.5,
    hot0=1543262, cold0=42618, delta_max=1500644 (1.58x gap). GAP-side: phys_a=83, chosen_delta=457302
    (best_score 18787.5), REJECTED (new_ratio 1.162671 vs ratio 1.163032, gain 0.000361 < 0.0005).
    MD-side: phys_a=86 (=hottest), chosen_delta=1500644, REJECTED (new_ratio 1.230901).
    Trajectories split via different tried-slot marking on rejection (:196-199), not via acceptance.
  * C2_300 (both) trial 152, layer 5, hot 2, cold 0: gap=1012872, gap/2=506436, hot0=1226306,
    cold0=67825, delta_max=1158481. GAP-side phys_a=36 chosen_delta=368388 ACCEPTED (1.161569->1.073748);
    MD-side phys_a=37 chosen_delta=1158481 REJECTED (new_ratio 1.196281 worsens).
  * C3_128 & C3_300 (both apertures) trial 126, layer 20, hot 7, cold 3: gap=2902555, gap/2=1451277.5,
    hot0=2947928, cold0=1753, delta_max=2946175. GAP-side phys_a=122 chosen_delta=1685353 ACCEPTED
    (1.434471->1.387267); MD-side phys_a=116 chosen_delta=2946175 REJECTED (1.445697).
  * C4_128 & C4_300 (both apertures) trial 108, layer 82, hot 6, cold 7: gap=57675, gap/2=28837.5,
    hot0=59256, cold0=4, delta_max=59252. GAP-side phys_a=105 chosen_delta=9011 ACCEPTED
    (1.512040->1.372505); MD-side phys_a=98 chosen_delta=59252 REJECTED (1.536459).

## C-5. Resolution of the apparent paradox (recorded zero flips vs different final plans)
The two recorded facts came from DIFFERENT instruments: plan lengths/final ratios come from the
control runs (semantics untouched by the observation layer; plans_identical ctrl-vs-traced 52/52
still valid), while the flip counter read a field that never existed -> empty by construction.
Recomputed from the same events, strict flips are abundant (9298/13141) and the plan divergence
happens exactly at such overshoot trials: where hot0 > gap, the maxdelta rule moves delta_max
>(=)gap and overshoots (cold rank becomes new hot -> new_ratio worsens -> rejected or wasted),
while the gap rule moves a near-gap/2 delta that passes the :180 acceptance. No contradiction
ever existed in the data; the zero was the broken gauge. The frozen-code guard at :156
(hot0 <= gap) is a SCALE-MIXED criterion (slot load vs rank-sum difference): when it sends the
greedy branch, delta_max <= hot0 <= gap always, but delta_max can still exceed gap/2 (9298 such
trials) - i.e., the guard bounds overshoot vs gap, NOT vs gap/2.

## C-6. Missing data declared (no auto-rerun)
- Recorded events contain NO delta_max and NO hot0_load keys (only hot_0_load) - recomputation above
  used hot_0_load/cold0_load/hot_rank_sum/cold_rank_sum already present; nothing re-executed.
- BI cells: no gapcalc/pair-level anchors -> trial-level reconstruction still impossible (unchanged).
- CLOSEOUT_g2.md F3/F4 text is frozen and NOT edited; this correction file + LEDGER row supersede it.
