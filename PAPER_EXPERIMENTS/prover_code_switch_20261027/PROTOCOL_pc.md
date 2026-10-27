# PROTOCOL - prover_code_switch_20261027 (pre-registered, frozen before first boot)

## 0. Origin and authorization
- Author finding 2026-10-27: cross_domain_e2e_20261027 ran the WRONG dataset
  (e3_multidomain_switch instead of paper.md 5.3.1 crossdomain_freq6); see
  LEDGER_xd.tsv AUTHOR_CHALLENGE / INVESTIGATION / RETEST_PLAN rows.
- Author new spec (this campaign): dual-domain switch stream of 2048 prover
  requests + 2048 code requests; configs per paper 5.3.1; ours arm = the NEW
  DESIGN scheme (frozen P0 AERO-base rev1); 4 rounds + rotation.
- Author mid-turn corrections (binding): output length = 1 for ALL rows
  (NOT paper freq6 O=10); external code data pull allowed if short - NOT
  needed: frozen e3 code block (DOMGROUP 5-source spliced: code +
  repobench-p_e + lcc_e + repobench-p + lcc, retok L512 s256) has exactly
  2048 rows with sha-locked lineage.
- Author decisions (AskUserQuestion 2026-10-27): stream structure = TWO BIG
  BLOCKS single switch; ED period = 100 per paper.
- GPU authorization: author retest order + spec messages ("看看效果") =
  authorization for <=24 formal boots + <=2 zero-data INFRA restarts.

## 1. Stream (frozen)
- B_data/streams/pc_prover_code_switch.jsonl
  sha256 = 01e33f82de54238da2ad342036998fe31a244c519339bfec91983ac45c773f69
  n = 4096, uniq text hashes = 4096.
- Block1 idx 0..2047    = prover_2048tok_out1.jsonl all rows (in ~2048 tok).
- Block2 idx 2048..4095 = e3_multidomain_switch.jsonl rows[6144:8192]
  (e3_code_00000..02047, in ~512 tok). Single switch at idx 2048.
- ALL rows: max_tokens = 1, temperature = 0. Registry: B_data/registry.json.

## 2. Arms (6) and paper 5.3.1 mapping
| arm | config | paper lineage |
|-----|--------|---------------|
| I   | bare deepep auto | identity 4.7 req/s (freq6) |
| ES  | normal + R16 + --init-expert-location datafore_prover_placement.json | EPLB static (prover-frozen) 4.5 |
| DF  | auto + --init-expert-location datafore_prover_placement.json (no redundancy) | DataForest-Remap (prover放置) 4.7 |
| MT  | auto + --init-expert-location placement_pinned_fair.json (no redundancy) | MoETuner prover fair-split ILP1 4.63 |
| ED  | normal + R16 + --enable-eplb + algorithm auto + period 100 + recorder buffer 32 | EPLB dynamic 4.4 (period per paper) |
| P0  | frozen AERO-base rev1 RG=0, v2.1 knobs, PB flags verbatim | ours = NEW DESIGN (author-specified; paper arm was legacy sw16 - disclosed delta) |
Disclosed deltas vs paper (uniform across arms, no arm favored):
  D1 stream = author-specified prover<->code two-block (paper: freq6 book<->prover 6seg).
  D2 O=1 all rows (author-specified; paper freq6 O=10).
  D3 mem-fraction 0.8 (Oct campaign standard; paper base used 0.78).
  D4 model path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 (same weights as /data/models copy).
  D5 harness bench_client_v2 (closed-loop conc=32, OpenAI completions, streaming, per-request jsonl) vs paper freq6_bench.py (closed-loop conc=32, /generate).
  D6 ours arm = current frozen implementation (code locks below), not the Sep legacy flags.
  D7 ED keeps --eplb-algorithm auto (Oct campaign standard; paper did not pass the flag).

## 3. Workload
N = 4096 per boot, conc = 32 (paper), BENCH_TIMEOUT = 1800s.
Primary metric: req/s = ok / wall. Secondary: tps, phase req/s (prover block
vs code block from per-request jsonl), boundary reaction (P0 windows, ED
rebalance timestamps vs boundary crossing).

## 4. Guard chain (per boot, run_pc_boot.sh)
stream sha -> code locks (async_swapper 2e4652a0…, controller 4964cfc8…,
sp_state 525e88f0…, window_v2 810c54da…) -> tool locks (launch_arm_pc.sh
cfbdaee752283935…, bench_client_v2 06668243…) -> placement locks
(code_lock/placement_shas.env: PL_ES=PL_DF=c1787ad1…, PL_MT=299cea28…) ->
boot_meta.json records all shas -> ready protocol (900s cap, ready to roll /
fatal-grep / pid-alive) -> bench -> gpu-clear kill -> mech_check_pc.py.
Guard fail rc=2 -> QUEUE HALT. Boot fail rc=1 -> log + continue (triage
within INFRA budget <=2 zero-data restarts, never silently dropped).

## 5. Rotation and budget
r1: ES ED MT DF P0 I | r2: ED MT DF P0 I ES | r3: MT DF P0 I ES ED |
r4: DF P0 I ES ED MT   (24 boots, same scheme as xd campaign)
Per boot: launch ~5-7min + bench est 3-8min (4096 req O1 conc32) + teardown
~2min. ED may run longer (period 100 => multiple full rebalances). Total est
4.5-6h. Queue detached (setsid) with log; polling only, no blocking waits.

## 6. Mechanism check - FACTS ONLY
ED rebalance starts RECORDED NOT GATED (lesson: xd campaign 16 log rows =
2 events x 8 DP ranks; with O=1 batch-level steps, event count is not
request-count predictable). Static arms (I/ES/DF/MT): expect 0 runtime
rebalance rows after init; init injection evidence recorded. P0:
v2p1_rank0.jsonl presence GATED (arm P only); window/swap/transition facts
recorded. No gate may create pressure toward triggering (anti-tuning).

## 7. Analysis plan (pre-registered)
- Per-arm per-round req/s table (all 16+8 values disclosed, no silent drops).
- Mean +/- sd over 4 rounds; P0 vs each of 5 arms: round-paired t-test
  df=3 + Holm x5 (analyze_xd.py infrastructure clone).
- Phase split at idx 2048 boundary: prover-block vs code-block req/s per
  arm; P0 boundary-reaction windows; ED event timeline vs boundary.
- Static-arm cross-domain mismatch quantified from phase split (prover-frozen
  calibration vs code-block execution) - the failure mode the paper studies.
- Standing directives: h_rate never interpreted as remaining acceleration;
  no payback solver claims; paper analysis models not described as
  implemented online algorithms.

## 8. Deliverables
REPORT_pc.md; /data archive + manifest; GitHub plumbing-only push (guard:
origin/main == a1992e010a7ee9bb1717b99ea011c5aec85cd65f); EXPERIMENT_MAP:
new row + CORRECTION of the cross_domain_e2e_20261027 row framing (e3
campaign reframed as prefill-heavy supplementary probe, not paper 5.3.1
main comparison); GH README + REPORT_xd.md correction addendum; LEDGER
append-only throughout.
