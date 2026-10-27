# PROTOCOL - prover_book_switch_20261027 (pre-registered, frozen before first boot)

## 0. Origin and authorization
- Parent campaign prover_code_switch_20261027 halted by author at 11/24 DONE
  (12:15Z): joint finding - on the prover->code O=1 stream the prover-frozen
  static arms showed NO code-block penalty (mismatch cost invisible at O=1),
  while P0 (new design) paid swap overhead in stage2 without recouping gains
  (code block 28.8-30.6 req/s, lowest of all arms; boundary detected at w29
  z=23.5 but adaptation bandwidth-limited: grow_blocked_headroom x10,
  1 swap/window).
- Author new spec 2026-10-27 (binding): stage2 = book requests instead of
  code; ONLY 2 reps; after the run author + agent analyze the problem
  together. Author asked 1024 book rows; the published book_4438tok_O10 set
  contains exactly 1000 rows (README table) - FULL published set used,
  disclosed here and to author, not silently adjusted, no external splicing.
- GPU authorization: author message ("我们现在把第二阶段换成1024个book请求
  看一下效果怎么样。而且我们只跑两次") = launch authorization for <=12
  formal boots + <=2 zero-data INFRA restarts.

## 1. Stream (frozen)
- B_data/streams/pb_prover_book_switch.jsonl
  sha256 = 2da3467263d4f52889ea2931fa84d9fb244ed3dfabe136fddb5c1b6feb7bf904
  n = 3048, uniq text hashes = 3048.
- Block1 idx 0..2047    = prover_2048tok_out1.jsonl all rows (in ~2048 tok),
  byte-source identical to pc campaign block1.
- Block2 idx 2048..3047 = book_4438tok_O10.jsonl all 1000 rows (BookCorpus,
  in ~4438 tok, paper freq6 book family; per-row max_tokens overridden
  10 -> 1 per author standing O=1 spec).
- ALL rows: max_tokens = 1, temperature = 0. Registry: B_data/registry.json.

## 2. Arms (6) - identical to pc campaign (launcher byte-identical, sha cfbdaee7…)
I bare auto | ES normal+R16+datafore_prover_placement.json | DF auto+same
file | MT auto+placement_pinned_fair.json | ED normal+R16+enable-eplb+auto+
period 100+recorder buffer 32 | P0 frozen AERO-base rev1 RG=0 v2.1 knobs.
Paper-lineage deltas D1-D7 of PROTOCOL_pc.md sec.2 carry over verbatim
(stream now prover->book two-block per author; stage2 domain = book makes
the mismatch axis comparable to paper freq6 book<->prover pair).

## 3. Workload
N = 3048 per boot, conc = 32, BENCH_TIMEOUT = 1800s. Primary metric req/s.
Phase split at idx 2048 (prover block vs book block); book block prefill is
~2.2x heavier per request than prover (4438 vs 2048 tok).

## 4. Guard chain
Identical structure to pc campaign (run_pb_boot.sh): stream sha -> code
locks x4 -> launcher sha cfbdaee7… + bench client 06668243… -> placement
locks (PL_ES=PL_DF=c1787ad1…, PL_MT=299cea28…) -> boot_meta -> ready
protocol -> bench -> gpu-clear -> mech_check_pb.py (DS_SHA = this stream).
rc=2 halts queue; rc=1 logged + continue.

## 5. Rotation and budget
r1: ES ED MT DF P0 I | r2: ED MT DF P0 I ES   (12 boots, 2 reps per author)
Per boot ~12 min est (launch ~1.5-2min + bench ~7-8min est: 8.6M prefill
tokens total + teardown). Total est ~2.4h. Detached queue + cron patrol.

## 6. Mechanism check - FACTS ONLY (unchanged clauses from pc sec.6)
ED events recorded not gated; static arms 0 runtime rebalance expected; P0
v2p1 presence gated for arm P only; anti-tuning clause in force.

## 7. Analysis plan (pre-registered; JOINT with author per his message)
- Per-arm per-round req/s (both rounds disclosed); mean of 2 rounds.
- Phase split idx2048: prover-block vs book-block req/s per arm; P0 boundary
  reaction (issued windows vs boundary wall time, z trajectory, swap count
  and timing); early/late book-block split (first 512 vs rest) to test the
  pc-campaign finding (P0 stage2 dip then partial recovery).
- P0-vs-5 paired stats NOT run at n=2 (df=1 too weak; author opted 2 reps
  for speed) - descriptive deltas only, explicitly labeled non-inferential.
- Comparison anchor: pc campaign partial (r1 full + r2 partial) same arms
  same stage1 - stage2 domain effect (code vs book) isolated descriptively.
- Standing directives: h_rate never interpreted as remaining acceleration;
  no payback solver claims; paper analysis models not described as
  implemented online algorithms.

## 8. Deliverables
Joint analysis session with author (his request) BEFORE report
finalization; then REPORT_pb.md, /data archive + manifest, GitHub
plumbing-only push (guard origin/main == a1992e010a7ee9bb1717b99ea011c5aec85cd65f),
EXPERIMENT_MAP rows (pc partial + pb) + correction duties for
cross_domain_e2e_20261027 framing (carried over from pc PROTOCOL sec.8),
LEDGER append-only throughout.
