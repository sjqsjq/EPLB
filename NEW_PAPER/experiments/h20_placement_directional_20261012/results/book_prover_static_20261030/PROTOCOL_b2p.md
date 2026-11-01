# PROTOCOL_b2p.md - book_prover_static_20261030 (FROZEN v2, author-checked 2026-10-30)

## 1. Question and framing
Two static layouts, same frozen prover workload: does the locked prover profile
(REF, historical ~19pct stage-gain static evidence) beat a book-derived profile
(OLD, cross-domain diagnostic) in full-service time? REF is NOT claimed optimal;
OLD is NOT claimed worst-case. Measured effects reported as-is, both directions
preserved. n=2 per cell: NON-INFERENTIAL descriptive.

## 2. Arms (identical injection path, sole delta = profile file)
- OLD (B2P_OLD): BASE + --init-expert-location placements/book_profile_placement.json
- REF (B2P_REF): BASE + --init-expert-location placements/datafore_prover_placement.json
- BASE = frozen launch_arm_pc.sh BASE (tp8 dp8 ep8 dp-attention deepep auto
  deep_gemm bf16 fp8 mem0.8 no-cuda-graph no-radix watchdog600 port30000).
- Loader path: {"logical_count": ...} -> init_by_eplb -> rebalance_experts
  (greedy least-loaded, zero redundancy). NO --enable-pb-oeplb, NO --enable-eplb:
  online rebalance OFF during timing.

## 3. Stream
b2p_prover_first2048.jsonl = byte-exact rows 1-2048 of frozen
pb_prover_book_switch.jsonl (src sha 2da34672..., PBS registry: block1
prover[0:2048], prover_2048tok_out1 lineage). sha f8e948f3..., 2048 uniq ids
pb_prover_00000..02047, max_tokens=1 temperature=0 (C32/O1, conc=32).

## 4. Matrix and bench
4 boots: OLD_b1 -> REF_b1 -> REF_b2 -> OLD_b2 (block1 OLD->REF, block2 REF->OLD,
arm order alternates per block; frozen pre-data). bench_client_v2.py (06668243,
final_plan scripts), NREQ=2048 CONC=32 BENCH_TIMEOUT=1800. PBS reference:
full 3048-row DF boot ~9min -> expected ~9-10min/boot, campaign <=1.5h GPU.

## 5. Mapping evidence naming (author fix 2, 2026-10-30)
File sha + offline deterministic replay + server.log init line = EXPECTED-MAPPING
EVIDENCE, never called actual p2l fingerprint. Runtime mapping export search:
0 endpoint hits in sglang/srt/entrypoints -> direct runtime p2l cross-check
NOT AVAILABLE, recorded as-is. server.log init line = runtime loader-path evidence.
Offline replay: scripts/check_p2l_expected.py via frozen eplb_algorithms entry,
outputs archived analysis/expected_mapping_{OLD,REF}.json.

## 6. Guards and halt conditions (rc=2 GUARD_ABORT stops queue)
stream sha / pb_oeplb 4-file code sha / loader-path 3-file sha / launcher+bench
tool sha / placement lock (PL_OLD PL_REF, written once) - any mismatch -> abort,
report specific problem, NO silent replacement, NO arm dropped, NO scope widened.
INFRA: <=2 zero-data restarts per arm, logged, never silently dropped.

## 7. Endpoints
Primary: total_time_s (full service) + actual completed throughput per boot.
Secondary: first-data latency, completion latency. Quartile buckets ONLY with
exact definitions. All 4 boots reported regardless of direction.

## 8. Preservation
LEDGER_b2p.tsv append-only (corrections appended). All costs and per-run results
kept. No mid-run tuning, no added repeats, no sampling by outcome. h_rate/h_w
never interpreted as headroom. Paper-analysis models never written as
implemented online algorithms.
