# PROTOCOL_pbd.md - prover_book_double_20261027 (pre-registration)

STATUS: DRAFT until PROTOCOL_FROZEN ledger row (written after stream sha fill, before first boot). Frozen = append-only corrections via LEDGER, never in-place edits of executed history.

## 1. Authorization and question
- Author directive 2026-10-27 (verbatim intent): 对 prover 和 book 数据集都扩充一倍看效果; 数据来源 modelscope: BookCorpus (书籍长文本) + DeepSeek-Prover-V1 (数学证明/形式化验证); 扩充之后跑两次 (六臂 x 2 轮 = 12 formal boots).
- Author capacity decision 2026-10-27 (AskUserQuestion): 纯V1 接受1.68x - prover 扩充只用剩余 DeepSeek-Prover-V1 语料, 不混入 Kimina; book 正常翻倍 1000->2000.
- Measured outcome under the byte-exact reproducing recipe: prover +1296 rows = 1.633x (lower than the 1.68x estimate because the verified greedy-pack recipe discards each straddling entry tail; the ENTIRE remaining V1 corpus is consumed, so 1296 is the corpus maximum). Disclosed, not silently adjusted.
- Research question (descriptive): do the pb campaign arm orderings (book: ED>P0>MT>I>ES>DF; prover: DF~MT>P0>ES>ED>I) and the P0 post-switch recovery/truncation dynamics replicate on a stream with 1.75x total length, a 1.63x prover block and a 2x book block?

## 2. Stream (B_data/streams/pbd_prover_book_double.jsonl)
- Structure: block1 prover[0:3344] = 2048 existing pb prover rows (raw-line byte-preserved) + 1296 new rows -> block2 book[3344:5344] = 1000 existing pb book rows (raw-line byte-preserved) + 1000 new rows. Single switch at idx 3344. n=5344.
- New prover rows: modelscope AI-ModelScope/DeepSeek-Prover-V1 dataset.jsonl, recipe v3 greedy-pack (per-entry chat-wrap, encode add_special_tokens=False, append until >=2048 tokens, emit 2048, discard straddler overflow) with skip-set {2064 thm_2079, 3061 thm_3080, 15894 thm_15937} (entries absent from the original builder source; diagnosed by global-absence test + non-injective-decode phantom-skip analysis, see LEDGER RECIPE_DIAGNOSIS). GATE: recipe reproduces ALL 2048 existing prover rows byte-exact (mismatches=0) before any new row is emitted; extension = entries[16782:27503), all included, structurally disjoint from existing.
- New book rows: modelscope BookCorpus books1/epubtxt (17868 txt), sorted-glob order, size>=24000B, lstrip + token-truncate to L_j = re-encoded token length of existing pb_book_j (index-paired length distribution; no RNG, deterministic). Disjointness: 200-char substring probes (raw offsets 0/8000/15000/25000 + lstrip head; raw and whitespace-normalized) vs concatenation of all 1000 existing book prompts; plus selected-head dedup. Disclosure: only 164/1000 existing book rows head-match books1, so disjointness is probe-enforced, not source-file-accounted.
- Row schema identical to pb: prover rows ignore_eos=false, book rows ignore_eos=true, ALL rows max_tokens=1, temperature=0. New ids pbd_prover_02048.. / pbd_book_01000..; new src fields carry full provenance.
- sha256: f864f2369858edcd6ba3d1b07fb57d67e26e72a54957754b7e1708fcf6964a9d (frozen at STREAM_FROZEN ledger row; verified at every boot by run_pbd_boot.sh, mismatch = rc2 halt).

## 3. Design
- 6 arms x 2 rounds = 12 formal boots, same rotation as pb: r1 ES ED MT DF P0 I | r2 ED MT DF P0 I ES. Labels <ARM>_r<k>.
- Workload: bench_client_v2.py (frozen sha 06668243...), NREQ=5344 CONC=32 BENCH_TIMEOUT=3600.
- Arms: ES/ED/MT/DF static paper-lineage placements (prover-frozen lineage, placement files byte-copied from pb); P = AERO-base online controller (params 0); I = pinned fair baseline. Identical launch_arm_pc.sh (byte-identical copy, sha cfbdaee7... verified).

## 4. Guard chain (cloned from pb, deltas: campaign paths, stream sha, NREQ 3048->5344, BENCH_TIMEOUT 1800->3600, purpose pbd_double)
- Per boot (run_pbd_boot.sh): stream sha -> pb_oeplb code locks (async_swapper 2e4652a0..., controller 4964cfc8..., sp_state 525e88f0..., window_v2 810c54da...) -> launcher sha cfbdaee7... + bench client sha 06668243... -> placement locks from code_lock/placement_shas.env (PL_ES=PL_DF, PL_MT; copied unmodified from pb) -> boot_meta.json -> ready protocol (225 x 4s, ready to roll) -> bench -> gpu-clear -> mech_check_pbd.py (facts-only, completeness-gated rc).
- rc=2 = guard abort halts queue (GUARD_ABORT ledger row); rc=1 = log+continue.

## 5. Queue behavior
- INFRA budget: <=2 zero-data restarts per arm (READY_FAIL / bench transport failure with no perreq data); every restart gets a LEDGER row; arms are NEVER silently dropped; mechanism errors are reported honestly, not retried into success.
- Queue completion writes QUEUE_PBD_COMPLETE (script-emitted ledger row, as in pb).

## 6. Pre-registered analysis (descriptive only; written before looking at results)
- Completeness: 12/12 boots with ok+errors=5344 and perreq lines=5344; MECH_CHECK completeness_pass.
- Endpoints per arm x round: tps, lat_mean_ms, ttft_mean_ms (bench json); phase-split via perreq order at switch idx 3344: prover phase [0,3344), book phase [3344,5344), book early/late halves (b_early/b_late as in pb).
- P0 dynamics: v2p1 z trajectory, swap counts, gb events, boundary windows vs pb r1 (27 swaps, b_late 3.97) and pb r2 (6 swaps, b_late 4.55). With ~1.75x stream length P0 observes proportionally more post-switch windows: tests whether the pc-era 75pct-truncated recovery and the pb r1/r2 swap-count spread were window-budget artifacts.
- Static vs online: per-arm phase means replicate the pb layered attribution (static gain shortfall = calibration non-transfer; P0 deficit = overhead side; O=1 compression of benefits); ED batched rebalance line counts recorded as mechanism evidence (expected ~21 lines x 8 per rebalance, facts only).
- Standing interpretation rules: h_rate is NOT 还可以加速多少 (no headroom reading); no payback solver exists; offline/paper analysis models are not written as implemented online algorithms; mechanism facts never gate rc (anti-tuning clause inherited).
- Cross-campaign comparison pbd vs pb uses per-phase/per-request metrics (stream lengths differ); no significance claims from 2 rounds - descriptive tables only.

## 7. Freeze rules
- This PROTOCOL + stream + code/placement locks freeze at first boot; post-hoc changes only as appended LEDGER CORRECTION rows + addendum sections.
- Analysis scripts (descriptive_pbd.py, cloned from descriptive_pb.py with new switch idx) are written from this section before results inspection.

## 8. Artifact index
- boots/<LABEL>/: server.log, bench.log, <LABEL>.json, <LABEL>_perreq.jsonl, boot_meta.json[.end], MECH_CHECK.json, v2p1_rank0.jsonl (P arm), sp_run_end.json.
- queue.log, LEDGER_pbd.tsv, B_data/registry.json, scripts/{run_pbd_boot.sh, queue_formal_pbd.sh, mech_check_pbd.py, launch_arm_pc.sh}, code_lock/placement_shas.env, placements/.
