# Cross-domain placement-mismatch screening (offline, read-only, 2026-10-30T06:53:08Z)

Caliber: same logical-expert basis as proven imbalance_offline.py. hr = P0 controller h_rate formula (8 ranks x 16 slots); r_raw = mean per-layer max/mean rank load; layouts = deepseek flat rebalance. B-dedicated reference layout is DIAGNOSTIC REFERENCE ONLY (built from full B counts incl all eval segments), NOT optimal; no count gap is converted to expected speedup. Segments = 4 contiguous slices of B counts (e3rec: equal step chunks; book_pub: file-order chunks).

## Provenance (input len / output rule / stage / sample; MISSING marked)
| count source | domain | input_len | output_rule | stage | sample |
|---|---|---|---|---|---|
| e3rec_sharegpt | sharegpt (mixed zh/en chat) | 512 tok (uniform retok) | max_tokens=1; temperature not separately recorded = MISSING | all-forward logical counts, identity basis, O=1 prefill-dominated; step=8192 hidden rows | 2048 reqs; slice steps[4:128] (124 steps); count_sum=763887616 |
| e3rec_prover | prover (DeepSeek-Prover-V1 retok) | 512 tok (uniform retok) | max_tokens=1; temperature MISSING | same recording | 2048 reqs; slice steps[136:256] (120 steps); count_sum=739246080 |
| e3rec_book | book (BookCorpus retok) | 512 tok (uniform retok) | max_tokens=1; temperature MISSING | same recording | 2048 reqs; slice steps[264:384] (120 steps); count_sum=739246080 |
| e3rec_code | code (retok L512 5-source splice, same population as pc stage2) | 512 tok (uniform retok) | max_tokens=1; temperature MISSING | same recording | 2048 reqs; slice steps[392:512] (120 steps); count_sum=739246080 |
| book_pub | book published traces (exact pb stage2 source population) | ~4438 tok | max_tokens=10 (O10); temperature MISSING | prefill forwards only (is_prefill filter) | 32 npz, 6400 forwards (3876 prefill), 1000 published rows; count_sum=2921972704 |
| datafore_DF_profile | prover (DF/ES deployed arm profile, paper 5.3.1 lineage) | MISSING (all source copies unreadable at run time) | MISSING | MISSING | MISSING |

## Primary matrix: 12 A->B pairs on e3rec counts (uniform in512/out1 caliber)
| A->B | hr_I | hr_Aprof | hr_Bref | gap(A-Bref) | gap(A-I) | r_raw I/A/Bref | layers worse/94 | seg gaps g1..g4 | persistent |
|---|---|---|---|---|---|---|---|---|---|
| sharegpt->prover | 0.09031 | 0.07948 | 0.00004 | +0.07944 | -0.01083 | 1.722/1.636/1.000 | 94 | 0.0787 0.0778 0.0779 0.0779 | YES |
| sharegpt->book | 0.07566 | 0.06770 | 0.00003 | +0.06767 | -0.00797 | 1.605/1.542/1.000 | 94 | 0.0649 0.0658 0.0645 0.0668 | YES |
| sharegpt->code | 0.06474 | 0.06096 | 0.00010 | +0.06087 | -0.00377 | 1.518/1.488/1.001 | 94 | 0.0606 0.0567 0.0471 0.0569 | YES |
| prover->sharegpt | 0.03081 | 0.03322 | 0.00028 | +0.03294 | +0.00241 | 1.246/1.266/1.002 | 94 | 0.0315 0.0286 0.0305 0.0290 | YES |
| prover->book | 0.07566 | 0.07593 | 0.00003 | +0.07591 | +0.00027 | 1.605/1.607/1.000 | 94 | 0.0730 0.0745 0.0730 0.0745 | YES |
| prover->code | 0.06474 | 0.07209 | 0.00010 | +0.07199 | +0.00735 | 1.518/1.577/1.001 | 94 | 0.0716 0.0673 0.0585 0.0679 | YES |
| book->sharegpt | 0.03081 | 0.02746 | 0.00028 | +0.02718 | -0.00335 | 1.246/1.220/1.002 | 94 | 0.0249 0.0232 0.0243 0.0244 | YES |
| book->prover | 0.09031 | 0.08533 | 0.00004 | +0.08529 | -0.00498 | 1.722/1.683/1.000 | 94 | 0.0844 0.0835 0.0841 0.0837 | YES |
| book->code | 0.06474 | 0.06143 | 0.00010 | +0.06133 | -0.00331 | 1.518/1.491/1.001 | 94 | 0.0609 0.0569 0.0486 0.0573 | YES |
| code->sharegpt | 0.03081 | 0.03099 | 0.00028 | +0.03071 | +0.00018 | 1.246/1.248/1.002 | 94 | 0.0291 0.0266 0.0284 0.0267 | YES |
| code->prover | 0.09031 | 0.08135 | 0.00004 | +0.08130 | -0.00896 | 1.722/1.651/1.000 | 94 | 0.0805 0.0795 0.0798 0.0801 | YES |
| code->book | 0.07566 | 0.07007 | 0.00003 | +0.07004 | -0.00560 | 1.605/1.561/1.000 | 94 | 0.0674 0.0682 0.0672 0.0687 | YES |

## Supplementary: B = book_pub (pb native caliber ~4438tok, prefill-only, O10)
| A->B | hr_I | hr_Aprof | hr_Bref | gap(A-Bref) | gap(A-I) | r_raw I/A/Bref | layers worse/94 | seg gaps g1..g4 | persistent |
|---|---|---|---|---|---|---|---|---|---|
| sharegpt->book_pub | 0.09110 | 0.08355 | 0.00002 | +0.08353 | -0.00754 | 1.729/1.668/1.000 | 94 | 0.0815 0.0806 0.0816 0.0814 | YES |
| prover->book_pub | 0.09110 | 0.09122 | 0.00002 | +0.09119 | +0.00012 | 1.729/1.730/1.000 | 94 | 0.0890 0.0884 0.0894 0.0889 | YES |
| book->book_pub | 0.09110 | 0.01845 | 0.00002 | +0.01843 | -0.07265 | 1.729/1.148/1.000 | 94 | 0.0163 0.0153 0.0172 0.0166 | YES |
| code->book_pub | 0.09110 | 0.08619 | 0.00002 | +0.08617 | -0.00491 | 1.729/1.690/1.000 | 94 | 0.0842 0.0833 0.0845 0.0837 | YES |

## DF deployed layout rows: MISSING this run (datafore_prover_counts.json unreadable at run time on all known copies; transient storage failures). Full-count DF numbers remain available in the archived imbalance_offline.json grid (prover_frozen_DF_repro rows); segment-level DF evaluation deferred until the source is readable again.

