# PBD_BP extraction: corrected book->prover stream (r4/r5) vs original prover->book (r1-r3)

NOTE: r4/r5 run on pbd_book_prover_double_out32_ie.jsonl (sha256 7b42568a..., author-ordered correction);
r1-r3 ran on the frozen prover->book stream (sha cdd1d864...). Cross-stream comparison is descriptive
(order-effect view), NOT a same-stream A/B. Missing boots marked n<expected.

Primary caliber = req_s (ok/total_time_s), identical to frozen T2 main-table caliber (agg tps tokens/s = req_s x 32 under O32 ignore_eos).

| arm | n(r4r5) | req_s_mean(r4r5) | req_s_vals | n(r1r3) | req_s_mean(r1r3) | delta% | lat_p99_mean(r4r5) | ttft_p95_mean(r4r5) |
|---|---|---|---|---|---|---|---|---|
| I | 1 | 6.094 | [6.094] | 3 | 5.875 | +3.7% | 65100.1 | 47868.1 |
| ES | 1 | 6.141 | [6.141] | 3 | 6.063 | +1.3% | 92325.5 | 46467.8 |
| ED | 0 | - | - | 3 | 6.112 | - | - | - |
| MT | 0 | - | - | 3 | 5.99 | - | - | - |
| DF | 1 | 6.239 | [6.239] | 3 | 5.939 | +5.1% | 64501.9 | 46601.7 |
| P0 | 1 | 6.46 | [6.46] | 3 | 6.13 | +5.4% | 69726.9 | 46490.8 |
