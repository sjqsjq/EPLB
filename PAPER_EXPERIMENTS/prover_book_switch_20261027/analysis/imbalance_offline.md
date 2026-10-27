# Offline imbalance comparison (author directive 2026-10-27)

Metric definitions: headroom_rate = P0 controller h_rate formula (physical space, 8 ranks x 16 slots); r_raw = mean per-layer max/mean rank load; cv = mean per-layer std/mean.

## 94-layer grid: SAME counts x layouts

### counts = code_e3rec_392_512
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.06474 | 1.5179 | 0.3074 |
| prover_frozen_DF_repro | 0.06373 | 1.5099 | 0.2944 |
| e3_fair_ILP | 0.05300 | 1.4240 | 0.2539 |
| MT_pinned_fair | 0.06662 | 1.5330 | 0.3153 |
| code_adapted_greedy_offline | 0.00010 | 1.0008 | 0.0005 |
| book_adapted_greedy_offline | 0.06143 | 1.4914 | 0.2905 |

### counts = code_e3rec_404_512
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.06405 | 1.5124 | 0.3050 |
| prover_frozen_DF_repro | 0.06320 | 1.5056 | 0.2920 |
| e3_fair_ILP | 0.05245 | 1.4196 | 0.2517 |
| MT_pinned_fair | 0.06606 | 1.5285 | 0.3132 |
| code_adapted_greedy_offline | 0.00093 | 1.0074 | 0.0049 |
| book_adapted_greedy_offline | 0.06094 | 1.4875 | 0.2884 |

### counts = book_e3rec_264_384
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.07566 | 1.6053 | 0.3536 |
| prover_frozen_DF_repro | 0.07372 | 1.5898 | 0.3335 |
| e3_fair_ILP | 0.06769 | 1.5415 | 0.3166 |
| MT_pinned_fair | 0.07324 | 1.5859 | 0.3312 |
| code_adapted_greedy_offline | 0.07007 | 1.5606 | 0.3331 |
| book_adapted_greedy_offline | 0.00003 | 1.0002 | 0.0002 |

### counts = prover_e3rec_136_256
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.09031 | 1.7225 | 0.3953 |
| prover_frozen_DF_repro | 0.01298 | 1.1038 | 0.0650 |
| e3_fair_ILP | 0.06787 | 1.5429 | 0.3199 |
| MT_pinned_fair | 0.01307 | 1.1046 | 0.0676 |
| code_adapted_greedy_offline | 0.08135 | 1.6508 | 0.3655 |
| book_adapted_greedy_offline | 0.08533 | 1.6827 | 0.3876 |

### counts = prover_datafore_full
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.09001 | 1.7201 | 0.3960 |
| prover_frozen_DF_repro | 0.00004 | 1.0003 | 0.0002 |
| e3_fair_ILP | 0.06507 | 1.5206 | 0.3094 |
| MT_pinned_fair | 0.00177 | 1.0141 | 0.0093 |
| code_adapted_greedy_offline | 0.08013 | 1.6411 | 0.3585 |
| book_adapted_greedy_offline | 0.08501 | 1.6801 | 0.3823 |

### counts = book_published_traces (exact, pb stage2 source, prefill forwards)
| layout | headroom_rate | r_raw | cv |
|---|---|---|---|
| identity | 0.09110 | 1.7288 | 0.4174 |
| prover_frozen_DF_repro | 0.08608 | 1.6886 | 0.3912 |
| e3_fair_ILP | 0.08228 | 1.6582 | 0.3840 |
| MT_pinned_fair | 0.09069 | 1.7255 | 0.4069 |
| code_adapted_greedy_offline | 0.08619 | 1.6895 | 0.4010 |
| book_adapted_greedy_offline | 0.01845 | 1.1476 | 0.1086 |
| book_adapted_greedy_on_pub | 0.00002 | 1.0002 | 0.0001 |

## cdx0 exact layers (eval-code counts, 2026-10-16 captures)

### layer_024
| column | headroom_rate | r_raw (ours) | r_raw (archived) | cv |
|---|---|---|---|---|
| identity | 0.07338 | 1.5870 | 1.5870 | 0.3007 |
| base-of-sharegpt | 0.08730 | 1.6984 | 1.6984 | 0.3322 |
| base-of-prover | 0.08836 | 1.7069 | 1.7069 | 0.4326 |
| base-of-book | 0.04259 | 1.3407 | 1.3407 | 0.2034 |
| base-of-code | 0.04221 | 1.3376 | 1.3376 | 0.1605 |

### layer_071
| column | headroom_rate | r_raw (ours) | r_raw (archived) | cv |
|---|---|---|---|---|
| identity | 0.07777 | 1.6222 | 1.6222 | 0.3631 |
| base-of-sharegpt | 0.04514 | 1.3611 | 1.3611 | 0.3066 |
| base-of-prover | 0.09303 | 1.7443 | 1.7443 | 0.4324 |
| base-of-book | 0.12366 | 1.9893 | 1.9893 | 0.4250 |
| base-of-code | 0.02412 | 1.1930 | 1.1930 | 0.0921 |

## P0 live h_rate (controller own metric on live traffic)
- pc P0_r1: prover plateau(w2-29)=0.01032 boundary(w30)=0.07530 z30=15.5 w31=0.02391 code plateau(w33-39)=0.01738 z plateau=3.51
- pc P0_r2: prover plateau(w2-29)=0.01032 boundary(w30)=0.07424 z30=12.6 w31=0.02404 code plateau(w33-39)=0.01698 z plateau=3.55
- pc P0_r3: prover plateau(w2-29)=0.01058 boundary(w30)=-1.00000 z30=-1.0 w31=-1.00000 code plateau(w33-39)=-1.00000 z plateau=-1.00
- pb P0_r1: 85 windows so far (trajectory in json)
- pb P0_r2: 56 windows so far (trajectory in json)

## Cross-checks
- prover slice vs datafore prover counts per-layer cosine: mean=0.9892 min=0.9786 max=0.9948
- cdx0 archived r_raw vs recomputed: see table above (identity column anchors pipeline)
