# Five-arm campaign summary (A00/A10/A01/A11/P11)

analysis sha256: 6d8ddf15eaca9263a5db60e782206aa36ffaa30f99e0005172c11515bdc1d3a2

| arm | n_valid | mean total_time_s | sd | min | max |
|---|---|---|---|---|---|
| A00 | 5/5 | 367.81 | 7.38 | 362.49 | 380.82 |
| A10 | 5/5 | 372.60 | 7.98 | 364.62 | 384.07 |
| A01 | 5/5 | 374.26 | 7.22 | 368.00 | 385.54 |
| A11 | 5/5 | 378.21 | 5.99 | 372.66 | 387.43 |
| P11 | 5/5 | 378.65 | 6.89 | 373.84 | 389.26 |

## Contrasts (paired by round; negative = first arm faster)

| contrast | n | mean diff (s) | 95% CI | excludes zero |
|---|---|---|---|---|
| primary_A11_minus_A00 | 5 | 10.40 | [5.00, 15.80] | True |
| primary_A11_minus_P11 | 5 | -0.44 | [-7.82, 6.95] | False |
| secondary_A11_minus_A10 | 5 | 5.61 | [-2.39, 13.61] | False |
| secondary_A11_minus_A01 | 5 | 3.95 | [-0.94, 8.84] | False |
| interaction_WxD | 5 | -0.84 | [-11.82, 10.13] | False |

Migration wall (alongside, never subtracted): {"A00": [2288.3, 1696.7, 1555.8, 1674.3, 2676.2], "A10": [1791.8, 2003.8, 3402.8, 1794.8, 2101.5], "A01": [2197.2, 1849.1, 3626.1, 1373.0, 1340.8], "A11": [2555.3, 2380.3, 1419.4, 1400.1, 2750.1], "P11": [1706.5, 3911.2, 2936.8, 1822.9, 2194.0]}

Missing starts: none
