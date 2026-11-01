# CD-X0 cross-domain replay pilot - frozen analysis summary

- study: h20-placement-cdx0-v1
- overall verdict (EXPLORATORY): **detected_exploratory**
- layer results (EXPLORATORY): {"layer-024": "clear_crossover_advantage", "layer-071": "clear_crossover_advantage"}
- valid starts: 32/32 (retired: 0)
- epsilon = 22.948043518066406 us (labels only; NOT part of any verdict)

Verdict scope: the chosen captures, layers 024/071 and the five candidate
columns at replay caliber ONLY. No whole-service benefit or harm claim in
either direction. 'not_detected' is not evidence of absence of structure.

## layer-024: clear_crossover_advantage (exploratory)

```
row                   identity            sharegpt              prover                book                code
sharegpt        1919.72+-77.53      1926.44+-16.67      1904.16+-67.87      1916.13+-59.38      1879.88+-43.26
prover          2429.24+-44.20      1886.33+-21.87      1462.39+-26.92      2262.72+-28.74      1790.25+-45.70
book            2355.91+-59.21      2663.48+-15.35      2830.46+-34.96      2280.60+-69.66      2480.15+-106.57
code            2827.97+-23.94      3001.40+-53.06      2945.35+-58.28      2512.27+-32.02      2534.81+-30.27
```

rows tracking own base: {'sharegpt': False, 'prover': True, 'book': True, 'code': False}
indeterminate rows: []
oracle gap (row-equal K, descriptive): 6.332% (137.49 us/forward)
oracle gap (K-tilde weighted, descriptive, assumption-flagged): 4.191% (97.62 us/forward)

Primary contrasts (exploratory; + means own base faster):

- book: T(base-of-code) - T(base-of-book) = +199.55 us [+157.32, +241.79] excludes_zero_positive; holm_adj_p=0.0032 (descriptive)
- book: T(base-of-prover) - T(base-of-book) = +549.87 us [+475.88, +623.85] excludes_zero_positive; holm_adj_p=0.0017 (descriptive)
- book: T(base-of-sharegpt) - T(base-of-book) = +382.88 us [+313.65, +452.11] excludes_zero_positive; holm_adj_p=0.0024 (descriptive)
- code: T(base-of-book) - T(base-of-code) = -22.54 us [-82.42, +37.34] compatible_with_zero; holm_adj_p=0.8262 (descriptive)
- code: T(base-of-prover) - T(base-of-code) = +410.54 us [+343.64, +477.45] excludes_zero_positive; holm_adj_p=0.0023 (descriptive)
- code: T(base-of-sharegpt) - T(base-of-code) = +466.59 us [+388.40, +544.79] excludes_zero_positive; holm_adj_p=0.0023 (descriptive)
- prover: T(base-of-book) - T(base-of-prover) = +800.33 us [+748.29, +852.36] excludes_zero_positive; holm_adj_p=0.0002 (descriptive)
- prover: T(base-of-code) - T(base-of-prover) = +327.86 us [+283.74, +371.98] excludes_zero_positive; holm_adj_p=0.0017 (descriptive)
- prover: T(base-of-sharegpt) - T(base-of-prover) = +423.94 us [+391.44, +456.44] excludes_zero_positive; holm_adj_p=0.0003 (descriptive)
- sharegpt: T(base-of-book) - T(base-of-sharegpt) = -10.31 us [-63.94, +43.33] compatible_with_zero; holm_adj_p=0.8262 (descriptive)
- sharegpt: T(base-of-code) - T(base-of-sharegpt) = -46.56 us [-73.50, -19.62] excludes_zero_negative; holm_adj_p=0.0473 (descriptive)
- sharegpt: T(base-of-prover) - T(base-of-sharegpt) = -22.28 us [-75.57, +31.01] compatible_with_zero; holm_adj_p=0.8262 (descriptive)

## layer-071: clear_crossover_advantage (exploratory)

```
row                   identity            sharegpt              prover                book                code
sharegpt        1911.20+-31.34      1861.55+-37.19      1973.92+-37.14      2100.00+-40.31      2011.18+-55.99
prover          2175.83+-65.93      1728.65+-100.34      1476.85+-72.08      2019.65+-45.17      1937.97+-27.51
book            2480.12+-92.67      2733.22+-11.86      2459.04+-52.67      2280.45+-32.33      2824.78+-48.14
code            2829.93+-84.79      2622.05+-62.07      3024.30+-130.98      3332.81+-58.46      2372.67+-81.11
```

rows tracking own base: {'sharegpt': True, 'prover': True, 'book': True, 'code': True}
indeterminate rows: []
oracle gap (row-equal K, descriptive): 10.550% (235.65 us/forward)
oracle gap (K-tilde weighted, descriptive, assumption-flagged): 8.771% (207.94 us/forward)

Primary contrasts (exploratory; + means own base faster):

- book: T(base-of-code) - T(base-of-book) = +544.33 us [+489.35, +599.30] excludes_zero_positive; holm_adj_p=0.0008 (descriptive)
- book: T(base-of-prover) - T(base-of-book) = +178.59 us [+136.64, +220.53] excludes_zero_positive; holm_adj_p=0.0052 (descriptive)
- book: T(base-of-sharegpt) - T(base-of-book) = +452.77 us [+409.27, +496.27] excludes_zero_positive; holm_adj_p=0.0007 (descriptive)
- code: T(base-of-book) - T(base-of-code) = +960.15 us [+847.17, +1073.12] excludes_zero_positive; holm_adj_p=0.0010 (descriptive)
- code: T(base-of-prover) - T(base-of-code) = +651.63 us [+581.79, +721.48] excludes_zero_positive; holm_adj_p=0.0008 (descriptive)
- code: T(base-of-sharegpt) - T(base-of-code) = +249.39 us [+184.31, +314.47] excludes_zero_positive; holm_adj_p=0.0053 (descriptive)
- prover: T(base-of-book) - T(base-of-prover) = +542.79 us [+443.78, +641.81] excludes_zero_positive; holm_adj_p=0.0029 (descriptive)
- prover: T(base-of-code) - T(base-of-prover) = +461.12 us [+400.01, +522.23] excludes_zero_positive; holm_adj_p=0.0013 (descriptive)
- prover: T(base-of-sharegpt) - T(base-of-prover) = +251.80 us [+110.94, +392.65] excludes_zero_positive; holm_adj_p=0.0215 (descriptive)
- sharegpt: T(base-of-book) - T(base-of-sharegpt) = +238.45 us [+178.69, +298.21] excludes_zero_positive; holm_adj_p=0.0053 (descriptive)
- sharegpt: T(base-of-code) - T(base-of-sharegpt) = +149.63 us [+59.17, +240.09] excludes_zero_positive; holm_adj_p=0.0215 (descriptive)
- sharegpt: T(base-of-prover) - T(base-of-sharegpt) = +112.37 us [+69.54, +155.20] excludes_zero_positive; holm_adj_p=0.0108 (descriptive)

