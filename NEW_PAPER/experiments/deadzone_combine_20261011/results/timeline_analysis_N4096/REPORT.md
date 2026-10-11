# Representative timelines (N=4096)

The static figure is reconstructed from same-device CUDA-event intervals in the main 5×100 timing run. Nsight is collected in separate one-sample runs and is not used for numerical latency estimates.

| path | delay | rank-3 delay (µs) | rank-3 ready gate (µs) | median combine after gate, all ranks (µs) | max layer time (µs) |
|---|---:|---:|---:|---:|---:|
| original | zero | 25.088 | 2.976 | 189.792 | 1417.104 |
| original | inside | 35.776 | 3.008 | 189.880 | 1415.072 |
| original | outside | 109.760 | 2.976 | 248.616 | 1470.944 |
| strict | zero | 25.184 | 49.744 | 161.776 | 1437.376 |
| strict | inside | 35.808 | 41.584 | 161.736 | 1437.904 |
| strict | outside | 109.856 | 21.072 | 161.696 | 1492.944 |

In the strict path, the post-gate combine duration remains approximately constant while the ready-gate wait shifts across ranks. Thus removing all possible pre-all-ready combine progress does not remove the plateau.

Negative baseline-subtracted values elsewhere in the experiment remain unmodified; intervals spanning zero are compatible with a zero effect, and their sign is not identified.
