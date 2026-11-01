Smoke C run 1 (2026-10-16 11:16-11:20) - PRESERVED FAILURE RECORD.
Result: all 8 ranks reported "RANK DESYNC" at window 16 and froze W at
w_init=16. Root cause: hash-consistency check compared the all-reduced hash
sum against hardcoded `16 * h` (the 16-window check interval was confused
with the rank count); world size is 8, so a perfectly consistent 8*h sum
always failed the check. NOT a real rank divergence. Safe-mode behavior
itself verified correct: deterministic freeze at w_init, no hang, service
completed 4000/4000, no legacy mutation lines, no v2 exceptions.
Fix: controller.py hash check now uses torch.distributed.get_world_size();
desync log includes own_h/sum/world for diagnosis. Fix hash: 2a4dedb2...
