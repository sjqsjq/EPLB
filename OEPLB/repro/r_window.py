"""Three r definitions from one recording, so the offline predictor can match
what the runtime balancer actually sees.

  r_agg    : ratio of counts aggregated over the whole run  (what r_avg.py did)
  r_fwd    : mean over single forward passes of the per-forward ratio
  r_win(w) : mean over w-forward windows of the per-window ratio  <-- DIAG's
             avg_ratio_before uses w = pb-oeplb-sync-window (16 here)

Jensen: r_agg <= r_win(w) <= r_fwd.  They coincide only for homogeneous loads;
for skewed prompt-length mixes the gap is large and r_agg badly underestimates
the r that actually costs time.
"""
import torch, sys

path, ep = sys.argv[1], int(sys.argv[2] if len(sys.argv) > 2 else 8)
W = [int(x) for x in (sys.argv[3:] or [16])]
lc = torch.load(path, map_location="cpu", weights_only=False)["logical_count"].double()
F, L, E = lc.shape
live = lc.sum(dim=(1, 2)) > 0
lc = lc[live]
per = E // ep
g = lc.reshape(lc.shape[0], L, ep, per).sum(-1)          # [F, L, ep]


def ratio(x):                                             # x: [..., L, ep]
    tot = x.sum(-1, keepdim=True)
    r = x.max(-1).values / (tot.squeeze(-1) / ep)
    return torch.where(tot.squeeze(-1) > 0, r, torch.ones_like(r))


print(f"{path.split('/')[-1]}  ep={ep}  live_forwards={lc.shape[0]}/{F}  "
      f"tokens={lc.sum().item():,.0f}")
print(f"  r_agg              = {ratio(g.sum(0)).mean().item():.4f}")
wt = g.sum(dim=(1, 2))                                    # tokens per forward
rf = ratio(g).mean(-1)                                    # [F] per-forward, layer-mean
print(f"  r_fwd (token-wt)   = {(rf*wt).sum().item()/wt.sum().item():.4f}")
for w in W:
    n = (lc.shape[0] // w) * w
    if n == 0: continue
    gw = g[:n].reshape(-1, w, L, ep).sum(1)               # [n/w, L, ep]
    ww = gw.sum(dim=(1, 2))
    rw = ratio(gw).mean(-1)
    print(f"  r_win(w={w:<3d})       = {(rw*ww).sum().item()/ww.sum().item():.4f}"
          f"   (n={gw.shape[0]} windows)")
