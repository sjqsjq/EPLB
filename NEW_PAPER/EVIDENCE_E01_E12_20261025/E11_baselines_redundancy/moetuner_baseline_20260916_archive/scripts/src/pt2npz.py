#!/usr/bin/env python3
"""Convert SGLang expert_distribution_recorder_*.pt to a P[L,E] npz for MoETuner ILP.

SGLang's `stat` recorder dumps one .pt per rank per /dump call.  Each contains:
  { "rank": int, "logical_count": tensor [dim_extra, L, E], "average_...": ... }
All ranks contain the SAME global counts after all_reduce, so we can use rank0
only (or sum any subset — they cancel).
"""
import argparse, glob, os, sys
import torch, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob", required=True, help="e.g. dump_dir/expert_distribution_recorder_*_0.pt or *.pt")
    ap.add_argument("--out", required=True, help="path to output .npz")
    args = ap.parse_args()

    files = sorted(glob.glob(args.src_glob))
    if not files:
        print(f"no files match {args.src_glob}", file=sys.stderr); sys.exit(2)

    # Prefer files ending in _0.pt (rank 0) if the recorder dumped per-rank.
    rank0 = [f for f in files if f.endswith("_0.pt")]
    src = rank0 if rank0 else files
    print(f"found {len(files)} files; using {len(src)}")

    P_accum = None
    total_steps = 0
    for f in src:
        d = torch.load(f, weights_only=False, map_location="cpu")
        lc = d["logical_count"]
        # shape [dim_extra, L, E]
        if not isinstance(lc, torch.Tensor):
            lc = torch.as_tensor(lc)
        P = lc.sum(dim=0).to(torch.int64).numpy()  # [L, E]
        steps = int(lc.shape[0])
        total_steps += steps
        P_accum = P if P_accum is None else P_accum + P
        print(f"  {os.path.basename(f)}: rank={d.get('rank')} shape={tuple(lc.shape)} steps={steps} sum={int(lc.sum())}")

    print(f"[out] P shape={P_accum.shape} total_steps={total_steps} sum={int(P_accum.sum())}")
    print(f"[out] mean(P.sum(axis=1)) per layer = {P_accum.sum(axis=1).mean():.1f}")
    print(f"[out] layer 0 imb (max/mean) = {P_accum[0].max()/max(P_accum[0].mean(),1):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, P=P_accum, num_forwards=total_steps)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
