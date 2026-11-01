#!/usr/bin/env python3
"""Compute per-layer prefill->decode Spearman rho from routing trace npz dirs.
Usage: compute_pd_rho.py <trace_dir> [tag]
"""
import sys, glob, os
import numpy as np
from scipy.stats import spearmanr

def compute(trace_dir):
    prefill = None  # [94, 128]
    decode = None
    n_pf = n_dc = 0
    files = sorted(glob.glob(os.path.join(trace_dir, "rank*_fwd_chunk*.npz")))
    if not files:
        return None
    for f in files:
        z = np.load(f)
        ip = z['is_prefill']
        lh = z['layer_hists']  # [N, 94, 128]
        pf = lh[ip].sum(axis=0)   # [94,128]
        dc = lh[~ip].sum(axis=0)
        prefill = pf if prefill is None else prefill + pf
        decode = dc if decode is None else decode + dc
        n_pf += int(ip.sum()); n_dc += int((~ip).sum())
    rhos = []
    ge07 = 0
    for L in range(prefill.shape[0]):
        p = prefill[L]; d = decode[L]
        if p.sum()==0 or d.sum()==0:
            rhos.append(np.nan); continue
        r, _ = spearmanr(p, d)
        rhos.append(r)
        if r >= 0.7: ge07 += 1
    rhos = np.array(rhos)
    valid = rhos[~np.isnan(rhos)]
    return {
        'n_prefill_forwards': n_pf, 'n_decode_forwards': n_dc,
        'rho_mean': float(np.round(valid.mean(),3)),
        'rho_min': float(np.round(valid.min(),3)),
        'rho_max': float(np.round(valid.max(),3)),
        'layers_ge07': f"{ge07}/{prefill.shape[0]}",
        'layers_valid': int(len(valid)),
    }

if __name__ == '__main__':
    d = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv)>2 else os.path.basename(d)
    r = compute(d)
    if r is None:
        print(f"{tag}: NO TRACE FILES in {d}"); sys.exit(1)
    print(f"{tag}: rho={r['rho_mean']} (min {r['rho_min']} max {r['rho_max']}) ge0.7={r['layers_ge07']} pf_fwd={r['n_prefill_forwards']} dc_fwd={r['n_decode_forwards']}")
