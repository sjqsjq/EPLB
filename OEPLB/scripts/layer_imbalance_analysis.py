"""Per-layer, per-forward-step imbalance analysis for dispatch/combine/expert.

Usage: python3 layer_imbalance_analysis.py <label> <snapshot_dir> [<snapshot_dir> ...]
Pass one or more profiler snapshot dirs (each containing 4 rank .trace.json.gz
files from /start_profile with num_steps=N, with_stack=False) to pool them.

Why this exists: aggregating a whole run's kernel time into one ratio (e.g.
"1.011 overall imbalance") hides the real picture -- per-(layer,step) the
imbalance is much higher (measured ~1.1-1.6 mean, up to ~3x peak) because hot
ranks average out over time even when no swap ever ran. This script recovers
per-(layer,step) attribution WITHOUT relying on any profiler hook
(--enable-layerwise-nvtx-marker uses torch.cuda.nvtx ranges, which do NOT show
up as "nn.Module: X" trace events; with_modules is also not set by SGLang's
start_profile). Instead it exploits the fact that kernels of the same
(category, exact name) fire in strict cyclic layer order, once per layer per
forward pass -- verified: count % 48 == 0 for every kernel variant seen.
"""
import gzip, json, glob, sys
from collections import defaultdict, Counter

def classify(name):
    nl = name.lower()
    if 'deep_ep' in nl and 'dispatch' in nl: return 'dispatch'
    if 'deep_ep' in nl and 'combine' in nl: return 'combine'
    if 'deep_gemm' in nl and ('768' in name or '1536' in name): return 'expert'
    return None

def load_rank_layer_step(fpath, period=48):
    """Hook-free layer attribution: kernels of the same (category, exact_name)
    fire in strict layer-cyclic order once per layer per forward pass (verified:
    count % 48 == 0 for every variant seen). Chunk chronologically by `period`
    per (category,name) group -> position_in_chunk = layer_id, chunk_index = step.
    Sum across name-variants (e.g. deep_gemm's 768-shape + 1536-shape kernels)
    into one per-(step,layer) value per category."""
    with gzip.open(fpath) as fh:
        data = json.load(fh)
    events = data.get('traceEvents', [])
    kernels = [e for e in events if e.get('cat') == 'kernel']
    by_catname = defaultdict(list)
    for e in kernels:
        cat = classify(e.get('name', ''))
        if cat:
            by_catname[(cat, e['name'])].append((e['ts'], e.get('dur', 0)))

    # steps[cat] = list of dict(layer_id -> summed dur) ; index = step
    steps = {'dispatch': [], 'combine': [], 'expert': []}
    max_steps = {'dispatch': None, 'combine': None, 'expert': None}
    for (cat, name), evs in by_catname.items():
        evs.sort()
        n_chunks = len(evs) // period
        if len(evs) % period != 0:
            evs = evs[:n_chunks * period]  # drop partial trailing chunk (edge of capture window)
        arr = steps[cat]
        while len(arr) < n_chunks:
            arr.append(defaultdict(float))
        for i, (ts, dur) in enumerate(evs):
            chunk = i // period
            layer = i % period
            arr[chunk][layer] += dur
    return steps

def analyze_config(label, snap_dirs):
    ratios = defaultdict(lambda: defaultdict(list))   # layer -> cat -> [ratios]
    hot_rank_seq = defaultdict(list)                  # layer -> [(global_step, hot_rank)] (expert only)
    global_step = 0
    for d in snap_dirs:
        files = sorted(glob.glob(f'{d}/*.trace.json.gz'))
        if len(files) < 4:
            print(f"  WARN: {d} has {len(files)} files, skipping", file=sys.stderr)
            continue
        per_rank = [load_rank_layer_step(f) for f in files]
        num_ranks = len(per_rank)
        for cat in ('dispatch', 'combine', 'expert'):
            n_steps = min(len(pr[cat]) for pr in per_rank)
            for si in range(n_steps):
                for lid in range(48):
                    vals = [per_rank[r][cat][si].get(lid, 0.0) for r in range(num_ranks)]
                    total = sum(vals)
                    if total <= 0:
                        continue
                    avg = total / num_ranks
                    mx = max(vals)
                    ratios[lid][cat].append(mx / avg)
                    if cat == 'expert':
                        hot_rank_seq[lid].append((global_step + si, vals.index(mx)))
        global_step += 100000  # keep step ids from different snapshots from colliding when merged
    return ratios, hot_rank_seq

if __name__ == '__main__':
    label = sys.argv[1]
    dirs = sys.argv[2:]
    ratios, hot_rank_seq = analyze_config(label, dirs)

    print(f"=== {label} ===")
    for cat in ('dispatch', 'combine', 'expert'):
        all_vals = [v for lid in ratios for v in ratios[lid][cat]]
        if not all_vals:
            print(f"  {cat}: NO DATA")
            continue
        per_layer_means = [sum(ratios[lid][cat]) / len(ratios[lid][cat]) for lid in ratios if ratios[lid][cat]]
        print(f"  {cat}: n_samples={len(all_vals)} mean_ratio={sum(all_vals)/len(all_vals):.3f} "
              f"max_ratio={max(all_vals):.3f} mean_of_per_layer_means={sum(per_layer_means)/len(per_layer_means):.3f}")

    seq = hot_rank_seq.get(24, [])
    print(f"\n  layer24 hot_rank sequence (first 30 of {len(seq)}): {[r for _, r in seq[:30]]}")
    if ratios[24]['expert']:
        vals = ratios[24]['expert']
        print(f"  layer24 expert imbalance: n={len(vals)} mean={sum(vals)/len(vals):.3f} max={max(vals):.3f}")

    import pickle
    with open(f'/tmp/imbalance_{label}.pkl', 'wb') as f:
        pickle.dump({'ratios': {k: dict(v) for k, v in ratios.items()}, 'hot_rank_seq': dict(hot_rank_seq)}, f)
    print(f"  saved /tmp/imbalance_{label}.pkl")
