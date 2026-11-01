#!/usr/bin/env python3
"""Zero-GPU mining of formal phase4 E3 boots: where does P win/lose vs I, and why.
Descriptive only; no budget; separate author_diag ledger."""
import json, glob, os, re, calendar, statistics as st
G = '/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020/phase4_core/boots'
REF = 6.7419  # frozen SLO E3 ref_p95_TTFT (k=1.0)

def pct(v, q):
    v = sorted(v); i = (len(v)-1)*q/100; lo = int(i); hi = min(lo+1, len(v)-1)
    return v[lo] + (v[hi]-v[lo])*(i-lo)

def load(arm):
    reqs, boots = [], []
    for b in sorted(glob.glob(os.path.join(G, f's_E3_{arm}_b*'))):
        lab = os.path.basename(b)
        f = os.path.join(b, lab + '_perreq.jsonl')
        rows = [json.loads(l) for l in open(f)]
        reqs += rows; boots.append((lab, rows))
    return reqs, boots

print('== E3 TTFT (s), pooled 8 boots/arm ==')
pooled = {}
for arm in 'PI':
    reqs, boots = load(arm)
    t = [r['ttft_s'] for r in reqs if r.get('ttft_s') is not None]
    pooled[arm] = (reqs, boots, t)
    good = sum(1 for x in t if x <= REF)
    print(f'{arm}: n={len(t)} p50={pct(t,50):.3f} p90={pct(t,90):.3f} p95={pct(t,95):.3f} '
          f'p99={pct(t,99):.3f} max={max(t):.3f} GOOD@1.0={good/len(t):.4f} n_over_ref={len(t)-good}')
print('per-boot p95:')
for arm in 'PI':
    reqs, boots, t = pooled[arm]
    print(' ', arm, ' '.join(f'{lab[-2:]}:{pct([r["ttft_s"] for r in rows],95):.2f}' for lab, rows in boots))

# where do P's over-ref requests sit in time? near swap windows?
print()
print('== P arm: TTFT>ref requests vs swap-window issuance ==')
win_re = re.compile(r'^\[([\d\-]+ [\d:]+) DP\d+.*window#(\d+): issued (\d+) swap')
for arm in 'P':
    reqs, boots, t = pooled[arm]
    tot_over = 0; near_win = 0; tot_req = 0
    for lab, rows in boots:
        slog = os.path.join(G, lab, 'server.log')
        wins = {}
        for line in open(slog, errors='ignore'):
            m = win_re.match(line)
            if m:
                ts = calendar.timegm(__import__('time').strptime(m.group(1), '%Y-%m-%d %H:%M:%S'))
                w = int(m.group(2))
                wins.setdefault(w, []).append(ts)
        wtimes = sorted(st.median(v) for v in wins.values())
        over = [r for r in rows if r['ttft_s'] > REF]
        tot_req += len(rows); tot_over += len(over)
        for r in over:
            # request sent at t_send; TTFT window overlaps [t_send, t_last]
            if any(abs(t_send - w) < 30 or (t_send <= w <= r['t_last_wall_s'])
                   for w in wtimes for t_send in [r['t_send_wall_s']]):
                near_win += 1
        print(f'  {lab}: windows={len(wtimes)} reqs={len(rows)} over_ref={len(over)}')
    print(f'  TOTAL over_ref={tot_over}/{tot_req}; over_ref within ±30s of a window issue: {near_win}')

# throughput bins
print()
print('== completion rate (reqs/20s bin), pooled mean profile ==')
for arm in 'PI':
    reqs, boots, t = pooled[arm]
    prof = []
    for lab, rows in boots:
        t0 = min(r['t_send_wall_s'] for r in rows)
        bins = {}
        for r in rows:
            bins[int((r['t_last_wall_s']-t0)//20)] = bins.get(int((r['t_last_wall_s']-t0)//20), 0)+1
        prof.append([bins.get(i, 0) for i in range(0, max(bins)+1)])
    L = min(len(p) for p in prof)
    avg = [st.mean(p[i] for p in prof) for i in range(L)]
    print(f'{arm}: ' + ' '.join(f'{x:.0f}' for x in avg[:24]))
