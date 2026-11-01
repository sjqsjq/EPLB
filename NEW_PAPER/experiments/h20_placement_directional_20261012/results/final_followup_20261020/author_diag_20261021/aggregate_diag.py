#!/usr/bin/env python3
"""Aggregate author_diag matrix (E3/E4/freq6 x gh_pristine/v2 x paper/campaign flags).
Descriptive only; separate ledger; medians over available rounds. Run after queues finish."""
import json, os, re, statistics as st
D = os.path.dirname(os.path.abspath(__file__))
rows = []
for line in open(os.path.join(D, 'LEDGER_adhoc.tsv')):
    if line.startswith('#') or line.startswith('label'): continue
    p = line.rstrip('\n').split('\t')
    if len(p) < 6 or p[3] == '-': continue
    rows.append(dict(zip(['label','start','end','rc','tps','code'], p)))

def cell(match, tps_field='steady'):
    vals = []
    for r in rows:
        if re.match(match, r['label']) and r['rc'] == '0':
            t = r['tps']
            if '/' in t:  # freq6 run1/run2 -> steady=run2
                v = float(t.split('/')[1])
            else:
                v = float(t)
            vals.append((r['label'], v))
    return vals

CELLS = [
    ('freq6 identity (paper flags)',      r'f6_I_paper'),
    ('freq6 v2 (paper flags) steady',     r'f6_v2_P_paper'),
    ('freq6 gh (paper flags) steady',     r'f6_gh_P_paper'),
    ('freq6 identity (campaign flags)',   r'f6_I_camp'),
    ('freq6 v2 (campaign flags) steady',  r'f6_v2_P_camp'),
    ('E3 identity (paper flags)',         r'e3_I_paper'),
    ('E3 v2 (paper flags)',               r'e3_v2_paper'),
    ('E3 gh (paper flags)',               r'e3_gh_paper'),
    ('E3 identity (campaign flags)',      r'adhoc_I_v2_E3'),
    ('E3 v2 (campaign flags)',            r'adhoc_P_v2_E3'),
    ('E3 gh (campaign flags)',            r'adhoc_P_gh_E3'),
    ('E4 identity (paper flags)',         r'e4_I_paper'),
    ('E4 v2 (paper flags)',               r'e4_v2_paper'),
    ('E4 gh (paper flags)',               r'e4_gh_paper'),
    ('E3 v2 (campaign, FORMAL 8-boot)',   None),
    ('E4 v2/I (campaign, FORMAL 8-boot)', None),
]
print('| cell | n | values | median |')
print('|---|---|---|---|')
for name, pat in CELLS:
    if pat is None:
        print(f'| {name} | 8 | see RESULTS_SUPP.md | E3: I=41.92 P=43.06; E4: I=2840.25 P=2613.95 |')
        continue
    vals = cell(pat)
    if not vals:
        print(f'| {name} | 0 | — | — |'); continue
    vs = [v for _, v in vals]
    lab = ', '.join(f'{l.replace("_paper","").replace("adhoc_","")}={v}' for l, v in vals)
    print(f'| {name} | {len(vs)} | {lab} | {st.median(vs):.2f} |')
# relative gains on medians
def med(pat):
    vs = [v for _, v in cell(pat)]
    return st.median(vs) if vs else None
print()
for ds, ip, vp, gp in [('freq6', r'f6_I_paper', r'f6_v2_P_paper', r'f6_gh_P_paper'),
                        ('E3', r'e3_I_paper', r'e3_v2_paper', r'e3_gh_paper'),
                        ('E4', r'e4_I_paper', r'e4_v2_paper', r'e4_gh_paper')]:
    i, v, g = med(ip), med(vp), med(gp)
    if i and v and g:
        print(f'{ds} (paper flags, medians): identity={i:.2f} v2={v:.2f} ({100*(v/i-1):+.1f}%) gh={g:.2f} ({100*(g/i-1):+.1f}%)')
