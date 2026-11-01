#!/usr/bin/env python3
"""Round-2 extra assert: DIAG-2 pairwise mirror symmetry on every live window.

Usage: assert_r2_extra.py ART_DIR JOB
Parses [PB-OEPLB-P2PPEERS] lines from server.log. For every unordered rank
pair (i,j): the chronological sequence of n_send(i->j) across rank i's lines
must equal the sequence of n_recv(j<-i) across rank j's lines (and vice
versa). Given PLANHASH identity (phase-2 assert), these sequences cover the
same windows in the same order; equality is the measured precondition for
FIX-3's aligned-budget-cut argument. Also checks per-line internal
consistency (totals, n_subgroups).
"""
import json, re, sys, collections

art, job = sys.argv[1], sys.argv[2]
log = open(f"{art}/server.log", errors="replace").read()
pat = re.compile(r"\[PB-OEPLB-P2PPEERS\] rank=(\d+) n_p2p_ops=(\d+) n_subgroups=(\d+) peer_send_recv=(\[\[.*?\]\])")
per_rank = collections.defaultdict(list)
lines = 0
for m in pat.finditer(log):
    r = int(m.group(1)); n_ops = int(m.group(2)); n_sub = int(m.group(3))
    d = {int(p): (int(ns), int(nr)) for p, ns, nr in json.loads(m.group(4))}
    per_rank[r].append((n_ops, n_sub, d))
    lines += 1

errors = []
for r, entries in per_rank.items():
    for k, (n_ops, n_sub, d) in enumerate(entries):
        if sum(ns + nr for ns, nr in d.values()) != n_ops:
            errors.append(f"rank{r} line{k}: peer totals {sum(ns+nr for ns,nr in d.values())} != n_p2p_ops {n_ops}")
        if len(d) != n_sub:
            errors.append(f"rank{r} line{k}: n_subgroups {n_sub} != peers {len(d)}")

pairs_checked = 0
ranks = sorted(per_rank)
for i in ranks:
    for j in ranks:
        if j <= i:
            continue
        seq_i = [d[j] for (_, _, d) in per_rank[i] if j in d]
        seq_j = [d[i] for (_, _, d) in per_rank[j] if i in d]
        if len(seq_i) != len(seq_j):
            errors.append(f"pair({i},{j}): window-count mismatch {len(seq_i)} vs {len(seq_j)}")
            continue
        for k, ((ns_i, nr_i), (ns_j, nr_j)) in enumerate(zip(seq_i, seq_j)):
            pairs_checked += 1
            if ns_i != nr_j:
                errors.append(f"pair({i},{j}) occ{k}: send(i->j)={ns_i} != recv(j<-i)={nr_j}")
            if nr_i != ns_j:
                errors.append(f"pair({i},{j}) occ{k}: recv(i<-j)={nr_i} != send(j->i)={ns_j}")

ok = (lines > 0) and not errors
res = {"job": job, "p2ppeers_lines": lines, "ranks_with_lines": ranks,
       "windows_per_rank": {r: len(per_rank[r]) for r in ranks},
       "pair_occurrences_checked": pairs_checked,
       "mirror_symmetry": "PASS" if ok else "FAIL", "errors": errors[:20]}
json.dump(res, open(f"{art}/assert_r2_extra.json", "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
