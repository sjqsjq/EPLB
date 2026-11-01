import csv, collections, itertools
rows=list(csv.DictReader(open("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/adw_v2p1/formal_eval/run_list_formal_eval_v2p1.csv")))
print("total rows:",len(rows))
for cond in ("W1","W2"):
    cr=[r for r in rows if r["condition"]==cond]
    pos=collections.Counter((r["arm"],int(r["arm_pos_in_block"])) for r in cr)
    print("\n== %s (%d starts) position counts (arm: pos1/pos2/pos3/pos4) =="%(cond,len(cr)))
    for arm in ("A","F","L","M"):
        print("  %s: %s"%(arm,[pos[(arm,p)] for p in (1,2,3,4)]))
    # precedence: within each block, which arm runs before which
    prec=collections.Counter()
    blocks=collections.defaultdict(list)
    for r in cr: blocks[r["round"]].append(r)
    for rnd,bl in blocks.items():
        bl=sorted(bl,key=lambda r:int(r["arm_pos_in_block"]))
        for a,b in itertools.combinations([r["arm"] for r in bl],2):
            prec[(a,b)]+=1  # a before b
    print("  precedence (X before Y, of 6 blocks):")
    for a,b in itertools.combinations(("A","F","L","M"),2):
        print("    %s<%s: %d : %d"%(a,b,prec[(a,b)],prec[(b,a)]))
    # condition block first/second half of round
    fh=collections.Counter(r["block_pos"] for r in cr)
    print("  block_pos in round:",dict(fh))
