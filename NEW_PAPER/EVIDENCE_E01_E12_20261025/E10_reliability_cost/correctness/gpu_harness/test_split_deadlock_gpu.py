"""ROUND-2 ROOT-CAUSE PROOF: NCCL coalesced-group budget-split deadlock.

Mechanism under test (pytorch/pytorch#174288, OPEN, repro'd on NCCL<=2.29):
  torch batch_isend_irecv() wraps ALL ops in ONE ncclGroupStart/End. NCCL
  splits a large p2p group into MULTIPLE SEQUENTIAL KERNEL PLANS via an
  internal budget (nccl enqueue.cc scheduleP2pTasksToPlan). NCCL source
  itself notes the cut point "could vary by rank". With ASYMMETRIC per-rank
  participation (exactly what a partitioned swap plan produces: rank0 has
  more ops than rank1, ...), matched send/recv pairs land in DIFFERENT plan
  indices on the two sides. Plans run sequentially on the stream ->
  circular wait between ranks' plan kernels -> silent deadlock: all ranks
  posted (SeqNum enqueued), kernels resident (GPU 100%), no NCCL WARN,
  watchdog fires at timeout. This is the B1/prep3 (and old m_E3_P_b3)
  window#17 signature: plan identical on all ranks (PLANHASH measured),
  fast cluster completed (aligned cuts), slow cluster all-posted-never-done.

Modes:
  --mode single : OLD posting path  = one batch_isend_irecv(all ops)
                  -> expected: HANG (watchdog abort at --timeout)
  --mode perpeer: FIX-3 candidate   = per-peer subgroups, ascending peer
                  order, host-wait each subgroup, sync at end
                  -> expected: PASS all iters, canary bad=0

Plan generator mimics window#17: 299 ops, PARTITIONED into clusters
{0,1,5,6} and {2,3,4,7}, skewed per-rank participation, 2 weights per op
(main ~13.9MB uint8 + scale ~0.9MB), unique (rank,layer,slot) per plan.
Deterministic from seed -> identical on all ranks (asserted via gloo
allgather of plan hash each iter).
No production code imported: posting paths implemented locally so the
deployed module is untouched until the mechanism is proven.
"""
import argparse, hashlib, os, random, sys, time
import torch
import torch.distributed as dist
from datetime import timedelta

NUM_LOCAL = 16
N_LAYERS = 94
NW_ROWS = 13_000_000   # main weight bytes (uint8) ~13.9MB w/ header
NS_ROWS = 900_000      # scale bytes
CLUSTERS = [[0, 1, 5, 6], [2, 3, 4, 7]]

def fill_coef(iter_i, layer, rank, slot, wi):
    A = 1 + 2 * ((iter_i * 131 + layer * 31 + rank * 17 + slot * 7 + wi * 3) % 100)
    B = (iter_i * 29 + layer * 11 + rank * 13 + slot * 5 + wi * 71) % 256
    return A, B

def make_tensor(iter_i, layer, rank, slot, wi, rows, device):
    A, B = fill_coef(iter_i, layer, rank, slot, wi)
    base = torch.arange(rows, dtype=torch.uint8, device=device)
    return ((base * A + B) % 251).to(torch.uint8)

def gen_plan(seed):
    """Identical on every rank. Returns list of (layer, ra, sa, rb, sb) with
    local slots; total 299 ops, partitioned, skewed participation."""
    rng = random.Random(seed)
    ops = []
    used = set()
    targets = [(CLUSTERS[0], 150, [30, 25, 20, 12, 8, 5]),
               (CLUSTERS[1], 149, [5, 8, 12, 20, 25, 30])]
    for cl, n_ops, wts in targets:
        pairs = [(cl[0], cl[1]), (cl[0], cl[2]), (cl[0], cl[3]),
                 (cl[1], cl[2]), (cl[1], cl[3]), (cl[2], cl[3])]
        made = 0
        guard = 0
        while made < n_ops and guard < n_ops * 200:
            guard += 1
            a, b = rng.choices(pairs, wts)[0]
            layer = rng.randrange(N_LAYERS)
            sa = rng.randrange(NUM_LOCAL); sb = rng.randrange(NUM_LOCAL)
            if (a, layer, sa) in used or (b, layer, sb) in used:
                continue
            used.add((a, layer, sa)); used.add((b, layer, sb))
            ops.append((layer, a, sa, b, sb))
            made += 1
    return ops

def plan_hash(plan):
    return hashlib.sha256(repr(plan).encode()).hexdigest()[:16]

def build_local_ops(plan, my_rank, iter_i, srcs, temps, device):
    """Mirror async_swapper.begin() walk: per participating op, 2 weights x
    [isend(src), irecv(temp)] in plan order."""
    p2p = []
    for i, (layer, ra, sa, rb, sb) in enumerate(plan):
        if my_rank == ra:
            peer, slot = rb, sa
            role = 'a'
        elif my_rank == rb:
            peer, slot = ra, sb
            role = 'b'
        else:
            continue
        srcs[(i, role)] = [make_tensor(iter_i, layer, my_rank, slot, wi, r, device)
                           for wi, r in ((0, NW_ROWS), (1, NS_ROWS))]
        temps[(i, role)] = [torch.empty(r, dtype=torch.uint8, device=device)
                            for r in (NW_ROWS, NS_ROWS)]
        for wi in (0, 1):
            p2p.append((peer, dist.P2POp(dist.isend, srcs[(i, role)][wi], peer)))
            p2p.append((peer, dist.P2POp(dist.irecv, temps[(i, role)][wi], peer)))
    return p2p

def check_canary(plan, my_rank, iter_i, temps, device):
    bad = 0
    for i, (layer, ra, sa, rb, sb) in enumerate(plan):
        if my_rank == ra:
            peer, slot = rb, sb
        elif my_rank == rb:
            peer, slot = ra, sa
        else:
            continue
        # received temp must equal SENDER's formula
        for wi, rows in ((0, NW_ROWS), (1, NS_ROWS)):
            A, B = fill_coef(iter_i, layer, peer, slot, wi)
            base = torch.arange(rows, dtype=torch.uint8, device=device)
            exp = ((base * A + B) % 251).to(torch.uint8)
            got = temps[(i, 'a' if my_rank == ra else 'b')][wi]
            bad += int((got != exp).sum().item())
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['single', 'perpeer'], required=True)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--seed0', type=int, default=70000)
    args = ap.parse_args()

    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=args.timeout),
                            rank=rank, world_size=world)
    gloo = dist.new_group(backend='gloo')
    if rank == 0:
        print(f"[repro] mode={args.mode} iters={args.iters} timeout={args.timeout}s "
              f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)

    total_bad = 0
    t0 = time.time()
    for it in range(args.iters):
        seed = args.seed0 + it * 1013
        plan = gen_plan(seed)
        # plan identity across ranks (gloo, cheap)
        h = torch.tensor([int(plan_hash(plan)[:8], 16)], dtype=torch.int64)
        hs = [torch.zeros_like(h) for _ in range(world)]
        dist.all_gather(hs, h, group=gloo)
        assert len(set(int(x) for x in hs)) == 1, f"iter{it} plan diverged"
        srcs = {}; temps = {}
        p2p = build_local_ops(plan, rank, it, srcs, temps, device)
        dist.barrier(group=gloo)
        t_post = time.time()
        if args.mode == 'single':
            reqs = dist.batch_isend_irecv([op for _, op in p2p])
            for r in reqs: r.wait()
            torch.cuda.synchronize()
        else:
            by_peer = {}
            for peer, op in p2p:
                by_peer.setdefault(peer, []).append(op)
            for peer in sorted(by_peer):
                reqs = dist.batch_isend_irecv(by_peer[peer])
                for r in reqs: r.wait()
            torch.cuda.synchronize()
        t_xfer = time.time()
        n_ops_local = len(p2p)
        bad = check_canary(plan, rank, it, temps, device)
        total_bad += bad
        del srcs, temps, p2p
        torch.cuda.empty_cache()
        if rank == 0 or bad:
            print(f"[repro] iter={it} rank={rank} n_p2p_ops={n_ops_local} "
                  f"xfer={1000*(t_xfer-t_post):.0f}ms canary_bad={bad} elapsed={time.time()-t0:.0f}s", flush=True)
        # progress marker on every rank (last line before a hang localizes it)
        print(f"[progress] rank={rank} iter={it} done bad={bad}", flush=True)
        dist.barrier(group=gloo)
    if rank == 0:
        print(f"[repro] MODE={args.mode} ALL {args.iters} ITERS COMPLETE total_canary_bad={total_bad} "
              f"wall={time.time()-t0:.0f}s -> {'PASS' if total_bad==0 else 'FAIL'}", flush=True)
    dist.barrier(group=gloo)
    dist.destroy_process_group()
    sys.exit(0 if total_bad == 0 else 2)

if __name__ == '__main__':
    main()
