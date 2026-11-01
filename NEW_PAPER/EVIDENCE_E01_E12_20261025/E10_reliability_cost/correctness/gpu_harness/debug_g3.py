"""Debug rerun of G3_hand8: identify actual origin of every bad row by
brute-force formula match, dump full plan + per-rank detail."""
import importlib.util, os, json, hashlib
from datetime import timedelta
import torch, torch.distributed as dist

DEP = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
SRC = "/workspace/EPLB/OEPLB/src"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
asw = _load("asw", f"{DEP}/async_swapper.py")
reb = _load("reb", f"{SRC}/rebalancer.py")
SwapOp = reb.SwapOp
EP, NUM_LOCAL, L, NW, NS = 8, 16, 12, 13_000_000, 900_000
rank = int(os.environ["RANK"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=300))

def fill(l, phys, wi, n):
    A = 1 + 2 * ((l * 31 + phys * 17 + wi * 7) % 100)
    B = (l * 131 + phys * 29 + wi * 11) % 256
    return (torch.arange(n, device=dev, dtype=torch.uint8) * A + B).to(torch.uint8)

rw = {}
for l in range(L):
    rw[l] = [torch.stack([fill(l, rank*NUM_LOCAL+j, 0, NW) for j in range(NUM_LOCAL)]),
             torch.stack([fill(l, rank*NUM_LOCAL+j, 1, NS) for j in range(NUM_LOCAL)])]
class FM: routed_experts_weights_of_layer = rw
class FMR: model = FM()
ex = asw.AsyncSwapExecutor(FMR(), rank, NUM_LOCAL)

def hand(n, seed):
    g = torch.Generator().manual_seed(seed); ops = []
    for i in range(n):
        l = int(torch.randint(0, L, (1,), generator=g))
        ra = int(torch.randint(0, EP, (1,), generator=g))
        rb = ra if i % 17 == 0 else int(torch.randint(0, EP, (1,), generator=g))
        sa = int(torch.randint(0, NUM_LOCAL, (1,), generator=g))
        sb = int(torch.randint(0, NUM_LOCAL, (1,), generator=g))
        ops.append(SwapOp(l, ra*NUM_LOCAL+sa, rb*NUM_LOCAL+sb, ra, rb, 0, 1, 0.5))
    return ops

plan = hand(8, 8)
if rank == 0:
    print("PLAN:", [(i, o.layer_id, o.phys_slot_a, o.phys_slot_b, o.rank_a, o.rank_b) for i, o in enumerate(plan)], flush=True)

def expected_origins(plan):
    M = {(l, wi, p): p for l in range(L) for wi in range(2) for p in range(EP*NUM_LOCAL)}
    for op in plan:
        if op.rank_a == op.rank_b:
            for wi in range(2):
                ka = (op.layer_id, wi, op.phys_slot_a); kb = (op.layer_id, wi, op.phys_slot_b)
                M[ka], M[kb] = M[kb], M[ka]
    snap = dict(M)
    for op in plan:
        if op.rank_a != op.rank_b:
            for wi in range(2):
                ka = (op.layer_id, wi, op.phys_slot_a); kb = (op.layer_id, wi, op.phys_slot_b)
                M[ka] = snap[kb]; M[kb] = snap[ka]
    return M

M = expected_origins(plan)
dist.barrier()
ex.begin(plan); ex.try_finish()
live = FM.routed_experts_weights_of_layer
sizes = (NW, NS)
bad = []
for l in range(L):
    for wi in range(2):
        for j in range(NUM_LOCAL):
            phys = rank*NUM_LOCAL + j
            want_org = M[(l, wi, phys)]
            if not torch.equal(live[l][wi][j], fill(l, want_org, wi, sizes[wi])):
                # brute force actual origin
                act = None
                for p in range(EP*NUM_LOCAL):
                    if torch.equal(live[l][wi][j], fill(l, p, wi, sizes[wi])):
                        act = p; break
                bad.append({"l": l, "wi": wi, "j": j, "phys": phys,
                            "expected_origin": want_org, "actual_origin": act,
                            "replica_says": want_org})
gl = [None]*8; dist.all_gather_object(gl, bad)
if rank == 0:
    allbad = [b for r in gl for b in r]
    print("BAD_ROWS:", json.dumps(allbad), flush=True)
    # for each bad row, list plan ops touching that (l, phys)
    for b in allbad:
        touching = [(i, o.layer_id, o.phys_slot_a, o.phys_slot_b, o.rank_a, o.rank_b)
                    for i, o in enumerate(plan)
                    if o.layer_id == b["l"] and b["phys"] in (o.phys_slot_a, o.phys_slot_b)]
        print("ROW", b, "plan_ops_touching:", touching, flush=True)
dist.barrier(); dist.destroy_process_group()
