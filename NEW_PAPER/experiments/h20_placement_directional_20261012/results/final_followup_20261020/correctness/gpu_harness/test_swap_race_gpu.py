"""GPU correctness harness for the MIGRATION-RACE FIX (followup phase 2).

SCEN=j4: 12 layers, real-scale row bytes (~13.9MB/op), full scenario matrix:
  empty / 1-op same-rank / 8 cross / 298 / 299 / 300 / 301(adjacent boundary)
  / consecutive 5x300 under side-stream SM churn / planner-generated cap300.
SCEN=j5: 94 layers (full real geometry, ~20.9GB/GPU), consecutive 3x300 +
  planner cap300 under churn -- memory/allocator pressure at production scale.

Drives the REAL deployed AsyncSwapExecutor (NCCL, cuda) + real planner.
Canary = formula fill f(l, phys, wi): replica tracks slot ORIGINS (ints)
through begin() semantics; verification recomputes rows -> zero cross-rank
tensor gathers. Also asserts per-rank comm-order construction determinism
and cross-rank plan-hash identity per scenario.
"""
import importlib.util, json, os, sys, time, hashlib, threading
from datetime import timedelta
import torch
import torch.distributed as dist

SCEN = os.environ.get("SCEN", "j4")
DEP = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
SRC = "/workspace/EPLB/OEPLB/src"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

asw = _load("deployed_async_swapper", f"{DEP}/async_swapper.py")
reb = _load("src_rebalancer", f"{SRC}/rebalancer.py")
SwapOp = reb.SwapOp

EP, NUM_LOCAL = 8, 16
L = 12 if SCEN == "j4" else 94
NW, NS = 13_000_000, 900_000           # ~13.9MB per op (2 tensors), production-scale
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))

def fill(l, phys, wi, n):
    A = 1 + 2 * ((l * 31 + phys * 17 + wi * 7) % 100)
    B = (l * 131 + phys * 29 + wi * 11) % 256
    return (torch.arange(n, device=dev, dtype=torch.uint8) * A + B).to(torch.uint8)

class FakeModel: routed_experts_weights_of_layer = None
class FakeMR:     model = FakeModel()
# build [W(16,NW), S(16,NS)] per layer
rw = {}
for l in range(L):
    W = torch.stack([fill(l, rank * NUM_LOCAL + j, 0, NW) for j in range(NUM_LOCAL)])
    S = torch.stack([fill(l, rank * NUM_LOCAL + j, 1, NS) for j in range(NUM_LOCAL)])
    rw[l] = [W, S]
FakeModel.routed_experts_weights_of_layer = rw
ex = asw.AsyncSwapExecutor(FakeMR(), rank, NUM_LOCAL)

def plan_hash(plan):
    return hashlib.sha256(repr([(o.layer_id, o.phys_slot_a, o.phys_slot_b)
                                for o in plan]).encode()).hexdigest()[:16]

def comm_order_hash(plan):
    """Replicate begin()'s per-rank p2p posting sequence twice; assert
    determinism; return hash of the constructed op order (isend/irecv/same)."""
    hs = []
    for _ in range(2):
        seq = []
        for op in plan:
            if rank == op.rank_a == op.rank_b: seq.append(("local", op.layer_id, op.phys_slot_a, op.phys_slot_b))
            elif rank == op.rank_a:
                for wi in range(2): seq.append(("send", op.layer_id, op.phys_slot_a, wi)); seq.append(("recv", op.layer_id, op.phys_slot_a, wi))
            elif rank == op.rank_b:
                for wi in range(2): seq.append(("recv", op.layer_id, op.phys_slot_b, wi)); seq.append(("send", op.layer_id, op.phys_slot_b, wi))
        hs.append(hashlib.sha256(repr(seq).encode()).hexdigest()[:16])
    assert hs[0] == hs[1], "comm-order construction nondeterministic"
    return hs[0]

# Persistent slot-origin state across scenarios (live tensors accumulate
# permutations; the replica must carry them too -- harness bug found by
# G3 debug rerun 2026-10-20: identity-reset replica flagged exactly the
# 4 rows the previous scenario had moved).
M_STATE = {(l, wi, p): p for l in range(L) for wi in range(2) for p in range(EP * NUM_LOCAL)}

def expected_origins(plan, M=None):
    """Slot-origin replica of begin(): pass1 same-rank swaps in walk order;
    group snapshot; pass2 cross copies in plan order from snapshot.
    Starts from M (default: persistent M_STATE)."""
    M = dict(M_STATE if M is None else M)
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

def check_canary(M, scen, results):
    live = FakeModel.routed_experts_weights_of_layer
    bad = 0; first = None
    sizes = (NW, NS)
    for l in range(L):
        for wi in range(2):
            for j in range(NUM_LOCAL):
                phys = rank * NUM_LOCAL + j
                want = fill(l, M[(l, wi, phys)], wi, sizes[wi])
                if not torch.equal(live[l][wi][j], want):
                    bad += 1
                    if first is None: first = (l, wi, j, phys)
    gl = [None] * world; dist.all_gather_object(gl, (bad, first))
    tot = sum(b for b, _ in gl)
    if rank == 0:
        results[scen]["canary_bad_rows_total"] = tot
        results[scen]["canary_first_bad"] = [g[1] for g in gl if g[1]][:4]
    assert tot == 0, f"{scen}: canary mismatch total={tot} first={ [g[1] for g in gl if g[1]][:4] }"

class Churn:
    """Side-stream SM contention; fresh thread+event per scenario (a stopped
    Thread cannot be restarted -- harness bug found in attempt 3)."""
    def __init__(self): self.t = None
    def start(self):
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target=self._work, daemon=True); self.t.start()
    def _work(self):
        s = torch.cuda.Stream(device=dev)
        a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
        with torch.cuda.stream(s):
            while not self.stop_ev.is_set():
                for _ in range(30):
                    a = (a @ b * 1e-3).half()
                s.synchronize()
                time.sleep(0.02)
    def stop(self):
        self.stop_ev.set()
        if self.t: self.t.join(timeout=30)
churn = Churn()

def run(scen, plan, results, use_churn=False):
    if rank == 0: results[scen] = {}
    h = plan_hash(plan); co = comm_order_hash(plan)
    hs = [None] * world; cs = [None] * world
    dist.all_gather_object(hs, h); dist.all_gather_object(cs, co)
    if rank == 0:
        results[scen].update(plan_hash=h, plan_hash_identical=len(set(hs)) == 1, n_ops=len(plan))
    assert len(set(hs)) == 1, f"{scen}: plan hash diverged {hs}"
    global M_STATE
    M = expected_origins(plan)
    dist.barrier()
    if use_churn: churn.start()
    t0 = time.time()
    ex.begin(plan)
    fin = ex.try_finish()
    dt = time.time() - t0
    assert fin is not None and not ex.busy
    if use_churn:
        churn.stop()
    check_canary(M, scen, results)
    M_STATE = M   # canary verified: adopt as new persistent origin state
    if rank == 0: results[scen]["begin_s"] = round(dt, 3)
    dist.barrier()

def hand(n, seed):
    """Slot-disjoint hand plan (matches real planner invariant: each
    (layer, phys_slot) appears at most once per plan). Identical on all
    ranks: same seed, same draw order, CPU generator."""
    g = torch.Generator().manual_seed(seed); ops = []; used = set()
    while len(ops) < n:
        l = int(torch.randint(0, L, (1,), generator=g))
        ra = int(torch.randint(0, EP, (1,), generator=g))
        same = len(ops) % 17 == 0
        rb = ra if same else int(torch.randint(0, EP, (1,), generator=g))
        if rb == ra and not same:
            continue
        sa = int(torch.randint(0, NUM_LOCAL, (1,), generator=g))
        sb = int(torch.randint(0, NUM_LOCAL, (1,), generator=g))
        pa, pb = ra * NUM_LOCAL + sa, rb * NUM_LOCAL + sb
        if pa == pb or (l, pa) in used or (l, pb) in used:
            continue
        used.add((l, pa)); used.add((l, pb))
        ops.append(SwapOp(l, pa, pb, ra, rb, 0, 1, 0.5))
    return ops

results = {}
if SCEN == "j4":
    run("G1_empty", [], results)
    run("G2_same_rank_1op", [SwapOp(3, 5 * NUM_LOCAL + 1, 5 * NUM_LOCAL + 2, 5, 5, 0, 1, 0.5)], results)
    run("G3_hand8", hand(8, 8), results)
    run("G4_hand298", hand(298, 298), results)
    run("G5_hand299", hand(299, 299), results)
    run("G6_hand300", hand(300, 300), results)
    run("G7_hand301_adjacent", hand(301, 301), results)
    for k in range(5):
        run(f"G8_consec_churn_w{k}", hand(300, 2000 + k), results, use_churn=True)
    load = torch.randint(1, 1000, (L, EP * NUM_LOCAL), generator=torch.Generator().manual_seed(7)).float()
    load[:, 0:NUM_LOCAL] *= 9.0
    p2l = torch.arange(EP * NUM_LOCAL).unsqueeze(0).repeat(L, 1) % 128
    pl = reb.try_build_swap_plan(logical_count=load, physical_to_logical_map=p2l, num_ranks=EP,
                                 num_local=NUM_LOCAL, threshold_ratio=1.02, max_swaps_per_layer=64,
                                 max_total_swap_layers=94, max_total_ops=300)
    run("G9_planner_cap300", pl, results, use_churn=True)
else:  # j5
    run("H1_consec300_w0", hand(300, 3001), results, use_churn=True)
    run("H2_consec300_w1", hand(300, 3002), results, use_churn=True)
    run("H3_consec300_w2", hand(300, 3003), results, use_churn=True)
    load = torch.randint(1, 1000, (L, EP * NUM_LOCAL), generator=torch.Generator().manual_seed(71)).float()
    load[:, 0:NUM_LOCAL] *= 9.0; load[:, NUM_LOCAL:2 * NUM_LOCAL] *= 4.0
    p2l = torch.arange(EP * NUM_LOCAL).unsqueeze(0).repeat(L, 1) % 128
    pl = reb.try_build_swap_plan(logical_count=load, physical_to_logical_map=p2l, num_ranks=EP,
                                 num_local=NUM_LOCAL, threshold_ratio=1.02, max_swaps_per_layer=64,
                                 max_total_swap_layers=94, max_total_ops=300)
    run("H4_planner_cap300", pl, results, use_churn=True)

if rank == 0:
    results["_scen"] = SCEN; results["_all_pass"] = True
    json.dump(results, open(f"{OUTDIR}/gpu_{SCEN}_results.json", "w"), indent=1)
    print(f"GPU_{SCEN.upper()}_PASS", json.dumps(results))
dist.barrier(); dist.destroy_process_group()
