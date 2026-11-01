"""CPU correctness test for the MIGRATION-RACE FIX (followup phase 2, r1).

Drives the REAL deployed AsyncSwapExecutor.begin()/try_finish() and the REAL
try_build_swap_plan() under torchrun x8 (gloo), with fixed transaction counts
and hard timeouts per scenario. Verifies: cross-rank plan identity (hash),
no hang on empty/small/299/300-op and consecutive plans, exact canary weight
permutation, busy semantics. Does NOT change any control law.
"""
import importlib.util, json, os, sys, time, hashlib
from datetime import timedelta
import torch
import torch.distributed as dist

DEP = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
SRC = "/workspace/EPLB/OEPLB/src"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_test_results.json")

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

asw = _load("deployed_async_swapper", f"{DEP}/async_swapper.py")   # deployed copy
reb = _load("src_rebalancer", f"{SRC}/rebalancer.py")
SwapOp = reb.SwapOp

L, NUM_LOCAL, EP, D = 94, 16, 8, 2048
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="gloo", timeout=timedelta(seconds=120))
assert world == EP == dist.get_world_size()

class FakeModel:  routed_experts_weights_of_layer = None
class FakeMR:     model = FakeModel()

def fresh_weights(tag):
    """Canary fill: row value encodes (tag, layer, phys_slot) uniquely."""
    rw = {}
    for l in range(L):
        W = torch.arange(NUM_LOCAL, dtype=torch.float32).unsqueeze(1).expand(NUM_LOCAL, D).clone()
        S = -W.clone()
        W += (tag * 100000 + l * 1000)
        S -= (tag * 100000 + l * 1000)
        rw[l] = [W, S]
    return rw

FakeModel.routed_experts_weights_of_layer = fresh_weights(0)
ex = asw.AsyncSwapExecutor(FakeMR(), rank, NUM_LOCAL)

def plan_hash(plan):
    return hashlib.sha256(repr([(o.layer_id, o.phys_slot_a, o.phys_slot_b)
                                for o in plan]).encode()).hexdigest()[:16]

def assert_identical(plan, scen, results):
    h = plan_hash(plan)
    hs = [None] * world
    dist.all_gather_object(hs, h)
    ok = len(set(hs)) == 1
    if rank == 0:
        results[scen]["plan_hash_identical"] = ok
        results[scen]["plan_hash"] = h
        results[scen]["n_ops"] = len(plan)
    assert ok, f"{scen}: plan hash diverged: {hs}"
    return h

def expected_after(G, plan):
    """Exact replica of begin() semantics on the abstract global slot map:
    pass 1 (walk order): same-rank ops swap LIVE values immediately;
    group execution reads rows AFTER pass 1 (snapshot);
    pass 2 (plan order): cross-rank copies overwrite with snapshot values.
    G[r,l,wi,j] = pre-scenario value of rank r's row j (all-gathered), so the
    replica stays valid across accumulated cross-scenario permutations."""
    M = {(l, wi, ps): G[ps // NUM_LOCAL, l, wi, ps % NUM_LOCAL].clone()
         for l in range(L) for wi in range(2) for ps in range(EP*NUM_LOCAL)}
    for op in plan:
        if op.rank_a == op.rank_b:
            for wi in range(2):
                ka = (op.layer_id, wi, op.phys_slot_a); kb = (op.layer_id, wi, op.phys_slot_b)
                M[ka], M[kb] = M[kb].clone(), M[ka].clone()
    snap = {k: v.clone() for k, v in M.items()}
    for op in plan:
        if op.rank_a != op.rank_b:
            for wi in range(2):
                ka = (op.layer_id, wi, op.phys_slot_a); kb = (op.layer_id, wi, op.phys_slot_b)
                M[ka] = snap[kb]; M[kb] = snap[ka]
    exp = {}
    for l in range(L):
        exp[l] = [torch.stack([M[(l, wi, rank*NUM_LOCAL + j)] for j in range(NUM_LOCAL)])
                  for wi in range(2)]
    return exp

def global_orig(orig):
    flat = torch.stack([torch.stack([orig[l][wi] for wi in range(2)]) for l in range(L)])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    return torch.stack(gathered)  # [R,L,2,NUM_LOCAL,D]

def check_canary(G, plan, scen, results):
    exp = expected_after(G, plan)
    live = FakeModel.routed_experts_weights_of_layer
    bad = sum(int(not torch.equal(live[l][wi], exp[l][wi]))
              for l in exp for wi in range(2))
    detail = []
    if bad:
        for l in range(L):
            for wi in range(2):
                if not torch.equal(live[l][wi], exp[l][wi]):
                    rows = (live[l][wi] != exp[l][wi]).any(dim=1).nonzero().flatten().tolist()
                    for j in rows[:3]:
                        phys = rank * NUM_LOCAL + j
                        ops = [(i, o.layer_id, o.phys_slot_a, o.phys_slot_b, o.rank_a, o.rank_b)
                               for i, o in enumerate(plan)
                               if o.layer_id == l and phys in (o.phys_slot_a, o.phys_slot_b)]
                        detail.append({"l": l, "wi": wi, "row": j, "phys": phys,
                                       "exp0": float(exp[l][wi][j][0]), "act0": float(live[l][wi][j][0]),
                                       "orig0": float(G[rank, l, wi, j][0]), "ops": ops})
                    if len(detail) >= 4: break
            if len(detail) >= 4: break
    gl = [None] * world; dist.all_gather_object(gl, bad)
    gd = [None] * world; dist.all_gather_object(gd, detail if detail else None)
    if rank == 0:
        results[scen]["canary_mismatched_tensors"] = sum(gl)
        results[scen]["debug"] = [d for dd in gd if dd for d in dd][:8]
        if sum(gl): print("CANARY_DEBUG", json.dumps(results[scen]["debug"], indent=1))
    assert sum(gl) == 0, f"{scen}: canary mismatch on ranks {[i for i,b in enumerate(gl) if b]}"

def run(scen, plan, results):
    if rank == 0: results[scen] = {}
    orig = {l: [t.clone() for t in FakeModel.routed_experts_weights_of_layer[l]] for l in range(L)}
    G = global_orig(orig)
    assert_identical(plan, scen, results)
    dist.barrier(); t0 = time.time()
    ex.begin(plan)
    fin = ex.try_finish()
    dt = time.time() - t0
    assert fin is not None and not ex.busy
    check_canary(G, plan, scen, results)
    if rank == 0: results[scen]["begin_s"] = round(dt, 3)
    dist.barrier()

results = {}
# ---- S1 empty plan ----
run("S1_empty", [], results)
# ---- S2 single same-rank op (only rank 3 active; others post nothing) ----
p2 = [SwapOp(5, 3*NUM_LOCAL+1, 3*NUM_LOCAL+2, 3, 3, 5*NUM_LOCAL+1, 5*NUM_LOCAL+2, 0.5)]
run("S2_same_rank_only", p2, results)
# ---- S3 handcrafted 8 cross-rank ops ----
p3 = [SwapOp(i, (i % EP)*NUM_LOCAL + i, ((i+1) % EP)*NUM_LOCAL + i, i % EP, (i+1) % EP, 0, 1, 0.5)
      for i in range(8)]
run("S3_hand_8ops", p3, results)
# ---- S4/S5 handcrafted exact 299 / 300 with mixed same-rank ops ----
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
run("S4_hand_299", hand(299, 299), results)
run("S5_hand_300", hand(300, 300), results)
# ---- S6 planner-generated realistic plans (identical skewed load on all ranks) ----
g = torch.Generator().manual_seed(20261020)
load = torch.randint(1, 1000, (L, EP*NUM_LOCAL), generator=g).float()
load[:, 0:NUM_LOCAL] *= 9.0   # heavy skew on rank0 slots
p2l = torch.arange(EP*NUM_LOCAL).unsqueeze(0).repeat(L, 1) % 128
for cap, scen in ((60, "S6_plan_cap60"), (300, "S7_plan_cap300")):
    plan = reb.try_build_swap_plan(logical_count=load, physical_to_logical_map=p2l,
                                   num_ranks=EP, num_local=NUM_LOCAL, threshold_ratio=1.02,
                                   max_swaps_per_layer=64, max_total_swap_layers=94,
                                   max_total_ops=cap)
    run(scen, plan, results)
# ---- S8 consecutive large windows (3x 300-op back-to-back, commit tracking) ----
plans = [hand(300, 1000+k) for k in range(3)]
for k, pl in enumerate(plans):
    run(f"S8_consec_w{k}", pl, results)
# ---- S9 busy semantics: begin while pending must raise on ALL ranks ----
ex.pending = {"plan": [], "result_ready": True}
raised = False
try: ex.begin([])
except RuntimeError: raised = True
gl = [None]*world; dist.all_gather_object(gl, raised); ex.pending = None
if rank == 0: results["S9_busy_raises"] = all(gl)
assert all(gl)
# ---- summary ----
gath = [None]*world; dist.all_gather_object(gath, results if rank == 0 else None)
if rank == 0:
    results["_all_pass"] = True
    json.dump(results, open(OUT, "w"), indent=1)
    print("CPU_TEST_PASS", json.dumps({k: v for k, v in results.items()}))
dist.barrier(); dist.destroy_process_group()
