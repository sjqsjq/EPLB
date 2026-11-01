"""Prep-boot assertions. MODE=plain (I/E arms, no PB) or pb (P arm)."""
import json, os, re, sys

art, label, nreq, mode, min_swaps = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
no_bench = "--no-bench" in sys.argv
res = {"label": label, "mode": mode, "checks": {}, "pass": False}
log = open(f"{art}/server.log", errors="replace").read()

def add(k, ok, detail=""):
    res["checks"][k] = {"ok": bool(ok), "detail": detail}

add("no_watchdog", not re.search(r"Watchdog caught", log))
add("no_scheduler_exc", not re.search(r"Scheduler hit an exception", log))
add("no_coalesced_err", not re.search(r"COALESCED", log))
if mode == "pb":
    add("no_rank_desync", not re.search(r"RANK DESYNC", log))
    ph = {}
    for m in re.finditer(r"\[PB-OEPLB-PLANHASH\] window#(\d+) rank=(\d+) n_ops=(\d+) plan_hash=(\w+)", log):
        ph.setdefault(int(m.group(1)), {})[int(m.group(2))] = (int(m.group(3)), m.group(4))
    ok_id = ph and all(len({h for _, h in p.values()}) == 1 and len({n for n, _ in p.values()}) == 1 and len(p) == 8 for p in ph.values())
    add("planhash_identity", ok_id, f"plan_windows={len(ph)}")
    issued = set(re.findall(r"\[PB-OEPLB\] window#(\d+): issued", log))
    add("swaps_min", len(issued) >= min_swaps, f"issued_windows={len(issued)} min={min_swaps}")
    vp = f"{art}/v2p1_rank0.jsonl"
    if os.path.exists(vp):
        rows = [json.loads(l) for l in open(vp) if l.strip()]
        add("v2p1_no_desync", rows and not any(r.get("desync") for r in rows), f"rows={len(rows)}")
        add("v2p1_hash_identical", rows and all(r.get("h_min") == r.get("h_max") for r in rows), "")
        if rows:
            pv = rows[-1].get("p2l_version")
            add("p2l_version_match", pv in (len(issued) - 1, len(issued)), f"p2l={pv} issued={len(issued)}")
    else:
        add("v2p1_no_desync", False, "missing v2p1")

if not no_bench:
    bpath = f"{art}/{label}.json"
    try:
        bj = json.load(open(bpath))
        ok, err = bj.get("ok", -1), bj.get("errors", bj.get("err", -1))
        add("bench_complete", ok + err == nreq and err <= max(8, nreq // 500), f"ok={ok} err={err} nreq={nreq}")
    except Exception as e:
        add("bench_complete", False, str(e))
    prq = f"{art}/{label}_perreq.jsonl"
    add("perreq_lines", os.path.exists(prq) and sum(1 for _ in open(prq)) == nreq, prq)

res["pass"] = all(c["ok"] for c in res["checks"].values())
json.dump(res, open(f"{art}/assert_result.json", "w"), indent=1)
print(("PASS " if res["pass"] else "FAIL ") + label + " " + json.dumps({k: v["ok"] for k, v in res["checks"].items()}))
sys.exit(0 if res["pass"] else 1)
