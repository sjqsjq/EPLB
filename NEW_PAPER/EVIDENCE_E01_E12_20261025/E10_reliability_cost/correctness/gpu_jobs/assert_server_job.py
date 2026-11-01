"""Post-run assertions for a GPU correctness server job (followup phase 2)."""
import json, os, re, sys

art, job, nreq, min_swaps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
no_bench = "--no-bench" in sys.argv
res = {"job": job, "checks": {}, "pass": False}
log = open(f"{art}/server.log", errors="replace").read()

def add(k, ok, detail=""):
    res["checks"][k] = {"ok": bool(ok), "detail": detail}

# 1. no watchdog / coalesced-group timeouts / scheduler exceptions / desync
wd = len(re.findall(r"Watchdog caught", log))
co = len(re.findall(r"COALESCED", log))
sx = len(re.findall(r"Scheduler hit an exception", log))
rd = len(re.findall(r"RANK DESYNC", log))
add("no_watchdog", wd == 0, f"watchdog={wd}")
add("no_coalesced_err", co == 0, f"COALESCED={co}")
add("no_scheduler_exc", sx == 0, f"scheduler_exc={sx}")
add("no_rank_desync", rd == 0, f"rank_desync={rd}")

# 2. PLANHASH cross-rank identity per window (DIAG-1)
ph = {}
for m in re.finditer(r"\[PB-OEPLB-PLANHASH\] window#(\d+) rank=(\d+) n_ops=(\d+) plan_hash=(\w+)", log):
    w, r, n, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    ph.setdefault(w, {})[r] = (n, h)
identity_ok, max_ops = True, 0
for w, per in ph.items():
    hs = {h for _, h in per.values()}; ns = {n for n, _ in per.values()}
    if len(hs) > 1 or len(ns) > 1 or len(per) < 8:
        identity_ok = False
    max_ops = max(max_ops, max(n for n, _ in per.values()))
add("planhash_identity", identity_ok and len(ph) > 0,
    f"plan_windows={len(ph)} {'ok' if identity_ok else 'DIVERGED/incomplete'}")
add("large_plan_exercised", max_ops >= 250, f"max_n_ops={max_ops}")

# 3. swaps issued: unique windows with an issue line >= min_swaps
issued_windows = set(re.findall(r"\[PB-OEPLB\] window#(\d+): issued", log))
add("swaps_min", len(issued_windows) >= min_swaps,
    f"issued_windows={len(issued_windows)} min={min_swaps}")

# 4. v2p1_rank0.jsonl: desync false, h_min==h_max, p2l_version==sp_swaps_started at end
vp = f"{art}/v2p1_rank0.jsonl"
if os.path.exists(vp):
    rows = [json.loads(l) for l in open(vp) if l.strip()]
    bad_d = sum(1 for r in rows if r.get("desync"))
    bad_h = sum(1 for r in rows if r.get("h_min") != r.get("h_max"))
    add("v2p1_no_desync", bad_d == 0, f"rows={len(rows)} desync_true={bad_d}")
    add("v2p1_hash_identical", bad_h == 0, f"hmin_ne_hmax_rows={bad_h}")
    if rows:
        last = rows[-1]
        pv = last.get("p2l_version")
        # correct invariant (validated on old VALID boots m_E3_P_b1/m_E3_F_b1:
        # p2l_version == issued-1, last swap in flight at bench end;
        # sp_swaps_started is the SP sampling counter, NOT total swaps)
        add("p2l_version_match", pv in (len(issued_windows) - 1, len(issued_windows)),
            f"final p2l_version={pv} issued_windows={len(issued_windows)}")
    else:
        add("p2l_version_match", False, "empty jsonl")
else:
    add("v2p1_no_desync", False, "v2p1_rank0.jsonl missing")

# 5. bench completeness
if not no_bench:
    bpath = f"{art}/{job}.json" if os.path.exists(f"{art}/{job}.json") else f"{art}/bench.json"
    try:
        bj = json.load(open(bpath))
        ok, err = bj.get("ok", -1), bj.get("errors", bj.get("err", -1))
        add("bench_complete", ok + err == nreq and err <= max(8, nreq // 500),
            f"ok={ok} err={err} nreq={nreq} file={os.path.basename(bpath)}")
    except Exception as e:
        add("bench_complete", False, f"bench summary missing/unreadable: {e}")
    prq = f"{art}/{job}_perreq.jsonl"
    if os.path.exists(prq):
        n_lines = sum(1 for _ in open(prq))
        add("perreq_lines", n_lines == nreq, f"lines={n_lines}")
    else:
        add("perreq_lines", False, "perreq file missing")

res["pass"] = all(c["ok"] for c in res["checks"].values())
json.dump(res, open(f"{art}/assert_result.json", "w"), indent=1)
print(("PASS " if res["pass"] else "FAIL ") + job + " " +
      json.dumps({k: v["ok"] for k, v in res["checks"].items()}))
sys.exit(0 if res["pass"] else 1)
