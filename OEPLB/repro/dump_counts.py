"""Start/stop/dump the SGLang expert-distribution recorder, then convert the
.pt dump into a compact JSON of per-layer per-logical-expert token counts."""
import json, os, sys, glob, time
import requests, torch

BASE = "http://127.0.0.1:30000"
DUMPDIR = "/workspace/logs/recdump"
out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/logs/counts.json"
action = sys.argv[2] if len(sys.argv) > 2 else "all"

def post(ep):
    r = requests.post(BASE + ep, timeout=600)
    print(ep, r.status_code, r.text.strip())

if action in ("start", "all"):
    post("/start_expert_distribution_record")
    if action == "start":
        sys.exit(0)

if action in ("stop", "all"):
    post("/stop_expert_distribution_record")
    before = set(glob.glob(DUMPDIR + "/*.pt"))
    post("/dump_expert_distribution_record")
    new = None
    for _ in range(60):
        cand = set(glob.glob(DUMPDIR + "/*.pt")) - before
        if cand:
            new = sorted(cand)[-1]
            break
        time.sleep(1)
    if new is None:
        print("ERROR: no new .pt appeared in", DUMPDIR); sys.exit(1)
    print("dump file:", new)
    d = torch.load(new, map_location="cpu", weights_only=False)
    print("keys:", list(d.keys()) if isinstance(d, dict) else type(d))
    lc = d["logical_count"] if isinstance(d, dict) else d
    lc = torch.as_tensor(lc)
    print("logical_count shape", tuple(lc.shape), "dtype", lc.dtype, "sum", int(lc.sum()))
    if lc.dim() == 3:      # (pass, layer, expert) -> sum over passes
        lc = lc.sum(0)
    counts = lc.to(torch.float64).tolist()
    json.dump({"counts": counts,
               "num_layers": len(counts),
               "num_experts": len(counts[0]),
               "src": new}, open(out, "w"))
    print("wrote", out, len(counts), "layers x", len(counts[0]), "experts")
