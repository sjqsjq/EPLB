import requests, json, time, os, gzip, glob, sys, re
import numpy as np

LABEL = sys.argv[1]
MODEL = "/workspace/Qwen3-30B-A3B-FP8"
DATASET = "/workspace/EPLB/OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl"
with open(DATASET) as f:
    reqs = [json.loads(l) for l in f]

# Warmup
for req in reqs[:80]:
    requests.post("http://localhost:30000/v1/completions", json={
        "model": MODEL, "prompt": req["prompt"], "max_tokens": 32, "temperature": 0}, timeout=60)

# Profile
trace_dir = f"/tmp/trace_sweep_{LABEL}"
os.makedirs(trace_dir, exist_ok=True)
requests.post("http://localhost:30000/start_profile", json={
    "output_dir": trace_dir, "num_steps": 100, "activities": ["CPU", "GPU"]})
for req in reqs[80:160]:
    requests.post("http://localhost:30000/v1/completions", json={
        "model": MODEL, "prompt": req["prompt"], "max_tokens": 32, "temperature": 0}, timeout=60)
time.sleep(3)
requests.post("http://localhost:30000/stop_profile")
time.sleep(3)

# Analyze trace
files = sorted(glob.glob(f'{trace_dir}/*.trace.json.gz'))
all_rank = {}
for rank, fname in enumerate(files):
    with gzip.open(fname) as f:
        data = json.load(f)
    events = data.get('traceEvents', [])
    mw = [(int(e['name'].split('_')[-1]), e['ts'], e['ts']+e['dur'])
          for e in events if e.get('name','').startswith('nn.Module: DeepEPMoE_')]
    mw.sort(key=lambda x: x[1])
    mk = [e for e in events if e.get('cat')=='kernel'
          and 'deep_gemm' in e.get('name','').lower()
          and ('768' in e['name'] or '1536' in e['name'])]
    lt = [(lid, sum(e['dur'] for e in mk if ts0<=e['ts']<=ts1)) for lid, ts0, ts1 in mw]
    steps = []
    for i in range(0, len(lt), 48):
        c = lt[i:i+48]
        if len(c)==48: steps.append({l:t for l,t in c})
    all_rank[rank] = steps

ns = min(len(all_rank[r]) for r in range(4))
total_bn, total_ideal = 0, 0
layer_avgs = []
for lid in range(48):
    ratios = []
    for step in range(ns):
        vals = [all_rank[r][step].get(lid, 0) for r in range(4)]
        avg = sum(vals)/4
        total_bn += max(vals); total_ideal += avg
        if avg > 0.01: ratios.append(max(vals)/avg)
    if ratios: layer_avgs.append(np.mean(ratios))

eff = total_bn / max(total_ideal, 1)

# Extract overhead from log
log = open(f"/tmp/sweep_{LABEL}.log").read()
prof = re.findall(r'record=([\d.]+)ms allreduce=([\d.]+)ms planbuild=([\d.]+)ms finalize=([\d.]+)ms', log)
swaps = re.findall(r'total=(\d+)', log)
o = {'record':0,'allreduce':0,'planbuild':0,'finalize':0}
if prof:
    p = prof[-1]
    o = {'record':float(p[0]),'allreduce':float(p[1]),'planbuild':float(p[2]),'finalize':float(p[3])}
total_swaps = int(swaps[-1]) if swaps else 0

print(f"RESULT {LABEL}: eff={eff:.4f} layer_imbal={np.mean(layer_avgs):.3f} steps={ns} "
      f"swaps={total_swaps} rec={o['record']:.0f} ar={o['allreduce']:.0f} "
      f"pb={o['planbuild']:.0f} fin={o['finalize']:.0f} total_oh={sum(o.values()):.0f}")
