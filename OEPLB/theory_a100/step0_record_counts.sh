#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
SLOG=/workspace/logs/server_a100_recorder.log; : > "$SLOG"
DUMP=/workspace/logs/a100_recorder_dump; rm -rf $DUMP; mkdir -p $DUMP

for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
setsid nohup bash $T/launch_recorder_a100.sh > "$SLOG" 2>&1 < /dev/null &
PID=$!
ready=0
for i in $(seq 1 600); do
  grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 3
done
[ $ready -ne 1 ] && { echo BOOT_FAIL; tail -30 "$SLOG"; exit 1; }
echo "[step0] recorder server ready"

post() { python3 -c "
import urllib.request
req=urllib.request.Request('http://127.0.0.1:30000/$1', data=b'', method='POST')
print(urllib.request.urlopen(req, timeout=120).read()[:100])
"; }

sleep 3
echo "[step0] start record:"; post start_expert_distribution_record
python3 $MB/bench_0914.py $T/data/pinned_head1024.jsonl 1024 profile_head1024
echo "[step0] stop record:"; post stop_expert_distribution_record
sleep 2
echo "[step0] dump:"; post dump_expert_distribution_record
sleep 5
ls -la $DUMP/ | head -5

kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null

echo "[step0] convert pt -> counts json + analysis"
python3 - <<'PY'
import torch, json, glob, numpy as np
fs = sorted(glob.glob("/workspace/logs/a100_recorder_dump/expert_distribution_recorder_*_0.pt"))
print("pt files:", [f.split('/')[-1] for f in fs])
d = torch.load(fs[-1], map_location="cpu", weights_only=False)
lc = torch.as_tensor(d["logical_count"]).sum(dim=0).to(torch.int64).numpy()   # [94,128]
print("counts shape:", lc.shape, "total tokens:", lc.sum())
json.dump({"counts": lc.tolist(), "num_layers": 94, "num_experts": 128},
          open("/workspace/EPLB/OEPLB/theory_a100/data/counts235b_a100.json","w"))
# A100 原生 identity r_before(逐层 max/mean 的均值)
C = lc.astype(np.float64)
r_id = np.mean([ (C[l].reshape(8,16).sum(1).max()) / (C[l].sum()/8) for l in range(94) ])
print(f"A100 原生 identity r_before = {r_id:.4f}")
# 与 H20 FP8 profile 对比
P = np.load('/data/minghua/sjq/OEPLBdata/experiment_logs/moetuner_baseline_20260916/artifacts/P_pinned_fair.npz')
P = P[list(P.keys())[0]].astype(np.float64)
r_id_h20 = np.mean([ (P[l].reshape(8,16).sum(1).max()) / (P[l].sum()/8) for l in range(94) ])
c1=(C/C.sum()).ravel(); c2=(P/P.sum()).ravel()
cos=float((c1*c2).sum()/(np.linalg.norm(c1)*np.linalg.norm(c2)))
print(f"H20 P_pinned_fair r_before = {r_id_h20:.4f} ; A100-vs-H20 余弦 = {cos:.4f}")
PY
echo STEP0_DONE
