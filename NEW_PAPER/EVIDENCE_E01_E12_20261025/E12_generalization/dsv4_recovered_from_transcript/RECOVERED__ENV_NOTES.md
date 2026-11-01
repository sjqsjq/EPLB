# DSV4 适配环境记录 (Phase 1, 2026-10-22)

目标: 在隔离环境跑通 DeepSeek-V4-Flash (284B, 官方 FP4 专家 + FP8 非专家, ~160GB)。
冻结部署 (/opt/conda sglang 0.5.6.post2 + pb_oeplb) 全程未动。

## 环境
- venv: /opt/dsv4_env (python 3.11.12, 隔离, 无 system-site-packages)
- sglang 0.5.22 (原生 DeepseekV4ForCausalLM: deepseek_v4*.py ×5)
- torch 2.14.1+cu130, CUDA 13.0; 驱动 580.105.08 / CUDA 13.0 → 兼容
- 镜像: aliyun 实测 0.12MB/s(不可用, torch 栈需 6-10h) → **tencent 33.7MB/s** (4.4GB/8min)
  (tuna 25.1MB/s 备选; mirrors.cloud.aliyuncs.com 内网不可达)

## 故障与修复 (全部环境级, 未改任何框架代码)
| try | 现象 | 根因 | 修复 |
|-----|------|------|------|
| 1 | deep_gemm 断言 major>12 or (12,minor>=9), 8 rank 全崩 | 系统 nvcc 12.8 < 要求 12.9 | CUDA_HOME/CUDA_PATH/PATH 指向 venv 自带 nvidia/cu13 工具链 |
| 2 | CCCL #error "CUDA compiler and CUDA toolkit headers are incompatible" | nvcc 13.4 vs runtime 头 13.0 (CUDART_VERSION 13000) | nvidia-cuda-nvcc/crt/nvjitlink 降到 ~=13.0.0 (13.0.88) |
| 3 | ptxas fatal: Unsupported .version 9.4; current 9.0 | **nvidia-nvvm**(cicc) 残留 13.4 发 PTX 9.4, ptxas 13.0 只认 9.0 | nvidia-nvvm==13.0.88; 微编译验证通过 |
| 4 | flashinfer fused_moe 编译 15min 后最终链接 `ld: cannot find -lcudart` | wheel 只带 libcudart.so.13, 无开发符号链接 libcudart.so | ln -s libcudart.so{,.13}/libnvrtc.so/libnvJitLink.so; smoke 脚本加 LIBRARY_PATH=$CUDA_HOME/lib |
| 4' | 脚本 20min ready 超时误报 SMOKE_FAIL (服务器实际存活在编译) | 首次 JIT (MHC prewarm 251s + CUTLASS moe ~15min) 超出等待 | 等待窗口 120→540×10s; watcher2 兜底 |
| 5 | curl 不存在 → /generate 探针未执行(脚本仍写 SMOKE_DONE) | 镜像 shell 无 curl | 手动 python urllib 补探针, GENERATE_OK 入台账 |

## try5 成功时间线 (UTC)
- 11:06:12 launch → 11:07:2x 权重 46/46 (~40s) → 11:08:03 43 层 MXFP4 专家重排完成(CUTLASS W4A16, 模块已缓存仅重链)
- 11:08:12 custom allreduce 对称内存 OK → KV 池 84.7GB/GPU → DeepGEMM warmup 16384 形状 → AutoTuner trtllm fused_moe
- **11:17:10 "fired up and ready"** (总 10m58s, 含全部首启 JIT)
- 11:17:37 /generate 探针: "The capital of France is" → " Paris." 16 tok, 0.38s, GENERATE_OK

## 复现要点 (下次冷启动)
1. pip 索引用 tencent; 2. 装完后强制 `pip install nvidia-cuda-nvcc~=13.0.0 nvidia-cuda-crt~=13.0.0 nvidia-nvjitlink~=13.0.0 nvidia-nvvm==13.0.88` (防浮动拉到 13.4);
3. cu13/lib 补三个 .so 符号链接; 4. launch 前 export CUDA_HOME/CUDA_PATH/PATH/LD_LIBRARY_PATH/LIBRARY_PATH (见 smoke_dsv4.sh FIX-ENV1/2);
5. JIT 缓存 (/root/.cache/sglang/deep_gemm, /root/.cache/sglang/.cache/flashinfer) 保留则重启 ~11min, 其中编译不再重复。

## 遗留观察 (Phase 2 输入)
- config: 43 层全部为 DeepseekV2MoE, 均可参与重均衡 (更正 2026-10-22: num_hash_layers=3 实为 Engram 记忆模块 —— EngramHasher n-gram 哈希嵌入 + wkv, deepseek_v4.py:2826-2828/4212-4217, 并非静态专家路由层; 早前"3 个 hash 层不可迁移"的判断作废), 256 routed + 1 shared, top-6
- 专家路径: MXFP4 → FlashInfer SM90 CUTLASS **W4A16** (humming 未启用)
- deep_gemm/flashinfer 均为 JIT 缓存敏感; pb_oeplb 移植后首启时间需重新预算

# EP-stack feasibility (Phase 1b, 2026-10-22 13:36Z CLOSED)

目标: identity EP8+DP8+deepep boot on 0.5.22 (boot_v4_ep_test.sh)。10 轮 boot，
两类根因，均已闭环；**test10 PASS**: ready 3m50s，8/8 capture 无 OOM，
/generate " Paris." rc=0，server left RUNNING port 30000。

## 根因 1: NVSHMEM (boots 2/4/5/6/7, FIX-NVSHMEM3+4)
- sgl-deep-ep 0.1.2.post1 (deep_ep 2.1.0 cu130) 强制 nvshmem-cu13 **3.4.5**（3.3.20 host → device lib version mismatch → exit 255）。
- 3.4.5-cu13 **无视所有传统 transport env**（REMOTE_TRANSPORT / IB_ENABLE_IBGDA 名字不存在；IBGDA_SUPPORT=0 parsed-but-inert，8-rank harness 证实）；IBGDA 在本 eRDMA 类 fabric 上 DCT/AH **致命**（旧 cu12 3.3.20 只是容忍为 WARN——冻结 campaign 的"生效"env 其实同样是 inert 的）。
- **修复**: `NVSHMEM_HCA_LIST=mlx5_bogus` → 无可用 IB 设备 → IBGDA/IBRC 优雅跳过 → 纯 P2P/NVLink。单节点下这正是冻结 campaign 的等效传输配置。harness 8/8 BUFFER_OK+BARRIER_OK；test8 起全程验证（含推理）。

## 根因 2: FP4 专家 × SM90 (boot 8, FIX-FP4DEQUANT) + capture OOM (boot 9, FIX-MEMFRAC)
- deepep a2a + auto runner → **deep_gemm masked GEMM**；packed-FP4 断言 arch 10|12（deepgemm layout.hpp:73），H20=SM90 → capture 死。flashinfer_mxfp4 (SM90 CUTLASS W4A16) 只在 `a2a==none` 时 auto 选中（model_overrides/deepseek_v4.py:77）——identity smoke 因此没事。
- **修复**: `SGLANG_DSV4_FP4_DEQUANT=1`（官方 Hopper 路径）：加载时 e2m1→e4m3 逐专家 cast（fp8.py:1755），走 `grouped_gemm_nt_f8f8bf16_masked`（SM90 原生）。约束 runner=auto（fp8.py:420 assert）。权重 29.90→**42.28GB**/GPU（+12.4GB=专家 2x）。dequant 的 warning_once 日志被抑制，以内存量+masked-fp8 通路为证。
- 0.85 下 capture headroom 不足（bs=32 graph+deepep masked workspace 吃 12.1GB，bs=24 OOM）→ **mem-fraction 0.85→0.80**（capture avail 13.81→18.51GB，floor 6.31GB，全过）。

## EP 复现要点（下次冷启动，在 Phase-1 五条之上追加）
6. boot 脚本已含全部 fix（FIX-NVSHMEM3+4 + FIX-FP4DEQUANT + mem-fraction 0.80）；
7. masked FP8 kernel JIT 首启会多花时间（bs=32 capture ~167s 冷、~40s 暖）；DG cache 暖则整 boot ~4min；
8. PB 臂（boot_v4_pb_p.sh）同样已带三 fix + PB_OEPLB_*/OEPLB_V21_* frozen knobs + `--ep-dispatch-algorithm static`。

# PB-arm feasibility (Phase 1c, 2026-10-22 13:55Z CLOSED)

boot_v4_pb_p.sh (pb_oeplb v2.1 frozen knobs, static dispatch) + run_e2_bench.sh:
- **boot1 FAIL_AFTER_READY** (mem-fraction 0.80): first swap window's P2P subgroup
  creation = NCCL comm-init storm; NCCL raw cudaMalloc (bypasses torch allocator)
  "Failed to CUDA calloc 10MB" at bench peak (96.3/97.8GB used) -> rank4 ctx
  poisoned (illegal access on empty_cache retry) -> abort. Forensics kept in
  bench_e2_p1/invalid_boot1/ (controller window#1 JSON fully formed pre-death =
  port logic validated independent of the memory failure).
- **FIX-MEMFRAC2: 0.75 for PB arm** (+4.8GB raw headroom; capture avail 23.23GB).
  boot2 PASS: ready 3m46s; first window 744-869ms (comm creation), steady-state
  48-56ms/300ops; 24/24 windows committed, ~6900 live swaps, 0 errors.
- **E2 bench PASS**: ok=4082/4082 errors=0, 116.9s, tps 34.9, ttft_p95 9.99s
  (pure-prefill prover stream). VERIFY-WEIGHT-MOVE mirror checksums CHANGED=True;
  DIAG-2 mirror symmetry 56/56 pairs 57328==57328 sends/recvs, 0 asymmetries.
- Rule of thumb: identity arm fits 0.80; PB arm needs <=0.75 on this host because
  per-peer NCCL comm buffers allocate OUTSIDE the torch caching allocator at the
  first swap window under peak traffic.

## End-to-end matrix r1 + diagnostics (Phase 1d, 2026-10-22)

Author directive: "我要看端到端效果" → formal I-arm (identity) vs P-arm
(pb_oeplb v2.1) same-protocol comparison on DSV4, campaign-frozen bench
protocol (bench_client_v2, CONC=256, frozen stream shas, fresh boot per cell).

### Matrix r1 (scripts/run_ip_matrix.sh, 14:36-15:07Z, all cells rc=0, 0 errors)

| stream | arm | ok | T_s | tps | ttft_p50 | ttft_p95 | tpot_mean |
|---|---|---|---|---|---|---|---|
| e2 | I | 4082 | 96.0 | 42.5 | 5914 | 6777 | - |
| e2 | P | 4082 | 116.9 | 34.9 | 6993 | 9988 | - |
| e3 | I | 8192 | 184.0 | 44.5 | 5738 | 6823 | - |
| e3 | P | 8192 | 215.3 | 38.1 | 6578 | 8604 | - |
| e4 | I | 2048 | 81.2 | 3227.1 | 3241 | 5841 | 54.1 |
| e4 | P | 2048 | 97.2 | 2696.6 | 4106 | 8736 | 62.1 |

P-vs-I r1: tps -17.9% (e2) / -14.4% (e3) / -16.4% (e4); ttft_p95 +47.4/+26.1/+49.6%.
Old-campaign calibration (Qwen3 main_table.csv medians, n=8/arm/cond):
P-I tps = +14.6% (E2), +2.2% (E3), -3.6% (E4) → DSV4 REVERSES the sign on the
prefill streams; P's old win vs I does not transfer with frozen v2.1 knobs.
(P's old headline win was vs E arm: E4 tps E=1000 vs P=2767.)

### Diagnostics (scripts/run_diag_cells.sh, 15:15-15:26Z) — E2 wall decomposition

| config | T_s | delta | share |
|---|---|---|---|
| I (default dispatch) | 96.0 | — | — |
| I + --ep-dispatch-algorithm static (Diag-B) | 101.5 | +5.5s | ~26% |
| P noswap, MAX_SWAP_*=0 (Diag-A; 24 windows, 0 swaps verified) | 105.7 | +9.7s cum | instr/windows ~20% |
| P full (25 swaps) | 116.9 | +20.9s cum | swap effects ~54% |

- Controller/window protocol cost is NEGLIGIBLE: c_window_ms sums 75-246ms/run;
  swap mechanical begin() ~30-50ms/window (TIMING lines, n_ops≈148).
- Swap-attributable wall (~11.2s / 25 swaps ≈ 450ms/swap) >> mechanical cost →
  post-swap placements are NET WORSE for DSV4 traffic. Leading hypothesis:
  frozen v2.1 knobs (OEPLB_V21_KAPPA/z_lo/z_hi/theta_r_*) were FITTED ON
  QWEN3 traffic statistics; DSV4 (256 routed experts, group-limited topk,
  shared expert, FP8-dequant sizes) produces different h_rate/z signals →
  swap decisions degrade balance instead of improving it.
- static dispatch flag is a real per-token cost on 0.5.22 (~5.5%/wall on E2),
  independent of pb_oeplb. Old campaign: I arm ran DEFAULT dispatch, P arm
  auto-forced static — same asymmetry preserved here for fidelity.

### Implication for Phase 2 (author gate)
The port is functionally correct (0 errors, ~7k swaps cumulative, checksums +
mirror symmetry green) but the frozen knob set does not transfer. A DSV4
recalibration (re-fit z/theta knobs on DSV4 traffic via calib streams, or
threshold sweep) is the natural Phase 2; without it, identity is the faster
production config on this stack.
r2 (median confirmation, 6 cells incl. fresh p_e2) launched 15:28Z.

## Root cause of the -17~19% P-vs-I penalty (Phase 2 forensics, 2026-10-22)

Author challenge: "-17~19% 开销不合理，哪一步没做好？是否真的迁移？逻辑物理是否匹配？"

### Verdict on the three suspected failure modes
1. **迁移是否真的发生？——是 (migration IS real).** 24 swap plans committed in the
   96s E2 run (p2l_version 0→23), every plan ~300 ops (MAX_TOTAL_OPS saturated),
   57,344 expert-slot moves total; 48/48 [VERIFY-WEIGHT-MOVE] checksums CHANGED=True;
   weights move via async_swapper P2P batch_isend_irecv (begin() median 70ms).
2. **逻辑/物理位置是否匹配？——匹配 (routing DOES follow the metadata).**
   ExpertLocationMetadata.update() mutates the GLOBAL object in place (including
   logical_to_rank_dispatch_physical_map); ExpertLocationDispatchInfo.init_new()
   re-reads it per-forward; both the normal path (topk.py _post_process_topk_ids,
   CUDA branch ungated) and the hash path (hash_topk.py:271/:294) apply
   topk_ids_logical_to_physical. The 0.5.22 _eplb_remap_enabled() gate only affects
   the HIP branch. Function is preserved — the penalty is NOT wrong-output garbage.
3. **哪一步没做好？——layer attribution in the load recorder (THE root cause).**

### The actual bug: round-robin layer attribution scramble
- DSV4 has num_hash_layers=3: layers 0,1,2 route via HashTopK and NEVER call
  select_experts, so the single record hook (topk.py:2896 →
  controller.record_next_layer) fires only 40×/forward (layers 3..42).
- The frozen controller attributes each record to a slot by round-robin:
  `layer_id = _layer_offset + (_layer_counter % _num_moe_layers)` (controller.py:404)
  with _num_moe_layers=43 (all 43 layers hold routed weights → dense-offset guard
  at :390-400 sees offset=0 and cannot detect hash layers).
- 40 records per forward against 43 slots → attribution rotates 3 slots every
  forward (gcd(3,43)=1) → every layer's window aggregate is a mixture of ~13
  different true layers, and consecutive windows' mixtures are disjoint.
- Proof: prof_record_calls/recorded_steps = 14760/369 = **40.0 exactly**
  (old Qwen3 campaign: 15792/168 = **94.0 exactly** — aligned, all layers recorded).

### Consequences (all observations explained)
- s_change pinned at 0.30-0.38 (fixed point of the 3-step rotation), never ~1e-3.
- z = s_change×max(n_fresh/L,1)/κ pinned at 39-52 ≫ z_lo=22.64 → state machine
  never reaches QUIET (old E2: 1-2 swaps → z 0.1-0.3 → QUIET).
- Planner builds max-size plans from wrong-layer stats: DIAG predicts
  avg_ratio 1.80→1.21 every window; measured ratio_before stays 1.67-1.80 for all
  24 windows — predicted gain never materializes → re-plan forever.
- Defer gate needs len(plan)<16; plans are ~300 ops → never defers → unbounded churn.
- Cost: ~450ms wall impact per swap window × 24 windows ≈ 11s ≈ the 54% swap-effect
  share of the E2 penalty; plus static-dispatch flag (~26%) and instrumentation
  (~20%) shares. Hence -17~19% tps and +37~66% ttft_p95.
- Counter non-bug: jsonl sp_swaps_started is per-window (re-zeroed at
  controller.py:1906), =1 every window; sp_run_end swaps_started_total=25 counts all.

### Fix (host-patch level; frozen pb_oeplb package stays byte-identical)
At the single record call site topk.py:2896, set
`_pb_ctrl._layer_counter = layer_id - _layer_offset` before record_next_layer()
so slot == true layer id. Post-fix slots 0-2 stay permanently zero — verified safe:
- rebalancer compute_ratio: zero row → ratio 0.0 < 1.02 threshold → never planned;
- change_score: valid-mask (rowsum>0) excludes zero rows from s_change;
- z: n_rows=43 vs 40 true rows = harmless ~7% n_layer rescale;
- total_tokens: write-only telemetry (:427 increments only at layer_id==0).
Expected after fix: s_change collapses to ~1e-3, z < z_lo → QUIET reachable,
swaps converge to 1-2 like the old campaign, churn penalty disappears.
Validation: E2-slice live check (s_change collapse + QUIET + near-zero swaps),
then full I-vs-P matrix re-run. Owner: fork agent (in progress).

## Phase 1e — Root-cause forensics + layer-alignment fix (2026-10-22, author challenge)

**Author challenge**: -17~19% P-vs-I overhead implausible; demanded root cause
(which porting step failed / is migration real / logical-physical mismatch).

**ROOT CAUSE — layer-attribution scramble (43 vs 40).**
DSV4 config: num_hidden_layers=43, num_hash_layers=3. Layers 0-2 route via
HashTopK (deepseek_v2.py:679-680 `is_hash = layer_id < n_hash_layers`;
:733 `self.topk = HashTopK(...)`) and NEVER call select_experts -> the frozen
recorder hook fires 40x/forward (proof: prof_record_calls/recorded_steps =
14760/369 = 40.0 exact; old Qwen3 = 15792/168 = 94.0 exact, aligned).
Frozen controller attributes layers by round-robin `slot = _layer_counter %
_num_moe_layers` with `_num_moe_layers = num_layers - _layer_offset = 43 - 0`
(boot log "Init v0.11: layers=43"; NO "detected leading dense" line: hash
layers DO own routed weights so routed_experts_weights_of_layer keys = 0..42,
min=0 — the dense-offset BUGFIX cannot see them). Attribution therefore
ROTATES 3 slots per recorded forward (40 = -3 mod 43, gcd(3,43)=1):
- each slot's window aggregate = mixture of a step-3 residue progression of
  ~13 true layers; consecutive windows' mixtures PROVABLY DISJOINT
  (k-m = 13 mod 43 unsolvable in range) -> s_change pinned 0.30-0.38
  (Qwen3: 7.4e-4), z pinned 39-52 (Qwen3: 0.1-0.3) -> never QUIET;
- planner builds plans from wrong-layer stats: PB-OEPLB-DIAG predicted
  avg_ratio 1.80 -> 1.21 every window, next-window measured stayed 1.67-1.80
  (gain never materialized, 24/24 windows); defer gate needs len(plan)<16,
  never fires with saturated 300-op plans -> permanent churn: 24 plans x ~300
  ops = 57,344 swap ops per 96s run, batch_isend_irecv up to 997ms/window;
- slots 0-2 received garbage stats too -> plans swapped HASH layers
  (VERIFY-WEIGHT-MOVE layer=2 CHANGED=True) — functionally safe (HashTopK
  consumes p2l: hash_topk.py:271/294; weights+map move atomically) but noise.

**Answers to the author's three questions**:
1. Which step wasn't done right: the port's layer-attribution assumption
   ("every layer in [offset, num_layers) calls select_experts once per
   forward") is false on DSV4 (3 hash layers). Weights/remap/dispatch porting
   itself is correct.
2. Migration IS real: 48/48 checksummed weight moves, p2l commits in-place,
   post-fix h_rate converged 0.0202 -> 0.0117 after 2 plans (real imbalance
   really removed), cross-arm probes token-identical after ~6x300-op relayouts.
3. Logical-physical mismatch: expert-level mapping was CONSISTENT (F1
   invariant held, version tripwire never fired). The mismatch was
   STAT-LAYER vs MODEL-LAYER (logical layer positions scrambled), not
   expert positions.

**fp_test1 reinterpretation**: verdict FUNCTION_CHANGED was confounded —
v2p1 jsonl shows swaps committed on EMPTY probe-phase windows (stale plans;
p2l_v 0->6 before traffic) so "noise floor" P_pre1-vs-P_pre2 had real weight
moves; probe0 flips at near-tie tokens (18-22) in noise floor too; meanwhile
I_pre1 vs P_pre1 token-identical (dlogprob <=0.36 = static-vs-default
dispatch numerics). Clean re-test (fp2) running: Arm A QUIET determinism,
Arm B forced swaps (Z_LO=1e-5 env) across real commits.

**FIX (host-patch level, frozen pb_oeplb byte-identical)**: topk.py hook
block (my ported region) sets `_pb_ctrl._layer_counter = layer_id -
_layer_offset` before record_next_layer -> slot == true layer id; hash-layer
slots 0-2 stay permanently zero and are skipped by all consumers (rebalancer
ratio 0<1.02; change_score valid-mask; total_tokens write-only). sha_pre
914e1ed5..., sha_post 354b0482..., backup scripts/host_patches/.

**VERIFICATION (fixcheck_p_e2, full frozen E2 stream)**: s_change 0.0006-0.002,
z 0.07-0.29, swaps_started_total=2 with real convergence, TIMING begin 16
(vs 192), tps 34.5 -> 39.2, total_time 118.3s -> 104.3s; P-vs-I gap
-19.1% -> -8.0% (I 42.6tps/95.7s). Residual fully explained by diag
decomposition: static-dispatch flag (+5.5s, faithful to old-campaign P-arm
protocol) + instrumentation (+4.2s) ~ +9.7s expected vs +8.6s observed.

**In flight**: fp2 (preservation) + matrix r3/r4 (p_e2/e3/e4 x2 + i_e2 spot).

## Phase 2: layer-attribution fix (2026-10-22)
Root cause of the -17~19% P-vs-I penalty: DSV4 num_hash_layers=3 -> record hook
fires 40x/forward but frozen controller round-robin assumes 43 slots -> attribution
rotates 3 slots/forward (gcd(3,43)=1) -> per-layer stats are disjoint mixtures ->
s_change pinned ~0.33, z pinned 39-52 (never QUIET), 300-op plans churn every window.
Fix (host patch, frozen pb_oeplb byte-identical): topk.py hook sets
_layer_counter = layer_id - _layer_offset before record_next_layer (sha_post=354b0482).
Fixcheck E2: swaps 24->2, z 39-52->0.06-0.29 (old-campaign scale), s_change 0.33->0.0008,
h_rate 0.095->0.011 (theta_r_act vicinity), steady state grow_blocked_headroom,
tps 34.5->39.2 (+13.6%), remaining gap vs I (-8.0%) = static-dispatch flag (~5.7%)
+ instrumentation (~4.4%) fixed 0.5.22 overheads (matches decomposition model).
See fix_comparison_table.md (full matrix) + fp_fix_art/ (function preservation retest).

## Phase 1f — r3b unpatched-baseline cells + final fix table (2026-10-22 ~17:35Z)

r3b (approved as unpatched baseline; no /opt/dsv4_env edits during run):
- p_fix_e3_r2: tps 40.7, wall 201.1s, 46 windows, 8 swaps, z_med 2.93, z_max 122.0
  (one genuine alert episode; 3 confirmed_grow, 29 grow_blocked_headroom), h_rate_med 0.0152.
- p_fix_e4_r2: tps 2811.1, wall 93.3s, 75 windows, 9 swaps, z_max 15.7 < z_lo 22.64
  (QUIET throughout), 60 init windows (53 empty + ref-zeroing cascade, see
  STATS_PATH_TRACE.md), h_rate_med 0.0217.
- i_e2_r3 (I-arm reproducibility spot): tps 42.2 vs r1 42.5 / r2 42.8 — within 1%.

FINAL matrix (medians, fix_table_final_r3b.md):
| stream | I | P_broken | P_fix | P_fix-vs-I |
|---|---|---|---|---|
| e2 | 42.6 | 34.5 | 39.8 | -6.7% |
| e3 | 44.9 | 37.2 | 39.7 | -11.7% |
| e4 | 3215.1 | 2624.5 | 2755.3 | -14.3% |
Broken-arm pathology (saturated 300-op plans every window, 57k swap ops, z pinned
39-52, s_change 0.21-0.33) is GONE: post-fix swaps 2-13/run, z_med 0.10-2.93,
s_change_med 0.0008-0.0586 (stream-dependent, genuine).
Residual gap decomposition (OVERHEAD_ANALYSIS.md + r3b): static-dispatch flag cost
scales PER-FORWARD (E4 1210 forwards/93s vs E2 384/104s -> -14.3% vs -6.7%);
instrumentation scales PER-BOUNDARY (E4 75 boundaries); E3 adds real swap work
(13 ops-bearing swaps in r1). Phase 2 (patches A/B + Plan A+ micro-backport per
planb_assessment.md) targets 2.5-3.5 pts of this; Plan B (DSV4 on 0.5.6 clone)
assessed INFEASIBLE (kernel ABI wall: nvcc>=12.9 + torch2.14/cu130 binaries).
Stats-path trace deliverable: STATS_PATH_TRACE.md (z/h_rate formulas, N_MIN gating,
E4 init-cascade mechanism, swap-issuance knobs, s_change semantics, w1 h_rate).
GPU activity STOPPED after r3b_done per coordinator hard rule; no new cells.

## Phase 3: overhead patches A+B (fixab) — 2026-10-22 18:05Z

Applied (host-side, frozen pb_oeplb untouched; backups .bak_pre_a/.bak_pre_b in place):
- Patch A: expert_location_dispatch.py — per-(layer_id, algo) dispatch-info cache keyed on metadata
  object identity. sha 5a9c78047106987f -> 5511f6546b6ad444.
- Patch B: topk.py — module-level controller handle cache + pre-gate replicating frozen record gates
  1&3 (batch flag, min_record_tokens). sha 354b04824b2b73f5 -> 84a138ccf5730e74.
  Layer-align fix remains inside the patched region (pre-set _layer_counter = layer_id - offset).

Results (full frozen protocol, CONC=256):
- e2 P_fixab 40.9 tps (P_fix 39.8, +2.8%; vs I 42.5 -> -3.8%) errors=0 59w/5sw z_med 0.095
- e3 P_fixab 41.4 tps (P_fix 39.7, +4.4%; vs I 44.9 -> -7.8%) errors=0 74w/17sw z_med 2.642
- e4 P_fixab 2763.1 tps (P_fix 2755.3, +0.3%; vs I 3215.1 -> -14.1%) errors=0 75w/11sw z_med 0.663
  (e4 decode-heavy: dispatch transform inside CUDA graphs + fewer record-eligible batches ->
  patches A/B savings structurally smaller; residual = instrumentation floor + real swap work)

Function preservation (e2 cell gate, PASS=true):
- fixab pre AND post probes token-id identical vs I refs and vs each other across 5 committed swaps.
- Pfix_pre vs fixab_pre logprobs BITWISE identical (dpre=0.0000): patches introduce zero numerical
  change in steady state; cached views follow in-place metadata commits correctly.
- Cross-arm/cross-boot dpre <= 0.35 = regrouping-noise scale (control Pfix pre-vs-post: 0.24).

Process note (honesty): first gate run FAILed on a script bug — gate compared x[0] (logprob floats)
instead of x[1] (token ids) from out_token_logprobs. Recomputed with token ids: PASS. run_fixab_verify.sh
corrected (line 54). Ledger: fixab_gate rc=2 (false alarm) + fixab_gate_recomputed PASS=true.

Final configuration: sglang 0.5.22 + layer-align fix + patches A/B. Plan B (old-sglang adaptation)
remains rejected per planb_assessment.md. Table: fixab_comparison.md.
