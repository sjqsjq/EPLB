# 死区/上限理论 A100 实测验证总纲(2026-09-28~29)

> 对应论文 §2.4 铰链模型 T(r)=T_flat·[1+β·max(0,r−r_k)]、EP 幂律 r_k−1=c·EP^1.52、
> 死区阈值处方(REPRODUCE.md §2/§3/§4)在 A100(BF16/Triton/NCCL,非-DeepEP)上的直接验证。
> 结论先行:**铰链形状、上限公式、幂律系数(GEMM主导硬件间)全部复现;"r_k 硬件无关、β 硬件相关"
> 的文档假设被修正——r_k 由 workload/EP/协议的时间份额比主导(硬件弱相关),β 近似硬件不变量。**

## 1. 实验矩阵(全部本机实测,脚本/数据见本目录与 results 前缀)

| # | 实验 | 配置 | 产物 |
|---|---|---|---|
| step0 | A100 原生路由计数 | identity+recorder(stat), prover head1024, 15.5亿token | counts235b_a100.json;**与 H20 FP8 计数余弦 0.9999,identity r 1.7163 vs 1.7154** |
| step1 | T(r) 扫描 11 点 | 235B/EP8, 256tok burst N=256×5run(丢r1) | `_scanA100_*`;两套拟合(含/不含 conc) |
| B1 | 协议鲁棒性 6 点 | 同上但 494tok | `_scanL494_*` |
| B2 | chunk 机制判别 7 点 | 同上但 --chunked-prefill-size 4096 | `_scanC4K_*` |
| step3 | η(burst)3 臂 | 固定W=16: (1.02,无预算)/(1.02,256)/(1.06,300) | **churn臂崩溃实录** + 2 有效臂 |
| step3b | η(持续)2 臂 | N=2048, W=4: churn-lite(1.02,300) vs gated(1.20,300) | `_eta2*`;无分离+机制解释 |
| step4 | EP 幂律 | **57B**(qwen2_moe 同款 dispatch-info 补丁已打), EP4(tp4)/EP2(tp2), L256 workload | `_scan57E4_*`,`_scan57E2_*`;**EP8 在 A100 物理不可行(28头%8≠0,H20 靠 dp-attention,A100 禁用)** |

## 2. 铰链拟合总表(A100)

| 配置 | T_flat | 斜率 | **r_k** | **β** | R² | holdout 最大误差 |
|---|---|---|---|---|---|---|
| 235B/EP8, 256tok, chunk8192(稳健口径,conc除外) | 5.46s | 1.80 | **1.053** | 0.330 | 0.9972 | id +0.9% / r130 −0.4% / conc −5.0% |
| 235B/EP8, 256tok(含conc全域) | 5.46s | 1.98 | 1.070 | 0.363 | 0.9998 | id +2.2% |
| 235B/EP8, **494tok**(B1) | 10.68s | 3.35 | **1.133** | 0.314 | 0.9998(铰链/直线 RSS 比 **242×**) | r130 −0.6% / id +0.9% |
| 235B/EP8, 256tok, **chunk4096**(B2) | 5.95s | 1.57 | **1.086** | 0.264 | 0.9960 | id +0.9% |
| **57B/EP4**, L256 | 4.26s | 3.30 | **1.026** | 0.775 | 0.9659 | r108 +1.1% / id +2.9% |
| **57B/EP2**, L256 | 7.22s | 4.36 | **1.016** | 0.604 | 0.9808 | id −0.1% / r107 +0.4% |

## 3. 四个理论判定

### 3.1 死区/铰链形状:✅ 三硬件成立
每套配置铰链 RSS 均显著优于直线(3.5×~242×),平坦区实测存在(235B: T(1.000)=T(1.020)=T(1.060)±0.4%),holdout ≤1%(235B)。conc(r=4.57)点显示 **r>~1.8 后线性外推低估 ~5%**(曲线上凸),模型适用域应标注 r≤1.8(真实负载自然上界内)。

### 3.2 r_k 的主导变量是"不敏感/敏感时间份额比",非硬件本身
$$r_k-1 \approx \kappa\,(I/S)^{\alpha},\quad I=comm+attn(L)+weight\_load+launch,\quad \beta \approx S/(I+S)$$
- **workload(最强)**:prompt 256→494tok,r_k−1 从 0.053~0.070 → 0.133(×2.0-2.5 ≈ 长度比 1.93)——attention(不敏感项)单边膨胀;
- **协议**:chunk 8192→4096,每-forward 固定开销翻倍,r_k +0.016~0.033(BLOCK_M 量化假说降级为次因:预测 ×2 实测 ×1.3);
- **EP**:57B EP2→EP4:1.016→1.026(comm 份额↑),方向与 H20 幂律一致;
- **硬件(最弱,GEMM主导对之间)**:H20 定律直接外推 A100:EP2 预测 1.0117/实测 1.016(+0.004),EP4 预测 1.0336/实测 1.026(−0.008)——**系数跨 H20/A100 可迁移(±0.008≈拟合噪声)**;235B@L~500:A100 1.133 vs H20 1.093(+3.7%,A100 attention 相对更贵,HBM 2 vs 4TB/s)。
- **硬件(非对称时剧烈)**:H800(DeepEP 通信主导,comm/GEMM=72/13)实测 r_k=1.7-2.3、β=0.03(用户 H800 实验,待归档)——I/S 爆炸,幂律系数失效点恰在通信栈接管处,**验证比值模型的边界条件**。

### 3.3 β 是硬件鲁棒量(文档假设装反)
235B/EP8:A100 0.330~0.363 vs H20 0.352(±3~8%);文档对 A100 拍的 0.284 被证伪。β 随模型/workload 按份额移动(57B@L256 attention 极小 → β 0.60~0.78),与"β=敏感份额"定义自洽。Δ_max(A100@256tok)=+21.1%(稳健口径)~+23.4%(全域),实测 DataForest +21.8% 落带内;文档预测 18.3% 偏低 3-5pp。

### 3.4 η/死区阈值的工程价值在 A100 换了一种表现
- burst 与持续负载下,thr=1.02 churn 臂与 thr≥r_k 门控臂**无吞吐分离**(churn 15.6 vs gated 15.4):规划器在均衡点附近自我限流(每窗提案<min_swap_ops=8 即静默),H20 57B/L256 的"死区内持续 swap"(η=26%)状态在 A100 235B/prover 上维持不住;
- **但死区+预算是稳定性必需**:thr=1.02+无预算 → 单窗 1590-op 巨批(≈60GB 同步 P2P)→ NCCL collective 超时 → 全服务器崩溃(step3 实录,H20 异步 DeepEP 搬运下同配置只是变慢)。A100 同步 P2P 路径上,max-total-ops 预算是安全装置而非纯优化;
- 门控阈值需加**测量噪声余量**:窗口测得 ratio(~1.13-1.18,小窗采样噪声)恒高于真实放置 r(~1.07),thr=1.06 关不住门(实测 2 窗),thr=1.20 才静默(1 窗)——处方应从 thr=r_k 修正为 thr=r_k+margin(W,burstiness)。

## 4. 对既有文档/代码的修正清单
1. `REPRODUCE_BASELINES.md` §8.3 表:A100 行(r_k 1.093/β 0.284/Δmax 18.3%)→ 实测替换;H800 行预测(r_k~1.0/β 0.048)→ 用户实测 r_k 1.7-2.3/β 0.03(方向对、r_k 数值错);
2. `bound_curve.py` docstring "β and r_k belong to (model, GPU-count); r belongs to dataset" → **被 B1 证伪**:r_k 同属 dataset(prompt 长度)与协议(chunk);建议升级为 (model, EP, 通信栈, L, chunk) 的份额比模型;
3. REPRODUCE.md §3 EP 幂律:补注"GEMM 主导硬件间系数可迁移(A100 ±0.008);通信主导栈(H800/DeepEP)失效,需按 I/S 重估";
4. 死区阈值处方:thr = r_k + 噪声余量;A100 类同步 P2P 路径必须配 max-total-ops 预算(稳定性);
5. qwen2_moe.py `_forward_router_experts` 漏传 dispatch info(与 qwen3_moe 同类 bug)——已修复并写入两份 patch_sglang.py。

## 5. 产物索引
- 放置: `placements/plcA100_*.json`(11), `plc57E4_*.json`(7), `plc57E2_*.json`(7)
- 计数: `data/counts235b_a100.json`(A100 原生), 复用 `repro/counts57b.json`(L256,identity r 1.2177 与 H20 一致)
- 结果: `benchmarks/results/_scanA100_*`(44), `_scanL494_*`(24), `_scanC4K_*`(28), `_scan57E4_*`/`_scan57E2_*`(56), `_eta*`(step3/3b)
- 拟合: `logs/fit_f3_a100.txt`, `fit_f3_L494.txt`, `fit_f3_C4K.txt`, `fit_f3_57E4.txt`, `fit_f3_57E2.txt`
- 脚本: step0_record_counts / step1_scan / step2_phases / step3_eta / step3b_eta_sustained / step4_ep_law + launch_scan{,2,57}_a100.sh + launch_eta{,2}_a100.sh + bench_scan.py
- server 日志: /workspace/logs/server_a100_{scan_*,sust_*,eta_*,57b_*}.log(含 step3 崩溃现场)
