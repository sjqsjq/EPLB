
## 2026-08-11 13:29:47 事故：site-packages/sglang 被整树覆盖，PB-OEPLB 补丁丢失
现象：`launch_server.py: error: unrecognized arguments: --enable-pb-oeplb ...`
原因：整个 `/opt/conda/lib/python3.11/site-packages/sglang/` 在 13:29:47 被同一秒
写入（pip 重装/层刷新），server_args.py 4822→4680 行，OEPLB 的 CLI 参数被抹掉。
`srt/managers/pb_oeplb/` 目录因不属于 wheel 而幸存，所以只是参数注册没了。
修复（已执行）：
  cp sglang_patches_backup/{server_args,model_runner,topk}.py.oeplb → site-packages 对应路径
  find $SP -name __pycache__ -type d -exec rm -rf {} +
  python3 -m sglang.launch_server --help | grep -c pb-oeplb   # 应为 39
校验：三个 .oeplb 相对 pristine 只有增行（142/33/6 add, 仅 model_runner 1 del），
base 版本一致，可直接覆盖。被覆盖前的 pristine 副本存于 /workspace/logs/pristine_1329/。
影响面：driver4(L512, 12:57-13:21)、driver5(13:22-13:24) 均在事故前完成，未受影响；
driver6 对照组(13:35-13:38) 用的是 base_g8（不需要 OEPLB 参数），跑在 pristine 上，
其输出与 driver5 patched build 的 swap 前输出逐字节相同 → 两个 build 数值等价，对照仍有效。

## 2026-08-11 15:05 driver10 trial-1 崩溃：swap 临时缓冲区 O(len(plan)) 内存导致 NCCL OOM → 全局挂死

**现象**：`server57b_t1.log` 14:51:28 第一次决策（135 ops, ratio 1.218→1.017）后，
DP2/DP5/DP6 三个 rank 抛出
`ncclUnhandledCudaError ... Failed to CUDA calloc 10485760 bytes`
（栈：controller.py:828 `_decide_and_begin_swap` → async_swapper.py:118 `batch_isend_irecv`
→ `_coalescing_manager._end_coalescing`）。异常被 controller 的
`except Exception: logger.error(...)` 吞掉，另外 5 个 rank 完成了 P2P 传输
→ rank 之间权重与集合通信状态不一致 → 后续 forward 全部挂住
→ 15:05:43 八个 rank 同时 `Watchdog timeout (600.0)`，进程退出，GPU 释放。
benchmark 客户端因每请求 600s 超时而空转 50 分钟（手动 kill）。

**根因**（不是 NCCL 贪心，而是我们自己的内存放大）：
`begin()` 旧实现在发起任何传输**之前**为 plan 里**每一个** op 分配接收缓冲
`temp = [torch.empty_like(w[local]) for w in weights]`。57B 每专家 fp8 权重
2560×3584 ≈ 9.2 MB × 3 (gate/up/down) + scale ≈ 27.5 MB/op；135 ops ≈ **1.2-1.9 GB**
瞬时额外显存。叠加 `--mem-fraction-static 0.85` 后剩余空间不足，NCCL 连自己
10 MB 的 P2P channel buffer 都 calloc 不出来。plan 越大越容易触发 → 这是一个
**随不均衡度/首次决策规模放大的可靠性 bug**。

**为什么之前没暴露**：以往首次决策发生在 16K 请求压测中；driver10 首次把
corpus_probe 提到 200 条 prompt，顺序请求积累够了决策窗口，使首次决策（最大的
plan，135 ops）落在探针阶段。规模是触发条件，时机只是让它更容易命中。

**修复**（已应用，三处同步：site-packages / OEPLB/src / benchmark/sglang_src）：
`begin()` 改为**分块执行**，`_SWAP_CHUNK`（环境变量 `PB_OEPLB_SWAP_CHUNK`，默认 16）
一块，每块 alloc → batch_isend_irecv → wait → copy back → `del` 释放。
瞬时额外显存从 O(len(plan)) 降到 O(chunk)（16 ops ≈ 440 MB）。
所有 rank 用同一个 plan、同一个分块顺序迭代，因此仍然保持 lockstep。
另加一次 `empty_cache()` 后重试。备份：`*.pre_chunk_bak`。

**受影响数据**：driver10 trial 1 作废（decisions=0，实际是 5/8 rank 做了半个 swap）。
trial 2/3 使用修复后的代码。

### 15:56 修正上面的修复：分块会破坏 lockstep，已回退

分块版本在 trial 2 立刻暴露了第二个 bug：rank 0/1 打印了完整的
`n_ops=132 n_chunks=9`，rank 2/5/6 卡死，watchdog 报告一部分 rank 停在
`SeqNum=4, OpType=COALESCED`、另一部分停在 `SeqNum=5` → **coalesced work 的
序号在 rank 之间分叉**。原因：一个 chunk 内某个 rank 可能不拥有任何被交换的槽位，
于是它的 `p2p_ops` 为空、**整个 `batch_isend_irecv` 被跳过**；而
ProcessGroupNCCL 按 process group 给每个 coalesced work 编号，参与度不一致 →
序号分叉 → 下一个集合通信死锁。单批次版本之所以一直安全，正是因为每个 rank
恰好参与一次。**结论：plan 必须单批次传输，不能分块。**

**最终修复**（已同步三处副本，`async_swapper.py.chunked_bad` 保留错误版本）：
1. 回退到单次 `batch_isend_irecv`（恢复 lockstep，与论文既有测量语义一致）；
2. 在 NCCL 调用**之前**插入一次 `torch.cuda.empty_cache()`——这才是 trial 1 OOM
   的正解：NCCL 用裸 `cudaMalloc` 分配 P2P channel buffer，绕过 PyTorch 缓存
   分配器；大 prefill 之后缓存分配器可能占住所有空闲块，导致 NCCL 连 10 MB 都
   要不到。把缓存块还给 driver 即可。只在首次 swap（创建 channel buffer 时）真正
   起作用，代价几毫秒；
3. 失败后 `empty_cache()` 重试一次，而不是让异常逃到 controller 的
   `except Exception` 里被吞掉（那是 rank 失步的直接原因）；
4. 瞬时显存仍是 O(len(plan))，要压就用 `--pb-oeplb-max-total-ops`（当前 300，
   实测 plan 132-135 ops ≈ 1.2 GB），docstring 里写明了为什么不能改成分块。

**教训（对论文有直接影响）**：§3.4 现在写"同步 P2P 无稳定性问题"。实际上同步路径
有两个真实故障模式：(a) plan 规模放大的瞬时显存把 NCCL 自己的缓冲挤掉；
(b) 任何让参与度不均的"优化"都会造成 coalesced 序号分叉死锁。两者都应写进
局限性，并说明 max_total_ops 是必需的安全阀而非调参项。

### 16:23 driver 之间互相 pkill

driver10/11/12 各自在 boot 前 `pkill -9 -f sglang.launch_server`，而它们通过
`grep D<N>_DONE` 串联；driver10 重启后 log 被截断，串联失效 → driver10 trial3 与
driver11 eplb 阶段同时起服务、互相 kill（`port_base at 30234 is not available`）。
已去掉 driver11/12 的跨 driver 等待，改为逐个显式启动。

## 2026-08-11 17:30 — f_sens 直接测量（driver12，8卡57B）：T(r) 是铰链，不是直线

**结论先说**：`T(r) = 82.86 + 23.60·max(0, r − 1.099)`，R²=0.9981，残差平方和比线性
拟合低 12×。存在**死区** r ≤ r_k=1.099，在其中压低 r 不产生任何收益。

7 个布局点 × 2 轮 = 14 次运行，16384/16384 成功，0 错误，轮间 CV 0.03–0.91%。

| 布局 | r_avg | 均值(s) | 相对 bal |
|---|---|---|---|
| bal | 1.010 | 82.72 | — |
| r110 | 1.073 | 83.00 | +0.34% |
| r122 | 1.148 | 83.84 | +1.35% |
| identity | 1.218 | 85.88 | +3.82% |
| r135 | 1.220 | 86.00 | +3.97% |
| r150 | 1.287 | 87.24 | +5.47% |
| conc | 1.550 | 93.48 | +13.0% |

### 三个副产品

1. **r 是充分统计量**。identity 与 r135 的 r 相差 0.002 但布局结构完全无关，
   实测时间差 0.14%（在 CV 内）。用单一标量参数化布局是合理的。
2. **f_sens = B·r_b/T(r_b) = 0.335**，不是从单点反解的 0.061。差 5.5×。
   论文"FLOP 占比高估 1.9–7.7×"里的 7.7× 是反解在小 x 下的失效产物，已撤回；
   修正为 1.4–1.9×。
3. **上界 +3.40%，经验天花板 +3.82%**（identity→bal，同一批数据，不依赖拟合），
   相差 0.42pp。OEPLB 实测 +1.0% → 系统效率 29%。**该配置收益小是因为空间只有
   3–4%，不是均衡器失效**；但 29% 的效率本身是真问题。

### 两个踩过的坑（都会让实验无声失效）

- **recorder 与 `--deepep-mode auto` 不兼容**：`expert_distribution.py:314` 直接
  `raise NotImplementedError`，gatherer 只认 `normal` / `low_latency`。录制服务改用
  `--deepep-mode normal --disable-cuda-graph`（路由计数由 router argmax 决定，与
  dispatch kernel 无关，所以计数仍有效）；**被计时的** sweep 仍用 `auto`，与 baseline 一致。
- **共享一张 placement 造不出不均衡**：初版 `gen_placement.py` 用跨层聚合计数挑一张
  排列给 28 层共用，四个目标点（1.10/1.20/1.35/1.60）全部退化到 r_agg=1.050，conc 只到
  1.115。单一排列在层间被平均掉。改为**逐层各一张排列**、都把该层热专家压向 GPU0，
  区间才从 0.11 扩到 0.54。若没发现，整条曲线埋在噪声里，会得出"f_sens≈0"的假结论。

### 校验：离线 r 与运行时自报 r 相互独立地吻合

| 配置 | 录制计数离线重算 identity r_avg | DIAG 运行时首次决策 avg_ratio_before |
|---|---|---|
| 8 卡 | 1.218 | 1.216 |
| 4 卡 | 1.107 | 1.113 |

### 待办

- driver13（4 卡扫描，进行中）：4 卡 identity 的 r_avg=1.107 几乎正好落在 8 卡的
  r_k=1.099 上。若 r_k 与 EP 规模无关 → 4 卡上界≈0 → 实测 +1.85% 是噪声。
- 235B 的 T(r) 扫描仍未做。把 r_k=1.10 借给 235B 会让系统效率升到 109%>100%，
  说明 r_k 不能跨模型借用。这是上界模型目前最大的未闭合项。
- `threshold_ratio` 默认 1.02 落在死区内 → 偏保守，应上调到 r_k 附近。§3.2 需改。

## 2026-08-11 18:30 — EP=4 的 T(r) 扫描（driver13，14 次运行全部完成）

拟合（6 个构造点，identity 留 held-out）：

    [affine]  T = 90.19 + 44.34*r                        R2=0.9940  RSS=1.186
    [hinge ]  T = 135.48 + 46.35*max(0, r-1.032)         R2=0.9992  RSS=0.156   (7.6x 更好)
    held-out identity(r=1.107): 实测 139.06  hinge -0.08%  affine +0.15%
    bound r_b=1.107 -> r_a=1.04:  x_eff=0.061  Delta_max=+2.29%  f_sens=0.369
    经验天花板 identity 139.06 -> bal 135.49 = +2.63%

**主结论：r_k 会移动，f_sens 不会。**
- r_k: 1.099 (EP=8) -> 1.032 (EP=4)。斜率 B: 23.60 -> 46.35（约 2x，正好是每卡专家 GEMM
  翻倍）；T_flat: 82.86 -> 135.48（1.63x）。可被重叠吸收的固定松弛占比变小 -> 死区变窄。
- f_sens: 0.335 (EP=8) vs 0.369 (EP=4)，差 9%。所以跨配置外推可以借 f_sens，不能借 r_k。
- 铰链形式在两个独立配置上都胜过直线（RSS 12.1x / 7.6x），不是过拟合。
- 两个配置的拟合上界都略低于经验天花板、误差 <=0.4pp（3.40 vs 3.82；2.29 vs 2.63）。

**推翻了我自己 17:45 的判断。** 当时说"若 r_k 与 EP 无关，4 卡上界≈0，+1.85% 只能是噪声"。
前提被否证：r_k=1.032 < r_after=1.04，死区在 4 卡根本不起作用，上界是 +2.29% 不是 0。

**顺带查出论文两处跨数据集拼接（比 r_k 的事更值得修）：**
1. 附录F 三行（L512_O1/多域/ShareGPT，4卡）的 r_before 全填 1.113，但那是在另一个负载上
   DIAG 记的，被三行共用；各自数据集的 r_before 从未测过 -> driver17（一次开服三次录制）。
2. §2.4/D.3 的 "4卡57B 实测 +1.85%" 是 ShareGPT 20K 的结果，却与 L256 的 r_before 配对，
   相除得系统效率没有意义 -> driver18 在 4 卡 L256 上补 OEPLB 臂（baseline 已有：id 139.06, n=2）。

新脚本：r_avg.py（counts -> identity r_avg + LPT 下界；已用 ep=8 1.2177 vs DIAG 1.216、
ep=4 1.1071 vs DIAG 1.113 双向校验，LPT 下界 1.0100/1.0039 也与实测 bal 布局吻合）。

## 2026-08-11 18:40 — driver15 等价性（oeplb 臂 t1 完成，baseline 臂在跑）

- 256 条 swap 日志，**0 次 P2P failed** —— 之前 chunk@64 那个 NCCL 崩溃已修掉。
- GSM8K 200 题：pre 82.00% -> post 84.00%；**只有 17/200 输出逐字节相同**，16 处对错翻转
  （10 涨 6 跌，净 +4，二项检验无信号）。
- corpus logprob: mean -1.623372 -> -1.616264（好 0.44%）。
- **判读必须等 baseline 臂**：baseline 不可能发生 swap，若它也给出约 183/200 不同 + 类似量级
  的翻转，则这些差异来自 FP8+DeepEP+动态 batching 的归约顺序不确定性，与 swap 无关。

### baseline 对照臂到位（t1），判读结论

    GSM8K(200)      acc            逐字节相同    对错翻转
    oeplb   82.00% -> 84.00%       17/200       16 (10涨 6跌)
    base    84.50% -> 84.00%       41/200       13 (6涨 7跌)
    -> 两臂翻转量无差别；GSM8K n=200 的 1σ≈2.6pp，2pp 的变化本来就不可分辨。
       注意 base 也只有 41/200 逐字节相同：采样路径的字节一致性在本系统里
       从来就不存在（动态 batching + FP8），不能当正确性判据。

    corpus logprob（teacher-forced，50385 token，固定文本）
    base    50385/50385 逐 bit 相同   max|d| = 0.0            <- 噪声底为零！
    oeplb     109/50385 相同          mean|d| = 9.92e-2  max|d| = 6.79
              mean logprob -1.623372 -> -1.616264  (+7.1e-3，方向还变好了)

    -> 布局固定时 teacher-forced logprob 完全确定；所以 oeplb 臂那 0.099 全部
       归因于"布局变了"。扰动近零均值（偏移只有幅度的 7%），像数值噪声而不像
       权重损坏（权重错了会系统性变差）。但 0.099 nat/token 不是舍入量级，值得查。

    判据 = driver19（已排到队首，约20分钟）：用 --init-expert-location 拿静态布局，
    swap 一次都不发生。
      idA vs idB   -> 跨进程重启的数值底
      id  vs r135  -> 纯布局扰动（零 swap）
    若 id-vs-r135 也给出 ~0.1，则 swap 机制没有额外贡献，是布局改变本身的数值效应；
    若仍≈0，则 swap 路径在破坏状态，是真 bug。

队列顺序已改（queue3.sh）：d15 → **d19** → d18 → d17 → d14(235B) → d16。

## 2026-08-12 03:00 — 通宵队列全部结果（d14/d16/d17/d18/d19/d20/d21）

### d14: 235B 的 T(r) 扫描 —— 论文最大的未闭合项关闭
    [hinge]  T = 167.07 + 58.78*max(0, r-1.093)   R2=0.9995  RSS 比 affine 好 25.2x
    上界(r_a=1.05) +22.08%   经验天花板 id 204.03 -> bal 167.07 = +22.12%（差 0.04pp）
    实测 +17.5%  ->  系统效率 79%
    f_sens = 0.496（实测） vs 0.384（nsys beta） vs 0.366（单点反解）
    离线 identity r=1.737 vs DIAG 1.721（第三次交叉校验，0.9%）

**r_k 由并行配置决定，不由模型决定**：235B/EP8 = 1.093，57B/EP8 = 1.099（两个完全不同的
模型几乎同值）；57B/EP4 = 1.032。所以"不能借 r_k"要精确成"不能跨并行配置借，可以跨模型借"。

109% 矛盾的真凶是 f_sens 不是 r_k：论文用的 nsys beta 值 0.384 低估了 26%。换成实测
0.496 后效率 79%，矛盾消失。

**conc 布局（r=4.686）把 DeepGEMM 打崩了**（两轮都没起来）：
    deep_gemm_wrapper/compile_utils.py:218 _empty_token_fp8
    torch.AcceleratorError: CUDA error: invalid configuration argument
极端不均衡下 max_m 过大导致缓冲区分配失败。这是均衡器的一条**安全性**论据，值得写进论文。

### d18: 4卡 L256 同源对照 —— PB-OEPLB 打到 100% 天花板
    baseline(id) 139.06 | 静态最优 bal 135.49 | OEPLB 135.40 (CV 0.24%)  -> +2.70%
    OEPLB 与静态最优不可区分（差 0.07%，在 CV 内）
    DIAG 实际达到 r_after=1.011（不是论文假设的 1.04）-> 上界重算 +2.57%，实测 +2.70%
    首次决策 avg_ratio_before=1.113，与论文的值精确吻合 -> 确认那个 1.113 是 L256 的

### d17: r_before 几乎与数据集无关（ep=4）
    L512 1.1125 | 多域 1.0980 | ShareGPT 1.0965 | L256 1.1071   彼此差 <1.5%
    -> 附录F"三行共用 1.113"方法上不对，结论上无害；现在都是实测值

### d16: r_k 对并发不敏感，f_sens 敏感
    conc   bal     r122    conc    B(s/单位r)  f_sens  r_k估计
    64     106.09  107.88  116.88  22.38       0.249   1.068
    256     82.72   83.84   93.48  23.98       0.342   1.101
    512     81.99   83.79   93.04  23.01       0.328   1.070
    不均衡的绝对代价（秒）几乎不随并发变；占比在低并发下降，因为总时间被拉长。
    r_k 在 8 倍并发范围内基本不动（对比 EP 8->4 时 1.099->1.032）-> 再次指向"r_k 由每卡工作量定"。

### d19+d21: swap 机制清白（因果链闭合）
    idA vs idB     同布局/新进程            100.0% 逐bit相同   mean|d|=0
    idA vs flagid  同布局/走 --init-expert-location 恒等映射  100.0%   mean|d|=0   <- d21 补的对照
    idA vs bal     换布局/零 swap             0.2%   mean|d|=9.588e-2
    idA vs r135    换布局/零 swap             0.2%   mean|d|=9.311e-2
    d15 oeplb      换布局/256 次真 swap        0.2%   mean|d|=9.922e-2
    -> 只有"布局变了"这一个因子产生扰动；swap 机制贡献为零。
    -> d21 排除了"传 flag 本身有副作用"这个混淆（d19 原设计里 idA/idB 走 base 脚本、
       bal/r135 走 fixed 脚本，同时变了两件事，是我漏掉的对照）。
    -> 推论：跨卡置换专家是恒等变换，扰动只能是 FP8 归约顺序 -> 任何 EPLB 类系统都做不到
       逐 bit 等价，等价性判据必须是分布级的。

### d20: 附录F 的 +4.7% 不复现
    base  140.72 139.97 -> 140.34 (cv 0.38%)   tps 58.4
    oeplb 137.16 136.97 -> 137.06 (cv 0.10%)   tps 59.8
    gain = +2.39%（附录F 单次测得 +4.7%）
    该数据集自己的上界：r_b=1.116(DIAG，与 d17 离线 1.1125 吻合 0.3%)，r_after=1.010(实测)，
    x_eff=0.0753 -> 上界 +2.86%，实测 +2.39% -> 效率 84%。原 +4.7% 是上界的 1.64 倍，不可能值。
    成因：每臂各飘约 1%（base 57.7->58.4，oeplb 60.4->59.8），比值飘 2%。附录F 每臂 n=1。

### 待办
- d22 在跑：多域(+2.6%，其上界 +2.27%，115%)、ShareGPT(+3.1%，上界 +2.22%，140%) 同法重测。
- 论文要改：§2.4 三张表 + 109% 那段整段重写为 79%；D.3；附录F.2/F.5 的三个头条数字；
  附录G 加 235B 与 d16/d19/d21；新写附录E（数值等价性 + DeepGEMM 崩溃）。

## 2026-08-12 04:10 — 上界的可画形式 + EP=2 的预注册预测

### 上界化简（与论文原式代数等价，已验）
    Delta_ceiling = T(r_b)/T_flat - 1 = beta * max(0, r_b - r_k),   beta = B/T_flat
    等价性: f_sens*x_eff/(1-f_sens*x_eff) = B(r_b-rk)/T(r_b) / (T_flat/T(r_b)) = beta(r_b-rk)  QED

    配置参数 (beta, r_k) 与数据集参数 (r_b) 彻底分离 -> 一条折线 = 一个配置，一个数据集 = 线上一点。
    beta 比 f_sens 稳定：0.2848 / 0.3421 / 0.3518  vs  f_sens 0.335 / 0.369 / 0.496。
    原因：f_sens = B*r_b/T(r_b) 把工作点 r_b 混进了配置参数，beta 不含。**论文应改用 beta 参数化。**

    config              beta     r_k | dataset    r_b     ceiling  measured  eta
    57B  EP=8         0.2848  1.099 | L512     1.2288    3.70%        -      -
                                    | L256     1.2177    3.38%    1.00%    30%
                                    | multi    1.2085    3.12%        -      -
                                    | ShareGPT 1.2040    2.99%        -      -
    57B  EP=4         0.3421  1.032 | L512     1.1125    2.75%    2.39%    87%
                                    | L256     1.1071    2.57%    2.70%   105%
                                    | multi    1.0980    2.26%        -      -
                                    | ShareGPT 1.0965    2.21%        -      -
    235B EP=8         0.3518  1.093 | L512     1.7370   22.66%   17.50%    77%
    脚本 bound_curve.py -> /workspace/EPLB/OEPLB/fig_bound.png

### 预注册：57B EP=2（在跑测之前写下，不得事后修改）
    identity r_avg(ep=2) = 1.0210，LPT 下界 1.0010（r_avg.py 离线算）
    -> 天然不均衡已低于 EP=8/EP=4 两个 r_k，EP=2 本身就在死区附近。

    预测 1: r_k(EP=2) ≈ 1.00~1.01。
      依据：r_k 对 EP 线性外推 1.099@8, 1.032@4 -> 0.898@2 (<1，不可能)；
            对 log2(EP) 外推 -> 0.965@2 (<1，不可能)。既然两种外推都撞下界，只能在 1.0 饱和。
    预测 2: beta(EP=2) ∈ [0.34, 0.45]（0.285@EP8 -> 0.342@EP4，继续升但趋饱和）
    预测 3: identity 处上界 = beta*(1.021-r_k) ≈ 0.4%~0.8%
    预测 4: PB-OEPLB 在 EP=2/L256 上实测收益 ≈ 0，扣开销后可能略负。

    这一条是模型**判别力**的检验（会不会说"不"）——原计划用 30B 做，但 30B/DS-V2-Lite
    已不在 /data/models 上，EP=2 是更省的替代，且同模型可控。

## 2026-08-12 04:55 — DIAG 的 avg_ratio_before 有小样本偏差（均衡器的真缺陷）

d22 的 ShareGPT/4卡 首窗自报 avg_ratio_before=2.161，而离线聚合只有 1.0965（差 97%）。
前四次交叉校验都 <1%（L256 0.5%、multi 1.0%、L512 0.3%、235B 0.9%），只有 ShareGPT 崩。

排查（新脚本 r_window.py，利用 dump 里 logical_count 的 [4000,28,64] 逐 forward 维度）：
    ShareGPT/ep=4:  r_agg=1.0965   r_win(w=16)=1.1000   r_fwd(token加权)=1.1230
    multi   /ep=4:  r_agg=1.0980   r_win(w=16)=1.1020   r_fwd=1.1040
    -> Jensen 效应只有 0.3%，解释不了 2.161。

真因 = **小 batch 的抽样方差**：
    逐窗口(w=16) 250个:  p50=1.103 p90=1.211 p99=1.320 max=1.355
    逐forward:           p50=1.131 p90=1.759 p99=2.107 max=2.250
    token < 中位1/10 的 forward（1364个）ratio均值 1.551，其余 1.128
    ratio 最大的窗口 token=3584，中位窗口 token=922432（差 250x）
    -> batch 越小，per-GPU 分布的抽样方差越大，ratio 越高；而小窗口不占时间。

结论 1（对模型）：离线 1.0965 是对的，ShareGPT 上界 2.21% 站得住。**r_before 必须用
token 加权口径**；DIAG 的不加权均值在异质负载上系统性偏高。r_window.py 现在给三种口径。

结论 2（对系统，更重要）：**PB-OEPLB 的决策就用这个不加权统计量**，所以它会被小样本噪声
触发——3584 token 的窗口报 r=2.16，均衡器信了，付 P2P 阻塞换回零收益。现有保护
`--pb-oeplb-min-prefill-tokens 256` 比中位窗口(92万 token)低四个数量级，等于没有。
这可能是 eta 偏低（8卡57B 只有 29%）的一部分成因，且可直接验证 -> d26。
