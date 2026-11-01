# author_diag_20261021 — 作者指令的临时吞吐诊断(非正式包)

**授权**:作者 2026-10-21 会话内直接指令("直接迅速跑一下 E3 上 P 和 I 的直接吞吐对比…
再跑最开始什么都没改的系统")。
**性质**:诊断性 4 boots(E3,P/I × {v2 修复版, 原始未修复版})。
**池隔离**:标签 adhoc_*,独立台账 LEDGER_adhoc.tsv;不进 s_* 补充池、不进正式 ≤104
预算池、不参与任何 Holm/假设检验;只作描述性吞吐对照,单 boot 噪声 ±1-3 tps 量级。
**代码切换**:第 3/4 boot 临时把 deployed pb_oeplb 的 async_swapper.py/controller.py
换成 code_lock 原始版(93fa4ccdd653443a… / c6d2f76454679dff…,即旧战役版本,
含已知 NCCL 预算切分死锁与迁移竞态——挂死风险已知且被容纳:bench timeout 1200s,
server watchdog 600s,挂死现场全保留)。队列结束(无论成败)无条件恢复 v2 冻结部署
(2e4652a0306cb315… / 4964cfc8e8af397a…)并校验 sha;src 树全程不动。
**对照背景**:phase-4 8 块均值 E3 tps P=43.06±1.25,I=41.92±0.98(commit 5038cd1f)。
