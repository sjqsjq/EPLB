# OEPLB 项目目录清理 — 执行记录

## 已执行清理

### logs/ (3.2GB → 52K)
删除：4个core dump(2.8GB, 崩溃残留) + 10个一次性driver脚本 + 35个一次性launch脚本 + 52个server/bench log + 12个旧/噪声trace目录 + VL(vision-language)脚本与日志 + 旧分析脚本。

**保留**（复现必需）：
- `driver_length_clean.sh` — O=10干净长度实验（3个完美数据集来源）
- `env_235b.sh` — NVSHMEM/NCCL环境
- `launch235b_identity.sh`, `launch235b_pb_oeplb.sh` — 标准benchmark启动
- `send_trace_requests_general.py` — 通用trace请求发送器
- `run_trace_bench.py` — trace benchmark
- `length_O10_clean_results.json`, `ob3_figure_data.json`, `ob3_extra_figure_data.json` — 干净figure数据

### OEPLB/*.md 文档
- **AUDIT**: 4个(AUDIT_findings/AUDIT2/AUDIT3/AUDIT_FINAL) → 合并为 `AUDIT.md`（4轮证据链审计合集，历史审稿记录）
- **删除** `PREFILL_DECODE_CORRELATION_STUDY.md`（含作废/污染的旧O=64/O=256/拼接book数据 + 失败的dormant/decode-heavy实验；干净版 = `PREFILLBOUNDARY_DATA.md`）
- **论文同步**: PAPER_en.md / PAPER_zh.md / PAPER_zh_v2.md(精简版) 的 §2.3 Obs3 + §3.6 已写入直接ρ数据(MMLU 0.833/94层, prover 0.980, book 0.967)

### 数据集整理
3个完美数据集+trace整理到 `/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/`（含README复现文档）

## 保留未动（需你后续确认）
- `PAPER.md`(598行, 旧英文短版) vs `PAPER_en.md`(1178, 完整版) — 未删，等你定权威版
- `OEPLB/` 其余辅助文档(ADAPTIVE_DESIGN/COMPREHENSIVE_EXPERIMENT_LOG/INTRODUCTION/SYSTEM/QUICKSTART/REVIEW_zh) — 内容可能已并入PAPER，未删待确认
- `/workspace/EPLB/*.md`(6个历史报告) — 未删待确认
- `benchmarks/results/`(1550 json, 7.5MB) — 可能有论文引用，保留
