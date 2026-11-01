# PAPER_EXPERIMENTS(指明文本)

按作者 2026-10-26 指示:归档数据量大,正式归档存放于 /data/minghua/sjq/PAPER_EXPERIMENTS/(不强制上 GitHub);本目录只保留指明文本。原话含 /data/minghe/sjq,经查该路径不存在,按笔误处理为既有数据盘 /data/minghua/sjq,已登记。

- 总索引:本目录 EXPERIMENT_MAP.md(与归档侧 /data/minghua/sjq/PAPER_EXPERIMENTS/EXPERIMENT_MAP.md 同文)
- 当前唯一登记实验:adaptive_prover_sd_20261026(论文第 5.4 节「自适应观测与规划调度消融」)
- 归档实体:/data/minghua/sjq/PAPER_EXPERIMENTS/adaptive_prover_sd_20261026/(README.md、preregistration/、reports/、scripts/、tools_frozen/、streams/、boots_phase1/ 2 boots、boots_phase2/ 24 boots、queue_logs/、ARCHIVE_SHA256SUMS.txt)
- 原始来源:/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/abl_prover_sd_20261026/(保留未动)
- 结果概括:相对固定稀疏档 F(4,16,64) 吞吐提高约 6.5%(+5.88 tps,Holm p=0.0112 显著);相对固定密集档 F(1,16,16) 未检出差异(+0.54 tps,Holm p=0.750)。
- 治理:归档为只读副本;本次未跑 GPU、未改参数、未改原始结果;经作者本次放宽,未做 git commit/push(如需将指明文本推送 GitHub,待作者一句话即可执行)。
