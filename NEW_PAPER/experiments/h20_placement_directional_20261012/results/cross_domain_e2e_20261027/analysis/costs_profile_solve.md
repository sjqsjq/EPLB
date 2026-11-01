# 画像与求解成本单列 (作者关键点 2)
- 录制 boot: 190.86s bench (ok=8192, 0 err, 42.9 tps), 基座同正式臂 + stat recorder + deepep normal + identity 布局
- recorder .pt: 48,130,049 B, sha256 fa04965525ecb44ccd89e67be525e815a5ae5474ccbb44c6c7c9f9c13dc6df4a
  - rank0 shape (1000, 94, 128), sum = 3,154,479,072; 每层 33,558,288 = 4,194,260 prompt tok x 8 topk (精确)
- P_e3.npz: pt2npz.py 对 dim_extra 求和 -> P[94,128]
- MT ILP: solve_ilp.py (Gurobi 13.0.3 restricted license), ILP1 greedy, G=8 K=16, time_limit=30s, mip_gap=0.005
  - 94 层全部求解, 总 2822.5s; 每层 solve_s 30.0-30.2s (全部触顶 time_limit); mean_imbalance=1.0000 max=1.0000
  - 产物 placement_e3_fair.json (sha ddb38bfd...) + meta (sha 351adaac...)
- ES/DF: 共用 datafore_e3_placement.json == datafore_e3_remap_placement.json (字节相同, sha ddc3943d...);
  无离线求解成本; 布局构建发生在 boot init (ES: init_by_eplb R16; DF: rebalance_experts greedy)
- 摊销口径: 校准成本 = 1 录制 boot (~15min 启动 + 191s bench) + 1 次 ILP (2822.5s, 一次性离线);
  静态臂运行时零求解零重平衡 (T6 表 runtime rebalance rows = 0)
