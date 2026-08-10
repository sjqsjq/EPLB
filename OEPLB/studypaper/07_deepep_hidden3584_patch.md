# DeepEP hidden_size=3584 支持补丁

## 背景

Qwen2-57B-A14B-Instruct 的 `hidden_size=3584`，不在 DeepEP v1.2.1
`SWITCH_HIDDEN` 宏的硬编码列表里（只支持 2048/2560/4096/5120/6144/7168/8192），
导致 `deepep-mode=auto` 报错 "Unsupported hidden"。

## 数学验证

DeepEP low_latency kernel 的整除约束：
- bf16 路径：`kHidden % 256 == 0`
- fp8 路径：`kHidden % 512 == 0`
- send unroll 路径：`kHidden % 512 == 0`

验证：`3584 % 512 = 0` ✓ （3584 = 7 × 512）

所以 3584 在数学上完全兼容所有整除约束，只是宏里漏列了。

## 补丁

`csrc/kernels/launch.cuh` 的 `SWITCH_HIDDEN` 宏中加一行：

```c
case 3584: case_macro(3584); \
```

## 重新编译

```bash
cd /tmp/deepep_gitcode
NVSHMEM_DIR=/tmp/nvshmem_official/root pip install -e . --no-build-isolation
```

## 效果

Qwen2-57B 现在可以用 `deepep-mode=auto`（decode 走 low_latency kernel，保留 CUDA graph），
而不是被迫用 `deepep-mode=normal`（禁用 CUDA graph，-68% decode 吞吐）。

这是 OEPLB 在 Qwen2 系列上能获得正收益的必要条件之一。
