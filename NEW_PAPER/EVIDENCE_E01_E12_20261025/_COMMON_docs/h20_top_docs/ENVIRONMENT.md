# Environment snapshot

- **Protocol date:** 2026-10-12
- **Snapshot status:** hardware/software confirmed; real-forward input and
  checkpoint-layer manifests are not yet frozen
- **Repository HEAD at experiment start:**
  `a59949d55737499cdc2f2d05808b2c11b27cc7e0`

## Hardware

- 8 × NVIDIA H20
- reported memory per GPU: 97,871 MiB
- full eight-GPU `NV18` peer topology
- driver: 580.105.08
- GPUs were idle before E0/E1 setup (no compute processes; approximately
  97,353 MiB free per GPU at inspection)

The main experiment must save a fresh `nvidia-smi`, topology, clocks/power state,
and active-process snapshot next to each run. The values above are the initial
pre-experiment check, not a substitute for run-time metadata.

## Software

- Python 3.11.12
- PyTorch 2.9.1+cu128
- CUDA runtime 12.8
- SGLang 0.5.6.post2
- DeepEP normal mode
- DeepGEMM FP8
- eager execution for E0/E1

### DeepEP

- source: `/workspace/build/DeepEP`
- source commit: `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`
- loaded package: `/workspace/build/DeepEP/deep_ep/__init__.py`
- extension SHA-256:
  `cca04a167e34c579163ea7765b7e8ccf8e1a3015e735d8f6c00a7a1344ad9441`
- extension path:
  `/workspace/build/DeepEP/deep_ep_cpp.cpython-311-x86_64-linux-gnu.so`

The external DeepEP tree already contained user changes before this experiment:

```text
 M csrc/kernels/internode_ll.cu
 M deep_ep/buffer.py
?? deep_ep_cpp.cpython-311-x86_64-linux-gnu.so
```

This experiment must not reset, overwrite, or silently absorb those changes. The
run manifest must record source diff/hash metadata because the loaded extension
may not be reproducible from the named commit alone.

### DeepGEMM

- source: `/workspace/build/DeepGEMM`
- source commit: `06fbbd449e36b7e0d97580b1e913140457ba563a`
- loaded extension:
  `/opt/conda/lib/python3.11/site-packages/deep_gemm/deep_gemm_cpp.abi3.so`
- loaded extension SHA-256:
  `f981f79e96233e1a57785dd94f233b03444fa6431a583c81b1d653ae0c1abfa4`

## Intended model geometry

These are protocol constraints, not yet a frozen checkpoint manifest:

- 128 routed experts
- 16 physical expert slots per rank at EP=8
- top-k=8
- hidden size=4096
- expert intermediate size=1536

The model identifier, exact revision, selected layer, tensor paths, shard hashes,
input snapshot hashes, and logical route hashes remain intentionally unspecified
until the real-forward capture is validated. They will be written to
`inputs/manifest.json`; this document does not fabricate them in advance.

## Timing-reference feasibility check

A pre-E0 standalone test on all eight H20s established:

- PyTorch CUDA-IPC events opened across devices and used in cross-device stream
  waits are unsafe in this environment: both tested variants ended in SIGSEGV.
  The experiment must not use that mechanism.
- A warmed NCCL stream-alignment + GPU-0 end-reduce completion fence succeeded.
  Its empty fenced interval was 57–68 us per block, or about 1.1–1.4 us per
  forward when amortized over 50 forwards.

The actual layer replay must remeasure this empty-fence distribution. The fenced
interval and `T_proxy` remain separately named and stored.
