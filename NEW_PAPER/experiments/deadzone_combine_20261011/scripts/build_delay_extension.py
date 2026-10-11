#!/usr/bin/env python3
"""Build the tiny stream-ordered CUDA delay library (no PyTorch headers)."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    source = root / "scripts" / "gpu_delay.cu"
    output = root / "build" / "libdeadzone_gpu_delay.so"
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force and output.stat().st_mtime >= source.stat().st_mtime:
        print(output)
        return

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the delay kernel")

    command = [
        nvcc,
        "-O3",
        "-lineinfo",
        "-std=c++17",
        "-arch=sm_90",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(source),
        "-o",
        str(output),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
