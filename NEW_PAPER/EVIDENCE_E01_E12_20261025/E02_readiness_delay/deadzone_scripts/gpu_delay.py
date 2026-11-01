"""ctypes wrapper around the stream-ordered one-thread CUDA delay kernel."""

from __future__ import annotations

import ctypes
import pathlib

import torch


class GpuDelay:
    def __init__(self, library: pathlib.Path | None = None) -> None:
        if library is None:
            library = (
                pathlib.Path(__file__).resolve().parents[1]
                / "build"
                / "libdeadzone_gpu_delay.so"
            )
        if not library.exists():
            raise FileNotFoundError(
                f"Missing {library}; run scripts/build_delay_extension.py first"
            )
        self._library = ctypes.CDLL(str(library))
        self._launch = self._library.launch_deadzone_gpu_delay_ns
        self._launch.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        self._launch.restype = ctypes.c_int

    def __call__(self, target_ns: int) -> None:
        if target_ns < 0:
            raise ValueError(f"target_ns must be nonnegative, got {target_ns}")
        stream_ptr = torch.cuda.current_stream().cuda_stream
        status = self._launch(int(target_ns), int(stream_ptr))
        if status != 0:
            raise RuntimeError(f"CUDA delay kernel launch failed with error {status}")
