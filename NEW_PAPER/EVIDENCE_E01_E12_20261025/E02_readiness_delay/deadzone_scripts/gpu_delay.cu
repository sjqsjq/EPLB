#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ std::uint64_t read_globaltimer_ns() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// One block is launched, but only lane 0 spins. This keeps the injected delay
// stream-ordered while consuming negligible SM capacity.
__global__ __launch_bounds__(32, 1) void deadzone_gpu_delay_ns(
        std::uint64_t target_ns) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const std::uint64_t start = read_globaltimer_ns();
        while (read_globaltimer_ns() - start < target_ns) {
            __nanosleep(64);
        }
    }
}

extern "C" int launch_deadzone_gpu_delay_ns(
        std::uint64_t target_ns,
        std::uintptr_t stream_ptr) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    deadzone_gpu_delay_ns<<<1, 32, 0, stream>>>(target_ns);
    return static_cast<int>(cudaGetLastError());
}
