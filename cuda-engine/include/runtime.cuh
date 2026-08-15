#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "sim.cuh"

namespace kagcuda {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", file, line, expr, cudaGetErrorString(err));
    std::exit(2);
}
#define KAGCUDA_CHECK(expr) ::kagcuda::cuda_check((expr), #expr, __FILE__, __LINE__)

struct TablesOwner {
    DeviceTables dev{};

    TablesOwner() {
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.price_cache),
                                 static_cast<size_t>(N_PRODUCTS) * PRICE_CACHE_SPAN * sizeof(int)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.price_prefix),
                                 static_cast<size_t>(N_PRODUCTS) * (PRICE_CACHE_SPAN + 1) * sizeof(int64_t)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.mt_base), PyRandom::N * sizeof(uint32_t)));

        const int total = N_PRODUCTS * PRICE_CACHE_SPAN;
        init_price_cache_kernel<<<(total + 255) / 256, 256>>>(dev);
        KAGCUDA_CHECK(cudaGetLastError());
        init_price_prefix_kernel<<<1, 32>>>(dev);
        KAGCUDA_CHECK(cudaGetLastError());
        init_mt_base_kernel<<<1, 1>>>(dev);
        KAGCUDA_CHECK(cudaGetLastError());
        KAGCUDA_CHECK(cudaDeviceSynchronize());
    }

    ~TablesOwner() {
        if (dev.price_cache) cudaFree(dev.price_cache);
        if (dev.price_prefix) cudaFree(dev.price_prefix);
        if (dev.mt_base) cudaFree(dev.mt_base);
    }

    TablesOwner(const TablesOwner&) = delete;
    TablesOwner& operator=(const TablesOwner&) = delete;
};

inline void configure_device() {
    // Sim/PyRandom use sizeable thread-local state. This is not a CPU fallback:
    // it only raises the CUDA device stack limit for the all-device episode kernel.
    KAGCUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));
}

}  // namespace kagcuda
