#include "runtime.cuh"
#include "tape_io.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace kagcuda;

__global__ void run_tape_batch_kernel(Config base, DeviceTables tables,
                                      const Action* tape0, int n0,
                                      const Action* tape1, int n1,
                                      int episodes, Sim* sims, double* rewards) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= episodes) return;
    Config cfg = base;
    cfg.seed = base.seed + static_cast<uint64_t>(id);
    Sim& sim = sims[id];
    sim.init(cfg, &tables);
    while (!sim.st.done) {
        const int i0 = sim.st.step < n0 ? sim.st.step : n0 - 1;
        const int i1 = sim.st.step < n1 ? sim.st.step : n1 - 1;
        const int units0 = sim.st.farms[0].n_units;
        const int units1 = sim.st.farms[1].n_units;
        sim.step_with_unit_counts(tape0[i0], units0, tape1[i1], units1);
    }
    rewards[id * 2] = sim.st.farms[0].money;
    rewards[id * 2 + 1] = sim.st.farms[1].money;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: tape_bench TAPE0.json TAPE1.json [SEED_START=62000] [EPISODES=4096] [TPB=64]\n");
        return 2;
    }
    const uint64_t seed_start = argc > 3 ? static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10)) : 62000ull;
    const int episodes = argc > 4 ? std::atoi(argv[4]) : 4096;
    const int tpb = argc > 5 ? std::atoi(argv[5]) : 64;
    if (episodes <= 0 || tpb <= 0 || tpb > 1024) return 2;

    NativeTape h0, h1;
    try {
        h0 = load_native_tape_json(argv[1]);
        h1 = load_native_tape_json(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tape parse error: %s\n", e.what());
        return 2;
    }

    configure_device();
    TablesOwner tables;
    Config cfg;
    cfg.seed = seed_start;

    Action *d0 = nullptr, *d1 = nullptr;
    Sim* d_sims = nullptr;
    double* d_rewards = nullptr;
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d0), h0.actions.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d1), h1.actions.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sims), static_cast<size_t>(episodes) * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rewards), static_cast<size_t>(episodes) * 2 * sizeof(double)));
    KAGCUDA_CHECK(cudaMemcpy(d0, h0.actions.data(), h0.actions.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d1, h1.actions.data(), h1.actions.size() * sizeof(Action), cudaMemcpyHostToDevice));

    run_tape_batch_kernel<<<(episodes + tpb - 1) / tpb, tpb>>>(
        cfg, tables.dev, d0, static_cast<int>(h0.actions.size()), d1, static_cast<int>(h1.actions.size()),
        episodes, d_sims, d_rewards);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    KAGCUDA_CHECK(cudaEventCreate(&start)); KAGCUDA_CHECK(cudaEventCreate(&stop));
    KAGCUDA_CHECK(cudaEventRecord(start));
    run_tape_batch_kernel<<<(episodes + tpb - 1) / tpb, tpb>>>(
        cfg, tables.dev, d0, static_cast<int>(h0.actions.size()), d1, static_cast<int>(h1.actions.size()),
        episodes, d_sims, d_rewards);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaEventRecord(stop)); KAGCUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f; KAGCUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<double> rewards(static_cast<size_t>(episodes) * 2);
    KAGCUDA_CHECK(cudaMemcpy(rewards.data(), d_rewards, rewards.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double sink = 0.0; for (double v : rewards) sink += v;
    const double sec = ms / 1000.0;
    const double eps = episodes / sec;
    std::printf("GPU live tapes: steps=(%zu,%zu) seeds=%llu..%llu episodes=%d in %.3f ms\n",
                h0.actions.size(), h1.actions.size(), static_cast<unsigned long long>(seed_start),
                static_cast<unsigned long long>(seed_start + episodes - 1), episodes, ms);
    std::printf("%.1f eps/s = %.3f M transitions/s block=%d sink=%.0f\n",
                eps, eps * 719.0 / 1e6, tpb, sink);
    const int show = episodes < 5 ? episodes : 5;
    for (int i = 0; i < show; ++i)
        std::printf("seed=%llu rewards=(%.0f,%.0f)\n",
                    static_cast<unsigned long long>(seed_start + i), rewards[i*2], rewards[i*2+1]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d0); cudaFree(d1); cudaFree(d_sims); cudaFree(d_rewards);
    return 0;
}
