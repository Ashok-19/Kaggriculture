#include "runtime.cuh"
#include "policy_runner.cuh"
#include <cstdio>
#include <vector>

using namespace kagcuda;

struct PassState { int calls; };
struct PassPolicy {
    __device__ __forceinline__ void reset(PassState& s, const ObservationView& obs) const {
        s.calls = obs.step();
    }
    __device__ __forceinline__ void act(PassState& s, const ObservationView& obs, Action& out) const {
        // Touch every observation category so compilation proves the legal view
        // is usable from real device policy code. The volatile expression does
        // not change the pass action.
        const PublicFarmView own = obs.farm(obs.player());
        const PublicFarmView opp = obs.farm(1 - obs.player());
        const PrivateFarmView priv = obs.private_farm();
        const MarketView market = obs.market();
        const TownView town = obs.town();
        volatile int sink = obs.day() + obs.hour() + own.n_units() + opp.n_units() +
                            priv.shed(WHEAT) + priv.seed(WHEAT) + market.price(WHEAT) +
                            market.inventory(WHEAT) + town.n_shops();
        (void)sink;
        ++s.calls;
        out.n_orders = 0;
    }
};

int main() {
    configure_device();
    TablesOwner tables;
    constexpr int N = 256;
    Sim* sims = nullptr;
    PassState *s0 = nullptr, *s1 = nullptr;
    double* rewards = nullptr;
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sims), N * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s0), N * sizeof(PassState)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s1), N * sizeof(PassState)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rewards), N * 2 * sizeof(double)));
    Config cfg; cfg.seed = 1000;
    run_policy_batch_kernel<<<4,64>>>(cfg, tables.dev, N, PassPolicy{}, s0, PassPolicy{}, s1, sims, rewards);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> host(N * 2);
    KAGCUDA_CHECK(cudaMemcpy(host.data(), rewards, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (double v : host) {
        if (v != 3000.0) {
            std::fprintf(stderr, "policy smoke mismatch reward=%.0f expected=3000\n", v);
            return 1;
        }
    }
    std::printf("PASS device-policy legal-observation rollout: %d pass-vs-pass episodes exact final=(3000,3000)\n", N);
    cudaFree(sims); cudaFree(s0); cudaFree(s1); cudaFree(rewards);
    return 0;
}
