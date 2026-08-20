#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "kaggriculture.h"

struct Holder {
    KaggricultureEnv env{};
    std::vector<float> obs = std::vector<float>(KAG_OBS_SIZE);
    float acts[KAG_NUM_ATNS]{};
    float rewards[1]{};
    float terminals[1]{};
};

int main(int argc, char** argv) {
    const int nenv = argc > 1 ? std::atoi(argv[1]) : 8;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
    const int rounds = argc > 3 ? std::atoi(argv[3]) : 20;
    std::vector<Holder> hs(nenv);
    for (int e = 0; e < nenv; ++e) {
        auto& h = hs[e];
        h.env.observations = h.obs.data();
        h.env.actions = h.acts;
        h.env.rewards = h.rewards;
        h.env.terminals = h.terminals;
        h.env.num_agents = 1;
        h.env.env_id = e;
        h.env.base_seed = 10000;
        h.env.opponent_kind = 1;
        h.env.sim = new kag::Sim();
        c_reset(&h.env);
        for (int a = 0; a < KAG_NUM_ATNS; ++a) h.acts[a] = 0;
    }
    const long long total = static_cast<long long>(rounds) * 720 * nenv;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; ++r) {
        for (int t = 0; t < 720; ++t) {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int e = 0; e < nenv; ++e) c_step(&hs[e].env);
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("envs=%d threads=%d steps=%lld seconds=%.6f sps=%.0f\n",
                nenv, threads, total, sec, total / sec);
    for (auto& h : hs) c_close(&h.env);
    return 0;
}
