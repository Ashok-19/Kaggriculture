#include "runtime.cuh"
#include "tape_io.hpp"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace kagcuda;

struct TapeDesc {
    int offset;
    int length;
};

__device__ __forceinline__ const Action& tape_at(const Action* actions, TapeDesc d, int step) {
    const int idx = step < d.length ? step : d.length - 1;
    return actions[d.offset + idx];
}

__global__ void make_tape_snapshots_kernel(Config base, DeviceTables tables,
                                           const Action* base0, int n0,
                                           const Action* base1, int n1,
                                           int prefix, int n_snapshots,
                                           Sim* sims, Snapshot* snapshots) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_snapshots) return;
    Config cfg = base;
    cfg.seed = base.seed + static_cast<uint64_t>(id);
    Sim& sim = sims[id];
    sim.init(cfg, &tables);
    for (int t = 0; t < prefix; ++t) {
        const int i0 = t < n0 ? t : n0 - 1;
        const int i1 = t < n1 ? t : n1 - 1;
        const int units0 = sim.st.farms[0].n_units;
        const int units1 = sim.st.farms[1].n_units;
        sim.step_with_unit_counts(base0[i0], units0, base1[i1], units1);
    }
    sim.save_snapshot(&snapshots[id]);
}

__global__ void branch_tape_candidates_kernel(DeviceTables tables,
                                              const Snapshot* snapshots, int n_snapshots,
                                              const Action* base0, int n0,
                                              const Action* base1, int n1,
                                              const Action* candidate_actions,
                                              const TapeDesc* candidate_descs, int n_candidates,
                                              int candidate_player, int prefix,
                                              const uint32_t* rng_cache,
                                              int cache_first_day, int cache_n_days,
                                              Sim* sims, double* rewards) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_snapshots * n_candidates;
    if (id >= total) return;
    const int snapshot_id = id / n_candidates;
    const int candidate_id = id - snapshot_id * n_candidates;

    const uint32_t* cache = nullptr;
    if (rng_cache && cache_n_days > 0)
        cache = rng_cache + static_cast<size_t>(snapshot_id) * cache_n_days * PyRandom::N;

    Sim& sim = sims[id];
    sim.load_snapshot(snapshots[snapshot_id], &tables, cache, cache_first_day, cache_n_days);
    const TapeDesc cand = candidate_descs[candidate_id];

    while (!sim.st.done) {
        const int t = sim.st.step;
        const int i0 = t < n0 ? t : n0 - 1;
        const int i1 = t < n1 ? t : n1 - 1;
        const Action& ca = tape_at(candidate_actions, cand, t);
        const Action& a0 = candidate_player == 0 ? ca : base0[i0];
        const Action& a1 = candidate_player == 1 ? ca : base1[i1];
        const int units0 = sim.st.farms[0].n_units;
        const int units1 = sim.st.farms[1].n_units;
        if (cache)
            sim.step_with_unit_counts_cached(a0, units0, a1, units1);
        else
            sim.step_with_unit_counts(a0, units0, a1, units1);
    }
    rewards[id * 2] = sim.st.farms[0].money;
    rewards[id * 2 + 1] = sim.st.farms[1].money;
}

static const char* basename_ptr(const char* path) {
    const char* out = path;
    for (const char* p = path; *p; ++p) if (*p == '/') out = p + 1;
    return out;
}

static void append_candidate_arg(const char* arg, std::vector<std::string>& out) {
    if (arg[0] != '@') {
        out.emplace_back(arg);
        return;
    }
    std::ifstream in(arg + 1);
    if (!in) throw std::runtime_error(std::string("cannot open candidate list: ") + (arg + 1));
    std::string line;
    while (std::getline(in, line)) {
        size_t a = 0, b = line.size();
        while (a < b && (line[a] == ' ' || line[a] == '\t' || line[a] == '\r')) ++a;
        while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t' || line[b - 1] == '\r')) --b;
        if (a == b || line[a] == '#') continue;
        out.emplace_back(line.substr(a, b - a));
    }
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: tape_branch BASE0.json BASE1.json CANDIDATE_PLAYER PREFIX SEED_START N_SNAPSHOTS CANDIDATE.json|@LIST [CANDIDATE.json ...]\n");
        return 2;
    }
    const int candidate_player = std::atoi(argv[3]);
    const int prefix = std::atoi(argv[4]);
    const uint64_t seed_start = static_cast<uint64_t>(std::strtoull(argv[5], nullptr, 10));
    const int n_snapshots = std::atoi(argv[6]);
    if ((candidate_player != 0 && candidate_player != 1) || prefix < 0 || prefix > 719 || n_snapshots <= 0)
        return 2;

    NativeTape base0, base1;
    std::vector<NativeTape> candidates;
    std::vector<std::string> candidate_paths;
    try {
        for (int i = 7; i < argc; ++i) append_candidate_arg(argv[i], candidate_paths);
        if (candidate_paths.empty()) throw std::runtime_error("candidate list is empty");
        base0 = load_native_tape_json(argv[1]);
        base1 = load_native_tape_json(argv[2]);
        candidates.reserve(candidate_paths.size());
        for (const std::string& path : candidate_paths) candidates.push_back(load_native_tape_json(path));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tape parse error: %s\n", e.what());
        return 2;
    }

    std::vector<Action> candidate_actions;    const int n_candidates = static_cast<int>(candidates.size());


    std::vector<TapeDesc> candidate_descs;
    size_t total_actions = 0;
    for (const auto& t : candidates) total_actions += t.actions.size();
    candidate_actions.reserve(total_actions);
    candidate_descs.reserve(candidates.size());
    for (const auto& t : candidates) {
        const int offset = static_cast<int>(candidate_actions.size());
        candidate_actions.insert(candidate_actions.end(), t.actions.begin(), t.actions.end());
        candidate_descs.push_back({offset, static_cast<int>(t.actions.size())});
    }

    configure_device();
    TablesOwner tables;
    Config cfg;
    cfg.seed = seed_start;
    const int total_branches = n_snapshots * n_candidates;
    const int cache_first_day = prefix / cfg.turns_per_day;
    const int eod_days = (cfg.episode_steps - 1) / cfg.turns_per_day;  // 29: days 0..28.
    const int last_eod_day = eod_days - 1;
    const int cache_n_days = cache_first_day <= last_eod_day ? last_eod_day - cache_first_day + 1 : 0;

    Action *d_base0 = nullptr, *d_base1 = nullptr, *d_candidates = nullptr;
    TapeDesc* d_descs = nullptr;
    Sim *d_prefix_sims = nullptr, *d_branch_sims = nullptr;
    Snapshot* d_snapshots = nullptr;
    uint32_t* d_rng_cache = nullptr;
    double* d_rewards = nullptr;

    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base0), base0.actions.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base1), base1.actions.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidates), candidate_actions.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_descs), candidate_descs.size() * sizeof(TapeDesc)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prefix_sims), static_cast<size_t>(n_snapshots) * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_snapshots), static_cast<size_t>(n_snapshots) * sizeof(Snapshot)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_branch_sims), static_cast<size_t>(total_branches) * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rewards), static_cast<size_t>(total_branches) * 2 * sizeof(double)));
    if (cache_n_days > 0) {
        const size_t words = static_cast<size_t>(n_snapshots) * cache_n_days * PyRandom::N;
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_cache), words * sizeof(uint32_t)));
    }
    KAGCUDA_CHECK(cudaMemcpy(d_base0, base0.actions.data(), base0.actions.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_base1, base1.actions.data(), base1.actions.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_candidates, candidate_actions.data(), candidate_actions.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_descs, candidate_descs.data(), candidate_descs.size() * sizeof(TapeDesc), cudaMemcpyHostToDevice));

    int tpb = 64;
    if (const char* s = std::getenv("KAGCUDA_TPB")) {
        const int v = std::atoi(s);
        if (v > 0 && v <= 1024) tpb = v;
    }
    // Warm all kernels/context before timing.
    make_tape_snapshots_kernel<<<(n_snapshots + tpb - 1) / tpb, tpb>>>(
        cfg, tables.dev, d_base0, static_cast<int>(base0.actions.size()),
        d_base1, static_cast<int>(base1.actions.size()), prefix, n_snapshots, d_prefix_sims, d_snapshots);
    KAGCUDA_CHECK(cudaGetLastError());
    if (cache_n_days > 0) {
        const int entries = n_snapshots * cache_n_days;
        build_daily_rng_cache_kernel<<<(entries + 127) / 128, 128>>>(
            d_snapshots, n_snapshots, cache_first_day, cache_n_days, tables.dev.mt_base, d_rng_cache);
        KAGCUDA_CHECK(cudaGetLastError());
    }
    branch_tape_candidates_kernel<<<(total_branches + tpb - 1) / tpb, tpb>>>(
        tables.dev, d_snapshots, n_snapshots, d_base0, static_cast<int>(base0.actions.size()),
        d_base1, static_cast<int>(base1.actions.size()), d_candidates, d_descs, n_candidates,
        candidate_player, prefix, d_rng_cache, cache_first_day, cache_n_days, d_branch_sims, d_rewards);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, prefix_done, cache_done, stop;
    KAGCUDA_CHECK(cudaEventCreate(&start)); KAGCUDA_CHECK(cudaEventCreate(&prefix_done));
    KAGCUDA_CHECK(cudaEventCreate(&cache_done)); KAGCUDA_CHECK(cudaEventCreate(&stop));
    KAGCUDA_CHECK(cudaEventRecord(start));
    make_tape_snapshots_kernel<<<(n_snapshots + tpb - 1) / tpb, tpb>>>(
        cfg, tables.dev, d_base0, static_cast<int>(base0.actions.size()),
        d_base1, static_cast<int>(base1.actions.size()), prefix, n_snapshots, d_prefix_sims, d_snapshots);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaEventRecord(prefix_done));
    if (cache_n_days > 0) {
        const int entries = n_snapshots * cache_n_days;
        build_daily_rng_cache_kernel<<<(entries + 127) / 128, 128>>>(
            d_snapshots, n_snapshots, cache_first_day, cache_n_days, tables.dev.mt_base, d_rng_cache);
        KAGCUDA_CHECK(cudaGetLastError());
    }
    KAGCUDA_CHECK(cudaEventRecord(cache_done));
    branch_tape_candidates_kernel<<<(total_branches + tpb - 1) / tpb, tpb>>>(
        tables.dev, d_snapshots, n_snapshots, d_base0, static_cast<int>(base0.actions.size()),
        d_base1, static_cast<int>(base1.actions.size()), d_candidates, d_descs, n_candidates,
        candidate_player, prefix, d_rng_cache, cache_first_day, cache_n_days, d_branch_sims, d_rewards);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaEventRecord(stop)); KAGCUDA_CHECK(cudaEventSynchronize(stop));

    float prefix_ms = 0.0f, cache_ms = 0.0f, branch_ms = 0.0f, total_ms = 0.0f;
    KAGCUDA_CHECK(cudaEventElapsedTime(&prefix_ms, start, prefix_done));
    KAGCUDA_CHECK(cudaEventElapsedTime(&cache_ms, prefix_done, cache_done));
    KAGCUDA_CHECK(cudaEventElapsedTime(&branch_ms, cache_done, stop));
    KAGCUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    std::vector<double> rewards(static_cast<size_t>(total_branches) * 2);
    KAGCUDA_CHECK(cudaMemcpy(rewards.data(), d_rewards, rewards.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::printf("CUDA tape branches: player=%d prefix=%d seeds=%llu..%llu snapshots=%d candidates=%d total=%d tpb=%d\n",
                candidate_player, prefix, static_cast<unsigned long long>(seed_start),
                static_cast<unsigned long long>(seed_start + n_snapshots - 1), n_snapshots, n_candidates, total_branches, tpb);
    std::printf("prefix=%.3f ms rng_cache=%.3f ms branch=%.3f ms total=%.3f ms\n",
                prefix_ms, cache_ms, branch_ms, total_ms);
    const double cps = total_branches / (branch_ms / 1000.0);
    const double first_cps = total_branches / (total_ms / 1000.0);
    std::printf("branch-only %.1f continuations/s; incl prefix+cache %.1f continuations/s\n", cps, first_cps);

    for (int c = 0; c < n_candidates; ++c) {
        int wins = 0;
        double sum = 0.0;
        double floor = DBL_MAX;
        for (int s = 0; s < n_snapshots; ++s) {
            const int id = s * n_candidates + c;
            const double r0 = rewards[id * 2], r1 = rewards[id * 2 + 1];
            const double margin = candidate_player == 0 ? r0 - r1 : r1 - r0;
            if (margin > 0) ++wins;
            sum += margin;
            floor = margin < floor ? margin : floor;
        }
        std::printf("candidate[%d] %s wins=%d/%d mean_margin=%.2f floor=%.0f",
                    c, basename_ptr(candidate_paths[c].c_str()), wins, n_snapshots, sum / n_snapshots, floor);
        if (n_snapshots == 1) {
            const int id = c;
            std::printf(" rewards=(%.0f,%.0f)", rewards[id*2], rewards[id*2+1]);
        }
        std::printf("\n");
    }

    cudaEventDestroy(start); cudaEventDestroy(prefix_done); cudaEventDestroy(cache_done); cudaEventDestroy(stop);
    cudaFree(d_base0); cudaFree(d_base1); cudaFree(d_candidates); cudaFree(d_descs);
    cudaFree(d_prefix_sims); cudaFree(d_snapshots); cudaFree(d_branch_sims); cudaFree(d_rewards);
    if (d_rng_cache) cudaFree(d_rng_cache);
    return 0;
}
