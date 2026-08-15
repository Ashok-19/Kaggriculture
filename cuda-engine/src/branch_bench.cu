#include "runtime.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace kagcuda;

struct CandidatePlan { int tape0; int tape1; };

static void read_config(std::istream& in, Config& cfg) {
    const std::streampos pos = in.tellg();
    std::string tok;
    if (!(in >> tok)) return;
    if (tok != "CONFIG") { in.clear(); in.seekg(pos); return; }
    in >> cfg.episode_steps >> cfg.board_size >> cfg.starting_money >> cfg.max_orders >> cfg.turns_per_day
       >> cfg.shed_capacity >> cfg.weed_chance >> cfg.shop_unlock_interval >> cfg.shop_sell_interval
       >> cfg.center_sell_interval >> cfg.hire_mult;
}

static bool load(const char* path, uint64_t& seed, Config& cfg, std::vector<Turn>& turns) {
    std::ifstream in(path);
    if (!in) return false;
    int n = 0;
    in >> seed >> n;
    read_config(in, cfg);
    turns.resize(n);
    for (int t = 0; t < n; ++t) for (int p = 0; p < 2; ++p) {
        int nu = 0, no = 0;
        in >> nu >> no;
        Action& a = turns[t].a[p];
        a.n_units = imin(nu, MAX_UNITS);
        for (int i = 0; i < nu; ++i) {
            int op, arg, cnt;
            in >> op >> arg >> cnt;
            if (i < MAX_UNITS)
                a.units[i] = {static_cast<uint8_t>(op), static_cast<uint8_t>(arg), static_cast<int16_t>(cnt)};
        }
        a.n_orders = imin(no, MAX_ORDERS);
        for (int i = 0; i < no; ++i) {
            int op, item, cnt;
            in >> op >> item >> cnt;
            if (i < MAX_ORDERS)
                a.orders[i] = {static_cast<uint8_t>(op), static_cast<uint8_t>(item), cnt};
        }
    }
    return true;
}

__global__ void make_snapshot_batch_kernel(Config base, DeviceTables tables, const Turn* turns,
                                           int prefix, int n_snapshots, Sim* sims, Snapshot* snapshots) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_snapshots) return;
    Config cfg = base;
    cfg.seed = base.seed + static_cast<uint64_t>(id);
    Sim& sim = sims[id];
    sim.init(cfg, &tables);
    for (int t = 0; t < prefix; ++t) {
        const int units0 = sim.st.farms[0].n_units;
        const int units1 = sim.st.farms[1].n_units;
        sim.step_with_unit_counts(turns[t].a[0], units0, turns[t].a[1], units1);
    }
    sim.save_snapshot(&snapshots[id]);
}

// Cross product of exact snapshots and candidate tape pairs. Every thread owns
// one (snapshot, candidate) continuation and performs no host/CPU simulation.
template <bool USE_RNG_CACHE>
__global__ void branch_grid_kernel(DeviceTables tables, const Snapshot* snapshots, int n_snapshots,
                                   const Action* player0_tapes, const Action* player1_tapes, int tape_stride,
                                   const CandidatePlan* candidates, int n_candidates, int start_step, int end_step,
                                   const uint32_t* seed_cache, int cache_first_day, int cache_n_days,
                                   Sim* sims, double* margins) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_snapshots * n_candidates;
    if (id >= total) return;
    const int snapshot_id = id / n_candidates;
    const int candidate_id = id - snapshot_id * n_candidates;

    const uint32_t* snapshot_cache = nullptr;
    if (seed_cache)
        snapshot_cache = seed_cache + static_cast<size_t>(snapshot_id) * cache_n_days * PyRandom::N;

    Sim& sim = sims[id];
    sim.load_snapshot(snapshots[snapshot_id], &tables, snapshot_cache, cache_first_day, cache_n_days);
    const CandidatePlan plan = candidates[candidate_id];
    const Action* a0 = player0_tapes + static_cast<size_t>(plan.tape0) * tape_stride;
    const Action* a1 = player1_tapes + static_cast<size_t>(plan.tape1) * tape_stride;
    for (int t = start_step; t < end_step; ++t) {
        const int units0 = sim.st.farms[0].n_units;
        const int units1 = sim.st.farms[1].n_units;
        if constexpr (USE_RNG_CACHE)
            sim.step_with_unit_counts_cached(a0[t], units0, a1[t], units1);
        else
            sim.step_with_unit_counts(a0[t], units0, a1[t], units1);
    }
    margins[id] = sim.st.farms[0].money - sim.st.farms[1].money;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: branch_bench TRACE [PREFIX=168] [CANDIDATES=28] [SNAPSHOTS=128] [TPB=128] [RNG_CACHE=1]\n");
        return 2;
    }
    const int prefix = argc > 2 ? std::atoi(argv[2]) : 168;
    const int n_candidates = argc > 3 ? std::atoi(argv[3]) : 28;
    const int n_snapshots = argc > 4 ? std::atoi(argv[4]) : 128;
    const int tpb = argc > 5 ? std::atoi(argv[5]) : 128;
    const bool use_rng_cache = argc > 6 ? std::atoi(argv[6]) != 0 : true;
    if (n_candidates <= 0 || n_snapshots <= 0 || tpb <= 0 || tpb > 1024) return 2;

    configure_device();
    TablesOwner tables;
    uint64_t seed = 0;
    Config cfg;
    std::vector<Turn> turns;
    if (!load(argv[1], seed, cfg, turns)) return 2;
    cfg.seed = seed;
    if (prefix < 0 || prefix > static_cast<int>(turns.size())) return 2;
    const int stride = static_cast<int>(turns.size());
    const int total = n_candidates * n_snapshots;

    // The benchmark repeats one tape pair. The API accepts arbitrary tape ids,
    // so real research uploads each distinct donor/policy tape once and points
    // candidate plans at that library.
    std::vector<Action> h0(turns.size()), h1(turns.size());
    for (size_t t = 0; t < turns.size(); ++t) {
        h0[t] = turns[t].a[0];
        h1[t] = turns[t].a[1];
    }
    std::vector<CandidatePlan> candidates(n_candidates);
    for (int i = 0; i < n_candidates; ++i) candidates[i] = {0, 0};

    Turn* d_turns = nullptr;
    Action *d0 = nullptr, *d1 = nullptr;
    CandidatePlan* d_candidates = nullptr;
    Sim *d_prefix_sims = nullptr, *d_branch_sims = nullptr;
    Snapshot* d_snapshots = nullptr;
    double* d_margins = nullptr;
    uint32_t* d_seed_cache = nullptr;

    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_turns), turns.size() * sizeof(Turn)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d0), h0.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d1), h1.size() * sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidates), candidates.size() * sizeof(CandidatePlan)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prefix_sims), static_cast<size_t>(n_snapshots) * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_snapshots), static_cast<size_t>(n_snapshots) * sizeof(Snapshot)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_branch_sims), static_cast<size_t>(total) * sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_margins), static_cast<size_t>(total) * sizeof(double)));
    KAGCUDA_CHECK(cudaMemcpy(d_turns, turns.data(), turns.size() * sizeof(Turn), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d0, h0.data(), h0.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d1, h1.data(), h1.size() * sizeof(Action), cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), candidates.size() * sizeof(CandidatePlan), cudaMemcpyHostToDevice));

    make_snapshot_batch_kernel<<<(n_snapshots + 127) / 128, 128>>>(
        cfg, tables.dev, d_turns, prefix, n_snapshots, d_prefix_sims, d_snapshots);
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaDeviceSynchronize());

    const int cache_first_day = prefix / cfg.turns_per_day;
    const int final_day = stride > 0 ? (stride - 1) / cfg.turns_per_day : cache_first_day;
    const int cache_n_days = final_day >= cache_first_day ? final_day - cache_first_day + 1 : 0;
    size_t cache_words = 0;
    if (use_rng_cache && cache_n_days > 0) {
        cache_words = static_cast<size_t>(n_snapshots) * cache_n_days * PyRandom::N;
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_seed_cache), cache_words * sizeof(uint32_t)));
        const int entries = n_snapshots * cache_n_days;
        build_daily_rng_cache_kernel<<<(entries + 127) / 128, 128>>>(
            d_snapshots, n_snapshots, cache_first_day, cache_n_days, tables.dev.mt_base, d_seed_cache);
        KAGCUDA_CHECK(cudaGetLastError());
        KAGCUDA_CHECK(cudaDeviceSynchronize());
    }

    if (use_rng_cache) {
        branch_grid_kernel<true><<<(total + tpb - 1) / tpb, tpb>>>(
            tables.dev, d_snapshots, n_snapshots, d0, d1, stride, d_candidates, n_candidates,
            prefix, stride, d_seed_cache, cache_first_day, cache_n_days, d_branch_sims, d_margins);
    } else {
        branch_grid_kernel<false><<<(total + tpb - 1) / tpb, tpb>>>(
            tables.dev, d_snapshots, n_snapshots, d0, d1, stride, d_candidates, n_candidates,
            prefix, stride, nullptr, 0, 0, d_branch_sims, d_margins);
    }
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    KAGCUDA_CHECK(cudaEventCreate(&start));
    KAGCUDA_CHECK(cudaEventCreate(&stop));
    KAGCUDA_CHECK(cudaEventRecord(start));
    if (use_rng_cache) {
        branch_grid_kernel<true><<<(total + tpb - 1) / tpb, tpb>>>(
            tables.dev, d_snapshots, n_snapshots, d0, d1, stride, d_candidates, n_candidates,
            prefix, stride, d_seed_cache, cache_first_day, cache_n_days, d_branch_sims, d_margins);
    } else {
        branch_grid_kernel<false><<<(total + tpb - 1) / tpb, tpb>>>(
            tables.dev, d_snapshots, n_snapshots, d0, d1, stride, d_candidates, n_candidates,
            prefix, stride, nullptr, 0, 0, d_branch_sims, d_margins);
    }
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaEventRecord(stop));
    KAGCUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    KAGCUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<double> margins(total);
    KAGCUDA_CHECK(cudaMemcpy(margins.data(), d_margins, margins.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double sink = 0.0;
    for (double v : margins) sink += v;
    const int remaining = stride - prefix;
    const double sec = ms / 1000.0;
    const double cps = total / sec;
    const double transitions = cps * remaining;
    std::printf("GPU snapshot grid: snapshots=%d candidates=%d total=%d prefix=%d remaining=%d in %.3f ms\n",
                n_snapshots, n_candidates, total, prefix, remaining, ms);
    std::printf("%.1f continuations/s = %.3f M transitions/s, block=%d sink=%.0f\n",
                cps, transitions / 1e6, tpb, sink);
    std::printf("snapshot_bytes=%zu sim_bytes=%zu branch_state_MiB=%.2f rng_cache=%s cache_MiB=%.2f\n",
                sizeof(Snapshot), sizeof(Sim), static_cast<double>(total) * sizeof(Sim) / (1024.0 * 1024.0),
                use_rng_cache ? "on" : "off", static_cast<double>(cache_words * sizeof(uint32_t)) / (1024.0 * 1024.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_turns); cudaFree(d0); cudaFree(d1); cudaFree(d_candidates);
    cudaFree(d_prefix_sims); cudaFree(d_snapshots); cudaFree(d_branch_sims); cudaFree(d_margins);
    if (d_seed_cache) cudaFree(d_seed_cache);
    return 0;
}
