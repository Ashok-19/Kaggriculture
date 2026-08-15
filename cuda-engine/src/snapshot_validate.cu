#include "runtime.cuh"
#include "logical_hash.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace kagcuda;

static void read_config(std::istream& in, Config& cfg) {
    const std::streampos pos = in.tellg();
    std::string tok;
    if (!(in >> tok)) return;
    if (tok != "CONFIG") { in.clear(); in.seekg(pos); return; }
    in >> cfg.episode_steps >> cfg.board_size >> cfg.starting_money
       >> cfg.max_orders >> cfg.turns_per_day >> cfg.shed_capacity
       >> cfg.weed_chance >> cfg.shop_unlock_interval
       >> cfg.shop_sell_interval >> cfg.center_sell_interval >> cfg.hire_mult;
}

static bool load(const char* path, uint64_t& seed, Config& cfg, std::vector<Turn>& turns) {
    std::ifstream in(path);
    if (!in) return false;
    int n = 0; in >> seed >> n; read_config(in, cfg); turns.resize(n);
    for (int t = 0; t < n; ++t) for (int p = 0; p < 2; ++p) {
        int nu = 0, no = 0; in >> nu >> no;
        Action& a = turns[t].a[p]; a.n_units = imin(nu, MAX_UNITS);
        for (int i = 0; i < nu; ++i) {
            int op, arg, cnt; in >> op >> arg >> cnt;
            if (i < MAX_UNITS) a.units[i] = {static_cast<uint8_t>(op), static_cast<uint8_t>(arg), static_cast<int16_t>(cnt)};
        }
        a.n_orders = imin(no, MAX_ORDERS);
        for (int i = 0; i < no; ++i) {
            int op, item, cnt; in >> op >> item >> cnt;
            if (i < MAX_ORDERS) a.orders[i] = {static_cast<uint8_t>(op), static_cast<uint8_t>(item), cnt};
        }
    }
    return true;
}

static uint64_t hash_state(const State& s) {
    KagLogicalHash q;
    q.add_int(s.step); q.add_int(s.day); q.add_int(s.hour); q.add_bool(s.done); q.add_int(s.n_shops);
    for (int i = 0; i < s.n_shops; ++i) q.add_int(s.shops[i]);
    for (int i = 0; i < N_PRODUCTS; ++i) { q.add_int(s.market.inventory[i]); q.add_int(s.market.prices[i]); }
    for (int p = 0; p < 2; ++p) {
        const Farm& f = s.farms[p];
        q.add_double(f.money); q.add_int(f.n_units); q.add_int(f.n_quadrants); q.add_int(f.hires_today);
        q.add_int(f.next_decay_step); q.add_int(f.shed_total); q.add_double(f.sell_revenue); q.add_double(f.total_spend);
        for (int i = 0; i < N_ITEMS; ++i) { q.add_int(f.shed[i]); q.add_int(f.discarded[i]); q.add_int(f.produced[i]); q.add_int(f.sold_units[i]); }
        for (int i = 0; i < N_CROPS; ++i) q.add_int(f.seeds[i]);
        for (int u = 0; u < f.n_units; ++u) {
            q.add_int(f.pos_x[u]); q.add_int(f.pos_y[u]); q.add_int(f.inv_nkeys[u]);
            for (int k = 0; k < f.inv_nkeys[u]; ++k) q.add_int(f.inv_keys[u][k]);
            for (int i = 0; i < N_ITEMS; ++i) q.add_int(f.inv[u][i]);
        }
        for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) {
            const Tile& t = f.tiles[y][x];
            q.add_int(t.kind); q.add_int(t.what); q.add_bool(t.has_animal); q.add_bool(t.watered_today);
            q.add_bool(t.fed_today); q.add_bool(t.cared_today); q.add_bool(t.fertilizer_available);
            q.add_int(t.consecutive_dry); q.add_int(t.yield_units); q.add_int(t.pending_care_bonus);
            q.add_int(t.planted_day); q.add_int(t.max_lifespan_step); q.add_int(t.fertilized_until_day);
        }
    }
    return q.h;
}

__global__ void continuous_kernel(Config cfg, DeviceTables tables, const Turn* turns, int n_turns,
                                  Sim* sim, State* final_state) {
    if (blockIdx.x || threadIdx.x) return;
    sim->init(cfg, &tables);
    for (int t = 0; t < n_turns; ++t) sim->step(turns[t].a[0], turns[t].a[1]);
    *final_state = sim->st;
}

__global__ void prefix_snapshot_kernel(Config cfg, DeviceTables tables, const Turn* turns, int prefix,
                                       Sim* sim, Snapshot* snap) {
    if (blockIdx.x || threadIdx.x) return;
    sim->init(cfg, &tables);
    for (int t = 0; t < prefix; ++t) sim->step(turns[t].a[0], turns[t].a[1]);
    sim->save_snapshot(snap);
}

template <bool USE_RNG_CACHE>
__global__ void resume_snapshot_kernel(DeviceTables tables, const Turn* turns, int prefix, int n_turns,
                                       const Snapshot* snap, const uint32_t* rng_cache,
                                       int cache_first_day, int cache_n_days,
                                       Sim* sim, State* final_state) {
    if (blockIdx.x || threadIdx.x) return;
    sim->load_snapshot(*snap, &tables, USE_RNG_CACHE ? rng_cache : nullptr,
                       USE_RNG_CACHE ? cache_first_day : 0,
                       USE_RNG_CACHE ? cache_n_days : 0);
    for (int t = prefix; t < n_turns; ++t) {
        if constexpr (USE_RNG_CACHE) sim->step_cached(turns[t].a[0], turns[t].a[1]);
        else sim->step(turns[t].a[0], turns[t].a[1]);
    }
    *final_state = sim->st;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: snapshot_validate TRACE [TRACE...]\n"); return 2; }
    configure_device(); TablesOwner tables; int failures = 0;
    const int requested[] = {0, 1, 23, 24, 71, 72, 119, 120, 143, 144, 167, 168, 300, 600, 718, 719};

    for (int arg = 1; arg < argc; ++arg) {
        uint64_t seed = 0; Config cfg; std::vector<Turn> turns;
        if (!load(argv[arg], seed, cfg, turns)) { ++failures; continue; }
        cfg.seed = seed;
        Turn* d_turns = nullptr; Sim *d_a = nullptr, *d_b = nullptr; Snapshot* d_snap = nullptr;
        State *d_ref = nullptr, *d_out = nullptr;
        uint32_t* d_rng_cache = nullptr;
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_turns), turns.size() * sizeof(Turn)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(Sim)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(Sim)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_snap), sizeof(Snapshot)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ref), sizeof(State)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(State)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_cache), static_cast<size_t>(32) * PyRandom::N * sizeof(uint32_t)));
        KAGCUDA_CHECK(cudaMemcpy(d_turns, turns.data(), turns.size() * sizeof(Turn), cudaMemcpyHostToDevice));

        continuous_kernel<<<1,1>>>(cfg, tables.dev, d_turns, static_cast<int>(turns.size()), d_a, d_ref);
        KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
        State ref{}; KAGCUDA_CHECK(cudaMemcpy(&ref, d_ref, sizeof(State), cudaMemcpyDeviceToHost));
        const uint64_t ref_hash = hash_state(ref);

        bool ok = true;
        for (int prefix : requested) {
            if (prefix > static_cast<int>(turns.size())) continue;
            prefix_snapshot_kernel<<<1,1>>>(cfg, tables.dev, d_turns, prefix, d_a, d_snap);
            KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
            resume_snapshot_kernel<false><<<1,1>>>(tables.dev, d_turns, prefix, static_cast<int>(turns.size()),
                                                    d_snap, nullptr, 0, 0, d_b, d_out);
            KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
            State out{}; KAGCUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(State), cudaMemcpyDeviceToHost));
            const uint64_t h = hash_state(out);
            if (h != ref_hash) {
                std::printf("FAIL-SNAPSHOT %s prefix=%d uncached_hash=%016llx ref=%016llx\n", argv[arg], prefix,
                            static_cast<unsigned long long>(h), static_cast<unsigned long long>(ref_hash));
                ok = false; break;
            }

            const int cache_first_day = prefix / cfg.turns_per_day;
            const int final_day = turns.empty() ? cache_first_day : (static_cast<int>(turns.size()) - 1) / cfg.turns_per_day;
            const int cache_n_days = final_day >= cache_first_day ? final_day - cache_first_day + 1 : 0;
            if (cache_n_days > 32) { std::printf("FAIL-SNAPSHOT cache range too large\n"); ok = false; break; }
            if (cache_n_days > 0) {
                build_daily_rng_cache_kernel<<<1,32>>>(d_snap,1,cache_first_day,cache_n_days,tables.dev.mt_base,d_rng_cache);
                KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
            }
            resume_snapshot_kernel<true><<<1,1>>>(tables.dev, d_turns, prefix, static_cast<int>(turns.size()),
                                                   d_snap, d_rng_cache, cache_first_day, cache_n_days, d_b, d_out);
            KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
            KAGCUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(State), cudaMemcpyDeviceToHost));
            const uint64_t cached_h = hash_state(out);
            if (cached_h != ref_hash) {
                std::printf("FAIL-SNAPSHOT %s prefix=%d cached_hash=%016llx ref=%016llx\n", argv[arg], prefix,
                            static_cast<unsigned long long>(cached_h), static_cast<unsigned long long>(ref_hash));
                ok = false; break;
            }
        }
        if (ok) std::printf("PASS-SNAPSHOT %s seed=%llu boundaries=16 uncached+cached exact\n", argv[arg], static_cast<unsigned long long>(seed));
        else ++failures;
        cudaFree(d_turns); cudaFree(d_a); cudaFree(d_b); cudaFree(d_snap); cudaFree(d_ref); cudaFree(d_out); cudaFree(d_rng_cache);
    }
    return failures ? 1 : 0;
}
