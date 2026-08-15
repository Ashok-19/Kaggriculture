#include "runtime.cuh"
#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace kagcuda;

struct Checkpoint {
    double money[2];
    int32_t inventory[N_PRODUCTS];
    int32_t prices[N_PRODUCTS];
    int step, day, hour, n_shops;
    uint8_t shops[MAX_SHOP_INSTANCES];
};

__device__ __forceinline__ void capture(const Sim& sim, Checkpoint& cp) {
    cp.money[0] = sim.st.farms[0].money;
    cp.money[1] = sim.st.farms[1].money;
    for (int i = 0; i < N_PRODUCTS; ++i) {
        cp.inventory[i] = sim.st.market.inventory[i];
        cp.prices[i] = sim.st.market.prices[i];
    }
    cp.step = sim.st.step; cp.day = sim.st.day; cp.hour = sim.st.hour; cp.n_shops = sim.st.n_shops;
    for (int i = 0; i < MAX_SHOP_INSTANCES; ++i) cp.shops[i] = sim.st.shops[i];
}

__global__ void run_trace_kernel(Config cfg, DeviceTables tables, const Turn* turns,
                                 int n_turns, Checkpoint* checkpoints, State* final_state,
                                 Sim* sim_mem) {
    if (blockIdx.x || threadIdx.x) return;
    Sim& sim = sim_mem[0];
    sim.init(cfg, &tables);
    capture(sim, checkpoints[0]);
    for (int t = 0; t < n_turns; ++t) {
        sim.step(turns[t].a[0], turns[t].a[1]);
        capture(sim, checkpoints[t + 1]);
    }
    *final_state = sim.st;
}

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

static bool load(const std::string& path, uint64_t& seed, Config& cfg,
                 std::vector<Turn>& turns,
                 std::vector<std::array<double,2>>& money,
                 std::vector<std::array<int,N_PRODUCTS>>& inv) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    int n = 0;
    in >> seed >> n;
    read_config(in, cfg);
    turns.resize(n);
    for (int t = 0; t < n; ++t) {
        for (int p = 0; p < 2; ++p) {
            int nu = 0, no = 0; in >> nu >> no;
            Action& a = turns[t].a[p];
            a.n_units = imin(nu, MAX_UNITS);
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
    }
    std::string tag; in >> tag;
    if (tag != "TRUTH") return false;
    double m0, m1;
    while (in >> m0 >> m1) {
        money.push_back({m0,m1});
        std::array<int,N_PRODUCTS> row{};
        for (int i = 0; i < N_PRODUCTS; ++i) in >> row[i];
        inv.push_back(row);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: validate TRACE [TRACE...]\n");
        return 2;
    }
    configure_device();
    TablesOwner tables;
    int failures = 0;

    for (int arg = 1; arg < argc; ++arg) {
        uint64_t seed = 0;
        Config cfg;
        std::vector<Turn> turns;
        std::vector<std::array<double,2>> money;
        std::vector<std::array<int,N_PRODUCTS>> inv;
        if (!load(argv[arg], seed, cfg, turns, money, inv)) { ++failures; continue; }
        cfg.seed = seed;
        if (money.size() != turns.size() + 1 || inv.size() != money.size()) {
            std::fprintf(stderr, "bad truth length in %s: turns=%zu truth=%zu\n", argv[arg], turns.size(), money.size());
            ++failures; continue;
        }

        Turn* d_turns = nullptr;
        Checkpoint* d_cp = nullptr;
        State* d_final = nullptr;
        Sim* d_sim = nullptr;
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_turns), turns.size() * sizeof(Turn)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cp), money.size() * sizeof(Checkpoint)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_final), sizeof(State)));
        KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sim), sizeof(Sim)));
        KAGCUDA_CHECK(cudaMemcpy(d_turns, turns.data(), turns.size() * sizeof(Turn), cudaMemcpyHostToDevice));

        run_trace_kernel<<<1,1>>>(cfg, tables.dev, d_turns, static_cast<int>(turns.size()), d_cp, d_final, d_sim);
        KAGCUDA_CHECK(cudaGetLastError());
        KAGCUDA_CHECK(cudaDeviceSynchronize());

        std::vector<Checkpoint> cp(money.size());
        KAGCUDA_CHECK(cudaMemcpy(cp.data(), d_cp, cp.size() * sizeof(Checkpoint), cudaMemcpyDeviceToHost));
        State final_state{};
        KAGCUDA_CHECK(cudaMemcpy(&final_state, d_final, sizeof(State), cudaMemcpyDeviceToHost));

        int first_bad = -1;
        const char* what = "";
        int bad_index = -1;
        for (size_t t = 0; t < cp.size() && first_bad < 0; ++t) {
            for (int p = 0; p < 2; ++p) if (cp[t].money[p] != money[t][p]) {
                first_bad = static_cast<int>(t); what = "money"; bad_index = p; break;
            }
            for (int i = 0; i < N_PRODUCTS && first_bad < 0; ++i) if (cp[t].inventory[i] != inv[t][i]) {
                first_bad = static_cast<int>(t); what = "inventory"; bad_index = i; break;
            }
        }

        if (first_bad >= 0) {
            std::printf("FAIL %s seed=%llu first %s mismatch checkpoint=%d index=%d\n",
                        argv[arg], static_cast<unsigned long long>(seed), what, first_bad, bad_index);
            if (what[0] == 'm')
                std::printf("     cuda=(%.0f,%.0f) truth=(%.0f,%.0f)\n",
                            cp[first_bad].money[0], cp[first_bad].money[1],
                            money[first_bad][0], money[first_bad][1]);
            else
                std::printf("     cuda inv[%d]=%d truth=%d\n", bad_index,
                            cp[first_bad].inventory[bad_index], inv[first_bad][bad_index]);
            ++failures;
        } else {
            std::printf("PASS %s seed=%-8llu %zu steps exact final=(%.0f,%.0f) shops=%d\n",
                        argv[arg], static_cast<unsigned long long>(seed), turns.size(),
                        final_state.farms[0].money, final_state.farms[1].money, final_state.n_shops);
        }

        cudaFree(d_turns); cudaFree(d_cp); cudaFree(d_final); cudaFree(d_sim);
    }
    return failures ? 1 : 0;
}
