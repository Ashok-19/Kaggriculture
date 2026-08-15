#include "runtime.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

static void load(const std::string& path, uint64_t& seed, Config& cfg, std::vector<Turn>& turns) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    int n = 0; in >> seed >> n; read_config(in,cfg); turns.resize(n);
    for (int t = 0; t < n; ++t) for (int p = 0; p < 2; ++p) {
        int nu=0,no=0; in >> nu >> no; Action& a=turns[t].a[p]; a.n_units=imin(nu,MAX_UNITS);
        for(int i=0;i<nu;++i){int op,arg,cnt;in>>op>>arg>>cnt;if(i<MAX_UNITS)a.units[i]={static_cast<uint8_t>(op),static_cast<uint8_t>(arg),static_cast<int16_t>(cnt)};}
        a.n_orders=imin(no,MAX_ORDERS);
        for(int i=0;i<no;++i){int op,item,cnt;in>>op>>item>>cnt;if(i<MAX_ORDERS)a.orders[i]={static_cast<uint8_t>(op),static_cast<uint8_t>(item),cnt};}
    }
}

__global__ void run_batch_kernel(Config base, DeviceTables tables, const Turn* turns,
                                 int n_turns, int episodes, double* rewards, Sim* sims) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= episodes) return;
    Config cfg = base;
    cfg.seed = base.seed + static_cast<uint64_t>(id);
    Sim& sim = sims[id];
    sim.init(cfg, &tables);
    for (int t = 0; t < n_turns; ++t) sim.step(turns[t].a[0], turns[t].a[1]);
    rewards[id * 2] = sim.st.farms[0].money;
    rewards[id * 2 + 1] = sim.st.farms[1].money;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench TRACE [EPISODES] [THREADS_PER_BLOCK]\n");
        return 2;
    }
    const int episodes = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 128;
    if (episodes <= 0 || threads <= 0 || threads > 1024) return 2;

    configure_device();
    TablesOwner tables;
    uint64_t seed = 0; Config cfg; std::vector<Turn> turns;
    load(argv[1],seed,cfg,turns); cfg.seed=seed;

    Turn* d_turns=nullptr; double* d_rewards=nullptr; Sim* d_sims=nullptr;
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_turns),turns.size()*sizeof(Turn)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rewards),static_cast<size_t>(episodes)*2*sizeof(double)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sims),static_cast<size_t>(episodes)*sizeof(Sim)));
    KAGCUDA_CHECK(cudaMemcpy(d_turns,turns.data(),turns.size()*sizeof(Turn),cudaMemcpyHostToDevice));

    // Warmup catches launch/resource errors and removes JIT/context startup from timing.
    run_batch_kernel<<<(episodes+threads-1)/threads,threads>>>(cfg,tables.dev,d_turns,static_cast<int>(turns.size()),episodes,d_rewards,d_sims);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start,stop; KAGCUDA_CHECK(cudaEventCreate(&start)); KAGCUDA_CHECK(cudaEventCreate(&stop));
    const auto wall0=std::chrono::steady_clock::now();
    KAGCUDA_CHECK(cudaEventRecord(start));
    run_batch_kernel<<<(episodes+threads-1)/threads,threads>>>(cfg,tables.dev,d_turns,static_cast<int>(turns.size()),episodes,d_rewards,d_sims);
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaEventRecord(stop)); KAGCUDA_CHECK(cudaEventSynchronize(stop));
    const auto wall1=std::chrono::steady_clock::now();
    float ms=0.0f; KAGCUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));

    std::vector<double> rewards(static_cast<size_t>(episodes)*2);
    KAGCUDA_CHECK(cudaMemcpy(rewards.data(),d_rewards,rewards.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double sink=0.0; for(double v:rewards) sink+=v;
    const double sec=ms/1000.0;
    const double eps=episodes/sec;
    const double transitions=eps*turns.size();
    const double wall=std::chrono::duration<double>(wall1-wall0).count();
    std::printf("GPU batch: %d episodes x %zu turns in %.3f ms = %.1f eps/s = %.3f M transitions/s\n",
                episodes,turns.size(),ms,eps,transitions/1e6);
    std::printf("host wall %.6fs, block=%d, sink=%.0f\n",wall,threads,sink);

    cudaEventDestroy(start); cudaEventDestroy(stop); cudaFree(d_turns); cudaFree(d_rewards); cudaFree(d_sims);
    return 0;
}
