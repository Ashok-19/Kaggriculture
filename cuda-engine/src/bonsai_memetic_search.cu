#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include "bonsai_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

namespace {

constexpr int SEARCH_MAX_RULES = 4;
constexpr int SEARCH_NICHES = 256;
constexpr int SEARCH_MAX_FAMILIES = 32;
constexpr int SEARCH_MODES = 8;
constexpr int SEARCH_LAYOUT_CATS = 8;
constexpr int SEARCH_DEFAULT_SUBISLANDS = 32;
constexpr uint64_t SEARCH_MAGIC = 0x314D454D474B4255ull; // "UBKGMEM1" little-endian-ish marker.
volatile std::sig_atomic_t SEARCH_STOP_REQUESTED = 0;
void search_signal_handler(int) { SEARCH_STOP_REQUESTED = 1; }

// Search modes intentionally map to deterministic Bonsai execution primitives.
enum SearchMode : uint8_t {
    SM_MANAGER = 0,
    SM_ROUTE_UNITS_MANAGER_MARKET = 1,
    SM_ROUTE_UNITS_PLANNED_MARKET = 2,
    SM_MANAGER_UNITS_ROUTE_MARKET = 3,
    SM_ROUTE = 4,
    SM_MANAGER_ENGINE_MARKET = 5,
    SM_ROUTE_UNITS_ENGINE_MARKET = 6,
    SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET = 7,
};

struct CompactRule {
    uint8_t feature = GPF_ALWAYS;
    uint8_t cmp = GPC_EQ;
    uint8_t item = 0;
    uint8_t value_lo = 0;
    uint8_t value_hi = 0;
    uint8_t mode = SM_MANAGER;
    uint8_t unit_route = 0;
    uint8_t market_route = 0;
};
static_assert(sizeof(CompactRule) == 8, "CompactRule must remain 8 bytes");

// Exactly 256 bytes. The hot search never stores 4 KiB GPProgram objects per candidate.
// Every field is a strategic Bonsai-compatible gene and deterministically distills to GPProgram.
struct SearchGenome {
    uint8_t phase_step12[GP_PHASES - 1] = {0};                       // 10
    uint8_t phase_layout_seed[GP_PHASES] = {0};                      // 11 => 21
    uint8_t phase_layout_weight4[(GP_PHASES * SEARCH_LAYOUT_CATS) / 2] = {0}; // 44 => 65
    uint8_t phase_target_quadrants[GP_PHASES] = {0};                 // 11 => 76
    uint8_t phase_hires_per_day[GP_PHASES] = {0};                    // 11 => 87
    uint8_t phase_cow_target[GP_PHASES] = {0};                       // 11 => 98
    uint8_t phase_sheep_target[GP_PHASES] = {0};                     // 11 => 109
    uint8_t phase_goose_target[GP_PHASES] = {0};                     // 11 => 120
    uint8_t phase_cash100[GP_PHASES] = {0};                          // 11 => 131
    uint8_t phase_mode[GP_PHASES] = {0};                             // 11 => 142
    uint8_t phase_unit_route[GP_PHASES] = {0};                       // 11 => 153
    uint8_t phase_market_route[GP_PHASES] = {0};                     // 11 => 164

    uint8_t hire_until_hour = 0;
    uint8_t max_hires_per_turn = 0;
    uint8_t max_sell_orders = 0;
    uint8_t seed_buy_cap = 0;
    uint8_t animal_buy_cap = 0;
    uint8_t feed_stock = 0;
    uint8_t feed_load = 0;
    uint8_t fert_load = 0;
    uint8_t drop_load = 0;
    uint8_t drop_hour = 0;
    uint8_t rebuild_step_div12 = 0;
    uint8_t care_enabled = 0;
    uint8_t collect_fert_enabled = 0;
    uint8_t dist_penalty = 0;
    uint8_t planned_horizon = 0;
    uint8_t planned_seed_horizon = 0;
    uint8_t planned_animal_horizon = 0;
    uint8_t planned_pickup_horizon = 0;                              // 18 => 182
    uint8_t planned_seed_buffer[N_CROPS] = {0};                      // 5 => 187
    uint8_t planned_animal_buffer = 0;
    uint8_t planned_feed_buffer = 0;
    uint8_t planned_fert_buffer = 0;
    uint8_t copy_route_infrastructure = 0;                           // 4 => 191
    uint8_t sell_floor_pct[N_PRODUCTS] = {0};                        // 9 => 200
    uint8_t sell_reserve[N_PRODUCTS] = {0};                          // 9 => 209
    uint8_t sell_batch = 0;                                         // 1 => 210
    uint8_t task_priority[10] = {0};                                 // 10 => 220
    CompactRule rules[SEARCH_MAX_RULES];                             // 32 => 252
    uint8_t n_rules = 0;
    uint8_t engine_profile = 0;       // lockstep/town/$1/shared-externality regime
    uint8_t engine_batch_bias = 0;    // market unit batch/order-index search
    uint8_t engine_pressure = 0;      // high nibble=withhold, low nibble=empty-space/RNG pressure => 256
};
static_assert(sizeof(SearchGenome) == 256, "SearchGenome must remain exactly 256 bytes");

struct DeviceFitness {
    int wins = 0;
    int losses = 0;
    int ties = 0;
    int cases = 0;
    int regressions = 0;
    int losing_families = SEARCH_MAX_FAMILIES;
    int evaluated_families = 0;
    int worst_family_milli = 0;
    double floor = -DBL_MAX;
    double sum = 0.0;
    double marginal_delta = 0.0; // current common-case mean minus parent/common reference mean.
    uint64_t rank_key = 0;
    uint64_t signature = 0;
    uint8_t niche = 0;
    uint8_t pad[7] = {0};
};

struct IslandState {
    SearchGenome best_genome{};
    DeviceFitness best_fit{};
    int best_index = -1;
    int current_best_index = -1;
    int stagnation = 0;
    int resets = 0;
    uint64_t generations = 0;
    uint64_t local_refine_evals = 0;
    uint64_t improvements = 0;
    uint8_t initialized = 0;
    uint8_t reset_requested = 0;
    uint8_t pad[6] = {0};
};

struct CheckpointHeader {
    uint64_t magic = SEARCH_MAGIC;
    uint32_t version = 3;
    uint32_t sub_islands = 0;
    uint64_t generation = 0;
    uint64_t search_seed = 0;
    uint64_t candidates_evaluated = 0;
    uint64_t games_simulated = 0;
    double elapsed_seconds = 0.0;
};

struct DevEntry {
    std::string filename;
    std::string bundle_path;
    std::string team;
    int family = -1;
    uint64_t source_seed = 0;
    std::array<uint8_t, 2> baseline_win{{0,0}};
};

struct TierCase {
    int dev_index = -1;
    int family = -1;
    int seat = 0;
    uint64_t seed = 0;
    uint8_t baseline_win = 0;
};

struct TierSpec {
    std::string name;
    std::vector<TierCase> cases;
    int max_step = 720;
};

struct SearchBuffers {
    SearchGenome* pop_a = nullptr;
    SearchGenome* pop_b = nullptr;
    GPManager* mgr_a = nullptr;
    GPManager* mgr_b = nullptr;
    DeviceFitness* fit = nullptr;
    double* parent_mean_a = nullptr;
    double* parent_mean_b = nullptr;
    uint8_t* family_losses = nullptr;
    uint8_t* family_ties = nullptr;
    uint8_t* family_cases = nullptr;
    uint64_t* rng = nullptr;
    Sim* sims = nullptr;
    double* rewards = nullptr;
    IslandState* islands = nullptr;
    uint32_t* niche_counts = nullptr;
    unsigned long long* generation_niche_vote = nullptr;
    SearchGenome* archive_genomes = nullptr;
    DeviceFitness* archive_fit = nullptr;
    uint8_t* archive_valid = nullptr;
    SearchGenome* promo_genomes = nullptr;
    GPManager* promo_mgr = nullptr;
    DeviceFitness* promo_fit = nullptr;
    uint8_t* promo_family_losses = nullptr;
    uint8_t* promo_family_ties = nullptr;
    uint8_t* promo_family_cases = nullptr;
    Sim* promo_sims = nullptr;
    double* promo_rewards = nullptr;
    SearchGenome* certified_genomes = nullptr;
    DeviceFitness* certified_fit = nullptr;
    uint8_t* certified_level = nullptr;
    unsigned long long* promotion_games = nullptr;

    void clear() {
        if(pop_a) cudaFree(pop_a); if(pop_b) cudaFree(pop_b);
        if(mgr_a) cudaFree(mgr_a); if(mgr_b) cudaFree(mgr_b);
        if(fit) cudaFree(fit);
        if(parent_mean_a) cudaFree(parent_mean_a); if(parent_mean_b) cudaFree(parent_mean_b);
        if(family_losses) cudaFree(family_losses); if(family_ties) cudaFree(family_ties); if(family_cases) cudaFree(family_cases);
        if(rng) cudaFree(rng); if(sims) cudaFree(sims); if(rewards) cudaFree(rewards);
        if(islands) cudaFree(islands); if(niche_counts) cudaFree(niche_counts); if(generation_niche_vote) cudaFree(generation_niche_vote);
        if(archive_genomes) cudaFree(archive_genomes); if(archive_fit) cudaFree(archive_fit); if(archive_valid) cudaFree(archive_valid);
        if(promo_genomes) cudaFree(promo_genomes); if(promo_mgr) cudaFree(promo_mgr); if(promo_fit) cudaFree(promo_fit);
        if(promo_family_losses) cudaFree(promo_family_losses); if(promo_family_ties) cudaFree(promo_family_ties); if(promo_family_cases) cudaFree(promo_family_cases);
        if(promo_sims) cudaFree(promo_sims); if(promo_rewards) cudaFree(promo_rewards);
        if(certified_genomes) cudaFree(certified_genomes); if(certified_fit) cudaFree(certified_fit); if(certified_level) cudaFree(certified_level);
        if(promotion_games) cudaFree(promotion_games);
        *this = SearchBuffers{};
    }
};

static std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> out; std::string x; std::stringstream ss(s);
    while(std::getline(ss,x,'\t')) { if(!x.empty() && x.back()=='\r') x.pop_back(); out.push_back(x); }
    return out;
}

static std::vector<std::string> read_list(const std::filesystem::path& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open list: " + path.string());
    const auto base = std::filesystem::absolute(path).parent_path();
    std::vector<std::string> out; std::string line;
    while(std::getline(in,line)) {
        const auto hash=line.find('#'); if(hash!=std::string::npos) line.resize(hash);
        const auto a=line.find_first_not_of(" \t\r\n"); if(a==std::string::npos) continue;
        const auto b=line.find_last_not_of(" \t\r\n");
        std::filesystem::path p=line.substr(a,b-a+1); if(p.is_relative()) p=base/p;
        out.push_back(p.lexically_normal().string());
    }
    return out;
}

static uint8_t leaf_kind_to_mode(uint8_t k) {
    if(k==GPL_ROUTE_UNITS_MANAGER_MARKET) return SM_ROUTE_UNITS_MANAGER_MARKET;
    if(k==GPL_ROUTE_UNITS_PLANNED_MARKET) return SM_ROUTE_UNITS_PLANNED_MARKET;
    if(k==GPL_MANAGER_UNITS_ROUTE_MARKET) return SM_MANAGER_UNITS_ROUTE_MARKET;
    if(k==GPL_MANAGER_ENGINE_MARKET) return SM_MANAGER_ENGINE_MARKET;
    if(k==GPL_ROUTE_UNITS_ENGINE_MARKET) return SM_ROUTE_UNITS_ENGINE_MARKET;
    if(k==GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET) return SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET;
    if(k==GPL_ROUTE || k==GPL_HYBRID) return SM_ROUTE;
    return SM_MANAGER;
}

static uint8_t mode_to_leaf_kind(uint8_t m) {
    switch(m % SEARCH_MODES) {
        case SM_ROUTE_UNITS_MANAGER_MARKET: return GPL_ROUTE_UNITS_MANAGER_MARKET;
        case SM_ROUTE_UNITS_PLANNED_MARKET: return GPL_ROUTE_UNITS_PLANNED_MARKET;
        case SM_MANAGER_UNITS_ROUTE_MARKET: return GPL_MANAGER_UNITS_ROUTE_MARKET;
        case SM_ROUTE: return GPL_ROUTE;
        case SM_MANAGER_ENGINE_MARKET: return GPL_MANAGER_ENGINE_MARKET;
        case SM_ROUTE_UNITS_ENGINE_MARKET: return GPL_ROUTE_UNITS_ENGINE_MARKET;
        case SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET: return GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET;
        default: return GPL_MANAGER;
    }
}

static uint8_t host_weight4_get(const SearchGenome& g, int ph, int k) {
    const int j=ph*SEARCH_LAYOUT_CATS+k, b=j>>1;
    return (j&1)?(g.phase_layout_weight4[b]>>4):(g.phase_layout_weight4[b]&15);
}
static void host_weight4_set(SearchGenome& g, int ph, int k, uint8_t v) {
    const int j=ph*SEARCH_LAYOUT_CATS+k,b=j>>1; v&=15;
    if(j&1) g.phase_layout_weight4[b]=(uint8_t)((g.phase_layout_weight4[b]&15)|(v<<4));
    else g.phase_layout_weight4[b]=(uint8_t)((g.phase_layout_weight4[b]&240)|v);
}

static SearchGenome encode_prior(const GPProgram& p, int n_routes) {
    SearchGenome g{}; const GPManager& m=p.manager;
    for(int i=0;i<GP_PHASES-1;++i) g.phase_step12[i]=(uint8_t)std::clamp((int)((m.phase_step[i]+6)/12),1,60);
    for(int ph=0;ph<GP_PHASES;++ph) {
        g.phase_layout_seed[ph]=m.phase_layout_seed[ph];
        for(int k=0;k<8;++k) host_weight4_set(g,ph,k,(uint8_t)std::clamp((int)((m.phase_layout_weight[ph][k]+2)/4),0,15));
        g.phase_target_quadrants[ph]=m.phase_target_quadrants[ph]; g.phase_hires_per_day[ph]=m.phase_hires_per_day[ph];
        g.phase_cow_target[ph]=m.phase_cow_target[ph]; g.phase_sheep_target[ph]=m.phase_sheep_target[ph]; g.phase_goose_target[ph]=m.phase_goose_target[ph];
        g.phase_cash100[ph]=(uint8_t)std::clamp((int)((m.phase_cash_reserve[ph]+50)/100),0,255);
    }
    GPLeaf fallback = p.rules[p.n_rules ? p.n_rules-1 : 0].leaf;
    for(int ph=0;ph<GP_PHASES;++ph) {
        g.phase_mode[ph]=leaf_kind_to_mode(fallback.kind);
        g.phase_unit_route[ph]=(uint8_t)(n_routes?fallback.unit_route%n_routes:0);
        g.phase_market_route[ph]=(uint8_t)(n_routes?fallback.market_route%n_routes:0);
    }
    g.hire_until_hour=m.hire_until_hour; g.max_hires_per_turn=m.max_hires_per_turn; g.max_sell_orders=m.max_sell_orders;
    g.seed_buy_cap=m.seed_buy_cap; g.animal_buy_cap=m.animal_buy_cap; g.feed_stock=m.feed_stock; g.feed_load=m.feed_load; g.fert_load=m.fert_load;
    g.drop_load=m.drop_load; g.drop_hour=m.drop_hour; g.rebuild_step_div12=m.rebuild_step_div12; g.care_enabled=m.care_enabled; g.collect_fert_enabled=m.collect_fert_enabled;
    g.dist_penalty=m.dist_penalty; g.planned_horizon=m.planned_horizon; g.planned_seed_horizon=m.planned_seed_horizon; g.planned_animal_horizon=m.planned_animal_horizon; g.planned_pickup_horizon=m.planned_pickup_horizon;
    for(int i=0;i<N_CROPS;++i) g.planned_seed_buffer[i]=m.planned_seed_buffer[i];
    g.planned_animal_buffer=m.planned_animal_buffer; g.planned_feed_buffer=m.planned_feed_buffer; g.planned_fert_buffer=m.planned_fert_buffer; g.copy_route_infrastructure=m.copy_route_infrastructure;
    for(int i=0;i<N_PRODUCTS;++i){g.sell_floor_pct[i]=m.sell_floor_pct[i];g.sell_reserve[i]=m.sell_reserve[i];}
    g.sell_batch=m.sell_batch; for(int i=0;i<10;++i)g.task_priority[i]=m.task_priority[i];
    g.n_rules=0;
    for(int r=0;r<p.n_rules && g.n_rules<SEARCH_MAX_RULES;++r) {
        const GPRule& q=p.rules[r]; if(q.n_conds<1) continue;
        const GPCond& c=q.conds[0]; CompactRule& z=g.rules[g.n_rules++];
        z.feature=c.feature;z.cmp=c.cmp;z.item=c.item; const int16_t v=(int16_t)std::clamp(c.value,-32768,32767);
        z.value_lo=(uint8_t)(v&255);z.value_hi=(uint8_t)((uint16_t)v>>8);z.mode=leaf_kind_to_mode(q.leaf.kind);
        z.unit_route=(uint8_t)(n_routes?q.leaf.unit_route%n_routes:0);z.market_route=(uint8_t)(n_routes?q.leaf.market_route%n_routes:0);
    }
    g.engine_batch_bias=p.pad[0];g.engine_pressure=p.pad[1];
    if(fallback.kind==GPL_MANAGER_ENGINE_MARKET || fallback.kind==GPL_ROUTE_UNITS_ENGINE_MARKET || fallback.kind==GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET) g.engine_profile=fallback.pad;
    return g;
}

static GPManager decode_manager_host(const SearchGenome& g) {
    GPManager m{};
    for(int i=0;i<GP_PHASES-1;++i)m.phase_step[i]=(uint16_t)g.phase_step12[i]*12;
    for(int ph=0;ph<GP_PHASES;++ph){
        m.phase_layout_seed[ph]=g.phase_layout_seed[ph] ? g.phase_layout_seed[ph] : 1;
        for(int k=0;k<8;++k)m.phase_layout_weight[ph][k]=(uint8_t)(host_weight4_get(g,ph,k)*4);
        if(ph>=4){int empty=(int)m.phase_layout_weight[ph][7]+2*(g.engine_pressure&15);if(empty>60)empty=60;m.phase_layout_weight[ph][7]=(uint8_t)empty;int thin=(g.engine_pressure>>4)&15;for(int k=2;k<=6;++k){int v=(int)m.phase_layout_weight[ph][k]-thin;if(v<0)v=0;m.phase_layout_weight[ph][k]=(uint8_t)v;}}
        m.phase_target_quadrants[ph]=g.phase_target_quadrants[ph];m.phase_hires_per_day[ph]=g.phase_hires_per_day[ph];
        m.phase_cow_target[ph]=g.phase_cow_target[ph];m.phase_sheep_target[ph]=g.phase_sheep_target[ph];m.phase_goose_target[ph]=g.phase_goose_target[ph];
        m.phase_cash_reserve[ph]=(uint16_t)g.phase_cash100[ph]*100;
    }
    m.hire_until_hour=g.hire_until_hour;m.max_hires_per_turn=g.max_hires_per_turn;m.max_sell_orders=g.max_sell_orders;m.seed_buy_cap=g.seed_buy_cap;m.animal_buy_cap=g.animal_buy_cap;
    m.feed_stock=g.feed_stock;m.feed_load=g.feed_load;m.fert_load=g.fert_load;m.drop_load=g.drop_load;m.drop_hour=g.drop_hour;m.rebuild_step_div12=g.rebuild_step_div12;
    m.care_enabled=g.care_enabled;m.collect_fert_enabled=g.collect_fert_enabled;m.dist_penalty=g.dist_penalty;m.planned_horizon=g.planned_horizon;m.planned_seed_horizon=g.planned_seed_horizon;
    m.planned_animal_horizon=g.planned_animal_horizon;m.planned_pickup_horizon=g.planned_pickup_horizon;
    for(int i=0;i<N_CROPS;++i)m.planned_seed_buffer[i]=g.planned_seed_buffer[i];
    m.planned_animal_buffer=g.planned_animal_buffer;m.planned_feed_buffer=g.planned_feed_buffer;m.planned_fert_buffer=g.planned_fert_buffer;m.copy_route_infrastructure=g.copy_route_infrastructure;
    for(int i=0;i<N_PRODUCTS;++i){m.sell_floor_pct[i]=g.sell_floor_pct[i];m.sell_reserve[i]=g.sell_reserve[i];}
    m.sell_batch=g.sell_batch;for(int i=0;i<10;++i)m.task_priority[i]=g.task_priority[i];
    return m;
}

static GPLeaf host_leaf(uint8_t mode,uint8_t ur,uint8_t mr,int n_routes,uint8_t engine_profile){
    GPLeaf l{};l.kind=mode_to_leaf_kind(mode);l.unit_route=(uint8_t)(n_routes?ur%n_routes:0);l.market_route=(uint8_t)(n_routes?mr%n_routes:0);
    l.pad=(l.kind==GPL_MANAGER_ENGINE_MARKET||l.kind==GPL_ROUTE_UNITS_ENGINE_MARKET||l.kind==GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET)?engine_profile:0;return l;
}

static GPProgram distill_program(const SearchGenome& g,int n_routes){
    GPProgram p{};p.manager=decode_manager_host(g);p.pad[0]=g.engine_batch_bias;p.pad[1]=g.engine_pressure;p.pad[2]=0;int nr=0;
    for(int r=0;r<g.n_rules && r<SEARCH_MAX_RULES && nr<GP_MAX_RULES-1;++r){
        const CompactRule& z=g.rules[r];GPRule q{};q.n_conds=1;q.conds[0].feature=z.feature;q.conds[0].cmp=z.cmp;q.conds[0].item=z.item;
        q.conds[0].value=(int16_t)((uint16_t)z.value_lo|((uint16_t)z.value_hi<<8));q.leaf=host_leaf(z.mode,z.unit_route,z.market_route,n_routes,g.engine_profile);p.rules[nr++]=q;
    }
    for(int ph=0;ph<GP_PHASES-1 && nr<GP_MAX_RULES-1;++ph){
        GPRule q{};q.n_conds=1;q.conds[0].feature=GPF_STEP;q.conds[0].cmp=GPC_LE;q.conds[0].value=(int)g.phase_step12[ph]*12-1;
        q.leaf=host_leaf(g.phase_mode[ph],g.phase_unit_route[ph],g.phase_market_route[ph],n_routes,g.engine_profile);p.rules[nr++]=q;
    }
    GPRule fallback{};fallback.n_conds=0;fallback.leaf=host_leaf(g.phase_mode[GP_PHASES-1],g.phase_unit_route[GP_PHASES-1],g.phase_market_route[GP_PHASES-1],n_routes,g.engine_profile);p.rules[nr++]=fallback;
    p.n_rules=(uint8_t)nr;return p;
}

static bool host_better(const DeviceFitness& a,const DeviceFitness& b){
    if(a.losing_families!=b.losing_families)return a.losing_families<b.losing_families;
    if(a.losses!=b.losses)return a.losses<b.losses;
    if(a.regressions!=b.regressions)return a.regressions<b.regressions;
    if(a.worst_family_milli!=b.worst_family_milli)return a.worst_family_milli>b.worst_family_milli;
    if(a.floor!=b.floor)return a.floor>b.floor;
    if(a.marginal_delta!=b.marginal_delta)return a.marginal_delta>b.marginal_delta;
    if(a.wins!=b.wins)return a.wins>b.wins;
    if(a.sum!=b.sum)return a.sum>b.sum;
    return a.ties<b.ties;
}

static std::vector<DevEntry> load_dev_entries(const std::filesystem::path& manifest,const std::filesystem::path& baseline,int& family_count){
    std::ifstream in(manifest);if(!in)throw std::runtime_error("cannot open dev manifest: "+manifest.string());
    std::string line;if(!std::getline(in,line))throw std::runtime_error("empty dev manifest");auto hdr=split_tab(line);std::map<std::string,int> col;for(int i=0;i<(int)hdr.size();++i)col[hdr[i]]=i;
    if(!col.count("source_seed"))throw std::runtime_error("dev manifest lacks replay-faithful source_seed column");
    std::map<std::string,int> fam;std::vector<DevEntry> out;
    while(std::getline(in,line)){auto v=split_tab(line);if(v.size()<hdr.size())continue;DevEntry e;e.team=v[col["team"]];e.bundle_path=v[col["bundle_path"]];e.filename=std::filesystem::path(e.bundle_path).filename().string();e.source_seed=std::stoull(v[col["source_seed"]]);auto it=fam.find(e.team);if(it==fam.end()){int id=(int)fam.size();fam[e.team]=id;e.family=id;}else e.family=it->second;out.push_back(e);}
    family_count=(int)fam.size();if(family_count>SEARCH_MAX_FAMILIES)throw std::runtime_error("too many development families for resident search");
    std::map<std::string,int> byname;for(int i=0;i<(int)out.size();++i)byname[out[i].filename]=i;
    std::ifstream b(baseline);if(!b)throw std::runtime_error("cannot open baseline cases: "+baseline.string());if(!std::getline(b,line))throw std::runtime_error("empty baseline cases");auto bh=split_tab(line);col.clear();for(int i=0;i<(int)bh.size();++i)col[bh[i]]=i;
    if(!col.count("seed"))throw std::runtime_error("baseline cases lack recorded seed column");std::set<std::pair<int,int>> baseline_seen;
    while(std::getline(b,line)){auto v=split_tab(line);if(v.size()<bh.size())continue;std::string fn=std::filesystem::path(v[col["path"]]).filename().string();auto it=byname.find(fn);if(it==byname.end())throw std::runtime_error("baseline references unknown dev tape: "+fn);int di=it->second,seat=std::stoi(v[col["seat"]]);if(seat<0||seat>1)throw std::runtime_error("invalid baseline seat");if(std::stoull(v[col["seed"]])!=out[di].source_seed)throw std::runtime_error("baseline/dev recorded seed mismatch for "+fn);if(!baseline_seen.insert({di,seat}).second)throw std::runtime_error("duplicate baseline tape/seat case");out[di].baseline_win[seat]=(uint8_t)(v[col["class"]]=="W");}
    if(baseline_seen.size()!=out.size()*2)throw std::runtime_error("baseline must cover every development tape in both seats");
    return out;
}

static TierSpec load_tier(const std::filesystem::path& list,const std::vector<DevEntry>& dev,const std::string& name,int max_step){
    std::map<std::string,int> byname;for(int i=0;i<(int)dev.size();++i)byname[dev[i].filename]=i;TierSpec t;t.name=name;t.max_step=max_step;
    std::ifstream in(list);if(!in)throw std::runtime_error("cannot open tier list: "+list.string());std::string line;
    while(std::getline(in,line)){const auto a=line.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;const auto b=line.find_last_not_of(" \t\r\n");std::string entry=line.substr(a,b-a+1);std::string fn=std::filesystem::path(entry).filename().string();auto it=byname.find(fn);if(it==byname.end())throw std::runtime_error("tier tape not in dev manifest: "+fn);const int di=it->second;for(int seat=0;seat<2;++seat)t.cases.push_back({di,dev[di].family,seat,dev[di].source_seed,dev[di].baseline_win[seat]});}
    return t;
}

__device__ __forceinline__ uint64_t rng_next(uint64_t& s){uint64_t x=s? s:0x9e3779b97f4a7c15ull;x^=x>>12;x^=x<<25;x^=x>>27;s=x;return x*2685821657736338717ull;}
__device__ __forceinline__ int rng_int(uint64_t& s,int lo,int hi){if(hi<=lo)return lo;return lo+(int)(rng_next(s)%(uint64_t)(hi-lo+1));}

__device__ __forceinline__ uint8_t d_weight4_get(const SearchGenome& g,int ph,int k){const int j=ph*8+k,b=j>>1;return (j&1)?(g.phase_layout_weight4[b]>>4):(g.phase_layout_weight4[b]&15);}
__device__ __forceinline__ void d_weight4_set(SearchGenome& g,int ph,int k,uint8_t v){const int j=ph*8+k,b=j>>1;v&=15;if(j&1)g.phase_layout_weight4[b]=(uint8_t)((g.phase_layout_weight4[b]&15)|(v<<4));else g.phase_layout_weight4[b]=(uint8_t)((g.phase_layout_weight4[b]&240)|v);}

__device__ __forceinline__ void normalize_genome(SearchGenome& g,int n_routes){
    int prev=0;for(int i=0;i<GP_PHASES-1;++i){int x=g.phase_step12[i];if(x<prev+1)x=prev+1;const int maxv=60-(GP_PHASES-2-i);if(x>maxv)x=maxv;g.phase_step12[i]=(uint8_t)x;prev=x;}
    int land=1,cow=0,sheep=0,goose=0;
    for(int ph=0;ph<GP_PHASES;++ph){
        int q=g.phase_target_quadrants[ph];if(q<land)q=land;if(q<1)q=1;if(q>4)q=4;g.phase_target_quadrants[ph]=(uint8_t)q;land=q;
        if(g.phase_cow_target[ph]<cow)g.phase_cow_target[ph]=(uint8_t)cow;else cow=g.phase_cow_target[ph];
        if(g.phase_sheep_target[ph]<sheep)g.phase_sheep_target[ph]=(uint8_t)sheep;else sheep=g.phase_sheep_target[ph];
        if(g.phase_goose_target[ph]<goose)g.phase_goose_target[ph]=(uint8_t)goose;else goose=g.phase_goose_target[ph];
        if(g.phase_hires_per_day[ph]>15)g.phase_hires_per_day[ph]=15;
        int sum=0;for(int k=0;k<8;++k)sum+=d_weight4_get(g,ph,k);if(sum==0)d_weight4_set(g,ph,7,1);
        g.phase_mode[ph]%=SEARCH_MODES;if(n_routes>0){g.phase_unit_route[ph]%=n_routes;g.phase_market_route[ph]%=n_routes;}else{g.phase_unit_route[ph]=g.phase_market_route[ph]=0;}
        if(g.phase_layout_seed[ph]==0)g.phase_layout_seed[ph]=1;
    }
    g.hire_until_hour%=6;if(g.max_hires_per_turn<1)g.max_hires_per_turn=1;if(g.max_hires_per_turn>12)g.max_hires_per_turn=12;
    if(g.max_sell_orders<1)g.max_sell_orders=1;if(g.max_sell_orders>10)g.max_sell_orders=10;
    if(g.drop_hour>23)g.drop_hour=23;if(g.dist_penalty<1)g.dist_penalty=1;if(g.n_rules>SEARCH_MAX_RULES)g.n_rules=SEARCH_MAX_RULES;
    for(int r=0;r<g.n_rules;++r){CompactRule& z=g.rules[r];z.feature%=29;z.cmp%=4;z.mode%=SEARCH_MODES;if(n_routes>0){z.unit_route%=n_routes;z.market_route%=n_routes;}else z.unit_route=z.market_route=0;}
}

__device__ __forceinline__ GPManager decode_manager_device(const SearchGenome& g){
    GPManager m{};
    for(int i=0;i<GP_PHASES-1;++i)m.phase_step[i]=(uint16_t)g.phase_step12[i]*12;
    for(int ph=0;ph<GP_PHASES;++ph){m.phase_layout_seed[ph]=g.phase_layout_seed[ph];for(int k=0;k<8;++k)m.phase_layout_weight[ph][k]=(uint8_t)(d_weight4_get(g,ph,k)*4);if(ph>=4){int empty=(int)m.phase_layout_weight[ph][7]+2*(g.engine_pressure&15);if(empty>60)empty=60;m.phase_layout_weight[ph][7]=(uint8_t)empty;int thin=(g.engine_pressure>>4)&15;for(int k=2;k<=6;++k){int v=(int)m.phase_layout_weight[ph][k]-thin;if(v<0)v=0;m.phase_layout_weight[ph][k]=(uint8_t)v;}}m.phase_target_quadrants[ph]=g.phase_target_quadrants[ph];m.phase_hires_per_day[ph]=g.phase_hires_per_day[ph];m.phase_cow_target[ph]=g.phase_cow_target[ph];m.phase_sheep_target[ph]=g.phase_sheep_target[ph];m.phase_goose_target[ph]=g.phase_goose_target[ph];m.phase_cash_reserve[ph]=(uint16_t)g.phase_cash100[ph]*100;}
    m.hire_until_hour=g.hire_until_hour;m.max_hires_per_turn=g.max_hires_per_turn;m.max_sell_orders=g.max_sell_orders;m.seed_buy_cap=g.seed_buy_cap;m.animal_buy_cap=g.animal_buy_cap;m.feed_stock=g.feed_stock;m.feed_load=g.feed_load;m.fert_load=g.fert_load;m.drop_load=g.drop_load;m.drop_hour=g.drop_hour;m.rebuild_step_div12=g.rebuild_step_div12;m.care_enabled=g.care_enabled;m.collect_fert_enabled=g.collect_fert_enabled;m.dist_penalty=g.dist_penalty;m.planned_horizon=g.planned_horizon;m.planned_seed_horizon=g.planned_seed_horizon;m.planned_animal_horizon=g.planned_animal_horizon;m.planned_pickup_horizon=g.planned_pickup_horizon;
    for(int i=0;i<N_CROPS;++i)m.planned_seed_buffer[i]=g.planned_seed_buffer[i];m.planned_animal_buffer=g.planned_animal_buffer;m.planned_feed_buffer=g.planned_feed_buffer;m.planned_fert_buffer=g.planned_fert_buffer;m.copy_route_infrastructure=g.copy_route_infrastructure;for(int i=0;i<N_PRODUCTS;++i){m.sell_floor_pct[i]=g.sell_floor_pct[i];m.sell_reserve[i]=g.sell_reserve[i];}m.sell_batch=g.sell_batch;for(int i=0;i<10;++i)m.task_priority[i]=g.task_priority[i];return m;
}

__device__ __forceinline__ GPLeaf d_leaf(uint8_t mode,uint8_t ur,uint8_t mr,int n_routes,uint8_t engine_profile){GPLeaf l{};switch(mode%SEARCH_MODES){case SM_ROUTE_UNITS_MANAGER_MARKET:l.kind=GPL_ROUTE_UNITS_MANAGER_MARKET;break;case SM_ROUTE_UNITS_PLANNED_MARKET:l.kind=GPL_ROUTE_UNITS_PLANNED_MARKET;break;case SM_MANAGER_UNITS_ROUTE_MARKET:l.kind=GPL_MANAGER_UNITS_ROUTE_MARKET;break;case SM_ROUTE:l.kind=GPL_ROUTE;break;case SM_MANAGER_ENGINE_MARKET:l.kind=GPL_MANAGER_ENGINE_MARKET;break;case SM_ROUTE_UNITS_ENGINE_MARKET:l.kind=GPL_ROUTE_UNITS_ENGINE_MARKET;break;case SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET:l.kind=GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET;break;default:l.kind=GPL_MANAGER;break;}l.unit_route=n_routes?ur%n_routes:0;l.market_route=n_routes?mr%n_routes:0;l.pad=(l.kind==GPL_MANAGER_ENGINE_MARKET||l.kind==GPL_ROUTE_UNITS_ENGINE_MARKET||l.kind==GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET)?engine_profile:0;return l;}

__device__ __forceinline__ bool compact_rule_matches(const CompactRule& z,const ObservationView& obs){GPCond c{};c.feature=z.feature;c.cmp=z.cmp;c.item=z.item;c.value=(int16_t)((uint16_t)z.value_lo|((uint16_t)z.value_hi<<8));return gp_eval_cond(c,obs);}

__device__ __forceinline__ Action search_act(const SearchGenome& g,const GPManager& m,const ObservationView& obs,const Action* routes,const GPRouteDesc* descs,int n_routes){
    int ph=gp_phase(m,obs.step());uint8_t mode=g.phase_mode[ph],ur=g.phase_unit_route[ph],mr=g.phase_market_route[ph];
    for(int r=0;r<g.n_rules && r<SEARCH_MAX_RULES;++r)if(compact_rule_matches(g.rules[r],obs)){mode=g.rules[r].mode;ur=g.rules[r].unit_route;mr=g.rules[r].market_route;break;}
    GPLeaf leaf=d_leaf(mode,ur,mr,n_routes,g.engine_profile);ur=leaf.unit_route;mr=leaf.market_route;const Action& ua=gp_route_at(routes,descs[ur],obs.step());const Action& ma=gp_route_at(routes,descs[mr],obs.step());Action out{};out.n_units=obs.farm(obs.player()).n_units();
    switch(leaf.kind){case GPL_ROUTE:out=gp_copy_action(ua,out.n_units);break;case GPL_ROUTE_UNITS_MANAGER_MARKET:out=gp_copy_action(ua,out.n_units);gp_manager_market(m,obs,out);break;case GPL_ROUTE_UNITS_PLANNED_MARKET:out=gp_copy_planned_action(ua,out.n_units,routes,descs[ur],obs.step());gp_planned_market(m,obs,routes,descs[ur],out);break;case GPL_MANAGER_UNITS_ROUTE_MARKET:gp_manager_units(m,obs,out);out.n_orders=ma.n_orders;for(int i=0;i<ma.n_orders && i<MAX_ORDERS;++i)out.orders[i]=ma.orders[i];break;case GPL_MANAGER_ENGINE_MARKET:gp_manager_units(m,obs,out);gp_manager_market(m,obs,out);gp_engine_market_overlay(m,obs,out,g.engine_profile,g.engine_batch_bias,g.engine_pressure);break;case GPL_ROUTE_UNITS_ENGINE_MARKET:out=gp_copy_action(ua,out.n_units);gp_manager_market(m,obs,out);gp_engine_market_overlay(m,obs,out,g.engine_profile,g.engine_batch_bias,g.engine_pressure);break;case GPL_ROUTE_UNITS_PLANNED_ENGINE_MARKET:out=gp_copy_planned_action(ua,out.n_units,routes,descs[ur],obs.step());gp_planned_market(m,obs,routes,descs[ur],out);gp_engine_market_overlay(m,obs,out,g.engine_profile,g.engine_batch_bias,g.engine_pressure);break;default:gp_manager_units(m,obs,out);gp_manager_market(m,obs,out);break;}return out;
}

__global__ void eval_case_kernel(Config base,DeviceTables tables,const SearchGenome* genomes,const GPManager* managers,int population,const Action* routes,const GPRouteDesc* descs,int n_candidate_routes,int opponent_desc,int candidate_seat,uint64_t seed,int max_step,Sim* sims,double* rewards){
    int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population)return;Config cfg=base;cfg.seed=seed;Sim& sim=sims[pi];sim.init(cfg,&tables);const SearchGenome& g=genomes[pi];const GPManager& m=managers[pi];const GPRouteDesc od=descs[opponent_desc];while(!sim.st.done && sim.st.step<max_step){int step=sim.st.step,opp_seat=1-candidate_seat;ObservationView obs=make_observation(sim,candidate_seat);Action cand=search_act(g,m,obs,routes,descs,n_candidate_routes);Action opp=gp_copy_action(gp_route_at(routes,od,step),sim.st.farms[opp_seat].n_units);if(candidate_seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);}rewards[(size_t)pi*2]=sim.st.farms[0].money;rewards[(size_t)pi*2+1]=sim.st.farms[1].money;
}

__global__ void eval_distilled_program_case_kernel(Config base,DeviceTables tables,const GPProgram* program,const Action* routes,const GPRouteDesc* descs,int n_candidate_routes,int opponent_desc,int candidate_seat,uint64_t seed,int max_step,Sim* sim_out,double* rewards){
    if(blockIdx.x||threadIdx.x)return;Config cfg=base;cfg.seed=seed;Sim& sim=*sim_out;sim.init(cfg,&tables);const GPRouteDesc od=descs[opponent_desc];while(!sim.st.done&&sim.st.step<max_step){int step=sim.st.step,opp_seat=1-candidate_seat;ObservationView obs=make_observation(sim,candidate_seat);Action cand=gp_act(*program,obs,routes,descs,n_candidate_routes);Action opp=gp_copy_action(gp_route_at(routes,od,step),sim.st.farms[opp_seat].n_units);if(candidate_seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);}rewards[0]=sim.st.farms[0].money;rewards[1]=sim.st.farms[1].money;
}

__global__ void reset_metrics_kernel(DeviceFitness* fit,int population){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=population)return;DeviceFitness f{};f.floor=DBL_MAX;f.losing_families=0;fit[i]=f;}

__global__ void accumulate_case_kernel(const double* rewards,DeviceFitness* fit,uint8_t* family_losses,uint8_t* family_ties,uint8_t* family_cases,int population,int family_count,int family,int seat,int baseline_win){int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population)return;double r0=rewards[(size_t)pi*2],r1=rewards[(size_t)pi*2+1],m=seat==0?r0-r1:r1-r0;DeviceFitness& f=fit[pi];++f.cases;if(m>0)++f.wins;else if(m<0)++f.losses;else ++f.ties;if(baseline_win && m<=0)++f.regressions;f.sum+=m;if(m<f.floor)f.floor=m;size_t j=(size_t)pi*family_count+family;uint8_t c=family_cases[j];if(c<255)family_cases[j]=c+1;if(m<0){uint8_t x=family_losses[j];if(x<255)family_losses[j]=x+1;}else if(m==0){uint8_t x=family_ties[j];if(x<255)family_ties[j]=x+1;}}

__device__ __forceinline__ uint64_t mix64(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;return x;}

__device__ __forceinline__ uint8_t niche_of(const SearchGenome& g){uint32_t x=2166136261u;auto add=[&](uint8_t v){x=(x^v)*16777619u;};add(g.phase_target_quadrants[GP_PHASES-1]);add(g.phase_cow_target[GP_PHASES-1]/2);add(g.phase_sheep_target[GP_PHASES-1]/2);add(g.phase_cash100[GP_PHASES-1]/5);add(g.phase_mode[6]);add(g.phase_unit_route[6]);add(g.sell_floor_pct[5]/10);add(g.hire_until_hour);add(g.engine_profile);add(g.engine_batch_bias>>4);add(g.engine_pressure>>4);add(g.engine_pressure);return (uint8_t)(x&255);}

__device__ __forceinline__ uint64_t signature_of(const SearchGenome& g){uint64_t x=0xcbf29ce484222325ull;for(int i=0;i<GP_PHASES;++i){x^=(uint64_t)g.phase_target_quadrants[i]+((uint64_t)g.phase_cow_target[i]<<8)+((uint64_t)g.phase_sheep_target[i]<<16)+((uint64_t)g.phase_mode[i]<<24)+((uint64_t)g.phase_unit_route[i]<<32);x*=0x100000001b3ull;}for(int i=0;i<N_PRODUCTS;++i){x^=g.sell_floor_pct[i];x*=0x100000001b3ull;}x^=(uint64_t)g.n_rules|((uint64_t)g.engine_profile<<8)|((uint64_t)g.engine_batch_bias<<16)|((uint64_t)g.engine_pressure<<24);return mix64(x);}

__device__ __forceinline__ uint64_t rank_key_of(const DeviceFitness& f){
    uint64_t a=(uint64_t)(31-(f.losing_families>31?31:f.losing_families));
    uint64_t b=(uint64_t)(1023-(f.losses>1023?1023:f.losses));
    uint64_t c=(uint64_t)(1023-(f.regressions>1023?1023:f.regressions));
    uint64_t d=(uint64_t)(f.worst_family_milli<0?0:(f.worst_family_milli>1023?1023:f.worst_family_milli));
    int fq=(int)(f.floor/8.0)+8192;if(fq<0)fq=0;if(fq>16383)fq=16383;
    double mean=f.cases?f.sum/f.cases:0.0;int mq=(int)(mean/16.0)+4096;if(mq<0)mq=0;if(mq>8191)mq=8191;
    const uint64_t md=f.marginal_delta>1000.0?3ull:(f.marginal_delta>100.0?2ull:(f.marginal_delta>0.0?1ull:0ull));
    return (a<<59)|(b<<49)|(c<<39)|(d<<29)|((uint64_t)fq<<15)|((uint64_t)mq<<2)|md;
}

__global__ void finalize_metrics_kernel(const SearchGenome* genomes,DeviceFitness* fit,const uint8_t* family_losses,const uint8_t* family_ties,const uint8_t* family_cases,const double* parent_mean,int population,int family_count,uint32_t* niche_counts){int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population)return;DeviceFitness& f=fit[pi];int losing=0,eval=0,worst=1000;for(int fam=0;fam<family_count;++fam){size_t j=(size_t)pi*family_count+fam;int cases=family_cases[j];if(!cases)continue;++eval;int losses=family_losses[j],ties=family_ties[j];if(losses>0)++losing;int wins=cases-losses-ties;int milli=(wins*1000+ties*500)/cases;if(milli<worst)worst=milli;}f.losing_families=losing;f.evaluated_families=eval;f.worst_family_milli=eval?worst:0;double mean=f.cases?f.sum/f.cases:0.0;double ref=parent_mean?parent_mean[pi]:DBL_MAX;f.marginal_delta=ref<DBL_MAX/4?mean-ref:0.0;f.niche=niche_of(genomes[pi]);f.signature=signature_of(genomes[pi]);f.rank_key=rank_key_of(f);atomicAdd(&niche_counts[f.niche],1u);}

__device__ __forceinline__ bool d_better(const DeviceFitness& a,const DeviceFitness& b){if(a.losing_families!=b.losing_families)return a.losing_families<b.losing_families;if(a.losses!=b.losses)return a.losses<b.losses;if(a.regressions!=b.regressions)return a.regressions<b.regressions;if(a.worst_family_milli!=b.worst_family_milli)return a.worst_family_milli>b.worst_family_milli;if(a.floor!=b.floor)return a.floor>b.floor;if(a.marginal_delta!=b.marginal_delta)return a.marginal_delta>b.marginal_delta;if(a.wins!=b.wins)return a.wins>b.wins;if(a.sum!=b.sum)return a.sum>b.sum;return a.ties<b.ties;}

__global__ void gather_promotion_kernel(const SearchGenome* archive,const uint8_t* valid,SearchGenome* promo,GPManager* promo_mgr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=SEARCH_NICHES)return;if(valid[i]){promo[i]=archive[i];promo_mgr[i]=decode_manager_device(archive[i]);}else{promo[i]=SearchGenome{};promo_mgr[i]=GPManager{};}}

__global__ void eval_case_masked_kernel(Config base,DeviceTables tables,const SearchGenome* genomes,const GPManager* managers,const uint8_t* valid,int population,const Action* routes,const GPRouteDesc* descs,int n_candidate_routes,int opponent_desc,int candidate_seat,uint64_t seed,int max_step,Sim* sims,double* rewards,unsigned long long* game_counter){int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population)return;if(!valid[pi]){rewards[(size_t)pi*2]=0;rewards[(size_t)pi*2+1]=0;return;}if(game_counter)atomicAdd(game_counter,1ull);Config cfg=base;cfg.seed=seed;Sim& sim=sims[pi];sim.init(cfg,&tables);const SearchGenome& g=genomes[pi];const GPManager& m=managers[pi];const GPRouteDesc od=descs[opponent_desc];while(!sim.st.done&&sim.st.step<max_step){int step=sim.st.step,opp_seat=1-candidate_seat;ObservationView obs=make_observation(sim,candidate_seat);Action cand=search_act(g,m,obs,routes,descs,n_candidate_routes);Action opp=gp_copy_action(gp_route_at(routes,od,step),sim.st.farms[opp_seat].n_units);if(candidate_seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);}rewards[(size_t)pi*2]=sim.st.farms[0].money;rewards[(size_t)pi*2+1]=sim.st.farms[1].money;}

__global__ void accumulate_case_masked_kernel(const double* rewards,const uint8_t* valid,DeviceFitness* fit,uint8_t* family_losses,uint8_t* family_ties,uint8_t* family_cases,int population,int family_count,int family,int seat,int baseline_win){int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population||!valid[pi])return;double r0=rewards[(size_t)pi*2],r1=rewards[(size_t)pi*2+1],m=seat==0?r0-r1:r1-r0;DeviceFitness& f=fit[pi];++f.cases;if(m>0)++f.wins;else if(m<0)++f.losses;else ++f.ties;if(baseline_win&&m<=0)++f.regressions;f.sum+=m;if(m<f.floor)f.floor=m;size_t j=(size_t)pi*family_count+family;++family_cases[j];if(m<0)++family_losses[j];else if(m==0)++family_ties[j];}

__global__ void finalize_promotion_kernel(const SearchGenome* genomes,const uint8_t* valid,DeviceFitness* fit,const uint8_t* family_losses,const uint8_t* family_ties,const uint8_t* family_cases,int population,int family_count){int pi=blockIdx.x*blockDim.x+threadIdx.x;if(pi>=population)return;if(!valid[pi]){fit[pi]=DeviceFitness{};fit[pi].losing_families=SEARCH_MAX_FAMILIES;fit[pi].losses=1<<20;fit[pi].regressions=1<<20;fit[pi].floor=-DBL_MAX;return;}DeviceFitness& f=fit[pi];int losing=0,eval=0,worst=1000;for(int fam=0;fam<family_count;++fam){size_t j=(size_t)pi*family_count+fam;int cases=family_cases[j];if(!cases)continue;++eval;int losses=family_losses[j],ties=family_ties[j];if(losses>0)++losing;int wins=cases-losses-ties;int milli=(wins*1000+ties*500)/cases;if(milli<worst)worst=milli;}f.losing_families=losing;f.evaluated_families=eval;f.worst_family_milli=eval?worst:0;f.niche=(uint8_t)pi;f.signature=signature_of(genomes[pi]);f.rank_key=rank_key_of(f);}

__global__ void commit_certified_kernel(const SearchGenome* promo,const DeviceFitness* fit,const uint8_t* valid,int level,SearchGenome* certified,DeviceFitness* certified_fit,uint8_t* certified_level){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=SEARCH_NICHES||!valid[i])return;uint8_t old=certified_level[i];if(level>old||(level==old&&d_better(fit[i],certified_fit[i]))){certified[i]=promo[i];certified_fit[i]=fit[i];certified_level[i]=(uint8_t)level;}}

__global__ void niche_vote_kernel(const DeviceFitness* fit,int population,unsigned long long* votes){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=population)return;uint64_t key40=fit[i].rank_key>>24;uint64_t packed=(key40<<24)|(uint64_t)(0xffffffu-(uint32_t)i);atomicMax(&votes[fit[i].niche],(unsigned long long)packed);}

__global__ void niche_commit_kernel(const SearchGenome* pop,const DeviceFitness* fit,int population,const unsigned long long* votes,SearchGenome* archive_genomes,DeviceFitness* archive_fit,uint8_t* archive_valid){int niche=blockIdx.x*blockDim.x+threadIdx.x;if(niche>=SEARCH_NICHES)return;uint64_t v=votes[niche];if(!v)return;int idx=(int)(0xffffffu-(uint32_t)(v&0xffffffu));if(idx<0||idx>=population)return;if(!archive_valid[niche]||d_better(fit[idx],archive_fit[niche])){archive_genomes[niche]=pop[idx];archive_fit[niche]=fit[idx];archive_valid[niche]=1;}}

__global__ void island_best_kernel(const SearchGenome* pop,const DeviceFitness* fit,int population,int sub_islands,IslandState* states){int isl=blockIdx.x;if(isl>=sub_islands)return;int begin=(int)((long long)population*isl/sub_islands),end=(int)((long long)population*(isl+1)/sub_islands);int best=-1;for(int i=begin+threadIdx.x;i<end;i+=blockDim.x)if(best<0||d_better(fit[i],fit[best]))best=i;__shared__ int sh[256];sh[threadIdx.x]=best;__syncthreads();for(int off=blockDim.x/2;off>0;off>>=1){if(threadIdx.x<off){int a=sh[threadIdx.x],b=sh[threadIdx.x+off];if(a<0||(b>=0&&d_better(fit[b],fit[a])))sh[threadIdx.x]=b;}__syncthreads();}if(threadIdx.x==0){int bi=sh[0];IslandState& s=states[isl];s.current_best_index=bi;++s.generations;if(!s.initialized||d_better(fit[bi],s.best_fit)){s.best_fit=fit[bi];s.best_genome=pop[bi];s.best_index=bi;s.stagnation=0;s.initialized=1;++s.improvements;}else ++s.stagnation;}}

__global__ void island_control_kernel(IslandState* states,int sub_islands,int stagnation_limit){int isl=blockIdx.x*blockDim.x+threadIdx.x;if(isl>=sub_islands)return;IslandState& s=states[isl];s.reset_requested=(uint8_t)(s.stagnation>=stagnation_limit);if(s.stagnation>=stagnation_limit/2){for(int j=0;j<isl;++j){const IslandState& q=states[j];if(q.initialized&&q.best_fit.signature==s.best_fit.signature){s.reset_requested=1;break;}}}if(s.reset_requested){++s.resets;s.stagnation=0;}}

__device__ __forceinline__ void mutate_byte(uint8_t& v,uint64_t& r,int amp){int x=(int)v+rng_int(r,-amp,amp);if(x<0)x=0;if(x>255)x=255;v=(uint8_t)x;}

__device__ __forceinline__ uint8_t engine_mode_from(uint8_t mode){switch(mode%SEARCH_MODES){case SM_ROUTE_UNITS_MANAGER_MARKET:return SM_ROUTE_UNITS_ENGINE_MARKET;case SM_ROUTE_UNITS_PLANNED_MARKET:return SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET;case SM_MANAGER:return SM_MANAGER_ENGINE_MARKET;case SM_MANAGER_ENGINE_MARKET:case SM_ROUTE_UNITS_ENGINE_MARKET:case SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET:return mode;default:return SM_MANAGER_ENGINE_MARKET;}}

__device__ void mutate_engine(SearchGenome& g,uint64_t& r,int n_routes,int strength){
    const int k=rng_int(r,0,6),ph=rng_int(r,0,GP_PHASES-1);
    if(k==0){int bit=rng_int(r,0,7);g.engine_profile^=(uint8_t)(1u<<bit);if(rng_int(r,0,3)==0)g.engine_profile=(uint8_t)rng_int(r,0,255);}
    else if(k==1){mutate_byte(g.engine_batch_bias,r,16*strength);mutate_byte(g.engine_pressure,r,16*strength);}
    else if(k==2){mutate_byte(g.engine_pressure,r,8*strength);int x=(int)d_weight4_get(g,ph,7)+rng_int(r,-strength,3*strength);d_weight4_set(g,ph,7,(uint8_t)(x<0?0:(x>15?15:x)));if(rng_int(r,0,2)==0)g.phase_layout_seed[ph]=(uint8_t)rng_int(r,1,255);}
    else if(k==3){g.phase_mode[ph]=engine_mode_from(g.phase_mode[ph]);if(n_routes>0){if(rng_int(r,0,1))g.phase_unit_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);if(rng_int(r,0,1))g.phase_market_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);}}
    else if(k==4){int item=rng_int(r,0,N_PRODUCTS-1);mutate_byte(g.sell_floor_pct[item],r,12*strength);mutate_byte(g.sell_reserve[item],r,4*strength);mutate_byte(g.sell_batch,r,8*strength);}
    else if(k==5){if(g.n_rules<SEARCH_MAX_RULES&&rng_int(r,0,2)==0)++g.n_rules;if(g.n_rules){CompactRule& z=g.rules[rng_int(r,0,g.n_rules-1)];const uint8_t feats[]={GPF_PRICE,GPF_MARKET_INV,GPF_SHOP_SEEN,GPF_SHED,GPF_OWN_CROP,GPF_OPP_CROP,GPF_OWN_ANIMAL,GPF_OPP_ANIMAL,GPF_OWN_MONEY,GPF_OPP_MONEY,GPF_OWN_WEEDS,GPF_OPP_WEEDS,GPF_STEP_BUCKET8};z.feature=feats[rng_int(r,0,(int)(sizeof(feats)/sizeof(feats[0]))-1)];z.cmp=(uint8_t)rng_int(r,0,3);if(z.feature==GPF_OWN_CROP||z.feature==GPF_OPP_CROP)z.item=(uint8_t)rng_int(r,0,N_CROPS-1);else if(z.feature==GPF_OWN_ANIMAL||z.feature==GPF_OPP_ANIMAL)z.item=(uint8_t)rng_int(r,0,N_ANIMALS-1);else if(z.feature==GPF_SHOP_SEEN)z.item=(uint8_t)rng_int(r,0,MAX_SHOP_INSTANCES-1);else z.item=(uint8_t)rng_int(r,0,N_PRODUCTS-1);int16_t v=(int16_t)rng_int(r,-256,4096);z.value_lo=(uint8_t)(v&255);z.value_hi=(uint8_t)((uint16_t)v>>8);z.mode=(uint8_t)rng_int(r,SM_MANAGER_ENGINE_MARKET,SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET);if(n_routes>0){z.unit_route=(uint8_t)rng_int(r,0,n_routes-1);z.market_route=(uint8_t)rng_int(r,0,n_routes-1);}}}
    else{mutate_byte(g.rebuild_step_div12,r,3*strength);for(int q=4;q<GP_PHASES;++q)if(rng_int(r,0,3)==0){int x=(int)d_weight4_get(g,q,7)+rng_int(r,0,2*strength);d_weight4_set(g,q,7,(uint8_t)(x>15?15:x));}}
    normalize_genome(g,n_routes);
}

__device__ void mutate_one(SearchGenome& g,uint64_t& r,int n_routes,int strength){int k=rng_int(r,0,12),ph=rng_int(r,0,GP_PHASES-1);if(k==0&&ph<GP_PHASES-1){int x=(int)g.phase_step12[ph]+rng_int(r,-strength,strength);g.phase_step12[ph]=(uint8_t)(x<1?1:(x>60?60:x));}else if(k==1){int cat=rng_int(r,0,7),x=(int)d_weight4_get(g,ph,cat)+rng_int(r,-strength,strength);d_weight4_set(g,ph,cat,(uint8_t)(x<0?0:(x>15?15:x)));}else if(k==2){int z=rng_int(r,0,5);if(z==0)mutate_byte(g.phase_target_quadrants[ph],r,1);else if(z==1)mutate_byte(g.phase_hires_per_day[ph],r,strength);else if(z==2)mutate_byte(g.phase_cow_target[ph],r,strength);else if(z==3)mutate_byte(g.phase_sheep_target[ph],r,strength);else if(z==4)mutate_byte(g.phase_goose_target[ph],r,strength);else mutate_byte(g.phase_cash100[ph],r,2*strength);}else if(k==3){g.phase_mode[ph]=(uint8_t)rng_int(r,0,SEARCH_MODES-1);g.phase_unit_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);g.phase_market_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);}else if(k==4){int i=rng_int(r,0,N_PRODUCTS-1);mutate_byte(g.sell_floor_pct[i],r,8*strength);mutate_byte(g.sell_reserve[i],r,2*strength);if(rng_int(r,0,3)==0)mutate_byte(g.sell_batch,r,4*strength);}else if(k==5){int z=rng_int(r,0,8);uint8_t* p[]={&g.hire_until_hour,&g.max_hires_per_turn,&g.max_sell_orders,&g.seed_buy_cap,&g.animal_buy_cap,&g.feed_stock,&g.feed_load,&g.drop_load,&g.dist_penalty};mutate_byte(*p[z],r,strength+1);}else if(k==6){uint8_t* p[]={&g.planned_horizon,&g.planned_seed_horizon,&g.planned_animal_horizon,&g.planned_pickup_horizon,&g.planned_animal_buffer,&g.planned_feed_buffer,&g.planned_fert_buffer};mutate_byte(*p[rng_int(r,0,6)],r,2*strength);}else if(k==7){int i=rng_int(r,0,9);mutate_byte(g.task_priority[i],r,8*strength);}else if(k==8){g.phase_layout_seed[ph]=(uint8_t)rng_int(r,1,255);}else if(k==9){if(g.n_rules<SEARCH_MAX_RULES&&rng_int(r,0,1))++g.n_rules;else if(g.n_rules>0&&rng_int(r,0,2)==0)--g.n_rules;if(g.n_rules){CompactRule& z=g.rules[rng_int(r,0,g.n_rules-1)];z.feature=(uint8_t)rng_int(r,0,28);z.cmp=(uint8_t)rng_int(r,0,3);z.item=(uint8_t)rng_int(r,0,15);int16_t v=(int16_t)rng_int(r,-512,12000);z.value_lo=(uint8_t)(v&255);z.value_hi=(uint8_t)((uint16_t)v>>8);z.mode=(uint8_t)rng_int(r,0,SEARCH_MODES-1);z.unit_route=(uint8_t)rng_int(r,0,n_routes-1);z.market_route=(uint8_t)rng_int(r,0,n_routes-1);}}else if(k==10||k==11){mutate_engine(g,r,n_routes,strength);}else{g.care_enabled^=1;if(rng_int(r,0,1))g.collect_fert_enabled^=1;mutate_byte(g.rebuild_step_div12,r,2*strength);}normalize_genome(g,n_routes);}

__device__ void crossover_modules(SearchGenome& c,const SearchGenome& b,uint64_t& r){int mod=rng_int(r,0,5);if(mod==0){int a=rng_int(r,0,7),z=rng_int(r,a,10);for(int ph=a;ph<=z;++ph){c.phase_layout_seed[ph]=b.phase_layout_seed[ph];for(int k=0;k<8;++k)d_weight4_set(c,ph,k,d_weight4_get(b,ph,k));c.phase_target_quadrants[ph]=b.phase_target_quadrants[ph];c.phase_hires_per_day[ph]=b.phase_hires_per_day[ph];c.phase_cow_target[ph]=b.phase_cow_target[ph];c.phase_sheep_target[ph]=b.phase_sheep_target[ph];c.phase_goose_target[ph]=b.phase_goose_target[ph];c.phase_cash100[ph]=b.phase_cash100[ph];c.phase_mode[ph]=b.phase_mode[ph];c.phase_unit_route[ph]=b.phase_unit_route[ph];c.phase_market_route[ph]=b.phase_market_route[ph];if(ph>0)c.phase_step12[ph-1]=b.phase_step12[ph-1];}}else if(mod==1){c.hire_until_hour=b.hire_until_hour;c.max_hires_per_turn=b.max_hires_per_turn;c.seed_buy_cap=b.seed_buy_cap;c.animal_buy_cap=b.animal_buy_cap;c.feed_stock=b.feed_stock;c.feed_load=b.feed_load;c.fert_load=b.fert_load;c.drop_load=b.drop_load;c.drop_hour=b.drop_hour;c.dist_penalty=b.dist_penalty;for(int i=0;i<10;++i)c.task_priority[i]=b.task_priority[i];}else if(mod==2){c.max_sell_orders=b.max_sell_orders;c.sell_batch=b.sell_batch;for(int i=0;i<N_PRODUCTS;++i){c.sell_floor_pct[i]=b.sell_floor_pct[i];c.sell_reserve[i]=b.sell_reserve[i];}}else if(mod==3){c.planned_horizon=b.planned_horizon;c.planned_seed_horizon=b.planned_seed_horizon;c.planned_animal_horizon=b.planned_animal_horizon;c.planned_pickup_horizon=b.planned_pickup_horizon;for(int i=0;i<N_CROPS;++i)c.planned_seed_buffer[i]=b.planned_seed_buffer[i];c.planned_animal_buffer=b.planned_animal_buffer;c.planned_feed_buffer=b.planned_feed_buffer;c.planned_fert_buffer=b.planned_fert_buffer;c.copy_route_infrastructure=b.copy_route_infrastructure;}else if(mod==4){c.n_rules=b.n_rules;for(int i=0;i<SEARCH_MAX_RULES;++i)c.rules[i]=b.rules[i];}else{c.engine_profile=b.engine_profile;c.engine_batch_bias=b.engine_batch_bias;c.engine_pressure=b.engine_pressure;for(int ph=0;ph<GP_PHASES;++ph)if(b.phase_mode[ph]>=SM_MANAGER_ENGINE_MARKET){c.phase_mode[ph]=b.phase_mode[ph];c.phase_unit_route[ph]=b.phase_unit_route[ph];c.phase_market_route[ph]=b.phase_market_route[ph];}}}

__device__ int select_parent(const DeviceFitness* fit,const uint32_t* niche_counts,uint64_t& r,int begin,int end){int best=rng_int(r,begin,end-1);for(int k=0;k<3;++k){int q=rng_int(r,begin,end-1);if(d_better(fit[q],fit[best]))best=q;else if(!d_better(fit[best],fit[q])&&niche_counts[fit[q].niche]<niche_counts[fit[best].niche])best=q;}return best;}

__device__ __forceinline__ double device_fit_mean(const DeviceFitness& f){return f.cases?f.sum/f.cases:0.0;}

__global__ void init_population_kernel(SearchGenome* pop,GPManager* mgr,double* parent_mean,uint64_t* rng,int population,int sub_islands,int engine_sub_islands,const SearchGenome* priors,int n_priors,int n_routes,uint64_t seed,int universe){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=population)return;
    uint64_t r=mix64(seed^(uint64_t)(i+1)*0x9e3779b97f4a7c15ull);
    int isl=(int)((long long)i*sub_islands/population),begin=(int)((long long)population*isl/sub_islands),local=i-begin;bool engine=isl<engine_sub_islands;
    SearchGenome g=priors[n_priors?isl%n_priors:0];
    if(engine){g.engine_profile=(uint8_t)rng_int(r,0,255);g.engine_batch_bias=(uint8_t)rng_int(r,16,240);g.engine_pressure=(uint8_t)rng_int(r,0,255);for(int ph=2;ph<GP_PHASES;++ph)if(rng_int(r,0,2)!=0)g.phase_mode[ph]=engine_mode_from(g.phase_mode[ph]);}
    if(local!=0){int edits=universe==0?rng_int(r,2,12):rng_int(r,18,72);for(int e=0;e<edits;++e){if(engine&&rng_int(r,0,99)<65)mutate_engine(g,r,n_routes,universe==0?1:rng_int(r,1,4));else mutate_one(g,r,n_routes,universe==0?1:rng_int(r,1,4));}if(universe==1&&rng_int(r,0,2)==0){for(int ph=4;ph<GP_PHASES;++ph){g.phase_mode[ph]=engine&&rng_int(r,0,1)?(uint8_t)rng_int(r,SM_MANAGER_ENGINE_MARKET,SM_ROUTE_UNITS_PLANNED_ENGINE_MARKET):(uint8_t)rng_int(r,0,SEARCH_MODES-1);g.phase_unit_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);g.phase_market_route[ph]=(uint8_t)rng_int(r,0,n_routes-1);}}}
    else if(engine){for(int e=0;e<3;++e)mutate_engine(g,r,n_routes,1);}
    normalize_genome(g,n_routes);pop[i]=g;mgr[i]=decode_manager_device(g);parent_mean[i]=DBL_MAX;rng[i]=r;
}

__global__ void evolve_kernel(const SearchGenome* cur,SearchGenome* next,GPManager* next_mgr,double* next_parent_mean,const DeviceFitness* fit,const uint32_t* niche_counts,uint64_t* rng,int population,int sub_islands,int engine_sub_islands,int n_routes,int universe,int local_per_island,const IslandState* states,const SearchGenome* priors,int n_priors,const SearchGenome* archive_genomes,const DeviceFitness* archive_fit,const uint8_t* archive_valid,const SearchGenome* certified_genomes,const DeviceFitness* certified_fit,const uint8_t* certified_level){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=population)return;
    int isl=(int)((long long)i*sub_islands/population),begin=(int)((long long)population*isl/sub_islands),end=(int)((long long)population*(isl+1)/sub_islands),local=i-begin;bool engine=isl<engine_sub_islands;uint64_t r=rng[i];const IslandState& st=states[isl];SearchGenome child{};double ref=DBL_MAX;
    if(local==0&&st.initialized){child=st.best_genome;ref=device_fit_mean(st.best_fit);}
    else if(st.reset_requested){bool used=false;for(int t=0;t<8&&!used;++t){int n=rng_int(r,0,SEARCH_NICHES-1);if(certified_level[n]>=2){child=certified_genomes[n];used=true;}}for(int t=0;t<8&&!used;++t){int n=rng_int(r,0,SEARCH_NICHES-1);if(archive_valid[n]){child=archive_genomes[n];ref=device_fit_mean(archive_fit[n]);used=true;}}if(!used)child=priors[rng_int(r,0,n_priors-1)];int edits=universe==0?rng_int(r,12,36):rng_int(r,36,96);for(int e=0;e<edits;++e){if(engine&&rng_int(r,0,99)<70)mutate_engine(child,r,n_routes,universe==0?2:rng_int(r,2,5));else mutate_one(child,r,n_routes,universe==0?2:rng_int(r,2,5));}}
    else if(local<local_per_island&&st.initialized){child=st.best_genome;ref=device_fit_mean(st.best_fit);if(engine){mutate_engine(child,r,n_routes,1);if(local%5==0)mutate_engine(child,r,n_routes,1);}else{mutate_one(child,r,n_routes,1);if(local%7==0)mutate_one(child,r,n_routes,1);}}
    else{int a=select_parent(fit,niche_counts,r,begin,end);child=cur[a];ref=device_fit_mean(fit[a]);int cross_pct=universe==0?28:16;if(rng_int(r,0,99)<cross_pct){int b=select_parent(fit,niche_counts,r,begin,end);crossover_modules(child,cur[b],r);}int edits=universe==0?rng_int(r,1,5):rng_int(r,3,12);int strength=universe==0?1:rng_int(r,1,3);for(int e=0;e<edits;++e){if(engine&&rng_int(r,0,99)<60)mutate_engine(child,r,n_routes,strength);else mutate_one(child,r,n_routes,strength);}}
    if(engine){int eph=4+(isl%7);if(eph>=GP_PHASES)eph=GP_PHASES-1;child.phase_mode[eph]=engine_mode_from(child.phase_mode[eph]);if(rng_int(r,0,7)==0)mutate_engine(child,r,n_routes,1);}
    normalize_genome(child,n_routes);next[i]=child;next_mgr[i]=decode_manager_device(child);next_parent_mean[i]=ref;rng[i]=r;
}

static bool try_alloc(SearchBuffers& b,int population,int family_count,int sub_islands){auto cm=[&](void** p,size_t n){cudaError_t e=cudaMalloc(p,n);return e==cudaSuccess;};size_t pf=(size_t)population*family_count,promof=(size_t)SEARCH_NICHES*family_count;if(!cm((void**)&b.pop_a,(size_t)population*sizeof(SearchGenome)))return false;if(!cm((void**)&b.pop_b,(size_t)population*sizeof(SearchGenome)))return false;if(!cm((void**)&b.mgr_a,(size_t)population*sizeof(GPManager)))return false;if(!cm((void**)&b.mgr_b,(size_t)population*sizeof(GPManager)))return false;if(!cm((void**)&b.fit,(size_t)population*sizeof(DeviceFitness)))return false;if(!cm((void**)&b.parent_mean_a,(size_t)population*sizeof(double)))return false;if(!cm((void**)&b.parent_mean_b,(size_t)population*sizeof(double)))return false;if(!cm((void**)&b.family_losses,pf))return false;if(!cm((void**)&b.family_ties,pf))return false;if(!cm((void**)&b.family_cases,pf))return false;if(!cm((void**)&b.rng,(size_t)population*sizeof(uint64_t)))return false;if(!cm((void**)&b.sims,(size_t)population*sizeof(Sim)))return false;if(!cm((void**)&b.rewards,(size_t)population*2*sizeof(double)))return false;if(!cm((void**)&b.islands,(size_t)sub_islands*sizeof(IslandState)))return false;if(!cm((void**)&b.niche_counts,SEARCH_NICHES*sizeof(uint32_t)))return false;if(!cm((void**)&b.generation_niche_vote,SEARCH_NICHES*sizeof(unsigned long long)))return false;if(!cm((void**)&b.archive_genomes,SEARCH_NICHES*sizeof(SearchGenome)))return false;if(!cm((void**)&b.archive_fit,SEARCH_NICHES*sizeof(DeviceFitness)))return false;if(!cm((void**)&b.archive_valid,SEARCH_NICHES))return false;if(!cm((void**)&b.promo_genomes,SEARCH_NICHES*sizeof(SearchGenome)))return false;if(!cm((void**)&b.promo_mgr,SEARCH_NICHES*sizeof(GPManager)))return false;if(!cm((void**)&b.promo_fit,SEARCH_NICHES*sizeof(DeviceFitness)))return false;if(!cm((void**)&b.promo_family_losses,promof))return false;if(!cm((void**)&b.promo_family_ties,promof))return false;if(!cm((void**)&b.promo_family_cases,promof))return false;if(!cm((void**)&b.promo_sims,SEARCH_NICHES*sizeof(Sim)))return false;if(!cm((void**)&b.promo_rewards,SEARCH_NICHES*2*sizeof(double)))return false;if(!cm((void**)&b.certified_genomes,SEARCH_NICHES*sizeof(SearchGenome)))return false;if(!cm((void**)&b.certified_fit,SEARCH_NICHES*sizeof(DeviceFitness)))return false;if(!cm((void**)&b.certified_level,SEARCH_NICHES))return false;if(!cm((void**)&b.promotion_games,sizeof(unsigned long long)))return false;return true;}

static size_t per_candidate_bytes(int family_count){return 2*sizeof(SearchGenome)+2*sizeof(GPManager)+sizeof(DeviceFitness)+2*sizeof(double)+(size_t)family_count*3+sizeof(uint64_t)+sizeof(Sim)+2*sizeof(double);}

static void write_program_binary(const std::filesystem::path& p,const GPProgram& g){std::filesystem::create_directories(p.parent_path());std::filesystem::path tmp=p;tmp += ".tmp";{std::ofstream out(tmp,std::ios::binary);out.write(reinterpret_cast<const char*>(&g),sizeof(g));if(!out)throw std::runtime_error("failed writing "+tmp.string());}std::filesystem::rename(tmp,p);}

static void write_checkpoint(const std::filesystem::path& out,const CheckpointHeader& h,const std::vector<IslandState>& islands,const std::vector<SearchGenome>& archive_genomes,const std::vector<DeviceFitness>& archive_fit,const std::vector<uint8_t>& archive_valid,const std::vector<SearchGenome>& certified_genomes,const std::vector<DeviceFitness>& certified_fit,const std::vector<uint8_t>& certified_level){std::filesystem::create_directories(out);std::filesystem::path p=out/"resident_checkpoint.bin",tmp=p;tmp += ".tmp";{std::ofstream f(tmp,std::ios::binary);f.write(reinterpret_cast<const char*>(&h),sizeof(h));f.write(reinterpret_cast<const char*>(islands.data()),(std::streamsize)islands.size()*sizeof(IslandState));f.write(reinterpret_cast<const char*>(archive_genomes.data()),SEARCH_NICHES*sizeof(SearchGenome));f.write(reinterpret_cast<const char*>(archive_fit.data()),SEARCH_NICHES*sizeof(DeviceFitness));f.write(reinterpret_cast<const char*>(archive_valid.data()),SEARCH_NICHES);f.write(reinterpret_cast<const char*>(certified_genomes.data()),SEARCH_NICHES*sizeof(SearchGenome));f.write(reinterpret_cast<const char*>(certified_fit.data()),SEARCH_NICHES*sizeof(DeviceFitness));f.write(reinterpret_cast<const char*>(certified_level.data()),SEARCH_NICHES);if(!f)throw std::runtime_error("checkpoint write failed");}std::filesystem::rename(tmp,p);std::ofstream meta(out/"checkpoint.json.tmp");meta<<"{\n  \"version\": 3,\n  \"generation\": "<<h.generation<<",\n  \"elapsed_seconds\": "<<h.elapsed_seconds<<",\n  \"candidates_evaluated\": "<<h.candidates_evaluated<<",\n  \"games_simulated\": "<<h.games_simulated<<",\n  \"sub_islands\": "<<h.sub_islands<<",\n  \"checkpoint_bytes\": "<<std::filesystem::file_size(p)<<"\n}\n";meta.close();std::filesystem::rename(out/"checkpoint.json.tmp",out/"checkpoint.json");}

static void write_subisland_archive(const std::filesystem::path& out,const std::vector<IslandState>& islands,int n_routes){for(size_t i=0;i<islands.size();++i)if(islands[i].initialized){GPProgram gp=distill_program(islands[i].best_genome,n_routes);std::ostringstream name;name<<"subisland_"<<std::setw(2)<<std::setfill('0')<<i<<"_best.bin";write_program_binary(out/name.str(),gp);}}

static void write_niche_archive(const std::filesystem::path& out,const std::vector<SearchGenome>& genomes,const std::vector<DeviceFitness>& fit,const std::vector<uint8_t>& valid,int n_routes){
    std::filesystem::create_directories(out);std::ofstream meta(out/"niche_archive.jsonl.tmp");int written=0;
    for(int n=0;n<SEARCH_NICHES;++n){if(!valid[n])continue;GPProgram gp=distill_program(genomes[n],n_routes);std::ostringstream name;name<<"niche_"<<std::setw(3)<<std::setfill('0')<<n<<"_best.bin";write_program_binary(out/name.str(),gp);const DeviceFitness& f=fit[n];meta<<"{\"niche\":"<<n<<",\"losing_families\":"<<f.losing_families<<",\"losses\":"<<f.losses<<",\"regressions\":"<<f.regressions<<",\"worst_family_milli\":"<<f.worst_family_milli<<",\"wins\":"<<f.wins<<",\"ties\":"<<f.ties<<",\"cases\":"<<f.cases<<",\"floor\":"<<f.floor<<",\"sum\":"<<f.sum<<",\"signature\":"<<f.signature<<",\"binary\":\""<<name.str()<<"\"}\n";++written;}
    meta.close();std::filesystem::rename(out/"niche_archive.jsonl.tmp",out/"niche_archive.jsonl");std::ofstream count(out/"niche_archive_count.txt.tmp");count<<written<<"\n";count.close();std::filesystem::rename(out/"niche_archive_count.txt.tmp",out/"niche_archive_count.txt");
}

static void write_certified_archive(const std::filesystem::path& out,const std::vector<SearchGenome>& genomes,const std::vector<DeviceFitness>& fit,const std::vector<uint8_t>& level,int n_routes){
    std::ofstream meta(out/"certified_archive.jsonl.tmp");for(int n=0;n<SEARCH_NICHES;++n){if(!level[n])continue;GPProgram gp=distill_program(genomes[n],n_routes);std::ostringstream name;name<<"certified_niche_"<<std::setw(3)<<std::setfill('0')<<n<<"_L"<<(int)level[n]<<".bin";write_program_binary(out/name.str(),gp);const DeviceFitness& f=fit[n];meta<<"{\"niche\":"<<n<<",\"fidelity_level\":"<<(int)level[n]<<",\"losing_families\":"<<f.losing_families<<",\"losses\":"<<f.losses<<",\"regressions\":"<<f.regressions<<",\"worst_family_milli\":"<<f.worst_family_milli<<",\"wins\":"<<f.wins<<",\"ties\":"<<f.ties<<",\"cases\":"<<f.cases<<",\"floor\":"<<f.floor<<",\"sum\":"<<f.sum<<",\"signature\":"<<f.signature<<",\"binary\":\""<<name.str()<<"\"}\n";}meta.close();std::filesystem::rename(out/"certified_archive.jsonl.tmp",out/"certified_archive.jsonl");
}

static std::vector<SearchGenome> load_checkpoint_seeds(const std::filesystem::path& path,int expected_islands){std::vector<SearchGenome> out;if(!std::filesystem::exists(path))return out;std::ifstream f(path,std::ios::binary);CheckpointHeader h{};f.read(reinterpret_cast<char*>(&h),sizeof(h));if(!f||h.magic!=SEARCH_MAGIC||h.version!=3)return out;std::vector<IslandState> s(h.sub_islands);f.read(reinterpret_cast<char*>(s.data()),(std::streamsize)s.size()*sizeof(IslandState));if(!f)return {};if(h.version==3){std::vector<SearchGenome> ag(SEARCH_NICHES),cg(SEARCH_NICHES);std::vector<DeviceFitness> af(SEARCH_NICHES),cf(SEARCH_NICHES);std::vector<uint8_t> av(SEARCH_NICHES),cl(SEARCH_NICHES);f.read(reinterpret_cast<char*>(ag.data()),SEARCH_NICHES*sizeof(SearchGenome));f.read(reinterpret_cast<char*>(af.data()),SEARCH_NICHES*sizeof(DeviceFitness));f.read(reinterpret_cast<char*>(av.data()),SEARCH_NICHES);f.read(reinterpret_cast<char*>(cg.data()),SEARCH_NICHES*sizeof(SearchGenome));f.read(reinterpret_cast<char*>(cf.data()),SEARCH_NICHES*sizeof(DeviceFitness));f.read(reinterpret_cast<char*>(cl.data()),SEARCH_NICHES);if(f){for(int i=0;i<SEARCH_NICHES;++i)if(cl[i]>=2)out.push_back(cg[i]);for(int i=0;i<SEARCH_NICHES;++i)if(av[i])out.push_back(ag[i]);}}for(auto& x:s)if(x.initialized)out.push_back(x.best_genome);if((int)out.size()>expected_islands)out.resize(expected_islands);return out;}

} // namespace

int main(int argc,char** argv){
    std::signal(SIGTERM, search_signal_handler);std::signal(SIGINT, search_signal_handler);
    std::filesystem::path bundle,output="memetic_search";std::vector<std::string> seed_program_paths;std::string universe_s="exploit";int population_override=0,sub_islands=SEARCH_DEFAULT_SUBISLANDS,engine_sub_islands=12,tpb=128,epoch_generations=8,stagnation_limit=80,local_per_island=64,promotion_family_every=8,promotion_medium_every=32,promotion_full_every=128;double walltime_seconds=3600.0,vram_target=0.95,reserve_mib=512.0,checkpoint_interval=120.0;uint64_t search_seed=340034;bool benchmark=false,resume=false,parity_check=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--bundle")bundle=need();else if(a=="--output")output=need();else if(a=="--universe")universe_s=need();else if(a=="--population")population_override=std::stoi(need());else if(a=="--sub-islands")sub_islands=std::stoi(need());else if(a=="--engine-sub-islands")engine_sub_islands=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--epoch-generations")epoch_generations=std::stoi(need());else if(a=="--stagnation-limit")stagnation_limit=std::stoi(need());else if(a=="--local-per-island")local_per_island=std::stoi(need());else if(a=="--promotion-family-every")promotion_family_every=std::stoi(need());else if(a=="--promotion-medium-every")promotion_medium_every=std::stoi(need());else if(a=="--promotion-full-every")promotion_full_every=std::stoi(need());else if(a=="--walltime-seconds")walltime_seconds=std::stod(need());else if(a=="--vram-target")vram_target=std::stod(need());else if(a=="--reserve-mib")reserve_mib=std::stod(need());else if(a=="--checkpoint-interval-seconds")checkpoint_interval=std::stod(need());else if(a=="--search-seed")search_seed=std::stoull(need());else if(a=="--seed-program")seed_program_paths.push_back(need());else if(a=="--benchmark")benchmark=true;else if(a=="--parity-check")parity_check=true;else if(a=="--resume")resume=true;else throw std::runtime_error("unknown argument: "+a);}
    if(bundle.empty())throw std::runtime_error("--bundle is required");if(sub_islands<4||sub_islands>64)throw std::runtime_error("--sub-islands must be 4..64");if(engine_sub_islands<0||engine_sub_islands>sub_islands)throw std::runtime_error("--engine-sub-islands must be 0..sub-islands");if(tpb!=64&&tpb!=128&&tpb!=256)throw std::runtime_error("--tpb must be 64,128,256");if(vram_target<0.50||vram_target>0.97)throw std::runtime_error("--vram-target must be 0.50..0.97");int universe=(universe_s=="explore")?1:0;

    const auto process_started=std::chrono::steady_clock::now();configure_device();TablesOwner tables;Config base{};cudaDeviceProp prop{};int dev_id=0;KAGCUDA_CHECK(cudaGetDevice(&dev_id));KAGCUDA_CHECK(cudaGetDeviceProperties(&prop,dev_id));

    int family_count=0;auto dev=load_dev_entries(bundle/"populations/dev_manifest.tsv",bundle/"populations/baseline_dev_cases.tsv",family_count);if(dev.size()!=168)throw std::runtime_error("resident search requires exactly 168 development tapes");if(family_count!=21)throw std::runtime_error("resident search requires exactly 21 development families");
    TierSpec tier0=load_tier(bundle/"populations/stage1_core12.txt",dev,"core12",120),tier1=load_tier(bundle/"populations/stage2_core21.txt",dev,"family21",240),tier2=load_tier(bundle/"populations/stage3_core42.txt",dev,"medium42",480);
    TierSpec tier3;tier3.name="dev168";tier3.max_step=720;for(int di=0;di<(int)dev.size();++di)for(int seat=0;seat<2;++seat)tier3.cases.push_back({di,dev[di].family,seat,dev[di].source_seed,dev[di].baseline_win[seat]});std::array<TierSpec,4> tiers={tier0,tier1,tier2,tier3};
    std::set<uint64_t> dev_seeds,core_seeds;for(const auto& e:dev)dev_seeds.insert(e.source_seed);for(const auto& c:tiers[0].cases)core_seeds.insert(c.seed);if(core_seeds.size()!=12)throw std::runtime_error("core12 must expose 12 distinct recorded seeds");std::cout<<"SEED_POLICY recorded_per_tape dev_unique="<<dev_seeds.size()<<" core12_unique="<<core_seeds.size()<<" core12_games="<<tiers[0].cases.size()<<"\n";

    auto candidate_paths=read_list(bundle/"actuators/v6_routes.txt");const int n_candidate_routes=(int)candidate_paths.size();if(n_candidate_routes<1||n_candidate_routes>255)throw std::runtime_error("invalid candidate route count");std::vector<Action> h_routes;std::vector<GPRouteDesc> h_desc;auto append=[&](const std::string& path){NativeTape t=load_native_tape_json(path);int off=(int)h_routes.size();h_routes.insert(h_routes.end(),t.actions.begin(),t.actions.end());h_desc.push_back({off,(int)t.actions.size()});};for(auto& p:candidate_paths)append(p);for(auto& e:dev)append((bundle/e.bundle_path).string());
    Action* d_routes=nullptr;GPRouteDesc* d_desc=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d_routes,h_routes.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&d_desc,h_desc.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMemcpy(d_routes,h_routes.data(),h_routes.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(d_desc,h_desc.data(),h_desc.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));

    std::vector<SearchGenome> priors;GPProgram default_program{};default_program.n_rules=1;default_program.rules[0].leaf={GPL_MANAGER,0,0,0};priors.push_back(encode_prior(default_program,n_candidate_routes));for(const auto& p:seed_program_paths){if(std::filesystem::exists(p)){priors.push_back(encode_prior(load_gp_program_binary(p),n_candidate_routes));}}
    if(resume){auto rs=load_checkpoint_seeds(output/"resident_checkpoint.bin",sub_islands);priors.insert(priors.begin(),rs.begin(),rs.end());}
    SearchGenome* d_priors=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d_priors,priors.size()*sizeof(SearchGenome)));KAGCUDA_CHECK(cudaMemcpy(d_priors,priors.data(),priors.size()*sizeof(SearchGenome),cudaMemcpyHostToDevice));

    if(parity_check){
        SearchGenome hg=priors.front();
        // Force the parity gate through the engine-exploitation execution path,
        // not merely through a legacy manager prior. This still uses only legal
        // observation-driven runtime inputs.
        hg.engine_profile=(uint8_t)(2|4|16|32|64);
        hg.engine_batch_bias=96;
        hg.engine_pressure=0x82;
        for(int ph=0;ph<GP_PHASES;++ph){
            hg.phase_mode[ph]=(uint8_t)(SM_MANAGER_ENGINE_MARKET+(ph%3));
            hg.phase_unit_route[ph]=(uint8_t)(ph%n_candidate_routes);
            hg.phase_market_route[ph]=(uint8_t)((ph*3+1)%n_candidate_routes);
        }
        GPManager hm=decode_manager_host(hg);GPProgram hp=distill_program(hg,n_candidate_routes);SearchGenome* dg=nullptr;GPManager* dm=nullptr;GPProgram* dp=nullptr;Sim* ds=nullptr;double* dr0=nullptr;double* dr1=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&dg,sizeof(SearchGenome)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,sizeof(GPManager)));KAGCUDA_CHECK(cudaMalloc((void**)&dp,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dr0,2*sizeof(double)));KAGCUDA_CHECK(cudaMalloc((void**)&dr1,2*sizeof(double)));KAGCUDA_CHECK(cudaMemcpy(dg,&hg,sizeof(hg),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dm,&hm,sizeof(hm),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dp,&hp,sizeof(hp),cudaMemcpyHostToDevice));int mismatches=0;for(const TierCase& c:tiers[0].cases){int od=n_candidate_routes+c.dev_index;eval_case_kernel<<<1,1>>>(base,tables.dev,dg,dm,1,d_routes,d_desc,n_candidate_routes,od,c.seat,c.seed,720,ds,dr0);eval_distilled_program_case_kernel<<<1,1>>>(base,tables.dev,dp,d_routes,d_desc,n_candidate_routes,od,c.seat,c.seed,720,ds,dr1);KAGCUDA_CHECK(cudaDeviceSynchronize());double a[2],b[2];KAGCUDA_CHECK(cudaMemcpy(a,dr0,sizeof(a),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(b,dr1,sizeof(b),cudaMemcpyDeviceToHost));if(a[0]!=b[0]||a[1]!=b[1]){++mismatches;std::cerr<<"PARITY_MISMATCH dev="<<c.dev_index<<" seat="<<c.seat<<" search="<<a[0]<<','<<a[1]<<" distilled="<<b[0]<<','<<b[1]<<"\n";}}cudaFree(dg);cudaFree(dm);cudaFree(dp);cudaFree(ds);cudaFree(dr0);cudaFree(dr1);std::cout<<"DISTILL_PARITY engine_mode=1 cases="<<tiers[0].cases.size()<<" mismatches="<<mismatches<<" sizeof_SearchGenome="<<sizeof(SearchGenome)<<" sizeof_GPProgram="<<sizeof(GPProgram)<<"\n";cudaFree(d_priors);cudaFree(d_routes);cudaFree(d_desc);return mismatches?3:0;
    }

    size_t free0=0,total0=0;KAGCUDA_CHECK(cudaMemGetInfo(&free0,&total0));size_t reserve=(size_t)(reserve_mib*1024.0*1024.0);size_t safe=free0>reserve?free0-reserve:(size_t)(free0*0.80);size_t budget=(size_t)(safe*vram_target);size_t per=per_candidate_bytes(family_count);int population=population_override>0?population_override:(int)(budget/per);if(population<sub_islands*256)population=sub_islands*256;population=(population/sub_islands)*sub_islands;
    SearchBuffers buf;while(population>=sub_islands*256){if(try_alloc(buf,population,family_count,sub_islands))break;cudaGetLastError();buf.clear();population=(int)(population*0.94);population=(population/sub_islands)*sub_islands;}if(!buf.pop_a)throw std::runtime_error("unable to allocate resident search buffers");
    KAGCUDA_CHECK(cudaMemset(buf.islands,0,sub_islands*sizeof(IslandState)));KAGCUDA_CHECK(cudaMemset(buf.archive_valid,0,SEARCH_NICHES));KAGCUDA_CHECK(cudaMemset(buf.certified_level,0,SEARCH_NICHES));KAGCUDA_CHECK(cudaMemset(buf.promotion_games,0,sizeof(unsigned long long)));
    size_t free1=0,total1=0;KAGCUDA_CHECK(cudaMemGetInfo(&free1,&total1));double alloc_frac=(double)(free0-free1)/(double)free0;double safe_alloc_frac=safe?((double)(free0-free1)/(double)safe):0.0;
    std::cout<<"RESIDENT_MEMORY gpu=\""<<prop.name<<"\" total="<<total0<<" free_before="<<free0<<" free_after="<<free1<<" target="<<vram_target<<" safe_available="<<safe<<" safe_budget="<<budget<<" population="<<population<<" per_candidate="<<per<<" allocated_fraction_of_initial_free="<<alloc_frac<<" allocated_fraction_of_safe="<<safe_alloc_frac<<" sizeof_SearchGenome="<<sizeof(SearchGenome)<<" sizeof_GPManager="<<sizeof(GPManager)<<" sizeof_Sim="<<sizeof(Sim)<<" sub_islands="<<sub_islands<<" engine_sub_islands="<<engine_sub_islands<<"\n";std::cout.flush();

    init_population_kernel<<<(population+tpb-1)/tpb,tpb>>>(buf.pop_a,buf.mgr_a,buf.parent_mean_a,buf.rng,population,sub_islands,engine_sub_islands,d_priors,(int)priors.size(),n_candidate_routes,search_seed,universe);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());
    SearchGenome* cur=buf.pop_a;SearchGenome* next=buf.pop_b;GPManager* cur_mgr=buf.mgr_a;GPManager* next_mgr=buf.mgr_b;double* cur_parent_mean=buf.parent_mean_a;double* next_parent_mean=buf.parent_mean_b;

    const auto search_started=std::chrono::steady_clock::now();const auto hard_deadline=search_started+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(walltime_seconds));double last_checkpoint=-1e30;uint64_t generation=0,candidates_evaluated=0,main_games_simulated=0,local_evals=0;std::vector<IslandState> host_islands(sub_islands);std::vector<SearchGenome> host_archive_genomes(SEARCH_NICHES),host_certified_genomes(SEARCH_NICHES);std::vector<DeviceFitness> host_archive_fit(SEARCH_NICHES),host_certified_fit(SEARCH_NICHES);std::vector<uint8_t> host_archive_valid(SEARCH_NICHES),host_certified_level(SEARCH_NICHES);
    auto enqueue_promotion=[&](int level){const TierSpec& tier=tiers[level];gather_promotion_kernel<<<1,256>>>(buf.archive_genomes,buf.archive_valid,buf.promo_genomes,buf.promo_mgr);KAGCUDA_CHECK(cudaMemsetAsync(buf.promo_family_losses,0,(size_t)SEARCH_NICHES*family_count));KAGCUDA_CHECK(cudaMemsetAsync(buf.promo_family_ties,0,(size_t)SEARCH_NICHES*family_count));KAGCUDA_CHECK(cudaMemsetAsync(buf.promo_family_cases,0,(size_t)SEARCH_NICHES*family_count));reset_metrics_kernel<<<1,256>>>(buf.promo_fit,SEARCH_NICHES);for(const TierCase& c:tier.cases){int opponent_desc=n_candidate_routes+c.dev_index;eval_case_masked_kernel<<<1,256>>>(base,tables.dev,buf.promo_genomes,buf.promo_mgr,buf.archive_valid,SEARCH_NICHES,d_routes,d_desc,n_candidate_routes,opponent_desc,c.seat,c.seed,tier.max_step,buf.promo_sims,buf.promo_rewards,buf.promotion_games);accumulate_case_masked_kernel<<<1,256>>>(buf.promo_rewards,buf.archive_valid,buf.promo_fit,buf.promo_family_losses,buf.promo_family_ties,buf.promo_family_cases,SEARCH_NICHES,family_count,c.family,c.seat,c.baseline_win);}finalize_promotion_kernel<<<1,256>>>(buf.promo_genomes,buf.archive_valid,buf.promo_fit,buf.promo_family_losses,buf.promo_family_ties,buf.promo_family_cases,SEARCH_NICHES,family_count);commit_certified_kernel<<<1,256>>>(buf.promo_genomes,buf.promo_fit,buf.archive_valid,level,buf.certified_genomes,buf.certified_fit,buf.certified_level);};
    auto enqueue_generation=[&](){const TierSpec& tier=tiers[0];KAGCUDA_CHECK(cudaMemsetAsync(buf.family_losses,0,(size_t)population*family_count));KAGCUDA_CHECK(cudaMemsetAsync(buf.family_ties,0,(size_t)population*family_count));KAGCUDA_CHECK(cudaMemsetAsync(buf.family_cases,0,(size_t)population*family_count));KAGCUDA_CHECK(cudaMemsetAsync(buf.niche_counts,0,SEARCH_NICHES*sizeof(uint32_t)));KAGCUDA_CHECK(cudaMemsetAsync(buf.generation_niche_vote,0,SEARCH_NICHES*sizeof(unsigned long long)));reset_metrics_kernel<<<(population+tpb-1)/tpb,tpb>>>(buf.fit,population);for(const TierCase& c:tier.cases){int opponent_desc=n_candidate_routes+c.dev_index;eval_case_kernel<<<(population+tpb-1)/tpb,tpb>>>(base,tables.dev,cur,cur_mgr,population,d_routes,d_desc,n_candidate_routes,opponent_desc,c.seat,c.seed,tier.max_step,buf.sims,buf.rewards);accumulate_case_kernel<<<(population+tpb-1)/tpb,tpb>>>(buf.rewards,buf.fit,buf.family_losses,buf.family_ties,buf.family_cases,population,family_count,c.family,c.seat,c.baseline_win);}finalize_metrics_kernel<<<(population+tpb-1)/tpb,tpb>>>(cur,buf.fit,buf.family_losses,buf.family_ties,buf.family_cases,cur_parent_mean,population,family_count,buf.niche_counts);niche_vote_kernel<<<(population+tpb-1)/tpb,tpb>>>(buf.fit,population,buf.generation_niche_vote);niche_commit_kernel<<<1,256>>>(cur,buf.fit,population,buf.generation_niche_vote,buf.archive_genomes,buf.archive_fit,buf.archive_valid);island_best_kernel<<<sub_islands,256>>>(cur,buf.fit,population,sub_islands,buf.islands);island_control_kernel<<<(sub_islands+63)/64,64>>>(buf.islands,sub_islands,stagnation_limit);uint64_t next_generation=generation+1;if(!benchmark){if(promotion_full_every>0&&next_generation%(uint64_t)promotion_full_every==0)enqueue_promotion(3);else if(promotion_medium_every>0&&next_generation%(uint64_t)promotion_medium_every==0)enqueue_promotion(2);else if(promotion_family_every>0&&next_generation%(uint64_t)promotion_family_every==0)enqueue_promotion(1);}evolve_kernel<<<(population+tpb-1)/tpb,tpb>>>(cur,next,next_mgr,next_parent_mean,buf.fit,buf.niche_counts,buf.rng,population,sub_islands,engine_sub_islands,n_candidate_routes,universe,local_per_island,buf.islands,d_priors,(int)priors.size(),buf.archive_genomes,buf.archive_fit,buf.archive_valid,buf.certified_genomes,buf.certified_fit,buf.certified_level);KAGCUDA_CHECK(cudaGetLastError());std::swap(cur,next);std::swap(cur_mgr,next_mgr);std::swap(cur_parent_mean,next_parent_mean);};

    double last_generation_seconds=0.0;
    while(!SEARCH_STOP_REQUESTED && std::chrono::steady_clock::now()<hard_deadline){auto epoch_started=std::chrono::steady_clock::now();double remain=std::chrono::duration<double>(hard_deadline-epoch_started).count();int queue_limit=1;if(last_generation_seconds>0.0){double safe_window=std::min(checkpoint_interval,std::max(0.0,remain-15.0));queue_limit=std::max(1,std::min(epoch_generations,(int)(safe_window/last_generation_seconds)));}int queued=0;for(;queued<queue_limit;){if(SEARCH_STOP_REQUESTED)break;if(last_generation_seconds>0.0&&std::chrono::duration<double>(hard_deadline-std::chrono::steady_clock::now()).count()<last_generation_seconds+10.0)break;enqueue_generation();candidates_evaluated+=(uint64_t)population;main_games_simulated+=(uint64_t)population*(uint64_t)tiers[0].cases.size();local_evals+=(uint64_t)sub_islands*(uint64_t)(local_per_island>1?local_per_island-1:0);++generation;++queued;if(benchmark)break;}if(queued==0)break;KAGCUDA_CHECK(cudaDeviceSynchronize());auto now=std::chrono::steady_clock::now();double epoch_seconds=std::chrono::duration<double>(now-epoch_started).count();last_generation_seconds=epoch_seconds/queued;double elapsed=std::chrono::duration<double>(now-search_started).count();KAGCUDA_CHECK(cudaMemcpy(host_islands.data(),buf.islands,sub_islands*sizeof(IslandState),cudaMemcpyDeviceToHost));int best=-1;for(int i=0;i<sub_islands;++i)if(host_islands[i].initialized&&(best<0||host_better(host_islands[i].best_fit,host_islands[best].best_fit)))best=i;uint64_t resets=0,improvements=0;for(auto& s:host_islands){resets+=s.resets;improvements+=s.improvements;}unsigned long long promotion_games=0;KAGCUDA_CHECK(cudaMemcpy(&promotion_games,buf.promotion_games,sizeof(promotion_games),cudaMemcpyDeviceToHost));uint64_t total_games=main_games_simulated+(uint64_t)promotion_games;double cps=elapsed>0?candidates_evaluated/elapsed:0.0,gps=elapsed>0?total_games/elapsed:0.0;std::cout<<"HEARTBEAT elapsed="<<elapsed<<" epoch_seconds="<<epoch_seconds<<" generation_seconds="<<last_generation_seconds<<" queued="<<queued<<" generation="<<generation<<" candidates="<<candidates_evaluated<<" games="<<total_games<<" main_games="<<main_games_simulated<<" promotion_games="<<promotion_games<<" candidates_per_sec="<<cps<<" games_per_sec="<<gps<<" resets="<<resets<<" improvements="<<improvements<<" local_refine_evals="<<local_evals<<" engine_sub_islands="<<engine_sub_islands;if(best>=0){auto& f=host_islands[best].best_fit;std::cout<<" best_island="<<best<<" losing_families="<<f.losing_families<<" losses="<<f.losses<<" regressions="<<f.regressions<<" worst_family_milli="<<f.worst_family_milli<<" floor="<<f.floor<<" marginal_delta="<<f.marginal_delta<<" wins="<<f.wins<<"/"<<f.cases;}std::cout<<"\n";std::cout.flush();if(elapsed-last_checkpoint>=checkpoint_interval||benchmark){KAGCUDA_CHECK(cudaMemcpy(host_archive_genomes.data(),buf.archive_genomes,SEARCH_NICHES*sizeof(SearchGenome),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_archive_fit.data(),buf.archive_fit,SEARCH_NICHES*sizeof(DeviceFitness),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_archive_valid.data(),buf.archive_valid,SEARCH_NICHES,cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_genomes.data(),buf.certified_genomes,SEARCH_NICHES*sizeof(SearchGenome),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_fit.data(),buf.certified_fit,SEARCH_NICHES*sizeof(DeviceFitness),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_level.data(),buf.certified_level,SEARCH_NICHES,cudaMemcpyDeviceToHost));CheckpointHeader h{};h.sub_islands=sub_islands;h.generation=generation;h.search_seed=search_seed;h.candidates_evaluated=candidates_evaluated;h.games_simulated=total_games;h.elapsed_seconds=elapsed;write_checkpoint(output,h,host_islands,host_archive_genomes,host_archive_fit,host_archive_valid,host_certified_genomes,host_certified_fit,host_certified_level);last_checkpoint=elapsed;}if(benchmark)break;}

    KAGCUDA_CHECK(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-search_started).count();KAGCUDA_CHECK(cudaMemcpy(host_islands.data(),buf.islands,sub_islands*sizeof(IslandState),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_archive_genomes.data(),buf.archive_genomes,SEARCH_NICHES*sizeof(SearchGenome),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_archive_fit.data(),buf.archive_fit,SEARCH_NICHES*sizeof(DeviceFitness),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_archive_valid.data(),buf.archive_valid,SEARCH_NICHES,cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_genomes.data(),buf.certified_genomes,SEARCH_NICHES*sizeof(SearchGenome),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_fit.data(),buf.certified_fit,SEARCH_NICHES*sizeof(DeviceFitness),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(host_certified_level.data(),buf.certified_level,SEARCH_NICHES,cudaMemcpyDeviceToHost));unsigned long long final_promotion_games=0;KAGCUDA_CHECK(cudaMemcpy(&final_promotion_games,buf.promotion_games,sizeof(final_promotion_games),cudaMemcpyDeviceToHost));uint64_t final_total_games=main_games_simulated+(uint64_t)final_promotion_games;CheckpointHeader h{};h.sub_islands=sub_islands;h.generation=generation;h.search_seed=search_seed;h.candidates_evaluated=candidates_evaluated;h.games_simulated=final_total_games;h.elapsed_seconds=elapsed;write_checkpoint(output,h,host_islands,host_archive_genomes,host_archive_fit,host_archive_valid,host_certified_genomes,host_certified_fit,host_certified_level);write_subisland_archive(output,host_islands,n_candidate_routes);write_niche_archive(output,host_archive_genomes,host_archive_fit,host_archive_valid,n_candidate_routes);write_certified_archive(output,host_certified_genomes,host_certified_fit,host_certified_level,n_candidate_routes);std::ofstream summary(output/"resident_summary.json");summary<<"{\n  \"status\": \"complete\",\n  \"universe\": \""<<universe_s<<"\",\n  \"gpu\": \""<<prop.name<<"\",\n  \"population\": "<<population<<",\n  \"sub_islands\": "<<sub_islands<<",\n  \"engine_sub_islands\": "<<engine_sub_islands<<",\n  \"engine_sub_island_fraction\": "<<(double)engine_sub_islands/sub_islands<<",\n  \"seed_policy\": \"recorded_per_tape\",\n  \"recorded_dev_unique_seeds\": "<<dev_seeds.size()<<",\n  \"recorded_core12_unique_seeds\": "<<core_seeds.size()<<",\n  \"generation\": "<<generation<<",\n  \"elapsed_seconds\": "<<elapsed<<",\n  \"candidates_evaluated\": "<<candidates_evaluated<<",\n  \"games_simulated\": "<<final_total_games<<",\n  \"main_games_simulated\": "<<main_games_simulated<<",\n  \"promotion_games_simulated\": "<<final_promotion_games<<",\n  \"local_refine_evals\": "<<local_evals<<",\n  \"vram_target\": "<<vram_target<<",\n  \"allocated_fraction_of_initial_free\": "<<alloc_frac<<",\n  \"allocated_fraction_of_safe\": "<<safe_alloc_frac<<"\n}\n";summary.close();
    std::cout<<"FINAL elapsed="<<elapsed<<" generation="<<generation<<" candidates="<<candidates_evaluated<<" games="<<final_total_games<<" main_games="<<main_games_simulated<<" promotion_games="<<final_promotion_games<<" local_refine_evals="<<local_evals<<" output="<<output<<"\n";
    buf.clear();cudaFree(d_priors);cudaFree(d_routes);cudaFree(d_desc);return 0;
}
