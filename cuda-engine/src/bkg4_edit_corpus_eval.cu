#include "../../bonsai-kg/include/bonsai_kg_v4.hpp"
#include "../../bonsai-kg/include/bonsai_kg_v4_io.hpp"
#include "../../bonsai-kg/cuda/kag_adapter_v4.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/extended_route_pack.hpp"
#include "../../bonsai-kg/integrations/bkg4_champion/assets/identity.hpp"
#include "../../bonsai-kg/integrations/stateful_v6/route_pack.hpp"
#include "../include/runtime.cuh"
#include "../include/tape_io.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

namespace {

constexpr char RESULT_MAGIC_TEXT[] = "B4EC1";

#pragma pack(push, 1)
struct ModuleCondBin {
    uint16_t feature_id;
    uint8_t cmp;
    uint8_t reserved0;
    int16_t index;
    uint16_t reserved1;
    int32_t value;
};
struct ModuleRuleBin {
    uint8_t n_conditions;
    uint8_t reserved0;
    uint16_t route_id;
    ModuleCondBin conditions[4];
};
struct ModuleGenomeBin {
    uint16_t prefix_step;
    uint8_t n_rules;
    uint8_t reserved0;
    ModuleRuleBin rules[8];
};
struct GenomeFileHeader {
    char magic[8];
    uint32_t version;
    uint32_t count;
};
struct ResultHeader {
    char magic[8];
    uint32_t version;
    uint32_t genomes;
    uint32_t cases;
};
struct ResultBin {
    int32_t own_money;
    int32_t opp_money;
    uint16_t action_changes;
    uint16_t activations;
    uint16_t flags;
    uint16_t reserved;
    uint64_t action_trace_hash;
};
#pragma pack(pop)

static_assert(sizeof(ModuleCondBin) == 12, "ModuleCondBin ABI drift");
static_assert(sizeof(ModuleRuleBin) == 52, "ModuleRuleBin ABI drift");
static_assert(sizeof(ModuleGenomeBin) == 420, "ModuleGenomeBin ABI drift");
static_assert(sizeof(ResultBin) == 24, "B4EC1 result ABI drift");

struct CaseHost {
    std::string path;
    int seat = 0;
    uint64_t seed = 0;
    int family = 0;
    int protected_flag = 0;
    int own_history = 0;
    int role = 0;
};
struct CaseDev {
    int tape = 0;
    int seat = 0;
    uint64_t seed = 0;
    int family = 0;
    int protected_flag = 0;
    int own_history = 0;
    int role = 0;
};
struct OppDesc { int first = 0, count = 0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4* n = nullptr;
    uint16_t* c = nullptr;
    bonsaikg4::BbDecl4* bb = nullptr;
    bonsaikg4::ValueRef4* v = nullptr;
    bonsaikg4::FeatureArgRef4* a = nullptr;
    bonsaikg4::Predicate4* q = nullptr;
    bonsaikg4::Lifecycle4* l = nullptr;
    bonsaikg4::Effect4* e = nullptr;
    ~DevicePolicyOwner4() {
        if (n) cudaFree(n); if (c) cudaFree(c); if (bb) cudaFree(bb); if (v) cudaFree(v);
        if (a) cudaFree(a); if (q) cudaFree(q); if (l) cudaFree(l); if (e) cudaFree(e);
    }
};

template <class T>
static T* cp(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d = nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&d, v.size() * sizeof(T)));
    KAGCUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> out;
    size_t p = 0;
    for (;;) {
        size_t q = s.find('\t', p);
        out.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

static std::vector<CaseHost> load_cases(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cases: " + path);
    std::vector<CaseHost> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto z = split_tab(line);
        if (z.size() != 7) throw std::runtime_error("edit-corpus case needs 7 tab fields");
        CaseHost c;
        c.path = z[0]; c.seat = std::stoi(z[1]); c.seed = std::stoull(z[2]);
        c.family = std::stoi(z[3]); c.protected_flag = std::stoi(z[4]);
        c.own_history = std::stoi(z[5]); c.role = std::stoi(z[6]);
        if (c.seat < 0 || c.seat > 1) throw std::runtime_error("invalid case seat");
        out.push_back(c);
    }
    if (out.empty()) throw std::runtime_error("empty cases");
    return out;
}

static std::vector<ModuleGenomeBin> load_genomes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open genomes: " + path);
    GenomeFileHeader h{};
    in.read((char*)&h, sizeof(h));
    const char magic[8] = {'B','4','G','M','1','\0','\0','\0'};
    if (!in || std::memcmp(h.magic, magic, 8) || h.version != 1 || h.count < 1 || h.count > 100000)
        throw std::runtime_error("invalid B4GM1 genome header");
    std::vector<ModuleGenomeBin> out(h.count);
    in.read((char*)out.data(), out.size() * sizeof(out[0]));
    char extra = 0;
    if (!in || in.read(&extra, 1)) throw std::runtime_error("invalid/trailing B4GM1 genome payload");
    for (const auto& g : out) {
        if (g.n_rules > 8 || g.prefix_step >= 720) throw std::runtime_error("invalid ModuleGenome bounds");
        for (int ri = 0; ri < g.n_rules; ++ri) {
            if (g.rules[ri].n_conditions < 1 || g.rules[ri].n_conditions > 4)
                throw std::runtime_error("invalid ModuleGenome condition count");
        }
    }
    return out;
}

static bonsaikg_v6::RoutePack load_legacy_pack(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open legacy route pack");
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), {});
    auto full = bonsaikg::sha256_bytes(raw);
    if (bonsaikg_v6::hash_hex(full) != bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)
        throw std::runtime_error("legacy route pack SHA mismatch");
    bonsaikg_v6::PackHeader h{};
    if (raw.size() < sizeof(h)) throw std::runtime_error("short legacy route pack");
    std::memcpy(&h, raw.data(), sizeof(h));
    size_t desc_bytes = (size_t)h.route_count * sizeof(bonsaikg_v6::RouteDesc);
    size_t expected = sizeof(h) + desc_bytes + (size_t)h.action_count * sizeof(bonsaikg_v6::PackedAction);
    if (raw.size() != expected) throw std::runtime_error("legacy route pack size mismatch");
    bonsaikg_v6::RoutePack out;
    out.routes.resize(h.route_count); out.actions.resize(h.action_count);
    std::memcpy(out.routes.data(), raw.data() + sizeof(h), desc_bytes);
    std::memcpy(out.actions.data(), raw.data() + sizeof(h) + desc_bytes, out.actions.size() * sizeof(out.actions[0]));
    return out;
}

struct DeviceBasePack {
    const bonsaikg_v6::RouteDesc* routes = nullptr;
    const bonsaikg_v6::PackedAction* actions = nullptr;
    int n = 0;
    __device__ __forceinline__ const bonsaikg_v6::PackedAction* at(uint16_t route, int step) const {
        for (int i = 0; i < n; ++i) if (routes[i].route_id == route) {
            int s = step;
            if (s < 0) s = 0;
            if (s >= routes[i].steps) s = routes[i].steps - 1;
            return actions + routes[i].first + s;
        }
        return nullptr;
    }
};

__device__ __forceinline__ bool same_unit(const Action& a, const Action& b, int i) {
    return a.units[i].op == b.units[i].op && a.units[i].arg == b.units[i].arg && a.units[i].n == b.units[i].n;
}

__device__ __forceinline__ bool same_order(const Order& a, const Order& b) {
    return a.op == b.op && a.item == b.item && a.n == b.n;
}

__device__ __forceinline__ bool same_action(const Action& a, const Action& b) {
    if (a.n_units != b.n_units || a.n_orders != b.n_orders) return false;
    for (int i = 0; i < a.n_units; ++i) if (!same_unit(a, b, i)) return false;
    for (int i = 0; i < a.n_orders; ++i) if (!same_order(a.orders[i], b.orders[i])) return false;
    return true;
}

__device__ __forceinline__ uint64_t trace_mix(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ull;
    return h;
}

__device__ __forceinline__ uint64_t action_trace_hash(uint64_t h, int step, const Action& a) {
    h = trace_mix(h, (uint64_t)(uint32_t)step);
    h = trace_mix(h, (uint64_t)(uint32_t)a.n_units);
    for (int i = 0; i < a.n_units; ++i) {
        const UnitAction& u = a.units[i];
        uint64_t x = (uint64_t)u.op | ((uint64_t)u.arg << 8) | ((uint64_t)(uint16_t)u.n << 16) | ((uint64_t)(uint32_t)i << 32);
        h = trace_mix(h, x);
    }
    h = trace_mix(h, (uint64_t)(uint32_t)a.n_orders);
    for (int i = 0; i < a.n_orders; ++i) {
        const Order& o = a.orders[i];
        uint64_t x = (uint64_t)o.op | ((uint64_t)o.item << 8) | ((uint64_t)(uint32_t)o.n << 16) | ((uint64_t)(uint32_t)i << 48);
        h = trace_mix(h, x);
    }
    return h;
}

__device__ __forceinline__ void reconcile_feed(const int32_t* bb, int first_slot, int count, Action& a) {
    for (int u = 0; u < count && u < a.n_units; ++u)
        if (bb[first_slot + u] && a.units[u].op == OP_FEED) a.units[u] = UnitAction{};
}

__device__ __forceinline__ bool cmp_i32(int32_t a, int32_t b, uint8_t cmp) {
    switch (cmp) {
        case bonsaikg4::C_EQ: return a == b;
        case bonsaikg4::C_NE: return a != b;
        case bonsaikg4::C_LE: return a <= b;
        case bonsaikg4::C_GE: return a >= b;
        default: return false;
    }
}

__device__ __forceinline__ uint16_t module_route(const ModuleGenomeBin& g, const ObservationView& raw) {
    if (raw.step() < g.prefix_step) return 0xffffu;
    bonsaikg4::CudaObservation4 legal(raw);
    int nr = g.n_rules > 8 ? 8 : g.n_rules;
    for (int ri = 0; ri < nr; ++ri) {
        const auto& rule = g.rules[ri];
        int nc = rule.n_conditions;
        if (nc < 1 || nc > 4) continue;
        bool ok = true;
        for (int ci = 0; ci < nc; ++ci) {
            const auto& c = rule.conditions[ci];
            int32_t got = legal.feature(c.feature_id, c.index, 0);
            if (!cmp_i32(got, c.value, c.cmp)) { ok = false; break; }
        }
        if (ok) return rule.route_id;
    }
    return 0xffffu;
}

__device__ __forceinline__ bool build_final_action(
    const ObservationView& raw,
    const DeviceBasePack& base_pack,
    const bonsaikg4_champion::CudaLegacyPack& legacy_pack,
    bonsaikg4_champion::CudaLegacyState& legacy,
    bonsaikg_v6::CudaEngineExploitState& exploit,
    const int32_t* bb,
    int feed_slot,
    int feed_count,
    int strategic_route_slot,
    uint16_t route_override,
    Action& out
) {
    int route = bb[1];
    if (strategic_route_slot >= 0 && bb[strategic_route_slot]) route = bb[strategic_route_slot];
    if (route_override != 0xffffu) route = route_override;
    const auto* packed = base_pack.at((uint16_t)route, raw.step());
    if (!packed) return false;
    bonsaikg_v6::materialize_cuda(*packed, raw.farm(raw.player()).n_units(), out);
    bonsaikg4_champion::apply_legacy(raw, legacy_pack, legacy, out);
    bonsaikg_v6::apply_cuda_engine_exploits(raw, out, exploit, legacy.external_ok != 0, 100);
    reconcile_feed(bb, feed_slot, feed_count, out);
    return true;
}

__global__ void eval_kernel(
    Config cfg0,
    DeviceTables tables,
    bonsaikg4::PolicyView4 policy,
    DeviceBasePack base_pack,
    bonsaikg4_champion::CudaLegacyPack legacy_pack,
    const Action* opp_actions,
    const OppDesc* opp_desc,
    const CaseDev* cases,
    int ncases,
    const ModuleGenomeBin* genomes,
    int ngenomes,
    int feed_slot,
    int feed_count,
    int strategic_route_slot,
    int32_t* bb,
    bonsaikg4::NodeState4* ns,
    bonsaikg4::RuntimeMeta4* meta,
    Sim* sims,
    ResultBin* out
) {
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ncases * ngenomes;
    if (job >= total) return;
    int gi = job / ncases;
    int ci = job - gi * ncases;
    const CaseDev c = cases[ci];
    Config cfg = cfg0;
    cfg.seed = c.seed;
    Sim& sim = sims[job];
    sim.init(cfg, &tables);
    bonsaikg4::RuntimeView4 st{
        policy.n_bb ? bb + (size_t)job * policy.n_bb : nullptr,
        policy.n_state ? ns + (size_t)job * policy.n_state : nullptr,
        meta + job,
    };
    bonsaikg4::reset_runtime(policy, st);
    bonsaikg4_champion::CudaLegacyState legacy{};
    bonsaikg_v6::CudaEngineExploitState exploit{};
    bonsaikg4_champion::reset_legacy(legacy);
    bonsaikg_v6::reset_cuda_engine_exploits(exploit);
    uint16_t action_changes = 0;
    uint16_t activations = 0;
    uint16_t flags = 0;
    uint64_t trace_hash = 1469598103934665603ull;
    const ModuleGenomeBin& genome = genomes[gi];

    while (!sim.st.done) {
        int opponent_seat = 1 - c.seat;
        ObservationView raw = make_observation(sim, c.seat);
        bonsaikg4::CudaObservation4 obs4(raw);
        bonsaikg4::ActionResult4 ignored{};
        auto status = bonsaikg4::tick4(policy, st, obs4, 0, ignored, nullptr);
        if (meta[job].fault || status != bonsaikg4::SUCCESS) { flags |= 1; break; }

        auto legacy_cmp = legacy;
        auto exploit_cmp = exploit;
        Action baseline{};
        if (!build_final_action(raw, base_pack, legacy_pack, legacy_cmp, exploit_cmp, st.blackboard,
                                feed_slot, feed_count, strategic_route_slot, 0xffffu, baseline)) {
            flags |= 2; break;
        }

        uint16_t override_route = module_route(genome, raw);
        if (override_route != 0xffffu && activations != 0xffffu) ++activations;
        Action candidate{};
        if (!build_final_action(raw, base_pack, legacy_pack, legacy, exploit, st.blackboard,
                                feed_slot, feed_count, strategic_route_slot, override_route, candidate)) {
            flags |= 2; break;
        }
        if (!same_action(candidate, baseline) && action_changes != 0xffffu) ++action_changes;
        trace_hash = action_trace_hash(trace_hash, sim.st.step, candidate);

        Action opponent{};
        OppDesc d = opp_desc[c.tape];
        if (d.count) {
            int s = sim.st.step < d.count ? sim.st.step : d.count - 1;
            opponent = opp_actions[d.first + s];
        }
        opponent.n_units = sim.st.farms[opponent_seat].n_units;
        if (c.seat == 0)
            sim.step_with_unit_counts(candidate, sim.st.farms[0].n_units, opponent, sim.st.farms[1].n_units);
        else
            sim.step_with_unit_counts(opponent, sim.st.farms[0].n_units, candidate, sim.st.farms[1].n_units);
    }

    ResultBin r{};
    r.own_money = (int32_t)sim.st.farms[c.seat].money;
    r.opp_money = (int32_t)sim.st.farms[1 - c.seat].money;
    r.action_changes = action_changes;
    r.activations = activations;
    r.flags = flags;
    r.reserved = 0;
    r.action_trace_hash = trace_hash;
    out[job] = r;
}

static void write_results(const std::string& path, const std::vector<ResultBin>& rows, int genomes, int cases) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write edit-corpus CUDA result file");
    ResultHeader h{{'B','4','E','C','1','\0','\0','\0'}, 1u, (uint32_t)genomes, (uint32_t)cases};
    out.write((char*)&h, sizeof(h));
    out.write((char*)rows.data(), rows.size() * sizeof(rows[0]));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string controller, routes, legacy, cases_path, genomes_path, out_path;
        int feed_slot = 5, feed_count = 13, strategic_route_slot = 19, tpb = 64;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&]() {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--controller") controller = need();
            else if (a == "--routes") routes = need();
            else if (a == "--legacy-routes") legacy = need();
            else if (a == "--cases") cases_path = need();
            else if (a == "--genomes") genomes_path = need();
            else if (a == "--out") out_path = need();
            else if (a == "--feed-slot") feed_slot = std::stoi(need());
            else if (a == "--feed-count") feed_count = std::stoi(need());
            else if (a == "--strategic-route-slot") strategic_route_slot = std::stoi(need());
            else if (a == "--tpb") tpb = std::stoi(need());
            else throw std::runtime_error("unknown argument " + a);
        }
        if (controller.empty() || routes.empty() || legacy.empty() || cases_path.empty() || genomes_path.empty() || out_path.empty())
            throw std::runtime_error("required: --controller --routes --legacy-routes --cases --genomes --out");
        if (feed_slot < 0 || feed_count < 0 || strategic_route_slot < -1 || tpb <= 0)
            throw std::runtime_error("invalid evaluator numeric argument");

        auto owned = bonsaikg4::load_compiled4(controller);
        auto hp = owned.view();
        if (hp.n_bb < (uint16_t)(feed_slot + feed_count)) throw std::runtime_error("feed slots outside policy BB");
        if (strategic_route_slot >= 0 && hp.n_bb <= (uint16_t)strategic_route_slot)
            throw std::runtime_error("strategic route slot outside policy BB");
        DevicePolicyOwner4 po;
        po.n = cp(owned.nodes); po.c = cp(owned.child_refs); po.bb = cp(owned.bb); po.v = cp(owned.values);
        po.a = cp(owned.feature_args); po.q = cp(owned.predicates); po.l = cp(owned.lifecycles); po.e = cp(owned.effects);
        auto dp = hp;
        dp.nodes = po.n; dp.child_refs = po.c; dp.bb = po.bb; dp.values = po.v;
        dp.feature_args = po.a; dp.predicates = po.q; dp.lifecycles = po.l; dp.effects = po.e;

        auto rp = bonsaikg4_champion::load_extended_route_pack(routes);
        auto lp = load_legacy_pack(legacy);
        auto* dr = cp(rp.routes); auto* da = cp(rp.actions);
        auto* dlr = cp(lp.routes); auto* dla = cp(lp.actions);
        DeviceBasePack base_pack{dr, da, (int)rp.routes.size()};
        bonsaikg4_champion::CudaLegacyPack legacy_pack{dlr, dla, (int)lp.routes.size()};

        auto host_cases = load_cases(cases_path);
        auto host_genomes = load_genomes(genomes_path);
        std::vector<Action> host_opp_actions;
        std::vector<OppDesc> host_opp_desc;
        std::vector<CaseDev> host_case_dev;
        for (const auto& c : host_cases) {
            auto tape = load_native_tape_json(c.path);
            int off = (int)host_opp_actions.size();
            host_opp_actions.insert(host_opp_actions.end(), tape.actions.begin(), tape.actions.end());
            host_opp_desc.push_back({off, (int)tape.actions.size()});
            host_case_dev.push_back({(int)host_opp_desc.size() - 1, c.seat, c.seed, c.family, c.protected_flag, c.own_history, c.role});
        }

        auto* d_opp_actions = cp(host_opp_actions);
        auto* d_opp_desc = cp(host_opp_desc);
        auto* d_cases = cp(host_case_dev);
        auto* d_genomes = cp(host_genomes);
        size_t jobs = (size_t)host_cases.size() * host_genomes.size();
        int32_t* d_bb = nullptr;
        bonsaikg4::NodeState4* d_ns = nullptr;
        bonsaikg4::RuntimeMeta4* d_meta = nullptr;
        Sim* d_sims = nullptr;
        ResultBin* d_out = nullptr;
        if (hp.n_bb) KAGCUDA_CHECK(cudaMalloc((void**)&d_bb, jobs * hp.n_bb * sizeof(int32_t)));
        if (hp.n_state) KAGCUDA_CHECK(cudaMalloc((void**)&d_ns, jobs * hp.n_state * sizeof(bonsaikg4::NodeState4)));
        KAGCUDA_CHECK(cudaMalloc((void**)&d_meta, jobs * sizeof(*d_meta)));
        KAGCUDA_CHECK(cudaMalloc((void**)&d_sims, jobs * sizeof(*d_sims)));
        KAGCUDA_CHECK(cudaMalloc((void**)&d_out, jobs * sizeof(*d_out)));

        configure_device();
        TablesOwner tables;
        Config cfg;
        cfg.episode_steps = 720;
        cudaEvent_t e0, e1;
        KAGCUDA_CHECK(cudaEventCreate(&e0)); KAGCUDA_CHECK(cudaEventCreate(&e1));
        KAGCUDA_CHECK(cudaEventRecord(e0));
        eval_kernel<<<(jobs + tpb - 1) / tpb, tpb>>>(
            cfg, tables.dev, dp, base_pack, legacy_pack,
            d_opp_actions, d_opp_desc, d_cases, (int)host_cases.size(),
            d_genomes, (int)host_genomes.size(), feed_slot, feed_count, strategic_route_slot,
            d_bb, d_ns, d_meta, d_sims, d_out);
        KAGCUDA_CHECK(cudaGetLastError());
        KAGCUDA_CHECK(cudaEventRecord(e1)); KAGCUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        KAGCUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<ResultBin> host_out(jobs);
        KAGCUDA_CHECK(cudaMemcpy(host_out.data(), d_out, jobs * sizeof(ResultBin), cudaMemcpyDeviceToHost));
        write_results(out_path, host_out, (int)host_genomes.size(), (int)host_cases.size());
        int bad = 0;
        unsigned long long changes = 0, activations = 0;
        for (const auto& r : host_out) { bad += r.flags != 0; changes += r.action_changes; activations += r.activations; }
        std::cout << RESULT_MAGIC_TEXT
                  << " genomes=" << host_genomes.size()
                  << " cases=" << host_cases.size()
                  << " games=" << jobs
                  << " gpu_ms=" << ms
                  << " games_s=" << (jobs / (ms / 1000.0))
                  << " action_changes=" << changes
                  << " activations=" << activations
                  << " bad=" << bad << "\n";

        cudaFree(dr); cudaFree(da); cudaFree(dlr); cudaFree(dla);
        cudaFree(d_opp_actions); cudaFree(d_opp_desc); cudaFree(d_cases); cudaFree(d_genomes);
        if (d_bb) cudaFree(d_bb); if (d_ns) cudaFree(d_ns); cudaFree(d_meta); cudaFree(d_sims); cudaFree(d_out);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
