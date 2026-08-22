#include "../integrations/bkg4_champion/cpp_compat.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct TapeHeader { char magic[8]; uint32_t version; int32_t source_seat; uint64_t source_seed; uint32_t action_count; };
struct TapeActionHeader { uint8_t n_units; uint8_t n_orders; uint16_t reserved; };
struct TapeUnit { uint8_t op; uint8_t arg; int16_t n; };
struct TapeOrder { uint8_t op; uint8_t item; int16_t reserved; int32_t n; };
#pragma pack(pop)

struct Tape { std::vector<kag::Action> actions; };
struct Case { std::string path; int seat = 0; uint64_t seed = 0; };

#pragma pack(push, 1)
struct ModuleCondBin { uint16_t feature_id; uint8_t cmp; uint8_t reserved0; int16_t index; uint16_t reserved1; int32_t value; };
struct ModuleRuleBin { uint8_t n_conditions; uint8_t reserved0; uint16_t route_id; ModuleCondBin conditions[4]; };
struct ModuleGenomeBin { uint16_t prefix_step; uint8_t n_rules; uint8_t reserved0; ModuleRuleBin rules[8]; };
struct GenomeFileHeader { char magic[8]; uint32_t version; uint32_t count; };
#pragma pack(pop)
static_assert(sizeof(ModuleCondBin) == 12, "ModuleCondBin ABI drift");
static_assert(sizeof(ModuleRuleBin) == 52, "ModuleRuleBin ABI drift");
static_assert(sizeof(ModuleGenomeBin) == 420, "ModuleGenomeBin ABI drift");

std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> out; size_t p = 0;
    for (;;) { size_t q = s.find('\t', p); out.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p)); if (q == std::string::npos) break; p = q + 1; }
    return out;
}

std::vector<Case> load_cases(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("cannot open cases");
    std::vector<Case> out; std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto z = split_tab(line); if (z.size() != 7) throw std::runtime_error("case needs 7 tab fields");
        Case c; c.path = z[0]; c.seat = std::stoi(z[1]); c.seed = std::stoull(z[2]);
        if (c.seat < 0 || c.seat > 1) throw std::runtime_error("invalid case seat");
        out.push_back(c);
    }
    if (out.empty()) throw std::runtime_error("empty cases");
    return out;
}

Tape load_tape(const std::string& path) {
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open B4TP1 tape");
    TapeHeader h{}; in.read(reinterpret_cast<char*>(&h), sizeof(h));
    const char magic[8] = {'B','4','T','P','1','\0','\0','\0'};
    if (!in || std::memcmp(h.magic, magic, 8) || h.version != 1 || h.source_seat < 0 || h.source_seat > 1 || h.action_count < 1 || h.action_count > 720)
        throw std::runtime_error("invalid B4TP1 tape header");
    Tape out; out.actions.resize(h.action_count);
    for (uint32_t ai = 0; ai < h.action_count; ++ai) {
        TapeActionHeader ah{}; in.read(reinterpret_cast<char*>(&ah), sizeof(ah));
        if (!in || ah.reserved || ah.n_units < 1 || ah.n_units > kag::MAX_UNITS || ah.n_orders > 16) throw std::runtime_error("invalid B4TP1 action header");
        auto& a = out.actions[ai]; a.n_units = ah.n_units; a.n_orders = ah.n_orders;
        for (int ui = 0; ui < a.n_units; ++ui) { TapeUnit u{}; in.read(reinterpret_cast<char*>(&u), sizeof(u)); if (!in) throw std::runtime_error("truncated B4TP1 unit"); a.units[ui].op = u.op; a.units[ui].arg = u.arg; a.units[ui].n = u.n; }
        for (int oi = 0; oi < a.n_orders; ++oi) { TapeOrder o{}; in.read(reinterpret_cast<char*>(&o), sizeof(o)); if (!in || o.reserved) throw std::runtime_error("invalid B4TP1 order"); a.orders[oi].op = o.op; a.orders[oi].item = o.item; a.orders[oi].n = o.n; }
    }
    char extra = 0; if (in.read(&extra, 1)) throw std::runtime_error("trailing B4TP1 bytes");
    return out;
}

std::vector<ModuleGenomeBin> load_genomes(const std::string& path) {
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open B4GM1 genomes");
    GenomeFileHeader h{}; in.read(reinterpret_cast<char*>(&h), sizeof(h));
    const char magic[8] = {'B','4','G','M','1','\0','\0','\0'};
    if (!in || std::memcmp(h.magic, magic, 8) || h.version != 1 || h.count < 1 || h.count > 100000) throw std::runtime_error("invalid B4GM1 header");
    std::vector<ModuleGenomeBin> out(h.count); in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(out[0])));
    char extra = 0; if (!in || in.read(&extra, 1)) throw std::runtime_error("invalid/trailing B4GM1 payload");
    return out;
}

uint16_t module_route(const ModuleGenomeBin& g, const bonsaikg4::CppObservation4& obs, int step) {
    if (step < g.prefix_step) return 0xffffu;
    const int nr = g.n_rules > 8 ? 8 : g.n_rules;
    for (int ri = 0; ri < nr; ++ri) {
        const auto& rule = g.rules[ri]; const int nc = rule.n_conditions;
        if (nc < 1 || nc > 4) continue;
        bool ok = true;
        for (int ci = 0; ci < nc; ++ci) {
            const auto& c = rule.conditions[ci];
            if (!bonsaikg4::compare_i32(obs.feature(c.feature_id, c.index, 0), c.value, c.cmp)) { ok = false; break; }
        }
        if (ok) return rule.route_id;
    }
    return 0xffffu;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string controller, routes, legacy, cases_path, genomes_path;
        int feed_slot = 5, feed_count = 13, strategic_route_slot = 19;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
            if (a == "--controller") controller = need();
            else if (a == "--routes") routes = need();
            else if (a == "--legacy-routes") legacy = need();
            else if (a == "--cases") cases_path = need();
            else if (a == "--genomes") genomes_path = need();
            else if (a == "--feed-slot") feed_slot = std::stoi(need());
            else if (a == "--feed-count") feed_count = std::stoi(need());
            else if (a == "--strategic-route-slot") strategic_route_slot = std::stoi(need());
            else throw std::runtime_error("unknown argument " + a);
        }
        if (controller.empty() || routes.empty() || legacy.empty() || cases_path.empty() || genomes_path.empty()) throw std::runtime_error("missing required evaluator path");
        bonsaikg4_champion::CppChampionAgent4 agent(controller, routes, legacy, feed_slot, feed_count, strategic_route_slot);
        const auto cases = load_cases(cases_path); const auto genomes = load_genomes(genomes_path);
        for (size_t gi = 0; gi < genomes.size(); ++gi) for (size_t ci = 0; ci < cases.size(); ++ci) {
            const auto& c = cases[ci]; const Tape tape = load_tape(c.path);
            kag::Config cfg; cfg.seed = c.seed; cfg.episode_steps = 720; kag::Sim sim(cfg);
            bonsaikg4_champion::CppChampionState4 state{}; agent.reset(state);
            uint64_t trace_hash = 1469598103934665603ull;
            while (!sim.st.done) {
                bonsaikg4::CppObservation4 obs(&sim, c.seat);
                const uint16_t override_route = module_route(genomes[gi], obs, sim.st.step);
                kag::Action candidate{}; agent.act(state, sim, c.seat, candidate, override_route);
                trace_hash = bonsaikg4_champion::cpp_action_trace_hash(trace_hash, sim.st.step, candidate);
                kag::Action opponent{}; const int oi = 1 - c.seat;
                if (!tape.actions.empty()) opponent = tape.actions[static_cast<size_t>(sim.st.step < static_cast<int>(tape.actions.size()) ? sim.st.step : static_cast<int>(tape.actions.size()) - 1)];
                opponent.n_units = sim.st.farms[oi].n_units;
                if (c.seat == 0) sim.step_with_unit_counts(candidate, sim.st.farms[0].n_units, opponent, sim.st.farms[1].n_units);
                else sim.step_with_unit_counts(opponent, sim.st.farms[0].n_units, candidate, sim.st.farms[1].n_units);
            }
            const int32_t own = static_cast<int32_t>(sim.st.farms[c.seat].money);
            const int32_t opp = static_cast<int32_t>(sim.st.farms[1 - c.seat].money);
            std::cout << gi << '\t' << ci << '\t' << own << '\t' << opp << '\t' << state.meta.fault << '\t' << trace_hash << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
