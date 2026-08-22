#pragma once

#include "../../cpp/kag_adapter_v4.hpp"
#include "../../include/bonsai_kg_v4.hpp"
#include "../../include/bonsai_kg_v4_io.hpp"
#include "../stateful_v6/cpp_runtime.hpp"
#include "../stateful_v6/engine_exploits.hpp"
#include "extended_route_pack.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg4_champion {

struct CppLegacyState4 {
    bool trigger_decided = false;
    bool h4_trigger = false;
    int first_shop = -1;
    bool r86 = false;
    bool ownweed168 = false;
    bool r253 = false;
    bool fy144 = false;
    bool ice86 = false;
    bool ice_v32 = false;
    bool ff168 = false;
    bool ff_low = false;
    bool external_ok = false;
    int external_route = -1;

    void reset() { *this = CppLegacyState4{}; }
};

inline bool cpp_weed_at(const kag::Farm& farm, int x, int y) {
    return x >= 0 && x < kag::BOARD && y >= 0 && y < kag::BOARD && farm.tiles[y][x].kind == kag::T_WEED;
}

inline int cpp_shop_at(const kag::Sim& sim, int i) {
    return i >= 0 && i < sim.st.n_shops ? static_cast<int>(sim.st.shops[i]) : -1;
}

inline bool cpp_shop_prefix4(const kag::Sim& sim, int a, int b, int c, int d) {
    return cpp_shop_at(sim, 0) == a && cpp_shop_at(sim, 1) == b && cpp_shop_at(sim, 2) == c && cpp_shop_at(sim, 3) == d;
}

inline void cpp_materialize_route(const bonsaikg_v6::RoutePack& pack, uint16_t route, int step, int n_units, kag::Action& out) {
    bonsaikg_v6::materialize_cpp(pack.at(route, step), n_units, out);
}

inline void apply_cpp_legacy(const kag::Sim& sim, int player, const bonsaikg_v6::RoutePack& pack,
                             CppLegacyState4& s, kag::Action& action) {
    const int step = sim.st.step;
    if (step == 0) { s.reset(); return; }
    const kag::Farm& own = sim.st.farms[player];
    const kag::Farm& other = sim.st.farms[1 - player];

    if (!s.trigger_decided) {
        s.h4_trigger = other.n_units == 5 && other.money <= 32.0;
        s.trigger_decided = true;
    }

    if (s.h4_trigger) {
        if (s.first_shop < 0 && sim.st.n_shops > 0) s.first_shop = cpp_shop_at(sim, 0);
        if (step == 86) {
            s.r86 = cpp_weed_at(other, 0, 2);
            s.ice86 = s.first_shop == kag::SHOP_ICE_CREAM_SHOP && cpp_weed_at(own, 4, 2);
        }
        if (step == 144) s.fy144 = cpp_shop_at(sim, 0) == kag::SHOP_FARMERS_MARKET && cpp_shop_at(sim, 1) == kag::SHOP_YARN_STORE;
        if (step == 168) {
            s.ownweed168 = cpp_weed_at(own, 6, 3);
            s.ff168 = cpp_shop_at(sim, 0) == kag::SHOP_FARMERS_MARKET && cpp_shop_at(sim, 1) == kag::SHOP_FARMERS_MARKET && cpp_weed_at(own, 8, 0);
        }
        if (step == 253) {
            s.r253 = s.ownweed168 && s.first_shop == kag::SHOP_PIZZA_SHOP && other.money < 2200.0;
            if (s.ice86) s.ice_v32 = other.money >= 2100.0;
            if (s.ff168) s.ff_low = other.money < 1700.0;
        }
        int route = -1;
        if (s.ice86 && step >= 86 && step < 96) route = 202;
        else if (s.ice86 && step >= 96) route = (s.ice_v32 && step >= 253) ? 206 : 205;
        else if (s.ff168 && step >= 168) route = (s.ff_low && step >= 253) ? 201 : 207;
        else if (s.r86 && step >= 86 && step < 96) route = 202;
        else if (s.fy144 && step >= 144) route = 204;
        else if (s.r253 && step >= 253) route = 203;
        else if (step < 72) route = 200;
        else route = s.first_shop == kag::SHOP_PIZZA_SHOP ? 200 : 201;
        if (route >= 0) cpp_materialize_route(pack, static_cast<uint16_t>(route), step, own.n_units, action);
    }

    if (step == 25) s.external_ok = !s.h4_trigger;
    if (step == 288 && s.external_ok) {
        if (cpp_shop_prefix4(sim, 2, 4, 7, 7)) s.external_route = 208;
        else if (cpp_shop_prefix4(sim, 2, 0, 5, 3)) s.external_route = 209;
        else if (cpp_shop_prefix4(sim, 1, 5, 0, 0)) s.external_route = 210;
        else if (cpp_shop_prefix4(sim, 0, 2, 1, 4)) s.external_route = 209;
        else s.external_route = -1;
    }
    if (s.external_route >= 0 && step >= 288)
        cpp_materialize_route(pack, static_cast<uint16_t>(s.external_route), step, own.n_units, action);
}

struct CppChampionState4 {
    std::vector<int32_t> blackboard;
    std::vector<bonsaikg4::NodeState4> node_state;
    bonsaikg4::RuntimeMeta4 meta{};
    CppLegacyState4 legacy{};
    bonsaikg_v6::CppEngineExploitState exploit{};

    bonsaikg4::RuntimeView4 view() {
        return {blackboard.empty() ? nullptr : blackboard.data(), node_state.empty() ? nullptr : node_state.data(), &meta};
    }
};

class CppChampionAgent4 {
public:
    CppChampionAgent4(const std::string& controller_path, const std::string& routes_path, const std::string& legacy_path,
                      int feed_slot, int feed_count, int strategic_route_slot)
        : owned_(bonsaikg4::load_compiled4(controller_path)), policy_(owned_.view()),
          base_pack_(load_extended_route_pack(routes_path)), legacy_pack_(load_extended_route_pack(legacy_path)),
          feed_slot_(feed_slot), feed_count_(feed_count), strategic_route_slot_(strategic_route_slot) {
        if (policy_.n_bb < static_cast<uint16_t>(feed_slot_ + feed_count_)) throw std::runtime_error("feed slots outside BKG4 policy");
        if (strategic_route_slot_ >= 0 && policy_.n_bb <= static_cast<uint16_t>(strategic_route_slot_))
            throw std::runtime_error("strategic route slot outside BKG4 policy");
    }

    void reset(CppChampionState4& s) const {
        s.blackboard.assign(policy_.n_bb, 0);
        s.node_state.assign(policy_.n_state, bonsaikg4::NodeState4{});
        s.meta = bonsaikg4::RuntimeMeta4{};
        bonsaikg4::reset_runtime(policy_, s.view());
        s.legacy.reset();
        s.exploit.reset();
    }

    void act(CppChampionState4& s, const kag::Sim& sim, int player, kag::Action& out, uint16_t module_route = 0xffffu) const {
        bonsaikg4::CppObservation4 obs(&sim, player);
        bonsaikg4::ActionResult4 ignored{};
        const auto status = bonsaikg4::tick4(policy_, s.view(), obs, 0, ignored, nullptr);
        if (s.meta.fault || status != bonsaikg4::SUCCESS)
            throw std::runtime_error("BKG4 controller fault/non-success");
        int route = s.blackboard[1];
        if (strategic_route_slot_ >= 0 && s.blackboard[strategic_route_slot_]) route = s.blackboard[strategic_route_slot_];
        if (module_route != 0xffffu) route = static_cast<int>(module_route);
        cpp_materialize_route(base_pack_, static_cast<uint16_t>(route), sim.st.step, sim.st.farms[player].n_units, out);
        apply_cpp_legacy(sim, player, legacy_pack_, s.legacy, out);
        bonsaikg_v6::apply_cpp_engine_exploits(sim, player, out, s.exploit, s.legacy.external_ok);
        for (int u = 0; u < feed_count_ && u < out.n_units; ++u)
            if (s.blackboard[feed_slot_ + u] && out.units[u].op == kag::OP_FEED) out.units[u] = kag::UnitAction{};
    }

private:
    bonsaikg4::OwnedPolicy4 owned_;
    bonsaikg4::PolicyView4 policy_{};
    bonsaikg_v6::RoutePack base_pack_;
    bonsaikg_v6::RoutePack legacy_pack_;
    int feed_slot_ = 0;
    int feed_count_ = 0;
    int strategic_route_slot_ = -1;
};

inline uint64_t cpp_trace_mix(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ull;
    return h;
}

inline uint64_t cpp_action_trace_hash(uint64_t h, int step, const kag::Action& a) {
    h = cpp_trace_mix(h, static_cast<uint64_t>(static_cast<uint32_t>(step)));
    h = cpp_trace_mix(h, static_cast<uint64_t>(static_cast<uint32_t>(a.n_units)));
    for (int i = 0; i < a.n_units; ++i) {
        const auto& u = a.units[i];
        const uint64_t x = static_cast<uint64_t>(u.op) | (static_cast<uint64_t>(u.arg) << 8) |
                           (static_cast<uint64_t>(static_cast<uint16_t>(u.n)) << 16) |
                           (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32);
        h = cpp_trace_mix(h, x);
    }
    h = cpp_trace_mix(h, static_cast<uint64_t>(static_cast<uint32_t>(a.n_orders)));
    for (int i = 0; i < a.n_orders; ++i) {
        const auto& o = a.orders[i];
        const uint64_t x = static_cast<uint64_t>(o.op) | (static_cast<uint64_t>(o.item) << 8) |
                           (static_cast<uint64_t>(static_cast<uint32_t>(o.n)) << 16) |
                           (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 48);
        h = cpp_trace_mix(h, x);
    }
    return h;
}

}  // namespace bonsaikg4_champion
