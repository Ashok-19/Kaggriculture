#pragma once

#include "bonsai_kg_v4_abi.hpp"

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define B4IM_HD __host__ __device__ __forceinline__
#else
#define B4IM_HD inline
#endif

namespace bonsaikg4::invariant {

constexpr uint8_t B4IM1_VERSION = 1;
constexpr int B4IM1_MAX_RULES = 3;
constexpr int B4IM1_MAX_GUARDS = 5;
constexpr int B4IM1_CANDIDATE_SIZE = 256;

enum Domain : uint8_t { DOMAIN_GLOBAL = 0, DOMAIN_EACH_OWN_UNIT = 1 };
enum Control : uint8_t { CONTROL_INSTANT = 0, CONTROL_COMMITTED = 1 };
enum ActionKind : uint8_t {
    ACTION_NOOP = 0,
    ACTION_ROUTE_OVERRIDE = 1,
    ACTION_UNIT_REWRITE = 2,
    ACTION_MARKET_INTENT = 3,
    ACTION_COMMITTED_ROUTE = 4,
};
enum ArgMode : uint8_t { ARG_CONST = 0, ARG_BOUND_TILE = 1, ARG_BOUND_UNIT = 2 };

#pragma pack(push, 1)
struct CandidateHeader {
    char magic[4];
    uint8_t version = 0;
    uint8_t n_rules = 0;
    uint16_t flags = 0;
    uint64_t reserved = 0;
};
struct GuardBin {
    uint16_t feature_id = 0;
    uint8_t cmp = 0;
    uint8_t arg_mode = 0;
    int32_t arg = 0;
    int32_t value = 0;
};
struct RuleBin {
    uint8_t domain = 0;
    uint8_t control = 0;
    uint8_t action = 0;
    uint8_t n_guard = 0;
    uint8_t n_exit = 0;
    uint8_t effect = 0;
    uint8_t authority = 0;
    uint8_t priority = 0;
    int16_t p0 = 0;
    int16_t p1 = 0;
    int16_t p2 = 0;
    int16_t p3 = 0;
    int16_t p4 = 0;
    uint16_t reserved = 0;
    GuardBin guards[B4IM1_MAX_GUARDS]{};
};
struct Candidate {
    CandidateHeader header{};
    RuleBin rules[B4IM1_MAX_RULES]{};
};
struct FileHeader {
    char magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
};
#pragma pack(pop)

static_assert(sizeof(CandidateHeader) == 16, "B4IM1 header ABI drift");
static_assert(sizeof(GuardBin) == 12, "B4IM1 guard ABI drift");
static_assert(sizeof(RuleBin) == 80, "B4IM1 rule ABI drift");
static_assert(sizeof(Candidate) == B4IM1_CANDIDATE_SIZE, "B4IM1 candidate ABI drift");
static_assert(sizeof(FileHeader) == 16, "B4IM1 file header ABI drift");

struct RuntimeState {
    uint8_t committed_mask = 0;
};

struct Decision {
    uint16_t route_override = NONE_U16;
    uint8_t activations = 0;
    uint8_t market_count = 0;
    uint64_t rewrite_masks[B4IM1_MAX_RULES]{};
    uint8_t market_rules[B4IM1_MAX_RULES]{};
};

B4IM_HD bool magic_ok(const Candidate& c) {
    return c.header.magic[0] == 'B' && c.header.magic[1] == '4' &&
           c.header.magic[2] == 'I' && c.header.magic[3] == 'M';
}

template <class T>
B4IM_HD bool bytes_zero(const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i) if (p[i] != 0) return false;
    return true;
}

B4IM_HD bool is_unit_feature(uint16_t f) {
    return f == F_OWN_WORKER_X || f == F_OWN_WORKER_Y || f == F_OPP_WORKER_X ||
           f == F_OPP_WORKER_Y || f == F_OWN_WORKER_TILE || f == F_OPP_WORKER_TILE;
}

B4IM_HD bool is_tile_feature(uint16_t f);

B4IM_HD bool valid_candidate(const Candidate& c) {
    if (!magic_ok(c) || c.header.version != B4IM1_VERSION || c.header.n_rules > B4IM1_MAX_RULES ||
        c.header.flags != 0 || c.header.reserved != 0) return false;
    for (int ri = 0; ri < c.header.n_rules; ++ri) {
        const RuleBin& r = c.rules[ri];
        if (r.domain > DOMAIN_EACH_OWN_UNIT || r.control > CONTROL_COMMITTED || r.action > ACTION_COMMITTED_ROUTE ||
            r.n_guard + r.n_exit > B4IM1_MAX_GUARDS || r.effect > 8 || r.authority > MAX_AUTHORITY || r.reserved != 0)
            return false;
        for (int gi = 0; gi < r.n_guard + r.n_exit; ++gi) {
            const GuardBin& g = r.guards[gi];
            if (g.feature_id > LAST_FEATURE_V1 || g.cmp > C_GE || g.arg_mode > ARG_BOUND_UNIT) return false;
            if (g.feature_id == F_OWN_INVENTORY && r.domain != DOMAIN_EACH_OWN_UNIT) return false;
            uint8_t expected = ARG_CONST;
            if (r.domain == DOMAIN_EACH_OWN_UNIT && is_tile_feature(g.feature_id)) expected = ARG_BOUND_TILE;
            else if (r.domain == DOMAIN_EACH_OWN_UNIT && is_unit_feature(g.feature_id)) expected = ARG_BOUND_UNIT;
            if (g.feature_id == F_OWN_INVENTORY) expected = ARG_CONST;
            if (g.arg_mode != expected) return false;
        }
        for (int gi = r.n_guard + r.n_exit; gi < B4IM1_MAX_GUARDS; ++gi)
            if (!bytes_zero(r.guards[gi])) return false;
        if (r.domain == DOMAIN_GLOBAL && r.action == ACTION_UNIT_REWRITE) return false;
        if (r.domain == DOMAIN_EACH_OWN_UNIT && r.action != ACTION_UNIT_REWRITE) return false;
        if ((r.action == ACTION_COMMITTED_ROUTE) != (r.control == CONTROL_COMMITTED)) return false;
        if (r.control == CONTROL_COMMITTED && r.n_exit == 0) return false;
        if (r.control == CONTROL_INSTANT && r.n_exit != 0) return false;
        if (r.action != ACTION_MARKET_INTENT &&
            (r.authority != AUTH_COMMITTED_STRATEGY || r.priority != 0)) return false;
        if (r.action == ACTION_NOOP && (r.p0 || r.p1 || r.p2 || r.p3 || r.p4)) return false;
        if ((r.action == ACTION_ROUTE_OVERRIDE || r.action == ACTION_COMMITTED_ROUTE) &&
            (r.p0 < 0 || r.p1 || r.p2 || r.p3 || r.p4)) return false;
        if (r.action == ACTION_UNIT_REWRITE) {
            if (r.p0 < U_PASS || r.p0 > U_CARE || r.p1 < U_PASS || r.p1 > U_CARE || r.p4) return false;
            if (r.p1 == U_PLANT) {
                if (r.p2 < 0 || r.p2 >= 5 || r.p3 != 1) return false;
            } else if (r.p1 == U_PICKUP || r.p1 == U_PLACE) {
                if (r.p2 < 0 || r.p2 >= 12 || r.p3 < 1) return false;
            } else if (r.p2 != 0 || r.p3 != 1) return false;
        }
        if (r.action == ACTION_MARKET_INTENT) {
            if (r.p0 < M_HIRE || r.p0 > M_SELL || r.p2 < 1 ||
                r.p3 != r.authority || r.p4 != r.priority) return false;
            if ((r.p0 == M_HIRE || r.p0 == M_BUY_LAND) && (r.p1 != 0 || r.p2 != 1)) return false;
            if (r.p0 == M_BUY_SEED && (r.p1 < 0 || r.p1 >= 5)) return false;
            if (r.p0 == M_BUY_PRODUCT && r.p1 != 0 && r.p1 != 8) return false;
            if (r.p0 == M_BUY_ANIMAL && (r.p1 < 9 || r.p1 > 11)) return false;
            if (r.p0 == M_SELL && (r.p1 < 0 || r.p1 >= 9)) return false;
        }
    }
    for (int ri = c.header.n_rules; ri < B4IM1_MAX_RULES; ++ri)
        if (!bytes_zero(c.rules[ri])) return false;
    return true;
}

B4IM_HD void reset(RuntimeState& s) { s.committed_mask = 0; }

B4IM_HD bool is_tile_feature(uint16_t f) {
    switch (f) {
        case F_OWN_TILE_KIND: case F_OPP_TILE_KIND:
        case F_OWN_TILE_ITEM: case F_OPP_TILE_ITEM:
        case F_OWN_TILE_YIELD: case F_OPP_TILE_YIELD:
        case F_OWN_TILE_WATERED: case F_OPP_TILE_WATERED:
        case F_OWN_TILE_FED: case F_OPP_TILE_FED:
        case F_OWN_TILE_CARED: case F_OPP_TILE_CARED:
        case F_OWN_TILE_FERT_AVAILABLE: case F_OPP_TILE_FERT_AVAILABLE:
        case F_OWN_TILE_PLANTED_DAY: case F_OPP_TILE_PLANTED_DAY:
        case F_OWN_TILE_CONSECUTIVE_UNWATERED: case F_OPP_TILE_CONSECUTIVE_UNWATERED:
        case F_OWN_TILE_MAX_LIFESPAN_STEP: case F_OPP_TILE_MAX_LIFESPAN_STEP:
        case F_OWN_TILE_FERTILIZED_UNTIL_DAY: case F_OPP_TILE_FERTILIZED_UNTIL_DAY:
        case F_OWN_TILE_PLACED_DAY: case F_OPP_TILE_PLACED_DAY:
        case F_OWN_TILE_CONSECUTIVE_UNFED: case F_OPP_TILE_CONSECUTIVE_UNFED:
        case F_OWN_TILE_PENDING_CARE_BONUS: case F_OPP_TILE_PENDING_CARE_BONUS:
        case F_OWN_TILE_HAS_ANIMAL: case F_OPP_TILE_HAS_ANIMAL:
            return true;
        default: return false;
    }
}

template <class Obs>
B4IM_HD int32_t guard_value(const GuardBin& g, const Obs& obs, int bound_unit) {
    if (g.arg_mode == ARG_BOUND_TILE) {
        const int32_t tile = obs.feature(F_OWN_WORKER_TILE, bound_unit, 0);
        return obs.feature(g.feature_id, tile, 0);
    }
    if (g.arg_mode == ARG_BOUND_UNIT)
        return obs.feature(g.feature_id, bound_unit, 0);
    if (g.feature_id == F_OWN_INVENTORY && bound_unit >= 0)
        return obs.feature(g.feature_id, bound_unit, g.arg);
    return obs.feature(g.feature_id, g.arg, 0);
}

template <class Obs>
B4IM_HD bool guard_matches(const RuleBin& r, const Obs& obs, int first, int count, int bound_unit = -1) {
    for (int i = 0; i < count; ++i) {
        const GuardBin& g = r.guards[first + i];
        if (!compare_i32(guard_value(g, obs, bound_unit), g.value, g.cmp)) return false;
    }
    return true;
}

B4IM_HD uint16_t priority_key(const RuleBin& r) {
    return make_priority_key(r.authority, r.priority);
}

B4IM_HD void insert_market_rule(const Candidate& c, Decision& d, uint8_t ri) {
    const uint16_t key = priority_key(c.rules[ri]);
    int pos = d.market_count;
    if (pos < B4IM1_MAX_RULES) ++d.market_count;
    else {
        const uint8_t worst = d.market_rules[B4IM1_MAX_RULES - 1];
        const uint16_t worst_key = priority_key(c.rules[worst]);
        if (key < worst_key || (key == worst_key && ri > worst)) return;
        pos = B4IM1_MAX_RULES - 1;
    }
    while (pos > 0) {
        const uint8_t prev = d.market_rules[pos - 1];
        const uint16_t prev_key = priority_key(c.rules[prev]);
        if (key < prev_key || (key == prev_key && ri > prev)) break;
        d.market_rules[pos] = prev;
        --pos;
    }
    d.market_rules[pos] = ri;
}

template <class Obs>
B4IM_HD Decision evaluate(const Candidate& c, RuntimeState& state, const Obs& obs) {
    Decision d{};
    if (!valid_candidate(c)) return d;
    int n_units = obs.feature(F_OWN_UNITS, 0, 0);
    if (n_units < 0) n_units = 0;
    if (n_units > MAX_UNITS) n_units = MAX_UNITS;
    for (uint8_t ri = 0; ri < c.header.n_rules; ++ri) {
        const RuleBin& r = c.rules[ri];
        if (r.domain == DOMAIN_EACH_OWN_UNIT) {
            if (r.action != ACTION_UNIT_REWRITE) continue;
            uint64_t mask = 0;
            for (int u = 0; u < n_units; ++u)
                if (guard_matches(r, obs, 0, r.n_guard, u)) mask |= (uint64_t(1) << u);
            d.rewrite_masks[ri] = mask;
            if (mask) ++d.activations;
            continue;
        }

        const bool enter = guard_matches(r, obs, 0, r.n_guard, -1);
        if (r.action == ACTION_COMMITTED_ROUTE) {
            const uint8_t bit = uint8_t(1u << ri);
            if (enter) state.committed_mask |= bit;
            if ((state.committed_mask & bit) && guard_matches(r, obs, r.n_guard, r.n_exit, -1))
                state.committed_mask &= uint8_t(~bit);
            if (state.committed_mask & bit) {
                d.route_override = static_cast<uint16_t>(r.p0);
                ++d.activations;
            }
        } else if (!enter) {
            continue;
        } else if (r.action == ACTION_ROUTE_OVERRIDE) {
            d.route_override = static_cast<uint16_t>(r.p0);
            ++d.activations;
        } else if (r.action == ACTION_MARKET_INTENT) {
            insert_market_rule(c, d, ri);
            ++d.activations;
        }
    }
    return d;
}

template <class Action>
B4IM_HD void apply_post(const Candidate& c, const Decision& d, Action& a, int max_market_orders) {
    int n_units = a.n_units;
    if (n_units < 0) n_units = 0;
    if (n_units > MAX_UNITS) n_units = MAX_UNITS;
    for (int ri = 0; ri < c.header.n_rules; ++ri) {
        const RuleBin& r = c.rules[ri];
        if (r.action != ACTION_UNIT_REWRITE) continue;
        const uint64_t mask = d.rewrite_masks[ri];
        for (int u = 0; u < n_units; ++u) {
            if (!(mask & (uint64_t(1) << u))) continue;
            auto& x = a.units[u];
            if (x.op != static_cast<uint8_t>(r.p0)) continue;
            x.op = static_cast<uint8_t>(r.p1);
            x.arg = static_cast<uint8_t>(r.p2);
            x.n = static_cast<decltype(x.n)>(r.p3);
        }
    }

    int cap = max_market_orders;
    if (cap < 0) cap = 0;
    if (cap > MAX_MARKET_INTENTS) cap = MAX_MARKET_INTENTS;
    int insert = d.market_count;
    if (insert > cap) insert = cap;
    int existing = a.n_orders;
    if (existing < 0) existing = 0;
    if (existing > cap - insert) existing = cap - insert;
    for (int i = existing - 1; i >= 0; --i) a.orders[i + insert] = a.orders[i];
    for (int i = 0; i < insert; ++i) {
        const RuleBin& r = c.rules[d.market_rules[i]];
        a.orders[i].op = static_cast<uint8_t>(r.p0);
        a.orders[i].item = static_cast<uint8_t>(r.p1);
        a.orders[i].n = static_cast<decltype(a.orders[i].n)>(r.p2);
    }
    a.n_orders = existing + insert;
}

}  // namespace bonsaikg4::invariant

#undef B4IM_HD
