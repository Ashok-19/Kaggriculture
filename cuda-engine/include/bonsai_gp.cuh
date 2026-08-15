#pragma once
#include "observation.cuh"
#include "tape_io.hpp"
#include <stdint.h>

namespace kagcuda {

// GPU-cheap Bonsai subset used for genetic program search.
// A Program is a memoryless/reactive Select over ordered rules. Each rule is a
// Sequence of legal-observation conditions followed by one executable leaf.
// This is intentionally compact enough to evaluate thousands of programs in
// parallel while retaining the semantics we care about: priority, reactivity,
// causal public-state branching, and coherent executable leaves.
constexpr int GP_MAX_RULES = 6;
constexpr int GP_MAX_CONDS = 4;

enum GPFeature : uint8_t {
    GPF_ALWAYS = 0,
    GPF_STEP,
    GPF_FIRST_SHOP,
    GPF_SECOND_SHOP,
    GPF_THIRD_SHOP,
    GPF_OWN_MONEY,
    GPF_OPP_MONEY,
    GPF_PRICE,
    GPF_MARKET_INV,
    GPF_SHED,
    GPF_UNITS,
    GPF_LAND,
};

enum GPCmp : uint8_t { GPC_EQ = 0, GPC_NE, GPC_GE, GPC_LE };

enum GPLeafKind : uint8_t {
    GPL_ROUTE = 0,                  // one donor route for units + market
    GPL_HYBRID = 1,                 // donor units + independently selected donor market
    GPL_ROUTE_UNITS_MANAGER_MARKET = 2,
    GPL_MANAGER = 3,                // fully reactive units + reactive market/economy
    GPL_MANAGER_UNITS_ROUTE_MARKET = 4,
    GPL_ROUTE_UNITS_PLANNED_MARKET = 5, // coherent actuator + route-demand-aware synthesized economy
};

struct GPCond {
    uint8_t feature = GPF_ALWAYS;
    uint8_t cmp = GPC_EQ;
    uint8_t item = 0;
    uint8_t pad = 0;
    int32_t value = 0;
};

struct GPLeaf {
    uint8_t kind = GPL_ROUTE;
    uint8_t unit_route = 0;
    uint8_t market_route = 0;
    // For manager-backed leaves, selects an optional external frozen manager
    // expert. Zero preserves legacy single-manager binaries and behavior.
    uint8_t pad = 0;
};

struct GPRule {
    uint8_t n_conds = 0;
    uint8_t pad[3] = {0,0,0};
    GPCond conds[GP_MAX_CONDS];
    GPLeaf leaf;
};

// Fully executable economic-program genes.  The manager is deliberately a
// phase-structured *economic blueprint*, not a replay.  Each phase describes
// public-state targets that the reactive executor continuously works toward.
// layout categories: pasture, coop, WHEAT, CARROT, TOMATO, STRAWBERRY, MELON,
// intentionally empty.  Phase boundaries and every target are evolvable.
constexpr int GP_PHASES = 11;
struct GPManager {
    // Start steps of phases 1..10. Phase 0 starts at step 0.
    uint16_t phase_step[GP_PHASES-1] = {24,48,72,144,216,264,288,336,432,576};
    uint8_t phase_layout_seed[GP_PHASES] = {1,2,3,4,5,6,7,8,9,10,11};
    // Seeded from broad public-state milestones of the current elite economic
    // family, but these are genes, not fixed donor instructions. Evolution can
    // change every phase/category independently.
    uint8_t phase_layout_weight[GP_PHASES][8] = {
        {4,0,7,0,0,0,12,2},   // opening: pasture + WHEAT + MELON
        {4,0,7,0,0,0,12,2},
        {5,0,7,0,0,0,12,1},
        {6,0,5,0,0,2,12,0},
        {12,0,5,0,0,11,20,2},
        {12,0,7,0,0,11,16,4},
        {14,0,12,0,0,22,8,19},
        {17,0,20,0,0,30,8,25},
        {18,0,24,0,0,34,8,16},
        {18,0,24,0,0,34,0,24},
        {18,0,30,2,0,22,0,28}
    };
    uint8_t phase_target_quadrants[GP_PHASES] = {1,1,1,1,2,2,3,4,4,4,4};
    // Number of HIRE orders desired during the current day. hires_today resets
    // nightly, so this represents labor throughput correctly.
    uint8_t phase_hires_per_day[GP_PHASES] = {5,0,4,5,8,12,12,10,10,12,10};
    uint8_t phase_cow_target[GP_PHASES] = {2,2,2,3,6,6,6,6,6,6,6};
    uint8_t phase_sheep_target[GP_PHASES] = {2,2,2,2,4,6,6,8,10,12,12};
    uint8_t phase_goose_target[GP_PHASES] = {0};
    uint16_t phase_cash_reserve[GP_PHASES] = {0,0,0,0,100,200,500,700,1000,1500,1500};

    uint8_t hire_until_hour = 2;
    uint8_t max_hires_per_turn = 6;
    uint8_t max_sell_orders = 4;
    uint8_t seed_buy_cap = 32;
    uint8_t animal_buy_cap = 8;
    uint8_t feed_stock = 8;   // minimum; runtime scales to ~2 units per live animal.
    uint8_t feed_load = 10;
    uint8_t fert_load = 3;
    uint8_t drop_load = 12;
    uint8_t drop_hour = 20;
    uint8_t rebuild_step_div12 = 24;  // allow deliberate reallocation from step 288.
    uint8_t care_enabled = 1;
    uint8_t collect_fert_enabled = 1;
    uint8_t dist_penalty = 4;
    // Route-demand economic planner genes. These are used by the fixed-actuator
    // evolutionary island: inspect our own executable unit program over this
    // many future turns and ensure its resource contract can actually execute.
    uint8_t planned_horizon = 24;          // feed/fertilizer/default lookahead
    uint8_t planned_seed_horizon = 24;
    uint8_t planned_animal_horizon = 24;
    // Explicit worker PICKUP quantities are a near-term shed deadline, not a
    // full-day inventory target. A short independent horizon prevents the
    // planner from hoarding tomorrow's pickup burst for ~24 turns and filling
    // the shed overnight. This is evolvable separately from feed demand.
    uint8_t planned_pickup_horizon = 0;
    uint8_t planned_seed_buffer[N_CROPS] = {0,0,0,0,0};
    uint8_t planned_animal_buffer = 0;
    uint8_t planned_feed_buffer = 0;
    uint8_t planned_fert_buffer = 0;
    uint8_t copy_route_infrastructure = 1;
    // price floor as percentage of the simulator base price; 0 means sell freely.
    uint8_t sell_floor_pct[N_PRODUCTS] = {70,70,70,80,80,70,80,80,65};
    uint8_t sell_reserve[N_PRODUCTS] = {6,2,2,4,2,2,4,4,0};
    uint8_t sell_batch = 36;
    // task priorities; higher is stronger. Indexed by GPTaskKind below.
    uint8_t task_priority[10] = {250,248,244,242,238,232,220,190,205,170};
};

struct GPProgram {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_MAX_RULES];
    GPManager manager;
};

struct GPRouteDesc { int offset; int length; };

__device__ __forceinline__ const Action& gp_route_at(const Action* actions, GPRouteDesc d, int step) {
    const int idx = step < d.length ? step : d.length - 1;
    return actions[d.offset + idx];
}

__device__ __forceinline__ Action gp_copy_action(const Action& src, int n_units) {
    Action out = src;
    out.n_units = n_units;
    return out;
}

__device__ __forceinline__ bool gp_cmp(int lhs, int rhs, uint8_t cmp) {
    switch (cmp) {
        case GPC_EQ: return lhs == rhs;
        case GPC_NE: return lhs != rhs;
        case GPC_GE: return lhs >= rhs;
        case GPC_LE: return lhs <= rhs;
    }
    return false;
}

__device__ __forceinline__ int gp_shop(const ObservationView& obs, int i) {
    const TownView town = obs.town();
    return i >= 0 && i < town.n_shops() ? town.shop(i) : -1;
}

__device__ __forceinline__ bool gp_eval_cond(const GPCond& c, const ObservationView& obs) {
    const int p = obs.player();
    int lhs = 0;
    switch (c.feature) {
        case GPF_ALWAYS: return true;
        case GPF_STEP: lhs = obs.step(); break;
        case GPF_FIRST_SHOP: lhs = gp_shop(obs, 0); break;
        case GPF_SECOND_SHOP: lhs = gp_shop(obs, 1); break;
        case GPF_THIRD_SHOP: lhs = gp_shop(obs, 2); break;
        case GPF_OWN_MONEY: lhs = static_cast<int>(obs.farm(p).money()); break;
        case GPF_OPP_MONEY: lhs = static_cast<int>(obs.farm(1-p).money()); break;
        case GPF_PRICE: lhs = c.item < N_PRODUCTS ? obs.market().price(c.item) : 0; break;
        case GPF_MARKET_INV: lhs = c.item < N_PRODUCTS ? obs.market().inventory(c.item) : 0; break;
        case GPF_SHED: lhs = c.item < N_ITEMS ? obs.private_farm().shed(c.item) : 0; break;
        case GPF_UNITS: lhs = obs.farm(p).n_units(); break;
        case GPF_LAND: lhs = obs.farm(p).n_quadrants(); break;
        default: return false;
    }
    return gp_cmp(lhs, c.value, c.cmp);
}

__device__ __forceinline__ uint32_t gp_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int gp_phase(const GPManager& g, int step) {
    int p = 0;
    while (p + 1 < GP_PHASES && step >= static_cast<int>(g.phase_step[p])) ++p;
    return p;
}

// 0 pasture, 1 coop, 2..6 crops, 7 intentionally empty.  Shed-adjacent cells
// are intentionally NOT reserved: the elite farms use them productively while
// workers can still access the shed from those positions.
__device__ __forceinline__ int gp_desired_category(const GPManager& g, int phase, int x, int y) {
    int total = 0;
    for (int i = 0; i < 8; ++i) total += g.phase_layout_weight[phase][i];
    if (total <= 0) return 7;
    const uint32_t h = gp_mix32(static_cast<uint32_t>(x + 17*y + 131*g.phase_layout_seed[phase]));
    int r = static_cast<int>(h % static_cast<uint32_t>(total));
    for (int i = 0; i < 8; ++i) {
        if (r < g.phase_layout_weight[phase][i]) return i;
        r -= g.phase_layout_weight[phase][i];
    }
    return 7;
}

__device__ __forceinline__ bool gp_is_shed_cell(int x, int y) {
    return (x == 4 || x == 5) && (y == 4 || y == 5);
}

__device__ __forceinline__ int gp_manhattan(int x0, int y0, int x1, int y1) {
    const int dx = x0 > x1 ? x0 - x1 : x1 - x0;
    const int dy = y0 > y1 ? y0 - y1 : y1 - y0;
    return dx + dy;
}

__device__ __forceinline__ UnitAction gp_step_toward(int x, int y, int tx, int ty) {
    UnitAction a{};
    if (x < tx) a.op = OP_EAST;
    else if (x > tx) a.op = OP_WEST;
    else if (y < ty) a.op = OP_SOUTH;
    else if (y > ty) a.op = OP_NORTH;
    else a.op = OP_PASS;
    return a;
}

__device__ __forceinline__ void gp_nearest_shed(int x, int y, int& tx, int& ty) {
    const int xs[4] = {4,5,4,5};
    const int ys[4] = {4,4,5,5};
    int best = 1 << 20;
    tx = 4; ty = 4;
    for (int k = 0; k < 4; ++k) {
        const int d = gp_manhattan(x,y,xs[k],ys[k]);
        if (d < best) { best = d; tx = xs[k]; ty = ys[k]; }
    }
}

__device__ __forceinline__ int gp_inv_total(const PrivateFarmView& priv, int u) {
    int n = 0;
    for (int it = 0; it < N_ITEMS; ++it) n += priv.inventory(u,it);
    return n;
}

__device__ __forceinline__ int gp_carried_item(const ObservationView& obs, int item) {
    const PublicFarmView own = obs.farm(obs.player());
    const PrivateFarmView priv = obs.private_farm();
    int n = 0;
    for (int u = 0; u < own.n_units(); ++u) n += priv.inventory(u,item);
    return n;
}

__device__ __forceinline__ int gp_count_category(const ObservationView& obs, int category) {
    const PublicFarmView own = obs.farm(obs.player());
    int n = 0;
    for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) {
        const Tile& t = own.tile(x,y);
        if (t.kind == T_LOCKED) continue;
        if (category == 0 && t.kind == T_PASTURE) ++n;
        else if (category == 1 && t.kind == T_COOP) ++n;
        else if (category >= 2 && category <= 6 && t.kind == T_PLANT && t.what == category-2) ++n;
    }
    return n;
}

__device__ __forceinline__ int gp_count_animal(const ObservationView& obs, uint8_t animal) {
    const PublicFarmView own = obs.farm(obs.player());
    int n = 0;
    for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) {
        const Tile& t = own.tile(x,y);
        if ((t.kind == T_PASTURE || t.kind == T_COOP) && t.has_animal && t.what == animal) ++n;
    }
    return n;
}

__device__ __forceinline__ int gp_target_category_count(const GPManager& g, const ObservationView& obs, int category) {
    const PublicFarmView own = obs.farm(obs.player());
    const int phase = gp_phase(g,obs.step());
    int n = 0;
    for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) {
        if (own.tile(x,y).kind == T_LOCKED) continue;
        if (gp_desired_category(g,phase,x,y) == category) ++n;
    }
    return n;
}

__device__ __forceinline__ int gp_base_price(int item) {
    const int base[N_PRODUCTS] = {25,35,60,120,250,50,160,200,100};
    return item >= 0 && item < N_PRODUCTS ? base[item] : 1;
}

__device__ __forceinline__ int gp_land_cost(int n_quadrants) {
    if (n_quadrants <= 1) return 1000;
    if (n_quadrants == 2) return 2000;
    if (n_quadrants == 3) return 4000;
    return 1 << 29;
}

__device__ __forceinline__ void gp_push_order(Action& out, uint8_t op, uint8_t item, int n) {
    if (out.n_orders >= 10 || n <= 0) return;
    Order& o = out.orders[out.n_orders++];
    o.op = op; o.item = item; o.n = n;
}

__device__ __forceinline__ void gp_manager_market(const GPManager& g, const ObservationView& obs, Action& out) {
    out.n_orders = 0;
    const int p = obs.player();
    const PublicFarmView own = obs.farm(p);
    const PrivateFarmView priv = obs.private_farm();
    const int hour = obs.hour();
    const int step = obs.step();
    const int phase = gp_phase(g,step);

    // Reactive liquidation first. High shed pressure relaxes floors so the
    // program cannot silently destroy output through capacity overflow.
    const bool pressure = priv.shed_total() >= 90;
    const bool terminal = step >= 710;
    int sale_orders = 0;
    for (int it = 0; it < N_PRODUCTS && out.n_orders < 10 && sale_orders < g.max_sell_orders; ++it) {
        const int stock = priv.shed(it);
        int sellable = stock - static_cast<int>(g.sell_reserve[it]);
        if (terminal) sellable = stock;
        if (sellable <= 0) continue;
        const int floor = (gp_base_price(it) * static_cast<int>(g.sell_floor_pct[it])) / 100;
        if (!pressure && !terminal && obs.market().price(it) < floor) continue;
        int q = sellable;
        if (!terminal && g.sell_batch > 0 && q > g.sell_batch) q = g.sell_batch;
        const int before = out.n_orders;
        gp_push_order(out, M_SELL, static_cast<uint8_t>(it), q);
        if (out.n_orders > before) ++sale_orders;
    }

    if (terminal) return;

    // Expansion is causal and current-state-driven; no future shop data.
    if (own.n_quadrants() < g.phase_target_quadrants[phase]) {
        const int cost = gp_land_cost(own.n_quadrants());
        if (own.money() >= static_cast<double>(cost + g.phase_cash_reserve[phase]) && out.n_orders < 10)
            gp_push_order(out, M_BUY_LAND, 0, 1);
    }

    // Daily labor is a renewable resource: hires_today resets at midnight.
    if (hour <= g.hire_until_hour && own.hires_today() < g.phase_hires_per_day[phase]) {
        int want = static_cast<int>(g.phase_hires_per_day[phase]) - own.hires_today();
        if (want > g.max_hires_per_turn) want = g.max_hires_per_turn;
        for (int k = 0; k < want && out.n_orders < 10; ++k) gp_push_order(out, M_HIRE, 0, 1);
    }

    // Elite openings buy several crop families in the same market action. Buy
    // every visible phase deficit that fits, rather than serializing the farm's
    // buildout across many turns.
    for (int c = 0; c < N_CROPS && out.n_orders < 10; ++c) {
        const int target = gp_target_category_count(g, obs, c + 2);
        const int planted = gp_count_category(obs, c + 2);
        int deficit = target - planted - priv.seed(c);
        if (deficit <= 0) continue;
        if (deficit > g.seed_buy_cap) deficit = g.seed_buy_cap;
        gp_push_order(out, M_BUY_SEED, static_cast<uint8_t>(c), deficit);
    }

    // Explicit phase livestock targets are economically meaningful genes. Count
    // animals already carried by workers so transport latency cannot cause the
    // controller to repeatedly overbuy them.
    const int targets[3] = {
        static_cast<int>(g.phase_goose_target[phase]),
        static_cast<int>(g.phase_cow_target[phase]),
        static_cast<int>(g.phase_sheep_target[phase])
    };
    for (int j = 0; j < 3 && out.n_orders < 10; ++j) {
        const int animal = GOOSE + j;
        int deficit = targets[j] - gp_count_animal(obs,static_cast<uint8_t>(animal))
                      - priv.shed(animal) - gp_carried_item(obs,animal);
        if (deficit <= 0) continue;
        if (deficit > g.animal_buy_cap) deficit = g.animal_buy_cap;
        gp_push_order(out,M_BUY_ANIMAL,static_cast<uint8_t>(animal),deficit);
    }

    const int animals = gp_count_animal(obs, GOOSE) + gp_count_animal(obs, COW) + gp_count_animal(obs, SHEEP);
    const int desired_feed = animals * 2 > g.feed_stock ? animals * 2 : g.feed_stock;
    if (animals > 0 && priv.shed(WHEAT) < desired_feed && out.n_orders < 10) {
        gp_push_order(out, M_BUY_PRODUCT, WHEAT, desired_feed - priv.shed(WHEAT));
    }
}

struct GPPlannedNeeds {
    int seeds[N_CROPS] = {0,0,0,0,0};
    int animals[N_ANIMALS] = {0,0,0};
    int feed = 0;
    int wheat_pickup = 0; // executable per-worker WHEAT pickup allocation
    int fertilizer = 0;
};

// The donor actuator often requests more WHEAT than a worker will actually use
// before the next refill/day reset.  Treat route pickup quantity as a capacity,
// not a mandatory economic target: count this same worker's downstream FEED
// actions until its next WHEAT pickup or the end of the day, then clamp to the
// donor pickup capacity.  This preserves route geometry/timing while making the
// resource quantity causal and executable.
__device__ __noinline__ int gp_worker_wheat_need_after_pickup(const Action* route_actions,
                                                                 GPRouteDesc route,
                                                                 int pickup_step, int unit, int cap) {
    if (cap <= 0) return 0;
    int day_end = ((pickup_step / 24) + 1) * 24;
    if (day_end > route.length) day_end = route.length;
    int need = 0;
    for (int t = pickup_step + 1; t < day_end; ++t) {
        const Action& a = gp_route_at(route_actions,route,t);
        if (unit >= a.n_units) continue;
        const UnitAction& x = a.units[unit];
        if (x.op == OP_PICKUP && x.arg == WHEAT) break;
        if (x.op == OP_FEED) {
            ++need;
            if (need >= cap) return cap;
        }
    }
    return need;
}

__device__ __forceinline__ Action gp_copy_planned_action(const Action& src, int n_units,
                                                          const Action* route_actions,
                                                          GPRouteDesc route, int step) {
    Action out = gp_copy_action(src,n_units);
    const int lim = n_units < src.n_units ? n_units : src.n_units;
    for (int u = 0; u < lim; ++u) {
        UnitAction& x = out.units[u];
        if (x.op != OP_PICKUP || x.arg != WHEAT) continue;
        const int q = gp_worker_wheat_need_after_pickup(route_actions,route,step,u,x.n);
        if (q > 0) x.n = static_cast<int16_t>(q);
        else { x.op = OP_PASS; x.arg = 0; x.n = 1; }
    }
    return out;
}

__device__ __forceinline__ GPPlannedNeeds gp_route_needs(const GPManager& g,
                                                          const Action* route_actions,
                                                          GPRouteDesc route,
                                                          int step) {
    GPPlannedNeeds need{};
    int max_h = static_cast<int>(g.planned_horizon);
    if (static_cast<int>(g.planned_seed_horizon) > max_h) max_h = g.planned_seed_horizon;
    if (static_cast<int>(g.planned_animal_horizon) > max_h) max_h = g.planned_animal_horizon;
    if (static_cast<int>(g.planned_pickup_horizon) > max_h) max_h = g.planned_pickup_horizon;
    int end = step + max_h;
    if (end > route.length) end = route.length;
    for (int t = step; t < end; ++t) {
        const int dt=t-step;
        const Action& a = gp_route_at(route_actions,route,t);
        for (int u = 0; u < a.n_units && u < MAX_UNITS; ++u) {
            const UnitAction& x = a.units[u];
            if (x.op == OP_PLANT && x.arg < N_CROPS && dt < g.planned_seed_horizon) ++need.seeds[x.arg];
            else if (x.op == OP_PLACE && is_animal(x.arg) && dt < g.planned_animal_horizon) ++need.animals[x.arg-GOOSE];
            else if (x.op == OP_FEED && dt < g.planned_horizon) ++need.feed;
            else if (x.op == OP_PICKUP && x.arg == WHEAT && dt < g.planned_pickup_horizon)
                need.wheat_pickup += gp_worker_wheat_need_after_pickup(route_actions,route,t,u,x.n);
            else if (x.op == OP_FERTILIZE && dt < g.planned_horizon) ++need.fertilizer;
        }
    }
    return need;
}

__device__ __forceinline__ int gp_planned_reserve(const GPManager& g,
                                                   const GPPlannedNeeds& need,
                                                   const ObservationView& obs,
                                                   int item) {
    int reserve = item < N_PRODUCTS ? static_cast<int>(g.sell_reserve[item]) : 0;
    if (item == WHEAT) {
        // The fixed actuator's explicit pickup contract is stronger than a
        // global FEED count: those units must physically exist in the shed when
        // specific workers arrive. Protect whichever requirement is larger.
        int x = need.wheat_pickup;
        const int feed = need.feed + static_cast<int>(g.planned_feed_buffer);
        if (feed > x) x = feed;
        if (x > reserve) reserve = x;
    } else if (item == FERTILIZER) {
        const int x = need.fertilizer + static_cast<int>(g.planned_fert_buffer);
        if (x > reserve) reserve = x;
    }
    // Net current carried inventory against generic future consumption. Raw
    // route PICKUP quantities are not assumed to be mandatory by default; the
    // zero pickup horizon therefore reproduces the validated feed-only planner.
    reserve -= gp_carried_item(obs,item);
    return reserve > 0 ? reserve : 0;
}

__device__ __forceinline__ void gp_copy_route_infrastructure(const GPManager& g,
                                                              const Action& route_now,
                                                              Action& out) {
    if (!g.copy_route_infrastructure) return;
    for (int i=0; i<route_now.n_orders && out.n_orders<MAX_ORDERS; ++i) {
        const Order& o=route_now.orders[i];
        if (o.op==M_HIRE || o.op==M_BUY_LAND) {
            Order& z=out.orders[out.n_orders++]; z=o;
        }
    }
}

__device__ __forceinline__ void gp_planned_market(const GPManager& g,
                                                   const ObservationView& obs,
                                                   const Action* route_actions,
                                                   GPRouteDesc unit_route,
                                                   Action& out) {
    out.n_orders=0;
    const int step=obs.step();
    const PublicFarmView own=obs.farm(obs.player());
    const PrivateFarmView priv=obs.private_farm();
    const int phase=gp_phase(g,step);
    const GPPlannedNeeds need=gp_route_needs(g,route_actions,unit_route,step);
    const bool pressure=priv.shed_total()>=90;
    const bool terminal=step>=710;

    // First monetize only true surplus. The reserve is derived from the unit
    // program's upcoming FEED/FERTILIZE consumption, so capital release cannot
    // silently starve the actuator that is supposed to execute the plan.
    int sale_orders=0;
    for(int it=0;it<N_PRODUCTS && out.n_orders<MAX_ORDERS && sale_orders<g.max_sell_orders;++it){
        const int stock=priv.shed(it);
        const int reserve=terminal?0:gp_planned_reserve(g,need,obs,it);
        int sellable=stock-reserve;
        if(sellable<=0)continue;
        const int floor=(gp_base_price(it)*static_cast<int>(g.sell_floor_pct[it]))/100;
        if(!pressure && !terminal && obs.market().price(it)<floor)continue;
        if(!terminal && g.sell_batch>0 && sellable>g.sell_batch)sellable=g.sell_batch;
        gp_push_order(out,M_SELL,static_cast<uint8_t>(it),sellable);++sale_orders;
    }
    if(terminal)return;

    // Preserve only the actuator's infrastructure contract. This keeps worker
    // count/land geometry coherent without inheriting donor sell/buy economics.
    const Action& route_now=gp_route_at(route_actions,unit_route,step);
    gp_copy_route_infrastructure(g,route_now,out);

    // Land is a persistent public-state milestone, not a one-turn command. If
    // the actuator's exact BUY_LAND turn was unaffordable, keep retrying after
    // liquidation until the phase target is actually visible in public state.
    // Avoid duplicating an exact route BUY_LAND already copied this turn.
    if(own.n_quadrants()<g.phase_target_quadrants[phase] && out.n_orders<MAX_ORDERS){
        bool has_land=false;
        for(int i=0;i<out.n_orders;++i) if(out.orders[i].op==M_BUY_LAND){has_land=true;break;}
        if(!has_land) gp_push_order(out,M_BUY_LAND,0,1);
    }

    // Protect working capital for the actuator's labor/infrastructure contract.
    // When current observable cash is below the evolvable phase reserve, defer
    // discretionary seed/animal stocking; liquidation, HIRE/LAND and feed still
    // execute so the farm cannot lose workers merely because a future animal or
    // crop became visible inside the lookahead window.
    const bool protect_discretionary = own.money() < static_cast<double>(g.phase_cash_reserve[phase]);

    // Buy seeds for the actual future PLANT commands in our own executable
    // program. Current seed stock is observable and deducted from the contract.
    if(!protect_discretionary) for(int c=0;c<N_CROPS && out.n_orders<MAX_ORDERS;++c){
        int q=need.seeds[c]+static_cast<int>(g.planned_seed_buffer[c])-priv.seed(c);
        if(q<=0)continue;
        if(q>g.seed_buy_cap)q=g.seed_buy_cap;
        gp_push_order(out,M_BUY_SEED,static_cast<uint8_t>(c),q);
    }

    // Likewise purchase only animals that upcoming PLACE commands can use.
    if(!protect_discretionary) for(int j=0;j<N_ANIMALS && out.n_orders<MAX_ORDERS;++j){
        const int item=GOOSE+j;
        int supply=priv.shed(item)+gp_carried_item(obs,item);
        int q=need.animals[j]+static_cast<int>(g.planned_animal_buffer)-supply;
        if(q<=0)continue;
        if(q>g.animal_buy_cap)q=g.animal_buy_cap;
        gp_push_order(out,M_BUY_ANIMAL,static_cast<uint8_t>(item),q);
    }

    // Top up WHEAT against the executable program's future FEED demand. An
    // optional short pickup horizon can raise the contract during experiments,
    // but carried WHEAT remains valid generic feed supply in the baseline.
    if(out.n_orders<MAX_ORDERS){
        int required=need.wheat_pickup;
        const int feed=need.feed+static_cast<int>(g.planned_feed_buffer);
        if(feed>required)required=feed;
        const int supply=priv.shed(WHEAT)+gp_carried_item(obs,WHEAT);
        int q=required-supply;
        if(q>0)gp_push_order(out,M_BUY_PRODUCT,WHEAT,q);
    }
}

enum GPTaskKind : uint8_t {
    GPT_DEPOSIT = 0,
    GPT_FEED,
    GPT_HARVEST,
    GPT_WATER,
    GPT_PLACE_ANIMAL,
    GPT_BUILD_OR_PLANT,
    GPT_DIG,
    GPT_CARE,
    GPT_COLLECT_FERT,
    GPT_FERTILIZE,
};

struct GPTask {
    int score = -(1 << 28);
    int x = 0, y = 0;
    uint8_t kind = GPT_BUILD_OR_PLANT;
    uint8_t arg = 0;
};

__device__ __forceinline__ void gp_consider_task(GPTask& best, const GPManager& g,
                                                  int unit, int n_units, int ordinal,
                                                  int ux, int uy, int x, int y,
                                                  uint8_t kind, uint8_t arg) {
    // Stable one-task/one-worker partition prevents every worker from piling
    // onto the same highest-priority tile while remaining completely reactive.
    // Keep ownership stable while the phase blueprint changes around it.
    if (n_units > 1 && ((ordinal + static_cast<int>(g.phase_layout_seed[0])) % n_units) != unit) return;
    const int d = gp_manhattan(ux,uy,x,y);
    const int score = static_cast<int>(g.task_priority[kind]) * 256 - d * static_cast<int>(g.dist_penalty);
    if (score > best.score) { best.score = score; best.x=x; best.y=y; best.kind=kind; best.arg=arg; }
}

__device__ __forceinline__ UnitAction gp_manager_unit(const GPManager& g, const ObservationView& obs, int u) {
    UnitAction pass{};
    const int p = obs.player();
    const PublicFarmView own = obs.farm(p);
    const PrivateFarmView priv = obs.private_farm();
    const int n_units = own.n_units();
    if (u >= n_units) return pass;
    const int ux = own.worker_x(u), uy = own.worker_y(u);

    // Inventory evacuation is per-worker and overrides assigned farm tasks.
    const int carried = gp_inv_total(priv,u);
    if (carried >= g.drop_load || (obs.hour() >= g.drop_hour && carried > 0)) {
        int sx,sy; gp_nearest_shed(ux,uy,sx,sy);
        if (gp_is_shed_cell(ux,uy)) { UnitAction a{}; a.op=OP_DROP; return a; }
        return gp_step_toward(ux,uy,sx,sy);
    }

    GPTask best;
    int ordinal = 0;
    const int phase = gp_phase(g,obs.step());
    const bool can_rebuild = obs.step() >= static_cast<int>(g.rebuild_step_div12) * 12;
    for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x, ++ordinal) {
        const Tile& t = own.tile(x,y);
        if (t.kind == T_LOCKED) continue;
        const int desired = gp_desired_category(g,phase,x,y);

        if (t.kind == T_PLANT) {
            if (t.yield_units > 0 && obs.day() - t.planted_day >= D_CROPS[t.what].first_yield_day)
                gp_consider_task(best,g,u,n_units,ordinal,ux,uy,x,y,GPT_HARVEST,t.what);
            if (!t.watered_today)
                gp_consider_task(best,g,u,n_units,ordinal+101,ux,uy,x,y,GPT_WATER,t.what);
            if (can_rebuild && desired != static_cast<int>(t.what)+2)
                gp_consider_task(best,g,u,n_units,ordinal+202,ux,uy,x,y,GPT_DIG,0);
            continue;
        }

        if (t.kind == T_PASTURE || t.kind == T_COOP) {
            if (t.has_animal) {
                if (t.yield_units > 0) gp_consider_task(best,g,u,n_units,ordinal,ux,uy,x,y,GPT_HARVEST,t.what);
                if (!t.fed_today) gp_consider_task(best,g,u,n_units,ordinal+101,ux,uy,x,y,GPT_FEED,t.what);
                if (g.care_enabled && !t.cared_today) gp_consider_task(best,g,u,n_units,ordinal+202,ux,uy,x,y,GPT_CARE,t.what);
                if (g.collect_fert_enabled && t.fertilizer_available)
                    gp_consider_task(best,g,u,n_units,ordinal+303,ux,uy,x,y,GPT_COLLECT_FERT,FERTILIZER);
            } else {
                uint8_t animal = GOOSE;
                if (t.kind == T_PASTURE) {
                    // Placement deficit is about animals physically on pasture;
                    // shed/carried supply must NOT satisfy this target. Supply is
                    // counted separately by the market controller to avoid overbuy.
                    const int cow_def = static_cast<int>(g.phase_cow_target[phase]) - gp_count_animal(obs,COW);
                    const int sheep_def = static_cast<int>(g.phase_sheep_target[phase]) - gp_count_animal(obs,SHEEP);
                    if (cow_def <= 0 && sheep_def <= 0) continue;
                    animal = static_cast<uint8_t>(sheep_def > cow_def ? SHEEP : COW);
                } else {
                    if (gp_count_animal(obs,GOOSE) >= g.phase_goose_target[phase]) continue;
                }
                gp_consider_task(best,g,u,n_units,ordinal+404,ux,uy,x,y,GPT_PLACE_ANIMAL,animal);
            }
            if (can_rebuild && ((t.kind == T_PASTURE && desired != 0) || (t.kind == T_COOP && desired != 1)))
                gp_consider_task(best,g,u,n_units,ordinal+505,ux,uy,x,y,GPT_DIG,0);
            continue;
        }

        if (t.kind == T_WEED) {
            if (desired != 7) gp_consider_task(best,g,u,n_units,ordinal+606,ux,uy,x,y,GPT_DIG,0);
            continue;
        }

        if (t.kind == T_EMPTY) {
            // For this task, arg is the desired layout category itself:
            // 0 pasture, 1 coop, 2..6 crop. Keeping the category avoids the
            // enum-value collision between ST_COOP/ST_PASTURE and WHEAT/CARROT.
            if (desired >= 0 && desired <= 6)
                gp_consider_task(best,g,u,n_units,ordinal+707,ux,uy,x,y,GPT_BUILD_OR_PLANT,static_cast<uint8_t>(desired));
        }
    }

    if (best.score <= -(1 << 27)) return pass;

    // Tasks requiring carried inputs first route the assigned worker through the
    // shed. The assignment remains stable next tick, so this forms an executable
    // multi-turn leaf without a replay trajectory.
    int need_item = -1, need_n = 1;
    if (best.kind == GPT_FEED) { need_item = WHEAT; need_n = g.feed_load; }
    else if (best.kind == GPT_PLACE_ANIMAL) { need_item = best.arg; need_n = 1; }
    else if (best.kind == GPT_FERTILIZE) { need_item = FERTILIZER; need_n = g.fert_load; }
    if (need_item >= 0 && priv.inventory(u,need_item) <= 0) {
        int sx,sy; gp_nearest_shed(ux,uy,sx,sy);
        if (gp_is_shed_cell(ux,uy) && priv.shed(need_item) > 0) {
            UnitAction a{}; a.op=OP_PICKUP; a.arg=static_cast<uint8_t>(need_item);
            a.n=static_cast<int16_t>(need_n < priv.shed(need_item) ? need_n : priv.shed(need_item));
            return a;
        }
        return gp_step_toward(ux,uy,sx,sy);
    }

    if (ux != best.x || uy != best.y) return gp_step_toward(ux,uy,best.x,best.y);

    UnitAction a{};
    switch (best.kind) {
        case GPT_FEED: a.op=OP_FEED; break;
        case GPT_HARVEST: a.op=OP_HARVEST; break;
        case GPT_WATER: a.op=OP_WATER; break;
        case GPT_PLACE_ANIMAL: a.op=OP_PLACE; a.arg=best.arg; a.n=1; break;
        case GPT_DIG: a.op=OP_DIG; break;
        case GPT_CARE: a.op=OP_CARE; break;
        case GPT_COLLECT_FERT: a.op=OP_COLLECT_FERTILIZER; break;
        case GPT_FERTILIZE: a.op=OP_FERTILIZE; break;
        case GPT_BUILD_OR_PLANT: {
            if (best.arg == 0) a.op=OP_BUILD_PASTURE;
            else if (best.arg == 1) a.op=OP_BUILD_COOP;
            else if (best.arg >= 2 && best.arg <= 6) { a.op=OP_PLANT; a.arg=static_cast<uint8_t>(best.arg-2); }
            else a.op=OP_PASS;
            break;
        }
        default: a.op=OP_PASS; break;
    }
    return a;
}

__device__ __forceinline__ void gp_manager_units(const GPManager& g, const ObservationView& obs, Action& out) {
    const int n = obs.farm(obs.player()).n_units();
    out.n_units = n;
    for (int u = 0; u < n; ++u) out.units[u] = gp_manager_unit(g,obs,u);
}

__device__ __forceinline__ GPLeaf gp_select_leaf(const GPProgram& prog, const ObservationView& obs) {
    for (int r = 0; r < prog.n_rules && r < GP_MAX_RULES; ++r) {
        const GPRule& rule = prog.rules[r];
        bool ok = true;
        for (int c = 0; c < rule.n_conds && c < GP_MAX_CONDS; ++c) {
            if (!gp_eval_cond(rule.conds[c],obs)) { ok=false; break; }
        }
        if (ok) return rule.leaf;
    }
    return prog.rules[prog.n_rules ? prog.n_rules-1 : 0].leaf;
}

__device__ __forceinline__ const GPManager& gp_leaf_manager(const GPProgram& prog, const GPLeaf& leaf,
                                                               const GPManager* manager_bank, int n_managers) {
    if (manager_bank && n_managers > 0 && static_cast<int>(leaf.pad) < n_managers) return manager_bank[leaf.pad];
    return prog.manager;
}

__device__ __forceinline__ Action gp_act_with_managers(const GPProgram& prog, const ObservationView& obs,
                                                       const Action* route_actions, const GPRouteDesc* route_descs,
                                                       int n_routes, const GPManager* manager_bank, int n_managers) {
    const GPLeaf leaf = gp_select_leaf(prog,obs);
    int ur = leaf.unit_route < n_routes ? leaf.unit_route : 0;
    int mr = leaf.market_route < n_routes ? leaf.market_route : ur;
    const Action& ua = gp_route_at(route_actions,route_descs[ur],obs.step());
    const Action& ma = gp_route_at(route_actions,route_descs[mr],obs.step());
    const GPManager& manager = gp_leaf_manager(prog,leaf,manager_bank,n_managers);
    Action out{};
    out.n_units = obs.farm(obs.player()).n_units();
    switch (leaf.kind) {
        case GPL_ROUTE:
            out = gp_copy_action(ua,out.n_units);
            break;
        case GPL_HYBRID:
            out = gp_copy_action(ua,out.n_units);
            out.n_orders = ma.n_orders;
            for (int i=0;i<ma.n_orders && i<MAX_ORDERS;++i) out.orders[i]=ma.orders[i];
            break;
        case GPL_ROUTE_UNITS_MANAGER_MARKET:
            out = gp_copy_action(ua,out.n_units);
            gp_manager_market(manager,obs,out);
            break;
        case GPL_MANAGER:
            gp_manager_units(manager,obs,out);
            gp_manager_market(manager,obs,out);
            break;
        case GPL_ROUTE_UNITS_PLANNED_MARKET:
            out = gp_copy_planned_action(ua,out.n_units,route_actions,route_descs[ur],obs.step());
            gp_planned_market(manager,obs,route_actions,route_descs[ur],out);
            break;
        case GPL_MANAGER_UNITS_ROUTE_MARKET:
            gp_manager_units(manager,obs,out);
            out.n_orders = ma.n_orders;
            for (int i=0;i<ma.n_orders && i<MAX_ORDERS;++i) out.orders[i]=ma.orders[i];
            break;
        default:
            out = gp_copy_action(ua,out.n_units);
            break;
    }
    return out;
}

__device__ __forceinline__ Action gp_act(const GPProgram& prog, const ObservationView& obs,
                                         const Action* route_actions, const GPRouteDesc* route_descs,
                                         int n_routes) {
    return gp_act_with_managers(prog,obs,route_actions,route_descs,n_routes,nullptr,0);
}

} // namespace kagcuda
