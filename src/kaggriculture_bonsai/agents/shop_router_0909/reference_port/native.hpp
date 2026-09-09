// SPDX-License-Identifier: Apache-2.0
// Translation of the attributed Shop Router 0909 callable; see PROVENANCE.md.
#pragma once
#include "bonsai_kg_v4.hpp"
#include "route_pack.hpp"

#ifdef __CUDACC__
#define BKG4_VM_HD __host__ __device__ inline
#else
#define BKG4_VM_HD inline
#endif

namespace shop_router_0909 {
using namespace bonsaikg4;
// ITEMS is the shared 12-name item domain; the first PRODUCTS of them are the
// donor's PRODUCTS tuple in its own order. UNITS bounds worker slots; QUEUE is
// the per-worker same-day repair queue, which the dawn reset keeps under a day.
constexpr int ITEMS = 12, PRODUCTS = 9, UNITS = 40, TAPE_UNITS = 13, QUEUE = 32;
constexpr int STEPS = 719, ROUTES = 13, LAST_STEP = 718, SHED_CAPACITY = 100, MAX_ORDERS = 10;
constexpr int TURNS_PER_DAY = 24;

struct Unit { int op = U_PASS, arg = 0, n = 1, arity = 1; };
struct Order { int op = 0, item = 0, n = 0; };
struct Worker {
    int x = 0, y = 0, weed = 0, inv_count = 0;
    int keys[ITEMS]{};   // inventory insertion order, as the donor iterates it
    int inv[ITEMS]{};
};
struct Observation {
    int step = 0, player = 0, board = 10, n_positions = 1, shop0 = -1, shop1 = -1;
    int shed[ITEMS]{}, shed_present[ITEMS]{}, prices[ITEMS]{};
    Worker workers[UNITS]{};
    BKG4_VM_HD int feature(int id, int a0 = 0, int a1 = 0) const {
        (void)a1;
        if (id == F_STEP) return step;
        if (id == F_SHOP_AT) return a0 == 0 ? shop0 : (a0 == 1 ? shop1 : -1);
        return 0;
    }
};
struct Action { int n_units = 0, n_market = 0; Unit units[TAPE_UNITS]{}; Order market[MAX_ORDERS]{}; };
struct State {
    // Two blackboard slots: the parent uses `route` only, the reinforced
    // candidate adds `endgame`. The policy header says how many are live.
    int initialized = 0, last_step = -1, day = -1, bb[2]{}, due = -1;
    int advanced[ITEMS]{};
    RuntimeMeta4 meta{};
    int counts[UNITS]{}, heads[UNITS]{};
    Unit queue[UNITS][QUEUE]{};
};
struct Result {
    Action action{};
    int last_step = -1, route = 0, day = -1, due = -1, endgame = 0, fault = 0;
    int advanced[ITEMS]{}, queue_counts[UNITS]{};
    uint64_t queue_hash = 0;
};
struct PackView { const bonsaikg_v6::PackedAction* records; const uint16_t* arity; };

BKG4_VM_HD int low(int a, int b) { return a < b ? a : b; }
BKG4_VM_HD int high(int a, int b) { return a > b ? a : b; }
BKG4_VM_HD bool beside_shed(const Observation& o, int i) {
    const int c = o.board / 2, x = o.workers[i].x, y = o.workers[i].y;
    return (x == c - 1 || x == c) && (y == c - 1 || y == c);
}
BKG4_VM_HD bool weed_blocked(int op) { return op == U_PLANT || op == U_BUILD_COOP || op == U_BUILD_PASTURE; }
BKG4_VM_HD Unit unpack(bonsaikg_v6::PackedUnit u, bool short_form) {
    Unit out{u.op, u.arg, u.n, 1};
    if (u.op == U_PLANT) out.arity = 2;
    if (u.op == U_PICKUP || u.op == U_PLACE) out.arity = short_form ? 2 : 3;
    return out;
}
// The donor keeps the tape's own hand count, never resizing it to the farm.
BKG4_VM_HD Action tape(PackView pack, int route, int step) {
    Action out{};
    if (step < 0 || step >= STEPS) return out;
    const int index = route * STEPS + step;
    const auto& a = pack.records[index];
    const uint16_t short_form = pack.arity[index];
    out.n_units = 1 + a.saved_hands;
    out.units[0] = unpack(a.farmer, short_form & 1);
    for (int i = 1; i < out.n_units; ++i) out.units[i] = unpack(a.hands[i - 1], short_form & (1 << i));
    out.n_market = a.n_market;
    for (int i = 0; i < a.n_market; ++i) out.market[i] = {a.market[i].op, a.market[i].item, a.market[i].n};
    return out;
}
BKG4_VM_HD void repair_weeds(Action& a, const Observation& o, State& s) {
    const int day = o.step / TURNS_PER_DAY;
    if (day != s.day) {
        s.day = day;
        for (int i = 0; i < UNITS; ++i) { s.counts[i] = 0; s.heads[i] = 0; }
    }
    const int n = low(a.n_units, o.n_positions);
    for (int i = 0; i < n; ++i) {
        if (s.counts[i] >= QUEUE) { s.meta.fault = 100; return; }
        s.queue[i][(s.heads[i] + s.counts[i]) % QUEUE] = a.units[i];
        ++s.counts[i];
        const Unit& head = s.queue[i][s.heads[i]];
        if (weed_blocked(head.op) && o.workers[i].weed) {
            a.units[i] = Unit{U_DIG, 0, 1, 1};
        } else {
            a.units[i] = head;
            s.heads[i] = (s.heads[i] + 1) % QUEUE;
            --s.counts[i];
        }
    }
}
// Lightweight stock estimate, in the donor's worker and inventory order.
BKG4_VM_HD void project(const Action& a, const Observation& o, int* stock) {
    int total = 0;
    for (int k = 0; k < ITEMS; ++k) { stock[k] = o.shed[k]; total += stock[k]; }
    const int n = low(a.n_units, o.n_positions);
    for (int i = 0; i < n; ++i) {
        if (!beside_shed(o, i)) continue;
        const Unit u = a.units[i];
        const Worker& w = o.workers[i];
        if (u.op == U_PICKUP && (u.arg < PRODUCTS || o.shed_present[u.arg])) {
            const int quantity = high(0, u.arity >= 3 ? u.n : 1);
            const int taken = low(stock[u.arg], quantity);
            stock[u.arg] -= taken; total -= taken;
        } else if (u.op == U_DROP) {
            for (int j = 0; j < w.inv_count; ++j) {
                const int k = w.keys[j];
                const int added = low(high(0, w.inv[k]), high(0, SHED_CAPACITY - total));
                if (added > 0) { stock[k] += added; total += added; }
            }
        } else if (u.op == U_PLACE && u.arg < PRODUCTS) {  // PRODUCTS..ITEMS are the animals
            const int quantity = high(0, u.arity >= 3 ? u.n : 1);
            const int added = low(low(quantity, high(0, w.inv[u.arg])), high(0, SHED_CAPACITY - total));
            if (added > 0) { stock[u.arg] += added; total += added; }
        }
    }
}
BKG4_VM_HD void subtract_advanced(Action& a, State& s, int step) {
    if (s.due == step) {
        int remaining[ITEMS];
        for (int k = 0; k < ITEMS; ++k) remaining[k] = s.advanced[k];
        for (int i = 0; i < a.n_market; ++i) {
            Order& ord = a.market[i];
            if (ord.op != M_SELL) continue;
            const int removed = low(high(0, ord.n), remaining[ord.item]);
            if (removed > 0) { ord.n -= removed; remaining[ord.item] -= removed; }
        }
    }
    for (int k = 0; k < ITEMS; ++k) s.advanced[k] = 0;
    s.due = -1;
}
BKG4_VM_HD void advance_sales(Action& a, const Observation& o, State& s, PackView pack, int route) {
    const int next_step = o.step + 1;
    if (next_step > LAST_STEP || next_step % 72 == 0 || o.step % 4 == 0) return;
    const Action future = tape(pack, route, next_step);
    int planned[ITEMS]{};
    bool already[ITEMS]{};
    for (int i = 0; i < future.n_market; ++i)
        if (future.market[i].op == M_SELL && future.market[i].item < PRODUCTS)
            planned[future.market[i].item] += high(0, future.market[i].n);
    for (int i = 0; i < a.n_market; ++i)
        if (a.market[i].op == M_SELL) already[a.market[i].item] = true;
    int stock[ITEMS];
    project(a, o, stock);
    bool any = false;
    for (int k = 0; k < PRODUCTS; ++k) {
        if (k == 0 || k == 8 || already[k]) continue;  // WHEAT and FERTILIZER are never led
        const int quantity = low(stock[k], planned[k]);
        if (quantity <= 0 || o.prices[k] < 2) continue;
        if (a.n_market >= MAX_ORDERS) break;
        a.market[a.n_market++] = {M_SELL, k, quantity};
        s.advanced[k] = quantity;
        any = true;
    }
    if (any) s.due = next_step;
}
BKG4_VM_HD Action liquidate(const Observation& o) {
    Action a{};
    a.n_units = o.n_positions;
    for (int i = 0; i < o.n_positions && i < TAPE_UNITS; ++i)
        if (beside_shed(o, i) && o.workers[i].inv_count) a.units[i] = Unit{U_DROP, 0, 1, 1};
    int stock[ITEMS];
    project(a, o, stock);
    for (int k = 0; k < PRODUCTS; ++k)
        if (stock[k] > 0) a.market[a.n_market++] = {M_SELL, k, stock[k]};
    // Stable descending price*quantity, retaining PRODUCTS order on ties.
    for (int i = 1; i < a.n_market; ++i) {
        const Order v = a.market[i];
        const int64_t value = static_cast<int64_t>(o.prices[v.item]) * v.n;
        int j = i;
        while (j > 0 && static_cast<int64_t>(o.prices[a.market[j - 1].item]) * a.market[j - 1].n < value) {
            a.market[j] = a.market[j - 1]; --j;
        }
        a.market[j] = v;
    }
    return a;
}
// Bounded terminal repair: top every product sell up to the projected shed.
// Mirrors the Python gate exactly, including the single pre-repair projection.
constexpr int FULL_SELL = 4;
BKG4_VM_HD void full_sell(Action& a, const Observation& o) {
    for (int i = 0; i < a.n_market; ++i)
        if (a.market[i].op != 0 && a.market[i].op != M_SELL) return;
    int stock[ITEMS];
    project(a, o, stock);
    for (int k = 0; k < PRODUCTS; ++k) {
        if (k == 0 || k == 8 || stock[k] <= 0 || o.prices[k] < 2) continue;
        bool found = false;
        for (int i = 0; i < a.n_market; ++i)
            if (a.market[i].op == M_SELL && a.market[i].item == k) { a.market[i].n = stock[k]; found = true; }
        if (!found && a.n_market < MAX_ORDERS) a.market[a.n_market++] = {M_SELL, k, stock[k]};
    }
}
BKG4_VM_HD uint64_t hash_word(uint64_t h, int v) {
    uint32_t u = static_cast<uint32_t>(v);
    for (int j = 0; j < 4; ++j) { h ^= (u >> (8 * j)) & 255; h *= 1099511628211ULL; }
    return h;
}
BKG4_VM_HD uint64_t queue_hash(const State& s) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < UNITS; ++i) {
        h = hash_word(h, s.counts[i]);
        for (int j = 0; j < s.counts[i]; ++j) {
            const Unit& u = s.queue[i][(s.heads[i] + j) % QUEUE];
            h = hash_word(h, u.op); h = hash_word(h, u.arg); h = hash_word(h, u.n); h = hash_word(h, u.arity);
        }
    }
    return h;
}
BKG4_VM_HD void tick(const PolicyView4& policy, PackView pack, const Observation& o, State& s, Result& out) {
    RuntimeView4 runtime{s.bb, nullptr, &s.meta};
    if (!s.initialized || o.step <= s.last_step) {
        reset_runtime(policy, runtime);
        s.initialized = 1; s.day = -1; s.due = -1;
        for (int k = 0; k < ITEMS; ++k) s.advanced[k] = 0;
        for (int i = 0; i < UNITS; ++i) { s.counts[i] = 0; s.heads[i] = 0; }
    }
    s.last_step = o.step;
    ActionResult4 ignored{};
    tick4(policy, runtime, o, 0, ignored);
    const int route = s.bb[0];
    const int endgame = policy.n_bb > 1 ? s.bb[1] : 0;
    Action a = tape(pack, route, o.step);
    repair_weeds(a, o, s);
    subtract_advanced(a, s, o.step);
    advance_sales(a, o, s, pack, route);
    if (endgame & FULL_SELL) full_sell(a, o);
    if (a.n_market > MAX_ORDERS) a.n_market = MAX_ORDERS;
    if (o.step == LAST_STEP) a = liquidate(o);
    out.action = a;
    out.last_step = s.last_step; out.route = route; out.day = s.day; out.due = s.due;
    out.endgame = endgame;
    for (int k = 0; k < ITEMS; ++k) out.advanced[k] = s.advanced[k];
    for (int i = 0; i < UNITS; ++i) out.queue_counts[i] = s.counts[i];
    out.fault = s.meta.fault;
    out.queue_hash = queue_hash(s);
}
}  // namespace shop_router_0909
#undef BKG4_VM_HD
