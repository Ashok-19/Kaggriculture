// Kaggriculture simulator — faithful C++ port of kaggriculture.py.
//
// Baseline: kaggle-environments 1.32.7, Kaggle commit
// 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c (PR #1399).
// Design goals, in order: (1) bit-identical results to the Python interpreter,
// (2) zero heap allocation per step, (3) trivially copyable state so search can
// snapshot and fork episodes.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "pyrandom.hpp"

namespace kag {

// ---------------------------------------------------------------- item space
// PRODUCTS order matches kaggriculture.py exactly. The first five products are
// also the five crops, in CROPS order, so a crop id doubles as a product id.
enum Item : uint8_t {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER,
    GOOSE, COW, SHEEP, N_ITEMS
};
constexpr int N_PRODUCTS = 9;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
inline bool is_crop(uint8_t i) { return i < N_CROPS; }
inline bool is_product(uint8_t i) { return i < N_PRODUCTS; }
inline bool is_animal(uint8_t i) { return i >= GOOSE && i < N_ITEMS; }

// ---------------------------------------------------------------- unit ops
enum Op : uint8_t {
    OP_PASS = 0, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST,
    OP_PICKUP, OP_DROP, OP_PLACE,
    OP_PLANT, OP_WATER, OP_HARVEST, OP_FERTILIZE, OP_DIG,
    OP_BUILD_COOP, OP_BUILD_PASTURE,
    OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE, OP_INVALID
};

enum MOp : uint8_t {
    M_NONE = 0, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL
};

// ---------------------------------------------------------------- static data
struct CropDef { int seed, first_yield_day, max_yield_day, interval, max_yield; bool ongoing; };
inline constexpr CropDef CROPS[N_CROPS] = {
    {  10, 2,  4, 0, 6, false },  // WHEAT
    {  20, 2,  3, 0, 4, false },  // CARROT
    {  50, 8,  8, 1, 4, true  },  // TOMATO
    { 100, 10, 10, 2, 4, true },  // STRAWBERRY
    {  80, 10, 12, 0, 6, false }, // MELON
};

enum Structure : uint8_t { ST_COOP = 0, ST_PASTURE = 1 };
struct AnimalDef { int cost; Structure structure; int first_yield_day, interval, max_held; Item product; };
inline constexpr AnimalDef ANIMALS[N_ANIMALS] = {
    { 300, ST_COOP,    4, 1, 4, EGG  },  // GOOSE
    { 400, ST_PASTURE, 8, 2, 6, MILK },  // COW
    { 500, ST_PASTURE, 6, 3, 6, WOOL },  // SHEEP
};

// 1.32.7 adds the HINGE scarcity curve for CARROT/TOMATO/EGG below I0.
enum Shape : uint8_t { F_LINEAR, F_SQ, F_SQRT, F_LOG, F_LOG10, F_HINGE };
constexpr double HINGE_GAIN = 8.0;
struct MarketDef { double base; int I0; double T; Shape below_f; double below_t; Shape above_f; double above_t; };
inline constexpr MarketDef MARKET[N_PRODUCTS] = {
    {  25, 10000, 400, F_SQRT,   0.80, F_LOG,    0.20 },  // WHEAT
    {  35, 10000, 450, F_HINGE,  1.00, F_SQRT,   0.70 },  // CARROT
    {  60, 10000, 200, F_HINGE,  0.40, F_SQRT,   0.60 },  // TOMATO
    { 120, 10000, 100, F_SQRT,   0.70, F_LINEAR, 1.60 },  // STRAWBERRY
    { 250, 10000, 300, F_LOG,    0.20, F_SQ,     3.60 },  // MELON
    {  50, 10000, 332, F_HINGE,  0.40, F_LOG,    0.20 },  // EGG
    { 160, 10000, 122, F_SQRT,   0.60, F_LINEAR, 1.60 },  // MILK
    { 200, 10000, 105, F_LOG,    0.20, F_SQ,     3.20 },  // WOOL
    { 100, 10000, 200, F_LINEAR, 0.40, F_LINEAR, 0.40 },  // FERTILIZER
};

inline double shape(Shape f, double x, double T = 0.0) {
    if (x < 0.0) x = 0.0;
    switch (f) {
        case F_LINEAR: return x;
        case F_SQ:     return x * x;
        case F_SQRT:   return std::sqrt(x);
        case F_LOG:    return std::log(1.0 + x);
        case F_LOG10:  return std::log10(1.0 + x);
        case F_HINGE: {
            if (T <= 0.0) return x;
            double u = x / T;
            double d = std::max(0.0, u - 1.0);
            return u + HINGE_GAIN * d * d;
        }
    }
    return x;
}

// Amplitudes are derived constants; precompute once.
struct MarketAmp { double below, above; };
inline const std::array<MarketAmp, N_PRODUCTS>& market_amps() {
    static const std::array<MarketAmp, N_PRODUCTS> a = [] {
        std::array<MarketAmp, N_PRODUCTS> r{};
        for (int i = 0; i < N_PRODUCTS; ++i) {
            r[i].below = MARKET[i].below_t * MARKET[i].base /
                         shape(MARKET[i].below_f, MARKET[i].T, MARKET[i].T);
            r[i].above = MARKET[i].above_t * MARKET[i].base /
                         shape(MARKET[i].above_f, MARKET[i].T, MARKET[i].T);
        }
        return r;
    }();
    return a;
}

// Python: max(PRICE_FLOOR, int(round(price))). Python's round() on a float is
// round-half-to-even, which is exactly nearbyint's default mode.
inline int market_price_formula(int item, int inv) {
    const MarketDef& p = MARKET[item];
    const MarketAmp& a = market_amps()[item];
    double price;
    if (inv < p.I0) {
        price = p.base + a.below * shape(p.below_f, static_cast<double>(p.I0 - inv), p.T);
    } else {
        price = p.base - a.above * shape(p.above_f, static_cast<double>(inv - p.I0), p.T);
    }
    int v = static_cast<int>(std::nearbyint(price));
    return v < 1 ? 1 : v;
}

// Market orders price one unit at a time, making the same inventory states hot
// across large simulation batches. Cache a broad practical range exactly once;
// unusual states fall back to the identical formula above. The cache changes no
// numerical semantics and allocates nothing per step.
constexpr int PRICE_CACHE_MIN = -20000;
constexpr int PRICE_CACHE_MAX = 50000;
constexpr int PRICE_CACHE_SPAN = PRICE_CACHE_MAX - PRICE_CACHE_MIN + 1;
inline const std::vector<int>& market_price_cache() {
    static const std::vector<int> cache = [] {
        std::vector<int> out(static_cast<size_t>(N_PRODUCTS) * PRICE_CACHE_SPAN);
        for (int item = 0; item < N_PRODUCTS; ++item)
            for (int inv = PRICE_CACHE_MIN; inv <= PRICE_CACHE_MAX; ++inv)
                out[static_cast<size_t>(item) * PRICE_CACHE_SPAN + (inv - PRICE_CACHE_MIN)] =
                    market_price_formula(item, inv);
        return out;
    }();
    return cache;
}

inline const std::vector<int64_t>& market_price_prefix() {
    static const std::vector<int64_t> prefix = [] {
        std::vector<int64_t> out(static_cast<size_t>(N_PRODUCTS) * (PRICE_CACHE_SPAN + 1), 0);
        const auto& prices = market_price_cache();
        for (int item = 0; item < N_PRODUCTS; ++item) {
            const size_t pbase = static_cast<size_t>(item) * PRICE_CACHE_SPAN;
            const size_t sbase = static_cast<size_t>(item) * (PRICE_CACHE_SPAN + 1);
            for (int k = 0; k < PRICE_CACHE_SPAN; ++k)
                out[sbase + k + 1] = out[sbase + k] + prices[pbase + k];
        }
        return out;
    }();
    return prefix;
}

inline bool cached_market_price_sum(int item, int lo_inv, int hi_inv, int64_t& out) {
    if (lo_inv > hi_inv) { out = 0; return true; }
    if (item < 0 || item >= N_PRODUCTS || lo_inv < PRICE_CACHE_MIN || hi_inv > PRICE_CACHE_MAX)
        return false;
    const auto& prefix = market_price_prefix();
    const size_t base = static_cast<size_t>(item) * (PRICE_CACHE_SPAN + 1);
    const int lo = lo_inv - PRICE_CACHE_MIN;
    const int hi = hi_inv - PRICE_CACHE_MIN;
    out = prefix[base + hi + 1] - prefix[base + lo];
    return true;
}

inline int market_price(int item, int inv) {
    if (inv >= PRICE_CACHE_MIN && inv <= PRICE_CACHE_MAX)
        return market_price_cache()[static_cast<size_t>(item) * PRICE_CACHE_SPAN +
                                    (inv - PRICE_CACHE_MIN)];
    return market_price_formula(item, inv);
}

// ---------------------------------------------------------------- shops / town
enum ShopId : uint8_t {
    SHOP_BAKERY = 0, SHOP_BRUNCH_SPOT, SHOP_FARMERS_MARKET, SHOP_ICE_CREAM_SHOP,
    SHOP_PET_CAFE, SHOP_PIZZA_SHOP, SHOP_SMOOTHIE_SHOP, SHOP_YARN_STORE, N_SHOPS
};
// NOTE: this array is in sorted(SHOPS) order, because the environment unlocks
// with `rng.choice(sorted(SHOPS))`.
inline constexpr uint16_t SHOP_MASK[N_SHOPS] = {
    (1u << EGG) | (1u << WHEAT),                                         // BAKERY
    (1u << EGG) | (1u << WHEAT) | (1u << STRAWBERRY),                    // BRUNCH_SPOT
    (1u << WHEAT) | (1u << CARROT) | (1u << TOMATO) | (1u << STRAWBERRY),// FARMERS_MARKET
    (1u << STRAWBERRY) | (1u << MILK) | (1u << WHEAT),                   // ICE_CREAM_SHOP
    (1u << CARROT),                                                       // PET_CAFE
    (1u << MILK) | (1u << TOMATO) | (1u << WHEAT),                       // PIZZA_SHOP
    (1u << STRAWBERRY) | (1u << MILK),                                   // SMOOTHIE_SHOP
    (1u << WOOL),                                                         // YARN_STORE
};
inline constexpr int SHOP_MULT[N_SHOPS] = { 1, 1, 1, 1, 2, 1, 1, 2 };
constexpr int MAX_SHOP_INSTANCES = 8;

// ---------------------------------------------------------------- config
struct Config {
    int episode_steps = 720;
    int board_size = 10;
    int starting_money = 3000;
    int max_orders = 10;
    int turns_per_day = 24;
    int shed_capacity = 100;
    double weed_chance = 0.005;
    int shop_unlock_interval = 3;
    int shop_sell_interval = 4;
    int center_sell_interval = 24;
    int hire_mult = 1;
    uint64_t seed = 0;
    bool cache_daily_rng_seed_state = false;  // internal snapshot-search acceleration
};

// ---------------------------------------------------------------- state
enum TileKind : uint8_t { T_EMPTY = 0, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT };

struct Tile {
    TileKind kind = T_EMPTY;
    uint8_t  what = 0;
    bool     has_animal = false;
    bool     watered_today = false;
    bool     fed_today = false;
    bool     cared_today = false;
    bool     fertilizer_available = false;
    int8_t   consecutive_dry = 0;
    int8_t   yield_units = 0;
    int8_t   pending_care_bonus = 0;
    int16_t  planted_day = 0;
    int32_t  max_lifespan_step = -1;
    int16_t  fertilized_until_day = -1;
};

constexpr int MAX_UNITS = 40;
constexpr int BOARD = 10;
constexpr int NO_DECAY_STEP = 1 << 30;

struct Farm {
    double money = 0;
    Tile tiles[BOARD][BOARD];
    int8_t pos_x[MAX_UNITS], pos_y[MAX_UNITS];
    int n_units = 1;
    int n_quadrants = 1;
    int hires_today = 0;
    int next_decay_step = NO_DECAY_STEP;

    int16_t shed[N_ITEMS] = {0};
    int shed_total = 0;
    int16_t seeds[N_CROPS] = {0};
    int16_t inv[MAX_UNITS][N_ITEMS] = {{0}};
    uint8_t inv_keys[MAX_UNITS][N_ITEMS] = {{0}};
    uint8_t inv_nkeys[MAX_UNITS] = {0};

    int32_t discarded[N_ITEMS] = {0};
    int32_t produced[N_ITEMS] = {0};
    int32_t sold_units[N_ITEMS] = {0};
    double  sell_revenue = 0;
    double  total_spend = 0;

    void inv_add(int u, int item, int n) {
        if (n <= 0) return;
        if (inv[u][item] == 0) inv_keys[u][inv_nkeys[u]++] = static_cast<uint8_t>(item);
        inv[u][item] += static_cast<int16_t>(n);
    }
    void inv_erase(int u, int item) {
        inv[u][item] = 0;
        int k = 0;
        while (k < inv_nkeys[u] && inv_keys[u][k] != item) ++k;
        if (k == inv_nkeys[u]) return;
        for (int j = k; j + 1 < inv_nkeys[u]; ++j) inv_keys[u][j] = inv_keys[u][j + 1];
        inv_nkeys[u]--;
    }
    bool inv_take(int u, int item, int n) {
        if (inv[u][item] < n) return false;
        inv[u][item] -= static_cast<int16_t>(n);
        if (inv[u][item] == 0) inv_erase(u, item);
        return true;
    }
    void inv_clear(int u) {
        for (int i = 0; i < N_ITEMS; ++i) inv[u][i] = 0;
        inv_nkeys[u] = 0;
    }
};

struct Market {
    int32_t inventory[N_PRODUCTS];
    int32_t prices[N_PRODUCTS];
};

struct State {
    Farm farms[2];
    Market market;
    uint8_t shops[MAX_SHOP_INSTANCES];
    int n_shops = 0;
    int step = 0;
    int day = 0;
    int hour = 0;
    bool done = false;
};

// ---------------------------------------------------------------- actions
struct UnitAction { uint8_t op = OP_PASS; uint8_t arg = 0; int16_t n = 1; };
struct Order { uint8_t op = M_NONE; uint8_t item = 0; int32_t n = 0; };

struct Action {
    UnitAction units[MAX_UNITS];
    int n_units = 1;
    Order orders[16];
    int n_orders = 0;
    void clear() { n_units = 1; units[0] = UnitAction{}; n_orders = 0; }
};

// ---------------------------------------------------------------- helpers
inline int quadrant_of(int x, int y, int bs) {
    int half = bs / 2;
    return (y < half ? 0 : 2) + (x < half ? 0 : 1);
}
inline constexpr int LAND_ORDER[3] = { 1, 2, 3 };
inline constexpr int LAND_PRICES[3] = { 1000, 2000, 4000 };

inline void shed_access_tiles(int bs, int out[4][2]) {
    int h = bs / 2;
    out[0][0] = h - 1; out[0][1] = h - 1;
    out[1][0] = h;     out[1][1] = h - 1;
    out[2][0] = h - 1; out[2][1] = h;
    out[3][0] = h;     out[3][1] = h;
}
inline bool is_shed_adjacent(int x, int y, int bs) {
    int h = bs / 2;
    return (x == h - 1 || x == h) && (y == h - 1 || y == h);
}

inline int fib(int n) {
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { int t = b; b = a + b; a = t; }
    return a;
}

class Sim {
public:
    Config cfg;
    State st;

    explicit Sim(const Config& c = Config{}) : cfg(c) { reset(); }

    void reset() {
        st = State{};
        int h = cfg.board_size / 2;
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            f.money = cfg.starting_money;
            for (int y = 0; y < cfg.board_size; ++y)
                for (int x = 0; x < cfg.board_size; ++x)
                    f.tiles[y][x].kind = (quadrant_of(x, y, cfg.board_size) == 0) ? T_EMPTY : T_LOCKED;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
        }
        for (int i = 0; i < N_PRODUCTS; ++i) {
            st.market.inventory[i] = MARKET[i].I0;
            st.market.prices[i] = static_cast<int>(MARKET[i].base);
        }
    }

    double reward(int p) const { return st.farms[p].money; }

    void step(const Action& a0, const Action& a1) {
        step_with_unit_counts(a0, a0.n_units, a1, a1.n_units);
    }

    // Replay wrappers pad/truncate hand actions to each farm's current unit
    // count. This overload supplies those effective counts without copying and
    // rewriting the large Action structs every turn.
    void step_with_unit_counts(const Action& a0, int n_units0,
                               const Action& a1, int n_units1) {
        if (st.done) return;
        const Action* acts[2] = { &a0, &a1 };
        const int action_units[2] = {
            std::max(0, std::min(n_units0, MAX_UNITS)),
            std::max(0, std::min(n_units1, MAX_UNITS)),
        };
        int step_i = st.step;
        int day = step_i / cfg.turns_per_day;

        for (int p = 0; p < 2; ++p) apply_unit_actions(p, *acts[p], day, action_units[p]);
        uint16_t market_dirty = process_market(*acts[0], *acts[1]);
        market_dirty |= town_consume(step_i);
        refresh_prices(market_dirty);
        for (int p = 0; p < 2; ++p) decay_plants(st.farms[p], step_i);
        if ((step_i + 1) % cfg.turns_per_day == 0) end_of_day(day);

        st.step = step_i + 1;
        st.day = st.step / cfg.turns_per_day;
        st.hour = st.step % cfg.turns_per_day;
        if (step_i >= cfg.episode_steps - 2) st.done = true;
    }

private:
    void apply_unit_actions(int p, const Action& a, int day, int action_units) {
        Farm& f = st.farms[p];
        int demand[N_CROPS] = {0};
        for (int i = 0; i < action_units; ++i)
            if (a.units[i].op == OP_PLANT && a.units[i].arg < N_CROPS) demand[a.units[i].arg]++;
        int n = std::min(action_units, f.n_units);
        bool blocked[N_CROPS];
        for (int c = 0; c < N_CROPS; ++c) blocked[c] = demand[c] > f.seeds[c];

        for (int i = 0; i < n; ++i) {
            UnitAction u = a.units[i];
            if (u.op == OP_PLANT && u.arg < N_CROPS && blocked[u.arg]) continue;
            apply_unit(f, i, u, day);
        }
    }

    void apply_unit(Farm& f, int idx, const UnitAction& u, int day) {
        const int bs = cfg.board_size;
        int fx = f.pos_x[idx], fy = f.pos_y[idx];
        int16_t* inv = f.inv[idx];

        switch (u.op) {
            case OP_PASS: return;
            case OP_NORTH: case OP_SOUTH: case OP_EAST: case OP_WEST: {
                int nx = fx + (u.op == OP_EAST) - (u.op == OP_WEST);
                int ny = fy + (u.op == OP_SOUTH) - (u.op == OP_NORTH);
                if (nx < 0 || nx >= bs || ny < 0 || ny >= bs) return;
                f.pos_x[idx] = static_cast<int8_t>(nx);
                f.pos_y[idx] = static_cast<int8_t>(ny);
                return;
            }
            default: break;
        }

        Tile& tile = f.tiles[fy][fx];

        if (u.op == OP_DROP) {
            if (!is_shed_adjacent(fx, fy, bs)) return;
            const int nk = f.inv_nkeys[idx];
            for (int k = 0; k < nk; ++k) {
                const int it = f.inv_keys[idx][k];
                const int amount = inv[it];
                if (amount <= 0) continue;
                const int room = std::max(0, cfg.shed_capacity - f.shed_total);
                const int take = std::min(amount, room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += amount - take;
                inv[it] = 0;
            }
            f.inv_nkeys[idx] = 0;
            return;
        }
        if (u.op == OP_PICKUP) {
            if (!is_shed_adjacent(fx, fy, bs)) return;
            if (u.n <= 0 || u.arg >= N_ITEMS) return;
            int take = std::min<int>(u.n, f.shed[u.arg]);
            if (take <= 0) return;
            f.shed[u.arg] -= take; f.shed_total -= take;
            f.inv_add(idx, u.arg, take);
            return;
        }
        if (u.op == OP_PLACE) {
            if (u.arg >= N_ITEMS) return;
            if (is_animal(u.arg)) {
                const AnimalDef& ad = ANIMALS[u.arg - GOOSE];
                TileKind want = (ad.structure == ST_COOP) ? T_COOP : T_PASTURE;
                if (tile.kind == want && !tile.has_animal) {
                    if (f.inv_take(idx, u.arg, 1)) {
                        Tile t{};
                        t.kind = want; t.what = u.arg; t.has_animal = true;
                        t.planted_day = static_cast<int16_t>(day);
                        tile = t;
                    }
                    return;
                }
            }
            if (is_shed_adjacent(fx, fy, bs)) {
                int n = std::min<int>(u.n, inv[u.arg]);
                if (n <= 0) return;
                n = std::min(n, std::max(0, cfg.shed_capacity - f.shed_total));
                if (n <= 0) return;
                f.inv_take(idx, u.arg, n);
                f.shed[u.arg] += n; f.shed_total += n;
            }
            return;
        }

        if (tile.kind == T_LOCKED) return;

        switch (u.op) {
            case OP_PLANT: {
                if (u.arg >= N_CROPS || tile.kind != T_EMPTY || f.seeds[u.arg] <= 0) return;
                f.seeds[u.arg] -= 1;
                const CropDef& cd = CROPS[u.arg];
                Tile t{};
                t.kind = T_PLANT; t.what = u.arg;
                t.planted_day = static_cast<int16_t>(day);
                t.consecutive_dry = 1;
                t.yield_units = cd.ongoing ? 0 : 1;
                t.max_lifespan_step = cd.ongoing ? -1
                    : (day + cd.max_yield_day + 1) * cfg.turns_per_day;
                tile = t;
                if (t.max_lifespan_step >= 0)
                    f.next_decay_step = std::min(f.next_decay_step, t.max_lifespan_step);
                return;
            }
            case OP_WATER: {
                if (tile.kind != T_PLANT || tile.watered_today) return;
                tile.watered_today = true;
                const CropDef& cd = CROPS[tile.what];
                if (!cd.ongoing) {
                    int age = day - tile.planted_day;
                    int w0 = (cd.max_yield_day + 1) / 2;
                    if (age >= w0 && age <= cd.max_yield_day) {
                        int bonus = (tile.fertilized_until_day >= day) ? 2 : 1;
                        tile.yield_units = static_cast<int8_t>(std::min(cd.max_yield, tile.yield_units + bonus));
                    }
                }
                return;
            }
            case OP_HARVEST: {
                if (tile.kind == T_EMPTY || tile.kind == T_WEED) return;
                if (tile.yield_units <= 0) return;
                if (tile.kind == T_PLANT) {
                    const CropDef& cd = CROPS[tile.what];
                    if (day - tile.planted_day < cd.first_yield_day) return;
                    f.inv_add(idx, tile.what, tile.yield_units);
                    f.produced[tile.what] += tile.yield_units;
                    tile.yield_units = 0;
                    if (!cd.ongoing) { Tile e{}; e.kind = T_EMPTY; tile = e; }
                } else if (tile.has_animal) {
                    f.inv_add(idx, ANIMALS[tile.what - GOOSE].product, tile.yield_units);
                    f.produced[ANIMALS[tile.what - GOOSE].product] += tile.yield_units;
                    tile.yield_units = 0;
                }
                return;
            }
            case OP_FERTILIZE: {
                if (tile.kind != T_PLANT) return;
                if (!f.inv_take(idx, FERTILIZER, 1)) return;
                tile.fertilized_until_day = std::max<int16_t>(
                    tile.fertilized_until_day, static_cast<int16_t>(day + 2));
                return;
            }
            case OP_DIG: {
                if (tile.kind == T_EMPTY) return;
                if (tile.has_animal) return;
                Tile e{}; e.kind = T_EMPTY; tile = e;
                return;
            }
            case OP_BUILD_COOP:
                if (tile.kind != T_EMPTY) return;
                { Tile t{}; t.kind = T_COOP; tile = t; }
                return;
            case OP_BUILD_PASTURE:
                if (tile.kind != T_EMPTY) return;
                { Tile t{}; t.kind = T_PASTURE; tile = t; }
                return;
            case OP_FEED:
                if (!tile.has_animal || tile.fed_today) return;
                if (!f.inv_take(idx, WHEAT, 1)) return;
                tile.fed_today = true;
                return;
            case OP_COLLECT_FERTILIZER:
                if (!tile.has_animal || !tile.fertilizer_available) return;
                tile.fertilizer_available = false;
                f.inv_add(idx, FERTILIZER, 1);
                f.produced[FERTILIZER] += 1;
                return;
            case OP_CARE:
                if (!tile.has_animal || tile.cared_today) return;
                tile.cared_today = true;
                return;
            default: return;
        }
    }

    struct OState { uint8_t type; uint8_t item; int32_t remaining; bool live; };

    int market_affect_item(const OState& os) const {
        if (!os.live || os.remaining <= 0) return -1;
        if (os.type == M_SELL && is_product(os.item)) return os.item;
        if (os.type == M_BUY_PRODUCT && (os.item == WHEAT || os.item == FERTILIZER)) return os.item;
        return -1;
    }

    uint16_t batch_independent_order(const OState& os, Farm& f) {
        if (!os.live || os.remaining <= 0) return 0;
        const int item = os.item;
        const int requested = os.remaining;

        if (os.type == M_BUY_SEED) {
            if (!is_crop(item)) return 0;
            const int price = CROPS[item].seed;
            const int affordable = price > 0 ? static_cast<int>(f.money / price) : requested;
            const int n = std::max(0, std::min(requested, affordable));
            if (n <= 0) return 0;
            const int64_t cost = static_cast<int64_t>(n) * price;
            f.money -= static_cast<double>(cost);
            f.total_spend += static_cast<double>(cost);
            f.seeds[item] += static_cast<int16_t>(n);
            return 0;
        }

        if (os.type == M_BUY_ANIMAL) {
            if (!is_animal(item)) return 0;
            const int price = ANIMALS[item - GOOSE].cost;
            const int room = std::max(0, cfg.shed_capacity - f.shed_total);
            const int affordable = price > 0 ? static_cast<int>(f.money / price) : requested;
            const int n = std::max(0, std::min({requested, room, affordable}));
            if (n <= 0) return 0;
            const int64_t cost = static_cast<int64_t>(n) * price;
            f.money -= static_cast<double>(cost);
            f.total_spend += static_cast<double>(cost);
            f.shed[item] += static_cast<int16_t>(n);
            f.shed_total += n;
            return 0;
        }

        if (os.type == M_SELL) {
            if (!is_product(item)) return 0;
            const int n = std::max(0, std::min<int>(requested, f.shed[item]));
            if (n <= 0) return 0;
            const int inv0 = st.market.inventory[item];
            int moving = 0;
            if (market_price(item, inv0) > 1) {
                if (market_price(item, inv0 + n - 1) > 1) {
                    moving = n;
                } else {
                    int lo = 0, hi = n;
                    while (lo < hi) {
                        const int mid = lo + (hi - lo + 1) / 2;
                        if (market_price(item, inv0 + mid - 1) > 1) lo = mid;
                        else hi = mid - 1;
                    }
                    moving = lo;
                }
            }
            int64_t revenue = 0;
            if (moving > 0 && !cached_market_price_sum(item, inv0, inv0 + moving - 1, revenue)) {
                // Rare range outside the cache: preserve exact unit semantics.
                revenue = 0;
                moving = 0;
                int inv = inv0;
                for (int k = 0; k < n; ++k) {
                    const int price = market_price(item, inv);
                    revenue += price;
                    if (price > 1) { ++inv; ++moving; }
                }
            } else {
                revenue += static_cast<int64_t>(n - moving);  // floor-price sales stay at inventory 1.
            }
            f.shed[item] -= static_cast<int16_t>(n);
            f.shed_total -= n;
            f.money += static_cast<double>(revenue);
            f.sold_units[item] += n;
            f.sell_revenue += static_cast<double>(revenue);
            st.market.inventory[item] += moving;
            return moving > 0 ? static_cast<uint16_t>(1u << item) : 0;
        }

        if (os.type == M_BUY_PRODUCT) {
            if (!(item == WHEAT || item == FERTILIZER)) return 0;
            const int room = std::max(0, cfg.shed_capacity - f.shed_total);
            const int limit = std::max(0, std::min(requested, room));
            if (limit <= 0) return 0;
            const int inv0 = st.market.inventory[item];
            auto cost_for = [&](int n, int64_t& cost) {
                if (n <= 0) { cost = 0; return true; }
                return cached_market_price_sum(item, inv0 - n, inv0 - 1, cost);
            };
            int n = 0;
            int64_t cost = 0;
            int64_t full_cost = 0;
            if (cost_for(limit, full_cost)) {
                if (static_cast<double>(full_cost) <= f.money) {
                    n = limit;
                    cost = full_cost;
                } else {
                    int lo = 0, hi = limit;
                    while (lo < hi) {
                        const int mid = lo + (hi - lo + 1) / 2;
                        int64_t mid_cost = 0;
                        cost_for(mid, mid_cost);
                        if (static_cast<double>(mid_cost) <= f.money) lo = mid;
                        else hi = mid - 1;
                    }
                    n = lo;
                    cost_for(n, cost);
                }
            } else {
                // Rare range outside the cache: exact specialized unit loop.
                int inv = inv0;
                double money = f.money;
                for (int k = 0; k < limit; ++k) {
                    const int price = market_price(item, inv - 1);
                    if (money < price) break;
                    money -= price;
                    cost += price;
                    --inv;
                    ++n;
                }
            }
            if (n <= 0) return 0;
            f.money -= static_cast<double>(cost);
            f.total_spend += static_cast<double>(cost);
            f.shed[item] += static_cast<int16_t>(n);
            f.shed_total += n;
            st.market.inventory[item] -= n;
            return static_cast<uint16_t>(1u << item);
        }

        return 0;
    }

    uint16_t batch_same_product_sells(const OState& a, const OState& b) {
        const int item = a.item;
        Farm& f0 = st.farms[0];
        Farm& f1 = st.farms[1];
        const int n0 = std::max(0, std::min<int>(a.remaining, f0.shed[item]));
        const int n1 = std::max(0, std::min<int>(b.remaining, f1.shed[item]));
        const int paired = std::min(n0, n1);
        uint16_t dirty = 0;

        if (paired > 0) {
            int inv = st.market.inventory[item];
            int64_t revenue_each = 0;
            int moving_sales = 0;
            for (int k = 0; k < paired; ++k) {
                const int price = market_price(item, inv);
                revenue_each += price;
                if (price > 1) {
                    inv += 2;
                    moving_sales += 2;
                }
            }
            f0.shed[item] -= static_cast<int16_t>(paired);
            f1.shed[item] -= static_cast<int16_t>(paired);
            f0.shed_total -= paired;
            f1.shed_total -= paired;
            f0.money += static_cast<double>(revenue_each);
            f1.money += static_cast<double>(revenue_each);
            f0.sold_units[item] += paired;
            f1.sold_units[item] += paired;
            f0.sell_revenue += static_cast<double>(revenue_each);
            f1.sell_revenue += static_cast<double>(revenue_each);
            st.market.inventory[item] = inv;
            if (moving_sales > 0) dirty |= static_cast<uint16_t>(1u << item);
        }

        if (n0 > paired) {
            OState rem = a;
            rem.remaining = n0 - paired;
            rem.live = true;
            dirty |= batch_independent_order(rem, f0);
        }
        if (n1 > paired) {
            OState rem = b;
            rem.remaining = n1 - paired;
            rem.live = true;
            dirty |= batch_independent_order(rem, f1);
        }
        return dirty;
    }

    uint16_t process_market(const Action& a0, const Action& a1) {
        uint16_t dirty = 0;
        const Action* acts[2] = { &a0, &a1 };
        int nq[2];
        for (int p = 0; p < 2; ++p) nq[p] = std::min(acts[p]->n_orders, cfg.max_orders);
        int max_len = std::max(nq[0], nq[1]);

        for (int i = 0; i < max_len; ++i) {
            OState os[2]{};
            for (int p = 0; p < 2; ++p) {
                os[p].live = false;
                if (i < nq[p]) {
                    const Order& o = acts[p]->orders[i];
                    if (o.op == M_HIRE || o.op == M_BUY_LAND) {
                        os[p].type = o.op; os[p].live = true; os[p].remaining = 1;
                    } else if (o.op != M_NONE && o.n > 0) {
                        os[p].type = o.op; os[p].item = o.item; os[p].remaining = o.n; os[p].live = true;
                    }
                }
            }
            for (int p = 0; p < 2; ++p) {
                if (!os[p].live) continue;
                if (os[p].type == M_HIRE) { do_hire(st.farms[p]); os[p].live = false; }
                else if (os[p].type == M_BUY_LAND) { do_buy_land(st.farms[p]); os[p].live = false; }
            }

            if (os[0].live && os[1].live && os[0].type == M_SELL && os[1].type == M_SELL &&
                os[0].item == os[1].item && is_product(os[0].item)) {
                dirty |= batch_same_product_sells(os[0], os[1]);
                continue;
            }

            const int affect0 = market_affect_item(os[0]);
            const int affect1 = market_affect_item(os[1]);
            if (affect0 < 0 || affect1 < 0 || affect0 != affect1) {
                for (int p = 0; p < 2; ++p)
                    if (os[p].live) dirty |= batch_independent_order(os[p], st.farms[p]);
                continue;
            }

            for (;;) {
                struct Q { bool ok = false; uint8_t type = 0; uint8_t item = 0; int price = 0; };
                Q q[2];
                for (int p = 0; p < 2; ++p) {
                    if (!os[p].live || os[p].remaining <= 0) continue;
                    uint8_t t = os[p].type, it = os[p].item;
                    if (t == M_SELL && is_product(it)) {
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it]) };
                    } else if (t == M_BUY_PRODUCT && (it == WHEAT || it == FERTILIZER)) {
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it] - 1) };
                    } else if (t == M_BUY_SEED && is_crop(it)) {
                        q[p] = { true, t, it, CROPS[it].seed };
                    } else if (t == M_BUY_ANIMAL && is_animal(it)) {
                        q[p] = { true, t, it, ANIMALS[it - GOOSE].cost };
                    } else {
                        os[p].live = false;
                    }
                }
                if (!q[0].ok && !q[1].ok) break;
                bool committed = false;
                for (int p = 0; p < 2; ++p) {
                    if (!q[p].ok) continue;
                    if (commit_unit(q[p].type, q[p].item, q[p].price, st.farms[p])) {
                        if (q[p].type == M_BUY_PRODUCT ||
                            (q[p].type == M_SELL && q[p].price > 1))
                            dirty |= static_cast<uint16_t>(1u << q[p].item);
                        os[p].remaining -= 1; committed = true;
                    } else {
                        os[p].live = false;
                    }
                }
                if (!committed) break;
            }
        }
        return dirty;
    }

    bool commit_unit(uint8_t op, uint8_t item, int price, Farm& f) {
        switch (op) {
            case M_SELL:
                if (f.shed[item] <= 0) return false;
                f.shed[item] -= 1; f.shed_total -= 1;
                f.money += price;
                f.sold_units[item] += 1;
                f.sell_revenue += price;
                if (price > 1) st.market.inventory[item] += 1;
                return true;
            case M_BUY_PRODUCT:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                st.market.inventory[item] -= 1;
                return true;
            case M_BUY_SEED:
                if (f.money < price) return false;
                f.money -= price; f.total_spend += price; f.seeds[item] += 1;
                return true;
            case M_BUY_ANIMAL:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                return true;
        }
        return false;
    }

    void refresh_prices(uint16_t mask = static_cast<uint16_t>((1u << N_PRODUCTS) - 1u)) {
        for (int i = 0; i < N_PRODUCTS; ++i)
            if (mask & (1u << i))
                st.market.prices[i] = market_price(i, st.market.inventory[i]);
    }

    void do_hire(Farm& f) {
        int cost = cfg.hire_mult * fib(f.hires_today);
        if (f.money < cost || f.n_units >= MAX_UNITS) return;
        f.money -= cost; f.total_spend += cost;
        f.hires_today += 1;
        int acc[4][2]; shed_access_tiles(cfg.board_size, acc);
        int occ[4] = {0,0,0,0};
        for (int u = 0; u < f.n_units; ++u)
            for (int k = 0; k < 4; ++k)
                if (f.pos_x[u] == acc[k][0] && f.pos_y[u] == acc[k][1]) occ[k]++;
        int best = 0;
        for (int k = 1; k < 4; ++k) if (occ[k] < occ[best]) best = k;
        int idx = f.n_units++;
        f.pos_x[idx] = static_cast<int8_t>(acc[best][0]);
        f.pos_y[idx] = static_cast<int8_t>(acc[best][1]);
        f.inv_clear(idx);
    }

    void do_buy_land(Farm& f) {
        int extra = f.n_quadrants - 1;
        if (extra >= 3) return;
        int cost = LAND_PRICES[extra];
        if (f.money < cost) return;
        f.money -= cost; f.total_spend += cost;
        int quad = LAND_ORDER[extra];
        f.n_quadrants += 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x)
                if (quadrant_of(x, y, cfg.board_size) == quad && f.tiles[y][x].kind == T_LOCKED)
                    f.tiles[y][x].kind = T_EMPTY;
    }

    uint16_t town_consume(int step_i) {
        uint16_t dirty = 0;
        if (step_i % cfg.shop_sell_interval == 0) {
            for (int s = 0; s < st.n_shops; ++s) {
                uint16_t mask = SHOP_MASK[st.shops[s]];
                int mult = SHOP_MULT[st.shops[s]];
                for (int it = 0; it < N_PRODUCTS; ++it)
                    if (mask & (1u << it)) {
                        st.market.inventory[it] -= mult;
                        dirty |= static_cast<uint16_t>(1u << it);
                    }
            }
        }
        if (step_i % cfg.center_sell_interval == 0) {
            for (int it = 0; it < N_PRODUCTS; ++it)
                if (it != FERTILIZER) {
                    st.market.inventory[it] -= 1;
                    dirty |= static_cast<uint16_t>(1u << it);
                }
        }
        return dirty;
    }

    void decay_plants(Farm& f, int step_i) {
        if (step_i < f.next_decay_step) return;
        int next = NO_DECAY_STEP;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind != T_PLANT || t.max_lifespan_step < 0) continue;
                if (step_i < t.max_lifespan_step) {
                    next = std::min(next, static_cast<int>(t.max_lifespan_step));
                    continue;
                }
                if ((step_i - t.max_lifespan_step) % 2 != 0) {
                    next = std::min(next, step_i + 1);
                    continue;
                }
                t.yield_units -= 1;
                if (t.yield_units <= 0) {
                    Tile w{}; w.kind = T_WEED; t = w;
                } else {
                    next = std::min(next, step_i + 2);
                }
            }
        f.next_decay_step = next;
    }

    void end_of_day(int day) {
        PyRandom rng((cfg.seed * 1000003ull) ^ static_cast<uint64_t>(day),
                     cfg.cache_daily_rng_seed_state);
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            daily_refresh_and_spawn(f, day, rng);
            drop_inventories(f);
            int h = cfg.board_size / 2;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
            f.hires_today = 0;
        }
        int next_day = day + 1;
        if (next_day > 0 && next_day % cfg.shop_unlock_interval == 0 && st.n_shops < MAX_SHOP_INSTANCES)
            st.shops[st.n_shops++] = static_cast<uint8_t>(rng.choice_index(N_SHOPS));
    }

    void daily_refresh_and_spawn(Farm& f, int day, PyRandom& rng) {
        const int next_day = day + 1;
        const int h = cfg.board_size / 2;
        for (int y = 0; y < cfg.board_size; ++y) {
            int xmax;
            if (y < h) {
                xmax = f.n_quadrants >= 2 ? cfg.board_size : h;
            } else {
                if (f.n_quadrants < 3) continue;
                xmax = f.n_quadrants >= 4 ? cfg.board_size : h;
            }
            for (int x = 0; x < xmax; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind == T_PLANT) {
                    const bool was_watered = t.watered_today;
                    t.consecutive_dry = was_watered ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                    t.watered_today = false;
                    if (t.consecutive_dry >= 2) {
                        Tile w{}; w.kind = T_WEED; t = w;
                    } else {
                        const CropDef& cd = CROPS[t.what];
                        if (cd.ongoing) {
                            const int since = next_day - t.planted_day - cd.first_yield_day;
                            if (since >= 0 && since % cd.interval == 0) {
                                const int count = since / cd.interval + 1;
                                if (count <= cd.max_yield) {
                                    const bool fert = was_watered && t.fertilized_until_day >= day;
                                    t.yield_units = static_cast<int8_t>(
                                        std::min(cd.max_yield, t.yield_units + (fert ? 2 : 1)));
                                    if (count == cd.max_yield) {
                                        t.max_lifespan_step = (next_day + 1) * cfg.turns_per_day;
                                        f.next_decay_step = std::min(
                                            f.next_decay_step, static_cast<int>(t.max_lifespan_step));
                                    }
                                }
                            }
                        }
                    }
                } else if (t.has_animal) {
                    t.consecutive_dry = t.fed_today ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                    if (t.consecutive_dry >= 2) {
                        const TileKind k = t.kind;
                        Tile s{}; s.kind = k; t = s;
                    } else {
                        const AnimalDef& ad = ANIMALS[t.what - GOOSE];
                        const int since = next_day - t.planted_day - ad.first_yield_day;
                        if (since >= 0 && since % ad.interval == 0) {
                            const int bonus = t.fed_today ? t.pending_care_bonus : 0;
                            t.yield_units = static_cast<int8_t>(
                                std::min(ad.max_held, t.yield_units + 1 + bonus));
                            t.pending_care_bonus = 0;
                        }
                        if (t.cared_today && t.fed_today) t.pending_care_bonus += 1;
                        t.fertilizer_available = true;
                        t.fed_today = false;
                        t.cared_today = false;
                    }
                }
                if (t.kind == T_EMPTY && rng.random() < cfg.weed_chance)
                    t.kind = T_WEED;
            }
        }
    }

    void drop_inventories(Farm& f) {
        for (int u = 0; u < f.n_units; ++u) {
            const int nk = f.inv_nkeys[u];
            for (int k = 0; k < nk; ++k) {
                const int it = f.inv_keys[u][k];
                const int amount = f.inv[u][it];
                if (amount <= 0) continue;
                const int room = std::max(0, cfg.shed_capacity - f.shed_total);
                const int take = std::min(amount, room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += amount - take;
                f.inv[u][it] = 0;
            }
            f.inv_nkeys[u] = 0;
        }
    }
};

}  // namespace kag
