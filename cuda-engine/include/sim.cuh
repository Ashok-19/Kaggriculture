#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "pyrandom.cuh"

namespace kagcuda {

enum Item : uint8_t {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER,
    GOOSE, COW, SHEEP, N_ITEMS
};
constexpr int N_PRODUCTS = 9;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
constexpr int MAX_UNITS = 40;
constexpr int BOARD = 10;
constexpr int MAX_ORDERS = 16;
constexpr int MAX_SHOP_INSTANCES = 8;
constexpr int NO_DECAY_STEP = 1 << 30;
constexpr int PRICE_CACHE_MIN = -20000;
constexpr int PRICE_CACHE_MAX = 50000;
constexpr int PRICE_CACHE_SPAN = PRICE_CACHE_MAX - PRICE_CACHE_MIN + 1;
constexpr double HINGE_GAIN = 8.0;

__host__ __device__ __forceinline__ bool is_crop(uint8_t i) { return i < N_CROPS; }
__host__ __device__ __forceinline__ bool is_product(uint8_t i) { return i < N_PRODUCTS; }
__host__ __device__ __forceinline__ bool is_animal(uint8_t i) { return i >= GOOSE && i < N_ITEMS; }
__host__ __device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

// Unit and market op ids are deliberately identical to cpp-engine/include/sim.hpp.
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

enum Structure : uint8_t { ST_COOP = 0, ST_PASTURE = 1 };
struct CropDef { int seed, first_yield_day, max_yield_day, interval, max_yield; bool ongoing; };
struct AnimalDef { int cost; Structure structure; int first_yield_day, interval, max_held; Item product; };
enum Shape : uint8_t { F_LINEAR, F_SQ, F_SQRT, F_LOG, F_LOG10, F_HINGE };
struct MarketDef { double base; int I0; double T; Shape below_f; double below_t; Shape above_f; double above_t; };

__device__ __constant__ CropDef D_CROPS[N_CROPS] = {
    {10,2,4,0,6,false}, {20,2,3,0,4,false}, {50,8,8,1,4,true},
    {100,10,10,2,4,true}, {80,10,12,0,6,false}
};
__device__ __constant__ AnimalDef D_ANIMALS[N_ANIMALS] = {
    {300,ST_COOP,4,1,4,EGG}, {400,ST_PASTURE,8,2,6,MILK}, {500,ST_PASTURE,6,3,6,WOOL}
};
__device__ __constant__ MarketDef D_MARKET[N_PRODUCTS] = {
    {25,10000,400,F_SQRT,0.80,F_LOG,0.20},
    {35,10000,450,F_HINGE,1.00,F_SQRT,0.70},
    {60,10000,200,F_HINGE,0.40,F_SQRT,0.60},
    {120,10000,100,F_SQRT,0.70,F_LINEAR,1.60},
    {250,10000,300,F_LOG,0.20,F_SQ,3.60},
    {50,10000,332,F_HINGE,0.40,F_LOG,0.20},
    {160,10000,122,F_SQRT,0.60,F_LINEAR,1.60},
    {200,10000,105,F_LOG,0.20,F_SQ,3.20},
    {100,10000,200,F_LINEAR,0.40,F_LINEAR,0.40}
};

__device__ __constant__ uint16_t D_SHOP_MASK[MAX_SHOP_INSTANCES] = {
    (1u<<EGG)|(1u<<WHEAT),
    (1u<<EGG)|(1u<<WHEAT)|(1u<<STRAWBERRY),
    (1u<<WHEAT)|(1u<<CARROT)|(1u<<TOMATO)|(1u<<STRAWBERRY),
    (1u<<STRAWBERRY)|(1u<<MILK)|(1u<<WHEAT),
    (1u<<CARROT),
    (1u<<MILK)|(1u<<TOMATO)|(1u<<WHEAT),
    (1u<<STRAWBERRY)|(1u<<MILK),
    (1u<<WOOL)
};
__device__ __constant__ int D_SHOP_MULT[MAX_SHOP_INSTANCES] = {1,1,1,1,2,1,1,2};
__device__ __constant__ int D_LAND_ORDER[3] = {1,2,3};
__device__ __constant__ int D_LAND_PRICES[3] = {1000,2000,4000};

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
};

enum TileKind : uint8_t { T_EMPTY = 0, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT };
struct Tile {
    TileKind kind = T_EMPTY;
    uint8_t what = 0;
    bool has_animal = false;
    bool watered_today = false;
    bool fed_today = false;
    bool cared_today = false;
    bool fertilizer_available = false;
    int8_t consecutive_dry = 0;
    int8_t yield_units = 0;
    int8_t pending_care_bonus = 0;
    int16_t planted_day = 0;
    int32_t max_lifespan_step = -1;
    int16_t fertilized_until_day = -1;
};

// CUDA 11.5 does not reliably propagate Tile's non-zero default member
// initializers through a large nested State{} assignment. Construct every tile
// explicitly so snapshot state is bit-for-bit logical parity with the CPU oracle.
__device__ __forceinline__ Tile make_tile(TileKind kind) {
    Tile t;
    t.kind = kind;
    t.what = 0;
    t.has_animal = false;
    t.watered_today = false;
    t.fed_today = false;
    t.cared_today = false;
    t.fertilizer_available = false;
    t.consecutive_dry = 0;
    t.yield_units = 0;
    t.pending_care_bonus = 0;
    t.planted_day = 0;
    t.max_lifespan_step = -1;
    t.fertilized_until_day = -1;
    return t;
}
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
    double sell_revenue = 0;
    double total_spend = 0;

    __device__ __forceinline__ void inv_add(int u, int item, int n) {
        if (n <= 0) return;
        if (inv[u][item] == 0) inv_keys[u][inv_nkeys[u]++] = static_cast<uint8_t>(item);
        inv[u][item] += static_cast<int16_t>(n);
    }
    __device__ __forceinline__ void inv_erase(int u, int item) {
        inv[u][item] = 0;
        int k = 0;
        while (k < inv_nkeys[u] && inv_keys[u][k] != item) ++k;
        if (k == inv_nkeys[u]) return;
        for (int j = k; j + 1 < inv_nkeys[u]; ++j) inv_keys[u][j] = inv_keys[u][j + 1];
        --inv_nkeys[u];
    }
    __device__ __forceinline__ bool inv_take(int u, int item, int n) {
        if (inv[u][item] < n) return false;
        inv[u][item] -= static_cast<int16_t>(n);
        if (inv[u][item] == 0) inv_erase(u, item);
        return true;
    }
    __device__ __forceinline__ void inv_clear(int u) {
        for (int i = 0; i < N_ITEMS; ++i) inv[u][i] = 0;
        inv_nkeys[u] = 0;
    }
};
struct Market { int32_t inventory[N_PRODUCTS]; int32_t prices[N_PRODUCTS]; };
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

// Portable snapshot payload. It contains only simulation semantics; GPU table
// pointers are deliberately excluded and rebound when a snapshot is loaded.
struct Snapshot {
    Config cfg;
    State st;
};

struct UnitAction { uint8_t op = OP_PASS; uint8_t arg = 0; int16_t n = 1; };
struct Order { uint8_t op = M_NONE; uint8_t item = 0; int32_t n = 0; };
struct Action {
    UnitAction units[MAX_UNITS];
    int n_units = 1;
    Order orders[MAX_ORDERS];
    int n_orders = 0;
};
struct Turn { Action a[2]; };

struct DeviceTables {
    int* price_cache;
    int64_t* price_prefix;
    uint32_t* mt_base;
};

__device__ __forceinline__ double shape(Shape f, double x, double T = 0.0) {
    if (x < 0.0) x = 0.0;
    switch (f) {
        case F_LINEAR: return x;
        case F_SQ: return x * x;
        case F_SQRT: return sqrt(x);
        case F_LOG: return log(1.0 + x);
        case F_LOG10: return log10(1.0 + x);
        case F_HINGE: {
            if (T <= 0.0) return x;
            const double u = x / T;
            const double d = u > 1.0 ? u - 1.0 : 0.0;
            return u + HINGE_GAIN * d * d;
        }
    }
    return x;
}

__device__ __forceinline__ int market_price_formula(int item, int inv) {
    const MarketDef p = D_MARKET[item];
    const double below_den = shape(p.below_f, p.T, p.T);
    const double above_den = shape(p.above_f, p.T, p.T);
    const double below_amp = p.below_t * p.base / below_den;
    const double above_amp = p.above_t * p.base / above_den;
    double price;
    if (inv < p.I0)
        price = p.base + below_amp * shape(p.below_f, static_cast<double>(p.I0 - inv), p.T);
    else
        price = p.base - above_amp * shape(p.above_f, static_cast<double>(inv - p.I0), p.T);
    int v = static_cast<int>(nearbyint(price));
    return v < 1 ? 1 : v;
}

__device__ __forceinline__ int market_price(const DeviceTables* tab, int item, int inv) {
    if (inv >= PRICE_CACHE_MIN && inv <= PRICE_CACHE_MAX)
        return tab->price_cache[item * PRICE_CACHE_SPAN + (inv - PRICE_CACHE_MIN)];
    return market_price_formula(item, inv);
}

__device__ __forceinline__ bool cached_market_price_sum(const DeviceTables* tab, int item,
                                                         int lo_inv, int hi_inv, int64_t& out) {
    if (lo_inv > hi_inv) { out = 0; return true; }
    if (item < 0 || item >= N_PRODUCTS || lo_inv < PRICE_CACHE_MIN || hi_inv > PRICE_CACHE_MAX)
        return false;
    const int64_t* p = tab->price_prefix + static_cast<int64_t>(item) * (PRICE_CACHE_SPAN + 1);
    const int lo = lo_inv - PRICE_CACHE_MIN;
    const int hi = hi_inv - PRICE_CACHE_MIN;
    out = p[hi + 1] - p[lo];
    return true;
}

__global__ void init_price_cache_kernel(DeviceTables tab) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N_PRODUCTS * PRICE_CACHE_SPAN;
    if (idx >= total) return;
    const int item = idx / PRICE_CACHE_SPAN;
    const int inv = PRICE_CACHE_MIN + idx - item * PRICE_CACHE_SPAN;
    tab.price_cache[idx] = market_price_formula(item, inv);
}

__global__ void init_price_prefix_kernel(DeviceTables tab) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= N_PRODUCTS) return;
    int64_t* out = tab.price_prefix + static_cast<int64_t>(item) * (PRICE_CACHE_SPAN + 1);
    const int* prices = tab.price_cache + item * PRICE_CACHE_SPAN;
    out[0] = 0;
    for (int k = 0; k < PRICE_CACHE_SPAN; ++k) out[k + 1] = out[k] + prices[k];
}

__global__ void init_mt_base_kernel(DeviceTables tab) {
    if (blockIdx.x || threadIdx.x) return;
    tab.mt_base[0] = 19650218u;
    for (int i = 1; i < PyRandom::N; ++i)
        tab.mt_base[i] = 1812433253u * (tab.mt_base[i - 1] ^ (tab.mt_base[i - 1] >> 30)) +
                         static_cast<uint32_t>(i);
}

// Build the first 624 exact tempered CPython-MT outputs once per
// (snapshot, day). Candidate continuations sharing a snapshot then consume the
// common RNG stream directly instead of copying/twisting a private MT state.
__global__ void build_daily_rng_cache_kernel(const Snapshot* snapshots, int n_snapshots,
                                             int first_day, int n_days, const uint32_t* mt_base,
                                             uint32_t* out) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_snapshots * n_days;
    if (id >= total) return;
    const int snapshot_id = id / n_days;
    const int day = first_day + (id - snapshot_id * n_days);
    PyRandom rng;
    const uint64_t key = (snapshots[snapshot_id].cfg.seed * 1000003ull) ^ static_cast<uint64_t>(day);
    rng.seed(key, mt_base);
    uint32_t* dst = out + static_cast<size_t>(id) * PyRandom::N;
    for (int k = 0; k < PyRandom::N; ++k) dst[k] = rng.genrand_uint32();
}

__device__ __forceinline__ int quadrant_of(int x, int y, int bs) {
    const int half = bs / 2;
    return (y < half ? 0 : 2) + (x < half ? 0 : 1);
}
__device__ __forceinline__ bool is_shed_adjacent(int x, int y, int bs) {
    const int h = bs / 2;
    return (x == h - 1 || x == h) && (y == h - 1 || y == h);
}
__device__ __forceinline__ int fib(int n) {
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { const int t = b; b = a + b; a = t; }
    return a;
}

struct Sim {
    Config cfg;
    State st;
    DeviceTables tab;
    const uint32_t* daily_rng_cache = nullptr;
    int daily_rng_first_day = 0;
    int daily_rng_n_days = 0;

    __device__ __forceinline__ void init(const Config& c, const DeviceTables* t) {
        cfg = c; tab = *t;
        daily_rng_cache = nullptr; daily_rng_first_day = 0; daily_rng_n_days = 0;
        reset();
    }

    __device__ __forceinline__ void save_snapshot(Snapshot* out) const {
        out->cfg = cfg;
        out->st = st;
    }

    __device__ __forceinline__ void bind_rng_cache(const uint32_t* rng_cache,
                                                   int cache_first_day, int cache_n_days) {
        daily_rng_cache = rng_cache;
        daily_rng_first_day = cache_first_day;
        daily_rng_n_days = cache_n_days;
    }

    __device__ __forceinline__ void load_snapshot(const Snapshot& snap, const DeviceTables* t,
                                                  const uint32_t* rng_cache = nullptr,
                                                  int cache_first_day = 0, int cache_n_days = 0) {
        cfg = snap.cfg;
        st = snap.st;
        tab = *t;
        daily_rng_cache = rng_cache;
        daily_rng_first_day = cache_first_day;
        daily_rng_n_days = cache_n_days;
    }

    __device__ void reset() {
        st = State{};
        const int h = cfg.board_size / 2;
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            f.money = cfg.starting_money;
            for (int y = 0; y < cfg.board_size; ++y)
                for (int x = 0; x < cfg.board_size; ++x)
                    f.tiles[y][x] = make_tile(quadrant_of(x,y,cfg.board_size) == 0 ? T_EMPTY : T_LOCKED);
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
        }
        for (int i = 0; i < N_PRODUCTS; ++i) {
            st.market.inventory[i] = 10000;
            st.market.prices[i] = static_cast<int32_t>(D_MARKET[i].base);
        }
    }

    __device__ __forceinline__ void step(const Action& a0, const Action& a1) {
        step_impl<false>(a0, a0.n_units, a1, a1.n_units);
    }

    __device__ __forceinline__ void step_cached(const Action& a0, const Action& a1) {
        step_impl<true>(a0, a0.n_units, a1, a1.n_units);
    }

    __device__ __forceinline__ void step_with_unit_counts(const Action& a0, int n_units0,
                                                           const Action& a1, int n_units1) {
        step_impl<false>(a0, n_units0, a1, n_units1);
    }

    __device__ __forceinline__ void step_with_unit_counts_cached(const Action& a0, int n_units0,
                                                                  const Action& a1, int n_units1) {
        step_impl<true>(a0, n_units0, a1, n_units1);
    }

    template <bool USE_RNG_CACHE>
    __device__ void step_impl(const Action& a0, int n_units0, const Action& a1, int n_units1) {
        if (st.done) return;
        const Action* acts[2] = {&a0,&a1};
        const int action_units[2] = {
            imin(imax(n_units0,0),MAX_UNITS), imin(imax(n_units1,0),MAX_UNITS)
        };
        const int step_i = st.step;
        const int day = step_i / cfg.turns_per_day;
        for (int p = 0; p < 2; ++p) apply_unit_actions(p,*acts[p],day,action_units[p]);
        uint16_t dirty = process_market(*acts[0],*acts[1]);
        dirty |= town_consume(step_i);
        refresh_prices(dirty);
        for (int p = 0; p < 2; ++p) decay_plants(st.farms[p],step_i);
        if ((step_i + 1) % cfg.turns_per_day == 0) {
            if constexpr (USE_RNG_CACHE) end_of_day_cached(day);
            else end_of_day_uncached(day);
        }
        st.step = step_i + 1;
        st.day = st.step / cfg.turns_per_day;
        st.hour = st.step % cfg.turns_per_day;
        if (step_i >= cfg.episode_steps - 2) st.done = true;
    }

    __device__ void apply_unit_actions(int p, const Action& a, int day, int action_units) {
        Farm& f = st.farms[p];
        int demand[N_CROPS] = {0,0,0,0,0};
        for (int i = 0; i < action_units; ++i)
            if (a.units[i].op == OP_PLANT && a.units[i].arg < N_CROPS) ++demand[a.units[i].arg];
        const int n = imin(action_units,f.n_units);
        bool blocked[N_CROPS];
        for (int c = 0; c < N_CROPS; ++c) blocked[c] = demand[c] > f.seeds[c];
        for (int i = 0; i < n; ++i) {
            const UnitAction u = a.units[i];
            if (u.op == OP_PLANT && u.arg < N_CROPS && blocked[u.arg]) continue;
            apply_unit(f,i,u,day);
        }
    }

    __device__ void apply_unit(Farm& f, int idx, const UnitAction& u, int day) {
        const int bs = cfg.board_size;
        int fx = f.pos_x[idx], fy = f.pos_y[idx];
        int16_t* inv = f.inv[idx];
        switch (u.op) {
            case OP_PASS: return;
            case OP_NORTH: case OP_SOUTH: case OP_EAST: case OP_WEST: {
                const int nx = fx + (u.op == OP_EAST) - (u.op == OP_WEST);
                const int ny = fy + (u.op == OP_SOUTH) - (u.op == OP_NORTH);
                if (nx < 0 || nx >= bs || ny < 0 || ny >= bs) return;
                f.pos_x[idx] = static_cast<int8_t>(nx); f.pos_y[idx] = static_cast<int8_t>(ny); return;
            }
            default: break;
        }
        Tile& tile = f.tiles[fy][fx];
        if (u.op == OP_DROP) {
            if (!is_shed_adjacent(fx,fy,bs)) return;
            const int nk = f.inv_nkeys[idx];
            for (int k = 0; k < nk; ++k) {
                const int it = f.inv_keys[idx][k];
                const int amount = inv[it];
                if (amount <= 0) continue;
                const int room = imax(0,cfg.shed_capacity - f.shed_total);
                const int take = imin(amount,room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += amount - take; inv[it] = 0;
            }
            f.inv_nkeys[idx] = 0; return;
        }
        if (u.op == OP_PICKUP) {
            if (!is_shed_adjacent(fx,fy,bs) || u.n <= 0 || u.arg >= N_ITEMS) return;
            const int take = imin(static_cast<int>(u.n),static_cast<int>(f.shed[u.arg]));
            if (take <= 0) return;
            f.shed[u.arg] -= take; f.shed_total -= take; f.inv_add(idx,u.arg,take); return;
        }
        if (u.op == OP_PLACE) {
            if (u.arg >= N_ITEMS) return;
            if (is_animal(u.arg)) {
                const AnimalDef ad = D_ANIMALS[u.arg - GOOSE];
                const TileKind want = ad.structure == ST_COOP ? T_COOP : T_PASTURE;
                if (tile.kind == want && !tile.has_animal) {
                    if (f.inv_take(idx,u.arg,1)) {
                        Tile t = make_tile(want); t.what = u.arg; t.has_animal = true;
                        t.planted_day = static_cast<int16_t>(day); tile = t;
                    }
                    return;
                }
            }
            if (is_shed_adjacent(fx,fy,bs)) {
                int n = imin(static_cast<int>(u.n),static_cast<int>(inv[u.arg]));
                if (n <= 0) return;
                n = imin(n,imax(0,cfg.shed_capacity - f.shed_total));
                if (n <= 0) return;
                f.inv_take(idx,u.arg,n); f.shed[u.arg] += n; f.shed_total += n;
            }
            return;
        }
        if (tile.kind == T_LOCKED) return;
        switch (u.op) {
            case OP_PLANT: {
                if (u.arg >= N_CROPS || tile.kind != T_EMPTY || f.seeds[u.arg] <= 0) return;
                --f.seeds[u.arg]; const CropDef cd = D_CROPS[u.arg]; Tile t = make_tile(T_PLANT);
                t.what=u.arg; t.planted_day=static_cast<int16_t>(day);
                t.consecutive_dry=1; t.yield_units=cd.ongoing ? 0 : 1;
                t.max_lifespan_step = cd.ongoing ? -1 : (day + cd.max_yield_day + 1) * cfg.turns_per_day;
                tile=t; if (t.max_lifespan_step >= 0) f.next_decay_step=imin(f.next_decay_step,t.max_lifespan_step); return;
            }
            case OP_WATER: {
                if (tile.kind != T_PLANT || tile.watered_today) return;
                tile.watered_today=true; const CropDef cd=D_CROPS[tile.what];
                if (!cd.ongoing) {
                    const int age=day-tile.planted_day; const int w0=(cd.max_yield_day+1)/2;
                    if (age>=w0 && age<=cd.max_yield_day) {
                        const int bonus=tile.fertilized_until_day>=day ? 2 : 1;
                        tile.yield_units=static_cast<int8_t>(imin(cd.max_yield,tile.yield_units+bonus));
                    }
                }
                return;
            }
            case OP_HARVEST: {
                if (tile.kind==T_EMPTY || tile.kind==T_WEED || tile.yield_units<=0) return;
                if (tile.kind==T_PLANT) {
                    const CropDef cd=D_CROPS[tile.what]; if (day-tile.planted_day<cd.first_yield_day) return;
                    f.inv_add(idx,tile.what,tile.yield_units); f.produced[tile.what]+=tile.yield_units; tile.yield_units=0;
                    if (!cd.ongoing) tile = make_tile(T_EMPTY);
                } else if (tile.has_animal) {
                    const Item product=D_ANIMALS[tile.what-GOOSE].product;
                    f.inv_add(idx,product,tile.yield_units); f.produced[product]+=tile.yield_units; tile.yield_units=0;
                }
                return;
            }
            case OP_FERTILIZE:
                if (tile.kind!=T_PLANT || !f.inv_take(idx,FERTILIZER,1)) return;
                tile.fertilized_until_day = static_cast<int16_t>(imax(tile.fertilized_until_day,day+2)); return;
            case OP_DIG:
                if (tile.kind==T_EMPTY || tile.has_animal) return; tile=make_tile(T_EMPTY); return;
            case OP_BUILD_COOP:
                if (tile.kind!=T_EMPTY) return; tile=make_tile(T_COOP); return;
            case OP_BUILD_PASTURE:
                if (tile.kind!=T_EMPTY) return; tile=make_tile(T_PASTURE); return;
            case OP_FEED:
                if (!tile.has_animal || tile.fed_today || !f.inv_take(idx,WHEAT,1)) return; tile.fed_today=true; return;
            case OP_COLLECT_FERTILIZER:
                if (!tile.has_animal || !tile.fertilizer_available) return; tile.fertilizer_available=false;
                f.inv_add(idx,FERTILIZER,1); ++f.produced[FERTILIZER]; return;
            case OP_CARE:
                if (!tile.has_animal || tile.cared_today) return; tile.cared_today=true; return;
            default: return;
        }
    }

    struct OState { uint8_t type=0; uint8_t item=0; int32_t remaining=0; bool live=false; };

    __device__ __forceinline__ int market_affect_item(const OState& os) const {
        if (!os.live || os.remaining<=0) return -1;
        if (os.type==M_SELL && is_product(os.item)) return os.item;
        if (os.type==M_BUY_PRODUCT && (os.item==WHEAT || os.item==FERTILIZER)) return os.item;
        return -1;
    }

    __device__ uint16_t batch_independent_order(const OState& os, Farm& f) {
        if (!os.live || os.remaining<=0) return 0;
        const int item=os.item, requested=os.remaining;
        if (os.type==M_BUY_SEED) {
            if (!is_crop(item)) return 0; const int price=D_CROPS[item].seed;
            const int affordable=price>0 ? static_cast<int>(f.money/price) : requested;
            const int n=imax(0,imin(requested,affordable)); if (n<=0) return 0;
            const int64_t cost=static_cast<int64_t>(n)*price; f.money-=static_cast<double>(cost);
            f.total_spend+=static_cast<double>(cost); f.seeds[item]+=static_cast<int16_t>(n); return 0;
        }
        if (os.type==M_BUY_ANIMAL) {
            if (!is_animal(item)) return 0; const int price=D_ANIMALS[item-GOOSE].cost;
            const int room=imax(0,cfg.shed_capacity-f.shed_total);
            const int affordable=price>0 ? static_cast<int>(f.money/price) : requested;
            const int n=imax(0,imin(requested,imin(room,affordable))); if (n<=0) return 0;
            const int64_t cost=static_cast<int64_t>(n)*price; f.money-=static_cast<double>(cost);
            f.total_spend+=static_cast<double>(cost); f.shed[item]+=static_cast<int16_t>(n); f.shed_total+=n; return 0;
        }
        if (os.type==M_SELL) {
            if (!is_product(item)) return 0; const int n=imax(0,imin(requested,static_cast<int>(f.shed[item])));
            if (n<=0) return 0; const int inv0=st.market.inventory[item]; int moving=0;
            if (market_price(&tab,item,inv0)>1) {
                if (market_price(&tab,item,inv0+n-1)>1) moving=n;
                else { int lo=0,hi=n; while(lo<hi){ const int mid=lo+(hi-lo+1)/2;
                    if(market_price(&tab,item,inv0+mid-1)>1) lo=mid; else hi=mid-1; } moving=lo; }
            }
            int64_t revenue=0;
            if (moving>0 && !cached_market_price_sum(&tab,item,inv0,inv0+moving-1,revenue)) {
                revenue=0; moving=0; int inv=inv0;
                for(int k=0;k<n;++k){ const int price=market_price(&tab,item,inv); revenue+=price; if(price>1){++inv;++moving;} }
            } else revenue += static_cast<int64_t>(n-moving);
            f.shed[item]-=static_cast<int16_t>(n); f.shed_total-=n; f.money+=static_cast<double>(revenue);
            f.sold_units[item]+=n; f.sell_revenue+=static_cast<double>(revenue); st.market.inventory[item]+=moving;
            return moving>0 ? static_cast<uint16_t>(1u<<item) : 0;
        }
        if (os.type==M_BUY_PRODUCT) {
            if (!(item==WHEAT || item==FERTILIZER)) return 0;
            const int room=imax(0,cfg.shed_capacity-f.shed_total); const int limit=imax(0,imin(requested,room));
            if(limit<=0) return 0; const int inv0=st.market.inventory[item]; int n=0; int64_t cost=0,full=0;
            const bool full_cached=cached_market_price_sum(&tab,item,inv0-limit,inv0-1,full);
            if(full_cached){
                if(static_cast<double>(full)<=f.money){ n=limit; cost=full; }
                else { int lo=0,hi=limit; while(lo<hi){ const int mid=lo+(hi-lo+1)/2; int64_t mc=0;
                    cached_market_price_sum(&tab,item,inv0-mid,inv0-1,mc); if(static_cast<double>(mc)<=f.money) lo=mid; else hi=mid-1; }
                    n=lo; cached_market_price_sum(&tab,item,inv0-n,inv0-1,cost); }
            } else {
                int inv=inv0; double money=f.money;
                for(int k=0;k<limit;++k){ const int price=market_price(&tab,item,inv-1); if(money<price) break;
                    money-=price; cost+=price; --inv; ++n; }
            }
            if(n<=0) return 0; f.money-=static_cast<double>(cost); f.total_spend+=static_cast<double>(cost);
            f.shed[item]+=static_cast<int16_t>(n); f.shed_total+=n; st.market.inventory[item]-=n;
            return static_cast<uint16_t>(1u<<item);
        }
        return 0;
    }

    __device__ uint16_t batch_same_product_sells(const OState& a, const OState& b) {
        const int item=a.item; Farm& f0=st.farms[0]; Farm& f1=st.farms[1];
        const int n0=imax(0,imin(a.remaining,static_cast<int>(f0.shed[item])));
        const int n1=imax(0,imin(b.remaining,static_cast<int>(f1.shed[item])));
        const int paired=imin(n0,n1); uint16_t dirty=0;
        if(paired>0){ int inv=st.market.inventory[item]; int64_t rev=0; int moving_sales=0;
            for(int k=0;k<paired;++k){ const int price=market_price(&tab,item,inv); rev+=price; if(price>1){inv+=2;moving_sales+=2;} }
            f0.shed[item]-=static_cast<int16_t>(paired); f1.shed[item]-=static_cast<int16_t>(paired);
            f0.shed_total-=paired; f1.shed_total-=paired; f0.money+=static_cast<double>(rev); f1.money+=static_cast<double>(rev);
            f0.sold_units[item]+=paired; f1.sold_units[item]+=paired; f0.sell_revenue+=static_cast<double>(rev); f1.sell_revenue+=static_cast<double>(rev);
            st.market.inventory[item]=inv; if(moving_sales>0) dirty|=static_cast<uint16_t>(1u<<item);
        }
        if(n0>paired){ OState r=a; r.remaining=n0-paired; r.live=true; dirty|=batch_independent_order(r,f0); }
        if(n1>paired){ OState r=b; r.remaining=n1-paired; r.live=true; dirty|=batch_independent_order(r,f1); }
        return dirty;
    }

    __device__ bool commit_unit(uint8_t op, uint8_t item, int price, Farm& f) {
        switch(op){
            case M_SELL:
                if(f.shed[item]<=0) return false; --f.shed[item]; --f.shed_total; f.money+=price; ++f.sold_units[item]; f.sell_revenue+=price;
                if(price>1) ++st.market.inventory[item]; return true;
            case M_BUY_PRODUCT:
                if(f.money<price || f.shed_total>=cfg.shed_capacity) return false; f.money-=price; f.total_spend+=price;
                ++f.shed[item]; ++f.shed_total; --st.market.inventory[item]; return true;
            case M_BUY_SEED:
                if(f.money<price) return false; f.money-=price; f.total_spend+=price; ++f.seeds[item]; return true;
            case M_BUY_ANIMAL:
                if(f.money<price || f.shed_total>=cfg.shed_capacity) return false; f.money-=price; f.total_spend+=price;
                ++f.shed[item]; ++f.shed_total; return true;
        }
        return false;
    }

    __device__ uint16_t process_market(const Action& a0, const Action& a1) {
        uint16_t dirty=0; const Action* acts[2]={&a0,&a1};
        int nq[2]={imin(acts[0]->n_orders,cfg.max_orders),imin(acts[1]->n_orders,cfg.max_orders)};
        const int max_len=imax(nq[0],nq[1]);
        for(int i=0;i<max_len;++i){
            OState os[2]{};
            for(int p=0;p<2;++p){
                if(i<nq[p]){ const Order& o=acts[p]->orders[i];
                    if(o.op==M_HIRE || o.op==M_BUY_LAND){os[p].type=o.op;os[p].live=true;os[p].remaining=1;}
                    else if(o.op!=M_NONE && o.n>0){os[p].type=o.op;os[p].item=o.item;os[p].remaining=o.n;os[p].live=true;}
                }
            }
            for(int p=0;p<2;++p){ if(!os[p].live) continue;
                if(os[p].type==M_HIRE){do_hire(st.farms[p]);os[p].live=false;}
                else if(os[p].type==M_BUY_LAND){do_buy_land(st.farms[p]);os[p].live=false;}
            }
            if(os[0].live&&os[1].live&&os[0].type==M_SELL&&os[1].type==M_SELL&&os[0].item==os[1].item&&is_product(os[0].item)){
                dirty|=batch_same_product_sells(os[0],os[1]); continue;
            }
            const int affect0=market_affect_item(os[0]), affect1=market_affect_item(os[1]);
            if(affect0<0 || affect1<0 || affect0!=affect1){
                for(int p=0;p<2;++p) if(os[p].live) dirty|=batch_independent_order(os[p],st.farms[p]); continue;
            }
            for(;;){
                struct Q { bool ok; uint8_t type,item; int price; } q[2]={{false,0,0,0},{false,0,0,0}};
                for(int p=0;p<2;++p){ if(!os[p].live||os[p].remaining<=0) continue; const uint8_t t=os[p].type,it=os[p].item;
                    if(t==M_SELL&&is_product(it)) q[p]={true,t,it,market_price(&tab,it,st.market.inventory[it])};
                    else if(t==M_BUY_PRODUCT&&(it==WHEAT||it==FERTILIZER)) q[p]={true,t,it,market_price(&tab,it,st.market.inventory[it]-1)};
                    else if(t==M_BUY_SEED&&is_crop(it)) q[p]={true,t,it,D_CROPS[it].seed};
                    else if(t==M_BUY_ANIMAL&&is_animal(it)) q[p]={true,t,it,D_ANIMALS[it-GOOSE].cost};
                    else os[p].live=false;
                }
                if(!q[0].ok&&!q[1].ok) break; bool committed=false;
                for(int p=0;p<2;++p){ if(!q[p].ok) continue;
                    if(commit_unit(q[p].type,q[p].item,q[p].price,st.farms[p])){
                        if(q[p].type==M_BUY_PRODUCT || (q[p].type==M_SELL&&q[p].price>1)) dirty|=static_cast<uint16_t>(1u<<q[p].item);
                        --os[p].remaining; committed=true;
                    } else os[p].live=false;
                }
                if(!committed) break;
            }
        }
        return dirty;
    }

    __device__ void refresh_prices(uint16_t mask) {
        for(int i=0;i<N_PRODUCTS;++i) if(mask&(1u<<i)) st.market.prices[i]=market_price(&tab,i,st.market.inventory[i]);
    }

    __device__ void do_hire(Farm& f) {
        const int cost=cfg.hire_mult*fib(f.hires_today); if(f.money<cost || f.n_units>=MAX_UNITS) return;
        f.money-=cost; f.total_spend+=cost; ++f.hires_today;
        const int h=cfg.board_size/2; int ax[4]={h-1,h,h-1,h}; int ay[4]={h-1,h-1,h,h}; int occ[4]={0,0,0,0};
        for(int u=0;u<f.n_units;++u) for(int k=0;k<4;++k) if(f.pos_x[u]==ax[k]&&f.pos_y[u]==ay[k]) ++occ[k];
        int best=0; for(int k=1;k<4;++k) if(occ[k]<occ[best]) best=k; const int idx=f.n_units++;
        f.pos_x[idx]=static_cast<int8_t>(ax[best]); f.pos_y[idx]=static_cast<int8_t>(ay[best]); f.inv_clear(idx);
    }

    __device__ void do_buy_land(Farm& f) {
        const int extra=f.n_quadrants-1; if(extra>=3) return; const int cost=D_LAND_PRICES[extra]; if(f.money<cost) return;
        f.money-=cost; f.total_spend+=cost; const int quad=D_LAND_ORDER[extra]; ++f.n_quadrants;
        for(int y=0;y<cfg.board_size;++y) for(int x=0;x<cfg.board_size;++x)
            if(quadrant_of(x,y,cfg.board_size)==quad && f.tiles[y][x].kind==T_LOCKED) f.tiles[y][x].kind=T_EMPTY;
    }

    __device__ uint16_t town_consume(int step_i) {
        uint16_t dirty=0;
        if(step_i%cfg.shop_sell_interval==0){
            for(int s=0;s<st.n_shops;++s){ const uint16_t mask=D_SHOP_MASK[st.shops[s]]; const int mult=D_SHOP_MULT[st.shops[s]];
                for(int it=0;it<N_PRODUCTS;++it) if(mask&(1u<<it)){st.market.inventory[it]-=mult;dirty|=static_cast<uint16_t>(1u<<it);} }
        }
        if(step_i%cfg.center_sell_interval==0) for(int it=0;it<N_PRODUCTS;++it) if(it!=FERTILIZER){--st.market.inventory[it];dirty|=static_cast<uint16_t>(1u<<it);}
        return dirty;
    }

    __device__ void decay_plants(Farm& f, int step_i) {
        if(step_i<f.next_decay_step) return; int next=NO_DECAY_STEP;
        for(int y=0;y<cfg.board_size;++y) for(int x=0;x<cfg.board_size;++x){ Tile& t=f.tiles[y][x];
            if(t.kind!=T_PLANT||t.max_lifespan_step<0) continue;
            if(step_i<t.max_lifespan_step){next=imin(next,t.max_lifespan_step);continue;}
            if((step_i-t.max_lifespan_step)%2!=0){next=imin(next,step_i+1);continue;}
            --t.yield_units; if(t.yield_units<=0)t=make_tile(T_WEED); else next=imin(next,step_i+2);
        }
        f.next_decay_step=next;
    }

    template <class RNG>
    __device__ __forceinline__ void end_of_day_with_rng(int day, RNG& rng) {
        for(int p=0;p<2;++p){ Farm& f=st.farms[p]; daily_refresh_and_spawn(f,day,rng); drop_inventories(f);
            const int h=cfg.board_size/2; f.n_units=1; f.pos_x[0]=static_cast<int8_t>(h-1); f.pos_y[0]=static_cast<int8_t>(h-1); f.hires_today=0; }
        const int next_day=day+1;
        if(next_day>0 && next_day%cfg.shop_unlock_interval==0 && st.n_shops<MAX_SHOP_INSTANCES)
            st.shops[st.n_shops++]=static_cast<uint8_t>(rng.choice_index(MAX_SHOP_INSTANCES));
    }

    __device__ __noinline__ void end_of_day_uncached(int day) {
        PyRandom rng;
        rng.seed((cfg.seed*1000003ull)^static_cast<uint64_t>(day),tab.mt_base);
        end_of_day_with_rng(day,rng);
    }

    __device__ __forceinline__ void end_of_day_cached(int day) {
        if (!daily_rng_cache || day < daily_rng_first_day || day >= daily_rng_first_day + daily_rng_n_days) {
            end_of_day_uncached(day);
            return;
        }
        const uint32_t* words = daily_rng_cache + static_cast<size_t>(day - daily_rng_first_day) * PyRandom::N;
        const uint64_t key = (cfg.seed*1000003ull)^static_cast<uint64_t>(day);
        CachedDayRandom rng(words,key,tab.mt_base);
        end_of_day_with_rng(day,rng);
    }

    template <class RNG>
    __device__ void daily_refresh_and_spawn(Farm& f, int day, RNG& rng) {
        const int next_day=day+1, h=cfg.board_size/2;
        for(int y=0;y<cfg.board_size;++y){ int xmax;
            if(y<h) xmax=f.n_quadrants>=2 ? cfg.board_size : h;
            else { if(f.n_quadrants<3) continue; xmax=f.n_quadrants>=4 ? cfg.board_size : h; }
            for(int x=0;x<xmax;++x){ Tile& t=f.tiles[y][x];
                if(t.kind==T_PLANT){ const bool was=t.watered_today; t.consecutive_dry=was?0:static_cast<int8_t>(t.consecutive_dry+1); t.watered_today=false;
                    if(t.consecutive_dry>=2)t=make_tile(T_WEED);
                    else { const CropDef cd=D_CROPS[t.what]; if(cd.ongoing){ const int since=next_day-t.planted_day-cd.first_yield_day;
                        if(since>=0 && since%cd.interval==0){ const int count=since/cd.interval+1; if(count<=cd.max_yield){
                            const bool fert=was && t.fertilized_until_day>=day; t.yield_units=static_cast<int8_t>(imin(cd.max_yield,t.yield_units+(fert?2:1)));
                            if(count==cd.max_yield){t.max_lifespan_step=(next_day+1)*cfg.turns_per_day;f.next_decay_step=imin(f.next_decay_step,t.max_lifespan_step);} } } } }
                } else if(t.has_animal){ t.consecutive_dry=t.fed_today?0:static_cast<int8_t>(t.consecutive_dry+1);
                    if(t.consecutive_dry>=2){const TileKind k=t.kind;t=make_tile(k);}
                    else { const AnimalDef ad=D_ANIMALS[t.what-GOOSE]; const int since=next_day-t.planted_day-ad.first_yield_day;
                        if(since>=0 && since%ad.interval==0){ const int bonus=t.fed_today?t.pending_care_bonus:0;
                            t.yield_units=static_cast<int8_t>(imin(ad.max_held,t.yield_units+1+bonus)); t.pending_care_bonus=0; }
                        if(t.cared_today&&t.fed_today) ++t.pending_care_bonus; t.fertilizer_available=true; t.fed_today=false; t.cared_today=false; }
                }
                if(t.kind==T_EMPTY && rng.random()<cfg.weed_chance) t.kind=T_WEED;
            }
        }
    }

    __device__ void drop_inventories(Farm& f) {
        for(int u=0;u<f.n_units;++u){ const int nk=f.inv_nkeys[u];
            for(int k=0;k<nk;++k){ const int it=f.inv_keys[u][k], amount=f.inv[u][it]; if(amount<=0) continue;
                const int room=imax(0,cfg.shed_capacity-f.shed_total), take=imin(amount,room);
                if(take>0){f.shed[it]+=take;f.shed_total+=take;} f.discarded[it]+=amount-take; f.inv[u][it]=0; }
            f.inv_nkeys[u]=0;
        }
    }
};

}  // namespace kagcuda
