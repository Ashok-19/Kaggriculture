#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "sim.hpp"

constexpr int KAG_OBS_SIZE = 6377;
constexpr int KAG_NUM_ATNS = 44;
constexpr int KAG_UNIT_HEADS = kag::MAX_UNITS;
constexpr int KAG_MARKET_SLOTS = 4;
constexpr int KAG_UNIT_CODE_SIZE = 212;
constexpr int KAG_MARKET_CODE_SIZE = 155;
static constexpr int KAG_QTY_BUCKETS[8] = {1, 2, 4, 8, 16, 32, 64, 100};

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float win;
    float tie;
    float margin;
    float unit_nonpass_rate;
    float market_nonempty_rate;
    float n;
};

struct KaggricultureEnv {
    Log log;
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    int env_id;
    uint64_t base_seed;
    uint64_t episode_index;
    int learner_player;
    int opponent_kind;
    int pending_reset;
    kag::Sim* sim;
    int episode_steps;
    int unit_slots_total;
    int unit_slots_nonpass;
    int market_slots_total;
    int market_slots_nonempty;
};

static inline uint64_t kag_mix_seed(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

static inline float kag_clip01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

static inline int kag_action_index(float x, int size) {
    if (!std::isfinite(x) || size <= 1) return 0;
    long v = static_cast<long>(x);
    if (v < 0) v = -v;
    return static_cast<int>(v % size);
}

static inline bool kag_is_shed_adjacent(const kag::Farm& f, int unit, int bs) {
    return kag::is_shed_adjacent(f.pos_x[unit], f.pos_y[unit], bs);
}

static inline bool kag_unit_code_to_action(const kag::Sim& sim, int player, int unit,
                                           int code, kag::UnitAction& out) {
    using namespace kag;
    out = UnitAction{};
    const Farm& f = sim.st.farms[player];
    if (unit < 0 || unit >= f.n_units || code < 0 || code >= KAG_UNIT_CODE_SIZE) return false;
    const int x = f.pos_x[unit];
    const int y = f.pos_y[unit];
    const Tile& tile = f.tiles[y][x];

    if (code == 0) { out.op = OP_PASS; return true; }
    if (code >= 1 && code <= 4) {
        static constexpr uint8_t ops[4] = {OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST};
        const uint8_t op = ops[code - 1];
        const int nx = x + (op == OP_EAST) - (op == OP_WEST);
        const int ny = y + (op == OP_SOUTH) - (op == OP_NORTH);
        if (nx < 0 || nx >= sim.cfg.board_size || ny < 0 || ny >= sim.cfg.board_size) return false;
        out.op = op;
        return true;
    }
    if (code == 5) {
        if (!kag_is_shed_adjacent(f, unit, sim.cfg.board_size) || f.inv_nkeys[unit] == 0) return false;
        out.op = OP_DROP;
        return true;
    }
    if (code == 6) {
        if (tile.kind != T_PLANT || tile.watered_today) return false;
        out.op = OP_WATER;
        return true;
    }
    if (code == 7) {
        if (tile.yield_units <= 0) return false;
        if (tile.kind == T_PLANT) {
            const CropDef& cd = CROPS[tile.what];
            if (sim.st.day - tile.planted_day < cd.first_yield_day) return false;
        } else if (!tile.has_animal) {
            return false;
        }
        out.op = OP_HARVEST;
        return true;
    }
    if (code == 8) {
        if (tile.kind != T_PLANT || f.inv[unit][FERTILIZER] <= 0) return false;
        out.op = OP_FERTILIZE;
        return true;
    }
    if (code == 9) {
        if (tile.kind == T_EMPTY || tile.has_animal) return false;
        out.op = OP_DIG;
        return true;
    }
    if (code == 10) {
        if (tile.kind != T_EMPTY) return false;
        out.op = OP_BUILD_COOP;
        return true;
    }
    if (code == 11) {
        if (tile.kind != T_EMPTY) return false;
        out.op = OP_BUILD_PASTURE;
        return true;
    }
    if (code == 12) {
        if (!tile.has_animal || tile.fed_today || f.inv[unit][WHEAT] <= 0) return false;
        out.op = OP_FEED;
        return true;
    }
    if (code == 13) {
        if (!tile.has_animal || !tile.fertilizer_available) return false;
        out.op = OP_COLLECT_FERTILIZER;
        return true;
    }
    if (code == 14) {
        if (!tile.has_animal || tile.cared_today) return false;
        out.op = OP_CARE;
        return true;
    }
    if (code >= 15 && code < 20) {
        const int crop = code - 15;
        if (tile.kind != T_EMPTY || f.seeds[crop] <= 0) return false;
        out.op = OP_PLANT;
        out.arg = static_cast<uint8_t>(crop);
        return true;
    }
    if (code >= 20 && code < 116) {
        const int z = code - 20;
        const int item = z / 8;
        const int q = KAG_QTY_BUCKETS[z % 8];
        if (!kag_is_shed_adjacent(f, unit, sim.cfg.board_size) || f.shed[item] <= 0) return false;
        out.op = OP_PICKUP;
        out.arg = static_cast<uint8_t>(item);
        out.n = static_cast<int16_t>(std::min<int>(q, f.shed[item]));
        return out.n > 0;
    }
    const int z = code - 116;
    const int item = z / 8;
    const int q = KAG_QTY_BUCKETS[z % 8];
    if (item < 0 || item >= N_ITEMS) return false;
    if (is_animal(static_cast<uint8_t>(item))) {
        const AnimalDef& ad = ANIMALS[item - GOOSE];
        const TileKind want = ad.structure == ST_COOP ? T_COOP : T_PASTURE;
        if (tile.kind != want || tile.has_animal || f.inv[unit][item] <= 0) return false;
        out.op = OP_PLACE;
        out.arg = static_cast<uint8_t>(item);
        out.n = 1;
        return true;
    }
    if (!kag_is_shed_adjacent(f, unit, sim.cfg.board_size) || f.inv[unit][item] <= 0) return false;
    out.op = OP_PLACE;
    out.arg = static_cast<uint8_t>(item);
    out.n = static_cast<int16_t>(std::min<int>(q, f.inv[unit][item]));
    return out.n > 0;
}

static inline kag::UnitAction kag_decode_legal_unit_action(const kag::Sim& sim, int player,
                                                           int unit, int raw_code) {
    kag::UnitAction legal[KAG_UNIT_CODE_SIZE];
    int n = 0;
    for (int code = 0; code < KAG_UNIT_CODE_SIZE; ++code) {
        kag::UnitAction a{};
        if (kag_unit_code_to_action(sim, player, unit, code, a)) legal[n++] = a;
    }
    if (n <= 0) return kag::UnitAction{};
    return legal[raw_code % n];
}

static inline bool kag_market_code_to_order(const kag::Sim& sim, int player,
                                            int code, kag::Order& out) {
    using namespace kag;
    out = Order{};
    const Farm& f = sim.st.farms[player];
    if (code < 0 || code >= KAG_MARKET_CODE_SIZE) return false;
    if (code == 0) return true;
    if (code == 1) {
        const int cost = sim.cfg.hire_mult * fib(f.hires_today);
        if (f.n_units >= MAX_UNITS || f.money < cost) return false;
        out.op = M_HIRE;
        out.n = 1;
        return true;
    }
    if (code == 2) {
        const int extra = f.n_quadrants - 1;
        if (extra < 0 || extra >= 3 || f.money < LAND_PRICES[extra]) return false;
        out.op = M_BUY_LAND;
        out.n = 1;
        return true;
    }
    if (code >= 3 && code < 43) {
        const int z = code - 3;
        const int crop = z / 8;
        const int q = KAG_QTY_BUCKETS[z % 8];
        const int price = CROPS[crop].seed;
        const int affordable = price > 0 ? static_cast<int>(f.money / price) : q;
        const int n = std::min(q, affordable);
        if (n <= 0) return false;
        out.op = M_BUY_SEED;
        out.item = static_cast<uint8_t>(crop);
        out.n = n;
        return true;
    }
    if (code >= 43 && code < 59) {
        const int z = code - 43;
        const int which = z / 8;
        const int item = which == 0 ? WHEAT : FERTILIZER;
        const int q = KAG_QTY_BUCKETS[z % 8];
        const int room = std::max(0, sim.cfg.shed_capacity - f.shed_total);
        if (room <= 0 || f.money < market_price(item, sim.st.market.inventory[item] - 1)) return false;
        out.op = M_BUY_PRODUCT;
        out.item = static_cast<uint8_t>(item);
        out.n = std::min(q, room);
        return out.n > 0;
    }
    if (code >= 59 && code < 83) {
        const int z = code - 59;
        const int animal = GOOSE + z / 8;
        const int q = KAG_QTY_BUCKETS[z % 8];
        const int room = std::max(0, sim.cfg.shed_capacity - f.shed_total);
        const int price = ANIMALS[animal - GOOSE].cost;
        const int affordable = price > 0 ? static_cast<int>(f.money / price) : q;
        const int n = std::min({q, room, affordable});
        if (n <= 0) return false;
        out.op = M_BUY_ANIMAL;
        out.item = static_cast<uint8_t>(animal);
        out.n = n;
        return true;
    }
    const int z = code - 83;
    const int item = z / 8;
    const int q = KAG_QTY_BUCKETS[z % 8];
    if (item < 0 || item >= N_PRODUCTS || f.shed[item] <= 0) return false;
    out.op = M_SELL;
    out.item = static_cast<uint8_t>(item);
    out.n = std::min<int>(q, f.shed[item]);
    return out.n > 0;
}

static inline kag::Order kag_decode_legal_market_order(const kag::Sim& sim, int player,
                                                       int raw_code) {
    kag::Order legal[KAG_MARKET_CODE_SIZE];
    int n = 0;
    for (int code = 0; code < KAG_MARKET_CODE_SIZE; ++code) {
        kag::Order o{};
        if (kag_market_code_to_order(sim, player, code, o)) legal[n++] = o;
    }
    if (n <= 0) return kag::Order{};
    return legal[raw_code % n];
}

static inline kag::Action kag_decode_learner_action(KaggricultureEnv* env) {
    using namespace kag;
    const Sim& sim = *env->sim;
    const int player = env->learner_player;
    const Farm& f = sim.st.farms[player];
    Action out{};
    out.n_units = f.n_units;
    out.n_orders = KAG_MARKET_SLOTS;
    env->unit_slots_total += f.n_units;
    for (int unit = 0; unit < f.n_units; ++unit) {
        const int raw_code = kag_action_index(env->actions[unit], KAG_UNIT_CODE_SIZE);
        const UnitAction a = kag_decode_legal_unit_action(sim, player, unit, raw_code);
        out.units[unit] = a;
        if (a.op != OP_PASS) env->unit_slots_nonpass++;
    }
    for (int slot = 0; slot < KAG_MARKET_SLOTS; ++slot) {
        env->market_slots_total++;
        const int raw_code = kag_action_index(env->actions[KAG_UNIT_HEADS + slot], KAG_MARKET_CODE_SIZE);
        const Order o = kag_decode_legal_market_order(sim, player, raw_code);
        out.orders[slot] = o;
        if (o.op != M_NONE) env->market_slots_nonempty++;
    }
    return out;
}

static inline kag::Action kag_pass_opponent(const kag::Sim& sim, int player) {
    kag::Action a{};
    a.n_units = sim.st.farms[player].n_units;
    return a;
}

static inline kag::Action kag_starter_opponent(const kag::Sim& sim, int player) {
    using namespace kag;
    Action a{};
    const Farm& f = sim.st.farms[player];
    a.n_units = f.n_units;
    const int x = f.pos_x[0];
    const int y = f.pos_y[0];
    const Tile& tile = f.tiles[y][x];
    if (f.shed[CARROT] > 0) a.orders[a.n_orders++] = Order{M_SELL, CARROT, f.shed[CARROT]};
    if (f.seeds[CARROT] == 0 && f.money >= CROPS[CARROT].seed)
        a.orders[a.n_orders++] = Order{M_BUY_SEED, CARROT, 1};
    if (tile.kind == T_EMPTY && f.seeds[CARROT] > 0) {
        a.units[0] = UnitAction{OP_PLANT, CARROT, 1};
    } else if (tile.kind == T_PLANT && tile.what == CARROT) {
        const int age = sim.st.day - tile.planted_day;
        if (age >= CROPS[CARROT].max_yield_day) a.units[0] = UnitAction{OP_HARVEST, 0, 1};
        else if (!tile.watered_today) a.units[0] = UnitAction{OP_WATER, 0, 1};
    }
    return a;
}

static inline kag::Action kag_make_opponent_action(const KaggricultureEnv* env) {
    const int p = 1 - env->learner_player;
    return env->opponent_kind == 0 ? kag_pass_opponent(*env->sim, p)
                                   : kag_starter_opponent(*env->sim, p);
}

static inline void kag_compute_observation(KaggricultureEnv* env) {
    using namespace kag;
    const Sim& sim = *env->sim;
    float* out = static_cast<float*>(env->observations);
    int i = 0;
    auto push = [&](float v) {
        if (i >= KAG_OBS_SIZE) std::abort();
        out[i++] = std::max(-1.0f, std::min(1.0f, v));
    };
    auto onehot = [&](int value, int n) {
        for (int k = 0; k < n; ++k) push(k == value ? 1.0f : 0.0f);
    };
    push(sim.st.step / 718.0f);
    push(sim.st.day / 29.0f);
    push(sim.st.hour / 23.0f);
    push(env->learner_player == 0 ? 1.0f : 0.0f);
    push(env->learner_player == 1 ? 1.0f : 0.0f);
    for (int p = 0; p < 2; ++p) {
        const Farm& f = sim.st.farms[p];
        push(std::tanh(static_cast<float>(f.money / 20000.0)));
        push(f.n_units / static_cast<float>(MAX_UNITS));
        push(f.n_quadrants / 4.0f);
        push(std::tanh(f.hires_today / 12.0f));
    }
    for (int p = 0; p < 2; ++p) {
        const Farm& f = sim.st.farms[p];
        for (int u = 0; u < MAX_UNITS; ++u) {
            const bool active = u < f.n_units;
            push(active ? 1.0f : 0.0f);
            push(active ? (2.0f * f.pos_x[u] / 9.0f - 1.0f) : 0.0f);
            push(active ? (2.0f * f.pos_y[u] / 9.0f - 1.0f) : 0.0f);
        }
    }
    for (int p = 0; p < 2; ++p) {
        const Farm& f = sim.st.farms[p];
        for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) {
            const Tile& t = f.tiles[y][x];
            onehot(static_cast<int>(t.kind), 6);
            const bool has_item = t.kind == T_PLANT || t.has_animal;
            for (int item = 0; item < N_ITEMS; ++item) push(has_item && t.what == item ? 1.0f : 0.0f);
            push(t.has_animal ? 1.0f : 0.0f);
            push(t.watered_today ? 1.0f : 0.0f);
            push(t.fed_today ? 1.0f : 0.0f);
            push(t.cared_today ? 1.0f : 0.0f);
            push(t.fertilizer_available ? 1.0f : 0.0f);
            push(kag_clip01(t.consecutive_dry / 2.0f));
            push(kag_clip01(t.yield_units / 6.0f));
            const bool alive = t.kind == T_PLANT || t.has_animal;
            push(alive ? kag_clip01((sim.st.day - t.planted_day) / 30.0f) : 0.0f);
            push(t.fertilized_until_day >= sim.st.day
                     ? kag_clip01((t.fertilized_until_day - sim.st.day + 1) / 3.0f) : 0.0f);
            push(t.max_lifespan_step >= 0
                     ? kag_clip01((t.max_lifespan_step - sim.st.step) / 720.0f) : 0.0f);
        }
    }
    for (int item = 0; item < N_PRODUCTS; ++item)
        push(std::tanh((sim.st.market.inventory[item] - 10000) / 1000.0f));
    for (int item = 0; item < N_PRODUCTS; ++item)
        push(std::tanh(sim.st.market.prices[item] / 500.0f));
    int shop_counts[N_SHOPS] = {0};
    for (int s = 0; s < sim.st.n_shops; ++s) if (sim.st.shops[s] < N_SHOPS) shop_counts[sim.st.shops[s]]++;
    for (int s = 0; s < N_SHOPS; ++s) push(shop_counts[s] / 8.0f);
    const Farm& own = sim.st.farms[env->learner_player];
    for (int item = 0; item < N_ITEMS; ++item) push(own.shed[item] / 100.0f);
    for (int crop = 0; crop < N_CROPS; ++crop) push(std::tanh(own.seeds[crop] / 20.0f));
    for (int u = 0; u < MAX_UNITS; ++u)
        for (int item = 0; item < N_ITEMS; ++item)
            push(u < own.n_units ? std::tanh(own.inv[u][item] / 20.0f) : 0.0f);
    push(own.shed_total / 100.0f);
    if (i != KAG_OBS_SIZE) {
        std::fprintf(stderr, "observation size mismatch: %d vs %d\n", i, KAG_OBS_SIZE);
        std::abort();
    }
}

static inline void kag_add_episode_log(KaggricultureEnv* env, float reward, double margin) {
    const float win = margin > 0 ? 1.0f : 0.0f;
    const float tie = margin == 0 ? 1.0f : 0.0f;
    env->log.perf += win + 0.5f * tie;
    env->log.score += static_cast<float>(margin);
    env->log.episode_return += reward;
    env->log.episode_length += static_cast<float>(env->episode_steps);
    env->log.win += win;
    env->log.tie += tie;
    env->log.margin += static_cast<float>(margin);
    env->log.unit_nonpass_rate += env->unit_slots_total > 0
        ? env->unit_slots_nonpass / static_cast<float>(env->unit_slots_total) : 0.0f;
    env->log.market_nonempty_rate += env->market_slots_total > 0
        ? env->market_slots_nonempty / static_cast<float>(env->market_slots_total) : 0.0f;
    env->log.n += 1.0f;
}

static inline void c_reset(KaggricultureEnv* env) {
    kag::Config cfg;
    const uint64_t ep = env->episode_index++;
    uint64_t seed = env->base_seed * 1000003ULL;
    seed += static_cast<uint64_t>(env->env_id + 1) * 10007ULL;
    seed += (ep + 1) * 97ULL;
    cfg.seed = kag_mix_seed(seed);
    env->sim->cfg = cfg;
    env->sim->reset();
    env->learner_player = static_cast<int>((ep + static_cast<uint64_t>(env->env_id)) & 1ULL);
    env->episode_steps = 0;
    env->unit_slots_total = 0;
    env->unit_slots_nonpass = 0;
    env->market_slots_total = 0;
    env->market_slots_nonempty = 0;
    env->pending_reset = 0;
    kag_compute_observation(env);
}

static inline void c_step(KaggricultureEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    // Puffer stores reward/terminal at the next network callback. After the
    // 719th real decision, keep the terminal state alive for one flush slot.
    // The sampled action in that slot is deliberately ignored; reset here so
    // the next 720-slot rollout starts at a fresh episode with zeroed MinGRU.
    if (env->pending_reset) {
        c_reset(env);
        return;
    }
    const kag::Action learner = kag_decode_learner_action(env);
    const kag::Action opponent = kag_make_opponent_action(env);
    if (env->learner_player == 0) env->sim->step(learner, opponent);
    else env->sim->step(opponent, learner);
    env->episode_steps++;
    if (env->sim->st.done) {
        const double own = env->sim->st.farms[env->learner_player].money;
        const double opp = env->sim->st.farms[1 - env->learner_player].money;
        const double margin = own - opp;
        const float sign = margin > 0 ? 1.0f : (margin < 0 ? -1.0f : 0.0f);
        const float reward = 0.9f * sign + 0.1f * std::tanh(static_cast<float>(margin / 5000.0));
        env->rewards[0] = reward;
        env->terminals[0] = 1.0f;
        kag_add_episode_log(env, reward, margin);
        env->pending_reset = 1;
        kag_compute_observation(env);
        return;
    }
    kag_compute_observation(env);
}

static inline void c_render(KaggricultureEnv*) {}
static inline void c_close(KaggricultureEnv* env) { delete env->sim; env->sim = nullptr; }
