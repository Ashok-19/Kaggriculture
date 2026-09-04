#include "sim.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace kag;

namespace {

constexpr const char* ITEM_NAMES[N_ITEMS] = {
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
    "GOOSE", "COW", "SHEEP"
};
constexpr const char* CROP_NAMES[N_CROPS] = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"};
constexpr const char* SHOP_NAMES[N_SHOPS] = {
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"
};

int item_id(const std::string& s) {
    for (int i = 0; i < N_ITEMS; ++i) if (s == ITEM_NAMES[i]) return i;
    return 255;
}

int unit_op(const std::string& s) {
    static constexpr const char* names[] = {
        "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
        "FEED", "COLLECT_FERTILIZER", "CARE"
    };
    for (int i = 0; i < 18; ++i) if (s == names[i]) return i;
    return OP_INVALID;
}

int market_op(const std::string& s) {
    if (s == "HIRE") return M_HIRE;
    if (s == "BUY_LAND") return M_BUY_LAND;
    if (s == "BUY_SEED") return M_BUY_SEED;
    if (s == "BUY_PRODUCT") return M_BUY_PRODUCT;
    if (s == "BUY_ANIMAL") return M_BUY_ANIMAL;
    if (s == "SELL") return M_SELL;
    return M_NONE;
}

UnitAction parse_unit(py::handle h) {
    UnitAction out{};
    if (h.is_none() || !(py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h))) return out;
    py::sequence seq = py::reinterpret_borrow<py::sequence>(h);
    if (seq.size() == 0 || !py::isinstance<py::str>(seq[0])) return out;
    out.op = static_cast<uint8_t>(unit_op(py::cast<std::string>(seq[0])));
    if (seq.size() >= 2 && py::isinstance<py::str>(seq[1]))
        out.arg = static_cast<uint8_t>(item_id(py::cast<std::string>(seq[1])));
    if (seq.size() >= 3) {
        try { out.n = static_cast<int16_t>(py::cast<int>(seq[2])); }
        catch (...) { out.n = 1; }
    }
    return out;
}

Order parse_order(py::handle h) {
    Order out{};
    if (h.is_none() || !(py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h))) return out;
    py::sequence seq = py::reinterpret_borrow<py::sequence>(h);
    if (seq.size() == 0 || !py::isinstance<py::str>(seq[0])) return out;
    out.op = static_cast<uint8_t>(market_op(py::cast<std::string>(seq[0])));
    if (out.op == M_HIRE || out.op == M_BUY_LAND) {
        out.n = 1;
        return out;
    }
    if (seq.size() < 3 || !py::isinstance<py::str>(seq[1])) {
        out.op = M_NONE;
        return out;
    }
    out.item = static_cast<uint8_t>(item_id(py::cast<std::string>(seq[1])));
    try { out.n = py::cast<int>(seq[2]); }
    catch (...) { out.n = 0; }
    return out;
}

Action parse_action(py::handle h) {
    Action out{};
    if (h.is_none() || !py::isinstance<py::dict>(h)) return out;
    py::dict d = py::reinterpret_borrow<py::dict>(h);
    out.n_units = 1;
    if (d.contains("farmer")) out.units[0] = parse_unit(d["farmer"]);
    if (d.contains("hands") && py::isinstance<py::list>(d["hands"])) {
        py::list hands = py::reinterpret_borrow<py::list>(d["hands"]);
        int count = std::min<int>(hands.size(), MAX_UNITS - 1);
        out.n_units = 1 + count;
        for (int i = 0; i < count; ++i) out.units[i + 1] = parse_unit(hands[i]);
    }
    if (d.contains("market") && py::isinstance<py::list>(d["market"])) {
        py::list orders = py::reinterpret_borrow<py::list>(d["market"]);
        out.n_orders = std::min<int>(orders.size(), 16);
        for (int i = 0; i < out.n_orders; ++i) out.orders[i] = parse_order(orders[i]);
    }
    return out;
}

void apply_config(Config& cfg, const py::dict& d) {
    auto set_int = [&](const char* key, int& dst) { if (d.contains(key)) dst = py::cast<int>(d[key]); };
    set_int("episodeSteps", cfg.episode_steps);
    set_int("boardSize", cfg.board_size);
    set_int("startingMoney", cfg.starting_money);
    set_int("maxMarketOrdersPerTurn", cfg.max_orders);
    set_int("turnsPerDay", cfg.turns_per_day);
    set_int("shedCapacity", cfg.shed_capacity);
    if (d.contains("weedSpawnChance")) cfg.weed_chance = py::cast<double>(d["weedSpawnChance"]);
    set_int("townShopUnlockInterval", cfg.shop_unlock_interval);
    set_int("townShopSellInterval", cfg.shop_sell_interval);
    set_int("townCenterSellInterval", cfg.center_sell_interval);
    set_int("farmHandCostMult", cfg.hire_mult);
    if (d.contains("marketParams")) {
        py::object value = d["marketParams"];
        if (!value.is_none() && py::len(value) != 0)
            throw std::runtime_error("C++ bridge does not support marketParams overrides");
    }
    if (cfg.board_size != BOARD)
        throw std::runtime_error("C++ bridge currently supports competition boardSize=10 only");
}

py::object tile_to_py(const Tile& t) {
    if (t.kind == T_EMPTY) return py::none();
    if (t.kind == T_LOCKED) return py::str("LOCKED");
    py::dict d;
    if (t.kind == T_WEED) {
        d["kind"] = "WEED";
        return std::move(d);
    }
    if (t.kind == T_PLANT) {
        d["kind"] = "PLANT";
        d["crop"] = CROP_NAMES[t.what];
        d["planted_day"] = t.planted_day;
        d["watered_today"] = t.watered_today;
        d["consecutive_unwatered"] = t.consecutive_dry;
        d["yield_units"] = t.yield_units;
        d["max_lifespan_step"] = t.max_lifespan_step;
        d["fertilized_until_day"] = t.fertilized_until_day;
        return std::move(d);
    }
    d["kind"] = (t.kind == T_COOP ? "COOP" : "PASTURE");
    if (t.has_animal) {
        d["animal"] = ITEM_NAMES[t.what];
        d["placed_day"] = t.planted_day;
        d["yield_units"] = t.yield_units;
        d["fed_today"] = t.fed_today;
        d["consecutive_unfed"] = t.consecutive_dry;
        d["cared_today"] = t.cared_today;
        d["fertilizer_available"] = t.fertilizer_available;
        d["pending_care_bonus"] = t.pending_care_bonus;
    }
    return std::move(d);
}

py::dict farm_to_py(const Farm& f, int board_size) {
    py::dict d;
    d["money"] = f.money;
    py::list rows;
    for (int y = 0; y < board_size; ++y) {
        py::list row;
        for (int x = 0; x < board_size; ++x) row.append(tile_to_py(f.tiles[y][x]));
        rows.append(row);
    }
    d["tiles"] = rows;
    py::list farmer; farmer.append(f.pos_x[0]); farmer.append(f.pos_y[0]);
    d["farmer"] = farmer;
    py::list hands;
    for (int u = 1; u < f.n_units; ++u) {
        py::list pos; pos.append(f.pos_x[u]); pos.append(f.pos_y[u]); hands.append(pos);
    }
    d["hands"] = hands;
    py::list quadrants;
    quadrants.append("NW");
    if (f.n_quadrants >= 2) quadrants.append("NE");
    if (f.n_quadrants >= 3) quadrants.append("SW");
    if (f.n_quadrants >= 4) quadrants.append("SE");
    d["unlocked_quadrants"] = quadrants;
    d["hires_today"] = f.hires_today;
    return d;
}

py::dict private_to_py(const Farm& f) {
    py::dict priv;
    py::dict shed;
    for (int i = 0; i < N_ITEMS; ++i) shed[ITEM_NAMES[i]] = f.shed[i];
    priv["shed"] = shed;
    py::dict seeds;
    for (int i = 0; i < N_CROPS; ++i) seeds[CROP_NAMES[i]] = f.seeds[i];
    priv["seeds"] = seeds;
    py::list inventories;
    for (int u = 0; u < f.n_units; ++u) {
        py::dict inv;
        for (int k = 0; k < f.inv_nkeys[u]; ++k) {
            int item = f.inv_keys[u][k];
            if (f.inv[u][item] > 0) inv[ITEM_NAMES[item]] = f.inv[u][item];
        }
        inventories.append(inv);
    }
    priv["inventories"] = inventories;
    return priv;
}

class CompiledTape;
class Snapshot;

class Game {
public:
    explicit Game(uint64_t seed, py::dict config = py::dict()) : sim(make_config(seed, config)) {}

    py::dict observation(int player) const {
        if (player < 0 || player > 1) throw std::runtime_error("player must be 0 or 1");
        py::dict obs;
        obs["step"] = sim.st.step;
        obs["player"] = player;
        py::list farms;
        farms.append(farm_to_py(sim.st.farms[0], sim.cfg.board_size));
        farms.append(farm_to_py(sim.st.farms[1], sim.cfg.board_size));
        obs["farms"] = farms;
        py::dict market;
        py::dict inventory;
        py::dict prices;
        for (int i = 0; i < N_PRODUCTS; ++i) {
            inventory[ITEM_NAMES[i]] = sim.st.market.inventory[i];
            prices[ITEM_NAMES[i]] = sim.st.market.prices[i];
        }
        market["inventory"] = inventory;
        market["prices"] = prices;
        obs["market"] = market;
        py::dict town;
        py::list shops;
        for (int i = 0; i < sim.st.n_shops; ++i) shops.append(SHOP_NAMES[sim.st.shops[i]]);
        town["unlocked_shops"] = shops;
        obs["town"] = town;
        obs["day"] = sim.st.day;
        obs["hour"] = sim.st.hour;
        obs["private"] = private_to_py(sim.st.farms[player]);
        return obs;
    }

    Snapshot snapshot() const;

    py::list observations_shared() const {
        // Fast-path for read-only agents: public state is identical for both
        // players, so materialize it once and share those nested objects. Each
        // player still receives a distinct top-level dict and private state.
        py::dict obs0 = observation(0);
        py::dict obs1;
        obs1["step"] = sim.st.step;
        obs1["player"] = 1;
        obs1["farms"] = obs0["farms"];
        obs1["market"] = obs0["market"];
        obs1["town"] = obs0["town"];
        obs1["day"] = sim.st.day;
        obs1["hour"] = sim.st.hour;
        obs1["private"] = private_to_py(sim.st.farms[1]);
        py::list out;
        out.append(std::move(obs0));
        out.append(std::move(obs1));
        return out;
    }

    void load_compiled_tape(const CompiledTape& tape, int tape_player);

    void load_tape(py::list tape_actions, int tape_player) {
        if (tape_player < 0 || tape_player > 1)
            throw std::runtime_error("tape_player must be 0 or 1");
        if (tape_actions.empty())
            throw std::runtime_error("tape_actions must not be empty");
        tape_.clear();
        tape_.reserve(tape_actions.size());
        for (py::handle action : tape_actions) tape_.push_back(parse_action(action));
        tape_player_ = tape_player;
    }

    void step_with_tape(py::object candidate_action) {
        if (tape_player_ < 0 || tape_.empty())
            throw std::runtime_error("load_tape() must be called before step_with_tape()");
        const int idx = std::min<int>(sim.st.step, static_cast<int>(tape_.size()) - 1);
        const Action& route = tape_[idx];

        // The project replay wrappers dynamically pad/truncate recorded hand
        // actions to the current counterfactual farm-hand count. The unused
        // Action slots are already PASS, so pass the effective count directly.
        const int current_units = sim.st.farms[tape_player_].n_units;
        Action candidate = parse_action(candidate_action);
        if (tape_player_ == 0)
            sim.step_with_unit_counts(route, current_units, candidate, candidate.n_units);
        else
            sim.step_with_unit_counts(candidate, candidate.n_units, route, current_units);
    }

    void step(py::object a0, py::object a1) {
        sim.step(parse_action(a0), parse_action(a1));
    }

    bool done() const { return sim.st.done; }

    py::list rewards() const {
        py::list out;
        out.append(sim.st.farms[0].money);
        out.append(sim.st.farms[1].money);
        return out;
    }

private:
    std::vector<Action> tape_;
    int tape_player_ = -1;

    Sim sim;

    static Config make_config(uint64_t seed, const py::dict& config) {
        Config cfg;
        apply_config(cfg, config);
        cfg.seed = seed;
        return cfg;
    }
};

class Snapshot {
public:
    explicit Snapshot(const Sim& sim) : sim_(sim) {}
    const Sim& sim() const { return sim_; }
    int step() const { return sim_.st.step; }

private:
    Sim sim_;
};

Snapshot Game::snapshot() const { return Snapshot(sim); }

std::vector<Action> parse_tape_actions(const py::list& actions) {
    if (actions.empty()) throw std::runtime_error("tape actions must not be empty");
    std::vector<Action> tape;
    tape.reserve(actions.size());
    for (py::handle action : actions) tape.push_back(parse_action(action));
    return tape;
}

class CompiledTape {
public:
    explicit CompiledTape(py::list actions) : actions_(parse_tape_actions(actions)) {}
    const std::vector<Action>& actions() const { return actions_; }
    size_t size() const { return actions_.size(); }

private:
    std::vector<Action> actions_;
};

void Game::load_compiled_tape(const CompiledTape& tape, int tape_player) {
    if (tape_player < 0 || tape_player > 1)
        throw std::runtime_error("tape_player must be 0 or 1");
    tape_ = tape.actions();
    tape_player_ = tape_player;
}

py::list run_tapes_batch(py::iterable seeds, py::list tape0_actions, py::list tape1_actions,
                         py::dict config = py::dict()) {
    std::vector<uint64_t> seed_values;
    for (py::handle seed : seeds) seed_values.push_back(py::cast<uint64_t>(seed));
    const std::vector<Action> tape0 = parse_tape_actions(tape0_actions);
    const std::vector<Action> tape1 = parse_tape_actions(tape1_actions);
    Config base;
    apply_config(base, config);

    std::vector<std::array<double, 2>> results(seed_values.size());
    {
        // Once Python tapes/seeds/config are decoded, the entire batch is pure
        // C++. Releasing the GIL also avoids blocking unrelated Python threads;
        // project tooling still enforces the two-worker local compute cap.
        py::gil_scoped_release release;
        for (size_t i = 0; i < seed_values.size(); ++i) {
            Config cfg = base;
            cfg.seed = seed_values[i];
            Sim sim(cfg);
            while (!sim.st.done) {
                const int idx0 = std::min<int>(sim.st.step, static_cast<int>(tape0.size()) - 1);
                const int idx1 = std::min<int>(sim.st.step, static_cast<int>(tape1.size()) - 1);
                const int units0 = sim.st.farms[0].n_units;
                const int units1 = sim.st.farms[1].n_units;
                sim.step_with_unit_counts(tape0[idx0], units0, tape1[idx1], units1);
            }
            results[i] = {sim.st.farms[0].money, sim.st.farms[1].money};
        }
    }

    py::list out;
    for (const auto& reward : results) {
        py::list row;
        row.append(reward[0]);
        row.append(reward[1]);
        out.append(std::move(row));
    }
    return out;
}

py::list run_snapshot_candidates(const Snapshot& snapshot, py::list candidate_tapes,
                                 const CompiledTape& opponent_tape, int candidate_player,
                                 int workers = 1) {
    if (candidate_player < 0 || candidate_player > 1)
        throw std::runtime_error("candidate_player must be 0 or 1");
    if (workers < 1 || workers > 2)
        throw std::runtime_error("workers must be 1 or 2 on this local workstation");

    std::vector<const CompiledTape*> candidates;
    candidates.reserve(candidate_tapes.size());
    for (py::handle obj : candidate_tapes)
        candidates.push_back(&py::cast<const CompiledTape&>(obj));
    const auto& opponent = opponent_tape.actions();
    std::vector<std::array<double, 2>> results(candidates.size());

    {
        py::gil_scoped_release release;
        auto run_range = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                Sim sim = snapshot.sim();
                sim.cfg.cache_daily_rng_seed_state = true;
                const auto& candidate = candidates[i]->actions();
                while (!sim.st.done) {
                    const int cidx = std::min<int>(sim.st.step, static_cast<int>(candidate.size()) - 1);
                    const int oidx = std::min<int>(sim.st.step, static_cast<int>(opponent.size()) - 1);
                    const int c_units = sim.st.farms[candidate_player].n_units;
                    const int o_player = 1 - candidate_player;
                    const int o_units = sim.st.farms[o_player].n_units;
                    if (candidate_player == 0)
                        sim.step_with_unit_counts(candidate[cidx], c_units, opponent[oidx], o_units);
                    else
                        sim.step_with_unit_counts(opponent[oidx], o_units, candidate[cidx], c_units);
                }
                results[i] = {sim.st.farms[0].money, sim.st.farms[1].money};
            }
        };
        if (workers == 2 && candidates.size() > 1) {
            const size_t mid = (candidates.size() + 1) / 2;
            std::thread other(run_range, mid, candidates.size());
            run_range(0, mid);
            other.join();
        } else {
            run_range(0, candidates.size());
        }
    }

    py::list out;
    for (const auto& reward : results) {
        py::list row;
        row.append(reward[0]);
        row.append(reward[1]);
        out.append(std::move(row));
    }
    return out;
}

py::list run_compiled_tapes_batch(py::iterable seeds, const CompiledTape& tape0,
                                  const CompiledTape& tape1,
                                  py::dict config = py::dict(), int workers = 1) {
    if (workers < 1 || workers > 2)
        throw std::runtime_error("workers must be 1 or 2 on this local workstation");
    std::vector<uint64_t> seed_values;
    for (py::handle seed : seeds) seed_values.push_back(py::cast<uint64_t>(seed));
    Config base;
    apply_config(base, config);
    const auto& a0 = tape0.actions();
    const auto& a1 = tape1.actions();

    std::vector<std::array<double, 2>> results(seed_values.size());
    {
        py::gil_scoped_release release;
        auto run_range = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                Config cfg = base;
                cfg.seed = seed_values[i];
                Sim sim(cfg);
                while (!sim.st.done) {
                    const int idx0 = std::min<int>(sim.st.step, static_cast<int>(a0.size()) - 1);
                    const int idx1 = std::min<int>(sim.st.step, static_cast<int>(a1.size()) - 1);
                    const int units0 = sim.st.farms[0].n_units;
                    const int units1 = sim.st.farms[1].n_units;
                    sim.step_with_unit_counts(a0[idx0], units0, a1[idx1], units1);
                }
                results[i] = {sim.st.farms[0].money, sim.st.farms[1].money};
            }
        };
        if (workers == 2 && seed_values.size() > 1) {
            const size_t mid = (seed_values.size() + 1) / 2;
            std::thread other(run_range, mid, seed_values.size());
            run_range(0, mid);
            other.join();
        } else {
            run_range(0, seed_values.size());
        }
    }

    py::list out;
    for (const auto& reward : results) {
        py::list row;
        row.append(reward[0]);
        row.append(reward[1]);
        out.append(std::move(row));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(kaggriculture_cpp, m) {
    m.doc() = "Parity-validated Kaggriculture 1.32.7 C++ simulator bridge";
    py::class_<CompiledTape>(m, "CompiledTape")
        .def(py::init<py::list>(), py::arg("actions"))
        .def_property_readonly("steps", &CompiledTape::size);
    py::class_<Snapshot>(m, "Snapshot")
        .def_property_readonly("step", &Snapshot::step);
    m.def("run_snapshot_candidates", &run_snapshot_candidates, py::arg("snapshot"),
          py::arg("candidate_tapes"), py::arg("opponent_tape"),
          py::arg("candidate_player"), py::arg("workers") = 1);
    m.def("run_compiled_tapes_batch", &run_compiled_tapes_batch, py::arg("seeds"),
          py::arg("tape0"), py::arg("tape1"), py::arg("config") = py::dict(),
          py::arg("workers") = 1);
    m.def("run_tapes_batch", &run_tapes_batch, py::arg("seeds"), py::arg("tape0_actions"),
          py::arg("tape1_actions"), py::arg("config") = py::dict());
    py::class_<Game>(m, "Game")
        .def(py::init<uint64_t, py::dict>(), py::arg("seed"), py::arg("config") = py::dict())
        .def("observation", &Game::observation)
        .def("observations_shared", &Game::observations_shared)
        .def("snapshot", &Game::snapshot)
        .def("step", &Game::step)
        .def("load_tape", &Game::load_tape, py::arg("tape_actions"), py::arg("tape_player"))
        .def("load_compiled_tape", &Game::load_compiled_tape,
             py::arg("tape"), py::arg("tape_player"))
        .def("step_with_tape", &Game::step_with_tape)
        .def_property_readonly("done", &Game::done)
        .def("rewards", &Game::rewards);
}
