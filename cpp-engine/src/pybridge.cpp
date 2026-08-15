#include "sim.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <stdexcept>
#include <string>

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
    Sim sim;

    static Config make_config(uint64_t seed, const py::dict& config) {
        Config cfg;
        apply_config(cfg, config);
        cfg.seed = seed;
        return cfg;
    }
};

}  // namespace

PYBIND11_MODULE(kaggriculture_cpp, m) {
    m.doc() = "Parity-validated Kaggriculture 1.32.7 C++ simulator bridge";
    py::class_<Game>(m, "Game")
        .def(py::init<uint64_t, py::dict>(), py::arg("seed"), py::arg("config") = py::dict())
        .def("observation", &Game::observation)
        .def("step", &Game::step)
        .def_property_readonly("done", &Game::done)
        .def("rewards", &Game::rewards);
}
