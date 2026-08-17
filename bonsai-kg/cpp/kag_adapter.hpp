#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include "../../cpp-engine/include/sim.hpp"
#include <cstdint>
#include <climits>

namespace bonsaikg {

struct CppObservation {
    const kag::Sim* sim = nullptr;
    int player = 0;

    static int32_t money_i32(double x) {
        if (x <= static_cast<double>(INT32_MIN)) return INT32_MIN;
        if (x >= static_cast<double>(INT32_MAX)) return INT32_MAX;
        return static_cast<int32_t>(x);
    }
    int32_t feature(uint16_t id, int32_t arg) const {
        const kag::State& s = sim->st; const int p = player, q = 1-player;
        const kag::Farm& own = s.farms[p]; const kag::Farm& opp = s.farms[q];
        switch (id) {
            case F_ALWAYS: return 1; case F_STEP: return s.step; case F_DAY: return s.day; case F_HOUR: return s.hour;
            case F_OWN_MONEY: return money_i32(own.money); case F_OPP_MONEY: return money_i32(opp.money);
            case F_OWN_UNITS: return own.n_units; case F_OPP_UNITS: return opp.n_units;
            case F_OWN_LAND: return own.n_quadrants; case F_OPP_LAND: return opp.n_quadrants;
            case F_OWN_HIRES_TODAY: return own.hires_today;
            case F_MARKET_PRICE: return arg>=0 && arg<kag::N_PRODUCTS ? s.market.prices[arg] : 0;
            case F_MARKET_INVENTORY: return arg>=0 && arg<kag::N_PRODUCTS ? s.market.inventory[arg] : 0;
            case F_OWN_SHED: return arg>=0 && arg<kag::N_ITEMS ? own.shed[arg] : 0;
            case F_OWN_SHED_TOTAL: return own.shed_total;
            case F_OWN_SEED: return arg>=0 && arg<kag::N_CROPS ? own.seeds[arg] : 0;
            case F_OWN_INVENTORY: { const int u=(arg>>8)&255,it=arg&255; return u<own.n_units&&it<kag::N_ITEMS ? own.inv[u][it] : 0; }
            case F_SHOP_COUNT: return s.n_shops;
            case F_SHOP_AT: return arg>=0 && arg<s.n_shops ? s.shops[arg] : -1;
            case F_SHOP_SEEN: { int n=0; for(int i=0;i<s.n_shops;++i) n += s.shops[i]==arg; return n; }
            case F_OWN_WORKER_X: return arg>=0&&arg<own.n_units ? own.pos_x[arg] : -1;
            case F_OWN_WORKER_Y: return arg>=0&&arg<own.n_units ? own.pos_y[arg] : -1;
            case F_OPP_WORKER_X: return arg>=0&&arg<opp.n_units ? opp.pos_x[arg] : -1;
            case F_OPP_WORKER_Y: return arg>=0&&arg<opp.n_units ? opp.pos_y[arg] : -1;
            default: break;
        }
        if (id >= F_OWN_TILE_KIND && id <= F_OPP_TILE_FERT_AVAILABLE) {
            if (arg < 0 || arg >= 100) return 0;
            const int y=arg/10,x=arg%10;
            const bool own_side = id==F_OWN_TILE_KIND||id==F_OWN_TILE_ITEM||id==F_OWN_TILE_YIELD||id==F_OWN_TILE_WATERED||id==F_OWN_TILE_FED||id==F_OWN_TILE_CARED||id==F_OWN_TILE_FERT_AVAILABLE;
            const kag::Tile& t = (own_side?own:opp).tiles[y][x];
            switch(id){
                case F_OWN_TILE_KIND: case F_OPP_TILE_KIND: return t.kind;
                case F_OWN_TILE_ITEM: case F_OPP_TILE_ITEM: return t.what;
                case F_OWN_TILE_YIELD: case F_OPP_TILE_YIELD: return t.yield_units;
                case F_OWN_TILE_WATERED: case F_OPP_TILE_WATERED: return t.watered_today;
                case F_OWN_TILE_FED: case F_OPP_TILE_FED: return t.fed_today;
                case F_OWN_TILE_CARED: case F_OPP_TILE_CARED: return t.cared_today;
                case F_OWN_TILE_FERT_AVAILABLE: case F_OPP_TILE_FERT_AVAILABLE: return t.fertilizer_available;
            }
        }
        return 0;
    }
};

inline void to_cpp_action(const ActionResult& src, kag::Action& out) {
    out.n_units = src.n_units;
    for (int i=0;i<src.n_units&&i<kag::MAX_UNITS;++i) { out.units[i].op=src.units[i].op; out.units[i].arg=src.units[i].arg; out.units[i].n=src.units[i].n; }
    out.n_orders = src.n_market;
    for (int i=0;i<src.n_market&&i<16;++i) { out.orders[i].op=src.market[i].op; out.orders[i].item=src.market[i].item; out.orders[i].n=src.market[i].n; }
}

} // namespace bonsaikg
