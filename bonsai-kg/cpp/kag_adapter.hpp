#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include "../../cpp-engine/include/sim.hpp"
#include <cstdint>
#include <climits>
#include <stdexcept>

namespace bonsaikg {

static_assert(kag::BOARD == BOARD_SIZE, "Bonsai-KG requires boardSize=10");

class CppObservation {
    const kag::Sim* sim_ = nullptr;
    int player_ = 0;

    static int32_t money_i32(double x) {
        if (x <= static_cast<double>(INT32_MIN)) return INT32_MIN;
        if (x >= static_cast<double>(INT32_MAX)) return INT32_MAX;
        return static_cast<int32_t>(x);
    }

public:
    CppObservation(const kag::Sim* sim, int player) : sim_(sim), player_(player) {
        if (!sim_ || player_ < 0 || player_ > 1) throw std::runtime_error("Bonsai-KG invalid C++ observation context");
        if (sim_->cfg.board_size != BOARD_SIZE) throw std::runtime_error("Bonsai-KG requires boardSize=10");
    }

    int32_t feature(uint16_t id, int32_t arg0 = 0, int32_t arg1 = 0) const {
        const kag::State& s = sim_->st; const int p = player_, q = 1-player_;
        const kag::Farm& own = s.farms[p]; const kag::Farm& opp = s.farms[q];
        switch (id) {
            case F_ALWAYS: return 1; case F_STEP: return s.step; case F_DAY: return s.day; case F_HOUR: return s.hour;
            case F_OWN_MONEY: return money_i32(own.money); case F_OPP_MONEY: return money_i32(opp.money);
            case F_OWN_UNITS: return own.n_units; case F_OPP_UNITS: return opp.n_units;
            case F_OWN_LAND: return own.n_quadrants; case F_OPP_LAND: return opp.n_quadrants;
            case F_OWN_HIRES_TODAY: return own.hires_today; case F_OPP_HIRES_TODAY: return opp.hires_today;
            case F_OWN_QUADRANT_UNLOCKED: return arg0>=0 && arg0<4 ? arg0<own.n_quadrants : 0;
            case F_OPP_QUADRANT_UNLOCKED: return arg0>=0 && arg0<4 ? arg0<opp.n_quadrants : 0;
            case F_MARKET_PRICE: return arg0>=0 && arg0<kag::N_PRODUCTS ? s.market.prices[arg0] : 0;
            case F_MARKET_INVENTORY: return arg0>=0 && arg0<kag::N_PRODUCTS ? s.market.inventory[arg0] : 0;
            case F_OWN_SHED: return arg0>=0 && arg0<kag::N_ITEMS ? own.shed[arg0] : 0;
            case F_OWN_SHED_TOTAL: return own.shed_total;
            case F_OWN_SEED: return arg0>=0 && arg0<kag::N_CROPS ? own.seeds[arg0] : 0;
            case F_OWN_INVENTORY: return arg0>=0 && arg0<own.n_units && arg1>=0 && arg1<kag::N_ITEMS ? own.inv[arg0][arg1] : 0;
            case F_SHOP_COUNT: return s.n_shops;
            case F_SHOP_AT: return arg0>=0 && arg0<s.n_shops ? s.shops[arg0] : -1;
            case F_SHOP_SEEN: { int n=0; for(int i=0;i<s.n_shops;++i) n += s.shops[i]==arg0; return n; }
            case F_OWN_WORKER_X: return arg0>=0&&arg0<own.n_units ? own.pos_x[arg0] : -1;
            case F_OWN_WORKER_Y: return arg0>=0&&arg0<own.n_units ? own.pos_y[arg0] : -1;
            case F_OPP_WORKER_X: return arg0>=0&&arg0<opp.n_units ? opp.pos_x[arg0] : -1;
            case F_OPP_WORKER_Y: return arg0>=0&&arg0<opp.n_units ? opp.pos_y[arg0] : -1;
            case F_OWN_WORKER_TILE: return arg0>=0&&arg0<own.n_units ? own.pos_y[arg0]*BOARD_SIZE+own.pos_x[arg0] : -1;
            case F_OPP_WORKER_TILE: return arg0>=0&&arg0<opp.n_units ? opp.pos_y[arg0]*BOARD_SIZE+opp.pos_x[arg0] : -1;
            default: break;
        }
        const bool tile_feature = (id>=F_OWN_TILE_KIND&&id<=F_OPP_TILE_FERT_AVAILABLE) ||
            (id>=F_OWN_TILE_PLANTED_DAY&&id<=F_OPP_TILE_HAS_ANIMAL);
        if (tile_feature) {
            if (arg0 < 0 || arg0 >= BOARD_SIZE*BOARD_SIZE) return 0;
            const int y=arg0/BOARD_SIZE,x=arg0%BOARD_SIZE;
            const bool own_side = id==F_OWN_TILE_KIND||id==F_OWN_TILE_ITEM||id==F_OWN_TILE_YIELD||id==F_OWN_TILE_WATERED||id==F_OWN_TILE_FED||id==F_OWN_TILE_CARED||id==F_OWN_TILE_FERT_AVAILABLE||
                id==F_OWN_TILE_PLANTED_DAY||id==F_OWN_TILE_CONSECUTIVE_UNWATERED||id==F_OWN_TILE_MAX_LIFESPAN_STEP||id==F_OWN_TILE_FERTILIZED_UNTIL_DAY||id==F_OWN_TILE_PLACED_DAY||id==F_OWN_TILE_CONSECUTIVE_UNFED||id==F_OWN_TILE_PENDING_CARE_BONUS||id==F_OWN_TILE_HAS_ANIMAL;
            const kag::Tile& t = (own_side?own:opp).tiles[y][x];
            switch(id){
                case F_OWN_TILE_KIND: case F_OPP_TILE_KIND: return t.kind;
                case F_OWN_TILE_ITEM: case F_OPP_TILE_ITEM: return t.what;
                case F_OWN_TILE_YIELD: case F_OPP_TILE_YIELD: return t.yield_units;
                case F_OWN_TILE_WATERED: case F_OPP_TILE_WATERED: return t.watered_today;
                case F_OWN_TILE_FED: case F_OPP_TILE_FED: return t.fed_today;
                case F_OWN_TILE_CARED: case F_OPP_TILE_CARED: return t.cared_today;
                case F_OWN_TILE_FERT_AVAILABLE: case F_OPP_TILE_FERT_AVAILABLE: return t.fertilizer_available;
                case F_OWN_TILE_PLANTED_DAY: case F_OPP_TILE_PLANTED_DAY: return t.kind==kag::T_PLANT ? t.planted_day : 0;
                case F_OWN_TILE_CONSECUTIVE_UNWATERED: case F_OPP_TILE_CONSECUTIVE_UNWATERED: return t.kind==kag::T_PLANT ? t.consecutive_dry : 0;
                case F_OWN_TILE_MAX_LIFESPAN_STEP: case F_OPP_TILE_MAX_LIFESPAN_STEP: return t.kind==kag::T_PLANT ? t.max_lifespan_step : -1;
                case F_OWN_TILE_FERTILIZED_UNTIL_DAY: case F_OPP_TILE_FERTILIZED_UNTIL_DAY: return t.kind==kag::T_PLANT ? t.fertilized_until_day : -1;
                case F_OWN_TILE_PLACED_DAY: case F_OPP_TILE_PLACED_DAY: return (t.kind==kag::T_COOP||t.kind==kag::T_PASTURE) ? t.planted_day : 0;
                case F_OWN_TILE_CONSECUTIVE_UNFED: case F_OPP_TILE_CONSECUTIVE_UNFED: return (t.kind==kag::T_COOP||t.kind==kag::T_PASTURE) ? t.consecutive_dry : 0;
                case F_OWN_TILE_PENDING_CARE_BONUS: case F_OPP_TILE_PENDING_CARE_BONUS: return (t.kind==kag::T_COOP||t.kind==kag::T_PASTURE) ? t.pending_care_bonus : 0;
                case F_OWN_TILE_HAS_ANIMAL: case F_OPP_TILE_HAS_ANIMAL: return (t.kind==kag::T_COOP||t.kind==kag::T_PASTURE) ? t.has_animal : 0;
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
