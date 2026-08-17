#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include "../../cuda-engine/include/observation.cuh"
#include <stdint.h>

namespace bonsaikg {

struct CudaObservation {
    kagcuda::ObservationView obs;

    __device__ __forceinline__ explicit CudaObservation(kagcuda::ObservationView observation)
        : obs(observation) {}

    __device__ __forceinline__ int32_t feature(uint16_t id, int32_t arg) const {
        const int player = obs.player();
        const int opponent = 1 - player;
        const auto own = obs.farm(player);
        const auto opp = obs.farm(opponent);

        switch (id) {
            case F_ALWAYS: return 1;
            case F_STEP: return obs.step();
            case F_DAY: return obs.day();
            case F_HOUR: return obs.hour();
            case F_OWN_MONEY: {
                const double x = own.money();
                return x <= -2147483648.0 ? INT32_MIN
                     : x >=  2147483647.0 ? INT32_MAX
                     : static_cast<int32_t>(x);
            }
            case F_OPP_MONEY: {
                const double x = opp.money();
                return x <= -2147483648.0 ? INT32_MIN
                     : x >=  2147483647.0 ? INT32_MAX
                     : static_cast<int32_t>(x);
            }
            case F_OWN_UNITS: return own.n_units();
            case F_OPP_UNITS: return opp.n_units();
            case F_OWN_LAND: return own.n_quadrants();
            case F_OPP_LAND: return opp.n_quadrants();
            case F_OWN_HIRES_TODAY: return own.hires_today();
            case F_MARKET_PRICE:
                return arg >= 0 && arg < kagcuda::N_PRODUCTS ? obs.market().price(arg) : 0;
            case F_MARKET_INVENTORY:
                return arg >= 0 && arg < kagcuda::N_PRODUCTS ? obs.market().inventory(arg) : 0;
            case F_OWN_SHED:
                // private_farm() is the acting player's private state only.
                return arg >= 0 && arg < kagcuda::N_ITEMS ? obs.private_farm().shed(arg) : 0;
            case F_OWN_SHED_TOTAL:
                return obs.private_farm().shed_total();
            case F_OWN_SEED:
                return arg >= 0 && arg < kagcuda::N_CROPS ? obs.private_farm().seed(arg) : 0;
            case F_OWN_INVENTORY: {
                const int unit = (arg >> 8) & 255;
                const int item = arg & 255;
                return unit < own.n_units() && item < kagcuda::N_ITEMS
                    ? obs.private_farm().inventory(unit, item)
                    : 0;
            }
            case F_SHOP_COUNT:
                return obs.town().n_shops();
            case F_SHOP_AT:
                return arg >= 0 && arg < obs.town().n_shops() ? obs.town().shop(arg) : -1;
            case F_SHOP_SEEN: {
                int count = 0;
                for (int i = 0; i < obs.town().n_shops(); ++i) {
                    count += obs.town().shop(i) == arg;
                }
                return count;
            }
            case F_OWN_WORKER_X:
                return arg >= 0 && arg < own.n_units() ? own.worker_x(arg) : -1;
            case F_OWN_WORKER_Y:
                return arg >= 0 && arg < own.n_units() ? own.worker_y(arg) : -1;
            case F_OPP_WORKER_X:
                return arg >= 0 && arg < opp.n_units() ? opp.worker_x(arg) : -1;
            case F_OPP_WORKER_Y:
                return arg >= 0 && arg < opp.n_units() ? opp.worker_y(arg) : -1;
            default:
                break;
        }

        if (id >= F_OWN_TILE_KIND && id <= F_OPP_TILE_FERT_AVAILABLE) {
            if (arg < 0 || arg >= 100) return 0;
            const int y = arg / 10;
            const int x = arg % 10;
            const bool own_side =
                id == F_OWN_TILE_KIND || id == F_OWN_TILE_ITEM || id == F_OWN_TILE_YIELD ||
                id == F_OWN_TILE_WATERED || id == F_OWN_TILE_FED || id == F_OWN_TILE_CARED ||
                id == F_OWN_TILE_FERT_AVAILABLE;
            const auto& tile = (own_side ? own : opp).tile(x, y);
            switch (id) {
                case F_OWN_TILE_KIND:
                case F_OPP_TILE_KIND:
                    return tile.kind;
                case F_OWN_TILE_ITEM:
                case F_OPP_TILE_ITEM:
                    return tile.what;
                case F_OWN_TILE_YIELD:
                case F_OPP_TILE_YIELD:
                    return tile.yield_units;
                case F_OWN_TILE_WATERED:
                case F_OPP_TILE_WATERED:
                    return tile.watered_today;
                case F_OWN_TILE_FED:
                case F_OPP_TILE_FED:
                    return tile.fed_today;
                case F_OWN_TILE_CARED:
                case F_OPP_TILE_CARED:
                    return tile.cared_today;
                case F_OWN_TILE_FERT_AVAILABLE:
                case F_OPP_TILE_FERT_AVAILABLE:
                    return tile.fertilizer_available;
                default:
                    break;
            }
        }
        return 0;
    }
};

__device__ __forceinline__ void to_cuda_action(const ActionResult& src, kagcuda::Action& out) {
    out.n_units = src.n_units;
    for (int i = 0; i < src.n_units && i < kagcuda::MAX_UNITS; ++i) {
        out.units[i].op = src.units[i].op;
        out.units[i].arg = src.units[i].arg;
        out.units[i].n = src.units[i].n;
    }
    out.n_orders = src.n_market;
    for (int i = 0; i < src.n_market && i < kagcuda::MAX_ORDERS; ++i) {
        out.orders[i].op = src.market[i].op;
        out.orders[i].item = src.market[i].item;
        out.orders[i].n = src.market[i].n;
    }
}

} // namespace bonsaikg
