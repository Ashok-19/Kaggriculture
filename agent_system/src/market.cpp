#include "kag_agent/market.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace kag {
namespace {

enum Shape : int { LINEAR = 0, SQ, SQRT, LOG, LOG10 };
struct Params {
    double base;
    int i0;
    double scale;
    Shape below_shape;
    double below_target;
    Shape above_shape;
    double above_target;
};

constexpr std::array<Params, kProducts> kParams{{
    {25, 10000, 400, SQRT, 0.80, LOG, 0.20},
    {35, 10000, 450, LOG, 0.20, SQRT, 0.70},
    {60, 10000, 200, LINEAR, 0.40, SQRT, 0.60},
    {120, 10000, 100, SQRT, 0.70, LINEAR, 1.60},
    {250, 10000, 300, LOG, 0.20, SQ, 3.60},
    {50, 10000, 332, LINEAR, 0.40, LOG, 0.20},
    {160, 10000, 122, SQRT, 0.60, LINEAR, 1.60},
    {200, 10000, 105, LOG, 0.20, SQ, 3.20},
    {100, 10000, 200, LINEAR, 0.40, LINEAR, 0.40},
}};

constexpr std::array<std::array<int, kProducts>, kShops> kShopProducts{{
    // BAKERY
    {1,0,0,0,0,1,0,0,0},
    // PIZZA_SHOP
    {1,0,1,0,0,0,1,0,0},
    // BRUNCH_SPOT
    {1,0,0,1,0,1,0,0,0},
    // YARN_STORE (2x single-product demand)
    {0,0,0,0,0,0,0,2,0},
    // ICE_CREAM_SHOP
    {1,0,0,1,0,0,1,0,0},
    // PET_CAFE (2x)
    {0,2,0,0,0,0,0,0,0},
    // SMOOTHIE_SHOP
    {0,0,0,1,0,0,1,0,0},
    // FARMERS_MARKET
    {1,1,1,1,0,0,0,0,0},
}};

}  // namespace

double shape_value(int shape_id, double x) {
    x = std::max(0.0, x);
    switch (shape_id) {
        case LINEAR: return x;
        case SQ: return x * x;
        case SQRT: return std::sqrt(x);
        case LOG: return std::log1p(x);
        case LOG10: return std::log10(1.0 + x);
        default: return x;
    }
}

int market_price(Product item, int inventory) {
    const auto& p = kParams.at(static_cast<int>(item));
    double value = p.base;
    if (inventory < p.i0) {
        const double amp = p.below_target * p.base / shape_value(p.below_shape, p.scale);
        value = p.base + amp * shape_value(p.below_shape, p.i0 - inventory);
    } else {
        const double amp = p.above_target * p.base / shape_value(p.above_shape, p.scale);
        value = p.base - amp * shape_value(p.above_shape, inventory - p.i0);
    }
    return std::max(1, static_cast<int>(std::llround(value)));
}

int town_demand_per_day(const KagFrame& frame, Product item) {
    const int ticks = std::max(1, frame.turns_per_day) / std::max(1, frame.town_shop_sell_interval);
    int demand = 0;
    for (int s = 0; s < kShops; ++s) {
        demand += frame.shop_counts[s] * ticks * kShopProducts[s][static_cast<int>(item)];
    }
    if (item != FERTILIZER) {
        demand += std::max(1, frame.turns_per_day) / std::max(1, frame.town_center_sell_interval);
    }
    return demand;
}

int town_demand_between(const KagFrame& frame, Product item, int from_step, int to_step) {
    if (to_step <= from_step) return 0;
    int demand = 0;
    const int shop_interval = std::max(1, frame.town_shop_sell_interval);
    const int center_interval = std::max(1, frame.town_center_sell_interval);
    for (int step = from_step + 1; step <= to_step; ++step) {
        if (step % shop_interval == 0) {
            for (int s = 0; s < kShops; ++s) {
                demand += frame.shop_counts[s] * kShopProducts[s][static_cast<int>(item)];
            }
        }
        if (item != FERTILIZER && step % center_interval == 0) ++demand;
    }
    return demand;
}

double sell_revenue(Product item, int start_inventory, int quantity) {
    double revenue = 0.0;
    int inv = start_inventory;
    for (int i = 0; i < quantity; ++i) {
        const int p = market_price(item, inv);
        revenue += p;
        if (p > 1) ++inv;
    }
    return revenue;
}

}  // namespace kag
