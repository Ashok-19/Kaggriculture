#include "kag_agent/router.hpp"

#include <algorithm>

#include "kag_agent/market.hpp"

namespace kag {

RouteScores score_routes(const KagFrame& f, double v27_similarity) {
    RouteScores out;
    const int wool_demand = town_demand_per_day(f, WOOL);
    const int milk_demand = town_demand_per_day(f, MILK);
    const int wheat_inv = f.market_inventory[WHEAT];
    const double wheat_pressure = std::max(0, 10000 - wheat_inv) + std::max(0, f.opp_crops[C_WHEAT] - f.own_crops[C_WHEAT]) * 4.0;
    const double capital_gap = f.opp_money - f.own_money;

    if (wool_demand >= milk_demand + 6 && f.step >= 72) out.regime = WOOL_HEAVY;
    else if (milk_demand >= wool_demand + 6 && f.step >= 72) out.regime = MILK_HEAVY;
    else if (wheat_pressure >= 18.0 && f.step <= 192) out.regime = WHEAT_PRESSURE;
    else if (capital_gap > 8000.0 && f.step > 240) out.regime = DEFENSIVE;
    else out.regime = GENERIC;

    out.score[ROUTE_V27] = 1.2 + 1.4 * v27_similarity;
    out.score[ROUTE_LEV] = 1.0 + 0.015 * milk_demand + 0.2 * (out.regime == MILK_HEAVY);
    out.score[ROUTE_JINCE] = 1.0 + 0.012 * wool_demand + 0.18 * (out.regime == WOOL_HEAVY);
    out.score[ROUTE_COUNTER] = 0.8 + 0.025 * wheat_pressure + 0.35 * (out.regime == WHEAT_PRESSURE);
    return out;
}

}  // namespace kag
