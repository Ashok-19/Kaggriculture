#pragma once

#include "kag_agent/types.hpp"

namespace kag {

double shape_value(int shape_id, double x);
int market_price(Product item, int inventory);
int town_demand_per_day(const KagFrame& frame, Product item);
int town_demand_between(const KagFrame& frame, Product item, int from_step, int to_step);
double sell_revenue(Product item, int start_inventory, int quantity);

}  // namespace kag
