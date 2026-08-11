#include "kag_agent/c_api.h"

#include <algorithm>

#include "kag_agent/controller.hpp"
#include "kag_agent/market.hpp"

namespace {
kag::Controller g_controller;

kag::KagFrame convert(const KagFrameC& c) {
    kag::KagFrame f;
    f.step = c.step;
    f.player = c.player;
    f.turns_per_day = c.turns_per_day;
    f.town_shop_sell_interval = c.town_shop_sell_interval;
    f.town_center_sell_interval = c.town_center_sell_interval;
    f.own_money = c.own_money;
    f.opp_money = c.opp_money;
    f.own_quadrants = c.own_quadrants;
    f.opp_quadrants = c.opp_quadrants;
    f.own_hands = c.own_hands;
    f.opp_hands = c.opp_hands;
    std::copy_n(c.own_crops, 5, f.own_crops.begin());
    std::copy_n(c.opp_crops, 5, f.opp_crops.begin());
    std::copy_n(c.own_animals, 3, f.own_animals.begin());
    std::copy_n(c.opp_animals, 3, f.opp_animals.begin());
    std::copy_n(c.shed, 9, f.shed.begin());
    std::copy_n(c.seeds, 5, f.seeds.begin());
    std::copy_n(c.market_inventory, 9, f.market_inventory.begin());
    std::copy_n(c.market_prices, 9, f.market_prices.begin());
    std::copy_n(c.shop_counts, 8, f.shop_counts.begin());
    for (int h = 0; h <= kag::kLookahead; ++h) {
        std::copy_n(c.planned_sell[h], 9, f.planned_sell[h].begin());
        f.planned_market_slots[h] = c.planned_market_slots[h];
    }
    return f;
}

KagDecisionC convert(const kag::KagDecision& d) {
    return KagDecisionC{d.regime, d.route_id, d.route_confidence, d.v27_similarity,
                        d.preempt_item, d.preempt_qty, d.preempt_horizon,
                        d.market_mode, d.estimated_edge};
}
}  // namespace

extern "C" int kag_agent_abi_version(void) { return 1; }
extern "C" void kag_agent_reset(int32_t seat) { g_controller.reset(seat); }
extern "C" KagDecisionC kag_agent_decide(const KagFrameC* frame) {
    if (!frame) return KagDecisionC{};
    return convert(g_controller.decide(convert(*frame)));
}
extern "C" int32_t kag_agent_market_price(int32_t product, int32_t inventory) {
    if (product < 0 || product >= kag::kProducts) return 1;
    return kag::market_price(static_cast<kag::Product>(product), inventory);
}
