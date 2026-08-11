#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KagFrameC {
    int32_t step;
    int32_t player;
    int32_t turns_per_day;
    int32_t town_shop_sell_interval;
    int32_t town_center_sell_interval;
    double own_money;
    double opp_money;
    int32_t own_quadrants;
    int32_t opp_quadrants;
    int32_t own_hands;
    int32_t opp_hands;
    int32_t own_crops[5];
    int32_t opp_crops[5];
    int32_t own_animals[3];
    int32_t opp_animals[3];
    int32_t shed[9];
    int32_t seeds[5];
    int32_t market_inventory[9];
    int32_t market_prices[9];
    int32_t shop_counts[8];
    int32_t planned_sell[5][9];
    int32_t planned_market_slots[5];
} KagFrameC;

typedef struct KagDecisionC {
    int32_t regime;
    int32_t route_id;
    double route_confidence;
    double v27_similarity;
    int32_t preempt_item;
    int32_t preempt_qty;
    int32_t preempt_horizon;
    int32_t market_mode;
    double estimated_edge;
} KagDecisionC;

int kag_agent_abi_version(void);
void kag_agent_reset(int32_t seat);
KagDecisionC kag_agent_decide(const KagFrameC* frame);
int32_t kag_agent_market_price(int32_t product, int32_t inventory);

#ifdef __cplusplus
}
#endif
