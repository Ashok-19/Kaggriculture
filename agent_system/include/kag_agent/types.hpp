#pragma once

#include <array>
#include <cstdint>

namespace kag {

constexpr int kProducts = 9;
constexpr int kCrops = 5;
constexpr int kAnimals = 3;
constexpr int kShops = 8;
constexpr int kLookahead = 4;

enum Product : int {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER
};

enum Crop : int { C_WHEAT = 0, C_CARROT, C_TOMATO, C_STRAWBERRY, C_MELON };
enum Animal : int { GOOSE = 0, COW, SHEEP };
enum Regime : int { GENERIC = 0, WOOL_HEAVY, MILK_HEAVY, WHEAT_PRESSURE, SCARCITY, DEFENSIVE };

enum RouteId : int { ROUTE_V27 = 0, ROUTE_LEV, ROUTE_JINCE, ROUTE_COUNTER, ROUTE_COUNT };

struct KagFrame {
    std::int32_t step = 0;
    std::int32_t player = 0;
    std::int32_t turns_per_day = 24;
    std::int32_t town_shop_sell_interval = 4;
    std::int32_t town_center_sell_interval = 24;
    double own_money = 0.0;
    double opp_money = 0.0;
    std::int32_t own_quadrants = 1;
    std::int32_t opp_quadrants = 1;
    std::int32_t own_hands = 0;
    std::int32_t opp_hands = 0;
    std::array<std::int32_t, kCrops> own_crops{};
    std::array<std::int32_t, kCrops> opp_crops{};
    std::array<std::int32_t, kAnimals> own_animals{};
    std::array<std::int32_t, kAnimals> opp_animals{};
    std::array<std::int32_t, kProducts> shed{};
    std::array<std::int32_t, kCrops> seeds{};
    std::array<std::int32_t, kProducts> market_inventory{};
    std::array<std::int32_t, kProducts> market_prices{};
    std::array<std::int32_t, kShops> shop_counts{};
    // Planned sells for current step and the next kLookahead steps.
    std::array<std::array<std::int32_t, kProducts>, kLookahead + 1> planned_sell{};
    std::array<std::int32_t, kLookahead + 1> planned_market_slots{};
};

struct KagDecision {
    std::int32_t regime = GENERIC;
    std::int32_t route_id = ROUTE_V27;
    double route_confidence = 0.0;
    double v27_similarity = 0.0;
    std::int32_t preempt_item = -1;
    std::int32_t preempt_qty = 0;
    std::int32_t preempt_horizon = 0;
    std::int32_t market_mode = 0;
    double estimated_edge = 0.0;
};

}  // namespace kag
