#include "kag_agent/controller.hpp"

#include <algorithm>
#include <cmath>

#include "kag_agent/market.hpp"
#include "kag_agent/router.hpp"

namespace kag {

Controller::Controller() {
    reset(0);
    reset(1);
}

void Controller::reset(int seat) {
    if (seat < 0 || seat > 1) return;
    state_[seat] = ControllerState{};
}

ControllerState Controller::state(int seat) const {
    return state_.at(std::clamp(seat, 0, 1));
}

void Controller::update_residuals(const KagFrame& f, ControllerState& s) {
    if (!s.initialized || f.step <= 0 || f.step <= s.last_step) {
        s.previous_market_inventory = f.market_inventory;
        s.initialized = true;
        s.last_step = f.step;
        return;
    }
    for (int p = 0; p < kProducts; ++p) {
        const int observed_delta = f.market_inventory[p] - s.previous_market_inventory[p];
        // Positive delta is net selling pressure. We deliberately do not claim
        // this is pure opponent flow; EMA is only a routing signal.
        const double residual = static_cast<double>(observed_delta);
        s.opponent_flow_ema[p] = 0.82 * s.opponent_flow_ema[p] + 0.18 * residual;
    }
    s.previous_market_inventory = f.market_inventory;
    s.last_step = f.step;
}

double Controller::update_v27_similarity(const KagFrame& f, ControllerState& s) const {
    // Identity-free public/economic fingerprint. v27-like openings are highly
    // concentrated around 1 cow / 4 sheep, 5 wheat + 5 melon, and 3 quadrants.
    double score = 1.0;
    if (f.step >= 24) {
        score -= 0.18 * std::abs(f.opp_animals[COW] - 1);
        score -= 0.10 * std::abs(f.opp_animals[SHEEP] - 4);
        score -= 0.05 * std::abs(f.opp_crops[C_WHEAT] - 5);
        score -= 0.05 * std::abs(f.opp_crops[C_MELON] - 5);
    }
    if (f.step >= 180) {
        score -= 0.10 * std::max(0, std::abs(f.opp_quadrants - 2));
    }
    if (f.step >= 300) {
        score -= 0.08 * std::max(0, std::abs(f.opp_quadrants - 3));
    }
    // Strong early wheat-market pressure is a known non-v27 family signal.
    if (f.step <= 192 && s.opponent_flow_ema[WHEAT] < -2.0) score -= 0.22;
    score = std::clamp(score, 0.0, 1.0);
    s.v27_similarity = 0.70 * s.v27_similarity + 0.30 * score;
    return s.v27_similarity;
}

KagDecision Controller::choose_market_action(const KagFrame& f, ControllerState& s, KagDecision d) const {
    // Only products that are not inputs to the current strong route are allowed
    // to move across turns in the first controller generation.
    constexpr std::array<bool, kProducts> safe{{false, true, true, true, true, true, true, true, false}};

    double best_edge = 0.0;
    int best_item = -1;
    int best_qty = 0;
    int best_h = 0;

    for (int h = 1; h <= kLookahead; ++h) {
        if (f.planned_market_slots[0] >= 10) break;
        for (int pi = 0; pi < kProducts; ++pi) {
            if (!safe[pi]) continue;
            const int planned = f.planned_sell[h][pi];
            if (planned <= 0) continue;
            const int current_planned = f.planned_sell[0][pi];
            const int available = std::max(0, f.shed[pi] - current_planned);
            const int qty = std::min(planned, available);
            if (qty <= 0) continue;

            const auto item = static_cast<Product>(pi);
            const int inv_now = f.market_inventory[pi];
            const int demand = town_demand_between(f, item, f.step, f.step + h);
            const int opp_pressure = std::max(0, static_cast<int>(std::llround(s.opponent_flow_ema[pi] * h)));
            const int inv_wait = inv_now - demand + opp_pressure;
            const double now_rev = sell_revenue(item, inv_now, qty);
            const double wait_rev = sell_revenue(item, inv_wait, qty);

            // If the opponent is v27-like, the same future route sale is a
            // plausible opponent hazard. Our earlier supply reduces their quote;
            // estimate that denial value in addition to our own timing value.
            const double opp_harm = s.v27_similarity * std::max(0.0,
                sell_revenue(item, inv_wait, planned) - sell_revenue(item, inv_wait + qty, planned));
            const double edge = now_rev - wait_rev + 0.85 * opp_harm;
            if (edge > best_edge + 0.5) {
                best_edge = edge;
                best_item = pi;
                best_qty = qty;
                best_h = h;
            }
        }
    }

    if (best_item >= 0 && best_edge > 1.0) {
        d.preempt_item = best_item;
        d.preempt_qty = best_qty;
        d.preempt_horizon = best_h;
        d.market_mode = s.v27_similarity >= 0.72 ? 2 : 1;
        d.estimated_edge = best_edge;
    }
    return d;
}

KagDecision Controller::decide(const KagFrame& f) {
    const int seat = std::clamp(f.player, 0, 1);
    auto& s = state_[seat];
    if (f.step == 0 || (s.initialized && f.step < s.last_step)) reset(seat);
    update_residuals(f, s);
    const double similarity = update_v27_similarity(f, s);
    const auto scores = score_routes(f, similarity);

    KagDecision d;
    d.regime = scores.regime;
    d.v27_similarity = similarity;
    d.route_id = ROUTE_V27;
    double best = scores.score[0];
    double second = -1e9;
    for (int i = 1; i < ROUTE_COUNT; ++i) {
        if (scores.score[i] > best) {
            second = best;
            best = scores.score[i];
            d.route_id = i;
        } else {
            second = std::max(second, scores.score[i]);
        }
    }
    d.route_confidence = std::clamp((best - second) / 1.5, 0.0, 1.0);
    return choose_market_action(f, s, d);
}

}  // namespace kag
