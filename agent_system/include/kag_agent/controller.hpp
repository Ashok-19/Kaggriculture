#pragma once

#include <array>

#include "kag_agent/types.hpp"

namespace kag {

struct ControllerState {
    bool initialized = false;
    int last_step = -1;
    std::array<int, kProducts> previous_market_inventory{};
    std::array<double, kProducts> opponent_flow_ema{};
    std::array<int, kProducts> sale_debt{};
    double v27_similarity = 1.0;
};

class Controller {
public:
    Controller();
    void reset(int seat);
    KagDecision decide(const KagFrame& frame);
    ControllerState state(int seat) const;

private:
    std::array<ControllerState, 2> state_{};
    void update_residuals(const KagFrame& frame, ControllerState& state);
    double update_v27_similarity(const KagFrame& frame, ControllerState& state) const;
    KagDecision choose_market_action(const KagFrame& frame, ControllerState& state, KagDecision base) const;
};

}  // namespace kag
