#pragma once

#include "kag_agent/types.hpp"

namespace kag {

struct RouteScores {
    std::array<double, ROUTE_COUNT> score{};
    Regime regime = GENERIC;
};

RouteScores score_routes(const KagFrame& frame, double v27_similarity);

}  // namespace kag
