#pragma once

#include "../include/bonsai_kg.hpp"
#include "kag_adapter.cuh"

namespace bonsaikg {

struct CudaPolicyState {
    RuntimeState runtime{};
};

struct CudaPolicy {
    PolicyView tree{};
    int max_market_orders = 10;

    __device__ inline void reset(CudaPolicyState& state, const kagcuda::ObservationView&) const {
        reset_runtime(tree, state.runtime);
    }

    __device__ inline void act(CudaPolicyState& state, const kagcuda::ObservationView& observation,
                               kagcuda::Action& out) const {
        const CudaObservation legal(observation);
        ActionResult result;
        tick(tree, state.runtime, legal, max_market_orders, result, nullptr);
        to_cuda_action(result, out);
    }
};

} // namespace bonsaikg
