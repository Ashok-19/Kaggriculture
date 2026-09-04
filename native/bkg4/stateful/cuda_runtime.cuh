#pragma once

#include "route_pack.hpp"
#include "kag_adapter.cuh"
#include "bonsai_kg.hpp"
#include "sim.cuh"

namespace bonsaikg_v6 {

struct CudaCompatState {
    bonsaikg::RuntimeState controller{};
    int last_step = -1;
    uint16_t compat_fault = 0;
    uint16_t pad = 0;
};

__device__ __forceinline__ kagcuda::UnitAction to_cuda_unit(const PackedUnit& p) {
    kagcuda::UnitAction out{}; out.op=p.op; out.arg=p.arg; out.n=p.n; return out;
}
__device__ __forceinline__ kagcuda::Order to_cuda_order(const PackedOrder& p) {
    kagcuda::Order out{}; out.op=p.op; out.item=p.item; out.n=p.n; return out;
}
__device__ __forceinline__ void materialize_cuda(const PackedAction& src, int n_units, kagcuda::Action& out) {
    if(n_units<1)n_units=1;
    if(n_units>kagcuda::MAX_UNITS)n_units=kagcuda::MAX_UNITS;
    out=kagcuda::Action{}; out.n_units=n_units; out.units[0]=to_cuda_unit(src.farmer);
    const int hands=n_units-1;
    for(int i=0;i<hands;++i) out.units[i+1]=i<src.saved_hands?to_cuda_unit(src.hands[i]):kagcuda::UnitAction{};
    out.n_orders=src.n_market;
    for(int i=0;i<out.n_orders;++i) out.orders[i]=to_cuda_order(src.market[i]);
}

struct CudaCompatPolicy {
    bonsaikg::PolicyView tree{};
    const RouteDesc* routes = nullptr;
    const PackedAction* actions = nullptr;
    int route_count = 0;

    __device__ __forceinline__ void reset(CudaCompatState& s, const kagcuda::ObservationView&) const {
        bonsaikg::reset_runtime(tree,s.controller); s.last_step=-1; s.compat_fault=0;
    }
    __device__ __forceinline__ const PackedAction* route_at(uint16_t route_id,int step) const {
        for(int i=0;i<route_count;++i) if(routes[i].route_id==route_id) {
            int s=step; if(s<0)s=0; if(s>=routes[i].steps)s=routes[i].steps-1;
            return actions+routes[i].first+s;
        }
        return nullptr;
    }
    __device__ __forceinline__ void act(CudaCompatState& s,const kagcuda::ObservationView& observation,kagcuda::Action& out) const {
        const int step=observation.step();
        if(step==0||step<=s.last_step){bonsaikg::reset_runtime(tree,s.controller);s.compat_fault=0;}
        s.last_step=step;
        if(s.compat_fault||s.controller.fault){out=kagcuda::Action{};out.n_units=observation.farm(observation.player()).n_units();return;}
        const bonsaikg::CudaObservation legal(observation);bonsaikg::ActionResult ignored{};
        const bonsaikg::Status st=bonsaikg::tick(tree,s.controller,legal,0,ignored,nullptr);
        if(s.controller.fault||st!=bonsaikg::SUCCESS){s.compat_fault=s.controller.fault?s.controller.fault:0x100u;out=kagcuda::Action{};out.n_units=observation.farm(observation.player()).n_units();return;}
        const PackedAction* src=route_at(static_cast<uint16_t>(s.controller.blackboard[1]),step);
        if(!src){s.compat_fault=0x101u;out=kagcuda::Action{};out.n_units=observation.farm(observation.player()).n_units();return;}
        materialize_cuda(*src,observation.farm(observation.player()).n_units(),out);
    }
};

} // namespace bonsaikg_v6
