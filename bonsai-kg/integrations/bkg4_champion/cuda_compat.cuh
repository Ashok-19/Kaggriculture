#pragma once

// Search/runtime compatibility layer for BKG4_BASE_0. This file deliberately does
// not alter BKG4 ABI 4. The BKG4 controller is semantically equivalent to the frozen
// BKG3-v6 controller; CUDA search reuses that already-proven controller implementation
// and applies the exact frozen public-state compatibility/exploit overlays afterwards.

#include "../stateful_v6/cuda_runtime.cuh"
#include "../stateful_v6/engine_exploits.cuh"

namespace bonsaikg4_champion {

struct CudaLegacyState {
    uint8_t trigger_decided = 0;
    uint8_t h4_trigger = 0;
    int8_t first_shop = -1;
    uint8_t r86 = 0;
    uint8_t ownweed168 = 0;
    uint8_t r253 = 0;
    uint8_t fy144 = 0;
    uint8_t ice86 = 0;
    uint8_t ice_v32 = 0;
    uint8_t ff168 = 0;
    uint8_t ff_low = 0;
    uint8_t external_ok = 0;
    uint16_t external_route = 0xffffu;
};

struct CudaChampionState {
    bonsaikg_v6::CudaCompatState base{};
    CudaLegacyState legacy{};
    bonsaikg_v6::CudaEngineExploitState exploit{};
};

struct CudaLegacyPack {
    const bonsaikg_v6::RouteDesc* routes = nullptr;
    const bonsaikg_v6::PackedAction* actions = nullptr;
    int route_count = 0;

    __device__ __forceinline__ const bonsaikg_v6::PackedAction* route_at(uint16_t route_id, int step) const {
        for (int i=0;i<route_count;++i) if (routes[i].route_id == route_id) {
            int s=step; if(s<0)s=0; if(s>=routes[i].steps)s=routes[i].steps-1;
            return actions + routes[i].first + s;
        }
        return nullptr;
    }
};

__device__ __forceinline__ void reset_legacy(CudaLegacyState& s) {
    s = CudaLegacyState{};
}

__device__ __forceinline__ int shop_at(const kagcuda::ObservationView& obs, int i) {
    const auto town=obs.town(); return i>=0 && i<town.n_shops() ? town.shop(i) : -1;
}

__device__ __forceinline__ bool shop_prefix4(const kagcuda::ObservationView& obs,int a,int b,int c,int d) {
    return shop_at(obs,0)==a && shop_at(obs,1)==b && shop_at(obs,2)==c && shop_at(obs,3)==d;
}

__device__ __forceinline__ bool weed_at(const kagcuda::PublicFarmView& farm,int x,int y) {
    return farm.tile(x,y).kind == kagcuda::T_WEED;
}

__device__ __forceinline__ bool materialize_legacy(const CudaLegacyPack& pack,uint16_t route,int step,int n_units,kagcuda::Action& out) {
    const auto* p=pack.route_at(route,step); if(!p)return false;
    bonsaikg_v6::materialize_cuda(*p,n_units,out); return true;
}

__device__ __forceinline__ void apply_legacy(const kagcuda::ObservationView& obs,
                                              const CudaLegacyPack& pack,
                                              CudaLegacyState& s,
                                              kagcuda::Action& action) {
    const int step=obs.step(), me=obs.player(), opp=1-me;
    if(step==0){reset_legacy(s);return;}
    const auto own=obs.farm(me), other=obs.farm(opp);

    if(!s.trigger_decided){
        s.h4_trigger=(uint8_t)(other.n_units()==5 && other.money()<=32.0); // farmer + four hands
        s.trigger_decided=1;
    }

    if(s.h4_trigger){
        if(s.first_shop<0 && obs.town().n_shops()>0)s.first_shop=(int8_t)shop_at(obs,0);
        if(step==86){
            s.r86=(uint8_t)weed_at(other,0,2);
            s.ice86=(uint8_t)(s.first_shop==3 && weed_at(own,4,2)); // ICE_CREAM_SHOP
        }
        if(step==144)s.fy144=(uint8_t)(shop_at(obs,0)==2 && shop_at(obs,1)==7); // FARMERS_MARKET,YARN
        if(step==168){
            s.ownweed168=(uint8_t)weed_at(own,6,3);
            s.ff168=(uint8_t)(shop_at(obs,0)==2 && shop_at(obs,1)==2 && weed_at(own,8,0));
        }
        if(step==253){
            s.r253=(uint8_t)(s.ownweed168 && s.first_shop==5 && other.money()<2200.0); // PIZZA
            if(s.ice86)s.ice_v32=(uint8_t)(other.money()>=2100.0);
            if(s.ff168)s.ff_low=(uint8_t)(other.money()<1700.0);
        }
        uint16_t route=0xffffu;
        if(s.ice86 && step>=86 && step<96) route=202;
        else if(s.ice86 && step>=96) route=(s.ice_v32 && step>=253)?206:205;
        else if(s.ff168 && step>=168) route=(s.ff_low && step>=253)?201:207;
        else if(s.r86 && step>=86 && step<96) route=202;
        else if(s.fy144 && step>=144) route=204;
        else if(s.r253 && step>=253) route=203;
        else if(step<72) route=200;
        else route=s.first_shop==5?200:201;
        if(route!=0xffffu)materialize_legacy(pack,route,step,own.n_units(),action);
    }

    if(step==25)s.external_ok=(uint8_t)!s.h4_trigger;
    if(step==288 && s.external_ok){
        if(shop_prefix4(obs,2,4,7,7))s.external_route=208;       // FM,PET,YARN,YARN
        else if(shop_prefix4(obs,2,0,5,3))s.external_route=209;  // FM,BAKERY,PIZZA,ICE
        else if(shop_prefix4(obs,1,5,0,0))s.external_route=210;  // BRUNCH,PIZZA,BAKERY,BAKERY
        else if(shop_prefix4(obs,0,2,1,4))s.external_route=209;  // BAKERY,FM,BRUNCH,PET
        else s.external_route=0xffffu;
    }
    if(s.external_route!=0xffffu && step>=288)
        materialize_legacy(pack,s.external_route,step,own.n_units(),action);
}

struct CudaChampionPolicy {
    bonsaikg_v6::CudaCompatPolicy base{};
    CudaLegacyPack legacy{};

    __device__ __forceinline__ void reset(CudaChampionState& s,const kagcuda::ObservationView& obs) const {
        base.reset(s.base,obs);reset_legacy(s.legacy);bonsaikg_v6::reset_cuda_engine_exploits(s.exploit);
    }

    __device__ __forceinline__ uint16_t selected_route(const CudaChampionState& s) const {
        const int x=s.base.controller.blackboard[1]; return x<0?0:(uint16_t)x;
    }

    __device__ __forceinline__ bool materialize_base_route(uint16_t route,int step,int n_units,kagcuda::Action& out) const {
        const auto* p=base.route_at(route,step);if(!p)return false;bonsaikg_v6::materialize_cuda(*p,n_units,out);return true;
    }

    // module_route==0xffff preserves the controller-selected route. The searched
    // BKG4 module executes after the frozen controller but before compatibility
    // overlays, exactly matching how build_assets.py appends SetBlackboard rules.
    __device__ __forceinline__ void act(CudaChampionState& s,const kagcuda::ObservationView& obs,
                                         uint16_t module_route,kagcuda::Action& out) const {
        base.act(s.base,obs,out);
        if(module_route!=0xffffu){
            const int n=obs.farm(obs.player()).n_units();
            if(!materialize_base_route(module_route,obs.step(),n,out))s.base.compat_fault=0x201u;
            else s.base.controller.blackboard[1]=(int32_t)module_route;
        }
        apply_legacy(obs,legacy,s.legacy,out);
        bonsaikg_v6::apply_cuda_engine_exploits(obs,out,s.exploit,s.legacy.external_ok!=0,100);
    }
};

} // namespace bonsaikg4_champion
