#pragma once
#include "../include/bonsai_kg_v4_abi.hpp"
#include "kag_adapter.cuh"

namespace bonsaikg4 {
static_assert((int)F_ALWAYS==(int)bonsaikg::F_ALWAYS && (int)F_STEP==(int)bonsaikg::F_STEP &&
              (int)F_OWN_INVENTORY==(int)bonsaikg::F_OWN_INVENTORY &&
              (int)F_OPP_TILE_HAS_ANIMAL==(int)bonsaikg::F_OPP_TILE_HAS_ANIMAL,
              "BKG4 feature-registry-v1 ids must match frozen BKG3 legal adapter");
class CudaObservation4 {
    bonsaikg::CudaObservation inner_;
public:
    __device__ __forceinline__ explicit CudaObservation4(kagcuda::ObservationView o):inner_(o){}
    __device__ __forceinline__ int32_t feature(uint16_t id,int32_t a0=0,int32_t a1=0)const{return inner_.feature(id,a0,a1);}
};
__device__ __forceinline__ void to_cuda_action4(const ActionResult4&src,kagcuda::Action&out){
    out.n_units=src.n_units;for(int i=0;i<src.n_units&&i<kagcuda::MAX_UNITS;++i){out.units[i].op=src.units[i].op;out.units[i].arg=src.units[i].arg;out.units[i].n=src.units[i].n;}
    out.n_orders=src.n_market;for(int i=0;i<src.n_market&&i<kagcuda::MAX_ORDERS;++i){out.orders[i].op=src.market[i].op;out.orders[i].item=src.market[i].item;out.orders[i].n=src.market[i].n;}
}
} // namespace bonsaikg4
