#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include <cstdint>

namespace bonsaikg_test {

#ifdef __CUDACC__
#define BKG_TEST_HD __host__ __device__ inline
#else
#define BKG_TEST_HD inline
#endif

BKG_TEST_HD uint64_t mix64(uint64_t h, uint32_t x){ h ^= static_cast<uint64_t>(x); return h * 1099511628211ull; }

template<class Obs>
BKG_TEST_HD uint64_t hash_feature_registry(const Obs& obs){
    uint64_t h=1469598103934665603ull;
    auto add=[&](uint16_t fid,int32_t arg){h=mix64(h,fid);h=mix64(h,static_cast<uint32_t>(arg));h=mix64(h,static_cast<uint32_t>(obs.feature(fid,arg)));};
    const uint16_t plain[]={
        bonsaikg::F_ALWAYS,bonsaikg::F_STEP,bonsaikg::F_DAY,bonsaikg::F_HOUR,bonsaikg::F_OWN_MONEY,bonsaikg::F_OPP_MONEY,
        bonsaikg::F_OWN_UNITS,bonsaikg::F_OPP_UNITS,bonsaikg::F_OWN_LAND,bonsaikg::F_OPP_LAND,bonsaikg::F_OWN_HIRES_TODAY,
        bonsaikg::F_OWN_SHED_TOTAL,bonsaikg::F_SHOP_COUNT
    };
    for(uint16_t fid:plain)add(fid,0);
    for(int i=0;i<9;++i){add(bonsaikg::F_MARKET_PRICE,i);add(bonsaikg::F_MARKET_INVENTORY,i);}
    for(int i=0;i<12;++i)add(bonsaikg::F_OWN_SHED,i);
    for(int i=0;i<5;++i)add(bonsaikg::F_OWN_SEED,i);
    for(int u=0;u<40;++u)for(int item=0;item<12;++item)add(bonsaikg::F_OWN_INVENTORY,(u<<8)|item);
    for(int i=0;i<8;++i){add(bonsaikg::F_SHOP_AT,i);add(bonsaikg::F_SHOP_SEEN,i);}
    for(int u=0;u<40;++u){add(bonsaikg::F_OWN_WORKER_X,u);add(bonsaikg::F_OWN_WORKER_Y,u);add(bonsaikg::F_OPP_WORKER_X,u);add(bonsaikg::F_OPP_WORKER_Y,u);}
    for(uint16_t fid=bonsaikg::F_OWN_TILE_KIND;fid<=bonsaikg::F_OPP_TILE_FERT_AVAILABLE;++fid)
        for(int tile=0;tile<100;++tile)add(fid,tile);
    return h;
}

#undef BKG_TEST_HD

} // namespace bonsaikg_test
