#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include <stdint.h>

#ifdef __CUDACC__
#define BKG_TEST_HD __host__ __device__ inline
#else
#define BKG_TEST_HD inline
#endif

namespace bonsaikg_test {

BKG_TEST_HD uint64_t mix64(uint64_t h,uint64_t x){h^=x+0x9e3779b97f4a7c15ull+(h<<6)+(h>>2);return h;}

template<class Obs>
BKG_TEST_HD uint64_t hash_feature_registry(const Obs&obs){
    uint64_t h=0x6a09e667f3bcc909ull;
    auto add=[&](uint16_t fid,int32_t a0=0,int32_t a1=0){h=mix64(h,fid);h=mix64(h,(uint32_t)a0);h=mix64(h,(uint32_t)a1);h=mix64(h,(uint32_t)obs.feature(fid,a0,a1));};
    const uint16_t plain[]={bonsaikg::F_ALWAYS,bonsaikg::F_STEP,bonsaikg::F_DAY,bonsaikg::F_HOUR,bonsaikg::F_OWN_MONEY,bonsaikg::F_OPP_MONEY,bonsaikg::F_OWN_UNITS,bonsaikg::F_OPP_UNITS,bonsaikg::F_OWN_LAND,bonsaikg::F_OPP_LAND,bonsaikg::F_OWN_HIRES_TODAY,bonsaikg::F_OPP_HIRES_TODAY,bonsaikg::F_OWN_SHED_TOTAL,bonsaikg::F_SHOP_COUNT};
    for(uint16_t fid:plain)add(fid);
    for(int q=0;q<4;++q){add(bonsaikg::F_OWN_QUADRANT_UNLOCKED,q);add(bonsaikg::F_OPP_QUADRANT_UNLOCKED,q);}
    for(int i=0;i<9;++i){add(bonsaikg::F_MARKET_PRICE,i);add(bonsaikg::F_MARKET_INVENTORY,i);}
    for(int i=0;i<12;++i)add(bonsaikg::F_OWN_SHED,i);
    for(int i=0;i<5;++i)add(bonsaikg::F_OWN_SEED,i);
    for(int u=0;u<40;++u)for(int item=0;item<12;++item)add(bonsaikg::F_OWN_INVENTORY,u,item);
    for(int i=0;i<8;++i){add(bonsaikg::F_SHOP_AT,i);add(bonsaikg::F_SHOP_SEEN,i);}
    for(int u=0;u<40;++u){add(bonsaikg::F_OWN_WORKER_X,u);add(bonsaikg::F_OWN_WORKER_Y,u);add(bonsaikg::F_OPP_WORKER_X,u);add(bonsaikg::F_OPP_WORKER_Y,u);add(bonsaikg::F_OWN_WORKER_TILE,u);add(bonsaikg::F_OPP_WORKER_TILE,u);}
    for(uint16_t fid=bonsaikg::F_OWN_TILE_KIND;fid<=bonsaikg::F_OPP_TILE_FERT_AVAILABLE;++fid)for(int tile=0;tile<100;++tile)add(fid,tile);
    for(uint16_t fid=bonsaikg::F_OWN_TILE_PLANTED_DAY;fid<=bonsaikg::F_OPP_TILE_HAS_ANIMAL;++fid)for(int tile=0;tile<100;++tile)add(fid,tile);
    return h;
}

constexpr int TARGETED_FEATURE_COUNT=38;

template<class Obs>
BKG_TEST_HD void targeted_features(const Obs&obs,int32_t*out){
    int k=0;
    out[k++]=obs.feature(bonsaikg::F_OWN_HIRES_TODAY);out[k++]=obs.feature(bonsaikg::F_OPP_HIRES_TODAY);
    for(int q=0;q<4;++q){out[k++]=obs.feature(bonsaikg::F_OWN_QUADRANT_UNLOCKED,q);out[k++]=obs.feature(bonsaikg::F_OPP_QUADRANT_UNLOCKED,q);}
    out[k++]=obs.feature(bonsaikg::F_OWN_WORKER_TILE,0);out[k++]=obs.feature(bonsaikg::F_OPP_WORKER_TILE,0);out[k++]=obs.feature(bonsaikg::F_OWN_WORKER_TILE,1);out[k++]=obs.feature(bonsaikg::F_OPP_WORKER_TILE,1);
    const int plant=44,animal=43;
    out[k++]=obs.feature(bonsaikg::F_OWN_TILE_PLANTED_DAY,plant);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_CONSECUTIVE_UNWATERED,plant);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_MAX_LIFESPAN_STEP,plant);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_FERTILIZED_UNTIL_DAY,plant);
    out[k++]=obs.feature(bonsaikg::F_OWN_TILE_HAS_ANIMAL,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_PLACED_DAY,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_CONSECUTIVE_UNFED,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_PENDING_CARE_BONUS,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_FED,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_CARED,animal);out[k++]=obs.feature(bonsaikg::F_OWN_TILE_FERT_AVAILABLE,animal);
    out[k++]=obs.feature(bonsaikg::F_OPP_TILE_PLANTED_DAY,plant);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_CONSECUTIVE_UNWATERED,plant);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_MAX_LIFESPAN_STEP,plant);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_FERTILIZED_UNTIL_DAY,plant);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_HAS_ANIMAL,animal);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_PLACED_DAY,animal);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_CONSECUTIVE_UNFED,animal);out[k++]=obs.feature(bonsaikg::F_OPP_TILE_PENDING_CARE_BONUS,animal);
    // Explicit acting-player private canaries. These are legal only for OWN_*.
    out[k++]=obs.feature(bonsaikg::F_OWN_SHED,0);out[k++]=obs.feature(bonsaikg::F_OWN_SHED,8);out[k++]=obs.feature(bonsaikg::F_OWN_SEED,0);out[k++]=obs.feature(bonsaikg::F_OWN_INVENTORY,0,0);out[k++]=obs.feature(bonsaikg::F_OWN_INVENTORY,1,0);
}

#undef BKG_TEST_HD
} // namespace bonsaikg_test
