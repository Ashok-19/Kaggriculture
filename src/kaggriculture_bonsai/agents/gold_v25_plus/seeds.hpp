// Derived own-route seed requirements; both possible late routes are protected.
#pragma once
#include "feed.hpp"
#ifdef __CUDACC__
#define GOLD_SEED_HD __host__ __device__ inline
#else
#define GOLD_SEED_HD inline
#endif
namespace gold_v25_plus {
struct SeedState {int route=-1;int need[649][5]{};};
GOLD_SEED_HD void before_seed_step(gold_v25::PackView pack,int route,int step,int*need){
    auto a=gold_v25::tape(pack,route,step,40);int plants[5]{},buys[5]{};
    for(int i=0;i<a.n_units;++i)if(a.units[i].op==bonsaikg4::U_PLANT&&a.units[i].arg<5)++plants[a.units[i].arg];
    for(int i=0;i<a.n_market;++i)if(a.market[i].op==3&&a.market[i].item<5)buys[a.market[i].item]+=gold_v25::high(0,a.market[i].n);
    for(int k=0;k<5;++k)need[k]=plants[k]+gold_v25::high(0,need[k]-buys[k]);
}
GOLD_SEED_HD void seeds(gold_v25::Action&a,const gold_v25::Observation&o,gold_v25::PackView pack,int route,SeedState&s){
    if(o.step<144||o.step>=647)return;
    if(s.route!=route){
        int tail2[5]{},tail3[5]{};
        for(int t=718;t>=648;--t){before_seed_step(pack,2,t,tail2);before_seed_step(pack,3,t,tail3);}
        for(int k=0;k<5;++k)s.need[648][k]=gold_v25::high(tail2[k],tail3[k]);
        for(int t=647;t>=0;--t){for(int k=0;k<5;++k)s.need[t][k]=s.need[t+1][k];before_seed_step(pack,route,t,s.need[t]);}
        s.route=route;
    }
    int available[5];bool planted[100]{};
    for(int k=0;k<5;++k)available[k]=o.seeds[k];
    for(int i=0;i<a.n_units;++i){auto u=a.units[i];auto w=o.workers[i];int pos=w.y*10+w.x;
        if(u.op==bonsaikg4::U_PLANT&&u.arg<5&&w.tile.kind==0&&!planted[pos]&&available[u.arg]>0){--available[u.arg];planted[pos]=true;}}
    for(int i=0;i<a.n_market;++i){auto&order=a.market[i];if(order.op!=3||order.item>=5)continue;
        order.n=gold_v25::low(order.n,gold_v25::high(0,s.need[o.step+1][order.item]-available[order.item]));
        available[order.item]+=gold_v25::high(0,order.n);
    }
}
}
#undef GOLD_SEED_HD
