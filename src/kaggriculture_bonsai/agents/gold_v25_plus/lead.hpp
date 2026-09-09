// Bounded own-schedule sale anticipation within the current demand cycle.
#pragma once
#include "../gold_v25/native.hpp"
#ifdef __CUDACC__
#define GOLD_PLUS_HD __host__ __device__ inline
#else
#define GOLD_PLUS_HD inline
#endif
namespace gold_v25_plus {
GOLD_PLUS_HD void lead(gold_v25::Action&a,const gold_v25::Observation&o,
                      gold_v25::PackView pack,int route,int horizon){
    if(o.step%4==0)return;
    for(int i=0;i<a.n_market;++i)if(a.market[i].op!=0&&a.market[i].op!=6)return;
    int planned[12]{},projected[12];bool already[12]{};
    gold_v25::project(a,o,projected);
    for(int i=0;i<a.n_market;++i)if(a.market[i].op==6)already[a.market[i].item]=true;
    for(int dt=1;dt<=horizon;++dt){
        int next=o.step+dt;if(next>718||next%72==0)break;
        auto future=gold_v25::tape(pack,route,next,1);
        for(int i=0;i<future.n_market;++i){auto order=future.market[i];
            if(order.op==6&&order.item>0&&order.item<8)planned[order.item]+=gold_v25::high(0,order.n);}
        if(next%4==0)break;
    }
    for(int k=1;k<8;++k)if(!already[k]&&o.prices[k]>=2&&a.n_market<10){
        int n=gold_v25::low(planned[k],projected[k]);
        if(n>0)a.market[a.n_market++]={6,k,n};
    }
}
}
#undef GOLD_PLUS_HD
