// Own scheduled pickup coverage; no opponent identity or hidden state.
#pragma once
#include "lead.hpp"
#ifdef __CUDACC__
#define GOLD_FEED_HD __host__ __device__ inline
#else
#define GOLD_FEED_HD inline
#endif
namespace gold_v25_plus {
struct RepairState {int last=-1;int targets[100]{};int count=0;gold_v25::Unit job[5]{};};
GOLD_FEED_HD void dispatch(gold_v25::Action&a,const gold_v25::Observation&o,
        gold_v25::PackView pack,int route,const gold_v25::State&parent,RepairState&s){
    if(o.step==0||o.step<=s.last){for(int i=0;i<100;++i)s.targets[i]=0;s.count=0;}
    s.last=o.step;
    for(int i=0;i<o.n_units;++i)for(int j=0;j<parent.counts[i];++j){auto p=parent.pending[i][j];
        if(p.unit.op==bonsaikg4::U_BUILD_PASTURE||p.unit.op==bonsaikg4::U_BUILD_COOP)s.targets[p.y*10+p.x]=p.unit.op;}
    for(int i=0;i<o.n_units;++i){auto w=o.workers[i];if(w.tile.kind==3||w.tile.kind==4)s.targets[w.y*10+w.x]=0;}
    if(!s.count&&a.units[0].op==bonsaikg4::U_PASS){
        auto w=o.workers[0];
        for(int target=0;target<100;++target)if(s.targets[target]){
            int tx=target%10,ty=target/10,dx=tx-w.x,dy=ty-w.y;
            int distance=(dx<0?-dx:dx)+(dy<0?-dy:dy),length=distance*2+1;
            if(distance<=0||distance>2||o.step+length>718)continue;
            bool idle=true;
            for(int t=o.step;t<o.step+length;++t)if(gold_v25::tape(pack,route,t,1).units[0].op!=bonsaikg4::U_PASS)idle=false;
            if(!idle)continue;
            int n=0;
            for(int k=0;k<(dx<0?-dx:dx);++k)s.job[n++]={dx>0?bonsaikg4::U_EAST:bonsaikg4::U_WEST,0,1,1};
            for(int k=0;k<(dy<0?-dy:dy);++k)s.job[n++]={dy>0?bonsaikg4::U_SOUTH:bonsaikg4::U_NORTH,0,1,1};
            s.job[n++]={s.targets[target],0,1,1};
            for(int k=distance-1;k>=0;--k){int op=s.job[k].op;
                int opposite=op==bonsaikg4::U_EAST?bonsaikg4::U_WEST:op==bonsaikg4::U_WEST?bonsaikg4::U_EAST:op==bonsaikg4::U_NORTH?bonsaikg4::U_SOUTH:bonsaikg4::U_NORTH;
                s.job[n++]={opposite,0,1,1};}
            s.count=n;s.targets[target]=0;break;
        }
    }
    if(s.count){a.units[0]=s.job[0];--s.count;for(int i=0;i<s.count;++i)s.job[i]=s.job[i+1];}
}
GOLD_FEED_HD void structure(gold_v25::Action&a,const gold_v25::Observation&o){
    for(int i=0;i<o.n_units;++i){auto&w=o.workers[i];auto u=a.units[i];
        for(int k=10;k<=12;++k){int animal=k==12?9:k;
            if(w.inv[animal]<=0)continue;
            int kind=animal==9?3:4;
            if(u.op==bonsaikg4::U_PLACE&&u.arg==animal&&w.tile.kind==0){
                a.units[i]={animal==9?bonsaikg4::U_BUILD_COOP:bonsaikg4::U_BUILD_PASTURE,0,1,1};break;}
            if(w.tile.kind==kind&&!w.tile.animal&&gold_v25::noop(u,w,o)){
                a.units[i]={bonsaikg4::U_PLACE,animal,1,2};break;}
        }
    }
}
GOLD_FEED_HD void feed(gold_v25::Action&a,const gold_v25::Observation&o,
                      gold_v25::PackView pack,int route,int start,int stop){
    if(o.step<start||o.step>=stop||a.n_market>=10)return;
    auto next=gold_v25::tape(pack,route,o.step+1,40);
    int demand=0;
    for(int i=0;i<next.n_units;++i)if(next.units[i].op==bonsaikg4::U_PICKUP&&next.units[i].arg==0)
        demand+=gold_v25::high(0,next.units[i].n);
    int projected[12];gold_v25::project(a,o,projected);int available=projected[0];
    for(int i=0;i<a.n_market;++i){auto order=a.market[i];if(order.item!=0)continue;
        if(order.op==4)available+=gold_v25::high(0,order.n);
        else if(order.op==6)available-=gold_v25::low(available,gold_v25::high(0,order.n));}
    int missing=gold_v25::high(0,demand-available);
    if(missing)a.market[a.n_market++]={4,0,missing};
}
}
#undef GOLD_FEED_HD
