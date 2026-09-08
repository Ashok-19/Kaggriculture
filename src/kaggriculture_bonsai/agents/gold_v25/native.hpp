// SPDX-License-Identifier: Apache-2.0
// Translation of the attributed gold-v25 chassis; see PROVENANCE.md and overlay.py.
#pragma once
#include "bonsai_kg_v4.hpp"
#include "route_pack.hpp"

#ifdef __CUDACC__
#define BKG4_VM_HD __host__ __device__ inline
#else
#define BKG4_VM_HD inline
#endif

namespace gold_v25 {
using namespace bonsaikg4;
constexpr int ITEMS = 12, PRODUCTS = 9, UNITS = 40, QUEUE = 719;
struct Unit { int op=0, arg=0, n=1, arity=1; };
struct Order { int op=0, item=0, n=0; };
struct Tile { int kind=0, animal=0, watered=0, yield=0, fed=0, fertilizer=0, cared=0; };
struct Worker { int x=0, y=0, inv_count=0; int keys[ITEMS]{}; int inv[ITEMS]{}; Tile tile{}; };
struct Observation {
    int step=0, player=0, board=10, n_units=1, yarn=0, egg=10000;
    int shed[ITEMS]{}, seeds[ITEMS]{}, prices[ITEMS]{};
    Worker workers[UNITS]{};
    BKG4_VM_HD int feature(int id, int a0=0, int a1=0) const {
        (void)a0; (void)a1;
        if(id==F_STEP)return step;
        if(id==F_OWN_UNITS)return n_units;
        if(id==F_SHOP_SEEN)return yarn;
        if(id==F_MARKET_INVENTORY)return egg;
        return 0;
    }
};
struct Action { int n_units=0, n_market=0; Unit units[UNITS]{}; Order market[10]{}; };
struct Pending { int x=0, y=0; Unit unit{}; };
struct State {
    int initialized=0, last_step=-1, bb[3]{}, due=-1, suppress[ITEMS]{};
    RuntimeMeta4 meta{};
    int counts[UNITS]{};
    Pending pending[UNITS][QUEUE]{};
};
struct Result {
    Action action{};
    int last_step=-1, route=0, day6=0, day27=0, due=-1, suppress[ITEMS]{}, fault=0;
    int pending_counts[UNITS]{};
    uint64_t pending_hash=0;
};
struct PackView { const bonsaikg_v6::PackedAction* records; const uint16_t* arity; };
BKG4_VM_HD int low(int a,int b){return a<b?a:b;}
BKG4_VM_HD int high(int a,int b){return a>b?a:b;}
BKG4_VM_HD bool move(int op){return op>=U_NORTH&&op<=U_WEST;}
BKG4_VM_HD bool adjacent(const Worker&w,int board){int h=board/2;return (w.x==h-1||w.x==h)&&(w.y==h-1||w.y==h);}
BKG4_VM_HD Unit unpack(bonsaikg_v6::PackedUnit u,bool short_form){
    Unit out{u.op,u.arg,u.n,1};
    if(u.op==U_PLANT)out.arity=2;
    if(u.op==U_PICKUP||u.op==U_PLACE)out.arity=short_form?2:3;
    return out;
}
BKG4_VM_HD Action tape(PackView pack,int route,int step,int count){
    Action out{};out.n_units=count;
    if(step<0||step>=719)return out;
    const int index=route*719+step;
    const auto&a=pack.records[index];const uint16_t short_form=pack.arity[index];
    out.units[0]=unpack(a.farmer,short_form&1);
    for(int i=1;i<count;++i)if(i<=a.saved_hands)out.units[i]=unpack(a.hands[i-1],short_form&(1<<i));
    out.n_market=a.n_market;
    for(int i=0;i<a.n_market;++i)out.market[i]={a.market[i].op,a.market[i].item,a.market[i].n};
    return out;
}
BKG4_VM_HD bool noop(Unit u,const Worker&w,const Observation&o){
    const Tile&t=w.tile;const bool adj=adjacent(w,o.board), animal=t.animal!=0;
    if(move(u.op)){
        int x=w.x+(u.op==U_EAST)-(u.op==U_WEST),y=w.y+(u.op==U_SOUTH)-(u.op==U_NORTH);
        return !(x>=0&&y>=0&&x<o.board&&y<o.board);
    }
    switch(u.op){
    case U_PASS:return true;
    case U_DROP:return !adj||w.inv_count==0;
    case U_PICKUP:return !adj;
    case U_PLACE:
        if(u.arg>=9&&((u.arg==9&&t.kind==3)||(u.arg>=10&&t.kind==4))&&!animal)return w.inv[u.arg]<=0;
        return !adj||w.inv[u.arg]<=0;
    default:break;
    }
    if(t.kind==-1)return true;
    switch(u.op){
    case U_PLANT:return t.kind!=0||o.seeds[u.arg]<=0;
    case U_WATER:return t.kind!=2||t.watered;
    case U_HARVEST:return t.kind<=0||t.yield<=0;
    case U_FERTILIZE:return t.kind!=2||w.inv[8]<=0;
    case U_DIG:return t.kind==0||animal;
    case U_BUILD_COOP:case U_BUILD_PASTURE:return t.kind!=0;
    case U_FEED:return !animal||t.fed||w.inv[0]<=0;
    case U_COLLECT_FERTILIZER:return !animal||!t.fertilizer;
    case U_CARE:return !animal||t.cared;
    default:return true;
    }
}
BKG4_VM_HD void append(State&s,int i,Pending p){
    if(s.counts[i]>=QUEUE){s.meta.fault=100;return;}
    s.pending[i][s.counts[i]++]=p;
}
BKG4_VM_HD void weeds(Action&a,const Observation&o,State&s,PackView pack,int route){
    Action next=tape(pack,route,o.step+1,o.n_units);
    for(int i=0;i<o.n_units;++i){
        const Worker&w=o.workers[i];Unit act=a.units[i];
        if(s.counts[i]&&(s.pending[i][0].x!=w.x||s.pending[i][0].y!=w.y))s.counts[i]=0;
        const bool weed=w.tile.kind==1, idle=noop(act,w,o);
        if((act.op==U_PLANT||act.op==U_BUILD_COOP||act.op==U_BUILD_PASTURE)&&weed){
            append(s,i,{w.x,w.y,act});act=Unit{U_DIG,0,1,1};
        }else if(s.counts[i]&&idle){
            Unit replay=s.pending[i][0].unit;
            if(replay.op==U_PLANT&&move(next.units[i].op))s.counts[i]=0;
            else{
                --s.counts[i];for(int j=0;j<s.counts[i];++j)s.pending[i][j]=s.pending[i][j+1];
                if(act.op!=U_PASS&&!move(act.op))append(s,i,{w.x,w.y,act});
                act=replay;
            }
        }else if(weed&&idle)act=Unit{U_DIG,0,1,1};
        a.units[i]=act;
    }
}
BKG4_VM_HD void project(const Action&a,const Observation&o,int*proj){
    int total=0;for(int k=0;k<ITEMS;++k){proj[k]=o.shed[k];total+=proj[k];}
    for(int i=0;i<o.n_units;++i){
        const Worker&w=o.workers[i];if(!adjacent(w,o.board))continue;
        const Unit u=a.units[i];
        if(u.op==U_PICKUP){int qty=low(proj[u.arg],high(0,u.n));proj[u.arg]-=qty;total-=qty;}
        else if(u.op==U_DROP){
            for(int j=0;j<w.inv_count;++j){int k=w.keys[j];int take=low(high(0,w.inv[k]),high(0,100-total));
                if(take>0){proj[k]+=take;total+=take;}}
        }else if(u.op==U_PLACE&&u.arg<9){
            int take=low(low(high(0,u.n),high(0,w.inv[u.arg])),high(0,100-total));
            if(take>0){proj[u.arg]+=take;total+=take;}
        }
    }
}
BKG4_VM_HD void sales(Action&a,const Observation&o,State&s,PackView pack,int route){
    if(s.due==o.step){
        int remaining[ITEMS];for(int k=0;k<ITEMS;++k)remaining[k]=s.suppress[k];
        for(int i=0;i<a.n_market;++i){auto&ord=a.market[i];
            if(ord.op==M_SELL&&remaining[ord.item]>0){int remove=low(high(0,ord.n),remaining[ord.item]);ord.n-=remove;remaining[ord.item]-=remove;}}
    }
    int projected[ITEMS];project(a,o,projected);
    s.due=-1;for(int k=0;k<ITEMS;++k)s.suppress[k]=0;
    int nxt=o.step+1;if(nxt>718||nxt%72==0||o.step%4==0)return;
    Action future=tape(pack,route,nxt,1);int planned[ITEMS]{};bool already[ITEMS]{};
    for(int i=0;i<future.n_market;++i){auto ord=future.market[i];if(ord.op==M_SELL&&ord.item<PRODUCTS)planned[ord.item]+=high(0,ord.n);}
    for(int i=0;i<a.n_market;++i)if(a.market[i].op==M_SELL)already[a.market[i].item]=true;
    for(int k=1;k<8;++k){
        if(planned[k]<=0||already[k])continue;
        int qty=low(projected[k],planned[k]);if(qty<=0||o.prices[k]<2)continue;
        if(a.n_market>=10)break;
        a.market[a.n_market++]={M_SELL,k,qty};projected[k]-=qty;s.suppress[k]+=qty;s.due=nxt;
    }
}
BKG4_VM_HD uint64_t hash_word(uint64_t h,int v){
    uint32_t u=static_cast<uint32_t>(v);
    for(int j=0;j<4;++j){h^=(u>>(8*j))&255;h*=1099511628211ULL;}return h;
}
BKG4_VM_HD uint64_t pending_hash(const State&s){
    uint64_t h=14695981039346656037ULL;
    for(int i=0;i<UNITS;++i){h=hash_word(h,s.counts[i]);for(int j=0;j<s.counts[i];++j){
        const auto&p=s.pending[i][j];h=hash_word(h,p.x);h=hash_word(h,p.y);h=hash_word(h,p.unit.op);
        h=hash_word(h,p.unit.arg);h=hash_word(h,p.unit.n);h=hash_word(h,p.unit.arity);}}
    return h;
}
BKG4_VM_HD void tick(const PolicyView4&policy,PackView pack,const Observation&o,State&s,Result&out){
    RuntimeView4 runtime{s.bb,nullptr,&s.meta};
    if(!s.initialized||o.step==0||o.step<=s.last_step){
        reset_runtime(policy,runtime);s.initialized=1;s.due=-1;
        for(int k=0;k<ITEMS;++k)s.suppress[k]=0;
        for(int i=0;i<UNITS;++i)s.counts[i]=0;
    }
    s.last_step=o.step;ActionResult4 ignored{};
    tick4(policy,runtime,o,0,ignored);
    const int route=s.bb[0];Action a=tape(pack,route,o.step,o.n_units);
    weeds(a,o,s,pack,route);sales(a,o,s,pack,route);
    if(o.step>=718){
        a=Action{};a.n_units=o.n_units;
        for(int i=0;i<o.n_units;++i)if(adjacent(o.workers[i],o.board)&&o.workers[i].inv_count)a.units[i]=Unit{U_DROP,0,1,1};
        int projected[ITEMS];project(a,o,projected);
        for(int k=0;k<PRODUCTS;++k)if(projected[k]>0)a.market[a.n_market++]={M_SELL,k,projected[k]};
        // Stable descending value, retaining PRODUCTS order on ties.
        for(int i=1;i<a.n_market;++i){Order v=a.market[i];int j=i;
            int64_t value=static_cast<int64_t>(o.prices[v.item])*v.n;
            while(j>0&&static_cast<int64_t>(o.prices[a.market[j-1].item])*a.market[j-1].n<value){a.market[j]=a.market[j-1];--j;}a.market[j]=v;}
    }
    out.action=a;out.last_step=s.last_step;out.route=route;out.day6=s.bb[1];out.day27=s.bb[2];out.due=s.due;
    for(int k=0;k<ITEMS;++k)out.suppress[k]=s.suppress[k];
    for(int i=0;i<UNITS;++i)out.pending_counts[i]=s.counts[i];
    out.fault=s.meta.fault;out.pending_hash=pending_hash(s);
}
} // namespace gold_v25
#undef BKG4_VM_HD
