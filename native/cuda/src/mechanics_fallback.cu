// Bounded executor/mechanics fallback evaluator. The frozen Shop Plus controller
// and route pack are reused byte-identically; a candidate is only a small
// deterministic overlay applied to the action the frozen executor already
// produced for the searching seat. An all-zero overlay is the reference.
#include "../../../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "extended_route_pack.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#ifdef __CUDACC__
#include "runtime.cuh"
namespace engine = kagcuda;
#define HD __device__
#else
#include "sim.hpp"
namespace engine = kag;
#define HD inline
#endif
namespace parent = shop_router_0909;
using namespace bonsaikg4;

// member 0 Shop Plus, 1 direct Shop controller, 2/3 route-3 and route-12
// continuation opponents built from broadcast packs. All remain live executors.
struct Case { uint64_t seed; int seat, member; };
// cap_mode  0 off, 1 value-preserving partial deposit
// cap_floor minimum units at risk before the deposit rule activates
// cap_room  pre-emptive sale target room (0 off)
// cap_look  lookahead steps used to confirm an upcoming deposit
// feed_mode 0 off, 1 wheat-sale clamp, 2 clamp + pickup top-up, 3 top-up only
// feed_res  wheat units held back beyond forecast demand
// feed_look forecast horizon in steps
struct Overlay { int cap_mode, cap_floor, cap_room, cap_look, feed_mode, feed_res, feed_look; };
struct Diag { int deposits=0, saved=0, saved_value=0, presales=0, presold=0, clamps=0, clamped=0, topups=0; };
struct Row {
    double money[2]{};
    uint64_t simulation=0, controller=0, actions=0;
    int fault=0, turns=0, route=0;
    int produced[12]{}, purchases=0, hires=0, discarded=0, discard_value=0, unsold=0, failed=0, empty_pens=0;
    Diag d{};
};
struct Hash {
    uint64_t value=14695981039346656037ULL;
    HD void add(int64_t x){uint64_t u=(uint64_t)x;for(int j=0;j<8;++j){value^=(u>>(j*8))&255;value*=1099511628211ULL;}}
    HD void real(double d){union{double d;uint64_t u;}b;b.d=d;add(b.u);}
};
HD void hash_state(Hash& h,const engine::Sim& sim){
    const auto&s=sim.st;const auto&c=sim.cfg;
    h.add(c.episode_steps);h.add(c.board_size);h.add(c.starting_money);h.add(c.max_orders);
    h.add(c.turns_per_day);h.add(c.shed_capacity);h.real(c.weed_chance);
    h.add(c.shop_unlock_interval);h.add(c.shop_sell_interval);h.add(c.center_sell_interval);
    h.add(c.hire_mult);h.add(c.seed);
    h.add(s.step);h.add(s.day);h.add(s.hour);h.add(s.done);h.add(s.n_shops);
    for(int i=0;i<s.n_shops;++i)h.add(s.shops[i]);
    for(int k=0;k<9;++k){h.add(s.market.inventory[k]);h.add(s.market.prices[k]);}
    for(int p=0;p<2;++p){const auto&f=s.farms[p];
        h.real(f.money);h.add(f.n_units);h.add(f.n_quadrants);h.add(f.hires_today);h.add(f.next_decay_step);
        h.add(f.shed_total);h.real(f.sell_revenue);h.real(f.total_spend);
        for(int k=0;k<12;++k){h.add(f.shed[k]);h.add(f.discarded[k]);h.add(f.produced[k]);h.add(f.sold_units[k]);}
        for(int k=0;k<5;++k)h.add(f.seeds[k]);
        for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];
            h.add(t.kind);h.add(t.what);h.add(t.has_animal);h.add(t.watered_today);h.add(t.fed_today);
            h.add(t.cared_today);h.add(t.fertilizer_available);h.add(t.consecutive_dry);h.add(t.yield_units);
            h.add(t.pending_care_bonus);h.add(t.planted_day);h.add(t.max_lifespan_step);h.add(t.fertilized_until_day);}
        for(int i=0;i<40;++i){h.add(f.inv_nkeys[i]);
            if(i<f.n_units){h.add(f.pos_x[i]);h.add(f.pos_y[i]);}
            for(int k=0;k<12;++k)h.add(f.inv[i][k]);
            for(int k=0;k<f.inv_nkeys[i];++k)h.add(f.inv_keys[i][k]);}
    }
}
HD void hash_controller(Hash&h,const parent::State&s){
    h.add(s.initialized);h.add(s.last_step);h.add(s.day);h.add(s.bb[0]);h.add(s.bb[1]);h.add(s.due);
    h.add(s.meta.fault);h.add(s.meta.reserved);h.add(s.meta.ticks);
    for(int k=0;k<12;++k)h.add(s.advanced[k]);
    for(int i=0;i<40;++i){h.add(s.counts[i]);h.add(s.heads[i]);
        for(int j=0;j<s.counts[i];++j){const auto&u=s.queue[i][(s.heads[i]+j)%parent::QUEUE];
            h.add(u.op);h.add(u.arg);h.add(u.n);h.add(u.arity);}}
}
HD void hash_action(Hash&h,const parent::Action&a){
    h.add(a.n_units);h.add(a.n_market);
    for(int i=0;i<a.n_units;++i){auto u=a.units[i];h.add(u.op);h.add(u.arg);h.add(u.n);h.add(u.arity);}
    for(int i=0;i<a.n_market;++i){auto o=a.market[i];h.add(o.op);h.add(o.item);h.add(o.n);}
}
HD parent::Observation observe(const engine::Sim&sim,int p){
    parent::Observation o{};o.step=sim.st.step;o.player=p;o.board=sim.cfg.board_size;
    const auto&f=sim.st.farms[p];o.n_positions=f.n_units;
    o.shop0=sim.st.n_shops>0?sim.st.shops[0]:-1;o.shop1=sim.st.n_shops>1?sim.st.shops[1]:-1;
    for(int k=0;k<12;++k){o.shed[k]=f.shed[k];o.shed_present[k]=k<9||f.shed[k]!=0;}
    for(int k=0;k<9;++k)o.prices[k]=sim.st.market.prices[k];
    for(int i=0;i<f.n_units;++i){auto&w=o.workers[i];w.x=f.pos_x[i];w.y=f.pos_y[i];
        w.weed=f.tiles[w.y][w.x].kind==engine::T_WEED;w.inv_count=f.inv_nkeys[i];
        for(int k=0;k<12;++k){w.keys[k]=f.inv_keys[i][k];w.inv[k]=f.inv[i][k];}}
    return o;
}
HD engine::Action convert(const parent::Action&a){
    engine::Action out{};out.n_units=a.n_units;out.n_orders=a.n_market;
    for(int i=0;i<a.n_units;++i)out.units[i]={(uint8_t)a.units[i].op,(uint8_t)a.units[i].arg,(int16_t)a.units[i].n};
    for(int i=0;i<a.n_market;++i)out.orders[i]={(uint8_t)a.market[i].op,(uint8_t)a.market[i].item,a.market[i].n};
    return out;
}
// ---------------------------------------------------------------- overlay ---
// Unit value used only to rank the searching seat's own carried goods. Animals
// are never priced by the market table, so they outrank every product.
HD int worth(const parent::Observation&o,int k){ return k<parent::PRODUCTS?o.prices[k]:1000; }
// Forecast of FEED operations in the frozen tape, over `look` steps, either for
// one lane or for every lane. This reads only the seat's own committed route.
HD int feed_demand(parent::PackView pack,int route,int step,int look,int lane){
    int n=0;
    for(int s=step;s<step+look&&s<=parent::LAST_STEP;++s){
        const parent::Action f=parent::tape(pack,route,s);
        for(int i=0;i<f.n_units&&i<parent::TAPE_UNITS;++i)
            if(f.units[i].op==U_FEED&&(lane<0||lane==i))++n;
    }
    return n;
}
HD bool deposit_ahead(parent::PackView pack,int route,int step,int look){
    for(int s=step+1;s<=step+look&&s<=parent::LAST_STEP;++s){
        const parent::Action f=parent::tape(pack,route,s);
        for(int i=0;i<f.n_units&&i<parent::TAPE_UNITS;++i)
            if(f.units[i].op==U_DROP||f.units[i].op==U_PLACE)return true;
    }
    return false;
}
HD void overlay(parent::Action&a,const parent::Observation&o,parent::PackView pack,int route,Overlay ov,Diag&d){
    if(o.step>=parent::LAST_STEP)return;
    if(!ov.cap_mode&&!ov.cap_room&&!ov.feed_mode)return;
    int shed[parent::ITEMS];int total=0;
    for(int k=0;k<parent::ITEMS;++k){shed[k]=o.shed[k];total+=shed[k];}
    const int n=parent::low(a.n_units,o.n_positions);
    // Family A, capacity-safe deposits. Workers resolve in order, so the running
    // shed total mirrors the engine's own sequencing.
    for(int i=0;i<n;++i){
        parent::Unit&u=a.units[i];
        const parent::Worker&w=o.workers[i];
        const bool near=parent::beside_shed(o,i);
        if(!near)continue;
        if(u.op==U_PICKUP){
            const int q=parent::high(0,u.arity>=3?u.n:1);
            const int take=parent::low(shed[u.arg],q);
            shed[u.arg]-=take;total-=take;continue;
        }
        if(u.op==U_PLACE&&u.arg<parent::PRODUCTS){
            const int q=parent::high(0,u.arity>=3?u.n:1);
            const int add=parent::low(parent::low(q,parent::high(0,w.inv[u.arg])),parent::high(0,parent::SHED_CAPACITY-total));
            shed[u.arg]+=add;total+=add;continue;
        }
        if(u.op!=U_DROP)continue;
        int carried=0;
        for(int j=0;j<w.inv_count;++j)carried+=parent::high(0,w.inv[w.keys[j]]);
        const int room=parent::high(0,parent::SHED_CAPACITY-total);
        const int at_risk=carried-room;
        if(ov.cap_mode&&at_risk>=ov.cap_floor&&at_risk>0){
            // DROP fills the shed in inventory-insertion order and destroys the
            // remainder. Deposit the most valuable carried item instead and keep
            // the rest, which stays legal cargo until room appears.
            // Canonical item order, so the choice never depends on the
            // worker's inventory insertion order.
            int best=-1,best_worth=-1;
            for(int k=0;k<parent::ITEMS;++k){
                if(w.inv[k]<=0)continue;
                const int val=worth(o,k);
                if(val>best_worth){best_worth=val;best=k;}
            }
            if(best>=0){
                int saved_value=0;
                // Value of the goods DROP would have destroyed, in key order.
                int left=room;
                for(int j=0;j<w.inv_count;++j){
                    const int k=w.keys[j];const int amount=parent::high(0,w.inv[k]);
                    const int take=parent::low(amount,parent::high(0,left));
                    left-=take;saved_value+=(amount-take)*worth(o,k);
                }
                const int put=parent::low(parent::high(0,w.inv[best]),room);
                if(put>0){u=parent::Unit{U_PLACE,best,put,3};shed[best]+=put;total+=put;}
                else u=parent::Unit{U_PASS,0,1,1};
                ++d.deposits;d.saved+=at_risk;d.saved_value+=saved_value;
            }
        }else{
            // Unmodified DROP: mirror the engine's key-order fill so the running
            // shed total stays exact for the workers that resolve after it.
            int left=room;
            for(int j=0;j<w.inv_count;++j){
                const int k=w.keys[j];const int amount=parent::high(0,w.inv[k]);
                const int take=parent::low(amount,parent::high(0,left));
                shed[k]+=take;left-=take;total+=take;
            }
        }
    }
    // Family A, pre-emptive capacity sale. Market orders resolve after unit
    // actions, so room bought now serves the next committed deposit.
    if(ov.cap_room>0&&a.n_market<parent::MAX_ORDERS){
        const int room=parent::high(0,parent::SHED_CAPACITY-total);
        if(room<ov.cap_room&&deposit_ahead(pack,route,o.step,ov.cap_look)){
            bool sold[parent::ITEMS]{};
            for(int i=0;i<a.n_market;++i)if(a.market[i].op==M_SELL)sold[a.market[i].item]=true;
            int pick=-1,pick_worth=1<<30;
            for(int k=1;k<parent::PRODUCTS;++k){
                if(k==8||sold[k]||shed[k]<=0||o.prices[k]<2)continue;  // never wheat or fertilizer
                if(o.prices[k]<pick_worth){pick_worth=o.prices[k];pick=k;}
            }
            if(pick>=0){
                const int want=parent::low(ov.cap_room-room,shed[pick]);
                if(want>0){
                    a.market[a.n_market++]={M_SELL,pick,want};
                    ++d.presales;d.presold+=want;
                }
            }
        }
    }
    // Family B, projected feed preservation. FEED consumes one WHEAT from the
    // acting worker's own cargo, so both the stock and its delivery matter.
    if(ov.feed_mode){
        const int demand=feed_demand(pack,route,o.step,ov.feed_look,-1);
        if(ov.feed_mode==1||ov.feed_mode==2){
            int wheat=shed[0];
            for(int i=0;i<o.n_positions&&i<parent::UNITS;++i)wheat+=parent::high(0,o.workers[i].inv[0]);
            const int keep=demand+ov.feed_res;
            for(int i=0;i<a.n_market;++i){
                parent::Order&ord=a.market[i];
                if(ord.op!=M_SELL||ord.item!=0||ord.n<=0)continue;
                const int allowed=parent::high(0,wheat-keep);
                if(ord.n>allowed){
                    // SELL is clamped to the shed by the engine, so only the
                    // units that would really have left the shed are counted.
                    ++d.clamps;
                    d.clamped+=parent::low(ord.n,shed[0])-parent::low(allowed,shed[0]);
                    ord.n=allowed;
                }
                wheat-=ord.n;
            }
        }
        if(ov.feed_mode==2||ov.feed_mode==3){
            for(int i=0;i<n;++i){
                parent::Unit&u=a.units[i];
                if(u.op!=U_PASS||!parent::beside_shed(o,i))continue;
                if(o.workers[i].inv[0]>0||shed[0]<=0)continue;
                const int lane=feed_demand(pack,route,o.step,ov.feed_look,i);
                if(lane<=0)continue;
                const int take=parent::low(shed[0],parent::low(lane,ov.feed_res>0?ov.feed_res:lane));
                if(take<=0)continue;
                u=parent::Unit{U_PICKUP,0,take,3};shed[0]-=take;total-=take;++d.topups;
            }
        }
    }
    if(a.n_market>parent::MAX_ORDERS)a.n_market=parent::MAX_ORDERS;
}
struct Sides {
    PolicyView4 policy[2];
    parent::PackView pack[2];
};
HD void turn(engine::Sim&sim,Sides sides,parent::State*states,Case c,Overlay ov,Row&r,Hash&sh,Hash&ch,Hash&ah,bool audit){
    if(audit)hash_state(sh,sim);
    engine::Action actions[2];
    for(int p=0;p<2;++p){auto o=observe(sim,p);parent::Result result{};
        const int side=(p==c.seat)?0:1;
        if(o.n_positions>parent::TAPE_UNITS){r.fault=201;return;}
        parent::tick(sides.policy[side],sides.pack[side],o,states[p],result);
        if(result.fault)r.fault=result.fault;
        if(p==c.seat){
            overlay(result.action,o,sides.pack[0],result.route,ov,r.d);
            for(int i=0;i<result.action.n_market;++i){auto op=result.action.market[i].op;
                if(op==engine::M_BUY_ANIMAL||op==engine::M_BUY_SEED||op==engine::M_BUY_PRODUCT)++r.purchases;
                if(op==engine::M_HIRE)++r.hires;}
        }
        if(audit){hash_controller(ch,states[p]);hash_action(ah,result.action);}
        actions[p]=convert(result.action);
    }
    sim.step(actions[0],actions[1]);++r.turns;
}
HD Row run(engine::Sim&sim,Sides sides,parent::State*states,Case c,Overlay ov,bool audit){
    Row r{};Hash sh,ch,ah;
    states[0]=parent::State{};states[1]=parent::State{};
    while(!sim.st.done){
        turn(sim,sides,states,c,ov,r,sh,ch,ah,audit);
        if(r.fault)break;
    }
    if(audit){hash_state(sh,sim);hash_controller(ch,states[0]);hash_controller(ch,states[1]);}
    const auto&f=sim.st.farms[c.seat];
    for(int k=0;k<12;++k){r.produced[k]=f.produced[k];r.discarded+=f.discarded[k];
        r.discard_value+=f.discarded[k]*(k<9?sim.st.market.prices[k]:1000);
        if(k<9)r.unsold+=f.shed[k];}
    for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];
        if(t.kind==engine::T_PLANT&&t.yield_units>0)++r.failed;
        if((t.kind==engine::T_COOP||t.kind==engine::T_PASTURE)&&!t.has_animal){++r.failed;++r.empty_pens;}}
    r.money[0]=sim.st.farms[0].money;r.money[1]=sim.st.farms[1].money;
    r.route=states[c.seat].bb[0];
    r.simulation=sh.value;r.controller=ch.value;r.actions=ah.value;return r;
}
#ifdef __CUDACC__
__global__ void evaluate(kagcuda::DeviceTables tables,PolicyView4 plus,PolicyView4 direct,
                         const bonsaikg_v6::PackedAction*base,const uint16_t*base_arity,
                         const bonsaikg_v6::PackedAction*route3,const bonsaikg_v6::PackedAction*route12,
                         parent::State*states,const Case*cases,int n_cases,
                         const Overlay*ovs,int n_candidates,bool audit,Row*rows){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n_cases*n_candidates)return;
    const int ci=id/n_cases, xi=id%n_cases;
    Case c=cases[xi];
    Sides sides{};
    sides.policy[0]=plus;sides.pack[0]={base,base_arity};
    sides.policy[1]=c.member==1?direct:plus;
    sides.pack[1]={c.member==2?route3:c.member==3?route12:base,base_arity};
    engine::Sim sim;engine::Config cfg;cfg.seed=c.seed;sim.init(cfg,&tables);
    rows[id]=run(sim,sides,states+2*id,c,ovs[ci],audit);
}
template<class T>T*upload(const T*data,size_t n){T*out=nullptr;if(!n)return out;
    KAGCUDA_CHECK(cudaMalloc(&out,n*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(out,data,n*sizeof(T),cudaMemcpyHostToDevice));return out;}
PolicyView4 upload_policy(OwnedPolicy4&owned){
    auto v=owned.view();
#define UP(field) v.field=upload(owned.field.data(),owned.field.size())
    UP(nodes);UP(child_refs);UP(bb);UP(values);UP(feature_args);UP(predicates);UP(lifecycles);UP(effects);
#undef UP
    return v;
}
#endif
static std::vector<bonsaikg_v6::PackedAction> broadcast(const std::vector<bonsaikg_v6::PackedAction>&base,int route){
    std::vector<bonsaikg_v6::PackedAction> out(base.size());
    for(int r=0;r<parent::ROUTES;++r)
        for(int s=0;s<parent::STEPS;++s)out[r*parent::STEPS+s]=base[route*parent::STEPS+s];
    return out;
}
int main(int argc,char**argv){
    if(argc!=7){std::cerr<<"plus_controller direct_controller routes arity cases candidates\n";return 2;}
    auto owned=load_compiled4(argv[1]);auto plus=owned.view();
    if(plus.n_bb!=2||plus.n_state!=0){std::cerr<<"unsupported parent state layout\n";return 2;}
    auto owned_direct=load_compiled4(argv[2]);auto direct=owned_direct.view();
    auto pack=bonsaikg4_champion::load_extended_route_pack(argv[3]);
    std::vector<uint16_t>arity(parent::ROUTES*parent::STEPS);
    {std::ifstream af(argv[4],std::ios::binary);af.read((char*)arity.data(),arity.size()*2);if(!af)return 2;}
    std::vector<Case>cases;Case c{};{std::ifstream cf(argv[5]);while(cf>>c.seed>>c.seat>>c.member)cases.push_back(c);}
    std::vector<Overlay>ovs;Overlay v{};
    {std::ifstream vf(argv[6]);while(vf>>v.cap_mode>>v.cap_floor>>v.cap_room>>v.cap_look>>v.feed_mode>>v.feed_res>>v.feed_look)ovs.push_back(v);}
    if(cases.empty()||ovs.empty())return 2;
    const bool audit=true;
    const auto route3=broadcast(pack.actions,3),route12=broadcast(pack.actions,12);
    const size_t total=cases.size()*ovs.size();
    std::vector<Row>rows(total);
    std::cerr<<"threads="<<total<<"\n";
#ifdef __CUDACC__
    kagcuda::configure_device();kagcuda::TablesOwner tables;
    plus=upload_policy(owned);direct=upload_policy(owned_direct);
    auto dev_base=upload(pack.actions.data(),pack.actions.size());
    auto dev_base_arity=upload(arity.data(),arity.size());
    auto dev_r3=upload(route3.data(),route3.size()),dev_r12=upload(route12.data(),route12.size());
    auto dev_cases=upload(cases.data(),cases.size());
    auto dev_ovs=upload(ovs.data(),ovs.size());
    Row*dev_rows=nullptr;parent::State*states=nullptr;
    KAGCUDA_CHECK(cudaMalloc(&dev_rows,total*sizeof(Row)));
    KAGCUDA_CHECK(cudaMalloc(&states,total*2*sizeof(parent::State)));
    evaluate<<<(total+31)/32,32>>>(tables.dev,plus,direct,dev_base,dev_base_arity,dev_r3,dev_r12,
                                   states,dev_cases,cases.size(),dev_ovs,ovs.size(),audit,dev_rows);
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaMemcpy(rows.data(),dev_rows,total*sizeof(Row),cudaMemcpyDeviceToHost));
#else
    auto states=std::make_unique<parent::State[]>(2*total);
    for(size_t id=0;id<total;++id){
        const size_t ci=id/cases.size(), xi=id%cases.size();
        Sides sides{};
        sides.policy[0]=plus;sides.pack[0]={pack.actions.data(),arity.data()};
        sides.policy[1]=cases[xi].member==1?direct:plus;
        sides.pack[1]={cases[xi].member==2?route3.data():cases[xi].member==3?route12.data():pack.actions.data(),arity.data()};
        engine::Config cfg;cfg.seed=cases[xi].seed;engine::Sim sim(cfg);
        rows[id]=run(sim,sides,states.get()+2*id,cases[xi],ovs[ci],audit);
    }
#endif
    for(size_t id=0;id<total;++id){const auto&r=rows[id];const auto&x=cases[id%cases.size()];
        std::cout<<(int)(id/cases.size())<<" "<<x.seed<<" "<<x.seat<<" "<<x.member
            <<" "<<r.money[0]<<" "<<r.money[1]<<" "<<r.simulation<<" "<<r.controller<<" "<<r.actions
            <<" "<<r.fault<<" "<<r.turns<<" "<<r.route
            <<" "<<r.purchases<<" "<<r.hires<<" "<<r.discarded<<" "<<r.discard_value<<" "<<r.unsold
            <<" "<<r.failed<<" "<<r.empty_pens
            <<" "<<r.d.deposits<<" "<<r.d.saved<<" "<<r.d.saved_value<<" "<<r.d.presales<<" "<<r.d.presold
            <<" "<<r.d.clamps<<" "<<r.d.clamped<<" "<<r.d.topups;
        for(int k=0;k<12;++k)std::cout<<" "<<r.produced[k];
        std::cout<<"\n";}
    std::cout.flush();
    return 0;
}
