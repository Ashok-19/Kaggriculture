// Isolated production-route search evaluator. The frozen Shop Plus controller
// and the audited engines are reused unchanged; a candidate is only an
// alternative route pack read by the searching seat.
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

// member 0 Shop Plus, 1 direct Shop controller, 2/3 route-3 and route-12
// continuation opponents built from broadcast packs. All remain live executors.
struct Case { uint64_t seed; int seat, member, checkpoint; };
struct Fork {
#ifdef __CUDACC__
    kagcuda::Snapshot game;
#else
    struct { engine::Config cfg; engine::State st; } game;
#endif
    parent::State controllers[2];
};
static_assert(std::is_trivially_copyable<Fork>::value, "snapshot must own its state");
struct Row {
    double money[2]{};
    uint64_t root=0, simulation=0, controller=0, actions=0;
    int fault=0, turns=0, route=0, shops[2]{-1,-1};
    int produced[12]{}, purchases=0, hires=0, discarded=0, unsold=0, failed=0;
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
struct Sides {
    bonsaikg4::PolicyView4 policy[2];
    parent::PackView pack[2];
};
#ifndef __CUDACC__
// Host-only annotation stream. It reports exact pre/post engine state for the
// searching seat so job mining never relies on a re-implemented simulator.
static bool g_dump=false;
static void dump_units(const engine::Sim&sim,int p,const parent::Action&a){
    const auto&f=sim.st.farms[p];
    const int n=a.n_units<f.n_units?a.n_units:f.n_units;
    for(int i=0;i<n;++i){
        const int x=f.pos_x[i],y=f.pos_y[i];const auto&t=f.tiles[y][x];
        std::cout<<"E "<<sim.st.step<<" "<<i<<" "<<(int)a.units[i].op<<" "<<a.units[i].arg<<" "<<a.units[i].n
                 <<" "<<x<<" "<<y<<" "<<(int)t.kind<<" "<<(int)t.what<<" "<<(int)t.has_animal
                 <<" "<<(int)t.yield_units<<" "<<(int)t.planted_day<<"\n";
    }
    for(int i=0;i<a.n_market;++i)
        std::cout<<"M "<<sim.st.step<<" "<<i<<" "<<a.market[i].op<<" "<<a.market[i].item<<" "<<a.market[i].n<<"\n";
    std::cout<<"S "<<sim.st.step<<" "<<(long long)f.money<<" "<<f.n_units;
    for(int k=0;k<5;++k)std::cout<<" "<<f.seeds[k];
    for(int k=0;k<12;++k)std::cout<<" "<<f.shed[k];
    for(int k=0;k<12;++k)std::cout<<" "<<f.produced[k];
    std::cout<<"\n";
}
static void dump_tiles(const engine::Sim&sim,int p){
    const auto&f=sim.st.farms[p];
    for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];
        if(t.kind==engine::T_EMPTY||t.kind==engine::T_LOCKED)continue;
        std::cout<<"T "<<sim.st.step<<" "<<x<<" "<<y<<" "<<(int)t.kind<<" "<<(int)t.what<<" "<<(int)t.has_animal
                 <<" "<<(int)t.yield_units<<" "<<(int)t.planted_day<<"\n";}
}
#endif
HD void turn(engine::Sim&sim,Sides sides,parent::State*states,Case c,Row&r,Hash&sh,Hash&ch,Hash&ah,bool audit){
    if(audit)hash_state(sh,sim);
    engine::Action actions[2];
    for(int p=0;p<2;++p){auto o=observe(sim,p);parent::Result result{};
        const int side=(p==c.seat)?0:1;
        if(o.n_positions>parent::TAPE_UNITS){r.fault=201;return;}
        parent::tick(sides.policy[side],sides.pack[side],o,states[p],result);
        if(result.fault)r.fault=result.fault;
        if(p==c.seat){for(int i=0;i<result.action.n_market;++i){auto op=result.action.market[i].op;
            if(op==engine::M_BUY_ANIMAL||op==engine::M_BUY_SEED||op==engine::M_BUY_PRODUCT)++r.purchases;
            if(op==engine::M_HIRE)++r.hires;}}
        if(audit){hash_controller(ch,states[p]);hash_action(ah,result.action);}
#ifndef __CUDACC__
        if(g_dump&&p==c.seat){dump_units(sim,p,result.action);dump_tiles(sim,p);}
#endif
        actions[p]=convert(result.action);
    }
    sim.step(actions[0],actions[1]);++r.turns;
}
HD Row run(engine::Sim&sim,Sides sides,Fork&fork,parent::State*states,Case c,int kind,bool audit){
    // kind 0 captures the checkpoint; kind 3 restores it and runs the suffix.
    Row r{};Hash sh,ch,ah;
    if(kind==3){
#ifdef __CUDACC__
        auto tables=sim.tab;sim.load_snapshot(fork.game,&tables);
#else
        sim.cfg=fork.game.cfg;sim.st=fork.game.st;
#endif
        states[0]=fork.controllers[0];states[1]=fork.controllers[1];
    }else{states[0]=parent::State{};states[1]=parent::State{};}
    while(!sim.st.done){
        if(sim.st.step==c.checkpoint){
            if(kind==0){
#ifdef __CUDACC__
                sim.save_snapshot(&fork.game);
#else
                fork.game.cfg=sim.cfg;fork.game.st=sim.st;
#endif
                fork.controllers[0]=states[0];fork.controllers[1]=states[1];
            }
            Hash h;hash_state(h,sim);hash_controller(h,states[0]);hash_controller(h,states[1]);r.root=h.value;
            r.route=states[c.seat].bb[0];auto o=observe(sim,c.seat);r.shops[0]=o.shop0;r.shops[1]=o.shop1;
            sh=Hash{};ch=Hash{};ah=Hash{};
        }
        turn(sim,sides,states,c,r,sh,ch,ah,audit);if(r.fault)break;
    }
    if(audit){hash_state(sh,sim);hash_controller(ch,states[0]);hash_controller(ch,states[1]);}
    const auto&f=sim.st.farms[c.seat];
    for(int k=0;k<12;++k){r.produced[k]=f.produced[k];r.discarded+=f.discarded[k];if(k<9)r.unsold+=f.shed[k];}
    for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];
        if(t.kind==engine::T_PLANT&&t.yield_units>0)++r.failed;
        if((t.kind==engine::T_COOP||t.kind==engine::T_PASTURE)&&!t.has_animal)++r.failed;}
    r.money[0]=sim.st.farms[0].money;r.money[1]=sim.st.farms[1].money;
    r.simulation=sh.value;r.controller=ch.value;r.actions=ah.value;return r;
}
#ifdef __CUDACC__
__global__ void evaluate(kagcuda::DeviceTables tables,bonsaikg4::PolicyView4 plus,bonsaikg4::PolicyView4 direct,
                         const bonsaikg_v6::PackedAction*const*packs,const uint16_t*const*arities,
                         const bonsaikg_v6::PackedAction*base,const uint16_t*base_arity,
                         const bonsaikg_v6::PackedAction*route3,const bonsaikg_v6::PackedAction*route12,
                         Fork*forks,parent::State*states,const Case*cases,int n_cases,
                         const int*candidates,int n_candidates,int kind,bool audit,Row*rows){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n_cases*n_candidates)return;
    const int ci=id/n_cases, xi=id%n_cases;
    Case c=cases[xi];
    Sides sides{};
    sides.policy[0]=plus;sides.pack[0]={candidates[ci]<0?base:packs[candidates[ci]],
                                        candidates[ci]<0?base_arity:arities[candidates[ci]]};
    sides.policy[1]=c.member==1?direct:plus;
    sides.pack[1]={c.member==2?route3:c.member==3?route12:base,base_arity};
    engine::Sim sim;engine::Config cfg;cfg.seed=c.seed;sim.init(cfg,&tables);
    rows[id]=run(sim,sides,forks[id],states+2*id,c,kind,audit);
}
template<class T>T*upload(const T*data,size_t n){T*out=nullptr;if(!n)return out;
    KAGCUDA_CHECK(cudaMalloc(&out,n*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(out,data,n*sizeof(T),cudaMemcpyHostToDevice));return out;}
bonsaikg4::PolicyView4 upload_policy(bonsaikg4::OwnedPolicy4&owned){
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
    if(argc!=9){std::cerr<<"plus_controller direct_controller routes arity cases candidate_list kind audit\n";return 2;}
    auto owned=bonsaikg4::load_compiled4(argv[1]);auto plus=owned.view();
    if(plus.n_bb!=2||plus.n_state!=0){std::cerr<<"unsupported parent state layout\n";return 2;}
    auto owned_direct=bonsaikg4::load_compiled4(argv[2]);auto direct=owned_direct.view();
    auto pack=bonsaikg4_champion::load_extended_route_pack(argv[3]);
    std::vector<uint16_t>arity(parent::ROUTES*parent::STEPS);
    {std::ifstream af(argv[4],std::ios::binary);af.read((char*)arity.data(),arity.size()*2);if(!af)return 2;}
    std::vector<Case>cases;Case c{};{std::ifstream cf(argv[5]);while(cf>>c.seed>>c.seat>>c.member>>c.checkpoint)cases.push_back(c);}
    // Candidate list: "-" for the zero-edit reference, otherwise pack path stems.
    std::vector<std::string>paths;{std::ifstream vf(argv[6]);std::string line;while(vf>>line)paths.push_back(line);}
    int kind=std::stoi(argv[7]);const bool audit=std::stoi(argv[8]);
#ifndef __CUDACC__
    if(kind==5){g_dump=true;kind=0;}
#endif
    if(cases.empty()||paths.empty())return 2;
    std::vector<std::vector<bonsaikg_v6::PackedAction>>owned_packs;
    std::vector<std::vector<uint16_t>>owned_arity;std::vector<int>candidates;
    for(auto&p:paths){
        if(p=="-"){candidates.push_back(-1);continue;}
        auto cp=bonsaikg4_champion::load_extended_route_pack(p+".bkv6");
        std::vector<uint16_t>ca(parent::ROUTES*parent::STEPS);
        {std::ifstream af(p+".arity",std::ios::binary);af.read((char*)ca.data(),ca.size()*2);if(!af)return 2;}
        candidates.push_back((int)owned_packs.size());
        owned_packs.push_back(std::move(cp.actions));owned_arity.push_back(std::move(ca));
    }
    auto route3=broadcast(pack.actions,3),route12=broadcast(pack.actions,12);
    const size_t total=cases.size()*candidates.size();
    std::vector<Row>rows(total);
    std::cerr<<"snapshot_bytes="<<sizeof(Fork)<<" threads="<<total<<"\n";
#ifdef __CUDACC__
    kagcuda::configure_device();kagcuda::TablesOwner tables;
    plus=upload_policy(owned);direct=upload_policy(owned_direct);
    auto dev_base=upload(pack.actions.data(),pack.actions.size());
    auto dev_base_arity=upload(arity.data(),arity.size());
    auto dev_r3=upload(route3.data(),route3.size()),dev_r12=upload(route12.data(),route12.size());
    std::vector<const bonsaikg_v6::PackedAction*>host_packs;std::vector<const uint16_t*>host_arity;
    for(size_t i=0;i<owned_packs.size();++i){host_packs.push_back(upload(owned_packs[i].data(),owned_packs[i].size()));
        host_arity.push_back(upload(owned_arity[i].data(),owned_arity[i].size()));}
    auto dev_packs=upload(host_packs.data(),host_packs.size());
    auto dev_arity=upload(host_arity.data(),host_arity.size());
    auto dev_cases=upload(cases.data(),cases.size());
    auto dev_candidates=upload(candidates.data(),candidates.size());
    Row*dev_rows=nullptr;Fork*forks=nullptr;parent::State*states=nullptr;
    KAGCUDA_CHECK(cudaMalloc(&dev_rows,total*sizeof(Row)));
    KAGCUDA_CHECK(cudaMalloc(&forks,total*sizeof(Fork)));
    KAGCUDA_CHECK(cudaMalloc(&states,total*2*sizeof(parent::State)));
#else
    auto forks=std::make_unique<Fork[]>(total);auto states=std::make_unique<parent::State[]>(2*total);
#endif
    const std::string snapshot_path=std::string(argv[5])+".snapshots";
    if(kind==3){
        std::vector<Fork>restored(cases.size());
        std::ifstream file(snapshot_path,std::ios::binary);
        file.read((char*)restored.data(),restored.size()*sizeof(Fork));
        if(!file||file.peek()!=std::ifstream::traits_type::eof())return 2;
        for(size_t i=0;i<restored.size();++i)
            if(restored[i].game.cfg.seed!=cases[i].seed||restored[i].game.st.step!=cases[i].checkpoint)return 2;
        std::vector<Fork>wide(total);
        for(size_t k=0;k<candidates.size();++k)
            for(size_t i=0;i<cases.size();++i)wide[k*cases.size()+i]=restored[i];
#ifdef __CUDACC__
        KAGCUDA_CHECK(cudaMemcpy(forks,wide.data(),wide.size()*sizeof(Fork),cudaMemcpyHostToDevice));
#else
        for(size_t i=0;i<wide.size();++i)forks[i]=wide[i];
#endif
    }
#ifdef __CUDACC__
    evaluate<<<(total+31)/32,32>>>(tables.dev,plus,direct,dev_packs,dev_arity,dev_base,dev_base_arity,dev_r3,dev_r12,
                                   forks,states,dev_cases,cases.size(),dev_candidates,candidates.size(),kind,audit,dev_rows);
    KAGCUDA_CHECK(cudaGetLastError());
    KAGCUDA_CHECK(cudaMemcpy(rows.data(),dev_rows,total*sizeof(Row),cudaMemcpyDeviceToHost));
    if(kind==0){
        std::vector<Fork>saved(total);
        KAGCUDA_CHECK(cudaMemcpy(saved.data(),forks,saved.size()*sizeof(Fork),cudaMemcpyDeviceToHost));
        std::vector<Fork>head(saved.begin(),saved.begin()+cases.size()),restored(cases.size());
        {std::ofstream f(snapshot_path,std::ios::binary);f.write((char*)head.data(),head.size()*sizeof(Fork));if(!f)return 2;}
        {std::ifstream f(snapshot_path,std::ios::binary);f.read((char*)restored.data(),restored.size()*sizeof(Fork));if(!f)return 2;}
        if(std::memcmp(head.data(),restored.data(),head.size()*sizeof(Fork))!=0)return 2;
    }
#else
    for(size_t id=0;id<total;++id){
        const size_t ci=id/cases.size(), xi=id%cases.size();
        Sides sides{};
        sides.policy[0]=plus;
        sides.pack[0]={candidates[ci]<0?pack.actions.data():owned_packs[candidates[ci]].data(),
                       candidates[ci]<0?arity.data():owned_arity[candidates[ci]].data()};
        sides.policy[1]=cases[xi].member==1?direct:plus;
        sides.pack[1]={cases[xi].member==2?route3.data():cases[xi].member==3?route12.data():pack.actions.data(),arity.data()};
        engine::Config cfg;cfg.seed=cases[xi].seed;engine::Sim sim(cfg);
        rows[id]=run(sim,sides,forks[id],states.get()+2*id,cases[xi],kind,audit);
    }
    if(kind==0){
        std::ofstream f(snapshot_path,std::ios::binary);
        f.write((char*)forks.get(),cases.size()*sizeof(Fork));if(!f)return 2;
    }
#endif
    for(size_t id=0;id<total;++id){const auto&r=rows[id];const auto&x=cases[id%cases.size()];
        std::cout<<(int)(id/cases.size())<<" "<<x.seed<<" "<<x.seat<<" "<<x.member<<" "<<x.checkpoint
            <<" "<<r.money[0]<<" "<<r.money[1]<<" "<<r.root<<" "<<r.simulation<<" "<<r.controller<<" "<<r.actions
            <<" "<<r.fault<<" "<<r.turns<<" "<<r.route<<" "<<r.shops[0]<<" "<<r.shops[1]
            <<" "<<r.purchases<<" "<<r.hires<<" "<<r.discarded<<" "<<r.unsold<<" "<<r.failed;
        for(int k=0;k<12;++k)std::cout<<" "<<r.produced[k];
        std::cout<<"\n";}
    std::cout.flush();
    return 0;
}
