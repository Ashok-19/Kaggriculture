// Isolated Phase-1 experiment. The frozen parent and both engines are reused.
#include "../../../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "extended_route_pack.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
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

// Phase-2 telemetry uses existing legal BKG4 features, never snapshot-only data.
constexpr int ROOT_FEATURES = 12;
struct LegalObservation {
    const engine::Sim& sim;
    int player;
    HD int feature(int id,int a0=0,int a1=0) const {
        (void)a1;const auto&f=sim.st.farms[player];
        switch(id){
            case bonsaikg4::F_STEP:return sim.st.step;
            case bonsaikg4::F_DAY:return sim.st.day;
            case bonsaikg4::F_HOUR:return sim.st.hour;
            case bonsaikg4::F_OWN_MONEY:return static_cast<int>(f.money);
            case bonsaikg4::F_OWN_UNITS:return f.n_units;
            case bonsaikg4::F_OWN_LAND:return f.n_quadrants;
            case bonsaikg4::F_MARKET_INVENTORY:return a0>=0&&a0<9?sim.st.market.inventory[a0]:0;
            case bonsaikg4::F_MARKET_PRICE:return a0>=0&&a0<9?sim.st.market.prices[a0]:0;
            case bonsaikg4::F_OWN_SHED:return a0>=0&&a0<12?f.shed[a0]:0;
            case bonsaikg4::F_OWN_SHED_TOTAL:return f.shed_total;
            case bonsaikg4::F_SHOP_AT:return a0>=0&&a0<sim.st.n_shops?sim.st.shops[a0]:-1;
            default:return 0;
        }
    }
};

struct Case { uint64_t seed; int seat, member, checkpoint; };
struct Candidate { int id, mode, duration; };
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
    int fault=0, changes=0, duration=0, turns=0, route=0, shops[2]{-1,-1};
    int features[ROOT_FEATURES]{};
};
// Logical fields only: no padding, pointer addresses or dead queue entries.
struct Hash {
    uint64_t value=14695981039346656037ULL;
    HD void add(int64_t x) {
        uint64_t u=static_cast<uint64_t>(x);
        for(int j=0;j<8;++j){value^=(u>>(j*8))&255;value*=1099511628211ULL;}
    }
    HD void real(double d) { union {double d; uint64_t u;} bits;bits.d=d;add(bits.u); }
};
HD void hash_state(Hash& h,const engine::Sim& sim) {
    const auto&s=sim.st; const auto&c=sim.cfg;
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
            h.add(t.pending_care_bonus);h.add(t.planted_day);h.add(t.max_lifespan_step);h.add(t.fertilized_until_day);
        }
        // Inactive worker inventories are included because a later hire reuses slots.
        for(int i=0;i<40;++i){h.add(f.inv_nkeys[i]);
            if(i<f.n_units){h.add(f.pos_x[i]);h.add(f.pos_y[i]);}
            for(int k=0;k<12;++k)h.add(f.inv[i][k]);
            for(int k=0;k<f.inv_nkeys[i];++k)h.add(f.inv_keys[i][k]);
        }
    }
}
HD void hash_controller(Hash&h,const parent::State&s){
    h.add(s.initialized);h.add(s.last_step);h.add(s.day);h.add(s.bb[0]);h.add(s.bb[1]);h.add(s.due);
    h.add(s.meta.fault);h.add(s.meta.reserved);h.add(s.meta.ticks);
    for(int k=0;k<12;++k)h.add(s.advanced[k]);
    for(int i=0;i<40;++i){h.add(s.counts[i]);h.add(s.heads[i]);
        for(int j=0;j<s.counts[i];++j){const auto&u=s.queue[i][(s.heads[i]+j)%parent::QUEUE];
            h.add(u.op);h.add(u.arg);h.add(u.n);h.add(u.arity);}
    }
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
        for(int k=0;k<12;++k){w.keys[k]=f.inv_keys[i][k];w.inv[k]=f.inv[i][k];}
    }return o;
}
HD engine::Action convert(const parent::Action&a){
    engine::Action out{};out.n_units=a.n_units;out.n_orders=a.n_market;
    for(int i=0;i<a.n_units;++i)out.units[i]={(uint8_t)a.units[i].op,(uint8_t)a.units[i].arg,(int16_t)a.units[i].n};
    for(int i=0;i<a.n_market;++i)out.orders[i]={(uint8_t)a.market[i].op,(uint8_t)a.market[i].item,a.market[i].n};
    return out;
}
// Compact action-family overrides. No controller state is reset or replaced.
HD void repair(parent::Action&a,const parent::Observation&o,int mode){
    if(mode==0)return;
    if(mode==1)parent::full_sell(a,o);
    if(mode==2||mode==3){
        for(int i=0;i<a.n_units&&i<o.n_positions;++i)
            if(parent::beside_shed(o,i)&&o.workers[i].inv_count)a.units[i]={6,0,1,1};
        if(mode==3)parent::full_sell(a,o);
    }
    if(mode==4){int stock[12];parent::project(a,o,stock);a.n_market=0;
        for(int k=0;k<9;++k)if(stock[k]>0)a.market[a.n_market++]={6,k,stock[k]};}
    if(mode==5){int n=0;for(int i=0;i<a.n_market;++i)if(a.market[i].op==6)a.market[n++]=a.market[i];a.n_market=n;}
    if(mode==6){
        for(int i=0;i<a.n_market;++i)if(a.market[i].op!=0&&a.market[i].op!=6)return;
        for(int i=1;i<a.n_market;++i){auto v=a.market[i];int j=i;
            while(j>0&&(int64_t)o.prices[a.market[j-1].item]*a.market[j-1].n<(int64_t)o.prices[v.item]*v.n){a.market[j]=a.market[j-1];--j;}
            a.market[j]=v;}
    }
    if(mode==7){for(int i=0;i<a.n_units&&i<o.n_positions;++i){auto&w=o.workers[i];
        if(!w.inv_count||parent::beside_shed(o,i))continue;
        a.units[i]={w.x<4?3:w.x>5?4:w.y<4?2:1,0,1,1};}}
    if(mode==8&&a.n_market<10)a.market[a.n_market++]={4,0,5};
    if(mode==9&&a.n_market<10)a.market[a.n_market++]={3,1,1};
    if(mode==10){for(int i=0;i<a.n_units;++i)if(a.units[i].op==8)a.units[i]={0,0,1,1};}
}
HD void turn(engine::Sim&sim,const bonsaikg4::PolicyView4&policy,parent::PackView pack,
             parent::State*states,Case c,Candidate candidate,Row&r,Hash&sh,Hash&ch,Hash&ah,bool audit,
             const bonsaikg4::PolicyView4&gate){
    if(audit)hash_state(sh,sim);
    engine::Action actions[2];
    for(int p=0;p<2;++p){auto o=observe(sim,p);parent::Result result{};
        if(o.n_positions>parent::TAPE_UNITS){r.fault=201;return;}
        parent::tick(policy,pack,o,states[p],result);if(result.fault)r.fault=result.fault;
        if(p!=c.seat&&c.member==1&&o.step>=576&&o.step<704)parent::full_sell(result.action,o);
        if(p==c.seat){
            Hash before;hash_action(before,result.action);
            int mode=o.step>=c.checkpoint&&o.step<c.checkpoint+candidate.duration?candidate.mode:0;
            if(gate.n_nodes&&candidate.mode){
                int32_t board[1]{};bonsaikg4::RuntimeMeta4 meta{};
                bonsaikg4::RuntimeView4 runtime{board,nullptr,&meta};bonsaikg4::reset_runtime(gate,runtime);
                bonsaikg4::ActionResult4 ignored{};LegalObservation legal{sim,p};
                bonsaikg4::tick4(gate,runtime,legal,0,ignored);
                if(meta.fault)r.fault=meta.fault;
                mode=board[0];
            }
            if(mode){
                repair(result.action,o,mode);++r.duration;
            }
            Hash after;hash_action(after,result.action);r.changes+=before.value!=after.value;
        }
        if(audit){hash_controller(ch,states[p]);hash_action(ah,result.action);}
        actions[p]=convert(result.action);
    }
    sim.step(actions[0],actions[1]);++r.turns;
}
HD Row run(engine::Sim&sim,const bonsaikg4::PolicyView4&policy,parent::PackView pack,
           Fork&fork,parent::State*states,Case c,Candidate candidate,int kind,bool audit,
           const bonsaikg4::PolicyView4&gate){
    // kind 0: prefix capture; 1: restore suffix; 2: matched uninterrupted intervention.
    Row r{};Hash sh,ch,ah;
    if(kind==1){
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
            LegalObservation legal{sim,c.seat};
            const int ids[ROOT_FEATURES]={bonsaikg4::F_OWN_MONEY,bonsaikg4::F_OWN_UNITS,bonsaikg4::F_OWN_LAND,
                bonsaikg4::F_MARKET_INVENTORY,bonsaikg4::F_MARKET_INVENTORY,bonsaikg4::F_MARKET_PRICE,bonsaikg4::F_MARKET_PRICE,
                bonsaikg4::F_OWN_SHED,bonsaikg4::F_OWN_SHED,bonsaikg4::F_OWN_SHED_TOTAL,bonsaikg4::F_SHOP_AT,bonsaikg4::F_SHOP_AT};
            const int args[ROOT_FEATURES]={0,0,0,2,7,2,7,2,7,0,0,1};
            for(int i=0;i<ROOT_FEATURES;++i)r.features[i]=legal.feature(ids[i],args[i]);
            // Compare only the matched suffix. Prefix work remains in the ledger.
            sh=Hash{};ch=Hash{};ah=Hash{};
        }
        Candidate active=kind==0?Candidate{}:candidate;
        turn(sim,policy,pack,states,c,active,r,sh,ch,ah,audit,gate);if(r.fault)break;
    }
    if(audit){hash_state(sh,sim);hash_controller(ch,states[0]);hash_controller(ch,states[1]);}
    r.money[0]=sim.st.farms[0].money;r.money[1]=sim.st.farms[1].money;
    r.simulation=sh.value;r.controller=ch.value;r.actions=ah.value;return r;
}
#ifdef __CUDACC__
__global__ void evaluate(kagcuda::DeviceTables tables,bonsaikg4::PolicyView4 policy,parent::PackView pack,
                        Fork*forks,parent::State*states,const Case*cases,int n,Candidate candidate,int kind,bool audit,Row*rows,
                        bonsaikg4::PolicyView4 gate){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    engine::Sim sim;engine::Config cfg;cfg.seed=cases[id].seed;sim.init(cfg,&tables);
    rows[id]=run(sim,policy,pack,forks[id],states+2*id,cases[id],candidate,kind,audit,gate);
}
template<class T>T*upload(const T*data,size_t n){T*out=nullptr;if(!n)return out;
    KAGCUDA_CHECK(cudaMalloc(&out,n*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(out,data,n*sizeof(T),cudaMemcpyHostToDevice));return out;}
#endif
int main(int argc,char**argv){
    if(argc<8||argc>10){std::cerr<<"controller routes arity cases candidates audit matched [legal_gate|-] [features]\n";return 2;}
    auto owned=bonsaikg4::load_compiled4(argv[1]);auto policy=owned.view();
    if(policy.n_bb!=2||policy.n_state!=0){std::cerr<<"unsupported parent state layout\n";return 2;}
    auto pack=bonsaikg4_champion::load_extended_route_pack(argv[2]);
    bonsaikg4::OwnedPolicy4 owned_gate;bonsaikg4::PolicyView4 gate{};
    if(argc>=9&&std::string(argv[8])!="-"){owned_gate=bonsaikg4::load_compiled4(argv[8]);gate=owned_gate.view();
        if(gate.n_bb!=1||gate.n_state!=0)return 2;}
    std::vector<uint16_t>arity(parent::ROUTES*parent::STEPS);std::ifstream af(argv[3],std::ios::binary);
    af.read((char*)arity.data(),arity.size()*2);if(!af)return 2;
    std::vector<Case>cases;Case c{};std::ifstream cf(argv[4]);while(cf>>c.seed>>c.seat>>c.member>>c.checkpoint)cases.push_back(c);
    std::vector<Candidate>candidates;Candidate candidate{};std::ifstream vf(argv[5]);
    while(vf>>candidate.id>>candidate.mode>>candidate.duration)candidates.push_back(candidate);
    if(cases.empty()||candidates.empty())return 2;
    const bool audit=std::stoi(argv[6]);const int matched=std::stoi(argv[7]);
    parent::PackView pv{pack.actions.data(),arity.data()};std::vector<Row>rows(cases.size());
    std::cerr<<"snapshot_bytes="<<sizeof(Fork)<<"\n";
#ifdef __CUDACC__
    kagcuda::configure_device();kagcuda::TablesOwner tables;
#define UP(field) policy.field=upload(owned.field.data(),owned.field.size())
    UP(nodes);UP(child_refs);UP(bb);UP(values);UP(feature_args);UP(predicates);UP(lifecycles);UP(effects);
#undef UP
    if(gate.n_nodes){
#define UP(field) gate.field=upload(owned_gate.field.data(),owned_gate.field.size())
        UP(nodes);UP(child_refs);UP(bb);UP(values);UP(feature_args);UP(predicates);UP(lifecycles);UP(effects);
#undef UP
    }
    pv={upload(pack.actions.data(),pack.actions.size()),upload(arity.data(),arity.size())};
    auto device_cases=upload(cases.data(),cases.size());Row*device_rows=nullptr;Fork*forks=nullptr;parent::State*states=nullptr;
    KAGCUDA_CHECK(cudaMalloc(&device_rows,rows.size()*sizeof(Row)));
    KAGCUDA_CHECK(cudaMalloc(&forks,rows.size()*sizeof(Fork)));
    KAGCUDA_CHECK(cudaMalloc(&states,rows.size()*2*sizeof(parent::State)));
#else
    auto forks=std::make_unique<Fork[]>(rows.size());auto states=std::make_unique<parent::State[]>(2);
#endif
    // Mode 3 reuses a context-pinned disk capture; mode 4 captures only.
    // The Python driver verifies the binary/case/parent identity and file digest.
    if(matched==3){
        std::vector<Fork> restored(rows.size());
#ifdef __CUDACC__
        std::string path=std::string(argv[4])+".cuda.snapshots";
#else
        std::string path=std::string(argv[4])+".cpu.snapshots";
#endif
        std::ifstream file(path,std::ios::binary);file.read((char*)restored.data(),restored.size()*sizeof(Fork));
        if(!file||file.peek()!=std::ifstream::traits_type::eof())return 2;
        for(size_t i=0;i<restored.size();++i)
            if(restored[i].game.cfg.seed!=cases[i].seed||restored[i].game.st.step!=cases[i].checkpoint)return 2;
#ifdef __CUDACC__
        KAGCUDA_CHECK(cudaMemcpy(forks,restored.data(),restored.size()*sizeof(Fork),cudaMemcpyHostToDevice));
#else
        for(size_t i=0;i<restored.size();++i)forks[i]=restored[i];
#endif
    }
    for(int index=matched==3?0:-1;index<(matched==4?0:(int)candidates.size());++index){
        candidate=index<0?Candidate{}:candidates[index];
        for(int kind=index<0?0:matched==2?2:1;kind<=(index<0?0:(matched==1||matched==2)?2:1);++kind){
#ifdef __CUDACC__
            evaluate<<<(rows.size()+31)/32,32>>>(tables.dev,policy,pv,forks,states,device_cases,rows.size(),candidate,kind,audit,device_rows,gate);
            KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaMemcpy(rows.data(),device_rows,rows.size()*sizeof(Row),cudaMemcpyDeviceToHost));
#else
            for(size_t i=0;i<rows.size();++i){engine::Config cfg;cfg.seed=cases[i].seed;engine::Sim sim(cfg);
                rows[i]=run(sim,policy,pv,forks[i],states.get(),cases[i],candidate,kind,audit,gate);}
#endif
            for(size_t i=0;i<rows.size();++i){const auto&r=rows[i];const auto&x=cases[i];
                std::cout<<kind<<" "<<candidate.id<<" "<<x.seed<<" "<<x.seat<<" "<<x.member<<" "<<x.checkpoint
                    <<" "<<r.money[0]<<" "<<r.money[1]<<" "<<r.root<<" "<<r.simulation<<" "<<r.controller<<" "<<r.actions
                    <<" "<<r.fault<<" "<<r.changes<<" "<<r.duration<<" "<<r.turns<<" "<<r.route<<" "<<r.shops[0]<<" "<<r.shops[1];
                if(argc==10)for(int j=0;j<ROOT_FEATURES;++j)std::cout<<" "<<r.features[j];
                std::cout<<"\n";
            }std::cout.flush();
            if(kind==0){
                // Exercise an actual host/disk round trip, not just a live device copy.
                std::vector<Fork> saved(rows.size()), restored(rows.size());
#ifdef __CUDACC__
                KAGCUDA_CHECK(cudaMemcpy(saved.data(),forks,saved.size()*sizeof(Fork),cudaMemcpyDeviceToHost));
                std::string path=std::string(argv[4])+".cuda.snapshots";
#else
                for(size_t i=0;i<saved.size();++i)saved[i]=forks[i];
                std::string path=std::string(argv[4])+".cpu.snapshots";
#endif
                {std::ofstream file(path,std::ios::binary);file.write((char*)saved.data(),saved.size()*sizeof(Fork));if(!file)return 2;}
                {std::ifstream file(path,std::ios::binary);file.read((char*)restored.data(),restored.size()*sizeof(Fork));if(!file)return 2;}
                if(std::memcmp(saved.data(),restored.data(),saved.size()*sizeof(Fork))!=0)return 2;
#ifdef __CUDACC__
                KAGCUDA_CHECK(cudaMemcpy(forks,restored.data(),restored.size()*sizeof(Fork),cudaMemcpyHostToDevice));
#else
                for(size_t i=0;i<restored.size();++i)forks[i]=restored[i];
#endif
            }
        }
    }
}
