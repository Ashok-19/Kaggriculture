// Representation/executive preflight for BKG4-projected hierarchical Expert Iteration.
// Dual-compiled with g++ and nvcc. The candidate consumes only Legal Observation
// fields copied into economic_plan_preflight::Observation; the frozen Shop Plus
// path remains byte-identical and is invoked directly for reference mode.
#include "../../../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "economic_plan_preflight.hpp"
#include "extended_route_pack.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#ifdef __CUDACC__
#include "runtime.cuh"
namespace engine=kagcuda;
#define HD __device__
#else
#include "sim.hpp"
namespace engine=kag;
#define HD inline
#endif
namespace parent=shop_router_0909;
namespace ep=economic_plan_preflight;
using namespace bonsaikg4;

struct Case { uint64_t seed; int seat,member; };
struct Row {
    double candidate_money=0,opp_money=0,reference_money=0,reference_opp_money=0;
    int turns=0,fault=0,first_divergence=-1,land=0,workers=0,hires=0,discarded=0,unsold=0,empty_pens=0;
    int crop[5]{},animal[3]{},produced[9]{};
    int failed_tasks=0,timeouts=0;
    uint64_t candidate_hash=0,reference_hash=0,plan_hash=0,first_action_hash=0;
};
struct Hash { uint64_t v=14695981039346656037ULL; HD void add(int64_t x){uint64_t u=(uint64_t)x;for(int j=0;j<8;++j){v^=(u>>(j*8))&255;v*=1099511628211ULL;}} HD void real(double d){union{double d;uint64_t u;}b;b.d=d;add((int64_t)b.u);} };

HD parent::Observation parent_observe(const engine::Sim&sim,int p){
    parent::Observation o{};o.step=sim.st.step;o.player=p;o.board=sim.cfg.board_size;const auto&f=sim.st.farms[p];o.n_positions=f.n_units;
    o.shop0=sim.st.n_shops>0?sim.st.shops[0]:-1;o.shop1=sim.st.n_shops>1?sim.st.shops[1]:-1;
    for(int k=0;k<12;++k){o.shed[k]=f.shed[k];o.shed_present[k]=k<9||f.shed[k]!=0;}for(int k=0;k<9;++k)o.prices[k]=sim.st.market.prices[k];
    for(int i=0;i<f.n_units&&i<parent::UNITS;++i){auto&w=o.workers[i];w.x=f.pos_x[i];w.y=f.pos_y[i];w.weed=f.tiles[w.y][w.x].kind==engine::T_WEED;w.inv_count=f.inv_nkeys[i];for(int k=0;k<12;++k){w.keys[k]=f.inv_keys[i][k];w.inv[k]=f.inv[i][k];}}
    return o;
}
HD ep::Observation observe(const engine::Sim&sim,int p){
    ep::Observation o{};o.step=sim.st.step;o.day=sim.st.day;o.hour=sim.st.hour;o.player=p;const auto&own=sim.st.farms[p];const auto&opp=sim.st.farms[1-p];
    o.own_money=(int)own.money;o.opp_money=(int)opp.money;o.own_units=own.n_units>ep::MAX_WORKERS?ep::MAX_WORKERS:own.n_units;o.opp_units=opp.n_units;
    o.own_land=own.n_quadrants;o.opp_land=opp.n_quadrants;o.hires_today=own.hires_today;o.shed_total=own.shed_total;
    for(int k=0;k<12;++k)o.shed[k]=own.shed[k];for(int c=0;c<5;++c)o.seeds[c]=own.seeds[c];for(int k=0;k<9;++k){o.market_inventory[k]=sim.st.market.inventory[k];o.price[k]=sim.st.market.prices[k];}
    for(int u=0;u<o.own_units;++u){o.workers[u].x=own.pos_x[u];o.workers[u].y=own.pos_y[u];for(int k=0;k<12;++k)o.workers[u].inv[k]=own.inv[u][k];}
    for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&a=own.tiles[y][x];auto&z=o.own[y][x];z.kind=a.kind;z.what=a.what;z.has_animal=a.has_animal;z.watered=a.watered_today;z.fed=a.fed_today;z.cared=a.cared_today;z.fertilizer=a.fertilizer_available;z.dry=a.consecutive_dry;z.yield=a.yield_units;z.planted_day=a.planted_day;
        const auto&b=opp.tiles[y][x];auto&q=o.opp[y][x];q.kind=b.kind;q.what=b.what;q.has_animal=b.has_animal;q.watered=b.watered_today;q.fed=b.fed_today;q.cared=b.cared_today;q.fertilizer=b.fertilizer_available;q.dry=b.consecutive_dry;q.yield=b.yield_units;q.planted_day=b.planted_day;}
    return o;
}
HD engine::Action convert(const parent::Action&a){engine::Action out{};out.n_units=a.n_units;out.n_orders=a.n_market;for(int i=0;i<a.n_units;++i)out.units[i]={(uint8_t)a.units[i].op,(uint8_t)a.units[i].arg,(int16_t)a.units[i].n};for(int i=0;i<a.n_market;++i)out.orders[i]={(uint8_t)a.market[i].op,(uint8_t)a.market[i].item,a.market[i].n};return out;}
HD engine::Action convert(const ep::Action&a){engine::Action out{};out.n_units=a.n_units;out.n_orders=a.n_orders;for(int i=0;i<a.n_units;++i)out.units[i]={(uint8_t)a.units[i].op,(uint8_t)a.units[i].arg,(int16_t)a.units[i].n};for(int i=0;i<a.n_orders;++i)out.orders[i]={(uint8_t)a.orders[i].op,(uint8_t)a.orders[i].item,a.orders[i].n};return out;}
HD bool same_action(const engine::Action&a,const engine::Action&b){if(a.n_units!=b.n_units||a.n_orders!=b.n_orders)return false;for(int i=0;i<a.n_units;++i)if(a.units[i].op!=b.units[i].op||a.units[i].arg!=b.units[i].arg||a.units[i].n!=b.units[i].n)return false;for(int i=0;i<a.n_orders;++i)if(a.orders[i].op!=b.orders[i].op||a.orders[i].item!=b.orders[i].item||a.orders[i].n!=b.orders[i].n)return false;return true;}
HD void hash_action(Hash&h,const engine::Action&a){h.add(a.n_units);h.add(a.n_orders);for(int i=0;i<a.n_units;++i){h.add(a.units[i].op);h.add(a.units[i].arg);h.add(a.units[i].n);}for(int i=0;i<a.n_orders;++i){h.add(a.orders[i].op);h.add(a.orders[i].item);h.add(a.orders[i].n);}}
HD uint64_t hash_bundle(const ep::Bundle&b){Hash h;h.add(b.n_stages);for(int s=0;s<ep::MAX_PLAN_STAGES;++s){const ep::Plan&p=b.stages[s];h.add(p.cow_target);h.add(p.sheep_target);h.add(p.goose_target);for(int c=0;c<5;++c)h.add(p.crop_target[c]);h.add(p.land_target);h.add(p.labour_target);h.add(p.cash_reserve);h.add(p.feed_mode);for(int i=0;i<6;++i)h.add(p.priority[i]);h.add(p.spatial_mode);h.add(p.sales_mode);h.add(p.duration);}return h.v;}
HD void hash_state(Hash&h,const engine::Sim&sim){h.add(sim.st.step);for(int p=0;p<2;++p){const auto&f=sim.st.farms[p];h.real(f.money);h.add(f.n_units);h.add(f.n_quadrants);h.add(f.hires_today);h.add(f.shed_total);for(int k=0;k<12;++k){h.add(f.shed[k]);h.add(f.produced[k]);h.add(f.discarded[k]);}for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];h.add(t.kind);h.add(t.what);h.add(t.has_animal);h.add(t.yield_units);h.add(t.watered_today);h.add(t.fed_today);h.add(t.cared_today);}}}

struct Sides { PolicyView4 policy[2]; parent::PackView pack[2]; };
HD void fill_sides(Sides&s,PolicyView4 plus,PolicyView4 direct,const bonsaikg_v6::PackedAction*base,const uint16_t*arity,const bonsaikg_v6::PackedAction*r3,const bonsaikg_v6::PackedAction*r12,const Case&c){s.policy[0]=plus;s.pack[0]={base,arity};s.policy[1]=c.member==1?direct:plus;s.pack[1]={c.member==2?r3:c.member==3?r12:base,arity};}

HD Row run_one(engine::Sim&sim,Sides sides,const Case&c,const ep::Bundle&bundle,bool candidate){
    Row r{};r.plan_hash=hash_bundle(bundle);parent::State opp_state{},subject_ref{},shadow{};ep::State exec{};Hash hh;
    while(!sim.st.done){engine::Action acts[2];for(int p=0;p<2;++p){const int side=p==c.seat?0:1;if(p!=c.seat){parent::Result z{};auto po=parent_observe(sim,p);if(po.n_positions>parent::TAPE_UNITS){r.fault=201;return r;}parent::tick(sides.policy[1],sides.pack[1],po,opp_state,z);if(z.fault){r.fault=z.fault;return r;}acts[p]=convert(z.action);continue;}
            if(!candidate){parent::Result z{};auto po=parent_observe(sim,p);if(po.n_positions>parent::TAPE_UNITS){r.fault=202;return r;}parent::tick(sides.policy[0],sides.pack[0],po,subject_ref,z);if(z.fault){r.fault=z.fault;return r;}acts[p]=convert(z.action);}else{
                auto eo=observe(sim,p);ep::Action ea=ep::tick(bundle,eo,exec);acts[p]=convert(ea);if(sim.st.step==0){Hash fh;hash_action(fh,acts[p]);r.first_action_hash=fh.v;}
                parent::Result z{};auto po=parent_observe(sim,p);if(po.n_positions<=parent::TAPE_UNITS){parent::tick(sides.policy[0],sides.pack[0],po,shadow,z);engine::Action pa=convert(z.action);if(r.first_divergence<0&&!same_action(acts[p],pa))r.first_divergence=sim.st.step;}
            }}
        hash_action(hh,acts[c.seat]);sim.step(acts[0],acts[1]);++r.turns;
    }
    hash_state(hh,sim);r.candidate_hash=hh.v;const auto&f=sim.st.farms[c.seat];r.candidate_money=f.money;r.opp_money=sim.st.farms[1-c.seat].money;r.land=f.n_quadrants;r.workers=f.n_units;r.hires=f.hires_today;
    for(int k=0;k<9;++k){r.produced[k]=f.produced[k];r.unsold+=f.shed[k];}for(int k=0;k<12;++k)r.discarded+=f.discarded[k];
    for(int y=0;y<10;++y)for(int x=0;x<10;++x){const auto&t=f.tiles[y][x];if(t.kind==engine::T_PLANT&&t.what<5)++r.crop[t.what];if(t.has_animal&&t.what>=engine::GOOSE&&t.what<=engine::SHEEP)++r.animal[t.what-engine::GOOSE];if((t.kind==engine::T_COOP||t.kind==engine::T_PASTURE)&&!t.has_animal)++r.empty_pens;}
    r.failed_tasks=exec.failed_tasks;r.timeouts=exec.timeouts;return r;
}
HD Row run_pair(engine::Sim&candidate_sim,engine::Sim&reference_sim,Sides sides,const Case&c,const ep::Bundle&b){Row a=run_one(candidate_sim,sides,c,b,true);Row r=run_one(reference_sim,sides,c,b,false);a.reference_money=r.candidate_money;a.reference_opp_money=r.opp_money;a.reference_hash=r.candidate_hash;return a;}

#ifdef __CUDACC__
__global__ void evaluate_representation_preflight(engine::DeviceTables tables,PolicyView4 plus,PolicyView4 direct,const bonsaikg_v6::PackedAction*base,const uint16_t*arity,const bonsaikg_v6::PackedAction*r3,const bonsaikg_v6::PackedAction*r12,const Case*cases,int nc,const ep::Bundle*bundles,int nb,Row*rows){int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nc*nb)return;int bi=id/nc,ci=id%nc;Case c=cases[ci];Sides s{};fill_sides(s,plus,direct,base,arity,r3,r12,c);engine::Config cfg;cfg.seed=c.seed;engine::Sim a,b;a.init(cfg,&tables);b.init(cfg,&tables);rows[id]=run_pair(a,b,s,c,bundles[bi]);}
template<class T>T*upload(const T*x,size_t n){T*out=nullptr;if(!n)return out;KAGCUDA_CHECK(cudaMalloc(&out,n*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(out,x,n*sizeof(T),cudaMemcpyHostToDevice));return out;}
PolicyView4 upload_policy(OwnedPolicy4&o){
    auto v=o.view();
    v.nodes=upload(o.nodes.data(),o.nodes.size());
    v.child_refs=upload(o.child_refs.data(),o.child_refs.size());
    v.bb=upload(o.bb.data(),o.bb.size());
    v.values=upload(o.values.data(),o.values.size());
    v.feature_args=upload(o.feature_args.data(),o.feature_args.size());
    v.predicates=upload(o.predicates.data(),o.predicates.size());
    v.lifecycles=upload(o.lifecycles.data(),o.lifecycles.size());
    v.effects=upload(o.effects.data(),o.effects.size());
    return v;
}
#endif
static std::vector<bonsaikg_v6::PackedAction> broadcast(const std::vector<bonsaikg_v6::PackedAction>&base,int route){std::vector<bonsaikg_v6::PackedAction>out(base.size());for(int r=0;r<parent::ROUTES;++r)for(int s=0;s<parent::STEPS;++s)out[r*parent::STEPS+s]=base[route*parent::STEPS+s];return out;}
int main(int argc,char**argv){
    if(argc!=7){std::cerr<<"plus_controller direct_controller routes arity cases plans\n";return 2;}
    auto op=load_compiled4(argv[1]);auto plus=op.view();auto od=load_compiled4(argv[2]);auto direct=od.view();if(plus.n_bb!=2||plus.n_state!=0){std::cerr<<"unsupported Shop Plus layout\n";return 2;}
    auto pack=bonsaikg4_champion::load_extended_route_pack(argv[3]);std::vector<uint16_t>arity(parent::ROUTES*parent::STEPS);{std::ifstream f(argv[4],std::ios::binary);f.read((char*)arity.data(),arity.size()*2);if(!f)return 2;}
    std::vector<Case>cases;Case c{};{std::ifstream f(argv[5]);while(f>>c.seed>>c.seat>>c.member)cases.push_back(c);}
    std::vector<ep::Bundle>bundles;{
        std::ifstream f(argv[6]);int bid=-1,stage=-1;
        while(f>>bid>>stage){int v[21]{};bool ok=true;for(int i=0;i<21;++i)if(!(f>>v[i])){ok=false;break;}if(!ok)break;
            if(bid<0||stage<0||stage>=ep::MAX_PLAN_STAGES) return 2;while((int)bundles.size()<=bid)bundles.push_back(ep::Bundle{});
            ep::Plan p{};p.cow_target=(uint8_t)v[0];p.sheep_target=(uint8_t)v[1];p.goose_target=(uint8_t)v[2];for(int i=0;i<5;++i)p.crop_target[i]=(uint8_t)v[3+i];p.land_target=(uint8_t)v[8];p.labour_target=(uint8_t)v[9];p.cash_reserve=(uint16_t)v[10];p.feed_mode=(uint8_t)v[11];for(int i=0;i<6;++i)p.priority[i]=(uint8_t)v[12+i];p.spatial_mode=(uint8_t)v[18];p.sales_mode=(uint8_t)v[19];p.duration=(uint8_t)v[20];bundles[bid].stages[stage]=p;if(stage+1>bundles[bid].n_stages)bundles[bid].n_stages=(uint8_t)(stage+1);
        }
    }
    if(cases.empty()||bundles.empty())return 2;auto r3=broadcast(pack.actions,3),r12=broadcast(pack.actions,12);size_t total=cases.size()*bundles.size();std::vector<Row>rows(total);
#ifdef __CUDACC__
    engine::configure_device();engine::TablesOwner tables;plus=upload_policy(op);direct=upload_policy(od);
    auto* db=upload(pack.actions.data(),pack.actions.size());
    auto* da=upload(arity.data(),arity.size());
    auto* d3=upload(r3.data(),r3.size());
    auto* d12=upload(r12.data(),r12.size());
    auto* dc=upload(cases.data(),cases.size());
    ep::Bundle* dbundles=upload(bundles.data(),bundles.size());
    Row*dr=nullptr;KAGCUDA_CHECK(cudaMalloc(&dr,total*sizeof(Row)));evaluate_representation_preflight<<<(total+31)/32,32>>>(tables.dev,plus,direct,db,da,d3,d12,dc,cases.size(),dbundles,bundles.size(),dr);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaMemcpy(rows.data(),dr,total*sizeof(Row),cudaMemcpyDeviceToHost));
#else
    for(size_t id=0;id<total;++id){size_t bi=id/cases.size(),ci=id%cases.size();Sides s{};fill_sides(s,plus,direct,pack.actions.data(),arity.data(),r3.data(),r12.data(),cases[ci]);engine::Config cfg;cfg.seed=cases[ci].seed;engine::Sim a(cfg),b(cfg);rows[id]=run_pair(a,b,s,cases[ci],bundles[bi]);}
#endif
    for(size_t id=0;id<total;++id){size_t pi=id/cases.size(),ci=id%cases.size();auto&r=rows[id];auto&x=cases[ci];std::cout<<pi<<" "<<x.seed<<" "<<x.seat<<" "<<x.member<<" "<<r.candidate_money<<" "<<r.opp_money<<" "<<r.reference_money<<" "<<r.reference_opp_money<<" "<<r.turns<<" "<<r.fault<<" "<<r.first_divergence<<" "<<r.land<<" "<<r.workers<<" "<<r.discarded<<" "<<r.unsold<<" "<<r.empty_pens<<" "<<r.failed_tasks<<" "<<r.timeouts<<" "<<r.candidate_hash<<" "<<r.reference_hash<<" "<<r.plan_hash<<" "<<r.first_action_hash;for(int k=0;k<5;++k)std::cout<<" "<<r.crop[k];for(int k=0;k<3;++k)std::cout<<" "<<r.animal[k];for(int k=0;k<9;++k)std::cout<<" "<<r.produced[k];std::cout<<"\n";}
    return 0;
}
