// Isolated bounded gold-v25 self-play evaluator; no parent source changes.
#include "../gold_v25/native.hpp"
#include "lead.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "extended_route_pack.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#ifdef __CUDACC__
#include "runtime.cuh"
namespace engine = kagcuda;
#define HD __device__
#else
#include "sim.hpp"
namespace engine = kag;
#define HD inline
#endif

struct Candidate {int id,start,end,mode;};
struct Row {double money[2];uint64_t hash;int fault,changes;};

HD gold_v25::Observation observe(const engine::Sim&s,int p){
    gold_v25::Observation o{};o.step=s.st.step;o.player=p;o.board=10;
    const auto&f=s.st.farms[p];o.n_units=f.n_units;o.egg=s.st.market.inventory[5];
    for(int j=0;j<s.st.n_shops;++j)if(s.st.shops[j]==7)o.yarn=1;
    for(int k=0;k<12;++k)o.shed[k]=f.shed[k];
    for(int k=0;k<5;++k)o.seeds[k]=f.seeds[k];
    for(int k=0;k<9;++k)o.prices[k]=s.st.market.prices[k];
    for(int i=0;i<f.n_units;++i){auto&w=o.workers[i];w.x=f.pos_x[i];w.y=f.pos_y[i];w.inv_count=f.inv_nkeys[i];
        for(int k=0;k<12;++k){w.keys[k]=f.inv_keys[i][k];w.inv[k]=f.inv[i][k];}
        const auto&t=f.tiles[w.y][w.x];
        w.tile.kind=t.kind==engine::T_LOCKED?-1:t.kind==engine::T_EMPTY?0:t.kind==engine::T_WEED?1:t.kind==engine::T_PLANT?2:t.kind==engine::T_COOP?3:4;
        w.tile.animal=t.has_animal;w.tile.watered=t.watered_today;w.tile.yield=t.yield_units;
        w.tile.fed=t.fed_today;w.tile.fertilizer=t.fertilizer_available;w.tile.cared=t.cared_today;
    }return o;
}
HD engine::Action convert(const gold_v25::Action&a){
    engine::Action out{};out.n_units=a.n_units;out.n_orders=a.n_market;
    for(int i=0;i<a.n_units;++i)out.units[i]={(uint8_t)a.units[i].op,(uint8_t)a.units[i].arg,(int16_t)a.units[i].n};
    for(int i=0;i<a.n_market;++i)out.orders[i]={(uint8_t)a.market[i].op,(uint8_t)a.market[i].item,a.market[i].n};
    return out;
}
HD void repair(gold_v25::Action&a,const gold_v25::Observation&o,Candidate c,gold_v25::PackView pack,int route){
    if(!c.mode||o.step<c.start||o.step>c.end)return;
    if(c.mode==9)c.mode=o.step>=708?5:1;
    if(c.mode==17)c.mode=o.step>=708?5:17;
    if(c.mode==33||c.mode==65){
        if(o.step>=708)c.mode=5;
        else gold_v25_plus::lead(a,o,pack,route,c.mode==33?2:3);
    }
    if(c.mode&2)for(int i=0;i<o.n_units;++i)if(gold_v25::adjacent(o.workers[i],o.board)&&o.workers[i].inv_count)a.units[i]={6,0,1,1};
    int projected[12];gold_v25::project(a,o,projected);
    if(c.mode&16){
        for(int i=0;i<a.n_market;++i)if(a.market[i].op!=0&&a.market[i].op!=6)return;
        for(int k=1;k<8;++k)if(projected[k]>0){
            bool found=false;
            for(int i=0;i<a.n_market;++i)if(a.market[i].op==6&&a.market[i].item==k){a.market[i].n=projected[k];found=true;}
            if(!found&&a.n_market<10)a.market[a.n_market++]={6,k,projected[k]};
        }
    }
    if(c.mode&4){a.n_market=0;for(int k=0;k<9;++k)if(projected[k]>0)a.market[a.n_market++]={6,k,projected[k]};}
    if(c.mode&1){
        // Only reorder all-SELL/no-op sequences. Never move purchases or hires.
        for(int i=0;i<a.n_market;++i)if(a.market[i].op!=0&&a.market[i].op!=6)return;
        for(int i=1;i<a.n_market;++i){auto v=a.market[i];int j=i;
            int64_t value=v.op==6?(int64_t)o.prices[v.item]*gold_v25::low(projected[v.item],v.n):0;
            while(j>0){auto prev=a.market[j-1];int64_t pv=prev.op==6?(int64_t)o.prices[prev.item]*gold_v25::low(projected[prev.item],prev.n):0;
                if(pv>=value)break;a.market[j]=prev;--j;}a.market[j]=v;}
    }
}
HD Row episode(engine::Sim&sim,const bonsaikg4::PolicyView4&policy,gold_v25::PackView pack,gold_v25::State*states,Candidate c,int seat,const bonsaikg4::PolicyView4*gate=nullptr){
    states[0].initialized=states[1].initialized=0;Row row{};row.hash=14695981039346656037ULL;
    while(!sim.st.done){
        engine::Action actions[2];
        for(int p=0;p<2;++p){auto o=observe(sim,p);gold_v25::Result result{};gold_v25::tick(policy,pack,o,states[p],result);
            if(result.fault)row.fault=result.fault;
            if(p==seat){auto old=result.action;
                bool enabled=true;
                if(gate){int32_t bb[1]{};bonsaikg4::RuntimeMeta4 meta{};bonsaikg4::RuntimeView4 state{bb,nullptr,&meta};
                    bonsaikg4::reset_runtime(*gate,state);bonsaikg4::ActionResult4 ignored{};
                    bonsaikg4::tick4(*gate,state,o,0,ignored);enabled=bb[0]!=0;
                    if(meta.fault||enabled!=(o.step>=c.start&&o.step<=c.end))row.fault=200;
                }
                if(enabled)repair(result.action,o,c,pack,result.route);
                bool changed=old.n_market!=result.action.n_market;
                for(int i=0;i<old.n_units;++i)changed|=old.units[i].op!=result.action.units[i].op||old.units[i].arg!=result.action.units[i].arg||old.units[i].n!=result.action.units[i].n;
                for(int i=0;i<old.n_market&&i<result.action.n_market;++i)changed|=old.market[i].op!=result.action.market[i].op||old.market[i].item!=result.action.market[i].item||old.market[i].n!=result.action.market[i].n;
                row.changes+=changed;
            }
            actions[p]=convert(result.action);
            for(int i=0;i<result.action.n_units;++i){auto u=result.action.units[i];row.hash=gold_v25::hash_word(row.hash,u.op);row.hash=gold_v25::hash_word(row.hash,u.arg);row.hash=gold_v25::hash_word(row.hash,u.n);}
            row.hash=gold_v25::hash_word(row.hash,result.action.n_market);
            for(int i=0;i<result.action.n_market;++i){auto z=result.action.market[i];row.hash=gold_v25::hash_word(row.hash,z.op);row.hash=gold_v25::hash_word(row.hash,z.item);row.hash=gold_v25::hash_word(row.hash,z.n);}
        }sim.step(actions[0],actions[1]);
    }
    row.money[0]=sim.st.farms[0].money;row.money[1]=sim.st.farms[1].money;return row;
}
#ifdef __CUDACC__
__global__ void evaluate(kagcuda::DeviceTables tables,bonsaikg4::PolicyView4 policy,gold_v25::PackView pack,gold_v25::State*states,Candidate c,const uint64_t*seeds,int n,Row*rows,bonsaikg4::PolicyView4 gate){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*2)return;
    engine::Sim sim;engine::Config cfg;cfg.seed=seeds[id/2];sim.init(cfg,&tables);
    rows[id]=episode(sim,policy,pack,states+id*2,c,id%2,gate.n_nodes?&gate:nullptr);
}
template<class T>T*upload(const T*data,size_t n){T*out=nullptr;if(!n)return out;KAGCUDA_CHECK(cudaMalloc(&out,n*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(out,data,n*sizeof(T),cudaMemcpyHostToDevice));return out;}
#endif
int main(int argc,char**argv){
    if(argc!=6&&argc!=7){std::cerr<<"controller routes arity candidates seeds [specialist]\n";return 2;}
    auto policy=bonsaikg4::load_compiled4(argv[1]);auto pack=bonsaikg4_champion::load_extended_route_pack(argv[2]);
    std::vector<uint16_t>arity(2876);std::ifstream af(argv[3],std::ios::binary);af.read((char*)arity.data(),5752);if(!af)return 2;
    std::vector<Candidate>candidates;Candidate c{};std::ifstream cf(argv[4]);while(cf>>c.id>>c.start>>c.end>>c.mode)candidates.push_back(c);
    std::vector<uint64_t>seeds;uint64_t seed;std::ifstream sf(argv[5]);while(sf>>seed)seeds.push_back(seed);
    auto view=policy.view();gold_v25::PackView pv{pack.actions.data(),arity.data()};std::vector<Row>rows(seeds.size()*2);
    bonsaikg4::OwnedPolicy4 owned_gate;bonsaikg4::PolicyView4 gate{};
    if(argc==7){owned_gate=bonsaikg4::load_compiled4(argv[6]);gate=owned_gate.view();}
#ifdef __CUDACC__
    kagcuda::configure_device();kagcuda::TablesOwner tables;
#define UP(field) view.field=upload(policy.field.data(),policy.field.size())
    UP(nodes);UP(child_refs);UP(bb);UP(values);UP(feature_args);UP(predicates);UP(lifecycles);UP(effects);
#undef UP
    if(gate.n_nodes){
#define UP(field) gate.field=upload(owned_gate.field.data(),owned_gate.field.size())
        UP(nodes);UP(child_refs);UP(bb);UP(values);UP(feature_args);UP(predicates);UP(lifecycles);UP(effects);
#undef UP
    }
    pv={upload(pack.actions.data(),pack.actions.size()),upload(arity.data(),arity.size())};
    auto device_seeds=upload(seeds.data(),seeds.size());Row*device_rows=nullptr;gold_v25::State*states=nullptr;
    KAGCUDA_CHECK(cudaMalloc(&device_rows,rows.size()*sizeof(Row)));KAGCUDA_CHECK(cudaMalloc(&states,rows.size()*2*sizeof(gold_v25::State)));
#else
    auto states=std::make_unique<gold_v25::State[]>(2);
#endif
    for(auto candidate:candidates){
#ifdef __CUDACC__
        evaluate<<<(rows.size()+31)/32,32>>>(tables.dev,view,pv,states,candidate,device_seeds,seeds.size(),device_rows,gate);
        KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaMemcpy(rows.data(),device_rows,rows.size()*sizeof(Row),cudaMemcpyDeviceToHost));
#else
        for(size_t i=0;i<rows.size();++i){engine::Config cfg;cfg.seed=seeds[i/2];engine::Sim sim(cfg);rows[i]=episode(sim,view,pv,states.get(),candidate,i%2,gate.n_nodes?&gate:nullptr);}
#endif
        for(size_t i=0;i<rows.size();++i){auto r=rows[i];std::cout<<candidate.id<<" "<<seeds[i/2]<<" "<<i%2<<" "<<r.money[0]<<" "<<r.money[1]<<" "<<r.hash<<" "<<r.fault<<" "<<r.changes<<"\n";}std::cout.flush();
    }
}
