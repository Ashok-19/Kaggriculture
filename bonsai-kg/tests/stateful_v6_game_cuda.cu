#include "../integrations/stateful_v6/cuda_runtime.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bonsaikg;
using namespace kagcuda;

struct DevicePolicyOwner{Node*nodes=nullptr;uint16_t*children=nullptr;BbDecl*bb=nullptr;ValueRef*values=nullptr;FeatureArgRef*feature_args=nullptr;Predicate*pred=nullptr;Lifecycle*lc=nullptr;Effect*effects=nullptr;~DevicePolicyOwner(){if(nodes)cudaFree(nodes);if(children)cudaFree(children);if(bb)cudaFree(bb);if(values)cudaFree(values);if(feature_args)cudaFree(feature_args);if(pred)cudaFree(pred);if(lc)cudaFree(lc);if(effects)cudaFree(effects);}};
template<class T>static T*copy_dev(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}

struct TickTrace{Action action{};int32_t leaf=255,route=64,s1=0,s2=0;uint16_t fault=0,compat_fault=0;};

__global__ void run_games(Config base,DeviceTables tables,bonsaikg_v6::CudaCompatPolicy policy,uint64_t seed0,int count,int seat,int max_ticks,Sim*sims,bonsaikg_v6::CudaCompatState*states,TickTrace*trace,int*ticks){
    int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=count)return;Config cfg=base;cfg.seed=seed0+(uint64_t)g;Sim&sim=sims[g];sim.init(cfg,&tables);bonsaikg_v6::CudaCompatState s{};reset_runtime(policy.tree,s.controller);s.last_step=-1;s.compat_fault=0;int t=0;
    while(!sim.st.done&&t<max_ticks){
        ObservationView o=make_observation(sim,seat);Action ca{};policy.act(s,o,ca);TickTrace&r=trace[(size_t)g*max_ticks+t];r.action=ca;r.leaf=s.controller.blackboard[0];r.route=s.controller.blackboard[1];r.s1=s.controller.blackboard[2];r.s2=s.controller.blackboard[3];r.fault=s.controller.fault;r.compat_fault=s.compat_fault;
        int opp=1-seat;Action oa{};oa.n_units=sim.st.farms[opp].n_units;if(seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);++t;
    }
    states[g]=s;ticks[g]=t;
}

static void print_tick(int g,uint64_t seed,int step,const TickTrace&r){
    const Action&a=r.action;std::cout<<"G="<<g<<" SEED="<<seed<<" T="<<step<<" L="<<r.leaf<<" R="<<r.route<<" S1="<<r.s1<<" S2="<<r.s2<<" F="<<r.fault<<" U="<<a.n_units<<":";
    for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}
    std::cout<<" M="<<a.n_orders<<":";for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}std::cout<<'\n';
}

int main(int argc,char**argv){
    try{
        if(argc<3||argc>7)throw std::runtime_error("usage: stateful_v6_game_cuda CONTROLLER ROUTE_PACK [SEED_START] [COUNT] [SEAT] [STEPS]");const uint64_t seed0=argc>=4?std::strtoull(argv[3],nullptr,10):990000ull;const int count=argc>=5?std::atoi(argv[4]):4,seat=argc>=6?std::atoi(argv[5]):0,steps=argc>=7?std::atoi(argv[6]):720;if(count<=0||seat<0||seat>1||steps<2)throw std::runtime_error("invalid game-runner argument");
        auto owned=load_compiled(argv[1]);if(bonsaikg_v6::controller_hash_hex(owned)!=bonsaikg_v6::CONTROLLER_SEMANTIC_HASH)throw std::runtime_error("controller identity mismatch");auto hp=owned.view();if(hp.max_intents)throw std::runtime_error("controller unexpectedly emits intents");DevicePolicyOwner d;d.nodes=copy_dev(owned.nodes);d.children=copy_dev(owned.child_refs);d.bb=copy_dev(owned.bb);d.values=copy_dev(owned.values);d.feature_args=copy_dev(owned.feature_args);d.pred=copy_dev(owned.predicates);d.lc=copy_dev(owned.lifecycles);d.effects=copy_dev(owned.effects);PolicyView p=hp;p.nodes=d.nodes;p.child_refs=d.children;p.bb=d.bb;p.values=d.values;p.feature_args=d.feature_args;p.predicates=d.pred;p.lifecycles=d.lc;p.effects=d.effects;
        auto pack=bonsaikg_v6::RoutePack::load(argv[2]);bonsaikg_v6::RouteDesc*dr=nullptr;bonsaikg_v6::PackedAction*da=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&dr,pack.routes.size()*sizeof(*dr)));KAGCUDA_CHECK(cudaMalloc((void**)&da,pack.actions.size()*sizeof(*da)));KAGCUDA_CHECK(cudaMemcpy(dr,pack.routes.data(),pack.routes.size()*sizeof(*dr),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(da,pack.actions.data(),pack.actions.size()*sizeof(*da),cudaMemcpyHostToDevice));bonsaikg_v6::CudaCompatPolicy policy{p,dr,da,(int)pack.routes.size()};
        configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=steps;Sim*ds=nullptr;bonsaikg_v6::CudaCompatState*dst=nullptr;TickTrace*dtrace=nullptr;int*dticks=nullptr;const int max_ticks=steps;KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)count*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dst,(size_t)count*sizeof(*dst)));KAGCUDA_CHECK(cudaMalloc((void**)&dtrace,(size_t)count*max_ticks*sizeof(*dtrace)));KAGCUDA_CHECK(cudaMalloc((void**)&dticks,(size_t)count*sizeof(*dticks)));KAGCUDA_CHECK(cudaMemset(dtrace,0,(size_t)count*max_ticks*sizeof(*dtrace)));
        run_games<<<(count+63)/64,64>>>(cfg,tables.dev,policy,seed0,count,seat,max_ticks,ds,dst,dtrace,dticks);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());std::vector<int>ticks(count);std::vector<TickTrace>trace((size_t)count*max_ticks);std::vector<bonsaikg_v6::CudaCompatState>states(count);KAGCUDA_CHECK(cudaMemcpy(ticks.data(),dticks,(size_t)count*sizeof(int),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(trace.data(),dtrace,trace.size()*sizeof(TickTrace),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(states.data(),dst,(size_t)count*sizeof(states[0]),cudaMemcpyDeviceToHost));
        for(int g=0;g<count;++g){for(int t=0;t<ticks[g];++t)print_tick(g,seed0+(uint64_t)g,t,trace[(size_t)g*max_ticks+t]);std::cerr<<"GAME="<<g<<" SEED="<<(seed0+(uint64_t)g)<<" SEAT="<<seat<<" TICKS="<<ticks[g]<<" CF="<<states[g].compat_fault<<" VF="<<states[g].controller.fault<<"\n";}
        cudaFree(dr);cudaFree(da);cudaFree(ds);cudaFree(dst);cudaFree(dtrace);cudaFree(dticks);return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
