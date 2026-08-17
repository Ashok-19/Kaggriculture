#include "../include/bonsai_kg.hpp"
#include "../include/bonsai_kg_io.hpp"
#include "../cuda/kag_adapter.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include "synthetic_io.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bonsaikg;using namespace kagcuda;
struct DevicePolicyOwner{Node*nodes=nullptr;uint16_t*children=nullptr;BbDecl*bb=nullptr;ValueRef*values=nullptr;FeatureArgRef*feature_args=nullptr;Predicate*pred=nullptr;Lifecycle*lc=nullptr;Effect*effects=nullptr;~DevicePolicyOwner(){if(nodes)cudaFree(nodes);if(children)cudaFree(children);if(bb)cudaFree(bb);if(values)cudaFree(values);if(feature_args)cudaFree(feature_args);if(pred)cudaFree(pred);if(lc)cudaFree(lc);if(effects)cudaFree(effects);}};
template<class T>static T*copy_dev(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
struct GameTickOut{uint8_t status=0,pad=0;uint16_t fault=0;ActionResult action{};int32_t blackboard[MAX_BB];NodeState state[MAX_STATE];};
struct GameFinal{int step=0,done=0;double reward0=0,reward1=0;};
__global__ void init_game_kernel(Config cfg,DeviceTables tables,PolicyView p,Sim*sim,RuntimeState*st){if(blockIdx.x||threadIdx.x)return;sim->init(cfg,&tables);reset_runtime(p,*st);}
__global__ void step_game_kernel(PolicyView p,Sim*sim,RuntimeState*st,GameTickOut*out,int seat){
    if(blockIdx.x||threadIdx.x||sim->st.done)return;CudaObservation obs(make_observation(*sim,seat));ActionResult ar{};Status status=tick(p,*st,obs,sim->cfg.max_orders,ar,nullptr);out->status=(uint8_t)status;out->fault=st->fault;out->action=ar;for(int i=0;i<MAX_BB;++i)out->blackboard[i]=st->blackboard[i];for(int i=0;i<MAX_STATE;++i)out->state[i]=st->node_state[i];
    Action a0{},a1{};a0.n_units=sim->st.farms[0].n_units;a1.n_units=sim->st.farms[1].n_units;if(seat==0)to_cuda_action(ar,a0);else to_cuda_action(ar,a1);sim->step_with_unit_counts(a0,sim->st.farms[0].n_units,a1,sim->st.farms[1].n_units);
}
__global__ void final_game_kernel(const Sim*sim,GameFinal*out){if(blockIdx.x||threadIdx.x)return;out->step=sim->st.step;out->done=sim->st.done;out->reward0=sim->st.farms[0].money;out->reward1=sim->st.farms[1].money;}
int main(int argc,char**argv){
    try{
        if(argc<2||argc>5)throw std::runtime_error("usage: cuda_game_runner POLICY [SEED] [EPISODE_STEPS] [SEAT]");const uint64_t seed=argc>=3?std::strtoull(argv[2],nullptr,10):12345ull;const int steps=argc>=4?std::atoi(argv[3]):16;const int seat=argc>=5?std::atoi(argv[4]):0;if(seat<0||seat>1)throw std::runtime_error("seat must be 0/1");
        auto owned=load_compiled(argv[1]);auto hp=owned.view();DevicePolicyOwner d;d.nodes=copy_dev(owned.nodes);d.children=copy_dev(owned.child_refs);d.bb=copy_dev(owned.bb);d.values=copy_dev(owned.values);d.feature_args=copy_dev(owned.feature_args);d.pred=copy_dev(owned.predicates);d.lc=copy_dev(owned.lifecycles);d.effects=copy_dev(owned.effects);PolicyView p=hp;p.nodes=d.nodes;p.child_refs=d.children;p.bb=d.bb;p.values=d.values;p.feature_args=d.feature_args;p.predicates=d.pred;p.lifecycles=d.lc;p.effects=d.effects;
        configure_device();TablesOwner tables;Config cfg;cfg.seed=seed;cfg.episode_steps=steps;if(cfg.board_size!=BOARD_SIZE)throw std::runtime_error("Bonsai-KG requires boardSize=10");Sim*sim=nullptr;RuntimeState*st=nullptr;GameTickOut*dout=nullptr;GameFinal*dfin=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&sim,sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&st,sizeof(RuntimeState)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,steps*sizeof(GameTickOut)));KAGCUDA_CHECK(cudaMalloc((void**)&dfin,sizeof(GameFinal)));KAGCUDA_CHECK(cudaMemset(dout,0,steps*sizeof(GameTickOut)));init_game_kernel<<<1,1>>>(cfg,tables.dev,p,sim,st);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());for(int ti=0;ti<steps-1;++ti){step_game_kernel<<<1,1>>>(p,sim,st,dout+ti,seat);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());}final_game_kernel<<<1,1>>>(sim,dfin);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());GameFinal fin{};KAGCUDA_CHECK(cudaMemcpy(&fin,dfin,sizeof(fin),cudaMemcpyDeviceToHost));const int ticks=fin.step;std::vector<GameTickOut>out(ticks);KAGCUDA_CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(GameTickOut),cudaMemcpyDeviceToHost));for(int ti=0;ti<ticks;++ti){RuntimeState hs{};hs.fault=out[ti].fault;for(int i=0;i<MAX_BB;++i)hs.blackboard[i]=out[ti].blackboard[i];for(int i=0;i<MAX_STATE;++i)hs.node_state[i]=out[ti].state[i];bonsaikg_test::print_snapshot(std::cout,0,ti,(Status)out[ti].status,hs,out[ti].action,hp.n_bb,hp.n_state);}std::cerr<<"FINAL ticks="<<ticks<<" done="<<fin.done<<" seat="<<seat<<" reward0="<<fin.reward0<<" reward1="<<fin.reward1<<"\n";cudaFree(sim);cudaFree(st);cudaFree(dout);cudaFree(dfin);return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
