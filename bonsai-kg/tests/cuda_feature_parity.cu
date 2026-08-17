#include "../cuda/kag_adapter.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include "feature_hash.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace kagcuda;

struct FeatureSnapshot { uint64_t hash; int32_t raw[bonsaikg_test::TARGETED_FEATURE_COUNT]; };

__device__ inline Action scripted(const Sim&sim){
    Action a{}; a.n_units=sim.st.farms[0].n_units; const int s=sim.st.step;
    if(s==0){a.n_orders=4;a.orders[0]={M_BUY_SEED,WHEAT,3};a.orders[1]={M_BUY_PRODUCT,WHEAT,2};a.orders[2]={M_BUY_PRODUCT,FERTILIZER,1};a.orders[3]={M_HIRE,0,1};}
    else if(s==1)a.units[0]={OP_PICKUP,WHEAT,1};
    else if(s==2)a.units[0]={OP_PLANT,WHEAT,1};
    else if(s==3)a.units[0]={OP_WATER,0,1};
    else if(s==4){a.units[0]={OP_PICKUP,FERTILIZER,1};a.n_orders=1;a.orders[0]={M_BUY_ANIMAL,GOOSE,1};}
    else if(s==5)a.units[0]={OP_FERTILIZE,0,1};
    else if(s==6)a.units[0]={OP_PICKUP,GOOSE,1};
    else if(s==7)a.units[0]={OP_WEST,0,1};
    else if(s==8)a.units[0]={OP_BUILD_COOP,0,1};
    else if(s==9)a.units[0]={OP_PLACE,GOOSE,1};
    else if(s==10)a.units[0]={OP_CARE,0,1};
    else if(s==11)a.units[0]={OP_FEED,0,1};
    return a;
}

__global__ void init_kernel(Config cfg,DeviceTables tables,Sim*sim){if(!blockIdx.x&&!threadIdx.x)sim->init(cfg,&tables);}
__global__ void step_kernel(Sim*sim,FeatureSnapshot*out){
    if(blockIdx.x||threadIdx.x||sim->st.done)return;
    bonsaikg::CudaObservation obs(make_observation(*sim,0));
    FeatureSnapshot& snap=out[sim->st.step]; snap.hash=bonsaikg_test::hash_feature_registry(obs); bonsaikg_test::targeted_features(obs,snap.raw);
    Action a0=scripted(*sim),a1{};a1.n_units=sim->st.farms[1].n_units;
    sim->step_with_unit_counts(a0,sim->st.farms[0].n_units,a1,sim->st.farms[1].n_units);
}

int main(int argc,char**argv){
    configure_device(); TablesOwner tables;
    const uint64_t seed=argc>1?std::strtoull(argv[1],nullptr,10):12345ull; const int steps=argc>2?std::atoi(argv[2]):80;
    Config cfg; cfg.seed=seed; cfg.episode_steps=steps;
    if(cfg.board_size!=bonsaikg::BOARD_SIZE){std::cerr<<"boardSize mismatch\n";return 2;}
    Sim*sim=nullptr; FeatureSnapshot*dout=nullptr; KAGCUDA_CHECK(cudaMalloc((void**)&sim,sizeof(Sim))); KAGCUDA_CHECK(cudaMalloc((void**)&dout,steps*sizeof(FeatureSnapshot)));
    init_kernel<<<1,1>>>(cfg,tables.dev,sim); KAGCUDA_CHECK(cudaDeviceSynchronize());
    for(int i=0;i<steps-1;++i){step_kernel<<<1,1>>>(sim,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());}
    std::vector<FeatureSnapshot>out(steps-1);KAGCUDA_CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(FeatureSnapshot),cudaMemcpyDeviceToHost));
    for(size_t t=0;t<out.size();++t){std::cout<<"T="<<t<<" H="<<out[t].hash<<" R=";for(int i=0;i<bonsaikg_test::TARGETED_FEATURE_COUNT;++i){if(i)std::cout<<',';std::cout<<out[t].raw[i];}std::cout<<"\n";}
    cudaFree(sim);cudaFree(dout);return 0;
}
