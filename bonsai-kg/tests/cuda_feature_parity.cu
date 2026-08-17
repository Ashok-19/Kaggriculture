#include "../cuda/kag_adapter.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include "feature_hash.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace kagcuda;

__device__ inline Action scripted(const Sim&sim){Action a{};a.n_units=sim.st.farms[0].n_units;int s=sim.st.step;
    if(s==0){a.n_orders=3;a.orders[0]={M_BUY_SEED,WHEAT,3};a.orders[1]={M_BUY_PRODUCT,WHEAT,2};a.orders[2]={M_HIRE,0,1};}
    else if(s==1)a.units[0]={OP_PICKUP,WHEAT,1};else if(s==2)a.units[0]={OP_PLANT,WHEAT,1};else if(s==3)a.units[0]={OP_WATER,0,1};
    else if(s==4){a.n_orders=1;a.orders[0]={M_BUY_ANIMAL,GOOSE,1};}else if(s==5)a.units[0]={OP_PICKUP,GOOSE,1};else if(s==6)a.units[0]={OP_WEST,0,1};
    else if(s==7)a.units[0]={OP_BUILD_COOP,0,1};else if(s==8)a.units[0]={OP_PLACE,GOOSE,1};else if(s==9)a.units[0]={OP_CARE,0,1};return a;}

__global__ void init_kernel(Config cfg,DeviceTables tables,Sim*sim){if(!blockIdx.x&&!threadIdx.x)sim->init(cfg,&tables);}
__global__ void step_kernel(Sim*sim,uint64_t*out){if(blockIdx.x||threadIdx.x||sim->st.done)return;bonsaikg::CudaObservation obs(make_observation(*sim,0));out[sim->st.step]=bonsaikg_test::hash_feature_registry(obs);Action a0=scripted(*sim),a1{};a1.n_units=sim->st.farms[1].n_units;sim->step_with_unit_counts(a0,sim->st.farms[0].n_units,a1,sim->st.farms[1].n_units);}

int main(int argc,char**argv){configure_device();TablesOwner tables;uint64_t seed=argc>1?std::strtoull(argv[1],nullptr,10):12345ull;int steps=argc>2?std::atoi(argv[2]):80;Config cfg;cfg.seed=seed;cfg.episode_steps=steps;Sim*sim=nullptr;uint64_t*dout=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&sim,sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,steps*sizeof(uint64_t)));init_kernel<<<1,1>>>(cfg,tables.dev,sim);KAGCUDA_CHECK(cudaDeviceSynchronize());for(int i=0;i<steps-1;++i){step_kernel<<<1,1>>>(sim,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());}std::vector<uint64_t>out(steps-1);KAGCUDA_CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));for(size_t i=0;i<out.size();++i)std::cout<<"T="<<i<<" H="<<out[i]<<"\n";cudaFree(sim);cudaFree(dout);return 0;}
