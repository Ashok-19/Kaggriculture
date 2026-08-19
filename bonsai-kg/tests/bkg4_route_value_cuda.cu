#include "../integrations/bkg4_champion/extended_route_pack.hpp"
#include "../integrations/stateful_v6/cuda_runtime.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

struct DevPack{const bonsaikg_v6::RouteDesc*r;const bonsaikg_v6::PackedAction*a;int n;__device__ const bonsaikg_v6::PackedAction*at(uint16_t id,int step)const{for(int i=0;i<n;++i)if(r[i].route_id==id){int s=step<0?0:step;if(s>=r[i].steps)s=r[i].steps-1;return a+r[i].first+s;}return nullptr;}};
__global__ void k(DevPack p,uint16_t base,int ov,int step,int n,kagcuda::Action*out,int*err){if(blockIdx.x||threadIdx.x)return;uint16_t route=ov?(uint16_t)ov:base;const auto*x=p.at(route,step);if(!x){*err=1;return;}bonsaikg_v6::materialize_cuda(*x,n,*out);*err=0;}

int main(int argc,char**argv){
    try{
        if(argc!=6)throw std::runtime_error("usage: bkg4_route_value_cuda PACK CONTROLLER_ROUTE OVERRIDE STEP N_UNITS");
        auto pack=bonsaikg4_champion::load_extended_route_pack(argv[1]);uint16_t base=(uint16_t)std::strtoul(argv[2],nullptr,10);int ov=std::atoi(argv[3]),step=std::atoi(argv[4]),n=std::atoi(argv[5]);
        bonsaikg_v6::RouteDesc*dr=nullptr;bonsaikg_v6::PackedAction*da=nullptr;kagcuda::Action*dout=nullptr;int*derr=nullptr;
        cudaMalloc((void**)&dr,pack.routes.size()*sizeof(*dr));cudaMalloc((void**)&da,pack.actions.size()*sizeof(*da));cudaMalloc((void**)&dout,sizeof(*dout));cudaMalloc((void**)&derr,sizeof(*derr));
        cudaMemcpy(dr,pack.routes.data(),pack.routes.size()*sizeof(*dr),cudaMemcpyHostToDevice);cudaMemcpy(da,pack.actions.data(),pack.actions.size()*sizeof(*da),cudaMemcpyHostToDevice);
        k<<<1,1>>>({dr,da,(int)pack.routes.size()},base,ov,step,n,dout,derr);if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("cuda kernel failed");kagcuda::Action a{};int err=0;cudaMemcpy(&a,dout,sizeof(a),cudaMemcpyDeviceToHost);cudaMemcpy(&err,derr,sizeof(err),cudaMemcpyDeviceToHost);if(err)throw std::runtime_error("route missing");
        uint16_t route=ov?(uint16_t)ov:base;std::cout<<route<<" U="<<a.n_units<<":";for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}std::cout<<" M="<<a.n_orders<<":";for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}std::cout<<'\n';
        cudaFree(dr);cudaFree(da);cudaFree(dout);cudaFree(derr);return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
