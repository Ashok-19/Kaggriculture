#include "../integrations/stateful_v6/cuda_runtime.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ void materialize_all(const bonsaikg_v6::PackedAction*src,int n,kagcuda::Action*out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    bonsaikg_v6::materialize_cuda(src[i],1+src[i].saved_hands,out[i]);
}
static void print_action(size_t idx,uint16_t rid,int step,const kagcuda::Action&a){
    std::cout<<"I="<<idx<<" R="<<rid<<" S="<<step<<" U="<<a.n_units<<":";
    for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}
    std::cout<<" M="<<a.n_orders<<":";
    for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}
    std::cout<<'\n';
}
int main(int argc,char**argv){
    try{
        if(argc!=2)throw std::runtime_error("usage: stateful_v6_pack_cuda ROUTE_PACK");
        auto p=bonsaikg_v6::RoutePack::load(argv[1]);
        bonsaikg_v6::PackedAction*ds=nullptr;kagcuda::Action*do_=nullptr;const int n=(int)p.actions.size();
        KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)n*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&do_,(size_t)n*sizeof(*do_)));
        KAGCUDA_CHECK(cudaMemcpy(ds,p.actions.data(),(size_t)n*sizeof(*ds),cudaMemcpyHostToDevice));
        materialize_all<<<(n+127)/128,128>>>(ds,n,do_);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());
        std::vector<kagcuda::Action>out(n);KAGCUDA_CHECK(cudaMemcpy(out.data(),do_,(size_t)n*sizeof(*do_),cudaMemcpyDeviceToHost));
        size_t idx=0;for(const auto&d:p.routes)for(int s=0;s<d.steps;++s,++idx)print_action(idx,d.route_id,s,out[idx]);
        std::cerr<<"ROUTES="<<p.routes.size()<<" ACTIONS="<<idx<<"\n";cudaFree(ds);cudaFree(do_);return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
