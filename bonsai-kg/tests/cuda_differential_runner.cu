#include "../include/bonsai_kg.hpp"
#include "../include/bonsai_kg_io.hpp"
#include "synthetic_io.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bonsaikg;
using namespace bonsaikg_test;

struct DevicePolicyOwner{
    Node*nodes=nullptr;uint16_t*children=nullptr;BbDecl*bb=nullptr;Predicate*pred=nullptr;Lifecycle*lc=nullptr;Effect*effects=nullptr;
    ~DevicePolicyOwner(){if(nodes)cudaFree(nodes);if(children)cudaFree(children);if(bb)cudaFree(bb);if(pred)cudaFree(pred);if(lc)cudaFree(lc);if(effects)cudaFree(effects);}
};
struct TickOut{uint8_t status=0;uint8_t pad[3]={0,0,0};uint16_t fault=0,pad2=0;ActionResult action{};int32_t blackboard[MAX_BB];NodeState state[MAX_STATE];};

template<class T> static T* copy_dev(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;cudaError_t e=cudaMalloc((void**)&d,v.size()*sizeof(T));if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));e=cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice);if(e!=cudaSuccess){cudaFree(d);throw std::runtime_error(cudaGetErrorString(e));}return d;}

__global__ void run_case_kernel(PolicyView p,const SynFrame*frames,int n,TickOut*out){
    if(blockIdx.x||threadIdx.x)return;RuntimeState st{};reset_runtime(p,st);
    for(int i=0;i<n;++i){SyntheticObservation obs{frames+i};ActionResult a{};Status r=tick(p,st,obs,10,a,nullptr);TickOut&o=out[i];o.status=(uint8_t)r;o.fault=st.fault;o.action=a;for(int j=0;j<MAX_BB;++j)o.blackboard[j]=st.blackboard[j];for(int j=0;j<MAX_STATE;++j)o.state[j]=st.node_state[j];}
}

static void check(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}

int main(int argc,char**argv){
    try{
        if(argc!=2)throw std::runtime_error("usage: cuda_differential_runner MANIFEST");auto cases=load_manifest(argv[1]);
        for(size_t ci=0;ci<cases.size();++ci){
            auto owned=load_compiled(cases[ci].first);auto hp=owned.view();auto frames=load_sequence(cases[ci].second);DevicePolicyOwner d;
            d.nodes=copy_dev(owned.nodes);d.children=copy_dev(owned.child_refs);d.bb=copy_dev(owned.bb);d.pred=copy_dev(owned.predicates);d.lc=copy_dev(owned.lifecycles);d.effects=copy_dev(owned.effects);
            PolicyView p=hp;p.nodes=d.nodes;p.child_refs=d.children;p.bb=d.bb;p.predicates=d.pred;p.lifecycles=d.lc;p.effects=d.effects;
            SynFrame*df=nullptr;TickOut*do_=nullptr;check(cudaMalloc((void**)&df,frames.size()*sizeof(SynFrame)),"cudaMalloc frames");check(cudaMalloc((void**)&do_,frames.size()*sizeof(TickOut)),"cudaMalloc output");check(cudaMemcpy(df,frames.data(),frames.size()*sizeof(SynFrame),cudaMemcpyHostToDevice),"copy frames");
            run_case_kernel<<<1,1>>>(p,df,(int)frames.size(),do_);check(cudaGetLastError(),"kernel launch");check(cudaDeviceSynchronize(),"kernel sync");std::vector<TickOut>out(frames.size());check(cudaMemcpy(out.data(),do_,out.size()*sizeof(TickOut),cudaMemcpyDeviceToHost),"copy output");cudaFree(df);cudaFree(do_);
            for(size_t ti=0;ti<out.size();++ti){RuntimeState st{};st.fault=out[ti].fault;for(int j=0;j<MAX_BB;++j)st.blackboard[j]=out[ti].blackboard[j];for(int j=0;j<MAX_STATE;++j)st.node_state[j]=out[ti].state[j];print_snapshot(std::cout,(int)ci,(int)ti,(Status)out[ti].status,st,out[ti].action,hp.n_bb,hp.n_state);}
        }
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
