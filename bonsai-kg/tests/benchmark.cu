#include "../include/bonsai_kg.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace bonsaikg;

struct BenchObs {
    int step = 0;
#ifdef __CUDACC__
    __host__ __device__ inline
#else
    inline
#endif
    int32_t feature(uint16_t id, int32_t arg, int32_t = 0) const {
        if (id == F_ALWAYS) return 1;
        if (id == F_STEP) return step;
        if (id == F_DAY) return step / 24;
        if (id == F_HOUR) return step % 24;
        if (id == F_OWN_UNITS || id == F_OPP_UNITS) return 1;
        if (id == F_OWN_WORKER_X || id == F_OPP_WORKER_X) return arg == 0 ? 4 : -1;
        if (id == F_OWN_WORKER_Y || id == F_OPP_WORKER_Y) return arg == 0 ? 4 : -1;
        return 0;
    }
};

struct OwnedTiny {
    Node nodes[5]{};
    uint16_t child_refs[4]{};
    ValueRef values[4]{};
    PolicyView view() const {
        PolicyView p{}; p.nodes=nodes; p.child_refs=child_refs; p.values=values; p.root=0; p.n_nodes=5;
        p.n_child_refs=4; p.n_state=2; p.n_values=4; p.instruction_budget=64; p.board_size=BOARD_SIZE; p.max_intents=1; return p;
    }
};

static OwnedTiny make_policy(){
    OwnedTiny o{};
    o.nodes[0].type=N_REACTIVE_SELECTOR; o.nodes[0].first_child=0; o.nodes[0].child_count=2; o.nodes[0].state_slot=0; o.nodes[0].lifecycle_slot=NONE_U16; o.nodes[0].subtree_end=5;
    o.nodes[1].type=N_SEQUENCE; o.nodes[1].first_child=2; o.nodes[1].child_count=2; o.nodes[1].state_slot=1; o.nodes[1].lifecycle_slot=NONE_U16; o.nodes[1].subtree_end=4;
    o.nodes[2].type=N_CONDITION; o.nodes[2].flags=C_GE; o.nodes[2].state_slot=NONE_U16; o.nodes[2].lifecycle_slot=NONE_U16; o.nodes[2].subtree_end=3; o.nodes[2].p0=0; o.nodes[2].p1=1;
    o.nodes[3].type=N_SUCCESS; o.nodes[3].state_slot=NONE_U16; o.nodes[3].lifecycle_slot=NONE_U16; o.nodes[3].subtree_end=4;
    o.nodes[4].type=N_MOVE_UNIT_TO; o.nodes[4].state_slot=NONE_U16; o.nodes[4].lifecycle_slot=NONE_U16; o.nodes[4].subtree_end=5; o.nodes[4].p0=2; o.nodes[4].p1=3; o.nodes[4].p2=100;
    o.child_refs[0]=1; o.child_refs[1]=4; o.child_refs[2]=2; o.child_refs[3]=3;
    o.values[0]={V_LEGAL_FEATURE,B_I32,F_STEP,0,0};
    o.values[1]={V_CONST,B_I32,0,0,1000000000};
    o.values[2]={V_CONST,B_UNIT,0,0,0};
    o.values[3]={V_CONST,B_TILE,0,0,0};
    return o;
}

__global__ void bench_kernel(PolicyView p, RuntimeState* states, int iters, uint32_t* checksum){
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    RuntimeState& st=states[id]; reset_runtime(p,st); uint32_t sum=0;
    for(int i=0;i<iters;++i){ BenchObs obs{i}; ActionResult a; const Status r=tick(p,st,obs,10,a,nullptr); sum += static_cast<uint32_t>(r) + static_cast<uint32_t>(a.units[0].op); }
    checksum[id]=sum + st.ticks;
}

static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(2);}}

template<class T> static T* dev_copy(const T* src,size_t n){T*d=nullptr;ck(cudaMalloc((void**)&d,n*sizeof(T)),"cudaMalloc table");ck(cudaMemcpy(d,src,n*sizeof(T),cudaMemcpyHostToDevice),"cudaMemcpy table");return d;}

int main(int argc,char**argv){
    const int cpu_iters=argc>1?std::atoi(argv[1]):5000000;
    const int n_envs=argc>2?std::atoi(argv[2]):65536;
    const int gpu_iters=argc>3?std::atoi(argv[3]):64;
    OwnedTiny host=make_policy(); PolicyView hp=host.view(); RuntimeState hs{}; reset_runtime(hp,hs);
    uint64_t cpu_checksum=0; auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<cpu_iters;++i){BenchObs ob{i};ActionResult a;auto r=tick(hp,hs,ob,10,a,nullptr);cpu_checksum += static_cast<unsigned>(r)+static_cast<unsigned>(a.units[0].op);}
    auto t1=std::chrono::steady_clock::now(); double cpu_s=std::chrono::duration<double>(t1-t0).count();

    Node*dn=dev_copy(host.nodes,5);uint16_t*dc=dev_copy(host.child_refs,4);ValueRef*dv=dev_copy(host.values,4);PolicyView dp=hp;dp.nodes=dn;dp.child_refs=dc;dp.values=dv;
    RuntimeState*states=nullptr;uint32_t*checks=nullptr;ck(cudaMalloc((void**)&states,(size_t)n_envs*sizeof(RuntimeState)),"cudaMalloc states");ck(cudaMalloc((void**)&checks,(size_t)n_envs*sizeof(uint32_t)),"cudaMalloc checks");
    cudaEvent_t a,b;ck(cudaEventCreate(&a),"event create");ck(cudaEventCreate(&b),"event create");ck(cudaEventRecord(a),"event record");
    bench_kernel<<<(n_envs+255)/256,256>>>(dp,states,gpu_iters,checks);ck(cudaGetLastError(),"bench launch");ck(cudaEventRecord(b),"event record2");ck(cudaEventSynchronize(b),"event sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");
    std::vector<uint32_t> hc(n_envs);ck(cudaMemcpy(hc.data(),checks,hc.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost),"copy checks");uint64_t gpu_checksum=0;for(auto x:hc)gpu_checksum+=x;
    const double gpu_ticks=(double)n_envs*gpu_iters; const double state_mib=(double)n_envs*sizeof(RuntimeState)/(1024.0*1024.0);
    std::cout<<"cpu_ticks="<<cpu_iters<<" cpu_seconds="<<cpu_s<<" cpu_ticks_per_s="<<(cpu_iters/cpu_s)<<" cpu_checksum="<<cpu_checksum<<"\n";
    std::cout<<"gpu_envs="<<n_envs<<" gpu_iters="<<gpu_iters<<" gpu_ticks="<<(uint64_t)gpu_ticks<<" gpu_ms="<<ms<<" gpu_ticks_per_s="<<(gpu_ticks/(ms/1000.0))<<" state_mib="<<state_mib<<" gpu_checksum="<<gpu_checksum<<"\n";
    cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(dn);cudaFree(dc);cudaFree(dv);cudaFree(states);cudaFree(checks);return 0;
}
