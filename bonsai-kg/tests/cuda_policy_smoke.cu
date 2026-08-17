#include "../cuda/policy.cuh"
#include "../../cuda-engine/include/runtime.cuh"
#include "../../cuda-engine/include/policy_runner.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace kagcuda;
using namespace bonsaikg;

struct PassState { int calls=0; };
struct PassPolicy {
    __device__ inline void reset(PassState& s,const ObservationView& obs) const { s.calls=obs.step(); }
    __device__ inline void act(PassState& s,const ObservationView&,Action& out) const { ++s.calls; out.n_orders=0; }
};

static double run_bonsai(Config cfg,DeviceTables tables,CudaPolicy policy,int n_envs){
    Sim*sims=nullptr;CudaPolicyState*s0=nullptr;PassState*s1=nullptr;double*rewards=nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&sims,(size_t)n_envs*sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc((void**)&s0,(size_t)n_envs*sizeof(CudaPolicyState)));
    KAGCUDA_CHECK(cudaMalloc((void**)&s1,(size_t)n_envs*sizeof(PassState)));
    KAGCUDA_CHECK(cudaMalloc((void**)&rewards,(size_t)n_envs*2*sizeof(double)));
    cudaEvent_t a,b;KAGCUDA_CHECK(cudaEventCreate(&a));KAGCUDA_CHECK(cudaEventCreate(&b));KAGCUDA_CHECK(cudaEventRecord(a));
    run_policy_batch_kernel<<<(n_envs+127)/128,128>>>(cfg,tables,n_envs,policy,s0,PassPolicy{},s1,sims,rewards);
    KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(b));KAGCUDA_CHECK(cudaEventSynchronize(b));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
    std::vector<double>host((size_t)n_envs*2);KAGCUDA_CHECK(cudaMemcpy(host.data(),rewards,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    for(double v:host)if(v!=3000.0){std::fprintf(stderr,"Bonsai policy smoke reward mismatch %.0f\n",v);std::exit(1);}
    cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(sims);cudaFree(s0);cudaFree(s1);cudaFree(rewards);return ms;
}

static double run_pass(Config cfg,DeviceTables tables,int n_envs){
    Sim*sims=nullptr;PassState*s0=nullptr,*s1=nullptr;double*rewards=nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&sims,(size_t)n_envs*sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc((void**)&s0,(size_t)n_envs*sizeof(PassState)));
    KAGCUDA_CHECK(cudaMalloc((void**)&s1,(size_t)n_envs*sizeof(PassState)));
    KAGCUDA_CHECK(cudaMalloc((void**)&rewards,(size_t)n_envs*2*sizeof(double)));
    cudaEvent_t a,b;KAGCUDA_CHECK(cudaEventCreate(&a));KAGCUDA_CHECK(cudaEventCreate(&b));KAGCUDA_CHECK(cudaEventRecord(a));
    run_policy_batch_kernel<<<(n_envs+127)/128,128>>>(cfg,tables,n_envs,PassPolicy{},s0,PassPolicy{},s1,sims,rewards);
    KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(b));KAGCUDA_CHECK(cudaEventSynchronize(b));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
    cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(sims);cudaFree(s0);cudaFree(s1);cudaFree(rewards);return ms;
}

int main(int argc,char**argv){
    configure_device();TablesOwner tables;const int n_envs=argc>1?std::atoi(argv[1]):4096;const int steps=argc>2?std::atoi(argv[2]):96;
    Node hnode{};hnode.type=N_SUCCESS;hnode.state_slot=NONE_U16;hnode.lifecycle_slot=NONE_U16;Node*dnode=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&dnode,sizeof(Node)));KAGCUDA_CHECK(cudaMemcpy(dnode,&hnode,sizeof(Node),cudaMemcpyHostToDevice));
    PolicyView p{};p.nodes=dnode;p.root=0;p.n_nodes=1;p.instruction_budget=64;CudaPolicy policy{p,10};Config cfg;cfg.seed=1000;cfg.episode_steps=steps;
    const double pass_ms=run_pass(cfg,tables.dev,n_envs);const double bonsai_ms=run_bonsai(cfg,tables.dev,policy,n_envs);const double turns=(double)n_envs*(steps-1);
    std::printf("PASS exact batched Bonsai-KG CUDA policy: envs=%d steps=%d pass_ms=%.3f bonsai_ms=%.3f pass_turns_s=%.0f bonsai_turns_s=%.0f overhead_ratio=%.4f runtime_state_bytes=%zu\n",n_envs,steps,pass_ms,bonsai_ms,turns/(pass_ms/1000.0),turns/(bonsai_ms/1000.0),bonsai_ms/pass_ms,sizeof(CudaPolicyState));
    cudaFree(dnode);return 0;
}
