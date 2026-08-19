#include "../include/bonsai_kg.hpp"
#include "../include/bonsai_kg_v4.hpp"
#include "../include/bonsai_kg_v4_io.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace b3 = bonsaikg;
namespace b4 = bonsaikg4;

struct Obs {
    int step=0;
    __host__ __device__ inline int32_t feature(uint16_t id,int32_t arg0=0,int32_t arg1=0) const {
        (void)arg1;
        if(id==b3::F_ALWAYS)return 1;
        if(id==b3::F_STEP)return step;
        if(id==b3::F_DAY)return step/24;
        if(id==b3::F_HOUR)return step%24;
        if(id==b3::F_OWN_UNITS||id==b3::F_OPP_UNITS)return 1;
        if(id==b3::F_OWN_WORKER_X||id==b3::F_OPP_WORKER_X)return arg0==0?4:-1;
        if(id==b3::F_OWN_WORKER_Y||id==b3::F_OPP_WORKER_Y)return arg0==0?4:-1;
        return 0;
    }
};

struct Tiny3 {
    b3::Node nodes[5]{};
    uint16_t child_refs[4]{};
    b3::ValueRef values[4]{};
    b3::PolicyView view() const {
        b3::PolicyView p{};p.nodes=nodes;p.child_refs=child_refs;p.values=values;p.root=0;p.n_nodes=5;
        p.n_child_refs=4;p.n_state=2;p.n_values=4;p.instruction_budget=64;p.board_size=b3::BOARD_SIZE;p.max_intents=1;return p;
    }
};
static Tiny3 tiny3(){
    Tiny3 o{};
    o.nodes[0].type=b3::N_REACTIVE_SELECTOR;o.nodes[0].first_child=0;o.nodes[0].child_count=2;o.nodes[0].state_slot=0;o.nodes[0].lifecycle_slot=b3::NONE_U16;o.nodes[0].subtree_end=5;
    o.nodes[1].type=b3::N_SEQUENCE;o.nodes[1].first_child=2;o.nodes[1].child_count=2;o.nodes[1].state_slot=1;o.nodes[1].lifecycle_slot=b3::NONE_U16;o.nodes[1].subtree_end=4;
    o.nodes[2].type=b3::N_CONDITION;o.nodes[2].flags=b3::C_GE;o.nodes[2].state_slot=b3::NONE_U16;o.nodes[2].lifecycle_slot=b3::NONE_U16;o.nodes[2].subtree_end=3;o.nodes[2].p0=0;o.nodes[2].p1=1;
    o.nodes[3].type=b3::N_SUCCESS;o.nodes[3].state_slot=b3::NONE_U16;o.nodes[3].lifecycle_slot=b3::NONE_U16;o.nodes[3].subtree_end=4;
    o.nodes[4].type=b3::N_MOVE_UNIT_TO;o.nodes[4].state_slot=b3::NONE_U16;o.nodes[4].lifecycle_slot=b3::NONE_U16;o.nodes[4].subtree_end=5;o.nodes[4].p0=2;o.nodes[4].p1=3;o.nodes[4].p2=100;
    o.child_refs[0]=1;o.child_refs[1]=4;o.child_refs[2]=2;o.child_refs[3]=3;
    o.values[0]={b3::V_LEGAL_FEATURE,b3::B_I32,b3::F_STEP,0,0};o.values[1]={b3::V_CONST,b3::B_I32,0,0,1000000000};o.values[2]={b3::V_CONST,b3::B_UNIT,0,0,0};o.values[3]={b3::V_CONST,b3::B_TILE,0,0,0};
    return o;
}

static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
template<class T>T* copy_arr(const T*src,size_t n){if(!n)return nullptr;T*d=nullptr;ck(cudaMalloc((void**)&d,n*sizeof(T)),"malloc");ck(cudaMemcpy(d,src,n*sizeof(T),cudaMemcpyHostToDevice),"copy");return d;}
template<class T>T* copy_vec(const std::vector<T>&v){return copy_arr(v.data(),v.size());}

__global__ void kernel3(b3::PolicyView p,b3::RuntimeState*states,int it,uint32_t*out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;b3::RuntimeState&st=states[id];b3::reset_runtime(p,st);uint32_t sum=0;
    for(int i=0;i<it;++i){Obs o{i};b3::ActionResult a{};auto r=b3::tick(p,st,o,10,a,nullptr);sum+=(uint32_t)r+(uint32_t)a.n_market+(uint32_t)a.units[0].op;}out[id]=sum+st.ticks;
}
__global__ void kernel4(b4::PolicyView4 p,int32_t*bb,b4::NodeState4*ns,b4::RuntimeMeta4*meta,int it,uint32_t*out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;b4::RuntimeView4 st{p.n_bb?bb+(size_t)id*p.n_bb:nullptr,p.n_state?ns+(size_t)id*p.n_state:nullptr,meta+id};b4::reset_runtime(p,st);uint32_t sum=0;
    for(int i=0;i<it;++i){Obs o{i};b4::ActionResult4 a{};auto r=b4::tick_specialized4<Obs,8,16>(p,st,o,10,a,nullptr);sum+=(uint32_t)r+(uint32_t)a.n_market+(uint32_t)a.units[0].op;}out[id]=sum+meta[id].ticks;
}

template<class Launch>static double timed_gpu(Launch launch){cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");ck(cudaEventRecord(a),"record");launch();ck(cudaGetLastError(),"launch");ck(cudaEventRecord(b),"record2");ck(cudaEventSynchronize(b),"sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");cudaEventDestroy(a);cudaEventDestroy(b);return ms;}

int main(int argc,char**argv){try{
    if(argc<2||argc>6)throw std::runtime_error("usage: v4_bkg3_comparison V4_POLICY [CPU_ITERS] [GPU_ENVS] [GPU_ITERS] [REPEATS]");
    int cpu=argc>2?std::atoi(argv[2]):200000,envs=argc>3?std::atoi(argv[3]):4096,it=argc>4?std::atoi(argv[4]):64,reps=argc>5?std::atoi(argv[5]):5;
    Tiny3 h3=tiny3();auto p3=h3.view();auto o4=b4::load_compiled4(argv[1]);auto p4=o4.view();
    if(p4.n_nodes!=5||p4.n_state!=1||p4.max_live_intents>8||p4.max_depth>16)throw std::runtime_error("unexpected BKG4 tiny fixture shape");
    b3::RuntimeState s3{};b3::reset_runtime(p3,s3);std::vector<int32_t>bb4(p4.n_bb);std::vector<b4::NodeState4>ns4(p4.n_state);b4::RuntimeMeta4 m4{};b4::RuntimeView4 s4{bb4.data(),ns4.data(),&m4};b4::reset_runtime(p4,s4);
    auto cpu_run3=[&](){uint64_t z=0;auto t0=std::chrono::steady_clock::now();for(int i=0;i<cpu;++i){Obs o{i};b3::ActionResult a{};auto r=b3::tick(p3,s3,o,10,a,nullptr);z+=(unsigned)r+(unsigned)a.n_market+(unsigned)a.units[0].op;}double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return std::pair<double,uint64_t>{cpu/sec,z};};
    auto cpu_run4=[&](){uint64_t z=0;auto t0=std::chrono::steady_clock::now();for(int i=0;i<cpu;++i){Obs o{i};b4::ActionResult4 a{};auto r=b4::tick4(p4,s4,o,10,a,nullptr);z+=(unsigned)r+(unsigned)a.n_market+(unsigned)a.units[0].op;}double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return std::pair<double,uint64_t>{cpu/sec,z};};
    auto c3=cpu_run3(),c4=cpu_run4();if(c3.second!=c4.second)throw std::runtime_error("CPU semantic checksum mismatch");
    b3::Node*dn3=copy_arr(h3.nodes,5);uint16_t*dc3=copy_arr(h3.child_refs,4);b3::ValueRef*dv3=copy_arr(h3.values,4);auto d3=p3;d3.nodes=dn3;d3.child_refs=dc3;d3.values=dv3;
    b4::Node4*dn4=copy_vec(o4.nodes);uint16_t*dc4=copy_vec(o4.child_refs);b4::BbDecl4*db4=copy_vec(o4.bb);b4::ValueRef4*dv4=copy_vec(o4.values);b4::FeatureArgRef4*da4=copy_vec(o4.feature_args);b4::Predicate4*dq4=copy_vec(o4.predicates);b4::Lifecycle4*dl4=copy_vec(o4.lifecycles);b4::Effect4*de4=copy_vec(o4.effects);auto d4=p4;d4.nodes=dn4;d4.child_refs=dc4;d4.bb=db4;d4.values=dv4;d4.feature_args=da4;d4.predicates=dq4;d4.lifecycles=dl4;d4.effects=de4;
    b3::RuntimeState*st3=nullptr;b4::NodeState4*st4=nullptr;int32_t*bbd4=nullptr;b4::RuntimeMeta4*md4=nullptr;uint32_t*out=nullptr;ck(cudaMalloc((void**)&st3,(size_t)envs*sizeof(b3::RuntimeState)),"st3");if(p4.n_state)ck(cudaMalloc((void**)&st4,(size_t)envs*p4.n_state*sizeof(b4::NodeState4)),"st4");if(p4.n_bb)ck(cudaMalloc((void**)&bbd4,(size_t)envs*p4.n_bb*sizeof(int32_t)),"bb4");ck(cudaMalloc((void**)&md4,(size_t)envs*sizeof(b4::RuntimeMeta4)),"meta4");ck(cudaMalloc((void**)&out,(size_t)envs*sizeof(uint32_t)),"out");
    std::vector<double>g3,g4;for(int r=0;r<reps;++r){double a=timed_gpu([&](){kernel3<<<(envs+127)/128,128>>>(d3,st3,it,out);});double b=timed_gpu([&](){kernel4<<<(envs+127)/128,128>>>(d4,bbd4,st4,md4,it,out);});g3.push_back((double)envs*it/(a/1000.0));g4.push_back((double)envs*it/(b/1000.0));}
    std::vector<uint32_t>gpu3(envs),gpu4(envs);kernel3<<<(envs+127)/128,128>>>(d3,st3,it,out);ck(cudaGetLastError(),"semantic kernel3");ck(cudaDeviceSynchronize(),"semantic sync3");ck(cudaMemcpy(gpu3.data(),out,(size_t)envs*sizeof(uint32_t),cudaMemcpyDeviceToHost),"semantic copy3");kernel4<<<(envs+127)/128,128>>>(d4,bbd4,st4,md4,it,out);ck(cudaGetLastError(),"semantic kernel4");ck(cudaDeviceSynchronize(),"semantic sync4");ck(cudaMemcpy(gpu4.data(),out,(size_t)envs*sizeof(uint32_t),cudaMemcpyDeviceToHost),"semantic copy4");if(gpu3!=gpu4)throw std::runtime_error("GPU semantic checksum mismatch");
    auto med=[](std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];};double m3=med(g3),m4v=med(g4);
    size_t rb3=sizeof(b3::RuntimeState),rb4=(size_t)p4.n_bb*4+(size_t)p4.n_state*8+sizeof(b4::RuntimeMeta4);
    std::cout<<"cpu_bkg3_ticks_s="<<c3.first<<" cpu_bkg4_ticks_s="<<c4.first<<" cpu_ratio="<<(c4.first/c3.first)
             <<" gpu_bkg3_median_ticks_s="<<m3<<" gpu_bkg4_median_ticks_s="<<m4v<<" gpu_ratio="<<(m4v/m3)
             <<" bkg3_runtime_bytes_env="<<rb3<<" bkg4_runtime_bytes_env="<<rb4<<" runtime_ratio="<<((double)rb4/rb3)
             <<" repeats="<<reps<<" checksum="<<c3.second<<"\n";
    cudaFree(dn3);cudaFree(dc3);cudaFree(dv3);cudaFree(dn4);cudaFree(dc4);if(db4)cudaFree(db4);cudaFree(dv4);if(da4)cudaFree(da4);if(dq4)cudaFree(dq4);if(dl4)cudaFree(dl4);if(de4)cudaFree(de4);cudaFree(st3);if(st4)cudaFree(st4);if(bbd4)cudaFree(bbd4);cudaFree(md4);cudaFree(out);return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
