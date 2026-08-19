#include "../../bonsai-kg/include/bonsai_kg_v4.hpp"
#include "../../bonsai-kg/include/bonsai_kg_v4_io.hpp"
#include "../../bonsai-kg/cuda/kag_adapter_v4.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/assets/identity.hpp"
#include "../../bonsai-kg/integrations/stateful_v6/route_pack.hpp"
#include "../include/runtime.cuh"
#include "../include/tape_io.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

namespace {
#pragma pack(push,1)
struct ResultHeader { char magic[8]; uint32_t version,genomes,cases; };
struct ResultBin { int32_t own_money,opp_money; uint16_t rewrites,flags; };
#pragma pack(pop)
static_assert(sizeof(ResultBin)==12,"result ABI");

struct CaseHost { std::string path; int seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0; };
struct CaseDev { int tape=0,seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0; };
struct OppDesc { int first=0,count=0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4* n=nullptr; uint16_t* c=nullptr; bonsaikg4::BbDecl4* bb=nullptr;
    bonsaikg4::ValueRef4* v=nullptr; bonsaikg4::FeatureArgRef4* a=nullptr;
    bonsaikg4::Predicate4* q=nullptr; bonsaikg4::Lifecycle4* l=nullptr; bonsaikg4::Effect4* e=nullptr;
    ~DevicePolicyOwner4(){if(n)cudaFree(n);if(c)cudaFree(c);if(bb)cudaFree(bb);if(v)cudaFree(v);if(a)cudaFree(a);if(q)cudaFree(q);if(l)cudaFree(l);if(e)cudaFree(e);}
};
template<class T> static T* cp(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
static std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost> load_cases(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open cases");std::vector<CaseHost>o;std::string line;while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;auto z=split_tab(line);if(z.size()!=6)throw std::runtime_error("case line needs 6 fields");o.push_back({z[0],std::stoi(z[1]),std::stoull(z[2]),std::stoi(z[3]),std::stoi(z[4]),std::stoi(z[5])});}if(o.empty())throw std::runtime_error("empty cases");return o;}

static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("legacy pack");
    std::vector<uint8_t>raw((std::istreambuf_iterator<char>(in)),{});
    auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy hash");
    bonsaikg_v6::PackHeader h{};if(raw.size()<sizeof(h))throw std::runtime_error("short legacy pack");std::memcpy(&h,raw.data(),sizeof(h));
    size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc);bonsaikg_v6::RoutePack o;o.routes.resize(h.route_count);o.actions.resize(h.action_count);
    std::memcpy(o.routes.data(),raw.data()+sizeof(h),db);std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0]));return o;
}

struct DeviceBasePack {
    const bonsaikg_v6::RouteDesc* routes=nullptr; const bonsaikg_v6::PackedAction* actions=nullptr; int n=0;
    __device__ __forceinline__ const bonsaikg_v6::PackedAction* at(uint16_t route,int step)const{
        for(int i=0;i<n;++i)if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;}return nullptr;
    }
};

__device__ __forceinline__ uint16_t reconcile_feed(const int32_t*bb,int first_slot,int count,kagcuda::Action&a){
    uint16_t changed=0;for(int u=0;u<count&&u<a.n_units;++u){if(!bb[first_slot+u]||a.units[u].op!=kagcuda::OP_FEED)continue;a.units[u]=kagcuda::UnitAction{};if(changed<65535)++changed;}return changed;
}

__global__ void eval_kernel(kagcuda::Config cfg0,kagcuda::DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack base_pack,bonsaikg4_champion::CudaLegacyPack legacy_pack,
                            const kagcuda::Action*opp,const OppDesc*od,const CaseDev*cases,int nc,int32_t*bb,bonsaikg4::NodeState4*ns,bonsaikg4::RuntimeMeta4*meta,
                            int feed_slot,int feed_count,kagcuda::Sim*sims,ResultBin*out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nc)return;const CaseDev c=cases[id];kagcuda::Config cfg=cfg0;cfg.seed=c.seed;kagcuda::Sim&sim=sims[id];sim.init(cfg,&tables);
    bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)id*policy.n_bb:nullptr,policy.n_state?ns+(size_t)id*policy.n_state:nullptr,meta+id};bonsaikg4::reset_runtime(policy,st);
    bonsaikg4_champion::CudaLegacyState legacy{};bonsaikg_v6::CudaEngineExploitState exploit{};bonsaikg4_champion::reset_legacy(legacy);bonsaikg_v6::reset_cuda_engine_exploits(exploit);uint16_t rew=0,flags=0;
    while(!sim.st.done){int step=sim.st.step,os=1-c.seat;auto raw=make_observation(sim,c.seat);bonsaikg4::CudaObservation4 obs4(raw);bonsaikg4::ActionResult4 ignored{};auto status=bonsaikg4::tick4(policy,st,obs4,0,ignored,nullptr);if(meta[id].fault||status!=bonsaikg4::SUCCESS){flags|=1;break;}
        int route=st.blackboard[1];kagcuda::Action ca{};const auto*p=base_pack.at((uint16_t)route,step);if(!p){flags|=2;break;}bonsaikg_v6::materialize_cuda(*p,raw.farm(c.seat).n_units(),ca);
        bonsaikg4_champion::apply_legacy(raw,legacy_pack,legacy,ca);bonsaikg_v6::apply_cuda_engine_exploits(raw,ca,exploit,legacy.external_ok!=0,100);uint16_t q=reconcile_feed(st.blackboard,feed_slot,feed_count,ca);rew=(uint16_t)imin(65535,(int)rew+(int)q);
        kagcuda::Action oa{};OppDesc d=od[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];}oa.n_units=sim.st.farms[os].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
    }
    ResultBin r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.rewrites=rew;r.flags=flags;out[id]=r;
}

static void write_results(const std::string&p,const std::vector<ResultBin>&r,int nc){std::ofstream o(p,std::ios::binary);ResultHeader h{{'B','4','M','R','1','\0','\0','\0'},1u,1u,(uint32_t)nc};o.write((char*)&h,sizeof(h));o.write((char*)r.data(),r.size()*sizeof(r[0]));}
}

int main(int argc,char**argv){try{
    std::string controller,routes,legacy,casesp,outp;int feed_slot=5,feed_count=13,tpb=64;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--out")outp=need();else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else throw std::runtime_error("unknown "+a);}if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||outp.empty())throw std::runtime_error("required args");
    auto owned=bonsaikg4::load_compiled4(controller);auto hp=owned.view();if(hp.n_bb<(uint16_t)(feed_slot+feed_count))throw std::runtime_error("feed slots outside policy BB");DevicePolicyOwner4 po;po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;
    auto rp=bonsaikg_v6::RoutePack::load(routes),lp=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lp.routes);auto*dla=cp(lp.actions);DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack ldp{dlr,dla,(int)lp.routes.size()};
    auto hc=load_cases(casesp);std::vector<kagcuda::Action>hoa;std::vector<OppDesc>hod;std::vector<CaseDev>hcd;for(const auto&c:hc){auto t=kagcuda::load_native_tape_json(c.path);int off=hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)hod.size()-1,c.seat,c.seed,c.family,c.protected_flag,c.own_history});}
    auto*doa=cp(hoa);auto*dod=cp(hod);auto*dc=cp(hcd);int nc=(int)hc.size();int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;kagcuda::Sim*ds=nullptr;ResultBin*dout=nullptr;if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,(size_t)nc*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,(size_t)nc*hp.n_state*sizeof(bonsaikg4::NodeState4)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,(size_t)nc*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)nc*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,(size_t)nc*sizeof(*dout)));
    configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));eval_kernel<<<(nc+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,ldp,doa,dod,dc,nc,dbb,dns,dm,feed_slot,feed_count,ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<ResultBin>hr(nc);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,hr.size()*sizeof(hr[0]),cudaMemcpyDeviceToHost));write_results(outp,hr,nc);int bad=0,rew=0;for(auto&r:hr){bad+=r.flags!=0;rew+=r.rewrites;}std::cout<<"cases="<<nc<<" gpu_ms="<<ms<<" games_s="<<(nc/(ms/1000.0))<<" policy_nodes="<<hp.n_nodes<<" bb="<<hp.n_bb<<" state="<<hp.n_state<<" bad="<<bad<<" rewrites="<<rew<<"\n";
    cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dc);if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
