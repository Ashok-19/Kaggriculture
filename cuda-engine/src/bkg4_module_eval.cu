#include "../../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/assets/identity.hpp"
#include "../include/runtime.cuh"
#include "../include/tape_io.hpp"
#include "../../bonsai-kg/include/bonsai_kg_io.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

namespace {

#pragma pack(push,1)
struct ModuleCondBin {
    uint16_t feature_id;
    uint8_t cmp;
    uint8_t reserved0;
    int16_t index;
    uint16_t reserved1;
    int32_t value;
};
struct ModuleRuleBin {
    uint8_t n_conditions;
    uint8_t reserved0;
    uint16_t route_id;
    ModuleCondBin conditions[4];
};
struct ModuleGenomeBin {
    uint16_t prefix_step;
    uint8_t n_rules;
    uint8_t reserved0;
    ModuleRuleBin rules[8];
};
struct GenomeFileHeader {
    char magic[8];
    uint32_t version;
    uint32_t count;
};
#pragma pack(pop)
static_assert(sizeof(ModuleCondBin)==12,"ModuleCondBin ABI drift");
static_assert(sizeof(ModuleRuleBin)==52,"ModuleRuleBin ABI drift");
static_assert(sizeof(ModuleGenomeBin)==420,"ModuleGenomeBin ABI drift");

struct CaseHost {
    std::string path;
    int seat=0;
    uint64_t seed=0;
    int family=0;
    int protected_flag=0;
    int own_history=0;
};
struct CaseDev {
    int tape=0;
    int seat=0;
    uint64_t seed=0;
    int family=0;
    int protected_flag=0;
    int own_history=0;
};
struct OppDesc { int first=0, count=0; };
struct Result {
    int32_t own_money=0;
    int32_t opp_money=0;
    uint16_t controller_fault=0;
    uint16_t compat_fault=0;
};

struct DevicePolicyOwner{
    bonsaikg::Node*nodes=nullptr;uint16_t*children=nullptr;bonsaikg::BbDecl*bb=nullptr;bonsaikg::ValueRef*values=nullptr;
    bonsaikg::FeatureArgRef*feature_args=nullptr;bonsaikg::Predicate*pred=nullptr;bonsaikg::Lifecycle*lc=nullptr;bonsaikg::Effect*effects=nullptr;
    ~DevicePolicyOwner(){if(nodes)cudaFree(nodes);if(children)cudaFree(children);if(bb)cudaFree(bb);if(values)cudaFree(values);if(feature_args)cudaFree(feature_args);if(pred)cudaFree(pred);if(lc)cudaFree(lc);if(effects)cudaFree(effects);}
};
template<class T>static T* copy_dev(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}

static std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;while(true){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost> load_cases(const std::string&path){
    std::ifstream in(path);if(!in)throw std::runtime_error("cannot open case manifest: "+path);std::vector<CaseHost> out;std::string line;
    while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;auto z=split_tab(line);if(z.size()!=6)throw std::runtime_error("case line must have 6 tab fields: path seat seed family protected own_history");CaseHost c;c.path=z[0];c.seat=std::stoi(z[1]);c.seed=std::stoull(z[2]);c.family=std::stoi(z[3]);c.protected_flag=std::stoi(z[4]);c.own_history=std::stoi(z[5]);if(c.seat<0||c.seat>1)throw std::runtime_error("case seat invalid");out.push_back(c);}
    if(out.empty())throw std::runtime_error("empty case manifest");return out;
}
static std::vector<ModuleGenomeBin> load_genomes(const std::string&path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open genome file: "+path);GenomeFileHeader h{};in.read((char*)&h,sizeof(h));const char magic[8]={'B','4','G','M','1','\0','\0','\0'};if(!in||std::memcmp(h.magic,magic,8)||h.version!=1||h.count==0||h.count>65536)throw std::runtime_error("invalid B4GM1 genome header");std::vector<ModuleGenomeBin> g(h.count);in.read((char*)g.data(),g.size()*sizeof(g[0]));char extra;if(!in||in.read(&extra,1))throw std::runtime_error("invalid/trailing genome payload");return g;
}

static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open legacy pack");std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),{});if(raw.size()<sizeof(bonsaikg_v6::PackHeader))throw std::runtime_error("short legacy pack");auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy pack SHA mismatch");bonsaikg_v6::PackHeader h{};std::memcpy(&h,raw.data(),sizeof(h));const char magic[8]={'B','K','V','6','R','P','1','\0'};if(std::memcmp(h.magic,magic,8)||h.version!=1||h.route_count!=11)throw std::runtime_error("legacy pack header mismatch");size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc),expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);if(raw.size()!=expected)throw std::runtime_error("legacy pack size mismatch");bonsaikg_v6::RoutePack out;out.routes.resize(h.route_count);out.actions.resize(h.action_count);std::memcpy(out.routes.data(),raw.data()+sizeof(h),db);std::memcpy(out.actions.data(),raw.data()+sizeof(h)+db,out.actions.size()*sizeof(out.actions[0]));return out;
}

__device__ __forceinline__ bool cmp_i32(int32_t a,int32_t b,uint8_t c){switch(c){case bonsaikg::C_EQ:return a==b;case bonsaikg::C_NE:return a!=b;case bonsaikg::C_LE:return a<=b;case bonsaikg::C_GE:return a>=b;default:return false;}}
__device__ __forceinline__ uint16_t module_route(const ModuleGenomeBin&g,const ObservationView&obs){
    if(obs.step()<g.prefix_step)return 0xffffu;const bonsaikg::CudaObservation legal(obs);int nr=g.n_rules;if(nr>8)nr=8;
    for(int r=0;r<nr;++r){const auto&rule=g.rules[r];int nc=rule.n_conditions;if(nc<1||nc>4)continue;bool ok=true;for(int k=0;k<nc;++k){const auto&c=rule.conditions[k];int32_t got=legal.feature(c.feature_id,c.index,0);if(!cmp_i32(got,c.value,c.cmp)){ok=false;break;}}if(ok)return rule.route_id;}
    return 0xffffu;
}

__global__ void eval_kernel(Config base,DeviceTables tables,bonsaikg4_champion::CudaChampionPolicy champion,
                            const Action*opp_actions,const OppDesc*opp_desc,const CaseDev*cases,int ncases,
                            const ModuleGenomeBin*genomes,int ngenomes,Result*out){
    int job=blockIdx.x*blockDim.x+threadIdx.x,total=ncases*ngenomes;if(job>=total)return;int gi=job/ncases,ci=job-gi*ncases;const CaseDev c=cases[ci];Config cfg=base;cfg.seed=c.seed;Sim sim;sim.init(cfg,&tables);bonsaikg4_champion::CudaChampionState st{};ObservationView initial=make_observation(sim,c.seat);champion.reset(st,initial);const auto&g=genomes[gi];
    while(!sim.st.done){const int step=sim.st.step,opp=1-c.seat;ObservationView obs=make_observation(sim,c.seat);uint16_t mr=module_route(g,obs);Action ca{};champion.act(st,obs,mr,ca);Action oa{};const auto d=opp_desc[c.tape];if(d.count>0){int s=step;if(s<0)s=0;if(s>=d.count)s=d.count-1;oa=opp_actions[d.first+s];}oa.n_units=sim.st.farms[opp].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);}
    Result r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.controller_fault=st.base.controller.fault;r.compat_fault=st.base.compat_fault;out[job]=r;
}

static void write_results(const std::string&path,const std::vector<Result>&r,int ngen,int ncases,double ms){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot write result file");const char magic[8]={'B','4','R','S','1','\0','\0','\0'};out.write(magic,8);uint32_t v=1,g=ngen,c=ncases;out.write((char*)&v,4);out.write((char*)&g,4);out.write((char*)&c,4);out.write((char*)r.data(),r.size()*sizeof(r[0]));
    std::cout<<"genomes="<<ngen<<" cases="<<ncases<<" games="<<((uint64_t)ngen*ncases)<<" gpu_ms="<<ms<<" games_s="<<(((double)ngen*ncases)/(ms/1000.0))<<" result="<<path<<"\n";
}

} // namespace

int main(int argc,char**argv){try{
    std::string controller,routes,legacy,cases_path,genomes_path,out_path;int tpb=64;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")cases_path=need();else if(a=="--genomes")genomes_path=need();else if(a=="--out")out_path=need();else if(a=="--tpb")tpb=std::stoi(need());else throw std::runtime_error("unknown argument "+a);}
    if(controller.empty()||routes.empty()||legacy.empty()||cases_path.empty()||genomes_path.empty()||out_path.empty())throw std::runtime_error("required: --controller --routes --legacy-routes --cases --genomes --out");
    auto owned=bonsaikg::load_compiled(controller);if(bonsaikg_v6::controller_hash_hex(owned)!=bonsaikg_v6::CONTROLLER_SEMANTIC_HASH)throw std::runtime_error("frozen controller identity mismatch");auto hp=owned.view();DevicePolicyOwner d;d.nodes=copy_dev(owned.nodes);d.children=copy_dev(owned.child_refs);d.bb=copy_dev(owned.bb);d.values=copy_dev(owned.values);d.feature_args=copy_dev(owned.feature_args);d.pred=copy_dev(owned.predicates);d.lc=copy_dev(owned.lifecycles);d.effects=copy_dev(owned.effects);bonsaikg::PolicyView dp=hp;dp.nodes=d.nodes;dp.child_refs=d.children;dp.bb=d.bb;dp.values=d.values;dp.feature_args=d.feature_args;dp.predicates=d.pred;dp.lifecycles=d.lc;dp.effects=d.effects;
    auto rp=bonsaikg_v6::RoutePack::load(routes),lp=load_legacy_pack(legacy);auto*dr=copy_dev(rp.routes);auto*da=copy_dev(rp.actions);auto*dlr=copy_dev(lp.routes);auto*dla=copy_dev(lp.actions);bonsaikg_v6::CudaCompatPolicy base{dp,dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaChampionPolicy champion{base,{dlr,dla,(int)lp.routes.size()}};
    auto hc=load_cases(cases_path);std::vector<NativeTape> tapes;std::vector<Action> hoa;std::vector<OppDesc> hod;std::vector<CaseDev> hcd;for(const auto&c:hc){NativeTape t=load_native_tape_json(c.path);int off=(int)hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)tapes.size(),c.seat,c.seed,c.family,c.protected_flag,c.own_history});tapes.push_back(std::move(t));}
    auto hg=load_genomes(genomes_path);Action*doa=copy_dev(hoa);OppDesc*dod=copy_dev(hod);CaseDev*dcd=copy_dev(hcd);ModuleGenomeBin*dg=copy_dev(hg);Result*dout=nullptr;size_t jobs=(size_t)hg.size()*hc.size();KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(Result)));
    configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,champion,doa,dod,dcd,(int)hc.size(),dg,(int)hg.size(),dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<Result> hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(Result),cudaMemcpyDeviceToHost));write_results(out_path,hr,(int)hg.size(),(int)hc.size(),ms);
    cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dcd);cudaFree(dg);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
