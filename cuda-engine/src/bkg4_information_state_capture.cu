#include "../../bonsai-kg/include/bonsai_kg_v4.hpp"
#include "../../bonsai-kg/include/bonsai_kg_v4_io.hpp"
#include "../../bonsai-kg/cuda/kag_adapter_v4.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/extended_route_pack.hpp"
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
#include <unordered_map>
#include <vector>

using namespace kagcuda;

namespace {

constexpr int R0_REGISTRY_WIDTH =
    14 +                         // scalar/no-argument features
    2 * N_PRODUCTS +             // market price + inventory
    N_ITEMS +                    // own shed by item
    N_CROPS +                    // own seeds
    MAX_UNITS * N_ITEMS +        // exact own per-unit inventories
    2 * MAX_SHOP_INSTANCES +     // SHOP_AT + SHOP_SEEN
    2 * 4 +                      // own/opp quadrant flags
    6 * MAX_UNITS +              // own/opp worker x/y/tile
    30 * BOARD * BOARD;          // every own/opp tile feature over every tile
constexpr int R0_COMPAT_WIDTH = 15; // 13 legacy + 2 engine-exploit fields
constexpr int R0_META_WIDTH = 2;    // fault + ticks (controller runtime meta)
constexpr int HIDDEN_TRUTH_WIDTH = 17; // offline only: opponent total holdings 12 + seeds 5

#pragma pack(push,1)
struct CaptureHeader {
    char magic[8];
    uint32_t version;
    uint32_t cases;
    uint32_t probes;
    uint32_t width;
    uint32_t registry_width;
    uint32_t bb_width;
    uint32_t compat_width;
    uint32_t meta_width;
};
#pragma pack(pop)
static_assert(sizeof(CaptureHeader) == 40, "capture header ABI");

struct CaseHost {
    std::string path;
    int seat = 0;
    uint64_t seed = 0;
    int family = 0;
    int protected_flag = 0;
    int own_history = 0;
    int role = 0;
};
struct CaseDev { int tape=0,seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0,role=0; };
struct OppDesc { int first=0,count=0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4* n=nullptr; uint16_t* c=nullptr; bonsaikg4::BbDecl4* bb=nullptr;
    bonsaikg4::ValueRef4* v=nullptr; bonsaikg4::FeatureArgRef4* a=nullptr;
    bonsaikg4::Predicate4* q=nullptr; bonsaikg4::Lifecycle4* l=nullptr; bonsaikg4::Effect4* e=nullptr;
    ~DevicePolicyOwner4(){if(n)cudaFree(n);if(c)cudaFree(c);if(bb)cudaFree(bb);if(v)cudaFree(v);if(a)cudaFree(a);if(q)cudaFree(q);if(l)cudaFree(l);if(e)cudaFree(e);}
};
template<class T> static T* cp(const std::vector<T>& v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}

static std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost> load_cases(const std::string&path){std::ifstream in(path);if(!in)throw std::runtime_error("cannot open cases: "+path);std::vector<CaseHost>o;std::string line;while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;auto z=split_tab(line);if(z.size()!=7)throw std::runtime_error("case needs seven tab fields");CaseHost c;c.path=z[0];c.seat=std::stoi(z[1]);c.seed=std::stoull(z[2]);c.family=std::stoi(z[3]);c.protected_flag=std::stoi(z[4]);c.own_history=std::stoi(z[5]);c.role=std::stoi(z[6]);if(c.seat<0||c.seat>1)throw std::runtime_error("seat");o.push_back(c);}if(o.empty())throw std::runtime_error("empty cases");return o;}
static std::vector<int> load_probes(const std::string&path){std::ifstream in(path);if(!in)throw std::runtime_error("cannot open probes: "+path);std::vector<int>o;int x=0;while(in>>x){if(x<0||x>=720)throw std::runtime_error("probe");o.push_back(x);}if(o.empty())throw std::runtime_error("empty probes");for(size_t i=1;i<o.size();++i)if(o[i]<=o[i-1])throw std::runtime_error("probes must be strictly increasing");return o;}

static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("legacy pack");std::vector<uint8_t>raw((std::istreambuf_iterator<char>(in)),{});auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy hash");bonsaikg_v6::PackHeader h{};if(raw.size()<sizeof(h))throw std::runtime_error("short legacy");std::memcpy(&h,raw.data(),sizeof(h));size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc),expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);if(raw.size()!=expected)throw std::runtime_error("legacy size");bonsaikg_v6::RoutePack o;o.routes.resize(h.route_count);o.actions.resize(h.action_count);std::memcpy(o.routes.data(),raw.data()+sizeof(h),db);std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0]));return o;}

struct DeviceBasePack {
    const bonsaikg_v6::RouteDesc* routes=nullptr; const bonsaikg_v6::PackedAction* actions=nullptr; int n=0;
    __device__ __forceinline__ const bonsaikg_v6::PackedAction* at(uint16_t route,int step)const{for(int i=0;i<n;++i)if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;}return nullptr;}
};

__device__ __forceinline__ void reconcile_feed(const int32_t*bb,int first_slot,int count,Action&a){for(int u=0;u<count&&u<a.n_units;++u)if(bb[first_slot+u]&&a.units[u].op==OP_FEED)a.units[u]=UnitAction{};}
__device__ __forceinline__ bool build_final_action(const ObservationView&raw,const DeviceBasePack&base_pack,const bonsaikg4_champion::CudaLegacyPack&legacy_pack,bonsaikg4_champion::CudaLegacyState&legacy,bonsaikg_v6::CudaEngineExploitState&exploit,const int32_t*bb,int feed_slot,int feed_count,int strategic_route_slot,Action&out){int route=bb[1];if(strategic_route_slot>=0&&bb[strategic_route_slot])route=bb[strategic_route_slot];const auto*p=base_pack.at((uint16_t)route,raw.step());if(!p)return false;bonsaikg_v6::materialize_cuda(*p,raw.farm(raw.player()).n_units(),out);bonsaikg4_champion::apply_legacy(raw,legacy_pack,legacy,out);bonsaikg_v6::apply_cuda_engine_exploits(raw,out,exploit,legacy.external_ok!=0,100);reconcile_feed(bb,feed_slot,feed_count,out);return true;}

__device__ __forceinline__ void put(int32_t*out,int&k,int32_t v){out[k++]=v;}
__device__ __forceinline__ int capture_registry(const bonsaikg4::CudaObservation4&o,int32_t*out){
    int k=0;
    const uint16_t scalars[]={
        bonsaikg4::F_ALWAYS,bonsaikg4::F_STEP,bonsaikg4::F_DAY,bonsaikg4::F_HOUR,
        bonsaikg4::F_OWN_MONEY,bonsaikg4::F_OPP_MONEY,bonsaikg4::F_OWN_UNITS,bonsaikg4::F_OPP_UNITS,
        bonsaikg4::F_OWN_LAND,bonsaikg4::F_OPP_LAND,bonsaikg4::F_OWN_HIRES_TODAY,
        bonsaikg4::F_OWN_SHED_TOTAL,bonsaikg4::F_SHOP_COUNT,bonsaikg4::F_OPP_HIRES_TODAY};
    for(int i=0;i<14;++i)put(out,k,o.feature(scalars[i]));
    for(int item=0;item<N_PRODUCTS;++item)put(out,k,o.feature(bonsaikg4::F_MARKET_PRICE,item));
    for(int item=0;item<N_PRODUCTS;++item)put(out,k,o.feature(bonsaikg4::F_MARKET_INVENTORY,item));
    for(int item=0;item<N_ITEMS;++item)put(out,k,o.feature(bonsaikg4::F_OWN_SHED,item));
    for(int item=0;item<N_CROPS;++item)put(out,k,o.feature(bonsaikg4::F_OWN_SEED,item));
    for(int u=0;u<MAX_UNITS;++u)for(int item=0;item<N_ITEMS;++item)put(out,k,o.feature(bonsaikg4::F_OWN_INVENTORY,u,item));
    for(int i=0;i<MAX_SHOP_INSTANCES;++i)put(out,k,o.feature(bonsaikg4::F_SHOP_AT,i));
    for(int i=0;i<MAX_SHOP_INSTANCES;++i)put(out,k,o.feature(bonsaikg4::F_SHOP_SEEN,i));
    for(int q=0;q<4;++q)put(out,k,o.feature(bonsaikg4::F_OWN_QUADRANT_UNLOCKED,q));
    for(int q=0;q<4;++q)put(out,k,o.feature(bonsaikg4::F_OPP_QUADRANT_UNLOCKED,q));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OWN_WORKER_X,u));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OWN_WORKER_Y,u));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OPP_WORKER_X,u));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OPP_WORKER_Y,u));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OWN_WORKER_TILE,u));
    for(int u=0;u<MAX_UNITS;++u)put(out,k,o.feature(bonsaikg4::F_OPP_WORKER_TILE,u));
    for(uint16_t fid=bonsaikg4::F_OWN_TILE_KIND;fid<=bonsaikg4::F_OPP_TILE_FERT_AVAILABLE;++fid)
        for(int tile=0;tile<BOARD*BOARD;++tile)put(out,k,o.feature(fid,tile));
    for(uint16_t fid=bonsaikg4::F_OWN_TILE_PLANTED_DAY;fid<=bonsaikg4::F_OPP_TILE_HAS_ANIMAL;++fid)
        for(int tile=0;tile<BOARD*BOARD;++tile)put(out,k,o.feature(fid,tile));
    return k;
}

__device__ __forceinline__ int capture_compat(const bonsaikg4_champion::CudaLegacyState&s,const bonsaikg_v6::CudaEngineExploitState&e,int32_t*out){int k=0;put(out,k,s.trigger_decided);put(out,k,s.h4_trigger);put(out,k,s.first_shop);put(out,k,s.r86);put(out,k,s.ownweed168);put(out,k,s.r253);put(out,k,s.fy144);put(out,k,s.ice86);put(out,k,s.ice_v32);put(out,k,s.ff168);put(out,k,s.ff_low);put(out,k,s.external_ok);put(out,k,s.external_route);put(out,k,e.high_labor);put(out,k,e.carrot_latched);return k;}

__device__ __forceinline__ void capture_hidden_truth(const Sim&sim,int opp,int32_t*out){
    const Farm& f=sim.st.farms[opp];int k=0;
    for(int item=0;item<N_ITEMS;++item){int total=f.shed[item];for(int u=0;u<f.n_units;++u)total+=f.inv[u][item];out[k++]=total;}
    for(int crop=0;crop<N_CROPS;++crop)out[k++]=f.seeds[crop];
}

__global__ void capture_kernel(Config cfg0,DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack base_pack,bonsaikg4_champion::CudaLegacyPack legacy_pack,const Action*opp,const OppDesc*od,const CaseDev*cases,int ncases,const int*probes,int nprobes,int feed_slot,int feed_count,int strategic_route_slot,int32_t*bb,bonsaikg4::NodeState4*ns,bonsaikg4::RuntimeMeta4*meta,Sim*sims,int32_t*frames,int32_t*truth,uint16_t*flags){
    const int ci=blockIdx.x*blockDim.x+threadIdx.x;if(ci>=ncases)return;const CaseDev c=cases[ci];Config cfg=cfg0;cfg.seed=c.seed;Sim&sim=sims[ci];sim.init(cfg,&tables);bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)ci*policy.n_bb:nullptr,policy.n_state?ns+(size_t)ci*policy.n_state:nullptr,meta+ci};bonsaikg4::reset_runtime(policy,st);bonsaikg4_champion::CudaLegacyState legacy{};bonsaikg_v6::CudaEngineExploitState exploit{};bonsaikg4_champion::reset_legacy(legacy);bonsaikg_v6::reset_cuda_engine_exploits(exploit);int pi=0;uint16_t fl=0;
    while(!sim.st.done){const int step=sim.st.step,os=1-c.seat;ObservationView raw=make_observation(sim,c.seat);bonsaikg4::CudaObservation4 obs4(raw);bonsaikg4::ActionResult4 ignored{};auto status=bonsaikg4::tick4(policy,st,obs4,0,ignored,nullptr);if(meta[ci].fault||status!=bonsaikg4::SUCCESS){fl|=1;break;}Action ca{};if(!build_final_action(raw,base_pack,legacy_pack,legacy,exploit,st.blackboard,feed_slot,feed_count,strategic_route_slot,ca)){fl|=2;break;}
        if(pi<nprobes&&step==probes[pi]){const int width=R0_REGISTRY_WIDTH+policy.n_bb+R0_COMPAT_WIDTH+R0_META_WIDTH;int32_t*out=frames+((size_t)ci*nprobes+pi)*width;int k=capture_registry(obs4,out);if(k!=R0_REGISTRY_WIDTH)fl|=4;for(int i=0;i<policy.n_bb;++i)put(out,k,st.blackboard[i]);k+=capture_compat(legacy,exploit,out+k);put(out,k,meta[ci].fault);put(out,k,(int32_t)meta[ci].ticks);if(k!=width)fl|=8;capture_hidden_truth(sim,1-c.seat,truth+((size_t)ci*nprobes+pi)*HIDDEN_TRUTH_WIDTH);++pi;}
        Action oa{};OppDesc d=od[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];}oa.n_units=sim.st.farms[os].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
    }
    if(pi!=nprobes)fl|=16;flags[ci]=fl;
}

static void write_capture(const std::string&path,const std::vector<int>&probes,const std::vector<int32_t>&frames,int cases,int width,int bb_width){std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("cannot write capture");CaptureHeader h{{'B','4','I','S','R','0','1','\0'},1u,(uint32_t)cases,(uint32_t)probes.size(),(uint32_t)width,(uint32_t)R0_REGISTRY_WIDTH,(uint32_t)bb_width,(uint32_t)R0_COMPAT_WIDTH,(uint32_t)R0_META_WIDTH};o.write((char*)&h,sizeof(h));for(int x:probes){int32_t y=x;o.write((char*)&y,sizeof(y));}o.write((char*)frames.data(),frames.size()*sizeof(frames[0]));}
static void write_truth(const std::string&path,const std::vector<int>&probes,const std::vector<int32_t>&truth,int cases){std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("cannot write truth");CaptureHeader h{{'B','4','I','S','T','1','1','\0'},1u,(uint32_t)cases,(uint32_t)probes.size(),(uint32_t)HIDDEN_TRUTH_WIDTH,(uint32_t)HIDDEN_TRUTH_WIDTH,0u,0u,0u};o.write((char*)&h,sizeof(h));for(int x:probes){int32_t y=x;o.write((char*)&y,sizeof(y));}o.write((char*)truth.data(),truth.size()*sizeof(truth[0]));}

} // namespace

int main(int argc,char**argv){try{
    std::string controller,routes,legacy,casesp,probesp,outp,truthp;int feed_slot=5,feed_count=13,strategic_route_slot=-1,tpb=64,device=-1;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--probes")probesp=need();else if(a=="--out")outp=need();else if(a=="--truth-out")truthp=need();else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--strategic-route-slot")strategic_route_slot=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--device")device=std::stoi(need());else throw std::runtime_error("unknown "+a);}
    if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||probesp.empty()||outp.empty()||truthp.empty())throw std::runtime_error("required args");
    if(device < -1)throw std::runtime_error("device");if(device>=0)KAGCUDA_CHECK(cudaSetDevice(device));
    auto owned=bonsaikg4::load_compiled4(controller);auto hp=owned.view();if(hp.n_bb<(uint16_t)(feed_slot+feed_count))throw std::runtime_error("feed slots");if(hp.n_state!=0)std::cerr<<"NOTE: capture supports state slots, current champion has "<<hp.n_state<<"\n";DevicePolicyOwner4 po;po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;
    auto rp=bonsaikg4_champion::load_extended_route_pack(routes),lp=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lp.routes);auto*dla=cp(lp.actions);DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack ldp{dlr,dla,(int)lp.routes.size()};
    auto hc=load_cases(casesp);auto probes=load_probes(probesp);std::vector<Action>hoa;std::vector<OppDesc>hod;std::vector<CaseDev>hcd;std::unordered_map<std::string,int>tape_cache;for(const auto&c:hc){int ti=-1;auto it=tape_cache.find(c.path);if(it!=tape_cache.end())ti=it->second;else{auto t=load_native_tape(c.path);int off=(int)hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());ti=(int)hod.size();hod.push_back({off,(int)t.actions.size()});tape_cache.emplace(c.path,ti);}hcd.push_back({ti,c.seat,c.seed,c.family,c.protected_flag,c.own_history,c.role});}
    auto*doa=cp(hoa);auto*dod=cp(hod);auto*dc=cp(hcd);auto*dprobes=cp(probes);const int ncases=(int)hc.size(),nprobes=(int)probes.size(),width=R0_REGISTRY_WIDTH+hp.n_bb+R0_COMPAT_WIDTH+R0_META_WIDTH;int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;Sim*ds=nullptr;int32_t*df=nullptr;int32_t*dt=nullptr;uint16_t*dflags=nullptr;if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,(size_t)ncases*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,(size_t)ncases*hp.n_state*sizeof(bonsaikg4::NodeState4)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,(size_t)ncases*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)ncases*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&df,(size_t)ncases*nprobes*width*sizeof(int32_t)));KAGCUDA_CHECK(cudaMalloc((void**)&dt,(size_t)ncases*nprobes*HIDDEN_TRUTH_WIDTH*sizeof(int32_t)));KAGCUDA_CHECK(cudaMalloc((void**)&dflags,(size_t)ncases*sizeof(uint16_t)));KAGCUDA_CHECK(cudaMemset(df,0,(size_t)ncases*nprobes*width*sizeof(int32_t)));
    configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));capture_kernel<<<(ncases+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,ldp,doa,dod,dc,ncases,dprobes,nprobes,feed_slot,feed_count,strategic_route_slot,dbb,dns,dm,ds,df,dt,dflags);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<int32_t>hf((size_t)ncases*nprobes*width);std::vector<int32_t>ht((size_t)ncases*nprobes*HIDDEN_TRUTH_WIDTH);std::vector<uint16_t>flags(ncases);KAGCUDA_CHECK(cudaMemcpy(hf.data(),df,hf.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(ht.data(),dt,ht.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(flags.data(),dflags,flags.size()*sizeof(uint16_t),cudaMemcpyDeviceToHost));int bad=0;for(auto x:flags)bad+=x!=0;write_capture(outp,probes,hf,ncases,width,hp.n_bb);write_truth(truthp,probes,ht,ncases);std::cout<<"r0 cases="<<ncases<<" probes="<<nprobes<<" width="<<width<<" registry="<<R0_REGISTRY_WIDTH<<" bb="<<hp.n_bb<<" compat="<<R0_COMPAT_WIDTH<<" meta="<<R0_META_WIDTH<<" gpu_ms="<<ms<<" bad="<<bad<<" output_MiB="<<(hf.size()*sizeof(int32_t)/(1024.0*1024.0))<<"\n";
    cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dc);cudaFree(dprobes);if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(df);cudaFree(dt);cudaFree(dflags);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
