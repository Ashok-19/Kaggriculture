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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;
namespace {

#pragma pack(push,1)
struct ResultHeader { char magic[8]; uint32_t version,candidates,cases,start_index,total_candidates; };
struct ResultBin { int32_t own_money,opp_money; uint16_t changes,flags; };
#pragma pack(pop)
static_assert(sizeof(ResultBin)==12,"result ABI");

struct Opening {
    uint8_t build=0,hires=0,cows=0,sheep=0,wseed=0,mseed=0,wbuy=0,order_template=0;
};
struct CaseHost { std::string path; int seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0,role=0; };
struct CaseDev { int tape=0,seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0,role=0; };
struct OppDesc { int first=0,count=0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4*n=nullptr;uint16_t*c=nullptr;bonsaikg4::BbDecl4*bb=nullptr;bonsaikg4::ValueRef4*v=nullptr;
    bonsaikg4::FeatureArgRef4*a=nullptr;bonsaikg4::Predicate4*q=nullptr;bonsaikg4::Lifecycle4*l=nullptr;bonsaikg4::Effect4*e=nullptr;
    ~DevicePolicyOwner4(){if(n)cudaFree(n);if(c)cudaFree(c);if(bb)cudaFree(bb);if(v)cudaFree(v);if(a)cudaFree(a);if(q)cudaFree(q);if(l)cudaFree(l);if(e)cudaFree(e);}
};
template<class T> static T* cp(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
static std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost> load_cases(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open cases");std::vector<CaseHost>o;std::string l;while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;auto z=split_tab(l);if(z.size()!=6&&z.size()!=7)throw std::runtime_error("case line needs 6 or 7 fields");CaseHost c;c.path=z[0];c.seat=std::stoi(z[1]);c.seed=std::stoull(z[2]);c.family=std::stoi(z[3]);c.protected_flag=std::stoi(z[4]);c.own_history=std::stoi(z[5]);if(z.size()==7)c.role=std::stoi(z[6]);o.push_back(c);}if(o.empty())throw std::runtime_error("empty cases");return o;}
static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("legacy pack");std::vector<uint8_t>raw((std::istreambuf_iterator<char>(in)),{});auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy hash");bonsaikg_v6::PackHeader h{};if(raw.size()<sizeof(h))throw std::runtime_error("short legacy");std::memcpy(&h,raw.data(),sizeof(h));size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc),expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);if(raw.size()!=expected)throw std::runtime_error("legacy size");bonsaikg_v6::RoutePack o;o.routes.resize(h.route_count);o.actions.resize(h.action_count);std::memcpy(o.routes.data(),raw.data()+sizeof(h),db);std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0]));return o;}
struct DeviceBasePack{const bonsaikg_v6::RouteDesc*routes=nullptr;const bonsaikg_v6::PackedAction*actions=nullptr;int n=0;__device__ __forceinline__ const bonsaikg_v6::PackedAction*at(uint16_t route,int step)const{for(int i=0;i<n;++i)if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;}return nullptr;}};

__device__ __forceinline__ uint16_t reconcile_feed(const int32_t*bb,int first,int count,Action&a){uint16_t n=0;for(int u=0;u<count&&u<a.n_units;++u)if(bb[first+u]&&a.units[u].op==OP_FEED){a.units[u]=UnitAction{};++n;}return n;}
__device__ __forceinline__ bool build_final_action(const ObservationView&raw,const DeviceBasePack&bp,const bonsaikg4_champion::CudaLegacyPack&lp,bonsaikg4_champion::CudaLegacyState&legacy,bonsaikg_v6::CudaEngineExploitState&exploit,const int32_t*bb,int feed_slot,int feed_count,int strategic_route_slot,uint16_t strategic_route_id,Action&out){int route=bb[1];if(strategic_route_slot>=0&&bb[strategic_route_slot])route=strategic_route_id==0xffffu?(uint16_t)bb[strategic_route_slot]:strategic_route_id;const auto*p=bp.at((uint16_t)route,raw.step());if(!p)return false;bonsaikg_v6::materialize_cuda(*p,raw.farm(raw.player()).n_units(),out);bonsaikg4_champion::apply_legacy(raw,lp,legacy,out);bonsaikg_v6::apply_cuda_engine_exploits(raw,out,exploit,legacy.external_ok!=0,100);reconcile_feed(bb,feed_slot,feed_count,out);return true;}
__device__ __forceinline__ void push(Action&a,uint8_t op,uint8_t item,int n){if(n<=0||a.n_orders>=MAX_ORDERS)return;a.orders[a.n_orders++]={op,item,n};}
__device__ __forceinline__ void hires(Action&a,int n){for(int i=0;i<n;++i)push(a,M_HIRE,0,1);}
__device__ __forceinline__ void animals(Action&a,const Opening&o,bool sheep_first){if(sheep_first){push(a,M_BUY_ANIMAL,SHEEP,o.sheep);push(a,M_BUY_ANIMAL,COW,o.cows);}else{push(a,M_BUY_ANIMAL,COW,o.cows);push(a,M_BUY_ANIMAL,SHEEP,o.sheep);}}
__device__ __forceinline__ void seeds(Action&a,const Opening&o){push(a,M_BUY_SEED,WHEAT,o.wseed);push(a,M_BUY_SEED,MELON,o.mseed);}
__device__ __forceinline__ void apply_opening(const Opening&o,int n_units,Action&a){a=Action{};a.n_units=n_units;a.units[0].op=o.build?OP_BUILD_PASTURE:OP_PASS;
    switch(o.order_template){
      case 0: hires(a,o.hires);animals(a,o,false);seeds(a,o);push(a,M_BUY_PRODUCT,WHEAT,o.wbuy);break; // incumbent
      case 1: hires(a,o.hires);animals(a,o,true);seeds(a,o);push(a,M_BUY_PRODUCT,WHEAT,o.wbuy);break;  // Ryo-like
      case 2: push(a,M_BUY_PRODUCT,WHEAT,o.wbuy);animals(a,o,false);seeds(a,o);hires(a,o.hires);break;  // tetsuya-like
      case 3: push(a,M_BUY_PRODUCT,WHEAT,o.wbuy);hires(a,o.hires);seeds(a,o);push(a,M_BUY_ANIMAL,SHEEP,o.sheep);push(a,M_BUY_ANIMAL,COW,o.cows);break; // Arman-like
      default: seeds(a,o);animals(a,o,false);hires(a,o.hires);push(a,M_BUY_PRODUCT,WHEAT,o.wbuy);break;
    }
}
__device__ __forceinline__ bool same_unit(const UnitAction&a,const UnitAction&b){return a.op==b.op&&a.arg==b.arg&&a.n==b.n;}
__device__ __forceinline__ bool same_order(const Order&a,const Order&b){return a.op==b.op&&a.item==b.item&&a.n==b.n;}
__device__ __forceinline__ bool same_action(const Action&a,const Action&b){if(a.n_units!=b.n_units||a.n_orders!=b.n_orders)return false;for(int i=0;i<a.n_units;++i)if(!same_unit(a.units[i],b.units[i]))return false;for(int i=0;i<a.n_orders;++i)if(!same_order(a.orders[i],b.orders[i]))return false;return true;}

static std::vector<Opening> build_grid(){
    std::vector<Opening> out;out.push_back(Opening{}); // candidate zero is exact frozen behavior, descriptor ignored.
    const int builds[]={0,1}, hs[]={2,3,4,5,6}, cs[]={0,1,2}, ss[]={2,3,4}, ws[]={5,6,7,8,10}, ms[]={0,7,11,12}, wb[]={4,5,6};
    for(int b:builds)for(int h:hs)for(int c:cs)for(int s:ss)for(int w:ws)for(int m:ms)for(int x:wb)for(int ot=0;ot<5;++ot){Opening o{};o.build=b;o.hires=h;o.cows=c;o.sheep=s;o.wseed=w;o.mseed=m;o.wbuy=x;o.order_template=ot;out.push_back(o);}
    return out;
}

__global__ void kernel(Config cfg0,DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack bp,bonsaikg4_champion::CudaLegacyPack lp,const Action*opp,const OppDesc*od,const CaseDev*cases,int nc,const Opening*all,int start,int count,int feed_slot,int feed_count,int strategic_route_slot,uint16_t strategic_route_id,int32_t*bb,bonsaikg4::NodeState4*ns,bonsaikg4::RuntimeMeta4*meta,Sim*sims,ResultBin*out){
    int job=blockIdx.x*blockDim.x+threadIdx.x,total=nc*count;if(job>=total)return;int li=job/nc,ci=job-li*nc,gi=start+li;const CaseDev c=cases[ci];Config cfg=cfg0;cfg.seed=c.seed;Sim&sim=sims[job];sim.init(cfg,&tables);bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)job*policy.n_bb:nullptr,policy.n_state?ns+(size_t)job*policy.n_state:nullptr,meta+job};bonsaikg4::reset_runtime(policy,st);bonsaikg4_champion::CudaLegacyState legacy{};bonsaikg_v6::CudaEngineExploitState exploit{};bonsaikg4_champion::reset_legacy(legacy);bonsaikg_v6::reset_cuda_engine_exploits(exploit);uint16_t changes=0,flags=0;
    while(!sim.st.done){int step=sim.st.step,os=1-c.seat;ObservationView raw=make_observation(sim,c.seat);bonsaikg4::ActionResult4 ignored{};auto status=bonsaikg4::tick4(policy,st,bonsaikg4::CudaObservation4(raw),0,ignored,nullptr);if(meta[job].fault||status!=bonsaikg4::SUCCESS){flags|=1;break;}Action ca{};if(!build_final_action(raw,bp,lp,legacy,exploit,st.blackboard,feed_slot,feed_count,strategic_route_slot,strategic_route_id,ca)){flags|=2;break;}if(gi>0&&step==0){Action base=ca;apply_opening(all[gi],raw.farm(raw.player()).n_units(),ca);if(!same_action(base,ca))++changes;}Action oa{};OppDesc d=od[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];}oa.n_units=sim.st.farms[os].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);}
    ResultBin r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.changes=changes;r.flags=flags;out[job]=r;
}
static void write_results(const std::string&p,const std::vector<ResultBin>&r,int start,int count,int nc,int total){std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot write result");ResultHeader h{{'B','4','O','P','1','\0','\0','\0'},1u,(uint32_t)count,(uint32_t)nc,(uint32_t)start,(uint32_t)total};o.write((char*)&h,sizeof(h));o.write((char*)r.data(),r.size()*sizeof(r[0]));}
static void dump_grid(const std::string&p,const std::vector<Opening>&g){std::ofstream o(p);o<<"id,build,hires,cows,sheep,wseed,mseed,wbuy,order_template\n";for(size_t i=0;i<g.size();++i){auto x=g[i];o<<i<<','<<int(x.build)<<','<<int(x.hires)<<','<<int(x.cows)<<','<<int(x.sheep)<<','<<int(x.wseed)<<','<<int(x.mseed)<<','<<int(x.wbuy)<<','<<int(x.order_template)<<'\n';}}
}

int main(int argc,char**argv){try{
    std::string controller,routes,legacy,casesp,outp,gridp;int start=0,count=-1,feed_slot=5,feed_count=13,strategic_route_slot=-1,strategic_route_id=0xffff,tpb=64;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--out")outp=need();else if(a=="--grid-out")gridp=need();else if(a=="--start")start=std::stoi(need());else if(a=="--count")count=std::stoi(need());else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--strategic-route-slot")strategic_route_slot=std::stoi(need());else if(a=="--strategic-route-id")strategic_route_id=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else throw std::runtime_error("unknown "+a);}if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||outp.empty())throw std::runtime_error("required args");auto grid=build_grid();if(!gridp.empty())dump_grid(gridp,grid);if(start<0||start>=(int)grid.size())throw std::runtime_error("bad start");if(count<0||start+count>(int)grid.size())count=(int)grid.size()-start;if(count<=0)throw std::runtime_error("bad count");
    auto owned=bonsaikg4::load_compiled4(controller);auto hp=owned.view();DevicePolicyOwner4 po;po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;auto rp=bonsaikg4_champion::load_extended_route_pack(routes),lg=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lg.routes);auto*dla=cp(lg.actions);DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack lp{dlr,dla,(int)lg.routes.size()};auto hc=load_cases(casesp);std::vector<Action>hoa;std::vector<OppDesc>hod;std::vector<CaseDev>hcd;for(const auto&c:hc){auto t=load_native_tape_json(c.path);int off=(int)hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)hod.size()-1,c.seat,c.seed,c.family,c.protected_flag,c.own_history,c.role});}auto*doa=cp(hoa);auto*dod=cp(hod);auto*dc=cp(hcd);auto*dg=cp(grid);size_t jobs=(size_t)count*hc.size();int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;Sim*ds=nullptr;ResultBin*dout=nullptr;if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,jobs*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,jobs*hp.n_state*sizeof(*dns)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,jobs*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(*dout)));configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,lp,doa,dod,dc,(int)hc.size(),dg,start,count,feed_slot,feed_count,strategic_route_slot,(uint16_t)strategic_route_id,dbb,dns,dm,ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<ResultBin>hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(*dout),cudaMemcpyDeviceToHost));int bad=0;for(auto&r:hr)bad+=r.flags!=0;write_results(outp,hr,start,count,(int)hc.size(),(int)grid.size());std::cout<<"candidates="<<count<<" start="<<start<<" total="<<grid.size()<<" cases="<<hc.size()<<" games="<<jobs<<" gpu_ms="<<ms<<" games_s="<<(jobs/(ms/1000.0))<<" bad="<<bad<<"\n";cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dc);cudaFree(dg);if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
