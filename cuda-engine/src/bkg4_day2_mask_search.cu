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

constexpr int N_SPECIAL = 16;
__constant__ int D_SPECIAL[N_SPECIAL];

#pragma pack(push,1)
struct ResultHeader {
    char magic[8];
    uint32_t version, start_mask, masks, cases, total_masks;
};
struct ResultBin {
    int32_t own_money, opp_money;
    uint16_t rewrites, flags;
};
#pragma pack(pop)
static_assert(sizeof(ResultBin) == 12, "day2 result ABI");

struct CaseHost {
    std::string path;
    int seat = 0;
    uint64_t seed = 0;
    int family = 0, protected_flag = 0, own_history = 0, role = 0;
};
struct CaseDev {
    int tape = 0, seat = 0;
    uint64_t seed = 0;
    int family = 0, protected_flag = 0, own_history = 0, role = 0;
};
struct TapeDesc { int first = 0, count = 0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4* n=nullptr; uint16_t* c=nullptr; bonsaikg4::BbDecl4* bb=nullptr;
    bonsaikg4::ValueRef4* v=nullptr; bonsaikg4::FeatureArgRef4* a=nullptr;
    bonsaikg4::Predicate4* q=nullptr; bonsaikg4::Lifecycle4* l=nullptr; bonsaikg4::Effect4* e=nullptr;
    ~DevicePolicyOwner4(){if(n)cudaFree(n);if(c)cudaFree(c);if(bb)cudaFree(bb);if(v)cudaFree(v);if(a)cudaFree(a);if(q)cudaFree(q);if(l)cudaFree(l);if(e)cudaFree(e);}
};
template<class T> static T* cp(const std::vector<T>& v) {
    if(v.empty()) return nullptr;
    T* d=nullptr; KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));
    KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return d;
}
static std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> o; size_t p=0;
    for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;} return o;
}
static std::vector<CaseHost> load_cases(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open cases");
    std::vector<CaseHost> o; std::string line;
    while(std::getline(in,line)) {
        if(line.empty()||line[0]=='#') continue;
        auto z=split_tab(line); if(z.size()!=6&&z.size()!=7) throw std::runtime_error("case line needs 6 or 7 fields");
        CaseHost c; c.path=z[0]; c.seat=std::stoi(z[1]); c.seed=std::stoull(z[2]); c.family=std::stoi(z[3]);
        c.protected_flag=std::stoi(z[4]); c.own_history=std::stoi(z[5]); if(z.size()==7)c.role=std::stoi(z[6]);
        if(c.seat<0||c.seat>1) throw std::runtime_error("bad seat"); o.push_back(c);
    }
    if(o.empty()) throw std::runtime_error("empty cases"); return o;
}
static bonsaikg_v6::RoutePack load_legacy_pack(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("legacy pack");
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),{});
    auto full=bonsaikg::sha256_bytes(raw);
    if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256) throw std::runtime_error("legacy hash");
    bonsaikg_v6::PackHeader h{}; if(raw.size()<sizeof(h)) throw std::runtime_error("short legacy"); std::memcpy(&h,raw.data(),sizeof(h));
    size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc), expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);
    if(raw.size()!=expected) throw std::runtime_error("legacy size");
    bonsaikg_v6::RoutePack o; o.routes.resize(h.route_count); o.actions.resize(h.action_count);
    std::memcpy(o.routes.data(),raw.data()+sizeof(h),db); std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0])); return o;
}
struct DeviceBasePack {
    const bonsaikg_v6::RouteDesc* routes=nullptr; const bonsaikg_v6::PackedAction* actions=nullptr; int n=0;
    __device__ __forceinline__ const bonsaikg_v6::PackedAction* at(uint16_t route,int step) const {
        for(int i=0;i<n;++i) if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;} return nullptr;
    }
};
__device__ __forceinline__ uint16_t reconcile_feed(const int32_t* bb,int first,int count,Action& a) {
    uint16_t changed=0;
    for(int u=0;u<count&&u<a.n_units;++u) if(bb[first+u]&&a.units[u].op==OP_FEED){a.units[u]=UnitAction{};if(changed<65535)++changed;}
    return changed;
}
__device__ __forceinline__ bool canonical_action(const ObservationView& raw,const DeviceBasePack& bp,const bonsaikg4_champion::CudaLegacyPack& lp,
                                                  bonsaikg4_champion::CudaLegacyState& legacy,bonsaikg_v6::CudaEngineExploitState& exploit,
                                                  const int32_t* bb,int feed_slot,int feed_count,Action& out) {
    const auto* p=bp.at((uint16_t)bb[1],raw.step()); if(!p) return false;
    bonsaikg_v6::materialize_cuda(*p,raw.farm(raw.player()).n_units(),out);
    bonsaikg4_champion::apply_legacy(raw,lp,legacy,out);
    bonsaikg_v6::apply_cuda_engine_exploits(raw,out,exploit,legacy.external_ok!=0,100);
    reconcile_feed(bb,feed_slot,feed_count,out); return true;
}
__device__ __forceinline__ int special_bit(int step) {
    #pragma unroll
    for(int i=0;i<N_SPECIAL;++i) if(D_SPECIAL[i]==step) return i;
    return -1;
}
__device__ __forceinline__ bool same_unit(const UnitAction& a,const UnitAction& b){return a.op==b.op&&a.arg==b.arg&&a.n==b.n;}
__device__ __forceinline__ bool same_order(const Order& a,const Order& b){return a.op==b.op&&a.item==b.item&&a.n==b.n;}
__device__ __forceinline__ bool same_action(const Action& a,const Action& b){
    if(a.n_units!=b.n_units||a.n_orders!=b.n_orders)return false;
    for(int i=0;i<a.n_units;++i)if(!same_unit(a.units[i],b.units[i]))return false;
    for(int i=0;i<a.n_orders;++i)if(!same_order(a.orders[i],b.orders[i]))return false;
    return true;
}

__global__ void kernel(Config cfg0,DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack bp,bonsaikg4_champion::CudaLegacyPack lp,
                       const Action* opp,const TapeDesc* od,const Action* donor,TapeDesc dd,const CaseDev* cases,int ncases,
                       uint32_t start_mask,int nmasks,int feed_slot,int feed_count,int32_t* bb,bonsaikg4::NodeState4* ns,
                       bonsaikg4::RuntimeMeta4* meta,Sim* sims,ResultBin* out) {
    int job=blockIdx.x*blockDim.x+threadIdx.x,total=ncases*nmasks; if(job>=total)return;
    int li=job/ncases,ci=job-li*ncases; uint32_t mask=start_mask+(uint32_t)li; const CaseDev c=cases[ci];
    Config cfg=cfg0; cfg.seed=c.seed; Sim& sim=sims[job]; sim.init(cfg,&tables);
    bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)job*policy.n_bb:nullptr,policy.n_state?ns+(size_t)job*policy.n_state:nullptr,meta+job};
    bonsaikg4::reset_runtime(policy,st); bonsaikg4_champion::CudaLegacyState legacy{}; bonsaikg_v6::CudaEngineExploitState exploit{};
    bonsaikg4_champion::reset_legacy(legacy); bonsaikg_v6::reset_cuda_engine_exploits(exploit);
    uint16_t rewrites=0,flags=0;
    while(!sim.st.done) {
        int step=sim.st.step,os=1-c.seat; ObservationView ob=make_observation(sim,c.seat); bonsaikg4::ActionResult4 ignored{};
        auto status=bonsaikg4::tick4(policy,st,bonsaikg4::CudaObservation4(ob),0,ignored,nullptr);
        if(meta[job].fault||status!=bonsaikg4::SUCCESS){flags|=1;break;}
        int bit=special_bit(step); bool use_donor=bit>=0 && ((mask>>bit)&1u);
        Action ca{};
        if(use_donor) {
            // Comparison baseline from copied overlay states only; actual overlay/exploit
            // state advances exactly once on the donor action chosen by this mask.
            auto lc=legacy; auto ec=exploit; Action base{};
            if(!canonical_action(ob,bp,lp,lc,ec,st.blackboard,feed_slot,feed_count,base)){flags|=2;break;}
            int ds=step<dd.count?step:dd.count-1; ca=donor[dd.first+ds]; ca.n_units=ob.farm(ob.player()).n_units();
            bonsaikg4_champion::apply_legacy(ob,lp,legacy,ca);
            bonsaikg_v6::apply_cuda_engine_exploits(ob,ca,exploit,legacy.external_ok!=0,100);
            reconcile_feed(st.blackboard,feed_slot,feed_count,ca);
            if(!same_action(ca,base)&&rewrites<65535)++rewrites;
        } else {
            if(!canonical_action(ob,bp,lp,legacy,exploit,st.blackboard,feed_slot,feed_count,ca)){flags|=2;break;}
        }
        Action oa{}; TapeDesc d=od[c.tape]; if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];} oa.n_units=sim.st.farms[os].n_units;
        if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
    }
    ResultBin r{}; r.own_money=(int32_t)sim.st.farms[c.seat].money; r.opp_money=(int32_t)sim.st.farms[1-c.seat].money; r.rewrites=rewrites; r.flags=flags; out[job]=r;
}
static void write_results(const std::string& path,const std::vector<ResultBin>& rows,uint32_t start,int nmasks,int ncases) {
    std::ofstream o(path,std::ios::binary); if(!o) throw std::runtime_error("cannot write results");
    ResultHeader h{{'B','4','D','M','1','\0','\0','\0'},1u,start,(uint32_t)nmasks,(uint32_t)ncases,65536u};
    o.write((char*)&h,sizeof(h)); o.write((char*)rows.data(),rows.size()*sizeof(rows[0]));
}
}

int main(int argc,char** argv){try{
    std::string controller,routes,legacy,casesp,donorp,outp; int start=0,count=-1,feed_slot=5,feed_count=13,tpb=64;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};
        if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();
        else if(a=="--donor")donorp=need();else if(a=="--out")outp=need();else if(a=="--start-mask")start=std::stoi(need());else if(a=="--count")count=std::stoi(need());
        else if(a=="--tpb")tpb=std::stoi(need());else throw std::runtime_error("unknown "+a);}
    if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||donorp.empty()||outp.empty())throw std::runtime_error("required args");
    if(start<0||start>=65536)throw std::runtime_error("bad start-mask"); if(count<0||start+count>65536)count=65536-start; if(count<=0)throw std::runtime_error("bad count");
    const int hs[N_SPECIAL]={26,27,28,29,30,31,32,33,36,37,38,39,40,41,46,47}; KAGCUDA_CHECK(cudaMemcpyToSymbol(D_SPECIAL,hs,sizeof(hs)));
    auto owned=bonsaikg4::load_compiled4(controller); auto hp=owned.view(); DevicePolicyOwner4 po;
    po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);
    auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;
    auto rp=bonsaikg_v6::RoutePack::load(routes),lg=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lg.routes);auto*dla=cp(lg.actions);
    DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack lp{dlr,dla,(int)lg.routes.size()};
    auto hc=load_cases(casesp);std::vector<Action>hoa;std::vector<TapeDesc>hod;std::vector<CaseDev>hcd;
    for(const auto&c:hc){auto t=load_native_tape_json(c.path);int off=(int)hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)hod.size()-1,c.seat,c.seed,c.family,c.protected_flag,c.own_history,c.role});}
    auto hd=load_native_tape_json(donorp);std::vector<Action>hda=hd.actions;TapeDesc hdd{0,(int)hda.size()};
    auto*doa=cp(hoa);auto*dod=cp(hod);auto*dda=cp(hda);auto*dc=cp(hcd);size_t jobs=(size_t)count*hc.size();
    int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;Sim*ds=nullptr;ResultBin*dout=nullptr;
    if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,jobs*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,jobs*hp.n_state*sizeof(*dns)));
    KAGCUDA_CHECK(cudaMalloc((void**)&dm,jobs*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(*dout)));
    configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));
    kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,lp,doa,dod,dda,hdd,dc,(int)hc.size(),(uint32_t)start,count,feed_slot,feed_count,dbb,dns,dm,ds,dout);
    KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    std::vector<ResultBin>hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(*dout),cudaMemcpyDeviceToHost));int bad=0;for(auto&r:hr)bad+=r.flags!=0;write_results(outp,hr,(uint32_t)start,count,(int)hc.size());
    std::cout<<"start="<<start<<" masks="<<count<<" cases="<<hc.size()<<" games="<<jobs<<" gpu_ms="<<ms<<" games_s="<<(jobs/(ms/1000.0))<<" bad="<<bad<<"\n";
    cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dda);cudaFree(dc);if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
