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

constexpr int ORACLE_UNITS = 13;
constexpr int U_LOCAL = 6;
constexpr int U_MARKET = 8;
constexpr int U_CANDIDATES = 1 + ORACLE_UNITS * U_LOCAL + U_MARKET;
constexpr int W_PAIRS = ORACLE_UNITS * (ORACLE_UNITS - 1) / 2;
constexpr int W_CANDIDATES = 1 + W_PAIRS + 2;
enum Branch : uint32_t { BRANCH_U = 1, BRANCH_W = 2, BRANCH_M = 3 };

#pragma pack(push,1)
struct ResultHeader {
    char magic[8];
    uint32_t version, branch, candidates, cases, probes, packed_offsets;
};
struct ResultBin { int32_t own_money, opp_money; uint16_t changes, flags; };
#pragma pack(pop)
static_assert(sizeof(ResultBin) == 12, "oracle result ABI");

struct CaseHost {
    std::string path;
    int seat = 0;
    uint64_t seed = 0;
    int family = 0;
    int protected_flag = 0;
    int own_history = 0;
    int role = 0;
};
struct CaseDev {
    int tape = 0, seat = 0;
    uint64_t seed = 0;
    int family = 0, protected_flag = 0, own_history = 0, role = 0;
};
struct OppDesc { int first = 0, count = 0; };
struct MacroCand { uint16_t route = 0xffffu, horizon = 0; };

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
    std::vector<std::string> out; size_t p=0;
    for(;;){size_t q=s.find('\t',p);out.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}
    return out;
}

static std::vector<CaseHost> load_cases(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open cases: "+path);
    std::vector<CaseHost> out; std::string line;
    while(std::getline(in,line)) {
        if(line.empty()||line[0]=='#') continue;
        auto z=split_tab(line); if(z.size()!=7) throw std::runtime_error("oracle case needs 7 tab fields");
        CaseHost c; c.path=z[0]; c.seat=std::stoi(z[1]); c.seed=std::stoull(z[2]); c.family=std::stoi(z[3]);
        c.protected_flag=std::stoi(z[4]); c.own_history=std::stoi(z[5]); c.role=std::stoi(z[6]);
        if(c.seat<0||c.seat>1) throw std::runtime_error("invalid seat"); out.push_back(c);
    }
    if(out.empty()) throw std::runtime_error("empty cases"); return out;
}

static std::vector<int> load_probes(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open probes: "+path);
    std::vector<int> out; int x=0; while(in>>x){if(x<0||x>=720)throw std::runtime_error("probe out of range");out.push_back(x);} if(out.empty())throw std::runtime_error("empty probes");return out;
}

static std::vector<int> parse_csv_int(const std::string& s) {
    std::vector<int> out; size_t p=0;
    for(;;){size_t q=s.find(',',p);std::string x=s.substr(p,q==std::string::npos?std::string::npos:q-p);if(!x.empty())out.push_back(std::stoi(x));if(q==std::string::npos)break;p=q+1;}
    return out;
}

static bonsaikg_v6::RoutePack load_legacy_pack(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("legacy pack");
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),{});
    auto full=bonsaikg::sha256_bytes(raw);
    if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256) throw std::runtime_error("legacy hash");
    bonsaikg_v6::PackHeader h{}; if(raw.size()<sizeof(h)) throw std::runtime_error("short legacy pack"); std::memcpy(&h,raw.data(),sizeof(h));
    size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc), expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);
    if(raw.size()!=expected) throw std::runtime_error("legacy pack size");
    bonsaikg_v6::RoutePack o; o.routes.resize(h.route_count); o.actions.resize(h.action_count);
    std::memcpy(o.routes.data(),raw.data()+sizeof(h),db); std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0])); return o;
}

struct DeviceBasePack {
    const bonsaikg_v6::RouteDesc* routes=nullptr; const bonsaikg_v6::PackedAction* actions=nullptr; int n=0;
    __device__ __forceinline__ const bonsaikg_v6::PackedAction* at(uint16_t route,int step) const {
        for(int i=0;i<n;++i) if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;} return nullptr;
    }
};

__device__ __forceinline__ bool same_unit(const UnitAction& a,const UnitAction& b){return a.op==b.op&&a.arg==b.arg&&a.n==b.n;}
__device__ __forceinline__ bool same_order(const Order& a,const Order& b){return a.op==b.op&&a.item==b.item&&a.n==b.n;}
__device__ __forceinline__ bool same_action(const Action& a,const Action& b){
    if(a.n_units!=b.n_units||a.n_orders!=b.n_orders)return false;
    for(int i=0;i<a.n_units;++i)if(!same_unit(a.units[i],b.units[i]))return false;
    for(int i=0;i<a.n_orders;++i)if(!same_order(a.orders[i],b.orders[i]))return false;
    return true;
}

__device__ __forceinline__ uint16_t reconcile_feed(const int32_t* bb,int first_slot,int count,Action& a){
    uint16_t changed=0; for(int u=0;u<count&&u<a.n_units;++u){if(!bb[first_slot+u]||a.units[u].op!=OP_FEED)continue;a.units[u]=UnitAction{};if(changed<65535)++changed;} return changed;
}

__device__ __forceinline__ bool harvest_legal(const ObservationView& obs,const Tile& t){
    if(t.yield_units<=0)return false;
    if(t.kind==T_PLANT){if(t.what>=N_CROPS)return false;return obs.day()-t.planted_day>=D_CROPS[t.what].first_yield_day;}
    return (t.kind==T_PASTURE||t.kind==T_COOP)&&t.has_animal;
}

__device__ __forceinline__ bool local_alt(const ObservationView& obs,int u,int which,UnitAction& out){
    const int p=obs.player(); const auto farm=obs.farm(p); const auto priv=obs.private_farm();
    if(u<0||u>=farm.n_units())return false; const int x=farm.worker_x(u),y=farm.worker_y(u); if(x<0||x>=BOARD||y<0||y>=BOARD)return false;
    const Tile& t=farm.tile(x,y); out=UnitAction{};
    switch(which){
        case 0: if(!harvest_legal(obs,t))return false; out.op=OP_HARVEST; return true;
        case 1: if(t.kind!=T_PLANT||t.watered_today)return false; out.op=OP_WATER; return true;
        case 2: if(!(t.kind==T_PASTURE||t.kind==T_COOP)||!t.has_animal||t.fed_today||priv.inventory(u,WHEAT)<=0)return false; out.op=OP_FEED; return true;
        case 3: if(!(t.kind==T_PASTURE||t.kind==T_COOP)||!t.has_animal||t.cared_today)return false; out.op=OP_CARE; return true;
        case 4: if(!(t.kind==T_PASTURE||t.kind==T_COOP)||!t.has_animal||!t.fertilizer_available)return false; out.op=OP_COLLECT_FERTILIZER; return true;
        case 5: if(t.kind!=T_PLANT||priv.inventory(u,FERTILIZER)<=0)return false; out.op=OP_FERTILIZE; return true;
    }
    return false;
}

__device__ __forceinline__ void compact_orders(Action& a,uint8_t drop_kind){
    int w=0; for(int i=0;i<a.n_orders;++i){const Order o=a.orders[i];bool drop=false;
        if(drop_kind==0)drop=o.op==M_SELL;
        else if(drop_kind==1)drop=o.op==M_HIRE;
        else if(drop_kind==2)drop=o.op==M_BUY_LAND;
        else if(drop_kind==3)drop=o.op==M_BUY_SEED||o.op==M_BUY_PRODUCT||o.op==M_BUY_ANIMAL;
        if(!drop)a.orders[w++]=o;
    } a.n_orders=w;
}

__device__ __forceinline__ int order_class(const Order& o,int mode){
    if(mode==0)return o.op==M_SELL?0:1;
    if(mode==1)return (o.op==M_BUY_SEED||o.op==M_BUY_PRODUCT||o.op==M_BUY_ANIMAL)?0:1;
    return 0;
}

__device__ __forceinline__ void stable_order_partition(Action& a,int mode){
    Order tmp[MAX_ORDERS];int w=0;for(int cls=0;cls<2;++cls)for(int i=0;i<a.n_orders;++i)if(order_class(a.orders[i],mode)==cls)tmp[w++]=a.orders[i];for(int i=0;i<w;++i)a.orders[i]=tmp[i];
}

__device__ __forceinline__ void apply_u(const ObservationView& obs,int candidate,Action& a){
    if(candidate<=0)return; int j=candidate-1;
    if(j<ORACLE_UNITS*U_LOCAL){int u=j/U_LOCAL,which=j%U_LOCAL;UnitAction alt{};if(local_alt(obs,u,which,alt))a.units[u]=alt;return;}
    const int m=j-ORACLE_UNITS*U_LOCAL;
    if(m==0)compact_orders(a,0);
    else if(m==1)compact_orders(a,1);
    else if(m==2)compact_orders(a,2);
    else if(m==3)compact_orders(a,3);
    else if(m==4)a.n_orders=0;
    else if(m==5)stable_order_partition(a,0);
    else if(m==6)stable_order_partition(a,1);
    else if(m==7)for(int i=0,jj=a.n_orders-1;i<jj;++i,--jj){Order t=a.orders[i];a.orders[i]=a.orders[jj];a.orders[jj]=t;}
}

__device__ __forceinline__ bool decode_pair(int candidate,int& a,int& b){
    int k=candidate-1; if(k<0||k>=W_PAIRS)return false; int cur=0;
    for(int i=0;i<ORACLE_UNITS;++i)for(int j=i+1;j<ORACLE_UNITS;++j){if(cur==k){a=i;b=j;return true;}++cur;}return false;
}

__device__ __forceinline__ void apply_w(int candidate,Action& a){
    if(candidate<=0)return; int x=0,y=0;
    if(decode_pair(candidate,x,y)){if(x<a.n_units&&y<a.n_units){UnitAction t=a.units[x];a.units[x]=a.units[y];a.units[y]=t;}return;}
    int n=a.n_units<ORACLE_UNITS?a.n_units:ORACLE_UNITS; if(n<=1)return;
    if(candidate==1+W_PAIRS){UnitAction first=a.units[0];for(int i=0;i+1<n;++i)a.units[i]=a.units[i+1];a.units[n-1]=first;}
    else if(candidate==2+W_PAIRS){UnitAction last=a.units[n-1];for(int i=n-1;i>0;--i)a.units[i]=a.units[i-1];a.units[0]=last;}
}

__device__ __forceinline__ void build_final_action(const ObservationView& raw,const DeviceBasePack& base_pack,const bonsaikg4_champion::CudaLegacyPack& legacy_pack,
                                                    bonsaikg4_champion::CudaLegacyState& legacy,bonsaikg_v6::CudaEngineExploitState& exploit,
                                                    const int32_t* bb,int feed_slot,int feed_count,uint16_t route_override,Action& out){
    int route=bb[1]; if(route_override!=0xffffu)route=route_override; const auto* p=base_pack.at((uint16_t)route,raw.step());
    if(!p){out=Action{};out.n_units=raw.farm(raw.player()).n_units();return;}
    bonsaikg_v6::materialize_cuda(*p,raw.farm(raw.player()).n_units(),out);
    bonsaikg4_champion::apply_legacy(raw,legacy_pack,legacy,out);
    bonsaikg_v6::apply_cuda_engine_exploits(raw,out,exploit,legacy.external_ok!=0,100);
    reconcile_feed(bb,feed_slot,feed_count,out);
}

__global__ void eval_kernel(Config cfg0,DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack base_pack,bonsaikg4_champion::CudaLegacyPack legacy_pack,
                            const Action* opp,const OppDesc* od,const CaseDev* cases,int ncases,const int* probes,int nprobes,int branch,int ncand,
                            const MacroCand* macros,int w_horizon,int32_t* bb,bonsaikg4::NodeState4* ns,bonsaikg4::RuntimeMeta4* meta,int feed_slot,int feed_count,
                            Sim* sims,ResultBin* out){
    const int job=blockIdx.x*blockDim.x+threadIdx.x,total=ncases*nprobes*ncand;if(job>=total)return;
    int z=job,candidate=z%ncand;z/=ncand;int pi=z%nprobes;int ci=z/nprobes;const CaseDev c=cases[ci];const int probe=probes[pi];
    Config cfg=cfg0;cfg.seed=c.seed;Sim& sim=sims[job];sim.init(cfg,&tables);
    bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)job*policy.n_bb:nullptr,policy.n_state?ns+(size_t)job*policy.n_state:nullptr,meta+job};bonsaikg4::reset_runtime(policy,st);
    bonsaikg4_champion::CudaLegacyState legacy{};bonsaikg_v6::CudaEngineExploitState exploit{};bonsaikg4_champion::reset_legacy(legacy);bonsaikg_v6::reset_cuda_engine_exploits(exploit);
    uint16_t changes=0,flags=0;
    while(!sim.st.done){
        const int step=sim.st.step,os=1-c.seat;ObservationView raw=make_observation(sim,c.seat);bonsaikg4::ActionResult4 ignored{};
        auto status=bonsaikg4::tick4(policy,st,bonsaikg4::CudaObservation4(raw),0,ignored,nullptr);if(meta[job].fault||status!=bonsaikg4::SUCCESS){flags|=1;break;}
        bool active=false;uint16_t route_override=0xffffu;
        if(branch==BRANCH_U)active=candidate>0&&step==probe;
        else if(branch==BRANCH_W)active=candidate>0&&step>=probe&&step<probe+w_horizon;
        else if(branch==BRANCH_M&&candidate>0){const MacroCand mc=macros[candidate];active=step>=probe&&step<probe+(int)mc.horizon;if(active)route_override=mc.route;}

        // Actual branch state uses the actual overlay states. Build a comparison action
        // from copied states only on active steps so `changes` means final executable
        // action difference after frozen overlays/feed deferral.
        Action ca{};
        if(active){
            auto legacy_cmp=legacy;auto exploit_cmp=exploit;Action baseline{};build_final_action(raw,base_pack,legacy_pack,legacy_cmp,exploit_cmp,st.blackboard,feed_slot,feed_count,0xffffu,baseline);
            // For U/W, build the same incumbent first, mutate it, then re-apply only
            // feed deferral after mutation. Legacy/exploit state progression remains
            // the canonical one because those layers are higher authority.
            if(branch==BRANCH_U||branch==BRANCH_W){
                build_final_action(raw,base_pack,legacy_pack,legacy,exploit,st.blackboard,feed_slot,0,0xffffu,ca);
                if(branch==BRANCH_U)apply_u(raw,candidate,ca);else apply_w(candidate,ca);
                reconcile_feed(st.blackboard,feed_slot,feed_count,ca);
            }else{
                build_final_action(raw,base_pack,legacy_pack,legacy,exploit,st.blackboard,feed_slot,feed_count,route_override,ca);
            }
            if(!same_action(ca,baseline)&&changes<65535)++changes;
        }else build_final_action(raw,base_pack,legacy_pack,legacy,exploit,st.blackboard,feed_slot,feed_count,0xffffu,ca);

        Action oa{};OppDesc d=od[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];}oa.n_units=sim.st.farms[os].n_units;
        if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
    }
    ResultBin r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.changes=changes;r.flags=flags;out[job]=r;
}

static void write_results(const std::string& path,const std::vector<ResultBin>& rows,int branch,int ncand,int ncases,int nprobes,int case_offset,int probe_offset){
    std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("cannot write result");ResultHeader h{{'B','4','O','R','1','\0','\0','\0'},1u,(uint32_t)branch,(uint32_t)ncand,(uint32_t)ncases,(uint32_t)nprobes,(uint32_t)(((case_offset&0xffff)<<16)|(probe_offset&0xffff))};o.write((char*)&h,sizeof(h));o.write((char*)rows.data(),rows.size()*sizeof(rows[0]));
}

} // namespace

int main(int argc,char** argv){try{
    std::string controller,routes,legacy,casesp,probesp,outp,branch_s,macro_routes_s="22,23,29,32,34,46,63,64,70,77,79",macro_horizons_s="24,72,144";int feed_slot=5,feed_count=13,w_horizon=24,tpb=64,case_start=0,case_count=0,probe_start=0,probe_count=0;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--probes")probesp=need();else if(a=="--out")outp=need();else if(a=="--branch")branch_s=need();else if(a=="--macro-routes")macro_routes_s=need();else if(a=="--macro-horizons")macro_horizons_s=need();else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--w-horizon")w_horizon=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--case-start")case_start=std::stoi(need());else if(a=="--case-count")case_count=std::stoi(need());else if(a=="--probe-start")probe_start=std::stoi(need());else if(a=="--probe-count")probe_count=std::stoi(need());else throw std::runtime_error("unknown "+a);}
    if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||probesp.empty()||outp.empty()||branch_s.empty())throw std::runtime_error("required args missing");
    int branch=branch_s=="U"?BRANCH_U:branch_s=="W"?BRANCH_W:branch_s=="M"?BRANCH_M:0;if(!branch)throw std::runtime_error("branch must be U/W/M");

    auto owned=bonsaikg4::load_compiled4(controller);auto hp=owned.view();if(hp.n_bb<(uint16_t)(feed_slot+feed_count))throw std::runtime_error("feed slots outside policy BB");DevicePolicyOwner4 po;po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;
    auto rp=bonsaikg_v6::RoutePack::load(routes),lp=load_legacy_pack(legacy);auto* dr=cp(rp.routes);auto* da=cp(rp.actions);auto* dlr=cp(lp.routes);auto* dla=cp(lp.actions);DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack ldp{dlr,dla,(int)lp.routes.size()};

    auto all_cases=load_cases(casesp);auto all_probes=load_probes(probesp);if(case_start<0||case_start>(int)all_cases.size())throw std::runtime_error("case-start");if(probe_start<0||probe_start>(int)all_probes.size())throw std::runtime_error("probe-start");int ce=case_count>0?(((case_start+case_count)<(int)all_cases.size())?(case_start+case_count):(int)all_cases.size()):(int)all_cases.size();int pe=probe_count>0?(((probe_start+probe_count)<(int)all_probes.size())?(probe_start+probe_count):(int)all_probes.size()):(int)all_probes.size();if(ce<=case_start||pe<=probe_start)throw std::runtime_error("empty slice");
    std::vector<CaseHost> hc(all_cases.begin()+case_start,all_cases.begin()+ce);std::vector<int> probes(all_probes.begin()+probe_start,all_probes.begin()+pe);
    std::vector<Action> hoa;std::vector<OppDesc> hod;std::vector<CaseDev> hcd;for(const auto& c:hc){auto t=load_native_tape_json(c.path);int off=(int)hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)hod.size()-1,c.seat,c.seed,c.family,c.protected_flag,c.own_history,c.role});}

    std::vector<MacroCand> hm;int ncand=branch==BRANCH_U?U_CANDIDATES:branch==BRANCH_W?W_CANDIDATES:1;hm.push_back({0xffffu,0});if(branch==BRANCH_M){auto mr=parse_csv_int(macro_routes_s),mh=parse_csv_int(macro_horizons_s);for(int r:mr){bool found=false;for(auto& d:rp.routes)if(d.route_id==r)found=true;if(!found)throw std::runtime_error("macro route absent: "+std::to_string(r));for(int h:mh){if(h<=0||h>720)throw std::runtime_error("macro horizon");hm.push_back({(uint16_t)r,(uint16_t)h});}}ncand=(int)hm.size();}

    auto* doa=cp(hoa);auto* dod=cp(hod);auto* dc=cp(hcd);auto* dprobes=cp(probes);auto* dmac=cp(hm);const int ncases=(int)hc.size(),nprobes=(int)probes.size();size_t jobs=(size_t)ncases*nprobes*ncand;
    int32_t* dbb=nullptr;bonsaikg4::NodeState4* dns=nullptr;bonsaikg4::RuntimeMeta4* dm=nullptr;Sim* ds=nullptr;ResultBin* dout=nullptr;if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,jobs*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,jobs*hp.n_state*sizeof(bonsaikg4::NodeState4)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,jobs*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(*dout)));
    configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,ldp,doa,dod,dc,ncases,dprobes,nprobes,branch,ncand,dmac,w_horizon,dbb,dns,dm,feed_slot,feed_count,ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<ResultBin> hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(ResultBin),cudaMemcpyDeviceToHost));write_results(outp,hr,branch,ncand,ncases,nprobes,case_start,probe_start);int bad=0,active=0;for(auto&r:hr){bad+=r.flags!=0;active+=r.changes>0;}std::cout<<"branch="<<branch_s<<" cases="<<ncases<<" probes="<<nprobes<<" candidates="<<ncand<<" games="<<jobs<<" gpu_ms="<<ms<<" games_s="<<(jobs/(ms/1000.0))<<" active_jobs="<<active<<" bad="<<bad<<" sim_MiB="<<(jobs*sizeof(Sim)/(1024.0*1024.0))<<"\n";
    cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dc);cudaFree(dprobes);if(dmac)cudaFree(dmac);if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
