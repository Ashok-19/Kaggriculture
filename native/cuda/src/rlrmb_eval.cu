#include "rlrmb_io.hpp"

#include "bonsai_kg_v4.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "kag_adapter_v4.cuh"
#include "cuda_compat.cuh"
#include "extended_route_pack.hpp"
#include "assets/champion/identity.hpp"
#include "route_pack.hpp"
#include "runtime.cuh"
#include "tape_io.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;
namespace {

#pragma pack(push,1)
struct ResultHeader { char magic[8]; uint32_t version,candidates,cases; };
#pragma pack(pop)
struct CaseHost { std::string path; int seat=0; uint64_t seed=0; };
struct CaseDev { int tape=0,seat=0; uint64_t seed=0; };
struct TapeDesc { int first=0,count=0; };

struct DevicePolicyOwner4 {
    bonsaikg4::Node4*n=nullptr;uint16_t*c=nullptr;bonsaikg4::BbDecl4*bb=nullptr;bonsaikg4::ValueRef4*v=nullptr;
    bonsaikg4::FeatureArgRef4*a=nullptr;bonsaikg4::Predicate4*q=nullptr;bonsaikg4::Lifecycle4*l=nullptr;bonsaikg4::Effect4*e=nullptr;
    ~DevicePolicyOwner4(){if(n)cudaFree(n);if(c)cudaFree(c);if(bb)cudaFree(bb);if(v)cudaFree(v);if(a)cudaFree(a);if(q)cudaFree(q);if(l)cudaFree(l);if(e)cudaFree(e);}
};
template<class T> static T* cp(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
static std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost> load_cases(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open cases");std::vector<CaseHost>o;std::string l;while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;auto z=split_tab(l);if(z.size()<3)throw std::runtime_error("case line needs path, seat, seed");o.push_back({z[0],std::stoi(z[1]),std::stoull(z[2])});}if(o.empty())throw std::runtime_error("empty cases");return o;}
static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("legacy pack");std::vector<uint8_t>raw((std::istreambuf_iterator<char>(in)),{});auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy hash");bonsaikg_v6::PackHeader h{};if(raw.size()<sizeof(h))throw std::runtime_error("short legacy");std::memcpy(&h,raw.data(),sizeof(h));size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc),expected=sizeof(h)+db+(size_t)h.action_count*sizeof(bonsaikg_v6::PackedAction);if(raw.size()!=expected)throw std::runtime_error("legacy size");bonsaikg_v6::RoutePack o;o.routes.resize(h.route_count);o.actions.resize(h.action_count);std::memcpy(o.routes.data(),raw.data()+sizeof(h),db);std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0]));return o;}
struct DeviceBasePack{const bonsaikg_v6::RouteDesc*routes=nullptr;const bonsaikg_v6::PackedAction*actions=nullptr;int n=0;__device__ __forceinline__ const bonsaikg_v6::PackedAction*at(uint16_t route,int step)const{for(int i=0;i<n;++i)if(routes[i].route_id==route){int s=step;if(s<0)s=0;if(s>=routes[i].steps)s=routes[i].steps-1;return actions+routes[i].first+s;}return nullptr;}};

__device__ __forceinline__ uint16_t reconcile_feed(const int32_t*bb,int first,int count,Action&a){uint16_t n=0;for(int u=0;u<count&&u<a.n_units;++u)if(bb[first+u]&&a.units[u].op==OP_FEED){a.units[u]=UnitAction{};++n;}return n;}
__device__ __forceinline__ bool same_unit(const UnitAction&a,const UnitAction&b){return a.op==b.op&&a.arg==b.arg&&a.n==b.n;}
__device__ __forceinline__ bool same_order(const Order&a,const Order&b){return a.op==b.op&&a.item==b.item&&a.n==b.n;}
__device__ __forceinline__ bool same_action(const Action&a,const Action&b){if(a.n_units!=b.n_units||a.n_orders!=b.n_orders)return false;for(int i=0;i<a.n_units;++i)if(!same_unit(a.units[i],b.units[i]))return false;for(int i=0;i<a.n_orders;++i)if(!same_order(a.orders[i],b.orders[i]))return false;return true;}
__device__ __forceinline__ bool materialize_parent(const ObservationView&raw,const DeviceBasePack&bp,const int32_t*bb,int strategic_route_slot,Action&out){int route=bb[1];if(strategic_route_slot>=0&&bb[strategic_route_slot])route=(uint16_t)bb[strategic_route_slot];const auto*p=bp.at((uint16_t)route,raw.step());if(!p)return false;bonsaikg_v6::materialize_cuda(*p,raw.farm(raw.player()).n_units(),out);return true;}
__device__ __forceinline__ Action unpack(const rlrmb::ActionRecord&r,int n_units){Action a{};a.n_units=n_units<1?1:(n_units>MAX_UNITS?MAX_UNITS:n_units);a.n_orders=r.n_orders>MAX_ORDERS?MAX_ORDERS:r.n_orders;for(int i=0;i<a.n_units;++i){a.units[i].op=r.units[i].op;a.units[i].arg=r.units[i].arg;a.units[i].n=r.units[i].n;}for(int i=0;i<a.n_orders;++i){a.orders[i].op=r.orders[i].op;a.orders[i].item=r.orders[i].item;a.orders[i].n=r.orders[i].n;}return a;}
__device__ __forceinline__ Action merge(const Action&p,const Action&s,uint8_t mask){Action a=p;if(mask&rlrmb::MASK_FARMER)a.units[0]=s.units[0];if(mask&rlrmb::MASK_HANDS){for(int i=1;i<a.n_units;++i)a.units[i]=s.units[i];}if(mask&rlrmb::MASK_MARKET){a.n_orders=s.n_orders;for(int i=0;i<s.n_orders;++i)a.orders[i]=s.orders[i];}return a;}
__device__ __forceinline__ int purchase_metric(uint8_t objective,const ObservationView&o){auto f=o.farm(o.player());if(objective==rlrmb::BUY_LAND)return f.n_quadrants();if(objective==rlrmb::HIRE)return f.hires_today();if(objective==rlrmb::BUY_SEED){int n=0;for(int i=0;i<N_CROPS;++i)n+=o.private_farm().seed(i);return n;}if(objective==rlrmb::BUY_PRODUCT||objective==rlrmb::BUY_ANIMAL)return o.private_farm().shed_total();return 0;}
__device__ __forceinline__ bool compare_value(int32_t lhs,uint16_t cmp,int32_t rhs){return cmp==0?lhs==rhs:cmp==1?lhs!=rhs:cmp==2?lhs<=rhs:cmp==3?lhs>=rhs:false;}

struct StrategicSnapshot { int money=0,opp_money=0,units=0,opp_units=0,land=0,opp_land=0,hires=0,opp_hires=0,shed=0,seed=0; };
__device__ __forceinline__ StrategicSnapshot strategic_snapshot(const ObservationView&o){StrategicSnapshot s{};auto own=o.farm(o.player()),opp=o.farm(1-o.player());s.money=own.money();s.opp_money=opp.money();s.units=own.n_units();s.opp_units=opp.n_units();s.land=own.n_quadrants();s.opp_land=opp.n_quadrants();s.hires=own.hires_today();s.opp_hires=opp.hires_today();s.shed=o.private_farm().shed_total();for(int i=0;i<N_CROPS;++i)s.seed+=o.private_farm().seed(i);return s;}
__device__ __forceinline__ int16_t compact_delta(int a,int b){int d=a-b;return (int16_t)(d>32767?32767:(d<-32768?-32768:d));}
__device__ __forceinline__ bool same_farmer(const Action&a,const Action&b){return a.n_units&&b.n_units&&same_unit(a.units[0],b.units[0]);}
__device__ __forceinline__ bool same_hands(const Action&a,const Action&b){if(a.n_units!=b.n_units)return false;for(int i=1;i<a.n_units;++i)if(!same_unit(a.units[i],b.units[i]))return false;return true;}
__device__ __forceinline__ bool same_market(const Action&a,const Action&b){if(a.n_orders!=b.n_orders)return false;for(int i=0;i<a.n_orders;++i)if(!same_order(a.orders[i],b.orders[i]))return false;return true;}
__device__ __forceinline__ uint64_t trace_word(uint64_t h,int32_t value){uint32_t v=(uint32_t)value;for(int i=0;i<4;++i){h^=(uint8_t)(v>>(8*i));h*=1099511628211ull;}return h;}
__device__ __forceinline__ uint64_t trace_action(uint64_t h,const Action&a){h=trace_word(h,a.n_units);for(int i=0;i<a.n_units;++i){h=trace_word(h,a.units[i].op);h=trace_word(h,a.units[i].arg);h=trace_word(h,a.units[i].n);}h=trace_word(h,a.n_orders);for(int i=0;i<a.n_orders;++i){h=trace_word(h,a.orders[i].op);h=trace_word(h,a.orders[i].item);h=trace_word(h,a.orders[i].n);}return h;}

__global__ void kernel(Config cfg0,DeviceTables tables,bonsaikg4::PolicyView4 policy,DeviceBasePack bp,bonsaikg4_champion::CudaLegacyPack lp,
 const Action*case_actions,const TapeDesc*case_desc,const CaseDev*cases,int nc,const rlrmb::ProgramRecord*programs,int np,const rlrmb::PhaseRecord*phases,
 const rlrmb::DecisionRecord*decisions,const rlrmb::MotifRecord*motifs,int nmotifs,const rlrmb::ActionRecord*motif_actions,int feed_slot,int feed_count,int strategic_route_slot,
 int ir_version,int32_t*bb,bonsaikg4::NodeState4*ns,bonsaikg4::RuntimeMeta4*meta,Sim*sims,rlrmb::ResultRecord*out){
 int job=blockIdx.x*blockDim.x+threadIdx.x,total=nc*np;if(job>=total)return;int pi=job/nc,ci=job-pi*nc;const auto p=programs[pi];const auto c=cases[ci];uint16_t initiation_step=p.initiation_step&0x03ffu,max_total_steps=p.max_total_steps&0x03ffu;uint16_t gate_count=ir_version==3?p.max_total_steps>>10:0,clause_count=ir_version==3?p.initiation_step>>10:0;uint32_t gate_first=p.phase_first+p.phase_count+((p.lane&0x80u)?1u:0u);Config cfg=cfg0;cfg.seed=c.seed;Sim&sim=sims[job];sim.init(cfg,&tables);bonsaikg4::RuntimeView4 st{policy.n_bb?bb+(size_t)job*policy.n_bb:nullptr,policy.n_state?ns+(size_t)job*policy.n_state:nullptr,meta+job};bonsaikg4::reset_runtime(policy,st);bonsaikg4_champion::CudaLegacyState legacy{};bonsaikg_v6::CudaEngineExploitState exploit{};bonsaikg4_champion::reset_legacy(legacy);bonsaikg_v6::reset_cuda_engine_exploits(exploit);rlrmb::RuntimeState rs{};uint16_t changes=0,first_step=0xffffu,last_step=0xffffu,activation_count=0,phase_path=0,farmer_changes=0,hands_changes=0,market_changes=0;uint64_t action_trace_hash=0;uint8_t activated=0,phase_usage=0,branch_path=0,completion_reason=rlrmb::COMPLETE_NONE;StrategicSnapshot entry{};
 if(p.phase_count<1||p.phase_count>rlrmb::MAX_PHASES){rs.fault=rlrmb::FAULT_PHASE_COUNT;}
 while(!sim.st.done&&!rs.fault){int step=sim.st.step,os=1-c.seat;ObservationView raw=make_observation(sim,c.seat);bonsaikg4::ActionResult4 ignored{};auto status=bonsaikg4::tick4(policy,st,bonsaikg4::CudaObservation4(raw),0,ignored,nullptr);if(meta[job].fault||status!=bonsaikg4::SUCCESS){rs.fault=rlrmb::FAULT_INTERNAL;break;}Action base{};if(!materialize_parent(raw,bp,st.blackboard,strategic_route_slot,base)){rs.fault=rlrmb::FAULT_INTERNAL;break;}Action ca=base;bool compatible=true;
  if(step>=initiation_step&&!rs.complete){if(ir_version==3&&!activated&&step==initiation_step){bool gate=false;bonsaikg4::CudaObservation4 o4(raw);for(uint16_t clause=0;clause<clause_count&&!gate;++clause){bool match=true;for(uint16_t i=0;i<gate_count;++i){const auto&g=phases[gate_first+i];if(g.reserved==clause&&!compare_value(o4.feature(g.feature,g.condition_index,0),g.comparator,g.condition_value)){match=false;break;}}if(match)gate=true;}if(!gate)rs.complete=1;}if(!rs.complete){uint32_t phase_offset=p.phase_first+(rs.in_recovery?p.phase_count:rs.phase);if(!rs.active){rs.active=1;rs.phase_steps=0;rs.entry_day=raw.day();rs.entry_shops=raw.town().n_shops();const auto&ep=phases[phase_offset];rs.entry_purchase=purchase_metric(ep.purchase,raw);if(!activated){entry=strategic_snapshot(raw);first_step=(uint16_t)step;action_trace_hash=1469598103934665603ull;}activated=1;last_step=(uint16_t)step;if(activation_count<65535)++activation_count;uint8_t phase_code=rs.in_recovery?p.phase_count:rs.phase;phase_usage|=(uint8_t)(1u<<phase_code);phase_path=(uint16_t)((phase_path<<3)|(phase_code+1));}else last_step=(uint16_t)step;const auto&ph=phases[phase_offset];if(ph.termination>rlrmb::TERM_MAX_STEPS){rs.fault=rlrmb::FAULT_TERMINATION;break;}if(ph.termination==rlrmb::TERM_PUBLIC_STATE&&ph.feature>bonsaikg4::LAST_FEATURE_V1){rs.fault=rlrmb::FAULT_FEATURE;break;}Action source=base;if(ph.source==rlrmb::SRC_DONOR_BLOCK||ph.source==rlrmb::SRC_SEMANTIC_SPLICE){if(ph.motif_index>=nmotifs){rs.fault=rlrmb::FAULT_MOTIF;break;}const auto&m=motifs[ph.motif_index];if(rs.phase_steps>=m.action_count){rs.fault=rlrmb::FAULT_MOTIF_EXHAUSTED;break;}source=unpack(motif_actions[m.action_first+rs.phase_steps],raw.farm(raw.player()).n_units());}else if(ph.source==rlrmb::SRC_PRIMITIVE){if(ph.primitive!=rlrmb::PRIM_LIQUIDITY_REBALANCE){rs.fault=rlrmb::FAULT_PRIMITIVE;break;}source=Action{};source.n_units=raw.farm(raw.player()).n_units();if(ph.primitive_enabled){source.n_orders=1;source.orders[0].op=M_BUY_PRODUCT;source.orders[0].item=ph.primitive_item;source.orders[0].n=ph.primitive_quantity;}}else if(ph.source!=rlrmb::SRC_PARENT){rs.fault=rlrmb::FAULT_SOURCE;break;}ca=merge(base,source,ph.mask);action_trace_hash=trace_action(action_trace_hash,ca);compatible=same_action(ca,base);if(!compatible&&changes<65535)++changes;if(!same_farmer(ca,base)&&farmer_changes<65535)++farmer_changes;if(!same_hands(ca,base)&&hands_changes<65535)++hands_changes;if(!same_market(ca,base)&&market_changes<65535)++market_changes;}}
  bonsaikg4_champion::apply_legacy(raw,lp,legacy,ca);bonsaikg_v6::apply_cuda_engine_exploits(raw,ca,exploit,legacy.external_ok!=0,100);reconcile_feed(st.blackboard,feed_slot,feed_count,ca);Action oa{};TapeDesc d=case_desc[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=case_actions[d.first+s];}oa.n_units=sim.st.farms[os].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
  if(rs.active){const auto&ph=phases[p.phase_first+(rs.in_recovery?p.phase_count:rs.phase)];++rs.phase_steps;++rs.program_steps;ObservationView post=make_observation(sim,c.seat);uint8_t reason=rlrmb::END_NONE;if(ph.termination==rlrmb::TERM_NEXT_EOD&&post.day()>rs.entry_day)reason=rlrmb::END_EOD;else if(ph.termination==rlrmb::TERM_NEXT_SHOP&&post.town().n_shops()>rs.entry_shops)reason=rlrmb::END_SHOP;else if(ph.termination==rlrmb::TERM_PURCHASE&&purchase_metric(ph.purchase,post)>=rs.entry_purchase+ph.target_count)reason=rlrmb::END_PURCHASE;else if(ph.termination==rlrmb::TERM_PUBLIC_STATE){bonsaikg4::CudaObservation4 o4(post);int32_t value=o4.feature(ph.feature,ph.condition_index,0);if(compare_value(value,ph.comparator,ph.condition_value))reason=rlrmb::END_PUBLIC_STATE;}else if(ph.termination==rlrmb::TERM_PARENT_COMPATIBLE&&compatible)reason=rlrmb::END_PARENT_COMPATIBLE;if(!reason&&rs.phase_steps>=ph.max_steps)reason=ph.termination==rlrmb::TERM_MAX_STEPS?rlrmb::END_MAX_STEPS:rlrmb::END_FIXED_CAP;if(!reason&&rs.program_steps>=max_total_steps)reason=rlrmb::END_PROGRAM_CAP;if(reason){rs.end_reason=reason;rs.active=0;if(rs.in_recovery||rs.program_steps>=max_total_steps){rs.complete=1;completion_reason=rs.in_recovery?rlrmb::COMPLETE_RECOVERY:rlrmb::COMPLETE_PROGRAM_CAP;}else{const rlrmb::DecisionRecord*decision=nullptr;for(uint32_t i=p.fallback;i<(uint32_t)p.fallback+p.reserved;++i)if(decisions[i].after_phase==rs.phase){decision=&decisions[i];break;}if(decision){if(decision->feature>bonsaikg4::LAST_FEATURE_V1){rs.fault=rlrmb::FAULT_FEATURE;break;}bonsaikg4::CudaObservation4 o4(post);bool matched=compare_value(o4.feature(decision->feature,decision->condition_index,0),decision->comparator,decision->condition_value);branch_path=(uint8_t)((branch_path<<2)|(matched?1:2));uint8_t kind=matched?decision->true_kind:decision->false_kind,target=matched?decision->true_phase:decision->false_phase;if(kind==rlrmb::CONT_ABORT){rs.complete=1;completion_reason=rlrmb::COMPLETE_ABORT;}else if(kind==rlrmb::CONT_RECOVERY){rs.in_recovery=1;}else if(kind==rlrmb::CONT_PHASE){if(target<=rs.phase||target>=p.phase_count){rs.fault=rlrmb::FAULT_CONTINUATION;break;}rs.phase=target;}else{rs.fault=rlrmb::FAULT_CONTINUATION;break;}}else if(rs.phase+1>=p.phase_count){rs.complete=1;completion_reason=rlrmb::COMPLETE_FINISHED;}else ++rs.phase;}}}
 }
 ObservationView terminal=make_observation(sim,c.seat);StrategicSnapshot finish=strategic_snapshot(terminal);rlrmb::ResultRecord r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.action_changes=changes;r.fault=rs.fault;r.activated=activated;r.final_phase=rs.phase;r.end_reason=rs.end_reason;r.complete=rs.complete;r.first_activation_step=first_step;r.last_activation_step=last_step;r.activation_count=activation_count;r.specialist_duration=rs.program_steps;r.phase_path=phase_path;r.phase_usage=phase_usage;r.branch_path=branch_path;r.completion_reason=rs.fault?rlrmb::COMPLETE_FAULT:completion_reason;r.family_farmer_changes=farmer_changes;r.family_hands_changes=hands_changes;r.family_market_changes=market_changes;r.action_trace_hash=action_trace_hash;if(activated){r.own_money_delta=compact_delta(finish.money,entry.money);r.opp_money_delta=compact_delta(finish.opp_money,entry.opp_money);r.own_units_delta=compact_delta(finish.units,entry.units);r.opp_units_delta=compact_delta(finish.opp_units,entry.opp_units);r.own_land_delta=compact_delta(finish.land,entry.land);r.opp_land_delta=compact_delta(finish.opp_land,entry.opp_land);r.own_hires_delta=compact_delta(finish.hires,entry.hires);r.opp_hires_delta=compact_delta(finish.opp_hires,entry.opp_hires);r.own_shed_delta=compact_delta(finish.shed,entry.shed);r.own_seed_delta=compact_delta(finish.seed,entry.seed);}for(int i=0;i<32;++i)r.semantic_id[i]=p.semantic_id[i];out[job]=r;
}

static void write_results(const std::string&p,const std::vector<rlrmb::ResultRecord>&r,int np,int nc){std::ofstream o(p,std::ios::binary);ResultHeader h{{'R','L','R','M','B','R','1','\0'},2u,(uint32_t)np,(uint32_t)nc};o.write((char*)&h,sizeof(h));o.write((char*)r.data(),r.size()*sizeof(r[0]));}

// Write result to stdout for the persistent protocol.
static void write_results_stdout(const std::vector<rlrmb::ResultRecord>&r,int np,int nc,uint32_t rid,float gpu_ms){
    size_t result_bytes=sizeof(ResultHeader)+r.size()*sizeof(rlrmb::ResultRecord);
    std::cout<<"RESULT "<<rid<<" "<<result_bytes<<" "<<gpu_ms<<"\n";
    ResultHeader h{{'R','L','R','M','B','R','1','\0'},2u,(uint32_t)np,(uint32_t)nc};
    std::cout.write(reinterpret_cast<const char*>(&h),sizeof(h));
    std::cout.write(reinterpret_cast<const char*>(r.data()),r.size()*sizeof(r[0]));
    std::cout.flush();
}

// Cached case panel (host + device allocations kept until shutdown).
struct PersistCasesEntry {
    int count=0;
    Action*dca=nullptr; TapeDesc*dcd=nullptr; CaseDev*dcases=nullptr;
    ~PersistCasesEntry(){if(dca)cudaFree(dca);if(dcd)cudaFree(dcd);if(dcases)cudaFree(dcases);}
    PersistCasesEntry()=default;
    PersistCasesEntry(const PersistCasesEntry&)=delete;
    PersistCasesEntry&operator=(const PersistCasesEntry&)=delete;
};

}

int main(int argc,char**argv){try{
 std::string controller,routes,legacy,casesp,programsp,outp;int feed_slot=5,feed_count=13,strategic_route_slot=-1,tpb=64;bool persist=false;
 for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--programs")programsp=need();else if(a=="--out")outp=need();else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--strategic-route-slot")strategic_route_slot=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--persist")persist=true;else throw std::runtime_error("unknown "+a);}
 if(persist){if(controller.empty()||routes.empty()||legacy.empty())throw std::runtime_error("--persist requires --controller --routes --legacy-routes");}
 else{if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||programsp.empty()||outp.empty())throw std::runtime_error("required args");}

 // Load static assets (policy, routes, legacy) once.
 auto owned=bonsaikg4::load_compiled4(controller);auto hp=owned.view();DevicePolicyOwner4 po;po.n=cp(owned.nodes);po.c=cp(owned.child_refs);po.bb=cp(owned.bb);po.v=cp(owned.values);po.a=cp(owned.feature_args);po.q=cp(owned.predicates);po.l=cp(owned.lifecycles);po.e=cp(owned.effects);auto dp=hp;dp.nodes=po.n;dp.child_refs=po.c;dp.bb=po.bb;dp.values=po.v;dp.feature_args=po.a;dp.predicates=po.q;dp.lifecycles=po.l;dp.effects=po.e;
 auto rp=bonsaikg4_champion::load_extended_route_pack(routes),lg=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lg.routes);auto*dla=cp(lg.actions);DeviceBasePack bp{dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaLegacyPack lp{dlr,dla,(int)lg.routes.size()};
 configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;

 if(persist){
  // Persistent mode: serve multiple batches from stdin.
  // Case panels are cached by path; per-batch program device memory is freed after each response.
  std::map<std::string,std::unique_ptr<PersistCasesEntry>> cases_cache;
  cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));
  std::string line;
  while(std::getline(std::cin,line)){
   if(line=="SHUTDOWN")break;
   // Parse: "BATCH rid batch_len cases_path"
   std::istringstream iss(line);std::string cmd;uint32_t rid,batch_len;std::string req_cases;
   iss>>cmd>>rid>>batch_len>>req_cases;
   if(cmd!="BATCH"){std::cerr<<"ERROR: expected BATCH, got "<<cmd<<"\n";return 2;}
   // Read batch bytes from stdin
   std::vector<uint8_t>raw(batch_len);
   if(!std::cin.read(reinterpret_cast<char*>(raw.data()),batch_len)){std::cerr<<"ERROR: short stdin\n";return 2;}
   // Load/cache cases
   auto&entry=cases_cache[req_cases];
   if(!entry){
    entry=std::make_unique<PersistCasesEntry>();
    auto hc=load_cases(req_cases);std::vector<Action>hca;std::vector<TapeDesc>hcd;std::vector<CaseDev>hcases;
    for(const auto&c:hc){auto t=load_native_tape(c.path);int off=(int)hca.size();hca.insert(hca.end(),t.actions.begin(),t.actions.end());hcd.push_back({off,(int)t.actions.size()});hcases.push_back({(int)hcd.size()-1,c.seat,c.seed});}
    entry->count=(int)hc.size();entry->dca=cp(hca);entry->dcd=cp(hcd);entry->dcases=cp(hcases);
   }
   PersistCasesEntry&cc=*entry;
   // Load batch from memory
   auto batch=rlrmb::load_batch_from_memory(raw);
   // Allocate per-batch device memory
   auto*dprograms=cp(batch.programs);auto*dphases=cp(batch.phases);auto*ddecisions=cp(batch.decisions);auto*dmotifs=cp(batch.motifs);auto*dactions=cp(batch.actions);
   size_t jobs=(size_t)batch.programs.size()*cc.count;
   int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;Sim*ds=nullptr;rlrmb::ResultRecord*dout=nullptr;
   if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,jobs*hp.n_bb*sizeof(int32_t)));
   if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,jobs*hp.n_state*sizeof(*dns)));
   KAGCUDA_CHECK(cudaMalloc((void**)&dm,jobs*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(*dout)));
   // Run kernel
   KAGCUDA_CHECK(cudaEventRecord(e0));
   kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,lp,cc.dca,cc.dcd,cc.dcases,cc.count,dprograms,(int)batch.programs.size(),dphases,ddecisions,dmotifs,(int)batch.motifs.size(),dactions,feed_slot,feed_count,strategic_route_slot,(int)batch.header.version,dbb,dns,dm,ds,dout);
   KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));
   float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
   std::vector<rlrmb::ResultRecord>hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(*dout),cudaMemcpyDeviceToHost));
   write_results_stdout(hr,(int)batch.programs.size(),cc.count,rid,ms);
   // Free per-batch device memory
   cudaFree(dprograms);cudaFree(dphases);cudaFree(ddecisions);cudaFree(dmotifs);cudaFree(dactions);
   if(dbb)cudaFree(dbb);if(dns)cudaFree(dns);cudaFree(dm);cudaFree(ds);cudaFree(dout);
  }
  cudaEventDestroy(e0);cudaEventDestroy(e1);
  return 0;
 }

 // One-shot mode (existing behaviour, unchanged).
 auto batch=rlrmb::load_batch(programsp);
 auto hc=load_cases(casesp);std::vector<Action>hca;std::vector<TapeDesc>hcd;std::vector<CaseDev>hcases;for(const auto&c:hc){auto t=load_native_tape(c.path);int off=(int)hca.size();hca.insert(hca.end(),t.actions.begin(),t.actions.end());hcd.push_back({off,(int)t.actions.size()});hcases.push_back({(int)hcd.size()-1,c.seat,c.seed});}
 auto*dca=cp(hca);auto*dcd=cp(hcd);auto*dcases=cp(hcases);auto*dprograms=cp(batch.programs);auto*dphases=cp(batch.phases);auto*ddecisions=cp(batch.decisions);auto*dmotifs=cp(batch.motifs);auto*dactions=cp(batch.actions);size_t jobs=(size_t)batch.programs.size()*hc.size();int32_t*dbb=nullptr;bonsaikg4::NodeState4*dns=nullptr;bonsaikg4::RuntimeMeta4*dm=nullptr;Sim*ds=nullptr;rlrmb::ResultRecord*dout=nullptr;if(hp.n_bb)KAGCUDA_CHECK(cudaMalloc((void**)&dbb,jobs*hp.n_bb*sizeof(int32_t)));if(hp.n_state)KAGCUDA_CHECK(cudaMalloc((void**)&dns,jobs*hp.n_state*sizeof(*dns)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,jobs*sizeof(*dm)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(*ds)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(*dout)));
 cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dp,bp,lp,dca,dcd,dcases,(int)hc.size(),dprograms,(int)batch.programs.size(),dphases,ddecisions,dmotifs,(int)batch.motifs.size(),dactions,feed_slot,feed_count,strategic_route_slot,(int)batch.header.version,dbb,dns,dm,ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<rlrmb::ResultRecord>hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(*dout),cudaMemcpyDeviceToHost));write_results(outp,hr,(int)batch.programs.size(),(int)hc.size());int bad=0;for(auto&r:hr)bad+=r.fault!=0;std::cout<<"candidates="<<batch.programs.size()<<" cases="<<hc.size()<<" games="<<jobs<<" launches=1 gpu_ms="<<ms<<" bad="<<bad<<"\n";return bad?1:0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
