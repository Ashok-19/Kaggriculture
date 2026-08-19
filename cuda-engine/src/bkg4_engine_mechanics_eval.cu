#include "../../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh"
#include "../../bonsai-kg/integrations/bkg4_champion/assets/identity.hpp"
#include "../include/runtime.cuh"
#include "../include/tape_io.hpp"
#include "../../bonsai-kg/include/bonsai_kg_io.hpp"

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
struct MechanicGenomeBin { uint8_t mode,animal_mask,replacement,flags; uint16_t wheat_cap,reserved; };
struct GenomeFileHeader { char magic[8]; uint32_t version,count; };
struct ResultFileHeader { char magic[8]; uint32_t version,genomes,cases; };
struct ResultBin { int32_t own_money,opp_money; uint16_t rewrites,flags; };
#pragma pack(pop)
static_assert(sizeof(MechanicGenomeBin)==8,"mechanic genome ABI");
static_assert(sizeof(ResultBin)==12,"result ABI");

constexpr uint8_t MODE_BASE=0,MODE_FEED_REWRITE=1;
constexpr uint8_t REPL_PASS=0,REPL_HARVEST=1,REPL_COLLECT=2,REPL_BEST=3;
constexpr uint8_t F_PROTECT_PRODUCTION=1,F_REQUIRE_NOT_CARED=2,F_REQUIRE_PENDING_ZERO=4;

struct CaseHost { std::string path; int seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0; };
struct CaseDev { int tape=0,seat=0; uint64_t seed=0; int family=0,protected_flag=0,own_history=0; };
struct OppDesc { int first=0,count=0; };
struct DevicePolicyOwner{bonsaikg::Node*nodes=nullptr;uint16_t*children=nullptr;bonsaikg::BbDecl*bb=nullptr;bonsaikg::ValueRef*values=nullptr;bonsaikg::FeatureArgRef*feature_args=nullptr;bonsaikg::Predicate*pred=nullptr;bonsaikg::Lifecycle*lc=nullptr;bonsaikg::Effect*effects=nullptr;~DevicePolicyOwner(){if(nodes)cudaFree(nodes);if(children)cudaFree(children);if(bb)cudaFree(bb);if(values)cudaFree(values);if(feature_args)cudaFree(feature_args);if(pred)cudaFree(pred);if(lc)cudaFree(lc);if(effects)cudaFree(effects);}};
template<class T>static T*cp(const std::vector<T>&v){if(v.empty())return nullptr;T*d=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&d,v.size()*sizeof(T)));KAGCUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
static std::vector<std::string>split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;while(true){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
static std::vector<CaseHost>load_cases(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open cases");std::vector<CaseHost>o;std::string l;while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;auto z=split_tab(l);if(z.size()!=6)throw std::runtime_error("case line needs 6 fields");CaseHost c;c.path=z[0];c.seat=std::stoi(z[1]);c.seed=std::stoull(z[2]);c.family=std::stoi(z[3]);c.protected_flag=std::stoi(z[4]);c.own_history=std::stoi(z[5]);o.push_back(c);}if(o.empty())throw std::runtime_error("empty cases");return o;}
static std::vector<MechanicGenomeBin>load_genomes(const std::string&p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open genomes");GenomeFileHeader h{};in.read((char*)&h,sizeof(h));const char m[8]={'B','4','M','X','1','\0','\0','\0'};if(!in||std::memcmp(h.magic,m,8)||h.version!=1||h.count<1||h.count>65536)throw std::runtime_error("bad genome header");std::vector<MechanicGenomeBin>g(h.count);in.read((char*)g.data(),g.size()*sizeof(g[0]));char x;if(!in||in.read(&x,1))throw std::runtime_error("bad genome payload");return g;}
static bonsaikg_v6::RoutePack load_legacy_pack(const std::string&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("legacy pack");std::vector<uint8_t>raw((std::istreambuf_iterator<char>(in)),{});auto full=bonsaikg::sha256_bytes(raw);if(bonsaikg_v6::hash_hex(full)!=bonsaikg4_champion::LEGACY_ROUTE_PACK_SHA256)throw std::runtime_error("legacy hash");bonsaikg_v6::PackHeader h{};std::memcpy(&h,raw.data(),sizeof(h));size_t db=(size_t)h.route_count*sizeof(bonsaikg_v6::RouteDesc);bonsaikg_v6::RoutePack o;o.routes.resize(h.route_count);o.actions.resize(h.action_count);std::memcpy(o.routes.data(),raw.data()+sizeof(h),db);std::memcpy(o.actions.data(),raw.data()+sizeof(h)+db,o.actions.size()*sizeof(o.actions[0]));return o;}

__device__ __forceinline__ int animal_bit(uint8_t what){return what==GOOSE?1:(what==COW?2:(what==SHEEP?4:0));}
__device__ __forceinline__ bool production_at_eod(const Tile&t,int day){if(!t.has_animal)return false;const AnimalDef a=D_ANIMALS[t.what-GOOSE];const int since=(day+1)-t.planted_day-a.first_yield_day;return since>=0&&since%a.interval==0;}
__device__ __forceinline__ UnitAction replacement(const MechanicGenomeBin&g,const Tile&t){
    UnitAction u{};u.op=OP_PASS;u.arg=0;u.n=1;
    if(g.replacement==REPL_HARVEST){if(t.yield_units>0)u.op=OP_HARVEST;return u;}
    if(g.replacement==REPL_COLLECT){if(t.fertilizer_available)u.op=OP_COLLECT_FERTILIZER;return u;}
    if(g.replacement==REPL_BEST){if(t.yield_units>0)u.op=OP_HARVEST;else if(t.fertilizer_available)u.op=OP_COLLECT_FERTILIZER;else if(!t.cared_today)u.op=OP_CARE;return u;}
    return u;
}
__device__ __forceinline__ uint16_t apply_mechanic(const MechanicGenomeBin&g,const ObservationView&obs,Action&a){
    if(g.mode==MODE_BASE||g.mode!=MODE_FEED_REWRITE)return 0;
    const int me=obs.player(),day=obs.day();const auto f=obs.farm(me);const auto priv=obs.private_farm();
    int wheat=priv.shed(WHEAT);for(int u=0;u<f.n_units();++u)wheat+=priv.inventory(u,WHEAT);
    if(g.wheat_cap&&wheat>(int)g.wheat_cap)return 0;
    uint16_t n=0;const int nu=f.n_units();
    for(int u=0;u<nu&&u<a.n_units;++u){
        if(a.units[u].op!=OP_FEED)continue;const int x=f.worker_x(u),y=f.worker_y(u);if(x<0||x>=BOARD||y<0||y>=BOARD)continue;const Tile&t=f.tile(x,y);
        if(!t.has_animal||!(g.animal_mask&animal_bit(t.what)))continue;
        if(t.fed_today||t.consecutive_dry!=0)continue;
        if((g.flags&F_PROTECT_PRODUCTION)&&production_at_eod(t,day))continue;
        if((g.flags&F_REQUIRE_NOT_CARED)&&t.cared_today)continue;
        if((g.flags&F_REQUIRE_PENDING_ZERO)&&t.pending_care_bonus!=0)continue;
        a.units[u]=replacement(g,t);if(n<65535)++n;
    }
    return n;
}

__global__ void eval_kernel(Config cfg0,DeviceTables tables,bonsaikg4_champion::CudaChampionPolicy champion,const Action*opp,const OppDesc*od,const CaseDev*cases,int nc,const MechanicGenomeBin*gen,int ng,Sim*sims,ResultBin*out){int job=blockIdx.x*blockDim.x+threadIdx.x,total=nc*ng;if(job>=total)return;int gi=job/nc,ci=job-gi*nc;const CaseDev c=cases[ci];Config cfg=cfg0;cfg.seed=c.seed;Sim&sim=sims[job];sim.init(cfg,&tables);bonsaikg4_champion::CudaChampionState st{};champion.reset(st,make_observation(sim,c.seat));uint16_t rew=0;while(!sim.st.done){int step=sim.st.step,os=1-c.seat;ObservationView ob=make_observation(sim,c.seat);Action ca{};champion.act(st,ob,0xffffu,ca);uint16_t q=apply_mechanic(gen[gi],ob,ca);rew=(uint16_t)imin(65535,(int)rew+(int)q);Action oa{};OppDesc d=od[c.tape];if(d.count){int s=step<d.count?step:d.count-1;oa=opp[d.first+s];}oa.n_units=sim.st.farms[os].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);}ResultBin r{};r.own_money=(int32_t)sim.st.farms[c.seat].money;r.opp_money=(int32_t)sim.st.farms[1-c.seat].money;r.rewrites=rew;r.flags=(uint16_t)((st.base.controller.fault?1:0)|(st.base.compat_fault?2:0));out[job]=r;}

static void write_results(const std::string&p,const std::vector<ResultBin>&r,int ng,int nc){std::ofstream o(p,std::ios::binary);ResultFileHeader h{{'B','4','M','R','1','\0','\0','\0'},1u,(uint32_t)ng,(uint32_t)nc};o.write((char*)&h,sizeof(h));o.write((char*)r.data(),r.size()*sizeof(r[0]));}
}

int main(int argc,char**argv){try{std::string controller,routes,legacy,casesp,genp,outp;int tpb=64;for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--genomes")genp=need();else if(a=="--out")outp=need();else if(a=="--tpb")tpb=std::stoi(need());else throw std::runtime_error("unknown "+a);}if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||genp.empty()||outp.empty())throw std::runtime_error("required args");auto owned=bonsaikg::load_compiled(controller);auto hp=owned.view();DevicePolicyOwner d;d.nodes=cp(owned.nodes);d.children=cp(owned.child_refs);d.bb=cp(owned.bb);d.values=cp(owned.values);d.feature_args=cp(owned.feature_args);d.pred=cp(owned.predicates);d.lc=cp(owned.lifecycles);d.effects=cp(owned.effects);bonsaikg::PolicyView dp=hp;dp.nodes=d.nodes;dp.child_refs=d.children;dp.bb=d.bb;dp.values=d.values;dp.feature_args=d.feature_args;dp.predicates=d.pred;dp.lifecycles=d.lc;dp.effects=d.effects;auto rp=bonsaikg_v6::RoutePack::load(routes),lp=load_legacy_pack(legacy);auto*dr=cp(rp.routes);auto*da=cp(rp.actions);auto*dlr=cp(lp.routes);auto*dla=cp(lp.actions);bonsaikg_v6::CudaCompatPolicy base{dp,dr,da,(int)rp.routes.size()};bonsaikg4_champion::CudaChampionPolicy champion{base,{dlr,dla,(int)lp.routes.size()}};auto hc=load_cases(casesp);std::vector<Action>hoa;std::vector<OppDesc>hod;std::vector<CaseDev>hcd;for(const auto&c:hc){NativeTape t=load_native_tape_json(c.path);int off=hoa.size();hoa.insert(hoa.end(),t.actions.begin(),t.actions.end());hod.push_back({off,(int)t.actions.size()});hcd.push_back({(int)hod.size()-1,c.seat,c.seed,c.family,c.protected_flag,c.own_history});}auto hg=load_genomes(genp);Action*doa=cp(hoa);OppDesc*dod=cp(hod);CaseDev*dc=cp(hcd);MechanicGenomeBin*dg=cp(hg);size_t jobs=(size_t)hc.size()*hg.size();Sim*ds=nullptr;ResultBin*dout=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&ds,jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,jobs*sizeof(ResultBin)));configure_device();TablesOwner tables;Config cfg;cfg.episode_steps=720;cudaEvent_t e0,e1;KAGCUDA_CHECK(cudaEventCreate(&e0));KAGCUDA_CHECK(cudaEventCreate(&e1));KAGCUDA_CHECK(cudaEventRecord(e0));eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,champion,doa,dod,dc,(int)hc.size(),dg,(int)hg.size(),ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaEventRecord(e1));KAGCUDA_CHECK(cudaEventSynchronize(e1));float ms=0;KAGCUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));std::vector<ResultBin>hr(jobs);KAGCUDA_CHECK(cudaMemcpy(hr.data(),dout,jobs*sizeof(ResultBin),cudaMemcpyDeviceToHost));write_results(outp,hr,(int)hg.size(),(int)hc.size());std::cout<<"genomes="<<hg.size()<<" cases="<<hc.size()<<" games="<<jobs<<" gpu_ms="<<ms<<" games_s="<<(jobs/(ms/1000.0))<<" sim_MiB="<<(jobs*sizeof(Sim)/(1024.0*1024.0))<<"\n";cudaFree(dr);cudaFree(da);cudaFree(dlr);cudaFree(dla);cudaFree(doa);cudaFree(dod);cudaFree(dc);cudaFree(dg);cudaFree(ds);cudaFree(dout);cudaEventDestroy(e0);cudaEventDestroy(e1);return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
