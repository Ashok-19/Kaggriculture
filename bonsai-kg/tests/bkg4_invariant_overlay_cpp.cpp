#include "../integrations/bkg4_champion/cpp_compat.hpp"
#include "../include/bkg4_invariant_module.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inv = bonsaikg4::invariant;

namespace {
#pragma pack(push,1)
struct TapeHeader { char magic[8]; uint32_t version; int32_t source_seat; uint64_t source_seed; uint32_t action_count; };
struct TapeActionHeader { uint8_t n_units; uint8_t n_orders; uint16_t reserved; };
struct TapeUnit { uint8_t op; uint8_t arg; int16_t n; };
struct TapeOrder { uint8_t op; uint8_t item; int16_t reserved; int32_t n; };
#pragma pack(pop)

struct Tape { std::vector<kag::Action> actions; };
struct Case { std::string path; int seat=0; uint64_t seed=0; };

std::vector<std::string> split_tab(const std::string&s){std::vector<std::string>o;size_t p=0;for(;;){size_t q=s.find('\t',p);o.push_back(s.substr(p,q==std::string::npos?std::string::npos:q-p));if(q==std::string::npos)break;p=q+1;}return o;}
std::vector<Case> load_cases(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open cases");std::vector<Case>o;std::string line;while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;auto z=split_tab(line);if(z.size()!=7)throw std::runtime_error("case needs 7 fields");Case c;c.path=z[0];c.seat=std::stoi(z[1]);c.seed=std::stoull(z[2]);if(c.seat<0||c.seat>1)throw std::runtime_error("invalid seat");o.push_back(c);}if(o.empty())throw std::runtime_error("empty cases");return o;}
Tape load_tape(const std::string&p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open B4TP1 tape");TapeHeader h{};in.read((char*)&h,sizeof(h));const char magic[8]={'B','4','T','P','1','\0','\0','\0'};if(!in||std::memcmp(h.magic,magic,8)||h.version!=1||h.action_count<1||h.action_count>720)throw std::runtime_error("invalid B4TP1 header");Tape t;t.actions.resize(h.action_count);for(uint32_t ai=0;ai<h.action_count;++ai){TapeActionHeader ah{};in.read((char*)&ah,sizeof(ah));if(!in||ah.reserved||ah.n_units<1||ah.n_units>kag::MAX_UNITS||ah.n_orders>16)throw std::runtime_error("invalid tape action");auto&a=t.actions[ai];a.n_units=ah.n_units;a.n_orders=ah.n_orders;for(int ui=0;ui<a.n_units;++ui){TapeUnit u{};in.read((char*)&u,sizeof(u));if(!in)throw std::runtime_error("truncated tape unit");a.units[ui].op=u.op;a.units[ui].arg=u.arg;a.units[ui].n=u.n;}for(int oi=0;oi<a.n_orders;++oi){TapeOrder r{};in.read((char*)&r,sizeof(r));if(!in||r.reserved)throw std::runtime_error("invalid tape order");a.orders[oi].op=r.op;a.orders[oi].item=r.item;a.orders[oi].n=r.n;}}char extra=0;if(in.read(&extra,1))throw std::runtime_error("trailing tape bytes");return t;}
std::vector<inv::Candidate> load_candidates(const std::string&p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open B4IM1 candidates");inv::FileHeader h{};in.read((char*)&h,sizeof(h));const char magic[8]={'B','4','I','M','1','\0','\0','\0'};if(!in||std::memcmp(h.magic,magic,8)||h.version!=1||h.count<1||h.count>100000)throw std::runtime_error("invalid B4IM1 header");std::vector<inv::Candidate>o(h.count);in.read((char*)o.data(),o.size()*sizeof(o[0]));char extra=0;if(!in||in.read(&extra,1))throw std::runtime_error("invalid B4IM1 payload");for(auto&c:o)if(!inv::valid_candidate(c))throw std::runtime_error("invalid B4IM1 candidate");return o;}
}

int main(int argc,char**argv){try{
    std::string controller,routes,legacy,casesp,candsp;int feed_slot=5,feed_count=13,strategic_slot=19;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing "+a);return std::string(argv[++i]);};if(a=="--controller")controller=need();else if(a=="--routes")routes=need();else if(a=="--legacy-routes")legacy=need();else if(a=="--cases")casesp=need();else if(a=="--candidates")candsp=need();else if(a=="--feed-slot")feed_slot=std::stoi(need());else if(a=="--feed-count")feed_count=std::stoi(need());else if(a=="--strategic-route-slot")strategic_slot=std::stoi(need());else throw std::runtime_error("unknown "+a);}if(controller.empty()||routes.empty()||legacy.empty()||casesp.empty()||candsp.empty())throw std::runtime_error("missing required evaluator path");
    bonsaikg4_champion::CppChampionAgent4 agent(controller,routes,legacy,feed_slot,feed_count,strategic_slot);auto cases=load_cases(casesp);auto cands=load_candidates(candsp);
    for(size_t gi=0;gi<cands.size();++gi)for(size_t ci=0;ci<cases.size();++ci){const auto&c=cases[ci];auto tape=load_tape(c.path);kag::Config cfg;cfg.seed=c.seed;cfg.episode_steps=720;kag::Sim sim(cfg);bonsaikg4_champion::CppChampionState4 st{};agent.reset(st);inv::RuntimeState ims{};uint64_t trace=1469598103934665603ull;uint32_t acts=0;
        while(!sim.st.done){bonsaikg4::CppObservation4 obs(&sim,c.seat);auto d=inv::evaluate(cands[gi],ims,obs);acts+=d.activations;kag::Action ca{};agent.act(st,sim,c.seat,ca,d.route_override);inv::apply_post(cands[gi],d,ca,10);trace=bonsaikg4_champion::cpp_action_trace_hash(trace,sim.st.step,ca);kag::Action oa{};int oi=1-c.seat;if(!tape.actions.empty())oa=tape.actions[(size_t)(sim.st.step<(int)tape.actions.size()?sim.st.step:(int)tape.actions.size()-1)];oa.n_units=sim.st.farms[oi].n_units;if(c.seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);}
        std::cout<<gi<<'\t'<<ci<<'\t'<<(int32_t)sim.st.farms[c.seat].money<<'\t'<<(int32_t)sim.st.farms[1-c.seat].money<<'\t'<<st.meta.fault<<'\t'<<trace<<'\t'<<acts<<'\n';
    }return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
