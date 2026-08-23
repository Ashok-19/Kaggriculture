#include "../integrations/bkg4_champion/cpp_compat.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#pragma pack(push,1)
struct TapeHeader { char magic[8]; uint32_t version; int32_t source_seat; uint64_t source_seed; uint32_t action_count; };
struct TapeActionHeader { uint8_t n_units; uint8_t n_orders; uint16_t reserved; };
struct TapeUnit { uint8_t op; uint8_t arg; int16_t n; };
struct TapeOrder { uint8_t op; uint8_t item; int16_t reserved; int32_t n; };
#pragma pack(pop)

struct Tape { std::vector<kag::Action> actions; };
struct Case { std::string path; int seat=0; uint64_t seed=0; };

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out; size_t pos=0;
    for (;;) { size_t next=s.find(sep,pos); out.push_back(s.substr(pos,next==std::string::npos?next:next-pos)); if(next==std::string::npos) return out; pos=next+1; }
}

std::vector<Case> load_cases(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open cases");
    std::vector<Case> out; std::string line;
    while(std::getline(in,line)) { if(line.empty()||line[0]=='#') continue; auto fields=split(line,'\t'); if(fields.size()!=7) throw std::runtime_error("case needs 7 fields"); out.push_back({fields[0],std::stoi(fields[1]),std::stoull(fields[2])}); }
    if(out.empty()) throw std::runtime_error("empty cases"); return out;
}

Tape load_tape(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("cannot open tape");
    TapeHeader h{}; in.read(reinterpret_cast<char*>(&h),sizeof(h)); const char magic[8]={'B','4','T','P','1','\0','\0','\0'};
    if(!in||std::memcmp(h.magic,magic,8)||h.version!=1||h.action_count<1||h.action_count>720) throw std::runtime_error("invalid tape header");
    Tape tape; tape.actions.resize(h.action_count);
    for(uint32_t ai=0;ai<h.action_count;++ai) {
        TapeActionHeader ah{}; in.read(reinterpret_cast<char*>(&ah),sizeof(ah));
        if(!in||ah.reserved||ah.n_units<1||ah.n_units>kag::MAX_UNITS||ah.n_orders>16) throw std::runtime_error("invalid tape action");
        auto& action=tape.actions[ai]; action.n_units=ah.n_units; action.n_orders=ah.n_orders;
        for(int ui=0;ui<action.n_units;++ui) { TapeUnit unit{}; in.read(reinterpret_cast<char*>(&unit),sizeof(unit)); action.units[ui].op=unit.op; action.units[ui].arg=unit.arg; action.units[ui].n=unit.n; }
        for(int oi=0;oi<action.n_orders;++oi) { TapeOrder order{}; in.read(reinterpret_cast<char*>(&order),sizeof(order)); if(order.reserved) throw std::runtime_error("invalid tape order"); action.orders[oi].op=order.op; action.orders[oi].item=order.item; action.orders[oi].n=order.n; }
        if(!in) throw std::runtime_error("truncated tape");
    }
    char extra=0; if(in.read(&extra,1)) throw std::runtime_error("trailing tape bytes"); return tape;
}

bonsaikg4_champion::CppInvariantUnitRewrite4 parse_rewrite(const std::string& value) {
    auto fields=split(value,','); if(fields.size()!=6) throw std::runtime_error("rewrite needs slot,unit,match,replacement,arg,quantity");
    bonsaikg4_champion::CppInvariantUnitRewrite4 row{};
    row.slot=std::stoi(fields[0]); row.unit=std::stoi(fields[1]); row.match_op=static_cast<uint8_t>(std::stoi(fields[2]));
    row.replacement_op=static_cast<uint8_t>(std::stoi(fields[3])); row.arg=static_cast<uint8_t>(std::stoi(fields[4])); row.quantity=static_cast<int16_t>(std::stoi(fields[5]));
    return row;
}
}

int main(int argc,char** argv) try {
    std::string controller,routes,legacy,cases_path; int feed_slot=5,feed_count=13,strategic_slot=19;
    std::vector<bonsaikg4_champion::CppInvariantUnitRewrite4> rewrites;
    for(int i=1;i<argc;++i) { std::string arg=argv[i]; auto need=[&](){ if(i+1>=argc) throw std::runtime_error("missing "+arg); return std::string(argv[++i]); };
        if(arg=="--controller") controller=need(); else if(arg=="--routes") routes=need(); else if(arg=="--legacy-routes") legacy=need(); else if(arg=="--cases") cases_path=need();
        else if(arg=="--feed-slot") feed_slot=std::stoi(need()); else if(arg=="--feed-count") feed_count=std::stoi(need()); else if(arg=="--strategic-route-slot") strategic_slot=std::stoi(need());
        else if(arg=="--rewrite") rewrites.push_back(parse_rewrite(need())); else throw std::runtime_error("unknown "+arg);
    }
    if(controller.empty()||routes.empty()||legacy.empty()||cases_path.empty()) throw std::runtime_error("missing required path");
    bonsaikg4_champion::CppChampionAgent4 agent(controller,routes,legacy,feed_slot,feed_count,strategic_slot,std::move(rewrites));
    auto cases=load_cases(cases_path);
    for(size_t ci=0;ci<cases.size();++ci) { const auto& c=cases[ci]; auto tape=load_tape(c.path); kag::Config cfg; cfg.seed=c.seed; cfg.episode_steps=720; kag::Sim sim(cfg);
        bonsaikg4_champion::CppChampionState4 state{}; agent.reset(state); uint64_t trace=1469598103934665603ull;
        while(!sim.st.done) { kag::Action own{}; agent.act(state,sim,c.seat,own); trace=bonsaikg4_champion::cpp_action_trace_hash(trace,sim.st.step,own); int opponent=1-c.seat; kag::Action other{};
            if(!tape.actions.empty()) other=tape.actions[static_cast<size_t>(sim.st.step<(int)tape.actions.size()?sim.st.step:(int)tape.actions.size()-1)]; other.n_units=sim.st.farms[opponent].n_units;
            if(c.seat==0) sim.step_with_unit_counts(own,sim.st.farms[0].n_units,other,sim.st.farms[1].n_units); else sim.step_with_unit_counts(other,sim.st.farms[0].n_units,own,sim.st.farms[1].n_units);
        }
        std::cout<<ci<<'\t'<<(int32_t)sim.st.farms[c.seat].money<<'\t'<<(int32_t)sim.st.farms[1-c.seat].money<<'\t'<<state.meta.fault<<'\t'<<trace<<'\n';
    }
    return 0;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 2; }
