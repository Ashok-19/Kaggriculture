#include "../integrations/stateful_v6/cpp_runtime.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void print_tick(int g,uint64_t seed,int step,const bonsaikg_v6::CppCompatState&s,const kag::Action&a){
    std::cout<<"G="<<g<<" SEED="<<seed<<" T="<<step
             <<" L="<<s.controller.blackboard[0]<<" R="<<s.controller.blackboard[1]
             <<" S1="<<s.controller.blackboard[2]<<" S2="<<s.controller.blackboard[3]
             <<" F="<<s.controller.fault<<" U="<<a.n_units<<":";
    for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}
    std::cout<<" M="<<a.n_orders<<":";
    for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}
    std::cout<<'\n';
}

int main(int argc,char**argv){
    try{
        if(argc<3||argc>7)throw std::runtime_error("usage: stateful_v6_game_cpp CONTROLLER ROUTE_PACK [SEED_START] [COUNT] [SEAT] [STEPS]");
        const uint64_t seed0=argc>=4?std::strtoull(argv[3],nullptr,10):990000ull;
        const int count=argc>=5?std::atoi(argv[4]):4,seat=argc>=6?std::atoi(argv[5]):0,steps=argc>=7?std::atoi(argv[6]):720;
        if(count<=0||seat<0||seat>1||steps<2)throw std::runtime_error("invalid game-runner argument");
        bonsaikg_v6::CppCompatAgent agent(argv[1],argv[2]);
        for(int g=0;g<count;++g){
            const uint64_t seed=seed0+(uint64_t)g;kag::Config cfg;cfg.seed=seed;cfg.episode_steps=steps;kag::Sim sim(cfg);bonsaikg_v6::CppCompatState state{};agent.reset(state);
            while(!sim.st.done){
                kag::Action ca{};agent.act(state,sim,seat,ca);print_tick(g,seed,sim.st.step,state,ca);
                const int opp=1-seat;kag::Action oa{};oa.n_units=sim.st.farms[opp].n_units;
                if(seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);
                else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);
            }
            std::cerr<<"GAME="<<g<<" SEED="<<seed<<" SEAT="<<seat<<" TICKS="<<sim.st.step<<"\n";
        }
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
