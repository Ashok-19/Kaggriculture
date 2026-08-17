#include "../integrations/stateful_v6/cpp_runtime.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static void print_obs(int g,uint64_t seed,const kag::Sim&sim,int seat){
    const int opp=1-seat;const auto&own=sim.st.farms[seat];const auto&other=sim.st.farms[opp];
    std::cout<<std::setprecision(17)<<"G="<<g<<" SEED="<<seed<<" T="<<sim.st.step<<" P="<<seat
             <<" OM="<<own.money<<" XM="<<other.money<<" H="<<(own.n_units-1)<<" SH="<<sim.st.n_shops<<":";
    for(int i=0;i<sim.st.n_shops;++i){if(i)std::cout<<',';std::cout<<(int)sim.st.shops[i];}
    std::cout<<" PR=9:";
    for(int i=0;i<kag::N_PRODUCTS;++i){if(i)std::cout<<',';std::cout<<sim.st.market.prices[i];}
    std::cout<<'\n';
}

int main(int argc,char**argv){
    try{
        if(argc<3||argc>7)throw std::runtime_error("usage: stateful_v6_observation_cpp CONTROLLER ROUTE_PACK [SEED_START] [COUNT] [SEAT] [STEPS]");
        const uint64_t seed0=argc>=4?std::strtoull(argv[3],nullptr,10):990000ull;const int count=argc>=5?std::atoi(argv[4]):4,seat=argc>=6?std::atoi(argv[5]):0,steps=argc>=7?std::atoi(argv[6]):720;
        if(count<=0||seat<0||seat>1||steps<2) throw std::runtime_error("invalid observation-runner argument");
        bonsaikg_v6::CppCompatAgent agent(argv[1],argv[2]);
        for(int g=0;g<count;++g){const uint64_t seed=seed0+(uint64_t)g;kag::Config cfg;cfg.seed=seed;cfg.episode_steps=steps;kag::Sim sim(cfg);bonsaikg_v6::CppCompatState state{};agent.reset(state);
            while(!sim.st.done){print_obs(g,seed,sim,seat);kag::Action ca{};agent.act(state,sim,seat,ca);const int opp=1-seat;kag::Action oa{};oa.n_units=sim.st.farms[opp].n_units;if(seat==0)sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);else sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);}
        }
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
