#include "../include/bonsai_kg.hpp"
#include "../include/bonsai_kg_io.hpp"
#include "../cpp/kag_adapter.hpp"
#include "synthetic_io.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc,char**argv){
    try{
        if(argc<2||argc>4)throw std::runtime_error("usage: cpp_game_runner POLICY [SEED] [EPISODE_STEPS]");
        const uint64_t seed=argc>=3?std::strtoull(argv[2],nullptr,10):12345ull;const int steps=argc>=4?std::atoi(argv[3]):16;
        auto owned=bonsaikg::load_compiled(argv[1]);auto p=owned.view();kag::Config cfg;cfg.seed=seed;cfg.episode_steps=steps;kag::Sim sim(cfg);
        bonsaikg::RuntimeState st{};bonsaikg::reset_runtime(p,st);int ti=0;
        while(!sim.st.done){
            bonsaikg::CppObservation obs{&sim,0};bonsaikg::ActionResult ar{};auto status=bonsaikg::tick(p,st,obs,cfg.max_orders,ar,nullptr);
            bonsaikg_test::print_snapshot(std::cout,0,ti,status,st,ar,p.n_bb,p.n_state);
            kag::Action a0{};bonsaikg::to_cpp_action(ar,a0);kag::Action a1{};a1.n_units=sim.st.farms[1].n_units;sim.step(a0,a1);++ti;
        }
        std::cerr<<"FINAL ticks="<<ti<<" reward0="<<sim.reward(0)<<" reward1="<<sim.reward(1)<<"\n";return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
