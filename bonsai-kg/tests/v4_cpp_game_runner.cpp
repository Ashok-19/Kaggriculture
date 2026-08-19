#include "../include/bonsai_kg_v4.hpp"
#include "../include/bonsai_kg_v4_io.hpp"
#include "../cpp/kag_adapter_v4.hpp"
#include "v4_synthetic_io.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
int main(int argc,char**argv){try{if(argc<2||argc>5)throw std::runtime_error("usage: v4_cpp_game_runner POLICY [SEED] [STEPS] [SEAT]");const uint64_t seed=argc>=3?std::strtoull(argv[2],nullptr,10):12345ull;const int steps=argc>=4?std::atoi(argv[3]):16;const int seat=argc>=5?std::atoi(argv[4]):0;if(seat<0||seat>1)throw std::runtime_error("seat must be 0/1");auto owned=bonsaikg4::load_compiled4(argv[1]);auto p=owned.view();kag::Config cfg;cfg.seed=seed;cfg.episode_steps=steps;kag::Sim sim(cfg);std::vector<int32_t>bb(p.n_bb);std::vector<bonsaikg4::NodeState4>ns(p.n_state);bonsaikg4::RuntimeMeta4 meta{};bonsaikg4::RuntimeView4 st{bb.data(),ns.data(),&meta};bonsaikg4::reset_runtime(p,st);int ti=0;while(!sim.st.done){bonsaikg4::CppObservation4 obs(&sim,seat);bonsaikg4::ActionResult4 ar{};auto status=bonsaikg4::tick4(p,st,obs,cfg.max_orders,ar,nullptr);bonsaikg4_test::print_snapshot(std::cout,0,ti,status,st,ar,p.n_bb,p.n_state);kag::Action a0{},a1{};a0.n_units=sim.st.farms[0].n_units;a1.n_units=sim.st.farms[1].n_units;if(seat==0)bonsaikg4::to_cpp_action4(ar,a0);else bonsaikg4::to_cpp_action4(ar,a1);sim.step(a0,a1);++ti;}std::cerr<<"FINAL ticks="<<ti<<" seat="<<seat<<" reward0="<<sim.reward(0)<<" reward1="<<sim.reward(1)<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
