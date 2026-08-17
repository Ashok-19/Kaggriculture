#include "../cpp/kag_adapter.hpp"
#include "feature_hash.hpp"
#include <cstdlib>
#include <iostream>

static kag::Action scripted(const kag::Sim& sim){
    kag::Action a{};a.n_units=sim.st.farms[0].n_units;const int s=sim.st.step;
    if(s==0){a.n_orders=3;a.orders[0]={kag::M_BUY_SEED,kag::WHEAT,3};a.orders[1]={kag::M_BUY_PRODUCT,kag::WHEAT,2};a.orders[2]={kag::M_HIRE,0,1};}
    else if(s==1){a.units[0]={kag::OP_PICKUP,kag::WHEAT,1};}
    else if(s==2){a.units[0]={kag::OP_PLANT,kag::WHEAT,1};}
    else if(s==3){a.units[0]={kag::OP_WATER,0,1};}
    else if(s==4){a.n_orders=1;a.orders[0]={kag::M_BUY_ANIMAL,kag::GOOSE,1};}
    else if(s==5){a.units[0]={kag::OP_PICKUP,kag::GOOSE,1};}
    else if(s==6){a.units[0]={kag::OP_WEST,0,1};}
    else if(s==7){a.units[0]={kag::OP_BUILD_COOP,0,1};}
    else if(s==8){a.units[0]={kag::OP_PLACE,kag::GOOSE,1};}
    else if(s==9){a.units[0]={kag::OP_CARE,0,1};}
    return a;
}

int main(int argc,char**argv){
    const uint64_t seed=argc>1?std::strtoull(argv[1],nullptr,10):12345ull;const int steps=argc>2?std::atoi(argv[2]):80;
    kag::Config cfg;cfg.seed=seed;cfg.episode_steps=steps;kag::Sim sim(cfg);int t=0;
    while(!sim.st.done){bonsaikg::CppObservation obs{&sim,0};std::cout<<"T="<<t<<" H="<<bonsaikg_test::hash_feature_registry(obs)<<"\n";kag::Action a0=scripted(sim),a1{};a1.n_units=sim.st.farms[1].n_units;sim.step(a0,a1);++t;}
    return 0;
}
