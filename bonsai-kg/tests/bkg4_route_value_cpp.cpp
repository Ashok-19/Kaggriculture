#include "../integrations/bkg4_champion/extended_route_pack.hpp"
#include "../integrations/stateful_v6/cpp_runtime.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc,char**argv){
    try{
        if(argc!=6)throw std::runtime_error("usage: bkg4_route_value_cpp PACK CONTROLLER_ROUTE OVERRIDE STEP N_UNITS");
        auto pack=bonsaikg4_champion::load_extended_route_pack(argv[1]);
        const uint16_t base=(uint16_t)std::strtoul(argv[2],nullptr,10);
        const int override_v=std::atoi(argv[3]),step=std::atoi(argv[4]),n=std::atoi(argv[5]);
        const uint16_t route=override_v?static_cast<uint16_t>(override_v):base;
        kag::Action a{};bonsaikg_v6::materialize_cpp(pack.at(route,step),n,a);
        std::cout<<route<<" U="<<a.n_units<<":";
        for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}
        std::cout<<" M="<<a.n_orders<<":";
        for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}
        std::cout<<'\n';return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
