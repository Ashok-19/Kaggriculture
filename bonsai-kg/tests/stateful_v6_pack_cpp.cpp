#include "../integrations/stateful_v6/cpp_runtime.hpp"
#include <iostream>
#include <stdexcept>

static void print_action(size_t idx,uint16_t rid,int step,const kag::Action&a){
    std::cout<<"I="<<idx<<" R="<<rid<<" S="<<step<<" U="<<a.n_units<<":";
    for(int i=0;i<a.n_units;++i){if(i)std::cout<<';';std::cout<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}
    std::cout<<" M="<<a.n_orders<<":";
    for(int i=0;i<a.n_orders;++i){if(i)std::cout<<';';std::cout<<(int)a.orders[i].op<<','<<(int)a.orders[i].item<<','<<a.orders[i].n;}
    std::cout<<'\n';
}

int main(int argc,char**argv){
    try{
        if(argc!=2)throw std::runtime_error("usage: stateful_v6_pack_cpp ROUTE_PACK");
        auto p=bonsaikg_v6::RoutePack::load(argv[1]);size_t idx=0;
        for(const auto&d:p.routes)for(int s=0;s<d.steps;++s,++idx){
            const auto&src=p.actions[d.first+s];kag::Action out{};
            bonsaikg_v6::materialize_cpp(src,1+src.saved_hands,out);print_action(idx,d.route_id,s,out);
        }
        std::cerr<<"ROUTES="<<p.routes.size()<<" ACTIONS="<<idx<<"\n";return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
