#include "../include/bonsai_kg.hpp"
#include "../include/bonsai_kg_io.hpp"
#include "synthetic_io.hpp"
#include <iostream>

int main(int argc,char**argv){
    try{
        if(argc!=2) throw std::runtime_error("usage: cpp_differential_runner MANIFEST");
        auto cases=bonsaikg_test::load_manifest(argv[1]);
        for(size_t ci=0;ci<cases.size();++ci){
            auto owned=bonsaikg::load_compiled(cases[ci].first); auto p=owned.view();
            auto frames=bonsaikg_test::load_sequence(cases[ci].second);
            bonsaikg::RuntimeState st{}; bonsaikg::reset_runtime(p,st);
            for(size_t ti=0;ti<frames.size();++ti){
                bonsaikg_test::SyntheticObservation obs{&frames[ti]}; bonsaikg::ActionResult a{};
                auto status=bonsaikg::tick(p,st,obs,10,a,nullptr);
                bonsaikg_test::print_snapshot(std::cout,static_cast<int>(ci),static_cast<int>(ti),status,st,a,p.n_bb,p.n_state);
            }
        }
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
