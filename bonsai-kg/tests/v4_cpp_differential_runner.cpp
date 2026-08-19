#include "../include/bonsai_kg_v4.hpp"
#include "../include/bonsai_kg_v4_io.hpp"
#include "v4_synthetic_io.hpp"
#include <iostream>
#include <vector>
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: v4_cpp_differential_runner MANIFEST");auto cases=bonsaikg_test::load_manifest(argv[1]);for(size_t ci=0;ci<cases.size();++ci){auto owned=bonsaikg4::load_compiled4(cases[ci].first);auto p=owned.view();auto frames=bonsaikg_test::load_sequence(cases[ci].second);std::vector<int32_t>bb(p.n_bb);std::vector<bonsaikg4::NodeState4>ns(p.n_state);bonsaikg4::RuntimeMeta4 meta{};bonsaikg4::RuntimeView4 st{bb.data(),ns.data(),&meta};bonsaikg4::reset_runtime(p,st);for(size_t ti=0;ti<frames.size();++ti){bonsaikg_test::SyntheticObservation obs{&frames[ti]};bonsaikg4::ActionResult4 a{};auto r=bonsaikg4::tick4(p,st,obs,10,a,nullptr);bonsaikg4_test::print_snapshot(std::cout,(int)ci,(int)ti,r,st,a,p.n_bb,p.n_state);}}return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
