#include "../include/bonsai_kg_v4_io.hpp"
#include <iostream>
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: v4_validate_compiled POLICY");auto p=bonsaikg4::load_compiled4(argv[1]);std::cout<<"OK nodes="<<p.nodes.size()<<" state="<<p.header.n_state<<" depth="<<p.header.max_depth<<" live="<<p.header.max_live_intents<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
