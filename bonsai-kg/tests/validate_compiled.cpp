#include "../include/bonsai_kg_io.hpp"
#include <iostream>

int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: validate_compiled POLICY\n";return 2;}
    try{
        auto p=bonsaikg::load_compiled(argv[1]);
        std::cout<<p.header.n_nodes<<" "<<p.header.n_state<<"\n";
        return 0;
    }catch(const std::exception&e){
        std::cerr<<e.what()<<"\n";
        return 1;
    }
}
