#include "../include/bonsai_kg_io.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: validate_compiled POLICY\n";return 2;}
    try{
        auto p=bonsaikg::load_compiled(argv[1]);
        std::cout<<p.header.n_nodes<<" "<<p.header.n_state<<" ";
        for(uint8_t b:p.header.hash)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(b);
        std::cout<<"\n";
        return 0;
    }catch(const std::exception&e){
        std::cerr<<e.what()<<"\n";
        return 1;
    }
}
