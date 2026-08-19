#pragma once
#include "synthetic_io.hpp"
#include "../include/bonsai_kg_v4_abi.hpp"
#include <ostream>
namespace bonsaikg4_test {
inline void print_snapshot(std::ostream&os,int case_id,int tick_id,bonsaikg4::Status status,bonsaikg4::RuntimeView4 st,const bonsaikg4::ActionResult4&a,uint16_t n_bb,uint16_t n_state){
    os<<"C="<<case_id<<" T="<<tick_id<<" S="<<(int)status<<" F="<<(st.meta?st.meta->fault:9999);os<<" U="<<a.n_units<<":";for(int i=0;i<a.n_units;++i){if(i)os<<';';os<<(int)a.units[i].op<<','<<(int)a.units[i].arg<<','<<a.units[i].n;}os<<" M="<<a.n_market<<":";for(int i=0;i<a.n_market;++i){if(i)os<<';';os<<(int)a.market[i].op<<','<<(int)a.market[i].item<<','<<a.market[i].n;}os<<" B=";for(uint16_t i=0;i<n_bb;++i){if(i)os<<',';os<<st.blackboard[i];}os<<" N=";for(uint16_t i=0;i<n_state;++i){if(i)os<<',';os<<st.node_state[i].a<<'/'<<st.node_state[i].b;}os<<'\n';
}
}
