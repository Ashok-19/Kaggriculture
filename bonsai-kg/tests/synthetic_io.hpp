#pragma once

#include "../include/bonsai_kg_abi.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg_test {

constexpr int MAX_SYN_ENTRIES = 128;
struct SynEntry { uint16_t feature = 0; uint16_t pad = 0; int32_t arg = 0; int32_t value = 0; };
static_assert(sizeof(SynEntry) == 12, "SynEntry size drift");
struct SynFrame {
    uint16_t n = 0;
    uint16_t pad = 0;
    SynEntry entries[MAX_SYN_ENTRIES];
};

struct SyntheticObservation {
    const SynFrame* frame = nullptr;
#ifdef __CUDACC__
    __host__ __device__ __forceinline__
#else
    inline
#endif
    int32_t feature(uint16_t id, int32_t arg) const {
        if (id == bonsaikg::F_ALWAYS) return 1;
        if (!frame) return 0;
        for (uint16_t i=0;i<frame->n;++i) if (frame->entries[i].feature==id && frame->entries[i].arg==arg) return frame->entries[i].value;
        if (arg != 0) for (uint16_t i=0;i<frame->n;++i) if (frame->entries[i].feature==id && frame->entries[i].arg==0) return frame->entries[i].value;
        return 0;
    }
};

inline std::vector<SynFrame> load_sequence(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open synthetic sequence: " + path);
    std::string magic; int version=0, ticks=0; in >> magic >> version >> ticks;
    if (!in || magic!="BKS1" || version!=1 || ticks<0 || ticks>4096) throw std::runtime_error("invalid synthetic sequence header");
    std::vector<SynFrame> out(static_cast<size_t>(ticks));
    for (int t=0;t<ticks;++t) {
        int n=0; in >> n; if (!in || n<0 || n>MAX_SYN_ENTRIES) throw std::runtime_error("invalid synthetic frame count");
        out[t].n=static_cast<uint16_t>(n);
        for (int i=0;i<n;++i) {
            int fid=0; int32_t arg=0,val=0; in >> fid >> arg >> val;
            if (!in || fid<0 || fid>65535) throw std::runtime_error("invalid synthetic entry");
            out[t].entries[i].feature=static_cast<uint16_t>(fid); out[t].entries[i].arg=arg; out[t].entries[i].value=val;
        }
    }
    std::string extra; if (in >> extra) throw std::runtime_error("trailing synthetic sequence tokens");
    return out;
}

inline std::vector<std::pair<std::string,std::string>> load_manifest(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open manifest: "+path);
    std::vector<std::pair<std::string,std::string>> out; std::string p,s;
    while(in>>p>>s) out.push_back({p,s});
    if(out.empty()) throw std::runtime_error("empty differential manifest");
    return out;
}

inline void print_snapshot(std::ostream& os, int case_id, int tick_id, bonsaikg::Status status,
                           const bonsaikg::RuntimeState& st, const bonsaikg::ActionResult& a,
                           uint16_t n_bb, uint16_t n_state) {
    os << "C="<<case_id<<" T="<<tick_id<<" S="<<static_cast<int>(status)<<" F="<<st.fault;
    os << " U="<<a.n_units<<":";
    for(int i=0;i<a.n_units;++i){ if(i)os<<';'; os<<static_cast<int>(a.units[i].op)<<','<<static_cast<int>(a.units[i].arg)<<','<<a.units[i].n; }
    os << " M="<<a.n_market<<":";
    for(int i=0;i<a.n_market;++i){ if(i)os<<';'; os<<static_cast<int>(a.market[i].op)<<','<<static_cast<int>(a.market[i].item)<<','<<a.market[i].n; }
    os << " B="; for(uint16_t i=0;i<n_bb;++i){if(i)os<<',';os<<st.blackboard[i];}
    os << " N="; for(uint16_t i=0;i<n_state;++i){if(i)os<<',';os<<st.node_state[i].a<<'/'<<st.node_state[i].b;}
    os << '\n';
}

} // namespace bonsaikg_test
