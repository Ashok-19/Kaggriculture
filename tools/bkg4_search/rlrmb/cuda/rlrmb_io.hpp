#pragma once

#include "rlrmb_ir.cuh"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlrmb {

struct HostBatch {
    FileHeader header{};
    std::vector<ProgramRecord> programs;
    std::vector<PhaseRecord> phases;
    std::vector<MotifRecord> motifs;
    std::vector<ActionRecord> actions;
};

template<class T> inline void read_vector(std::ifstream& in, std::vector<T>& values, uint32_t count) {
    values.resize(count);
    if (count) in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count)*sizeof(T));
    if (!in) throw std::runtime_error("short RLRMB device batch");
}

inline HostBatch load_batch(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open RLRMB device batch: "+path);
    HostBatch out;
    in.read(reinterpret_cast<char*>(&out.header), sizeof(out.header));
    if (!in || out.header.magic!=IR_MAGIC) throw std::runtime_error("invalid RLRMB batch magic");
    if (out.header.version!=IR_VERSION) throw std::runtime_error("unsupported RLRMB batch version");
    read_vector(in,out.programs,out.header.programs);
    read_vector(in,out.phases,out.header.phases);
    read_vector(in,out.motifs,out.header.motifs);
    read_vector(in,out.actions,out.header.actions);
    if (in.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("trailing RLRMB batch bytes");
    for(const auto&p:out.programs){
        if(p.phase_count<1||p.phase_count>MAX_PHASES||p.phase_first+p.phase_count>out.phases.size()) throw std::runtime_error("invalid program phase range");
        if(p.initiation_step>=719||p.max_total_steps<1||p.initiation_step+p.max_total_steps>719) throw std::runtime_error("invalid program horizon");
    }
    for(const auto&m:out.motifs) if(m.action_count<1||m.action_first+m.action_count>out.actions.size()) throw std::runtime_error("invalid motif range");
    return out;
}

} // namespace rlrmb
