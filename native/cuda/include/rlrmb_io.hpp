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
    std::vector<DecisionRecord> decisions;
    std::vector<MotifRecord> motifs;
    std::vector<ActionRecord> actions;
};

template<class T> inline void read_vector(std::ifstream& in, std::vector<T>& values, uint32_t count) {
    values.resize(count);
    if (count) in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count)*sizeof(T));
    if (!in) throw std::runtime_error("short RLRMB device batch");
}

// Memory-buffer reader used by the persistent evaluator's stdin protocol.
template<class T> inline void read_vector_mem(const uint8_t* data, size_t total, size_t& pos,
                                               std::vector<T>& values, uint32_t count) {
    values.resize(count);
    size_t bytes = count * sizeof(T);
    if (pos + bytes > total) throw std::runtime_error("short RLRMB device batch (memory)");
    if (count) std::memcpy(values.data(), data + pos, bytes);
    pos += bytes;
}

inline HostBatch load_batch_from_memory(const std::vector<uint8_t>& raw) {
    const uint8_t* data = raw.data();
    size_t total = raw.size();
    size_t pos = 0;
    HostBatch out;
    if (pos + sizeof(out.header) > total) throw std::runtime_error("short RLRMB device batch (memory)");
    std::memcpy(&out.header, data + pos, sizeof(out.header));
    pos += sizeof(out.header);
    if (out.header.magic != IR_MAGIC) throw std::runtime_error("invalid RLRMB batch magic (memory)");
    if (out.header.version != IR_VERSION_V1 && out.header.version != IR_VERSION_V2 && out.header.version != IR_VERSION_V3)
        throw std::runtime_error("unsupported RLRMB batch version (memory)");
    read_vector_mem(data, total, pos, out.programs, out.header.programs);
    read_vector_mem(data, total, pos, out.phases, out.header.phases);
    uint32_t decision_count = 0;
    if (out.header.version >= IR_VERSION_V2)
        for (const auto& p : out.programs) decision_count += p.reserved;
    read_vector_mem(data, total, pos, out.decisions, decision_count);
    read_vector_mem(data, total, pos, out.motifs, out.header.motifs);
    read_vector_mem(data, total, pos, out.actions, out.header.actions);
    if (pos != total) throw std::runtime_error("trailing RLRMB batch bytes (memory)");
    // Same validation as load_batch
    for (const auto& p : out.programs) {
        if (p.phase_count < 1 || p.phase_count > MAX_PHASES || p.phase_first + p.phase_count > out.phases.size())
            throw std::runtime_error("invalid program phase range");
        uint16_t initiation_step = p.initiation_step & 0x03ffu;
        uint16_t max_total_steps = p.max_total_steps & 0x03ffu;
        uint16_t gate_count = out.header.version == IR_VERSION_V3 ? p.max_total_steps >> 10 : 0;
        uint16_t clause_count = out.header.version == IR_VERSION_V3 ? p.initiation_step >> 10 : 0;
        bool has_recovery = (p.lane & 0x80u) != 0;
        if (initiation_step >= 719 || max_total_steps < 1 || initiation_step + max_total_steps > 719)
            throw std::runtime_error("invalid program horizon");
        uint32_t gate_first = p.phase_first + p.phase_count + (has_recovery ? 1u : 0u);
        if (out.header.version == IR_VERSION_V3 && (clause_count < 1 || clause_count > 6 || gate_count < clause_count || gate_count > 24 || gate_first + gate_count > out.phases.size()))
            throw std::runtime_error("invalid specialist gate range");
        for (uint32_t i = gate_first; i < gate_first + gate_count; ++i)
            if (out.phases[i].feature > MAX_PUBLIC_FEATURE || out.phases[i].comparator > 3 || out.phases[i].reserved >= clause_count)
                throw std::runtime_error("invalid specialist gate condition");
        if ((p.lane & 0x7fu) > 7) throw std::runtime_error("invalid program lane");
        if (out.header.version == IR_VERSION_V1) {
            if (p.fallback || p.reserved || (p.lane & 0x80u)) throw std::runtime_error("reactive fields in v1 program");
            continue;
        }
        if (p.reserved > MAX_DECISIONS || p.fallback + p.reserved > out.decisions.size())
            throw std::runtime_error("invalid decision range");
        bool recovery = false; int aborts = 0;
        for (uint32_t i = p.fallback; i < (uint32_t)p.fallback + p.reserved; ++i) {
            const auto& d = out.decisions[i];
            if (d.after_phase >= p.phase_count || d.comparator > 3) throw std::runtime_error("invalid reactive decision");
            for (int branch = 0; branch < 2; ++branch) {
                uint8_t kind = branch ? d.false_kind : d.true_kind, target = branch ? d.false_phase : d.true_phase;
                if (kind == CONT_PHASE) { if (target <= d.after_phase || target >= p.phase_count) throw std::runtime_error("invalid reactive continuation"); }
                else if (kind == CONT_ABORT) { ++aborts; }
                else if (kind == CONT_RECOVERY) { recovery = true; }
                else throw std::runtime_error("invalid reactive continuation");
            }
        }
        if (aborts > 1) throw std::runtime_error("multiple abort paths");
        if (recovery != has_recovery) throw std::runtime_error("invalid recovery descriptor");
        if (has_recovery && p.phase_first + p.phase_count >= out.phases.size()) throw std::runtime_error("missing recovery phase");
    }
    for (const auto& m : out.motifs)
        if (m.action_count < 1 || m.action_first + m.action_count > out.actions.size())
            throw std::runtime_error("invalid motif range");
    return out;
}

inline HostBatch load_batch(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open RLRMB device batch: "+path);
    HostBatch out;
    in.read(reinterpret_cast<char*>(&out.header), sizeof(out.header));
    if (!in || out.header.magic!=IR_MAGIC) throw std::runtime_error("invalid RLRMB batch magic");
    if (out.header.version!=IR_VERSION_V1&&out.header.version!=IR_VERSION_V2&&out.header.version!=IR_VERSION_V3) throw std::runtime_error("unsupported RLRMB batch version");
    read_vector(in,out.programs,out.header.programs);
    read_vector(in,out.phases,out.header.phases);
    uint32_t decision_count=0;
    if(out.header.version>=IR_VERSION_V2) for(const auto&p:out.programs) decision_count+=p.reserved;
    read_vector(in,out.decisions,decision_count);
    read_vector(in,out.motifs,out.header.motifs);
    read_vector(in,out.actions,out.header.actions);
    if (in.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("trailing RLRMB batch bytes");
    for(const auto&p:out.programs){
        if(p.phase_count<1||p.phase_count>MAX_PHASES||p.phase_first+p.phase_count>out.phases.size()) throw std::runtime_error("invalid program phase range");
        uint16_t initiation_step=p.initiation_step&0x03ffu,max_total_steps=p.max_total_steps&0x03ffu;
        uint16_t gate_count=out.header.version==IR_VERSION_V3?p.max_total_steps>>10:0,clause_count=out.header.version==IR_VERSION_V3?p.initiation_step>>10:0;
        bool has_recovery=(p.lane&0x80u)!=0;uint32_t gate_first=p.phase_first+p.phase_count+(has_recovery?1u:0u);
        if(initiation_step>=719||max_total_steps<1||initiation_step+max_total_steps>719) throw std::runtime_error("invalid program horizon");
        if(out.header.version==IR_VERSION_V3&&(clause_count<1||clause_count>6||gate_count<clause_count||gate_count>24||gate_first+gate_count>out.phases.size()))throw std::runtime_error("invalid specialist gate range");
        for(uint32_t i=gate_first;i<gate_first+gate_count;++i)if(out.phases[i].feature>MAX_PUBLIC_FEATURE||out.phases[i].comparator>3||out.phases[i].reserved>=clause_count)throw std::runtime_error("invalid specialist gate condition");
        if((p.lane&0x7fu)>7)throw std::runtime_error("invalid program lane");
        if(out.header.version==IR_VERSION_V1){if(p.fallback||p.reserved||(p.lane&0x80u))throw std::runtime_error("reactive fields in v1 program");continue;}
        if(p.reserved>MAX_DECISIONS||p.fallback+p.reserved>out.decisions.size())throw std::runtime_error("invalid decision range");
        bool recovery=false;int aborts=0;
        for(uint32_t i=p.fallback;i<(uint32_t)p.fallback+p.reserved;++i){const auto&d=out.decisions[i];
            if(d.after_phase>=p.phase_count||d.comparator>3)throw std::runtime_error("invalid reactive decision");
            for(int branch=0;branch<2;++branch){uint8_t kind=branch?d.false_kind:d.true_kind,target=branch?d.false_phase:d.true_phase;
                if(kind==CONT_PHASE){if(target<=d.after_phase||target>=p.phase_count)throw std::runtime_error("invalid reactive continuation");}
                else if(kind==CONT_ABORT){++aborts;}else if(kind==CONT_RECOVERY){recovery=true;}else throw std::runtime_error("invalid reactive continuation");
            }
        }
        if(aborts>1)throw std::runtime_error("multiple abort paths");
        if(recovery!=has_recovery)throw std::runtime_error("invalid recovery descriptor");
        if(has_recovery&&p.phase_first+p.phase_count>=out.phases.size())throw std::runtime_error("missing recovery phase");
    }
    for(const auto&m:out.motifs) if(m.action_count<1||m.action_first+m.action_count>out.actions.size()) throw std::runtime_error("invalid motif range");
    return out;
}

} // namespace rlrmb
