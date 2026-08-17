#pragma once

#include "bonsai_kg_abi.hpp"
#include <cstring>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg {

struct OwnedPolicy {
    Header header{};
    std::vector<Node> nodes;
    std::vector<uint16_t> child_refs;
    std::vector<BbDecl> bb;
    std::vector<ValueRef> values;
    std::vector<Predicate> predicates;
    std::vector<Lifecycle> lifecycles;
    std::vector<Effect> effects;

    PolicyView view() const {
        PolicyView p;
        p.nodes = nodes.data(); p.child_refs = child_refs.data(); p.bb = bb.data();
        p.values = values.data(); p.predicates = predicates.data(); p.lifecycles = lifecycles.data(); p.effects = effects.data();
        p.root = header.root; p.n_nodes = header.n_nodes; p.n_child_refs = header.n_child_refs;
        p.n_state = header.n_state; p.n_bb = header.n_bb; p.n_values = header.n_values; p.n_predicates = header.n_predicates;
        p.n_lifecycles = header.n_lifecycles; p.n_effects = header.n_effects;
        p.instruction_budget = header.instruction_budget;
        p.board_size = header.board_size;
        p.max_intents = header.max_intents;
        return p;
    }
};

inline bool feature_arg_ok(uint16_t fid, int32_t arg) {
    if (fid == F_MARKET_PRICE || fid == F_MARKET_INVENTORY) return arg >= 0 && arg < 9;
    if (fid == F_OWN_SHED) return arg >= 0 && arg < 12;
    if (fid == F_OWN_SEED) return arg >= 0 && arg < 5;
    if (fid == F_OWN_INVENTORY) {
        const int u = (arg >> 8) & 0xff, it = arg & 0xff;
        return arg >= 0 && arg <= 0xffff && u < 40 && it < 12;
    }
    if (fid == F_SHOP_AT || fid == F_SHOP_SEEN) return arg >= 0 && arg < 8;
    if (fid == F_OWN_QUADRANT_UNLOCKED || fid == F_OPP_QUADRANT_UNLOCKED) return arg >= 0 && arg < 4;
    if ((fid >= F_OWN_WORKER_X && fid <= F_OPP_WORKER_Y) || fid == F_OWN_WORKER_TILE || fid == F_OPP_WORKER_TILE)
        return arg >= 0 && arg < MAX_UNITS;
    if ((fid >= F_OWN_TILE_KIND && fid <= F_OPP_TILE_FERT_AVAILABLE) ||
        (fid >= F_OWN_TILE_PLANTED_DAY && fid <= F_OPP_TILE_HAS_ANIMAL))
        return arg >= 0 && arg < BOARD_SIZE * BOARD_SIZE;
    return arg == 0;
}

inline bool bb_value_ok(uint8_t type, int32_t value) { return valid_bb_value(type, value); }

inline uint8_t feature_type(uint16_t fid) {
    return fid == F_OWN_WORKER_TILE || fid == F_OPP_WORKER_TILE ? B_TILE : B_I32;
}

inline bool zero_bytes(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) if (p[i] != 0) return false;
    return true;
}

inline void sem_u8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }
inline void sem_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v)); out.push_back(static_cast<uint8_t>(v >> 8));
}
inline void sem_i32(std::vector<uint8_t>& out, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    out.push_back(static_cast<uint8_t>(u)); out.push_back(static_cast<uint8_t>(u >> 8));
    out.push_back(static_cast<uint8_t>(u >> 16)); out.push_back(static_cast<uint8_t>(u >> 24));
}

inline std::vector<uint8_t> semantic_bytes(const OwnedPolicy& o) {
    std::vector<uint8_t> out;
    out.reserve(128 + o.nodes.size() * 26 + o.child_refs.size() * 2 + o.bb.size() * 5 +
                o.values.size() * 12 + o.predicates.size() * 5 + o.lifecycles.size() * 24 + o.effects.size() * 6);
    const uint8_t prefix[8] = {'B','K','G','2','S','E','M',0}; for (uint8_t v : prefix) out.push_back(v);
    const Header& h = o.header;
    const uint16_t header_values[13] = {ABI_VERSION, h.board_size, h.root, h.n_nodes, h.n_child_refs, h.n_state,
                                        h.n_bb, h.n_values, h.n_predicates, h.n_lifecycles, h.n_effects, h.instruction_budget, h.max_intents};
    for (uint16_t v : header_values) sem_u16(out, v);
    for (const Node& n : o.nodes) {
        sem_u8(out, n.type); sem_u8(out, n.flags);
        sem_u16(out, n.first_child); sem_u16(out, n.child_count); sem_u16(out, n.state_slot); sem_u16(out, n.lifecycle_slot);
        sem_i32(out, n.p0); sem_i32(out, n.p1); sem_i32(out, n.p2); sem_i32(out, n.p3);
    }
    for (uint16_t c : o.child_refs) sem_u16(out, c);
    for (const BbDecl& b : o.bb) { sem_u8(out, b.type); sem_i32(out, b.initial); }
    for (const ValueRef& v : o.values) {
        sem_u8(out, v.kind); sem_u8(out, v.type); sem_u16(out, v.id); sem_i32(out, v.arg); sem_i32(out, v.value);
    }
    for (const Predicate& q : o.predicates) { sem_u16(out, q.lhs); sem_u16(out, q.rhs); sem_u8(out, q.cmp); }
    for (const Lifecycle& l : o.lifecycles) {
        const uint16_t vals[12] = {l.failure_if,l.success_if,l.skip_if,l.while_pred,l.on_success_start,l.on_success_count,
                                   l.on_failure_start,l.on_failure_count,l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};
        for (uint16_t v : vals) sem_u16(out, v);
    }
    for (const Effect& e : o.effects) { sem_u8(out, e.op); sem_u8(out, e.slot); sem_i32(out, e.value); }
    return out;
}

inline uint32_t sha_rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
inline std::array<uint8_t, 32> sha256_bytes(const std::vector<uint8_t>& input) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    std::vector<uint8_t> msg = input;
    const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8u;
    msg.push_back(0x80u);
    while ((msg.size() & 63u) != 56u) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>(bit_len >> (i * 8)));
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const size_t j = off + static_cast<size_t>(i) * 4;
            w[i] = (static_cast<uint32_t>(msg[j]) << 24) | (static_cast<uint32_t>(msg[j+1]) << 16) |
                   (static_cast<uint32_t>(msg[j+2]) << 8) | static_cast<uint32_t>(msg[j+3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = sha_rotr(w[i-15],7) ^ sha_rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const uint32_t s1 = sha_rotr(w[i-2],17) ^ sha_rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = sha_rotr(e,6) ^ sha_rotr(e,11) ^ sha_rotr(e,25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = sha_rotr(a,2) ^ sha_rotr(a,13) ^ sha_rotr(a,22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    std::array<uint8_t,32> out{};
    for (int i = 0; i < 8; ++i) {
        out[i*4] = static_cast<uint8_t>(h[i] >> 24); out[i*4+1] = static_cast<uint8_t>(h[i] >> 16);
        out[i*4+2] = static_cast<uint8_t>(h[i] >> 8); out[i*4+3] = static_cast<uint8_t>(h[i]);
    }
    return out;
}

inline std::array<uint8_t,32> semantic_hash(const OwnedPolicy& o) { return sha256_bytes(semantic_bytes(o)); }

inline void validate_owned_policy(const OwnedPolicy& o) {
    const Header& h = o.header;
    if (std::memcmp(h.magic, "BKG2", 4) != 0 || h.abi_version != ABI_VERSION || h.header_size != sizeof(Header))
        throw std::runtime_error("Bonsai-KG invalid header");
    if (!h.n_nodes || h.n_nodes > MAX_NODES || h.n_child_refs > MAX_CHILD_REFS || h.n_state > MAX_STATE ||
        h.n_bb > MAX_BB || h.n_values > MAX_VALUE_REFS || h.n_predicates > MAX_PRED ||
        h.n_lifecycles > MAX_LIFECYCLE || h.n_effects > MAX_EFFECT)
        throw std::runtime_error("Bonsai-KG table count out of bounds");
    if (h.root >= h.n_nodes || h.instruction_budget == 0 || h.instruction_budget > 4096 ||
        h.board_size != BOARD_SIZE || h.max_intents > MAX_DEFERRED_INTENTS)
        throw std::runtime_error("Bonsai-KG root/budget/max-intents invalid");
    if (o.nodes.size() != h.n_nodes || o.child_refs.size() != h.n_child_refs || o.bb.size() != h.n_bb ||
        o.values.size() != h.n_values || o.predicates.size() != h.n_predicates ||
        o.lifecycles.size() != h.n_lifecycles || o.effects.size() != h.n_effects)
        throw std::runtime_error("Bonsai-KG host table size mismatch");
    for (uint16_t i = 0; i < h.n_bb; ++i) {
        const BbDecl& b = o.bb[i];
        if (!zero_bytes(b.pad, 3) || !bb_value_ok(b.type, b.initial)) throw std::runtime_error("Bonsai-KG blackboard type/initial invalid");
    }
    for (uint16_t i = 0; i < h.n_values; ++i) {
        const ValueRef& v = o.values[i];
        if (v.type != B_I32 && v.type != B_TILE && v.type != B_UNIT) throw std::runtime_error("Bonsai-KG ValueRef type invalid");
        if (v.kind == V_CONST) {
            if (v.id != 0 || v.arg != 0 || !bb_value_ok(v.type, v.value)) throw std::runtime_error("Bonsai-KG CONST ValueRef invalid");
        } else if (v.kind == V_BB) {
            if (v.id >= h.n_bb || v.arg != 0 || v.value != 0) throw std::runtime_error("Bonsai-KG BB ValueRef invalid");
            const uint8_t declared = o.bb[v.id].type;
            const uint8_t effective = declared == B_TILE || declared == B_UNIT ? declared : static_cast<uint8_t>(B_I32);
            if (v.type != effective) throw std::runtime_error("Bonsai-KG BB ValueRef type mismatch");
        } else if (v.kind == V_LEGAL_FEATURE) {
            if (v.id > F_OPP_TILE_HAS_ANIMAL || !feature_arg_ok(v.id, v.arg) || v.value != 0 || v.type != feature_type(v.id))
                throw std::runtime_error("Bonsai-KG LEGAL_FEATURE ValueRef invalid");
        } else throw std::runtime_error("Bonsai-KG ValueRef kind invalid");
    }
    for (uint16_t i = 0; i < h.n_predicates; ++i) {
        const Predicate& q = o.predicates[i];
        if (q.cmp > C_GE || q.lhs >= h.n_values || q.rhs >= h.n_values || !zero_bytes(q.pad, 7) ||
            o.values[q.lhs].type != o.values[q.rhs].type) throw std::runtime_error("Bonsai-KG predicate encoding invalid");
    }
    std::vector<uint8_t> value_used(h.n_values, 0);
    for (uint16_t i = 0; i < h.n_predicates; ++i) {
        value_used[o.predicates[i].lhs] = 1; value_used[o.predicates[i].rhs] = 1;
    }
    for (uint16_t i = 0; i < h.n_effects; ++i) {
        const Effect& e = o.effects[i];
        if (e.op > E_ADD || e.slot >= h.n_bb || e.reserved != 0)
            throw std::runtime_error("Bonsai-KG effect invalid");
        if (e.op == E_CLEAR) {
            if (e.value != 0) throw std::runtime_error("Bonsai-KG CLEAR value noncanonical");
            continue;
        }
        if (e.value < 0 || e.value >= h.n_values) throw std::runtime_error("Bonsai-KG effect ValueRef invalid");
        const uint8_t declared = o.bb[e.slot].type;
        const uint8_t expected = declared == B_TILE || declared == B_UNIT ? declared : static_cast<uint8_t>(B_I32);
        if (o.values[e.value].type != expected) throw std::runtime_error("Bonsai-KG effect ValueRef type mismatch");
        if (e.op == E_ADD && declared != B_I32) throw std::runtime_error("Bonsai-KG ADD requires I32");
        value_used[e.value] = 1;
    }
    std::vector<uint8_t> predicate_used(h.n_predicates, 0), effect_used(h.n_effects, 0);
    for (uint16_t i = 0; i < h.n_lifecycles; ++i) {
        const Lifecycle& l = o.lifecycles[i];
        const uint16_t ps[4] = {l.failure_if,l.success_if,l.skip_if,l.while_pred};
        for (uint16_t x : ps) if (x != NONE_U16) {
            if (x >= h.n_predicates || predicate_used[x]) throw std::runtime_error("Bonsai-KG lifecycle predicate invalid/aliased");
            predicate_used[x] = 1;
        }
        const uint16_t ranges[8] = {l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,
                                    l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};
        for (int k = 0; k < 8; k += 2) {
            if (static_cast<uint32_t>(ranges[k]) + ranges[k+1] > h.n_effects || (ranges[k+1] == 0 && ranges[k] != 0))
                throw std::runtime_error("Bonsai-KG lifecycle effect range invalid");
            for (uint16_t e = ranges[k]; e < static_cast<uint16_t>(ranges[k] + ranges[k+1]); ++e) {
                if (effect_used[e]) throw std::runtime_error("Bonsai-KG lifecycle effects aliased");
                effect_used[e] = 1;
            }
        }
    }
    for (uint16_t i = 0; i < h.n_predicates; ++i) if (!predicate_used[i]) throw std::runtime_error("Bonsai-KG predicate table not dense/used");
    for (uint16_t i = 0; i < h.n_effects; ++i) if (!effect_used[i]) throw std::runtime_error("Bonsai-KG effect table not dense/used");
    std::vector<uint8_t> state_used(h.n_state, 0), child_ref_used(h.n_child_refs, 0), lifecycle_used(h.n_lifecycles, 0);
    uint16_t static_max_intents = 0;
    for (uint16_t i = 0; i < h.n_nodes; ++i) {
        const Node& n = o.nodes[i];
        if (!valid_node_type(n.type) || static_cast<uint32_t>(n.first_child) + n.child_count > h.n_child_refs)
            throw std::runtime_error("Bonsai-KG node encoding invalid");
        if (n.type == N_EMIT_UNIT_INTENT || n.type == N_EMIT_MARKET_INTENT || n.type == N_MOVE_UNIT_TO) ++static_max_intents;
        if (n.reserved != 0) throw std::runtime_error("Bonsai-KG node reserved field nonzero");
        if (n.child_count == 0 && n.first_child != 0) throw std::runtime_error("Bonsai-KG empty child range noncanonical");
        for (uint16_t pos = n.first_child; pos < static_cast<uint16_t>(n.first_child + n.child_count); ++pos) {
            if (child_ref_used[pos]) throw std::runtime_error("Bonsai-KG child-reference ranges overlap");
            child_ref_used[pos] = 1;
        }
        if (n.state_slot != NONE_U16 && n.state_slot >= h.n_state) throw std::runtime_error("Bonsai-KG state slot invalid");
        if (n.lifecycle_slot != NONE_U16) {
            if (n.lifecycle_slot >= h.n_lifecycles || lifecycle_used[n.lifecycle_slot]) throw std::runtime_error("Bonsai-KG lifecycle slot invalid/aliased");
            lifecycle_used[n.lifecycle_slot] = 1;
        }
        if (n.debug_id != i) throw std::runtime_error("Bonsai-KG debug id invalid");
        const bool core_stateful = n.type == N_SEQUENCE || n.type == N_SELECTOR || n.type == N_REACTIVE_SEQUENCE ||
                                   n.type == N_REACTIVE_SELECTOR || n.type == N_IF || n.type == N_PARALLEL ||
                                   n.type == N_TURN_TIMEOUT || n.type == N_RETRY || n.type == N_REPEAT || n.type == N_WAIT_TURNS;
        const bool needs_state = core_stateful || n.lifecycle_slot != NONE_U16;
        if (needs_state && n.state_slot == NONE_U16) throw std::runtime_error("Bonsai-KG missing required state slot");
        if (!needs_state && n.state_slot != NONE_U16) throw std::runtime_error("Bonsai-KG unnecessary state slot");
        if (n.state_slot != NONE_U16) {
            if (state_used[n.state_slot]) throw std::runtime_error("Bonsai-KG aliased state slot");
            state_used[n.state_slot] = 1;
        }
        for (uint16_t k = 0; k < n.child_count; ++k) if (o.child_refs[n.first_child+k] >= h.n_nodes) throw std::runtime_error("Bonsai-KG child ref invalid");
        const bool composite = n.type <= N_PARALLEL;
        if ((n.type == N_SEQUENCE || n.type == N_SELECTOR || n.type == N_REACTIVE_SEQUENCE || n.type == N_REACTIVE_SELECTOR) && n.child_count == 0)
            throw std::runtime_error("Bonsai-KG empty composite");
        if (n.type == N_IF && n.child_count != 3) throw std::runtime_error("Bonsai-KG If cardinality invalid");
        if (n.type == N_PARALLEL && (n.child_count < 2 || n.child_count > 16 || n.p0 < 1 || n.p0 > n.child_count || n.p1 < 1 || n.p1 > n.child_count))
            throw std::runtime_error("Bonsai-KG Parallel invalid");
        if (is_decorator(n.type) && n.child_count != 1) throw std::runtime_error("Bonsai-KG decorator cardinality invalid");
        if (!composite && !is_decorator(n.type) && n.child_count != 0) throw std::runtime_error("Bonsai-KG leaf has children");
        if (n.type != N_CONDITION && n.type != N_EMIT_UNIT_INTENT && n.flags != 0) throw std::runtime_error("Bonsai-KG unused node flags nonzero");
        bool allow[4] = {false,false,false,false};
        if (n.type == N_PARALLEL) allow[0]=allow[1]=true;
        else if (n.type==N_TURN_TIMEOUT||n.type==N_RETRY||n.type==N_REPEAT||n.type==N_WAIT_TURNS) allow[0]=true;
        else if (n.type==N_CONDITION||n.type==N_SET_BLACKBOARD||n.type==N_ADD_BLACKBOARD) allow[0]=allow[1]=true;
        else if (n.type==N_EMIT_UNIT_INTENT||n.type==N_EMIT_MARKET_INTENT) allow[0]=allow[1]=allow[2]=allow[3]=true;
        else if (n.type==N_MOVE_UNIT_TO) allow[0]=allow[1]=allow[2]=true;
        const int32_t params[4]={n.p0,n.p1,n.p2,n.p3};
        for(int k=0;k<4;++k)if(!allow[k]&&params[k]!=0)throw std::runtime_error("Bonsai-KG unused node parameter nonzero");
        if ((n.type == N_TURN_TIMEOUT || n.type == N_WAIT_TURNS) && (n.p0 < 1 || n.p0 > 720)) throw std::runtime_error("Bonsai-KG turn bound invalid");
        if ((n.type == N_RETRY || n.type == N_REPEAT) && (n.p0 < 1 || n.p0 > 255)) throw std::runtime_error("Bonsai-KG retry/repeat bound invalid");
        if (n.type == N_CONDITION) {
            if (n.flags > C_GE || n.p0 < 0 || n.p0 >= h.n_values || n.p1 < 0 || n.p1 >= h.n_values ||
                o.values[n.p0].type != o.values[n.p1].type) throw std::runtime_error("Bonsai-KG condition invalid");
        }
        if (n.type == N_CONDITION) { value_used[n.p0] = 1; value_used[n.p1] = 1; }
        if (n.type == N_SET_BLACKBOARD || n.type == N_ADD_BLACKBOARD) {
            if (n.p0 < 0 || n.p0 >= h.n_bb || n.p1 < 0 || n.p1 >= h.n_values)
                throw std::runtime_error("Bonsai-KG blackboard leaf operand invalid");
            const uint8_t declared = o.bb[n.p0].type;
            const uint8_t expected = declared == B_TILE || declared == B_UNIT ? declared : static_cast<uint8_t>(B_I32);
            if (o.values[n.p1].type != expected) throw std::runtime_error("Bonsai-KG blackboard leaf ValueRef type mismatch");
            if (n.type == N_ADD_BLACKBOARD && declared != B_I32) throw std::runtime_error("Bonsai-KG AddBlackboard requires I32");
            value_used[n.p1] = 1;
        }
        if (n.type == N_EMIT_UNIT_INTENT) {
            if (n.p0 < 0 || n.p0 >= MAX_UNITS || n.p1 < U_PASS || n.p1 > U_CARE || n.p3 < 1 || n.p3 > 32767)
                throw std::runtime_error("Bonsai-KG unit intent invalid");
            if (n.p1 == U_PLANT && (n.p2 < 0 || n.p2 >= 5)) throw std::runtime_error("Bonsai-KG PLANT item invalid");
            if ((n.p1 == U_PICKUP || n.p1 == U_PLACE) && (n.p2 < 0 || n.p2 >= 12)) throw std::runtime_error("Bonsai-KG unit item invalid");
            if (n.p1 != U_PLANT && n.p1 != U_PICKUP && n.p1 != U_PLACE && n.p2 != 0) throw std::runtime_error("Bonsai-KG no-item unit arg nonzero");
            if (n.p1 != U_PICKUP && n.p1 != U_PLACE && n.p3 != 1) throw std::runtime_error("Bonsai-KG ignored unit quantity noncanonical");
        }
        if (n.type == N_EMIT_MARKET_INTENT) {
            if (n.p0 < M_HIRE || n.p0 > M_SELL || n.p2 < 1 || n.p2 > 65535 || n.p3 < 0 || n.p3 > 255)
                throw std::runtime_error("Bonsai-KG market intent invalid");
            if ((n.p0 == M_HIRE || n.p0 == M_BUY_LAND) && (n.p1 != 0 || n.p2 != 1)) throw std::runtime_error("Bonsai-KG no-arg market intent invalid");
            if (n.p0 == M_BUY_SEED && (n.p1 < 0 || n.p1 >= 5)) throw std::runtime_error("Bonsai-KG BUY_SEED item invalid");
            if (n.p0 == M_BUY_PRODUCT && n.p1 != 0 && n.p1 != 8) throw std::runtime_error("Bonsai-KG BUY_PRODUCT item invalid");
            if (n.p0 == M_BUY_ANIMAL && (n.p1 < 9 || n.p1 > 11)) throw std::runtime_error("Bonsai-KG BUY_ANIMAL item invalid");
            if (n.p0 == M_SELL && (n.p1 < 0 || n.p1 >= 9)) throw std::runtime_error("Bonsai-KG SELL item invalid");
        }
        if (n.type == N_MOVE_UNIT_TO && (n.p0 < 0 || n.p0 >= h.n_values || n.p1 < 0 || n.p1 >= h.n_values ||
            o.values[n.p0].type != B_UNIT || o.values[n.p1].type != B_TILE || n.p2 < 0 || n.p2 > 255))
            throw std::runtime_error("Bonsai-KG MoveUnitTo invalid");
        if (n.type == N_MOVE_UNIT_TO) { value_used[n.p0] = 1; value_used[n.p1] = 1; }
    }
    if (static_max_intents != h.max_intents) throw std::runtime_error("Bonsai-KG max-intents bound is noncanonical");
    for (uint16_t i = 0; i < h.n_lifecycles; ++i) if (!lifecycle_used[i]) throw std::runtime_error("Bonsai-KG lifecycle table not dense/used");
    for (uint16_t i = 0; i < h.n_child_refs; ++i) if (!child_ref_used[i]) throw std::runtime_error("Bonsai-KG child-reference table not dense/used");
    for (uint16_t i = 0; i < h.n_values; ++i) if (!value_used[i]) throw std::runtime_error("Bonsai-KG ValueRef table not dense/used");
    for (uint16_t i = 0; i < h.n_state; ++i) if (!state_used[i]) throw std::runtime_error("Bonsai-KG state slots are not dense/used");
    std::vector<uint8_t> parent_count(h.n_nodes, 0);
    for (uint16_t i = 0; i < h.n_nodes; ++i) {
        const Node& n = o.nodes[i];
        for (uint16_t k = 0; k < n.child_count; ++k) {
            const uint16_t c = o.child_refs[n.first_child+k];
            if (c == h.root || ++parent_count[c] > 1) throw std::runtime_error("Bonsai-KG structure is not a tree");
        }
    }
    struct WalkFrame { uint16_t id, next; uint8_t depth; };
    std::vector<uint8_t> color(h.n_nodes, 0);
    std::vector<WalkFrame> stack;
    stack.push_back({h.root, 0, 1}); color[h.root] = 1;
    while (!stack.empty()) {
        WalkFrame& f = stack.back(); const Node& n = o.nodes[f.id];
        if (f.depth > MAX_DEPTH) throw std::runtime_error("Bonsai-KG depth exceeds limit");
        if (f.next < n.child_count) {
            const uint16_t c = o.child_refs[n.first_child + f.next++];
            if (color[c] == 1) throw std::runtime_error("Bonsai-KG cycle detected");
            if (color[c] == 0) { color[c] = 1; stack.push_back({c, 0, static_cast<uint8_t>(f.depth + 1)}); }
        } else { color[f.id] = 2; stack.pop_back(); }
    }
    for (uint16_t i = 0; i < h.n_nodes; ++i) if (color[i] == 0) throw std::runtime_error("Bonsai-KG unreachable node");
}

inline OwnedPolicy load_compiled(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open Bonsai-KG policy: " + path);
    OwnedPolicy o;
    in.read(reinterpret_cast<char*>(&o.header), sizeof(o.header));
    if (!in) throw std::runtime_error("short Bonsai-KG header");
    const Header& h = o.header;
    if (std::memcmp(h.magic, "BKG2", 4) != 0 || h.abi_version != ABI_VERSION || h.header_size != sizeof(Header))
        throw std::runtime_error("invalid Bonsai-KG header");
    if (h.n_nodes > MAX_NODES || h.n_child_refs > MAX_CHILD_REFS || h.n_state > MAX_STATE || h.n_bb > MAX_BB ||
        h.n_values > MAX_VALUE_REFS || h.n_predicates > MAX_PRED || h.n_lifecycles > MAX_LIFECYCLE || h.n_effects > MAX_EFFECT)
        throw std::runtime_error("Bonsai-KG counts exceed limits");
    o.nodes.resize(h.n_nodes); o.child_refs.resize(h.n_child_refs); o.bb.resize(h.n_bb); o.values.resize(h.n_values);
    o.predicates.resize(h.n_predicates); o.lifecycles.resize(h.n_lifecycles); o.effects.resize(h.n_effects);
    auto read_vec = [&](auto& v) { if (!v.empty()) in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(v[0]))); };
    read_vec(o.nodes); read_vec(o.child_refs); read_vec(o.bb); read_vec(o.values); read_vec(o.predicates); read_vec(o.lifecycles); read_vec(o.effects);
    if (!in) throw std::runtime_error("short Bonsai-KG payload");
    char extra; if (in.read(&extra,1)) throw std::runtime_error("trailing Bonsai-KG payload bytes");
    validate_owned_policy(o);
    const auto actual_hash = semantic_hash(o);
    if (std::memcmp(actual_hash.data(), o.header.hash, actual_hash.size()) != 0)
        throw std::runtime_error("Bonsai-KG semantic hash mismatch");
    return o;
}

} // namespace bonsaikg
