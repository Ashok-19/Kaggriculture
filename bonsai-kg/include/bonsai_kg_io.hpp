#pragma once

#include "bonsai_kg_abi.hpp"
#include <cstring>
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
    std::vector<Predicate> predicates;
    std::vector<Lifecycle> lifecycles;
    std::vector<Effect> effects;

    PolicyView view() const {
        PolicyView p;
        p.nodes = nodes.data(); p.child_refs = child_refs.data(); p.bb = bb.data();
        p.predicates = predicates.data(); p.lifecycles = lifecycles.data(); p.effects = effects.data();
        p.root = header.root; p.n_nodes = header.n_nodes; p.n_child_refs = header.n_child_refs;
        p.n_state = header.n_state; p.n_bb = header.n_bb; p.n_predicates = header.n_predicates;
        p.n_lifecycles = header.n_lifecycles; p.n_effects = header.n_effects;
        p.instruction_budget = header.instruction_budget;
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
    if (fid >= F_OWN_WORKER_X && fid <= F_OPP_WORKER_Y) return arg >= 0 && arg < 40;
    if (fid >= F_OWN_TILE_KIND && fid <= F_OPP_TILE_FERT_AVAILABLE) return arg >= 0 && arg < 100;
    return arg == 0;
}

inline bool bb_value_ok(uint8_t type, int32_t value) {
    if (type == 0) return value == 0 || value == 1;
    if (type == 1) return true;
    if (type == 2) return value >= 0 && value <= 65535;
    if (type == 3) return value >= 0 && value <= 255;
    if (type == 4) return value >= 0 && value <= 99;
    if (type == 5) return value >= 0 && value < MAX_UNITS;
    return false;
}

inline void validate_owned_policy(const OwnedPolicy& o) {
    const Header& h = o.header;
    if (std::memcmp(h.magic, "BKG1", 4) != 0 || h.abi_version != ABI_VERSION || h.header_size != sizeof(Header))
        throw std::runtime_error("Bonsai-KG invalid header");
    if (!h.n_nodes || h.n_nodes > MAX_NODES || h.n_child_refs > MAX_CHILD_REFS || h.n_state > MAX_STATE ||
        h.n_bb > MAX_BB || h.n_predicates > MAX_PRED || h.n_lifecycles > MAX_LIFECYCLE || h.n_effects > MAX_EFFECT)
        throw std::runtime_error("Bonsai-KG table count out of bounds");
    if (h.root >= h.n_nodes || h.instruction_budget == 0 || h.instruction_budget > 4096)
        throw std::runtime_error("Bonsai-KG root/budget invalid");
    if (o.nodes.size() != h.n_nodes || o.child_refs.size() != h.n_child_refs || o.bb.size() != h.n_bb ||
        o.predicates.size() != h.n_predicates || o.lifecycles.size() != h.n_lifecycles || o.effects.size() != h.n_effects)
        throw std::runtime_error("Bonsai-KG host table size mismatch");
    for (uint16_t i = 0; i < h.n_bb; ++i) {
        const BbDecl& b = o.bb[i];
        if (!bb_value_ok(b.type, b.initial)) throw std::runtime_error("Bonsai-KG blackboard type/initial invalid");
    }
    for (uint16_t i = 0; i < h.n_predicates; ++i) {
        const Predicate& q = o.predicates[i];
        if (q.cmp > C_GE || q.source > 1) throw std::runtime_error("Bonsai-KG predicate encoding invalid");
        if (q.source == 0) {
            if (q.id > F_OPP_TILE_FERT_AVAILABLE || !feature_arg_ok(q.id, q.arg))
                throw std::runtime_error("Bonsai-KG feature id/arg invalid");
        } else if (q.id >= h.n_bb || q.arg != 0) throw std::runtime_error("Bonsai-KG blackboard predicate invalid");
    }
    for (uint16_t i = 0; i < h.n_effects; ++i) {
        const Effect& e = o.effects[i];
        if (e.op > E_ADD || e.slot >= h.n_bb) throw std::runtime_error("Bonsai-KG effect invalid");
        if (e.op == E_ADD && o.bb[e.slot].type != 1) throw std::runtime_error("Bonsai-KG ADD requires I32");
        if (e.op == E_SET && !bb_value_ok(o.bb[e.slot].type, e.value)) throw std::runtime_error("Bonsai-KG SET value/type invalid");
    }
    for (uint16_t i = 0; i < h.n_lifecycles; ++i) {
        const Lifecycle& l = o.lifecycles[i];
        const uint16_t ps[4] = {l.failure_if,l.success_if,l.skip_if,l.while_pred};
        for (uint16_t x : ps) if (x != NONE_U16 && x >= h.n_predicates) throw std::runtime_error("Bonsai-KG lifecycle predicate invalid");
        const uint16_t ranges[8] = {l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,
                                    l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};
        for (int k = 0; k < 8; k += 2)
            if (static_cast<uint32_t>(ranges[k]) + ranges[k+1] > h.n_effects) throw std::runtime_error("Bonsai-KG lifecycle effect range invalid");
    }
    std::vector<uint8_t> state_used(h.n_state, 0);
    for (uint16_t i = 0; i < h.n_nodes; ++i) {
        const Node& n = o.nodes[i];
        if (!valid_node_type(n.type) || static_cast<uint32_t>(n.first_child) + n.child_count > h.n_child_refs)
            throw std::runtime_error("Bonsai-KG node encoding invalid");
        if (n.state_slot != NONE_U16 && n.state_slot >= h.n_state) throw std::runtime_error("Bonsai-KG state slot invalid");
        if (n.lifecycle_slot != NONE_U16 && n.lifecycle_slot >= h.n_lifecycles) throw std::runtime_error("Bonsai-KG lifecycle slot invalid");
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
        if ((n.type == N_TURN_TIMEOUT || n.type == N_WAIT_TURNS) && (n.p0 < 1 || n.p0 > 720)) throw std::runtime_error("Bonsai-KG turn bound invalid");
        if ((n.type == N_RETRY || n.type == N_REPEAT) && (n.p0 < 1 || n.p0 > 255)) throw std::runtime_error("Bonsai-KG retry/repeat bound invalid");
        if (n.type == N_CONDITION) {
            if (n.flags > C_GE || n.p0 < 0 || n.p0 > 1) throw std::runtime_error("Bonsai-KG condition invalid");
            if (n.p0 == 0 && (n.p1 < 0 || n.p1 > F_OPP_TILE_FERT_AVAILABLE || !feature_arg_ok(static_cast<uint16_t>(n.p1), n.p2))) throw std::runtime_error("Bonsai-KG condition feature invalid");
            if (n.p0 == 1 && (n.p1 < 0 || n.p1 >= h.n_bb)) throw std::runtime_error("Bonsai-KG condition blackboard invalid");
        }
        if (n.type == N_SET_BLACKBOARD || n.type == N_ADD_BLACKBOARD) {
            if (n.p0 < 0 || n.p0 >= h.n_bb) throw std::runtime_error("Bonsai-KG blackboard leaf slot invalid");
            if (n.type == N_SET_BLACKBOARD && !bb_value_ok(o.bb[n.p0].type, n.p1)) throw std::runtime_error("Bonsai-KG SetBlackboard value invalid");
            if (n.type == N_ADD_BLACKBOARD && o.bb[n.p0].type != 1) throw std::runtime_error("Bonsai-KG AddBlackboard requires I32");
        }
        if (n.type == N_EMIT_UNIT_INTENT) {
            if (n.p0 < 0 || n.p0 >= MAX_UNITS || n.p1 < U_PASS || n.p1 > U_CARE || n.p3 < 1 || n.p3 > 32767)
                throw std::runtime_error("Bonsai-KG unit intent invalid");
            if (n.p1 == U_PLANT && (n.p2 < 0 || n.p2 >= 5)) throw std::runtime_error("Bonsai-KG PLANT item invalid");
            if ((n.p1 == U_PICKUP || n.p1 == U_PLACE) && (n.p2 < 0 || n.p2 >= 12)) throw std::runtime_error("Bonsai-KG unit item invalid");
        }
        if (n.type == N_EMIT_MARKET_INTENT) {
            if (n.p0 < M_HIRE || n.p0 > M_SELL || n.p2 <= 0 || n.p3 < 0 || n.p3 > 255)
                throw std::runtime_error("Bonsai-KG market intent invalid");
            if ((n.p0 == M_HIRE || n.p0 == M_BUY_LAND) && (n.p1 != 0 || n.p2 != 1)) throw std::runtime_error("Bonsai-KG no-arg market intent invalid");
            if (n.p0 == M_BUY_SEED && (n.p1 < 0 || n.p1 >= 5)) throw std::runtime_error("Bonsai-KG BUY_SEED item invalid");
            if (n.p0 == M_BUY_PRODUCT && n.p1 != 0 && n.p1 != 8) throw std::runtime_error("Bonsai-KG BUY_PRODUCT item invalid");
            if (n.p0 == M_BUY_ANIMAL && (n.p1 < 9 || n.p1 > 11)) throw std::runtime_error("Bonsai-KG BUY_ANIMAL item invalid");
            if (n.p0 == M_SELL && (n.p1 < 0 || n.p1 >= 9)) throw std::runtime_error("Bonsai-KG SELL item invalid");
        }
        if (n.type == N_MOVE_UNIT_TO && (n.p0 < 0 || n.p0 >= MAX_UNITS || n.p1 < 0 || n.p1 >= 10 || n.p2 < 0 || n.p2 >= 10 || n.p3 < 0 || n.p3 > 255))
            throw std::runtime_error("Bonsai-KG MoveUnitTo invalid");
    }
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
    if (std::memcmp(h.magic, "BKG1", 4) != 0 || h.abi_version != ABI_VERSION || h.header_size != sizeof(Header))
        throw std::runtime_error("invalid Bonsai-KG header");
    if (h.n_nodes > MAX_NODES || h.n_child_refs > MAX_CHILD_REFS || h.n_state > MAX_STATE || h.n_bb > MAX_BB ||
        h.n_predicates > MAX_PRED || h.n_lifecycles > MAX_LIFECYCLE || h.n_effects > MAX_EFFECT)
        throw std::runtime_error("Bonsai-KG counts exceed limits");
    o.nodes.resize(h.n_nodes); o.child_refs.resize(h.n_child_refs); o.bb.resize(h.n_bb);
    o.predicates.resize(h.n_predicates); o.lifecycles.resize(h.n_lifecycles); o.effects.resize(h.n_effects);
    auto read_vec = [&](auto& v) { if (!v.empty()) in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(v[0]))); };
    read_vec(o.nodes); read_vec(o.child_refs); read_vec(o.bb); read_vec(o.predicates); read_vec(o.lifecycles); read_vec(o.effects);
    if (!in) throw std::runtime_error("short Bonsai-KG payload");
    char extra; if (in.read(&extra,1)) throw std::runtime_error("trailing Bonsai-KG payload bytes");
    validate_owned_policy(o);
    return o;
}

} // namespace bonsaikg
