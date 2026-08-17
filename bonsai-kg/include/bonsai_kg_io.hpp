#pragma once

#include "bonsai_kg_abi.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg {

inline uint16_t static_max_intents(const std::vector<Node>& nodes) {
    uint16_t n = 0;
    for (const Node& node : nodes)
        n += node.type == N_EMIT_UNIT_INTENT || node.type == N_EMIT_MARKET_INTENT || node.type == N_MOVE_UNIT_TO;
    return n;
}

struct OwnedPolicy {
    Header header{};
    std::vector<Node> nodes;
    std::vector<uint16_t> child_refs;
    std::vector<BbDecl> bb;
    std::vector<ValueRef> values;
    std::vector<FeatureArgRef> feature_args;
    std::vector<Predicate> predicates;
    std::vector<Lifecycle> lifecycles;
    std::vector<Effect> effects;

    PolicyView view() const {
        PolicyView p;
        p.nodes=nodes.data(); p.child_refs=child_refs.data(); p.bb=bb.data(); p.values=values.data();
        p.feature_args=feature_args.data(); p.predicates=predicates.data(); p.lifecycles=lifecycles.data(); p.effects=effects.data();
        p.root=header.root; p.n_nodes=header.n_nodes; p.n_child_refs=header.n_child_refs; p.n_state=header.n_state;
        p.n_bb=header.n_bb; p.n_values=header.n_values; p.n_feature_args=header.n_feature_args;
        p.n_predicates=header.n_predicates; p.n_lifecycles=header.n_lifecycles; p.n_effects=header.n_effects;
        p.instruction_budget=header.instruction_budget; p.board_size=header.board_size; p.max_intents=static_max_intents(nodes);
        return p;
    }
};

inline uint8_t feature_type(uint16_t fid) {
    if (fid == F_OWN_WORKER_TILE || fid == F_OPP_WORKER_TILE) return B_TILE;
    if (fid == F_OWN_TILE_ITEM || fid == F_OPP_TILE_ITEM) return B_ITEM;
    return B_I32;
}
inline uint8_t feature_arg_count(uint16_t fid) {
    if (fid == F_OWN_INVENTORY) return 2;
    if (fid == F_MARKET_PRICE || fid == F_MARKET_INVENTORY || fid == F_OWN_SHED || fid == F_OWN_SEED ||
        fid == F_SHOP_AT || fid == F_SHOP_SEEN || fid == F_OWN_QUADRANT_UNLOCKED || fid == F_OPP_QUADRANT_UNLOCKED ||
        (fid >= F_OWN_WORKER_X && fid <= F_OPP_WORKER_Y) || fid == F_OWN_WORKER_TILE || fid == F_OPP_WORKER_TILE ||
        (fid >= F_OWN_TILE_KIND && fid <= F_OPP_TILE_FERT_AVAILABLE) ||
        (fid >= F_OWN_TILE_PLANTED_DAY && fid <= F_OPP_TILE_HAS_ANIMAL)) return 1;
    return 0;
}
inline uint8_t feature_arg_type(uint16_t fid, uint8_t pos) {
    if (fid == F_OWN_INVENTORY) return pos == 0 ? B_UNIT : B_ITEM;
    if (fid == F_MARKET_PRICE || fid == F_MARKET_INVENTORY || fid == F_OWN_SHED || fid == F_OWN_SEED) return B_ITEM;
    if ((fid >= F_OWN_WORKER_X && fid <= F_OPP_WORKER_Y) || fid == F_OWN_WORKER_TILE || fid == F_OPP_WORKER_TILE) return B_UNIT;
    if ((fid >= F_OWN_TILE_KIND && fid <= F_OPP_TILE_FERT_AVAILABLE) ||
        (fid >= F_OWN_TILE_PLANTED_DAY && fid <= F_OPP_TILE_HAS_ANIMAL)) return B_TILE;
    return B_I32;
}
inline bool feature_const_arg_ok(uint16_t fid, uint8_t pos, int32_t v) {
    const uint8_t type = feature_arg_type(fid,pos);
    if (type == B_TILE) return v >= -1 && v < BOARD_SIZE*BOARD_SIZE;
    if (type == B_UNIT) return v >= -1 && v < MAX_UNITS;
    if (type == B_ITEM) return v >= -1 && v < MAX_ITEMS;
    if (fid == F_SHOP_AT || fid == F_SHOP_SEEN) return v >= 0 && v < 8;
    if (fid == F_OWN_QUADRANT_UNLOCKED || fid == F_OPP_QUADRANT_UNLOCKED) return v >= 0 && v < 4;
    return true;
}
inline bool zero_bytes(const uint8_t* p,size_t n){for(size_t i=0;i<n;++i)if(p[i])return false;return true;}

inline void sem_u8(std::vector<uint8_t>&o,uint8_t v){o.push_back(v);}
inline void sem_u16(std::vector<uint8_t>&o,uint16_t v){o.push_back((uint8_t)v);o.push_back((uint8_t)(v>>8));}
inline void sem_i32(std::vector<uint8_t>&o,int32_t v){uint32_t u=(uint32_t)v;o.push_back((uint8_t)u);o.push_back((uint8_t)(u>>8));o.push_back((uint8_t)(u>>16));o.push_back((uint8_t)(u>>24));}

inline std::vector<uint8_t> semantic_bytes(const OwnedPolicy&o){
    std::vector<uint8_t> out;
    out.reserve(128+o.nodes.size()*28+o.child_refs.size()*2+o.bb.size()*5+o.values.size()*12+o.feature_args.size()*8+o.predicates.size()*5+o.lifecycles.size()*24+o.effects.size()*6);
    const uint8_t prefix[8]={'B','K','G','3','S','E','M',0};for(uint8_t v:prefix)out.push_back(v);
    const Header&h=o.header;
    const uint16_t hv[14]={ABI_VERSION,h.board_size,h.root,h.n_nodes,h.n_child_refs,h.n_state,h.n_bb,h.n_values,h.n_feature_args,h.n_predicates,h.n_lifecycles,h.n_effects,h.instruction_budget,static_max_intents(o.nodes)};
    for(uint16_t v:hv)sem_u16(out,v);
    for(const Node&n:o.nodes){sem_u8(out,n.type);sem_u8(out,n.flags);sem_u16(out,n.first_child);sem_u16(out,n.child_count);sem_u16(out,n.state_slot);sem_u16(out,n.lifecycle_slot);sem_u16(out,n.subtree_end);sem_i32(out,n.p0);sem_i32(out,n.p1);sem_i32(out,n.p2);sem_i32(out,n.p3);}
    for(uint16_t c:o.child_refs)sem_u16(out,c);
    for(const BbDecl&b:o.bb){sem_u8(out,b.type);sem_i32(out,b.initial);}
    for(const ValueRef&v:o.values){sem_u8(out,v.kind);sem_u8(out,v.type);sem_u16(out,v.id);sem_i32(out,v.arg);sem_i32(out,v.value);}
    for(const FeatureArgRef&a:o.feature_args){sem_u8(out,a.kind);sem_u8(out,a.type);sem_u16(out,a.id);sem_i32(out,a.value);}
    for(const Predicate&q:o.predicates){sem_u16(out,q.lhs);sem_u16(out,q.rhs);sem_u8(out,q.cmp);}
    for(const Lifecycle&l:o.lifecycles){const uint16_t x[12]={l.failure_if,l.success_if,l.skip_if,l.while_pred,l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};for(uint16_t v:x)sem_u16(out,v);}
    for(const Effect&e:o.effects){sem_u8(out,e.op);sem_u8(out,e.slot);sem_i32(out,e.value);}
    return out;
}

inline uint32_t sha_rotr(uint32_t x,uint32_t n){return(x>>n)|(x<<(32u-n));}
inline std::array<uint8_t,32> sha256_bytes(const std::vector<uint8_t>&input){
    static const uint32_t k[64]={
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    std::vector<uint8_t>msg=input;const uint64_t bit_len=(uint64_t)msg.size()*8u;msg.push_back(0x80u);while((msg.size()&63u)!=56u)msg.push_back(0);for(int i=7;i>=0;--i)msg.push_back((uint8_t)(bit_len>>(i*8)));
    uint32_t h[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    for(size_t off=0;off<msg.size();off+=64){uint32_t w[64];for(int i=0;i<16;++i){size_t j=off+(size_t)i*4;w[i]=((uint32_t)msg[j]<<24)|((uint32_t)msg[j+1]<<16)|((uint32_t)msg[j+2]<<8)|(uint32_t)msg[j+3];}for(int i=16;i<64;++i){uint32_t s0=sha_rotr(w[i-15],7)^sha_rotr(w[i-15],18)^(w[i-15]>>3);uint32_t s1=sha_rotr(w[i-2],17)^sha_rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];for(int i=0;i<64;++i){const uint32_t ki=k[i];uint32_t s1=sha_rotr(e,6)^sha_rotr(e,11)^sha_rotr(e,25);uint32_t ch=(e&f)^((~e)&g);uint32_t t1=hh+s1+ch+ki+w[i];uint32_t s0=sha_rotr(a,2)^sha_rotr(a,13)^sha_rotr(a,22);uint32_t maj=(a&b)^(a&c)^(b&c);uint32_t t2=s0+maj;hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;}
    std::array<uint8_t,32>out{};for(int i=0;i<8;++i){out[i*4]=(uint8_t)(h[i]>>24);out[i*4+1]=(uint8_t)(h[i]>>16);out[i*4+2]=(uint8_t)(h[i]>>8);out[i*4+3]=(uint8_t)h[i];}return out;
}
inline std::array<uint8_t,32> semantic_hash(const OwnedPolicy&o){return sha256_bytes(semantic_bytes(o));}

inline void validate_owned_policy(const OwnedPolicy&o){
    const Header&h=o.header;
    if(std::memcmp(h.magic,"BKG3",4)!=0||h.abi_version!=ABI_VERSION||h.header_size!=sizeof(Header))throw std::runtime_error("Bonsai-KG invalid header");
    if(!h.n_nodes||h.root!=0||h.n_nodes>MAX_NODES||h.n_child_refs>MAX_CHILD_REFS||h.n_state>MAX_STATE||h.n_bb>MAX_BB||h.n_values>MAX_VALUE_REFS||h.n_feature_args>MAX_FEATURE_ARGS||h.n_predicates>MAX_PRED||h.n_lifecycles>MAX_LIFECYCLE||h.n_effects>MAX_EFFECT||h.instruction_budget==0||h.instruction_budget>4096||h.board_size!=BOARD_SIZE)throw std::runtime_error("Bonsai-KG header/count/budget invalid");
    if(o.nodes.size()!=h.n_nodes||o.child_refs.size()!=h.n_child_refs||o.bb.size()!=h.n_bb||o.values.size()!=h.n_values||o.feature_args.size()!=h.n_feature_args||o.predicates.size()!=h.n_predicates||o.lifecycles.size()!=h.n_lifecycles||o.effects.size()!=h.n_effects)throw std::runtime_error("Bonsai-KG host table size mismatch");
    for(const BbDecl&b:o.bb)if(!zero_bytes(b.pad,3)||!valid_bb_value(b.type,b.initial))throw std::runtime_error("Bonsai-KG blackboard invalid");
    for(const FeatureArgRef&a:o.feature_args){if(a.type!=B_I32&&a.type!=B_TILE&&a.type!=B_UNIT&&a.type!=B_ITEM)throw std::runtime_error("Bonsai-KG FeatureArg type invalid");if(a.kind==A_CONST){if(a.id!=0||!valid_bb_value(a.type,a.value))throw std::runtime_error("Bonsai-KG FeatureArg CONST invalid");}else if(a.kind==A_BB){if(a.id>=h.n_bb||a.value!=0||a.type!=value_type_for_bb(o.bb[a.id].type))throw std::runtime_error("Bonsai-KG FeatureArg BB invalid");}else throw std::runtime_error("Bonsai-KG FeatureArg kind invalid");}
    std::vector<uint8_t> feature_arg_used(h.n_feature_args,0),value_used(h.n_values,0);
    for(uint16_t i=0;i<h.n_values;++i){const ValueRef&v=o.values[i];if(v.type!=B_I32&&v.type!=B_TILE&&v.type!=B_UNIT&&v.type!=B_ITEM)throw std::runtime_error("Bonsai-KG ValueRef type invalid");if(v.kind==V_CONST){if(v.id!=0||v.arg!=0||!valid_bb_value(v.type,v.value))throw std::runtime_error("Bonsai-KG CONST ValueRef invalid");}else if(v.kind==V_BB){if(v.id>=h.n_bb||v.arg!=0||v.value!=0||v.type!=value_type_for_bb(o.bb[v.id].type))throw std::runtime_error("Bonsai-KG BB ValueRef invalid");}else if(v.kind==V_LEGAL_FEATURE){if(v.id>F_OPP_TILE_HAS_ANIMAL||v.type!=feature_type(v.id)||v.value!=feature_arg_count(v.id)||v.value<0||v.arg<0||((uint32_t)v.arg+(uint32_t)v.value)>h.n_feature_args||(v.value==0&&v.arg!=0))throw std::runtime_error("Bonsai-KG LEGAL_FEATURE ValueRef invalid");for(int32_t pos=0;pos<v.value;++pos){const uint16_t ai=(uint16_t)(v.arg+pos);if(feature_arg_used[ai])throw std::runtime_error("Bonsai-KG feature arg alias");feature_arg_used[ai]=1;const FeatureArgRef&a=o.feature_args[ai];if(a.type!=feature_arg_type(v.id,(uint8_t)pos))throw std::runtime_error("Bonsai-KG feature arg type mismatch");if(a.kind==A_CONST&&!feature_const_arg_ok(v.id,(uint8_t)pos,a.value))throw std::runtime_error("Bonsai-KG feature const arg invalid");}}else throw std::runtime_error("Bonsai-KG ValueRef kind invalid");}
    for(uint8_t u:feature_arg_used)if(!u)throw std::runtime_error("Bonsai-KG FeatureArg table not dense/used");
    for(const Predicate&q:o.predicates)if(q.cmp>C_GE||q.lhs>=h.n_values||q.rhs>=h.n_values||!zero_bytes(q.pad,7)||o.values[q.lhs].type!=o.values[q.rhs].type)throw std::runtime_error("Bonsai-KG predicate invalid");
    for(const Predicate&q:o.predicates){value_used[q.lhs]=1;value_used[q.rhs]=1;}
    for(const Effect&e:o.effects){if(e.op>E_ADD||e.slot>=h.n_bb||e.reserved)throw std::runtime_error("Bonsai-KG effect invalid");if(e.op==E_CLEAR){if(e.value)throw std::runtime_error("Bonsai-KG CLEAR noncanonical");continue;}if(e.value<0||e.value>=h.n_values)throw std::runtime_error("Bonsai-KG effect ValueRef invalid");const uint8_t expected=value_type_for_bb(o.bb[e.slot].type);if(o.values[e.value].type!=expected||(e.op==E_ADD&&o.bb[e.slot].type!=B_I32))throw std::runtime_error("Bonsai-KG effect type invalid");value_used[e.value]=1;}
    std::vector<uint8_t>pred_used(h.n_predicates,0),eff_used(h.n_effects,0),life_used(h.n_lifecycles,0),state_used(h.n_state,0),child_used(h.n_child_refs,0),parents(h.n_nodes,0);
    for(const Lifecycle&l:o.lifecycles){const uint16_t ps[4]={l.failure_if,l.success_if,l.skip_if,l.while_pred};for(uint16_t q:ps)if(q!=NONE_U16){if(q>=h.n_predicates||pred_used[q])throw std::runtime_error("Bonsai-KG lifecycle predicate invalid/aliased");pred_used[q]=1;}const uint16_t ranges[8]={l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};for(int k=0;k<8;k+=2){uint16_t st=ranges[k],ct=ranges[k+1];if((uint32_t)st+ct>h.n_effects||(ct==0&&st!=0))throw std::runtime_error("Bonsai-KG lifecycle effect range invalid");for(uint16_t e=st;e<st+ct;++e){if(eff_used[e])throw std::runtime_error("Bonsai-KG lifecycle effect alias");eff_used[e]=1;}}}
    for(uint8_t u:pred_used)if(!u)throw std::runtime_error("Bonsai-KG predicate table not dense/used");
    for(uint8_t u:eff_used)if(!u)throw std::runtime_error("Bonsai-KG effect table not dense/used");
    auto require_ref=[&](int32_t r,uint8_t t){if(r<0||r>=h.n_values||o.values[r].type!=t)throw std::runtime_error("Bonsai-KG node ValueRef invalid");value_used[r]=1;};
    for(uint16_t i=0;i<h.n_nodes;++i){const Node&n=o.nodes[i];if(!valid_node_type(n.type)||n.debug_id!=i||n.subtree_end<=i||n.subtree_end>h.n_nodes||(uint32_t)n.first_child+n.child_count>h.n_child_refs||(n.child_count==0&&n.first_child!=0))throw std::runtime_error("Bonsai-KG node encoding invalid");for(uint16_t pos=n.first_child;pos<n.first_child+n.child_count;++pos){if(child_used[pos])throw std::runtime_error("Bonsai-KG child range alias");child_used[pos]=1;const uint16_t c=o.child_refs[pos];if(c>=h.n_nodes||++parents[c]>1)throw std::runtime_error("Bonsai-KG child/tree invalid");}const bool core=n.type==N_SEQUENCE||n.type==N_SELECTOR||n.type==N_REACTIVE_SEQUENCE||n.type==N_REACTIVE_SELECTOR||n.type==N_IF||n.type==N_PARALLEL||n.type==N_TURN_TIMEOUT||n.type==N_RETRY||n.type==N_REPEAT||n.type==N_WAIT_TURNS;const bool needs=core||n.lifecycle_slot!=NONE_U16;if(needs!=(n.state_slot!=NONE_U16))throw std::runtime_error("Bonsai-KG state-slot requirement invalid");if(n.state_slot!=NONE_U16){if(n.state_slot>=h.n_state||state_used[n.state_slot])throw std::runtime_error("Bonsai-KG state slot invalid/aliased");state_used[n.state_slot]=1;}if(n.lifecycle_slot!=NONE_U16){if(n.lifecycle_slot>=h.n_lifecycles||life_used[n.lifecycle_slot])throw std::runtime_error("Bonsai-KG lifecycle invalid/aliased");life_used[n.lifecycle_slot]=1;}const bool composite=n.type<=N_PARALLEL;if((n.type==N_SEQUENCE||n.type==N_SELECTOR||n.type==N_REACTIVE_SEQUENCE||n.type==N_REACTIVE_SELECTOR)&&n.child_count==0)throw std::runtime_error("Bonsai-KG empty composite");if(n.type==N_IF&&n.child_count!=3)throw std::runtime_error("Bonsai-KG If invalid");if(n.type==N_PARALLEL&&(n.child_count<2||n.child_count>16||n.p0<1||n.p0>n.child_count||n.p1<1||n.p1>n.child_count))throw std::runtime_error("Bonsai-KG Parallel invalid");if(is_decorator(n.type)&&n.child_count!=1)throw std::runtime_error("Bonsai-KG decorator invalid");if(!composite&&!is_decorator(n.type)&&n.child_count!=0)throw std::runtime_error("Bonsai-KG leaf children invalid");if(n.type!=N_CONDITION&&n.type!=N_EMIT_UNIT_INTENT&&n.flags)throw std::runtime_error("Bonsai-KG flags noncanonical");bool allow[4]={0,0,0,0};if(n.type==N_PARALLEL)allow[0]=allow[1]=1;else if(n.type==N_TURN_TIMEOUT||n.type==N_RETRY||n.type==N_REPEAT||n.type==N_WAIT_TURNS)allow[0]=1;else if(n.type==N_CONDITION||n.type==N_SET_BLACKBOARD||n.type==N_ADD_BLACKBOARD)allow[0]=allow[1]=1;else if(n.type==N_EMIT_UNIT_INTENT||n.type==N_EMIT_MARKET_INTENT)allow[0]=allow[1]=allow[2]=allow[3]=1;else if(n.type==N_MOVE_UNIT_TO)allow[0]=allow[1]=allow[2]=1;const int32_t ps[4]={n.p0,n.p1,n.p2,n.p3};for(int k=0;k<4;++k)if(!allow[k]&&ps[k])throw std::runtime_error("Bonsai-KG parameter noncanonical");if((n.type==N_TURN_TIMEOUT||n.type==N_WAIT_TURNS)&&(n.p0<1||n.p0>720))throw std::runtime_error("Bonsai-KG turn bound invalid");if((n.type==N_RETRY||n.type==N_REPEAT)&&(n.p0<1||n.p0>255))throw std::runtime_error("Bonsai-KG count invalid");if(n.type==N_CONDITION){if(n.flags>C_GE||n.p0<0||n.p1<0||n.p0>=h.n_values||n.p1>=h.n_values||o.values[n.p0].type!=o.values[n.p1].type)throw std::runtime_error("Bonsai-KG Condition invalid");value_used[n.p0]=value_used[n.p1]=1;}else if(n.type==N_SET_BLACKBOARD||n.type==N_ADD_BLACKBOARD){if(n.p0<0||n.p0>=h.n_bb)throw std::runtime_error("Bonsai-KG BB leaf slot invalid");require_ref(n.p1,value_type_for_bb(o.bb[n.p0].type));if(n.type==N_ADD_BLACKBOARD&&o.bb[n.p0].type!=B_I32)throw std::runtime_error("Bonsai-KG Add requires I32");}else if(n.type==N_EMIT_UNIT_INTENT){if(n.p1<U_PASS||n.p1>U_CARE)throw std::runtime_error("Bonsai-KG unit op invalid");require_ref(n.p0,B_UNIT);require_ref(n.p2,B_ITEM);require_ref(n.p3,B_I32);}else if(n.type==N_EMIT_MARKET_INTENT){if(n.p0<M_HIRE||n.p0>M_SELL||n.p3<0||n.p3>255)throw std::runtime_error("Bonsai-KG market op invalid");require_ref(n.p1,B_ITEM);require_ref(n.p2,B_I32);}else if(n.type==N_MOVE_UNIT_TO){require_ref(n.p0,B_UNIT);require_ref(n.p1,B_TILE);if(n.p2<0||n.p2>255)throw std::runtime_error("Bonsai-KG Move priority invalid");}}
    for(uint8_t u:child_used)if(!u)throw std::runtime_error("Bonsai-KG child table not dense/used");
    for(uint8_t u:life_used)if(!u)throw std::runtime_error("Bonsai-KG lifecycle table not dense/used");
    for(uint8_t u:state_used)if(!u)throw std::runtime_error("Bonsai-KG state table not dense/used");
    for(uint8_t u:value_used)if(!u)throw std::runtime_error("Bonsai-KG ValueRef table not dense/used");
    if(parents[0])throw std::runtime_error("Bonsai-KG root has parent");
    for(uint16_t i=1;i<h.n_nodes;++i)if(parents[i]!=1)throw std::runtime_error("Bonsai-KG node parent invalid");
    struct Frame{uint16_t id,next;uint8_t depth;};std::vector<Frame>stack;uint16_t expected=0;if(h.root!=0)throw std::runtime_error("Bonsai-KG root preorder invalid");stack.push_back({0,0,1});expected=1;while(!stack.empty()){Frame&f=stack.back();const Node&n=o.nodes[f.id];if(f.depth>MAX_DEPTH)throw std::runtime_error("Bonsai-KG depth invalid");if(f.next<n.child_count){uint16_t c=o.child_refs[n.first_child+f.next++];if(c!=expected)throw std::runtime_error("Bonsai-KG node order not canonical preorder");++expected;stack.push_back({c,0,(uint8_t)(f.depth+1)});}else{if(n.subtree_end!=expected)throw std::runtime_error("Bonsai-KG subtree_end noncanonical");stack.pop_back();}}if(expected!=h.n_nodes)throw std::runtime_error("Bonsai-KG unreachable node");
}

inline OwnedPolicy load_compiled(const std::string&path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open Bonsai-KG policy: "+path);OwnedPolicy o;in.read((char*)&o.header,sizeof(o.header));if(!in)throw std::runtime_error("short Bonsai-KG header");const Header&h=o.header;if(std::memcmp(h.magic,"BKG3",4)!=0||h.abi_version!=ABI_VERSION||h.header_size!=sizeof(Header))throw std::runtime_error("invalid Bonsai-KG header");if(h.n_nodes>MAX_NODES||h.n_child_refs>MAX_CHILD_REFS||h.n_state>MAX_STATE||h.n_bb>MAX_BB||h.n_values>MAX_VALUE_REFS||h.n_feature_args>MAX_FEATURE_ARGS||h.n_predicates>MAX_PRED||h.n_lifecycles>MAX_LIFECYCLE||h.n_effects>MAX_EFFECT)throw std::runtime_error("Bonsai-KG counts exceed limits");o.nodes.resize(h.n_nodes);o.child_refs.resize(h.n_child_refs);o.bb.resize(h.n_bb);o.values.resize(h.n_values);o.feature_args.resize(h.n_feature_args);o.predicates.resize(h.n_predicates);o.lifecycles.resize(h.n_lifecycles);o.effects.resize(h.n_effects);auto read_vec=[&](auto&v){if(!v.empty())in.read((char*)v.data(),(std::streamsize)(v.size()*sizeof(v[0])));};read_vec(o.nodes);read_vec(o.child_refs);read_vec(o.bb);read_vec(o.values);read_vec(o.feature_args);read_vec(o.predicates);read_vec(o.lifecycles);read_vec(o.effects);if(!in)throw std::runtime_error("short Bonsai-KG payload");char extra;if(in.read(&extra,1))throw std::runtime_error("trailing Bonsai-KG payload bytes");validate_owned_policy(o);const auto actual=semantic_hash(o);if(std::memcmp(actual.data(),o.header.hash,actual.size())!=0)throw std::runtime_error("Bonsai-KG semantic hash mismatch");return o;
}

} // namespace bonsaikg
