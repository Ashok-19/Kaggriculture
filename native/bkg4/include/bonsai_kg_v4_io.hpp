#pragma once

#include "bonsai_kg_v4_abi.hpp"
#include "bonsai_kg_io.hpp" // reuse only the already-tested portable SHA-256 utility
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg4 {

struct OwnedPolicy4 {
    Header4 header{};
    std::vector<Node4> nodes;
    std::vector<uint16_t> child_refs;
    std::vector<BbDecl4> bb;
    std::vector<ValueRef4> values;
    std::vector<FeatureArgRef4> feature_args;
    std::vector<Predicate4> predicates;
    std::vector<Lifecycle4> lifecycles;
    std::vector<Effect4> effects;

    PolicyView4 view() const {
        PolicyView4 p;
        p.nodes=nodes.data();p.child_refs=child_refs.data();p.bb=bb.data();p.values=values.data();
        p.feature_args=feature_args.data();p.predicates=predicates.data();p.lifecycles=lifecycles.data();p.effects=effects.data();
        p.n_nodes=header.n_nodes;p.n_state=header.n_state;p.n_bb=header.n_bb;p.n_values=header.n_values;
        p.n_feature_args=header.n_feature_args;p.n_predicates=header.n_predicates;p.n_lifecycles=header.n_lifecycles;
        p.n_effects=header.n_effects;p.instruction_budget=header.instruction_budget;p.max_live_intents=header.max_live_intents;
        p.max_depth=header.max_depth;p.feature_registry_version=header.feature_registry_version;return p;
    }
};

struct StaticInfo4 {
    bool may_run=false, needs_state=false;
    uint16_t depth=1;
    uint32_t visits=1, live_intents=0, emissions=0;
};

inline bool zero_bytes4(const uint8_t* p,size_t n){for(size_t i=0;i<n;++i)if(p[i])return false;return true;}
inline uint8_t feature_type4(uint16_t fid){
    if(fid==F_OWN_WORKER_TILE||fid==F_OPP_WORKER_TILE)return B_TILE;
    if(fid==F_OWN_TILE_ITEM||fid==F_OPP_TILE_ITEM)return B_ITEM;
    return B_I32;
}
inline uint8_t feature_arg_count4(uint16_t fid){
    if(fid==F_OWN_INVENTORY)return 2;
    if(fid==F_MARKET_PRICE||fid==F_MARKET_INVENTORY||fid==F_OWN_SHED||fid==F_OWN_SEED||fid==F_SHOP_AT||fid==F_SHOP_SEEN||
       fid==F_OWN_QUADRANT_UNLOCKED||fid==F_OPP_QUADRANT_UNLOCKED||(fid>=F_OWN_WORKER_X&&fid<=F_OPP_WORKER_Y)||
       fid==F_OWN_WORKER_TILE||fid==F_OPP_WORKER_TILE||(fid>=F_OWN_TILE_KIND&&fid<=F_OPP_TILE_FERT_AVAILABLE)||
       (fid>=F_OWN_TILE_PLANTED_DAY&&fid<=F_OPP_TILE_HAS_ANIMAL))return 1;
    return 0;
}
inline uint8_t feature_arg_type4(uint16_t fid,uint8_t pos){
    if(fid==F_OWN_INVENTORY)return pos==0?B_UNIT:B_ITEM;
    if(fid==F_MARKET_PRICE||fid==F_MARKET_INVENTORY||fid==F_OWN_SHED||fid==F_OWN_SEED)return B_ITEM;
    if((fid>=F_OWN_WORKER_X&&fid<=F_OPP_WORKER_Y)||fid==F_OWN_WORKER_TILE||fid==F_OPP_WORKER_TILE)return B_UNIT;
    if((fid>=F_OWN_TILE_KIND&&fid<=F_OPP_TILE_FERT_AVAILABLE)||(fid>=F_OWN_TILE_PLANTED_DAY&&fid<=F_OPP_TILE_HAS_ANIMAL))return B_TILE;
    return B_I32;
}
inline bool feature_const_arg_ok4(uint16_t fid,uint8_t pos,int32_t v){
    const uint8_t t=feature_arg_type4(fid,pos);
    if(t==B_TILE)return v>=-1&&v<BOARD_SIZE*BOARD_SIZE;
    if(t==B_UNIT)return v>=-1&&v<MAX_UNITS;
    if(t==B_ITEM)return v>=-1&&v<MAX_ITEMS;
    if(fid==F_SHOP_AT||fid==F_SHOP_SEEN)return v>=0&&v<8;
    if(fid==F_OWN_QUADRANT_UNLOCKED||fid==F_OPP_QUADRANT_UNLOCKED)return v>=0&&v<4;
    return true;
}
inline uint16_t value_tag4(const ValueRef4& v){return (v.kind==V_CONST||v.kind==V_BB)&&v.arg>0&&v.arg<=65535?static_cast<uint16_t>(v.arg):0;}

inline std::vector<StaticInfo4> analyze_static4(const OwnedPolicy4& o){
    const auto& nodes=o.nodes;const auto& ch=o.child_refs;std::vector<StaticInfo4>a(nodes.size());
    for(int ii=(int)nodes.size()-1;ii>=0;--ii){
        const Node4&n=nodes[(size_t)ii];StaticInfo4 x{};x.depth=1;bool any=false;uint32_t sumv=0,suml=0,sume=0,maxv=0,maxl=0;
        for(uint16_t k=0;k<n.child_count;++k){const StaticInfo4&c=a[ch[n.first_child+k]];any|=c.may_run;sumv+=c.visits;suml+=c.live_intents;sume+=c.emissions;if(c.visits>maxv)maxv=c.visits;if(c.live_intents>maxl)maxl=c.live_intents;if((uint32_t)c.depth+1>x.depth)x.depth=(uint16_t)(c.depth+1);}
        switch(n.type){
            case N_WAIT_TURNS:case N_MOVE_UNIT_TO:case N_KEEP_RUNNING_UNTIL_FAILURE:x.may_run=true;break;
            case N_RETRY:case N_REPEAT:x.may_run=n.p0>1||any;break;
            case N_SEQUENCE:case N_SELECTOR:case N_REACTIVE_SEQUENCE:case N_REACTIVE_SELECTOR:case N_IF:case N_PARALLEL:
            case N_TURN_TIMEOUT:case N_INVERT:case N_FORCE_SUCCESS:case N_FORCE_FAILURE:case N_SWITCH_ENUM:x.may_run=any;break;
            case N_REACTIVE_IF:
                if(n.child_count!=3)throw std::runtime_error("BKG4 ReactiveIf cardinality invalid");
                if(a[ch[n.first_child]].may_run)throw std::runtime_error("BKG4 ReactiveIf condition must be synchronous");
                x.may_run=a[ch[n.first_child+1]].may_run||a[ch[n.first_child+2]].may_run;break;
            default:x.may_run=false;break;
        }
        bool semantic=false;
        switch(n.type){case N_SEQUENCE:case N_SELECTOR:case N_REACTIVE_SEQUENCE:case N_REACTIVE_SELECTOR:case N_IF:case N_PARALLEL:case N_REACTIVE_IF:case N_SWITCH_ENUM:case N_RETRY:case N_REPEAT:case N_WAIT_TURNS:semantic=true;break;default:break;}
        if(n.type==N_TURN_TIMEOUT)semantic=any;
        x.needs_state=semantic&&x.may_run;if(n.lifecycle_slot!=NONE_U16&&x.may_run)x.needs_state=true;
        if(n.type==N_SWITCH_ENUM)x.visits=1+maxv;
        else if(n.type==N_IF||n.type==N_REACTIVE_IF){uint32_t cond=n.child_count?a[ch[n.first_child]].visits:0,br=0;for(uint16_t k=1;k<n.child_count;++k)if(a[ch[n.first_child+k]].visits>br)br=a[ch[n.first_child+k]].visits;x.visits=1+cond+br;}
        else if(is_decorator(n.type))x.visits=1+(n.child_count?a[ch[n.first_child]].visits:0);
        else x.visits=1+sumv;
        if(n.type==N_SELECTOR||n.type==N_REACTIVE_SELECTOR||n.type==N_SWITCH_ENUM)x.live_intents=maxl;
        else if(n.type==N_IF||n.type==N_REACTIVE_IF){uint32_t cond=n.child_count?a[ch[n.first_child]].live_intents:0,br=0;for(uint16_t k=1;k<n.child_count;++k)if(a[ch[n.first_child+k]].live_intents>br)br=a[ch[n.first_child+k]].live_intents;x.live_intents=cond+br;}
        else if(is_decorator(n.type))x.live_intents=n.child_count?a[ch[n.first_child]].live_intents:0;
        else x.live_intents=suml;
        if(n.type==N_EMIT_UNIT_INTENT||n.type==N_EMIT_MARKET_INTENT||n.type==N_MOVE_UNIT_TO)x.live_intents=x.live_intents<1?1:x.live_intents;
        x.emissions=sume+((n.type==N_EMIT_UNIT_INTENT||n.type==N_EMIT_MARKET_INTENT||n.type==N_MOVE_UNIT_TO)?1u:0u);
        if(x.emissions>65534u)throw std::runtime_error("BKG4 emission-order proof exceeds uint16 capacity");
        a[(size_t)ii]=x;
    }
    return a;
}

inline void sem4_u8(std::vector<uint8_t>&o,uint8_t v){o.push_back(v);} inline void sem4_u16(std::vector<uint8_t>&o,uint16_t v){o.push_back((uint8_t)v);o.push_back((uint8_t)(v>>8));}
inline void sem4_i32(std::vector<uint8_t>&o,int32_t v){uint32_t u=(uint32_t)v;o.push_back((uint8_t)u);o.push_back((uint8_t)(u>>8));o.push_back((uint8_t)(u>>16));o.push_back((uint8_t)(u>>24));}
inline void sem4_u32(std::vector<uint8_t>&o,uint32_t u){o.push_back((uint8_t)u);o.push_back((uint8_t)(u>>8));o.push_back((uint8_t)(u>>16));o.push_back((uint8_t)(u>>24));}
inline std::vector<uint8_t> semantic_bytes4(const OwnedPolicy4&o){
    std::vector<uint8_t>out;const uint8_t prefix[8]={'B','K','G','4','S','E','M',0};for(uint8_t v:prefix)out.push_back(v);const Header4&h=o.header;
    const uint16_t hv[14]={ABI_VERSION,h.feature_registry_version,h.n_nodes,h.n_state,h.n_bb,h.n_values,h.n_feature_args,h.n_predicates,h.n_lifecycles,h.n_effects,h.instruction_budget,h.max_live_intents,h.max_depth,BOARD_SIZE};for(uint16_t v:hv)sem4_u16(out,v);
    for(const Node4&n:o.nodes){sem4_u8(out,n.type);sem4_u8(out,n.flags);sem4_u16(out,n.first_child);sem4_u16(out,n.child_count);sem4_u16(out,n.state_slot);sem4_u16(out,n.lifecycle_slot);sem4_u16(out,n.subtree_end);sem4_i32(out,n.p0);sem4_i32(out,n.p1);sem4_i32(out,n.p2);sem4_i32(out,n.p3);sem4_u32(out,n.aux);}
    for(uint16_t c:o.child_refs)sem4_u16(out,c);
    for(const BbDecl4&b:o.bb){sem4_u8(out,b.type);sem4_u8(out,b.enum_size);sem4_u16(out,b.enum_tag);sem4_i32(out,b.initial);}
    for(const ValueRef4&v:o.values){sem4_u8(out,v.kind);sem4_u8(out,v.type);sem4_u16(out,v.id);sem4_i32(out,v.arg);sem4_i32(out,v.value);}
    for(const FeatureArgRef4&a:o.feature_args){sem4_u8(out,a.kind);sem4_u8(out,a.type);sem4_u16(out,a.id);sem4_i32(out,a.value);}
    for(const Predicate4&q:o.predicates){sem4_u16(out,q.lhs);sem4_u16(out,q.rhs);sem4_u8(out,q.cmp);}
    for(const Lifecycle4&l:o.lifecycles){const uint16_t z[12]={l.failure_if,l.success_if,l.skip_if,l.while_pred,l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};for(uint16_t v:z)sem4_u16(out,v);}
    for(const Effect4&e:o.effects){sem4_u8(out,e.op);sem4_u8(out,e.slot);sem4_i32(out,e.value);}
    return out;
}
inline std::array<uint8_t,32> semantic_hash4(const OwnedPolicy4&o){return bonsaikg::sha256_bytes(semantic_bytes4(o));}

inline void validate_action4(const OwnedPolicy4&o,const Node4&n){
    auto exact=[&](int32_t r,uint8_t t,int32_t value){if(r<0||r>=(int32_t)o.values.size())return false;const auto&v=o.values[(size_t)r];return v.kind==V_CONST&&v.type==t&&v.arg==0&&v.value==value;};
    if(n.type==N_EMIT_UNIT_INTENT){if(n.p1<U_PASS||n.p1>U_CARE)throw std::runtime_error("BKG4 unit op invalid");const auto&i=o.values[(size_t)n.p2];const auto&q=o.values[(size_t)n.p3];if(n.p1==U_PLANT){if(!exact(n.p3,B_I32,1))throw std::runtime_error("BKG4 PLANT qty noncanonical");if(i.kind==V_CONST&&(i.value<0||i.value>=5))throw std::runtime_error("BKG4 PLANT item invalid");}else if(n.p1==U_PICKUP||n.p1==U_PLACE){if(i.kind==V_CONST&&(i.value<0||i.value>=MAX_ITEMS))throw std::runtime_error("BKG4 PICKUP/PLACE item invalid");if(q.kind==V_CONST&&(q.value<1||q.value>32767))throw std::runtime_error("BKG4 PICKUP/PLACE qty invalid");}else if(!exact(n.p2,B_ITEM,-1)||!exact(n.p3,B_I32,1))throw std::runtime_error("BKG4 unit hidden operands noncanonical");}
    else if(n.type==N_EMIT_MARKET_INTENT){if(n.p0<M_HIRE||n.p0>M_SELL)throw std::runtime_error("BKG4 market op invalid");const auto&i=o.values[(size_t)n.p1];const auto&q=o.values[(size_t)n.p2];if(n.p0==M_HIRE||n.p0==M_BUY_LAND){if(!exact(n.p1,B_ITEM,-1)||!exact(n.p2,B_I32,1))throw std::runtime_error("BKG4 atomic market operands noncanonical");}else{if(i.kind==V_CONST){const int v=i.value;if(n.p0==M_BUY_SEED&&(v<0||v>=5))throw std::runtime_error("BKG4 BUY_SEED item invalid");if(n.p0==M_BUY_PRODUCT&&v!=0&&v!=8)throw std::runtime_error("BKG4 BUY_PRODUCT item invalid");if(n.p0==M_BUY_ANIMAL&&(v<9||v>11))throw std::runtime_error("BKG4 BUY_ANIMAL item invalid");if(n.p0==M_SELL&&(v<0||v>=9))throw std::runtime_error("BKG4 SELL item invalid");}if(q.kind==V_CONST&&(q.value<1||q.value>65535))throw std::runtime_error("BKG4 market qty invalid");}}
}

inline void validate_owned_policy4(const OwnedPolicy4&o){
    const Header4&h=o.header;if(std::memcmp(h.magic,"BKG4",4)!=0||h.abi_version!=ABI_VERSION||h.header_size!=sizeof(Header4))throw std::runtime_error("BKG4 invalid header identity");
    if(!h.n_nodes||h.n_nodes>MAX_NODES||h.n_state>MAX_STATE||h.n_bb>MAX_BB||h.n_values>MAX_TABLE||h.n_feature_args>MAX_TABLE||h.n_predicates>MAX_TABLE||h.n_lifecycles>MAX_TABLE||h.n_effects>MAX_TABLE||!h.instruction_budget||h.instruction_budget>MAX_INSTRUCTION_BUDGET||h.max_live_intents>MAX_LIVE_INTENTS||!h.max_depth||h.max_depth>MAX_DEPTH||h.feature_registry_version!=FEATURE_REGISTRY_VERSION)throw std::runtime_error("BKG4 header counts/proofs invalid");
    if(o.nodes.size()!=h.n_nodes||o.child_refs.size()!=static_cast<size_t>(h.n_nodes-1)||o.bb.size()!=h.n_bb||o.values.size()!=h.n_values||o.feature_args.size()!=h.n_feature_args||o.predicates.size()!=h.n_predicates||o.lifecycles.size()!=h.n_lifecycles||o.effects.size()!=h.n_effects)throw std::runtime_error("BKG4 host table size mismatch");
    std::vector<uint16_t> enum_size(65536,0);uint16_t max_tag=0;for(const auto&b:o.bb){if(!valid_bb_value(b.type,b.initial))throw std::runtime_error("BKG4 blackboard invalid");if(b.enum_tag){if(b.type!=B_ENUM8||!b.enum_size||b.initial<0||b.initial>=b.enum_size)throw std::runtime_error("BKG4 named enum BB invalid");if(enum_size[b.enum_tag]&&enum_size[b.enum_tag]!=b.enum_size)throw std::runtime_error("BKG4 enum tag conflict");enum_size[b.enum_tag]=b.enum_size;if(b.enum_tag>max_tag)max_tag=b.enum_tag;}else if(b.enum_size)throw std::runtime_error("BKG4 raw BB enum_size invalid");}for(uint16_t t=1;t<=max_tag;++t)if(!enum_size[t])throw std::runtime_error("BKG4 enum tags not dense");
    std::vector<uint8_t> used_arg(h.n_feature_args,0),used_value(h.n_values,0),used_pred(h.n_predicates,0),used_eff(h.n_effects,0),used_lc(h.n_lifecycles,0),used_state(h.n_state,0),child_used(o.child_refs.size(),0);std::vector<uint8_t>parents(h.n_nodes,0);
    for(uint16_t i=0;i<h.n_feature_args;++i){const auto&a=o.feature_args[i];if((a.kind!=A_CONST&&a.kind!=A_BB)||(a.type!=B_I32&&a.type!=B_TILE&&a.type!=B_UNIT&&a.type!=B_ITEM))throw std::runtime_error("BKG4 FeatureArg invalid");if(a.kind==A_CONST){if(a.id||!valid_bb_value(a.type,a.value))throw std::runtime_error("BKG4 FeatureArg CONST invalid");}else if(a.id>=h.n_bb||a.value||o.bb[a.id].enum_tag||a.type!=value_type_for_bb(o.bb[a.id]))throw std::runtime_error("BKG4 FeatureArg BB invalid");}
    for(uint16_t i=0;i<h.n_values;++i){const auto&v=o.values[i];if((v.kind>V_LEGAL_FEATURE)||(v.type!=B_I32&&v.type!=B_TILE&&v.type!=B_UNIT&&v.type!=B_ITEM))throw std::runtime_error("BKG4 ValueRef invalid");if(v.kind==V_CONST){if(v.id)throw std::runtime_error("BKG4 CONST id invalid");if(v.arg){if(v.type!=B_I32||v.arg<1||v.arg>65535||!enum_size[(uint16_t)v.arg]||v.value<0||v.value>=enum_size[(uint16_t)v.arg])throw std::runtime_error("BKG4 enum CONST invalid");}else if(!valid_bb_value(v.type,v.value))throw std::runtime_error("BKG4 CONST invalid");}else if(v.kind==V_BB){if(v.id>=h.n_bb||v.value||v.type!=value_type_for_bb(o.bb[v.id])||v.arg!=o.bb[v.id].enum_tag)throw std::runtime_error("BKG4 BB ValueRef invalid");}else{if(v.id>LAST_FEATURE_V1||v.type!=feature_type4(v.id)||v.value!=feature_arg_count4(v.id)||v.value<0||v.arg<0||(uint32_t)v.arg+(uint32_t)v.value>h.n_feature_args||(v.value==0&&v.arg))throw std::runtime_error("BKG4 legal feature invalid");for(int32_t k=0;k<v.value;++k){uint16_t ai=(uint16_t)(v.arg+k);used_arg[ai]=1;const auto&a=o.feature_args[ai];if(a.type!=feature_arg_type4(v.id,(uint8_t)k)||(a.kind==A_CONST&&!feature_const_arg_ok4(v.id,(uint8_t)k,a.value)))throw std::runtime_error("BKG4 legal feature arg invalid");}}}
    for(uint16_t i=0;i<h.n_predicates;++i){const auto&q=o.predicates[i];if(q.cmp>C_GE||q.lhs>=h.n_values||q.rhs>=h.n_values||!zero_bytes4(q.pad,7))throw std::runtime_error("BKG4 predicate invalid");const auto&a=o.values[q.lhs],&b=o.values[q.rhs];if(a.type!=b.type||value_tag4(a)!=value_tag4(b)||(value_tag4(a)&&q.cmp!=C_EQ&&q.cmp!=C_NE))throw std::runtime_error("BKG4 predicate type/tag invalid");}
    for(uint16_t i=0;i<h.n_effects;++i){const auto&e=o.effects[i];if(e.op>E_ADD||e.slot>=h.n_bb||e.reserved)throw std::runtime_error("BKG4 effect invalid");if(e.op==E_CLEAR){if(e.value)throw std::runtime_error("BKG4 CLEAR value invalid");continue;}if(e.value<0||e.value>=h.n_values)throw std::runtime_error("BKG4 effect ValueRef invalid");const auto&v=o.values[(size_t)e.value];if(v.type!=value_type_for_bb(o.bb[e.slot])||value_tag4(v)!=o.bb[e.slot].enum_tag||(e.op==E_ADD&&o.bb[e.slot].type!=B_I32))throw std::runtime_error("BKG4 effect type invalid");}
    auto require_ref=[&](int32_t r,uint8_t typ,uint16_t tag=0){if(r<0||r>=h.n_values)throw std::runtime_error("BKG4 node ValueRef range invalid");const auto&v=o.values[(size_t)r];if(v.type!=typ||value_tag4(v)!=tag)throw std::runtime_error("BKG4 node ValueRef type/tag invalid");used_value[(size_t)r]=1;};
    for(uint16_t i=0;i<h.n_nodes;++i){const Node4&n=o.nodes[i];if(!valid_node_type(n.type)||n.subtree_end<=i||n.subtree_end>h.n_nodes||(uint32_t)n.first_child+n.child_count>o.child_refs.size()||(n.child_count==0&&n.first_child))throw std::runtime_error("BKG4 node encoding invalid");for(uint16_t p=n.first_child;p<n.first_child+n.child_count;++p){if(child_used[p])throw std::runtime_error("BKG4 child range alias");child_used[p]=1;uint16_t c=o.child_refs[p];if(c>=h.n_nodes||++parents[c]>1)throw std::runtime_error("BKG4 strict tree invalid");}if(n.state_slot!=NONE_U16){if(n.state_slot>=h.n_state||used_state[n.state_slot])throw std::runtime_error("BKG4 state slot invalid/aliased");used_state[n.state_slot]=1;}if(n.lifecycle_slot!=NONE_U16){if(n.lifecycle_slot>=h.n_lifecycles)throw std::runtime_error("BKG4 lifecycle slot invalid");used_lc[n.lifecycle_slot]=1;}
        if((n.type==N_SEQUENCE||n.type==N_SELECTOR||n.type==N_REACTIVE_SEQUENCE||n.type==N_REACTIVE_SELECTOR)&&!n.child_count)throw std::runtime_error("BKG4 empty composite");
        if((n.type==N_IF||n.type==N_REACTIVE_IF)&&n.child_count!=3)throw std::runtime_error("BKG4 If cardinality invalid");
        if(n.type==N_PARALLEL&&(n.child_count<2||n.child_count>16||n.p0<1||n.p0>n.child_count||n.p1<1||n.p1>n.child_count))throw std::runtime_error("BKG4 Parallel invalid");
        if(is_decorator(n.type)&&n.child_count!=1)throw std::runtime_error("BKG4 decorator invalid");
        if(n.type==N_SWITCH_ENUM){if(n.p0<0||n.p0>=h.n_bb)throw std::runtime_error("BKG4 Switch slot invalid");const auto&b=o.bb[(size_t)n.p0];if(!b.enum_tag||n.p1!=b.enum_tag||n.p2!=b.enum_size||n.child_count!=b.enum_size+1)throw std::runtime_error("BKG4 Switch encoding invalid");}
        if(n.type==N_CONDITION){if(n.flags>C_GE)throw std::runtime_error("BKG4 Condition compare invalid");uint16_t tag=(n.p1>=0&&n.p1<h.n_values)?value_tag4(o.values[(size_t)n.p1]):0;uint8_t typ=(n.p1>=0&&n.p1<h.n_values)?o.values[(size_t)n.p1].type:255;require_ref(n.p0,typ,tag);require_ref(n.p1,typ,tag);if(tag&&n.flags!=C_EQ&&n.flags!=C_NE)throw std::runtime_error("BKG4 enum Condition invalid");}else if(n.type==N_SET_BLACKBOARD||n.type==N_ADD_BLACKBOARD){if(n.p0<0||n.p0>=h.n_bb)throw std::runtime_error("BKG4 BB leaf invalid");const auto&b=o.bb[(size_t)n.p0];require_ref(n.p1,value_type_for_bb(b),b.enum_tag);if(n.type==N_ADD_BLACKBOARD&&b.type!=B_I32)throw std::runtime_error("BKG4 Add requires I32");}else if(n.type==N_EMIT_UNIT_INTENT){require_ref(n.p0,B_UNIT);require_ref(n.p2,B_ITEM);require_ref(n.p3,B_I32);if(!valid_priority_key(n.aux))throw std::runtime_error("BKG4 unit authority invalid");validate_action4(o,n);}else if(n.type==N_EMIT_MARKET_INTENT){require_ref(n.p1,B_ITEM);require_ref(n.p2,B_I32);if(!valid_priority_key(n.aux))throw std::runtime_error("BKG4 market authority invalid");validate_action4(o,n);}else if(n.type==N_MOVE_UNIT_TO){require_ref(n.p0,B_UNIT);require_ref(n.p1,B_TILE);if(!valid_priority_key(n.aux))throw std::runtime_error("BKG4 move authority invalid");}else if((n.type==N_WAIT_TURNS||n.type==N_TURN_TIMEOUT)&&(n.p0<1||n.p0>720))throw std::runtime_error("BKG4 turn bound invalid");else if((n.type==N_RETRY||n.type==N_REPEAT)&&(n.p0<1||n.p0>255))throw std::runtime_error("BKG4 count invalid");}
    if(parents[0])throw std::runtime_error("BKG4 root has parent");
    for(uint16_t i=1;i<h.n_nodes;++i)if(parents[i]!=1)throw std::runtime_error("BKG4 parent uniqueness invalid");
    for(uint8_t u:child_used)if(!u)throw std::runtime_error("BKG4 child ref unused");
    struct F{uint16_t id,next,depth;};std::vector<F>st;uint32_t expected=1;st.push_back({0,0,1});while(!st.empty()){F&f=st.back();const auto&n=o.nodes[f.id];if(f.depth>MAX_DEPTH)throw std::runtime_error("BKG4 depth invalid");if(f.next<n.child_count){uint16_t c=o.child_refs[n.first_child+f.next++];if(c!=expected)throw std::runtime_error("BKG4 preorder invalid");++expected;st.push_back({c,0,(uint16_t)(f.depth+1)});}else{if(n.subtree_end!=expected)throw std::runtime_error("BKG4 subtree_end invalid");st.pop_back();}}if(expected!=h.n_nodes)throw std::runtime_error("BKG4 unreachable node");
    const auto proof=analyze_static4(o);uint16_t state=0;for(uint16_t i=0;i<h.n_nodes;++i){if(proof[i].needs_state){if(o.nodes[i].state_slot!=state++)throw std::runtime_error("BKG4 state-elision proof mismatch");}else if(o.nodes[i].state_slot!=NONE_U16)throw std::runtime_error("BKG4 unnecessary state slot");}if(state!=h.n_state)throw std::runtime_error("BKG4 state count mismatch");if(proof[0].depth!=h.max_depth)throw std::runtime_error("BKG4 max_depth proof mismatch");if(proof[0].live_intents!=h.max_live_intents)throw std::runtime_error("BKG4 max_live_intents proof mismatch");if(proof[0].visits>h.instruction_budget)throw std::runtime_error("BKG4 instruction budget under proof");
    for(const auto&l:o.lifecycles){const uint16_t ps[4]={l.failure_if,l.success_if,l.skip_if,l.while_pred};for(uint16_t q:ps)if(q!=NONE_U16){if(q>=h.n_predicates)throw std::runtime_error("BKG4 lifecycle predicate invalid");used_pred[q]=1;}const uint16_t rr[8]={l.on_success_start,l.on_success_count,l.on_failure_start,l.on_failure_count,l.on_halted_start,l.on_halted_count,l.post_start,l.post_count};for(int k=0;k<8;k+=2){uint16_t s=rr[k],c=rr[k+1];if((uint32_t)s+c>h.n_effects||(c==0&&s))throw std::runtime_error("BKG4 lifecycle effect range invalid");for(uint16_t e=s;e<s+c;++e)used_eff[e]=1;}}
    for(const auto&q:o.predicates){used_value[q.lhs]=used_value[q.rhs]=1;}for(const auto&e:o.effects)if(e.op!=E_CLEAR)used_value[(size_t)e.value]=1;
    for(uint8_t u:used_arg)if(!u)throw std::runtime_error("BKG4 FeatureArg unused");
    for(uint8_t u:used_value)if(!u)throw std::runtime_error("BKG4 ValueRef unused");
    for(uint8_t u:used_pred)if(!u)throw std::runtime_error("BKG4 predicate unused");
    for(uint8_t u:used_eff)if(!u)throw std::runtime_error("BKG4 effect unused");
    for(uint8_t u:used_lc)if(!u)throw std::runtime_error("BKG4 lifecycle unused");
    for(uint8_t u:used_state)if(!u)throw std::runtime_error("BKG4 state unused");
}

inline OwnedPolicy4 load_compiled4(const std::string&path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open BKG4 policy: "+path);OwnedPolicy4 o;in.read((char*)&o.header,sizeof(o.header));if(!in)throw std::runtime_error("short BKG4 header");const auto&h=o.header;if(std::memcmp(h.magic,"BKG4",4)!=0||h.abi_version!=ABI_VERSION||h.header_size!=sizeof(Header4))throw std::runtime_error("invalid BKG4 header");if(!h.n_nodes||h.n_nodes>MAX_NODES||h.n_state>MAX_STATE||h.n_bb>MAX_BB||h.n_values>MAX_TABLE||h.n_feature_args>MAX_TABLE||h.n_predicates>MAX_TABLE||h.n_lifecycles>MAX_TABLE||h.n_effects>MAX_TABLE)throw std::runtime_error("BKG4 counts exceed limits");const uint32_t nc=h.n_nodes-1;o.nodes.resize(h.n_nodes);o.child_refs.resize(nc);o.bb.resize(h.n_bb);o.values.resize(h.n_values);o.feature_args.resize(h.n_feature_args);o.predicates.resize(h.n_predicates);o.lifecycles.resize(h.n_lifecycles);o.effects.resize(h.n_effects);auto readv=[&](auto&v){if(!v.empty())in.read((char*)v.data(),(std::streamsize)(v.size()*sizeof(v[0])));};readv(o.nodes);readv(o.child_refs);readv(o.bb);readv(o.values);readv(o.feature_args);readv(o.predicates);readv(o.lifecycles);readv(o.effects);if(!in)throw std::runtime_error("short BKG4 payload");char extra;if(in.read(&extra,1))throw std::runtime_error("trailing BKG4 bytes");validate_owned_policy4(o);const auto actual=semantic_hash4(o);if(std::memcmp(actual.data(),h.hash,32)!=0)throw std::runtime_error("BKG4 semantic hash mismatch");return o;
}

} // namespace bonsaikg4
