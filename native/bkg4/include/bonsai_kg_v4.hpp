#pragma once

#include "bonsai_kg_v4_abi.hpp"

#ifdef __CUDACC__
#define BKG4_VM_HD __host__ __device__ inline
#define BKG4_VM_NOINLINE __host__ __device__ __noinline__
#else
#define BKG4_VM_HD inline
#define BKG4_VM_NOINLINE inline
#endif

namespace bonsaikg4 {

struct EvalFrame4 {
    uint16_t node=0,cursor=0,aux=0,intent_start=0;
    uint8_t phase=0,skipped=0;
};
static_assert(sizeof(EvalFrame4)==10,"BKG4 EvalFrame size drift");
struct HaltFrame4 { uint16_t node=0,cursor=0;uint8_t entered=0,pad[3]={0,0,0}; };
static_assert(sizeof(HaltFrame4)==8,"BKG4 HaltFrame size drift");

template<class Obs>
struct EvaluatorCore4 {
    const PolicyView4&p;RuntimeView4 s;const Obs&obs;TraceBuffer4*trace;
    DeferredIntent4* intents=nullptr;uint16_t intent_cap=0;
    EvalFrame4* stack=nullptr;HaltFrame4* halt_stack=nullptr;uint16_t depth_cap=0;
    uint16_t n_intents=0,emit_seq=0,visits=0;int sp=0;Status last=FAILURE;bool have_last=false;
    BKG4_VM_HD EvaluatorCore4(const PolicyView4&pp,RuntimeView4 ss,const Obs&o,TraceBuffer4*t,DeferredIntent4*ii,uint16_t ic,EvalFrame4*es,HaltFrame4*hs,uint16_t dc):p(pp),s(ss),obs(o),trace(t),intents(ii),intent_cap(ic),stack(es),halt_stack(hs),depth_cap(dc){}
    BKG4_VM_HD void fault(uint16_t c){if(s.meta&&!s.meta->fault)s.meta->fault=c;}
    BKG4_VM_HD bool faulted()const{return !s.meta||s.meta->fault!=0;}
    BKG4_VM_HD int32_t feature(uint16_t id,int32_t a0=0,int32_t a1=0)const{return obs.feature(id,a0,a1);}
    BKG4_VM_HD NodeState4* st(const Node4&n){return n.state_slot!=NONE_U16&&n.state_slot<p.n_state?s.node_state+n.state_slot:nullptr;}
    BKG4_VM_HD bool active(const Node4&n){NodeState4*x=st(n);return x&&(x->b&ACTIVE_BIT);}
    BKG4_VM_HD void set_active(const Node4&n,bool v){NodeState4*x=st(n);if(!x)return;if(v)x->b|=ACTIVE_BIT;else x->b&=~ACTIVE_BIT;}
    BKG4_VM_HD uint16_t child(const Node4&n,uint16_t k)const{return p.child_refs[n.first_child+k];}
    BKG4_VM_HD void emit_trace(uint16_t node,Status status,uint8_t event){if(!trace||!trace->events||trace->count>=trace->capacity)return;TraceEvent4&e=trace->events[trace->count++];int32_t step=feature(F_STEP);if(step<-32768)step=-32768;if(step>32767)step=32767;e.step=(int16_t)step;e.node=node;e.event=event;e.status=(uint8_t)status;e.pad=0;}
    BKG4_VM_HD int32_t feature_arg(uint16_t i)const{if(i>=p.n_feature_args)return 0;const auto&a=p.feature_args[i];return a.kind==A_CONST?a.value:(a.id<p.n_bb?s.blackboard[a.id]:0);}
    BKG4_VM_HD int32_t value(uint16_t i)const{if(i>=p.n_values)return 0;const auto&v=p.values[i];if(v.kind==V_CONST)return v.value;if(v.kind==V_BB)return v.id<p.n_bb?s.blackboard[v.id]:0;const int32_t a0=v.value>0?feature_arg((uint16_t)v.arg):0,a1=v.value>1?feature_arg((uint16_t)(v.arg+1)):0;return feature(v.id,a0,a1);}
    BKG4_VM_HD bool pred(uint16_t i)const{if(i==NONE_U16||i>=p.n_predicates)return false;const auto&q=p.predicates[i];return compare_i32(value(q.lhs),value(q.rhs),q.cmp);}
    BKG4_VM_HD void effects(uint16_t start,uint16_t count){for(uint16_t k=0;k<count;++k){uint32_t i=(uint32_t)start+k;if(faulted())return;if(i>=p.n_effects){fault(3);return;}const auto&e=p.effects[i];if(e.slot>=p.n_bb){fault(3);return;}if(e.op==E_SET){int32_t v=value((uint16_t)e.value);const auto&b=p.bb[e.slot];if(!valid_bb_value(b.type,v)||(b.enum_tag&&(v<0||v>=b.enum_size))){fault(5);return;}s.blackboard[e.slot]=v;}else if(e.op==E_CLEAR)s.blackboard[e.slot]=p.bb[e.slot].initial;else if(e.op==E_ADD){bool ok=false;int32_t v=clamp_i32_add(s.blackboard[e.slot],value((uint16_t)e.value),ok);if(!ok){fault(2);return;}s.blackboard[e.slot]=v;}else{fault(3);return;}}}
    BKG4_VM_HD void cancel_subtree_intents(uint16_t root){if(intent_cap==0||root>=p.n_nodes)return;uint16_t hi=p.nodes[root].subtree_end,out=0;for(uint16_t i=0;i<n_intents;++i){uint16_t src=intents[i].source;if(src>=root&&src<hi)continue;if(out!=i)intents[out]=intents[i];++out;}n_intents=out;}
    BKG4_VM_HD void halt_node(uint16_t root){cancel_subtree_intents(root);HaltFrame4*hs=halt_stack;int hp=0;hs[hp++]={root,0,0,{0,0,0}};while(hp>0){auto&f=hs[hp-1];if(f.node>=p.n_nodes){fault(3);--hp;continue;}const Node4&n=p.nodes[f.node];NodeState4*x=st(n);if(!f.entered){f.entered=1;f.cursor=0;}int next=-1;if(n.type==N_SEQUENCE||n.type==N_SELECTOR){if(f.cursor==0&&x&&x->a<n.child_count){next=child(n,(uint16_t)x->a);f.cursor=1;}}else if(n.type==N_REACTIVE_SEQUENCE||n.type==N_REACTIVE_SELECTOR){if(f.cursor==0&&x&&x->a&&x->a-1<n.child_count){next=child(n,(uint16_t)(x->a-1));f.cursor=1;}}else if(n.type==N_IF){if(f.cursor==0&&x&&n.child_count==3){uint32_t ph=x->a;uint16_t ci=ph==0?0:(uint16_t)ph;if(ci<3)next=child(n,ci);f.cursor=1;}}else if(n.type==N_REACTIVE_IF){if(f.cursor==0&&x&&(x->a==1||x->a==2)){next=child(n,(uint16_t)x->a);f.cursor=1;}}else if(n.type==N_SWITCH_ENUM){if(f.cursor==0&&x&&x->a&&x->a-1<n.child_count){next=child(n,(uint16_t)(x->a-1));f.cursor=1;}}else if(n.type==N_PARALLEL){uint32_t running=x?(x->b&RUNNING_MASK16):0;while(f.cursor<n.child_count&&!(running&(1u<<f.cursor)))++f.cursor;if(f.cursor<n.child_count)next=child(n,f.cursor++);}else if(is_decorator(n.type)){if(f.cursor==0&&n.child_count==1){next=child(n,0);f.cursor=1;}}
        if(next>=0){if(hp>=depth_cap){fault(1);return;}hs[hp++]={(uint16_t)next,0,0,{0,0,0}};continue;}bool was=active(n);if(x){x->a=0;x->b=0;}if(was&&n.lifecycle_slot!=NONE_U16&&n.lifecycle_slot<p.n_lifecycles){const auto&lc=p.lifecycles[n.lifecycle_slot];effects(lc.on_halted_start,lc.on_halted_count);}if(was)emit_trace(f.node,SKIPPED,2);--hp;}}
    BKG4_VM_HD bool push(uint16_t i){++visits;if(visits>p.instruction_budget||sp>=depth_cap||i>=p.n_nodes){fault(i>=p.n_nodes?3:1);last=FAILURE;have_last=true;return false;}stack[sp++]={i,0,0,n_intents,0,0};have_last=false;return true;}
    BKG4_VM_HD void finish(Status r,uint8_t event=0){EvalFrame4 fr=stack[sp-1];uint16_t i=fr.node;const Node4&n=p.nodes[i];if(r==RUNNING)set_active(n,true);else{set_active(n,false);if(n.lifecycle_slot!=NONE_U16&&n.lifecycle_slot<p.n_lifecycles){const auto&lc=p.lifecycles[n.lifecycle_slot];if(r==SUCCESS){effects(lc.on_success_start,lc.on_success_count);effects(lc.post_start,lc.post_count);}else if(r==FAILURE){effects(lc.on_failure_start,lc.on_failure_count);effects(lc.post_start,lc.post_count);}}}if(faulted()||r==FAILURE||r==SKIPPED)n_intents=fr.intent_start<n_intents?fr.intent_start:n_intents;emit_trace(i,r,event);--sp;last=r;have_last=true;}
    BKG4_VM_HD void emit(uint16_t key,uint16_t source,uint8_t op,uint8_t arg,uint8_t subject,uint16_t qty){uint16_t order=emit_seq++;if(intent_cap==0||n_intents>=intent_cap){fault(4);return;}intents[n_intents++]={key,source,order,qty,op,arg,subject,0};}
    BKG4_VM_HD void push_parallel_next(EvalFrame4&f,const Node4&n){NodeState4*x=st(n);if(!x){while(f.cursor<n.child_count){push(child(n,f.cursor));return;}finish(FAILURE);return;}uint32_t success=x->a&0xFFFFu,failure=(x->a>>16)&0xFFFFu;while(f.cursor<n.child_count){uint32_t bit=1u<<f.cursor;if(!(success&bit)&&!(failure&bit)){push(child(n,f.cursor));return;}++f.cursor;}uint32_t running=x->b&RUNNING_MASK16;if(running){finish(RUNNING);return;}int sc=0,fc=0;for(uint16_t i=0;i<n.child_count;++i){sc+=(success>>i)&1u;fc+=(failure>>i)&1u;}bool all_skipped=sc==0&&fc==0&&f.skipped==n.child_count;x->a=x->b=0;finish(all_skipped?SKIPPED:FAILURE);}
    BKG4_VM_HD Status run(uint16_t root=0){push(root);while(sp>0){if(faulted()){n_intents=0;sp=0;last=FAILURE;have_last=true;break;}EvalFrame4&f=stack[sp-1];const Node4&n=p.nodes[f.node];NodeState4*x=st(n);if(f.phase==0){if(n.lifecycle_slot!=NONE_U16){if(n.lifecycle_slot>=p.n_lifecycles){fault(3);finish(FAILURE);continue;}const auto&lc=p.lifecycles[n.lifecycle_slot];bool act=active(n);if(!act){if(lc.failure_if!=NONE_U16&&pred(lc.failure_if)){finish(FAILURE,1);continue;}if(lc.success_if!=NONE_U16&&pred(lc.success_if)){finish(SUCCESS,1);continue;}if(lc.skip_if!=NONE_U16&&pred(lc.skip_if)){finish(SKIPPED,1);continue;}if(lc.while_pred!=NONE_U16&&!pred(lc.while_pred)){finish(SKIPPED,1);continue;}}else if(lc.while_pred!=NONE_U16&&!pred(lc.while_pred)){halt_node(f.node);finish(SKIPPED,3);continue;}}
        switch(n.type){
        case N_SEQUENCE:f.cursor=x?(uint16_t)x->a:0;f.phase=1;if(f.cursor>=n.child_count){if(x)x->a=0;finish(SUCCESS);}else push(child(n,f.cursor));break;
        case N_SELECTOR:f.cursor=x?(uint16_t)x->a:0;f.phase=2;if(f.cursor>=n.child_count){if(x)x->a=0;finish(FAILURE);}else push(child(n,f.cursor));break;
        case N_REACTIVE_SEQUENCE:f.aux=x?(uint16_t)x->a:0;f.cursor=0;f.phase=3;push(child(n,0));break;
        case N_REACTIVE_SELECTOR:f.aux=x?(uint16_t)x->a:0;f.cursor=0;f.phase=4;push(child(n,0));break;
        case N_IF:f.aux=x?(uint16_t)x->a:0;if(!x||x->a==0){f.phase=5;push(child(n,0));}else{f.phase=6;push(child(n,(uint16_t)x->a));}break;
        case N_REACTIVE_IF:f.phase=13;push(child(n,0));break;
        case N_SWITCH_ENUM:{int32_t v=s.blackboard[n.p0];uint16_t sel=(v>=0&&v<n.p2)?(uint16_t)v:(uint16_t)n.p2;int old=x&&active(n)&&x->a?(int)x->a-1:-1;if(old>=0&&old!=(int)sel)halt_node(child(n,(uint16_t)old));f.aux=sel;f.phase=14;push(child(n,sel));break;}
        case N_PARALLEL:f.cursor=0;f.skipped=0;f.phase=7;if(!x){push(child(n,0));}else push_parallel_next(f,n);break;
        case N_INVERT:case N_FORCE_SUCCESS:case N_FORCE_FAILURE:f.phase=8;push(child(n,0));break;
        case N_TURN_TIMEOUT:if(!x){f.phase=9;push(child(n,0));}else{int32_t now=feature(F_STEP);if(!active(n))x->a=(uint32_t)now;else if(now-(int32_t)x->a>=n.p0){halt_node(child(n,0));x->a=0;finish(FAILURE);break;}f.phase=9;push(child(n,0));}break;
        case N_RETRY:f.phase=10;push(child(n,0));break;case N_REPEAT:f.phase=11;push(child(n,0));break;case N_KEEP_RUNNING_UNTIL_FAILURE:f.phase=12;push(child(n,0));break;
        case N_SUCCESS:finish(SUCCESS);break;case N_FAILURE:finish(FAILURE);break;
        case N_CONDITION:finish(compare_i32(value((uint16_t)n.p0),value((uint16_t)n.p1),n.flags)?SUCCESS:FAILURE);break;
        case N_SET_BLACKBOARD:{int32_t v=value((uint16_t)n.p1);const auto&b=p.bb[n.p0];if(!valid_bb_value(b.type,v)||(b.enum_tag&&(v<0||v>=b.enum_size))){fault(5);finish(FAILURE);}else{s.blackboard[n.p0]=v;finish(SUCCESS);}break;}
        case N_ADD_BLACKBOARD:{bool ok=false;int32_t v=clamp_i32_add(s.blackboard[n.p0],value((uint16_t)n.p1),ok);if(!ok){fault(2);finish(FAILURE);}else{s.blackboard[n.p0]=v;finish(SUCCESS);}break;}
        case N_EMIT_UNIT_INTENT:{int unit=value((uint16_t)n.p0),item=value((uint16_t)n.p2),qty=value((uint16_t)n.p3);bool valid=unit>=0&&unit<feature(F_OWN_UNITS);if(n.p1==U_PLANT)valid=valid&&item>=0&&item<5&&qty==1;else if(n.p1==U_PICKUP||n.p1==U_PLACE)valid=valid&&item>=0&&item<MAX_ITEMS&&qty>=1&&qty<=32767;else valid=valid&&item==-1&&qty==1;if(!valid)finish(FAILURE);else{emit((uint16_t)n.aux,f.node,(uint8_t)n.p1,(uint8_t)(item<0?0:item),(uint8_t)unit,(uint16_t)qty);finish(SUCCESS);}break;}
        case N_EMIT_MARKET_INTENT:{int item=value((uint16_t)n.p1),qty=value((uint16_t)n.p2);bool valid=qty>=1&&qty<=65535;if(n.p0==M_HIRE||n.p0==M_BUY_LAND)valid=valid&&item==-1&&qty==1;else if(n.p0==M_BUY_SEED)valid=valid&&item>=0&&item<5;else if(n.p0==M_BUY_PRODUCT)valid=valid&&(item==0||item==8);else if(n.p0==M_BUY_ANIMAL)valid=valid&&item>=9&&item<=11;else if(n.p0==M_SELL)valid=valid&&item>=0&&item<9;else valid=false;if(!valid)finish(FAILURE);else{emit((uint16_t)n.aux,f.node,(uint8_t)n.p0,(uint8_t)(item<0?0:item),INTENT_MARKET_BIT,(uint16_t)qty);finish(SUCCESS);}break;}
        case N_WAIT_TURNS:{if(!x){fault(3);finish(FAILURE);break;}int32_t now=feature(F_STEP);if(!active(n)){x->a=(uint32_t)now;finish(RUNNING);}else if(now-(int32_t)x->a>=n.p0){x->a=0;finish(SUCCESS);}else finish(RUNNING);break;}
        case N_MOVE_UNIT_TO:{int unit=value((uint16_t)n.p0),target=value((uint16_t)n.p1);if(unit<0||target<0||target>=100||unit>=feature(F_OWN_UNITS)){finish(FAILURE);break;}int px=feature(F_OWN_WORKER_X,unit),py=feature(F_OWN_WORKER_Y,unit),tx=target%10,ty=target/10;if(px==tx&&py==ty){finish(SUCCESS);break;}int op=px<tx?U_EAST:px>tx?U_WEST:py<ty?U_SOUTH:U_NORTH;emit((uint16_t)n.aux,f.node,(uint8_t)op,0,(uint8_t)unit,1);finish(RUNNING);break;}
        default:fault(3);finish(FAILURE);break;}continue;}
        if(!have_last){fault(3);finish(FAILURE);continue;}Status r=last;have_last=false;switch(f.phase){
        case 1:if(r==SUCCESS||r==SKIPPED){++f.cursor;if(x)x->a=f.cursor;if(f.cursor>=n.child_count){if(x)x->a=0;finish(SUCCESS);}else push(child(n,f.cursor));}else if(r==RUNNING){if(!x){fault(3);finish(FAILURE);}else{x->a=f.cursor;finish(RUNNING);}}else{if(x)x->a=0;finish(FAILURE);}break;
        case 2:if(r==FAILURE||r==SKIPPED){++f.cursor;if(x)x->a=f.cursor;if(f.cursor>=n.child_count){if(x)x->a=0;finish(FAILURE);}else push(child(n,f.cursor));}else if(r==RUNNING){if(!x){fault(3);finish(FAILURE);}else{x->a=f.cursor;finish(RUNNING);}}else{if(x)x->a=0;finish(SUCCESS);}break;
        case 3:case 4:{int prev=f.aux?(int)f.aux-1:-1;if(prev==(int)f.cursor){f.aux=0;prev=-1;}bool advance=f.phase==3?(r==SUCCESS||r==SKIPPED):(r==FAILURE||r==SKIPPED);if(advance){++f.cursor;if(f.cursor>=n.child_count){if(prev>=0)halt_node(child(n,(uint16_t)prev));if(x)x->a=0;finish(f.phase==3?SUCCESS:FAILURE);}else push(child(n,f.cursor));}else{if(prev>=0)halt_node(child(n,(uint16_t)prev));if(r==RUNNING){if(!x){fault(3);finish(FAILURE);break;}x->a=f.cursor+1;}else if(x)x->a=0;finish(r);}break;}
        case 5:if(r==RUNNING){if(!x){fault(3);finish(FAILURE);}else finish(RUNNING);}else if(r==SKIPPED){if(x)x->a=0;finish(SKIPPED);}else{uint16_t ph=r==SUCCESS?1:2;if(x)x->a=ph;f.aux=ph;f.phase=6;push(child(n,ph));}break;
        case 6:if(r==RUNNING&&!x){fault(3);finish(FAILURE);}else{if(r!=RUNNING&&x)x->a=0;finish(r);}break;
        case 7:{uint16_t idx=f.cursor;uint32_t bit=1u<<idx;if(!x){if(r==RUNNING){fault(3);finish(FAILURE);break;}uint16_t sc=f.aux&0xFFu,fc=(f.aux>>8)&0xFFu;if(r==SUCCESS)++sc;else if(r==FAILURE)++fc;else ++f.skipped;f.aux=(uint16_t)(sc|(fc<<8));int possible=(int)n.child_count-(int)fc-(int)f.skipped;if(sc>=n.p0||fc>=n.p1||possible<n.p0){finish(sc>=n.p0?SUCCESS:FAILURE);break;}++f.cursor;if(f.cursor>=n.child_count)finish(sc>=n.p0?SUCCESS:FAILURE);else push(child(n,f.cursor));break;}uint32_t success=x->a&0xFFFFu,failure=(x->a>>16)&0xFFFFu,running=x->b&RUNNING_MASK16;if(r==SUCCESS){success|=bit;running&=~bit;}else if(r==FAILURE){failure|=bit;running&=~bit;}else if(r==RUNNING)running|=bit;else{++f.skipped;running&=~bit;}x->a=success|(failure<<16);x->b=(x->b&ACTIVE_BIT)|running;int sc=0,fc=0;for(uint16_t k=0;k<n.child_count;++k){sc+=(success>>k)&1u;fc+=(failure>>k)&1u;}int possible=(int)n.child_count-fc-(int)f.skipped;if(sc>=n.p0||fc>=n.p1||possible<n.p0){for(uint16_t k=0;k<n.child_count;++k)if(running&(1u<<k))halt_node(child(n,k));Status out=sc>=n.p0?SUCCESS:FAILURE;x->a=x->b=0;finish(out);}else{++f.cursor;push_parallel_next(f,n);}break;}
        case 8:if(n.type==N_INVERT)finish(r==SUCCESS?FAILURE:r==FAILURE?SUCCESS:r);else if(r==RUNNING||r==SKIPPED)finish(r);else finish(n.type==N_FORCE_SUCCESS?SUCCESS:FAILURE);break;
        case 9:if(r==RUNNING&&!x){fault(3);finish(FAILURE);}else{if(r!=RUNNING&&x)x->a=0;finish(r);}break;
        case 10:if(r==SUCCESS){if(x)x->a=0;finish(SUCCESS);}else if(r==FAILURE){if(n.p0<=1){finish(FAILURE);}else if(!x){fault(3);finish(FAILURE);}else{++x->a;if(x->a>=(uint32_t)n.p0){x->a=0;finish(FAILURE);}else finish(RUNNING);}}else finish(r);break;
        case 11:if(r==FAILURE){if(x)x->a=0;finish(FAILURE);}else if(r==SUCCESS){if(n.p0<=1){finish(SUCCESS);}else if(!x){fault(3);finish(FAILURE);}else{++x->a;if(x->a>=(uint32_t)n.p0){x->a=0;finish(SUCCESS);}else finish(RUNNING);}}else finish(r);break;
        case 12:finish(r==FAILURE?SUCCESS:r==SUCCESS?RUNNING:r);break;
        case 13:{if(r==RUNNING){fault(3);finish(FAILURE);break;}if(r==SKIPPED){finish(SKIPPED);break;}uint16_t sel=r==SUCCESS?1:2;int old=x&&active(n)?(int)x->a:0;if(old&&old!=(int)sel)halt_node(child(n,(uint16_t)old));f.aux=sel;f.phase=15;push(child(n,sel));break;}
        case 14:if(r==RUNNING){if(!x){fault(3);finish(FAILURE);}else{x->a=f.aux+1;finish(RUNNING);}}else{if(x)x->a=0;finish(r);}break;
        case 15:if(r==RUNNING){if(!x){fault(3);finish(FAILURE);}else{x->a=f.aux;finish(RUNNING);}}else{if(x)x->a=0;finish(r);}break;
        default:fault(3);finish(FAILURE);break;}}
        return last;}
};

template<class Obs>
BKG4_VM_NOINLINE Status tick_core4(const PolicyView4&p,RuntimeView4 s,const Obs&obs,int max_market_orders,ActionResult4&out,TraceBuffer4*trace,DeferredIntent4*intent_mem,uint16_t intent_cap,EvalFrame4*eval_mem,HaltFrame4*halt_mem,uint16_t depth_cap){
    int units=obs.feature(F_OWN_UNITS,0,0);if(units<0)units=0;if(units>MAX_UNITS)units=MAX_UNITS;out.n_units=units;for(int i=0;i<units;++i)out.units[i]=UnitIntentAction4{};out.n_market=0;if(trace)trace->count=0;EvaluatorCore4<Obs>ev(p,s,obs,trace,intent_mem,intent_cap,eval_mem,halt_mem,depth_cap);Status raw=FAILURE;if(s.meta&&!s.meta->fault)raw=ev.run(0);if(!s.meta||s.meta->fault||raw==FAILURE||raw==SKIPPED)ev.n_intents=0;UnitClaim4 uc[MAX_UNITS]{};MarketClaim4 mc[MAX_MARKET_INTENTS]{};uint16_t nm=0;if(s.meta&&!s.meta->fault&&(raw==SUCCESS||raw==RUNNING)){for(uint16_t i=0;i<ev.n_intents;++i){const auto&z=ev.intents[i];if(!(z.subject_kind&INTENT_MARKET_BIT)){uint8_t u=z.subject_kind;if(u>=MAX_UNITS)continue;auto&old=uc[u];if(!old.present||z.priority_key>old.priority_key||(z.priority_key==old.priority_key&&z.order<old.order)){old.present=1;old.op=z.op;old.arg=z.arg;old.quantity=(int16_t)z.quantity;old.priority_key=z.priority_key;old.order=z.order;old.reserved=0;}}else{MarketClaim4 cand{z.priority_key,z.order,z.op,z.arg,0,(int32_t)z.quantity};int pos=nm;if(pos<MAX_MARKET_INTENTS)++nm;else{const auto&w=mc[MAX_MARKET_INTENTS-1];if(cand.priority_key<w.priority_key||(cand.priority_key==w.priority_key&&cand.order>=w.order))continue;pos=MAX_MARKET_INTENTS-1;}while(pos>0){const auto&prev=mc[pos-1];if(cand.priority_key<prev.priority_key||(cand.priority_key==prev.priority_key&&cand.order>=prev.order))break;mc[pos]=prev;--pos;}mc[pos]=cand;}}}
    for(int i=0;i<units;++i){if(uc[i].present)out.units[i]={uc[i].op,uc[i].arg,uc[i].quantity};}
    int cap=max_market_orders;if(cap<0)cap=0;if(cap>MAX_MARKET_INTENTS)cap=MAX_MARKET_INTENTS;
    out.n_market=nm<cap?nm:cap;for(int i=0;i<out.n_market;++i)out.market[i]={mc[i].op,mc[i].item,0,mc[i].quantity};
    Status status=!s.meta||s.meta->fault?FAILURE:raw==SKIPPED?SUCCESS:raw;if(s.meta)++s.meta->ticks;return status;
}

// Only scratch layout is specialized. The interpreter body above is instantiated once
// per observation type, avoiding a 24-copy CUDA codegen explosion while preserving
// compile-time local-memory sizing for each journal/depth class.
template<class Obs,int IC,int DC>
BKG4_VM_HD Status tick_specialized4(const PolicyView4&p,RuntimeView4 s,const Obs&obs,int max_orders,ActionResult4&out,TraceBuffer4*t=nullptr){DeferredIntent4 intents[IC>0?IC:1];EvalFrame4 eval[DC];HaltFrame4 halt[DC];return tick_core4(p,s,obs,max_orders,out,t,intents,IC,eval,halt,DC);}

template<class Obs,int IC>
BKG4_VM_HD Status tick_depth4(const PolicyView4&p,RuntimeView4 s,const Obs&obs,int max_orders,ActionResult4&out,TraceBuffer4*t){if(p.max_depth<=16)return tick_specialized4<Obs,IC,16>(p,s,obs,max_orders,out,t);if(p.max_depth<=24)return tick_specialized4<Obs,IC,24>(p,s,obs,max_orders,out,t);if(p.max_depth<=32)return tick_specialized4<Obs,IC,32>(p,s,obs,max_orders,out,t);return tick_specialized4<Obs,IC,48>(p,s,obs,max_orders,out,t);}

template<class Obs>
BKG4_VM_HD Status tick4(const PolicyView4&p,RuntimeView4 s,const Obs&obs,int max_orders,ActionResult4&out,TraceBuffer4*t=nullptr){if(p.max_live_intents==0)return tick_depth4<Obs,0>(p,s,obs,max_orders,out,t);if(p.max_live_intents<=8)return tick_depth4<Obs,8>(p,s,obs,max_orders,out,t);if(p.max_live_intents<=16)return tick_depth4<Obs,16>(p,s,obs,max_orders,out,t);if(p.max_live_intents<=32)return tick_depth4<Obs,32>(p,s,obs,max_orders,out,t);if(p.max_live_intents<=64)return tick_depth4<Obs,64>(p,s,obs,max_orders,out,t);return tick_depth4<Obs,128>(p,s,obs,max_orders,out,t);}

} // namespace bonsaikg4

#undef BKG4_VM_HD
#undef BKG4_VM_NOINLINE
