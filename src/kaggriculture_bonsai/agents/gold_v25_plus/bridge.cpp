// Native submission wrapper: frozen parent first, then a compiled BKG4 specialist.
#define gold_create parent_gold_create
#define gold_destroy parent_gold_destroy
#define gold_tick parent_gold_tick
#define gold_pending parent_gold_pending
#include "../gold_v25/bridge.cu"
#undef gold_create
#undef gold_destroy
#undef gold_tick
#undef gold_pending
#include "lead.hpp"

struct PlusHandle {
    void*parent=nullptr;
    bonsaikg4::OwnedPolicy4 gate;
    ~PlusHandle(){if(parent)parent_gold_destroy(parent);}
};
extern "C" void*gold_create(const char*controller,const char*routes,const char*hash,const char*arity,int gpu){
    try{
        if(gpu)throw std::runtime_error("CPU submission wrapper; use the CUDA evaluator for device execution");
        auto h=std::make_unique<PlusHandle>();h->parent=parent_gold_create(controller,routes,hash,arity,0);
        if(!h->parent)throw std::runtime_error(gold_v25::error);
        std::string path=controller;path=path.substr(0,path.find_last_of('/')+1)+"specialist.bkgc";
        h->gate=bonsaikg4::load_compiled4(path);
        if(h->gate.header.n_bb!=1||h->gate.header.n_state!=0)throw std::runtime_error("unexpected specialist footprint");
        return h.release();
    }catch(const std::exception&e){gold_v25::error=e.what();return nullptr;}
}
extern "C" void gold_destroy(void*p){delete static_cast<PlusHandle*>(p);}
extern "C" int gold_pending(void*p,int player,int*buffer,int capacity){return parent_gold_pending(static_cast<PlusHandle*>(p)->parent,player,buffer,capacity);}
extern "C" int gold_tick(void*p,const gold_v25::Observation*o,gold_v25::Result*out){
    auto*h=static_cast<PlusHandle*>(p);
    if(parent_gold_tick(h->parent,o,out))return -1;
    auto gate=h->gate.view();int32_t bb[1]{};bonsaikg4::RuntimeMeta4 meta{};
    bonsaikg4::RuntimeView4 state{bb,nullptr,&meta};bonsaikg4::reset_runtime(gate,state);
    bonsaikg4::ActionResult4 ignored{};bonsaikg4::tick4(gate,state,*o,0,ignored);
    if(meta.fault){gold_v25::error="specialist BKG4 fault";return -1;}
    if(!bb[0])return 0;
    if(bb[0]==33||bb[0]==65){
        auto*parent=static_cast<gold_v25::Handle*>(h->parent);
        gold_v25_plus::lead(out->action,*o,{parent->pack.actions.data(),parent->arity.data()},out->route,bb[0]==33?2:3);
    }
    auto&a=out->action;int projected[12];gold_v25::project(a,*o,projected);
    // The original v1 gate used BOOL-like 1; retain its frozen >=708 semantics.
    if(bb[0]==5||o->step>=708){
        a.n_market=0;
        for(int k=0;k<9;++k)if(projected[k]>0)a.market[a.n_market++]={6,k,projected[k]};
    }else for(int i=0;i<a.n_market;++i)if(a.market[i].op!=0&&a.market[i].op!=6)return 0;
    if(bb[0]&16){
        for(int k=1;k<8;++k)if(projected[k]>0){
            bool found=false;
            for(int i=0;i<a.n_market;++i)if(a.market[i].op==6&&a.market[i].item==k){a.market[i].n=projected[k];found=true;}
            if(!found&&a.n_market<10)a.market[a.n_market++]={6,k,projected[k]};
        }
    }
    for(int i=1;i<a.n_market;++i){auto order=a.market[i];int j=i;
        int64_t value=order.op==6?(int64_t)o->prices[order.item]*gold_v25::low(projected[order.item],order.n):0;
        while(j>0){auto previous=a.market[j-1];int64_t prior=previous.op==6?(int64_t)o->prices[previous.item]*gold_v25::low(projected[previous.item],previous.n):0;
            if(prior>=value)break;
            a.market[j]=previous;--j;}a.market[j]=order;}
    return 0;
}
