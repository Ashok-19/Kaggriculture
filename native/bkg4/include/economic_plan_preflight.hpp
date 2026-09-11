#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define EP_HD __host__ __device__ inline
#else
#define EP_HD inline
#endif

namespace economic_plan_preflight {

constexpr int BOARD=10, ITEMS=12, PRODUCTS=9, CROPS=5, ANIMALS=3;
constexpr int MAX_WORKERS=16, MAX_ORDERS=10, LAST_STEP=718;
enum Item : uint8_t { WHEAT=0,CARROT,TOMATO,STRAWBERRY,MELON,EGG,MILK,WOOL,FERTILIZER,GOOSE,COW,SHEEP };
enum TileKind : uint8_t { T_EMPTY=0,T_LOCKED,T_WEED,T_COOP,T_PASTURE,T_PLANT };
enum UnitOp : uint8_t { U_PASS=0,U_NORTH,U_SOUTH,U_EAST,U_WEST,U_PICKUP,U_DROP,U_PLACE,U_PLANT,U_WATER,U_HARVEST,U_FERTILIZE,U_DIG,U_BUILD_COOP,U_BUILD_PASTURE,U_FEED,U_COLLECT_FERTILIZER,U_CARE };
enum MarketOp : uint8_t { M_NONE=0,M_HIRE,M_BUY_LAND,M_BUY_SEED,M_BUY_PRODUCT,M_BUY_ANIMAL,M_SELL };
enum FeedMode : uint8_t { FEED_MAINTAIN=0, FEED_BUYBACK=1 };
enum SpatialMode : uint8_t { SP_SHED_DISTANCE=0, SP_ROW_MAJOR=1, SP_COLUMN_MAJOR=2, SP_QUADRANT_LOCAL=3 };
enum SalesMode : uint8_t { SELL_ALL=0, SELL_EXCESS=1, SELL_DAY_END=2 };
enum TaskKind : uint8_t { TK_NONE=0,TK_DEPOSIT,TK_FEED,TK_HARVEST,TK_WATER,TK_CARE,TK_PLACE_ANIMAL,TK_BUILD,TK_PLANT,TK_DIG,TK_COLLECT_FERT };

struct Tile {
    uint8_t kind=0, what=0, has_animal=0, watered=0, fed=0, cared=0, fertilizer=0;
    int8_t dry=0, yield=0;
    int16_t planted_day=0;
};
struct Worker {
    int8_t x=0,y=0;
    int16_t inv[ITEMS]{};
};
struct Observation {
    int step=0,day=0,hour=0,player=0;
    int own_money=0,opp_money=0,own_units=1,opp_units=1,own_land=1,opp_land=1,hires_today=0;
    int shed_total=0;
    int16_t shed[ITEMS]{}, seeds[CROPS]{};
    int32_t market_inventory[PRODUCTS]{}, price[PRODUCTS]{};
    Worker workers[MAX_WORKERS]{};
    Tile own[BOARD][BOARD]{}, opp[BOARD][BOARD]{};
};
struct UnitAction { uint8_t op=U_PASS,arg=0; int16_t n=1; };
struct Order { uint8_t op=M_NONE,item=0; int32_t n=0; };
struct Action { int n_units=1,n_orders=0; UnitAction units[MAX_WORKERS]{}; Order orders[MAX_ORDERS]{}; };

// Strategic plan only. All economic choices live here; the executor may not override them.
struct Plan {
    uint8_t cow_target=0,sheep_target=0,goose_target=0;
    uint8_t crop_target[CROPS]{};
    uint8_t land_target=1, labour_target=0;
    uint16_t cash_reserve=0;
    uint8_t feed_mode=FEED_MAINTAIN;
    // Task priority ranks, high value first: feed,water,care,harvest,expansion,plant.
    uint8_t priority[6]={250,230,180,240,200,210};
    uint8_t spatial_mode=SP_SHED_DISTANCE;
    uint8_t sales_mode=SELL_EXCESS;
    uint8_t duration=72;
};
constexpr int MAX_PLAN_STAGES=10;
struct Bundle {
    uint8_t n_stages=1;
    Plan stages[MAX_PLAN_STAGES]{};
};
struct TaskState {
    uint8_t kind=TK_NONE,arg=0;
    int8_t x=-1,y=-1;
    int16_t since=-1,deadline=-1;
};
struct State {
    int initialized=0,last_step=-1,commitment_start=0,commitment_stage=0;
    int previous_money=0,last_day_money=0,three_day_money=0;
    TaskState tasks[MAX_WORKERS]{};
    uint16_t failed_tasks=0,timeouts=0;
};

EP_HD int min2(int a,int b){return a<b?a:b;} EP_HD int max2(int a,int b){return a>b?a:b;}
EP_HD int abs2(int a){return a<0?-a:a;} EP_HD int dist(int x,int y,int tx,int ty){return abs2(x-tx)+abs2(y-ty);}
EP_HD bool shed_cell(int x,int y){return (x==4||x==5)&&(y==4||y==5);}
EP_HD void nearest_shed(int x,int y,int&tx,int&ty){const int xs[4]={4,5,4,5},ys[4]={4,4,5,5};int bd=999;tx=4;ty=4;for(int k=0;k<4;++k){int d=dist(x,y,xs[k],ys[k]);if(d<bd){bd=d;tx=xs[k];ty=ys[k];}}}
EP_HD UnitAction step_toward(int x,int y,int tx,int ty){UnitAction a{};if(x<tx)a.op=U_EAST;else if(x>tx)a.op=U_WEST;else if(y<ty)a.op=U_SOUTH;else if(y>ty)a.op=U_NORTH;return a;}
EP_HD int carried_total(const Observation&o,int u){int n=0;for(int i=0;i<ITEMS;++i)n+=o.workers[u].inv[i];return n;}
EP_HD int carried_item(const Observation&o,int item){int n=0;for(int u=0;u<o.own_units&&u<MAX_WORKERS;++u)n+=o.workers[u].inv[item];return n;}
EP_HD bool animal_item(int i){return i>=GOOSE&&i<=SHEEP;}
EP_HD int animal_count(const Observation&o,int item){int n=0;for(int y=0;y<BOARD;++y)for(int x=0;x<BOARD;++x){const Tile&t=o.own[y][x];if(t.has_animal&&t.what==item)++n;}return n;}
EP_HD int crop_count(const Observation&o,int crop){int n=0;for(int y=0;y<BOARD;++y)for(int x=0;x<BOARD;++x){const Tile&t=o.own[y][x];if(t.kind==T_PLANT&&t.what==crop)++n;}return n;}
EP_HD int desired_total_sites(const Plan&p){int n=p.cow_target+p.sheep_target+p.goose_target;for(int c=0;c<CROPS;++c)n+=p.crop_target[c];return n;}
EP_HD int spatial_key(const Plan&p,int x,int y){
    if(p.spatial_mode==SP_ROW_MAJOR)return y*BOARD+x;
    if(p.spatial_mode==SP_COLUMN_MAJOR)return x*BOARD+y;
    if(p.spatial_mode==SP_QUADRANT_LOCAL){int q=(y>=5?2:0)+(x>=5?1:0);return q*100+(y%5)*5+(x%5);}
    int d=dist(x,y,4,4);return d*128+y*BOARD+x;
}
EP_HD bool site_eligible(const Observation&o,int x,int y){return o.own[y][x].kind!=T_LOCKED && !shed_cell(x,y);}
EP_HD int site_rank(const Plan&p,const Observation&o,int x,int y){
    const int key=spatial_key(p,x,y);int r=0;
    for(int yy=0;yy<BOARD;++yy)for(int xx=0;xx<BOARD;++xx)if(site_eligible(o,xx,yy)){
        int k=spatial_key(p,xx,yy);if(k<key||(k==key&&(yy<y||(yy==y&&xx<x))))++r;
    }return r;
}
// category: 0 pasture-cow, 1 pasture-sheep, 2 coop-goose, 3..7 crops, 8 unused.
EP_HD int desired_category(const Plan&p,const Observation&o,int x,int y){
    if(!site_eligible(o,x,y))return 8;int r=site_rank(p,o,x,y);
    if(r<p.cow_target)return 0;r-=p.cow_target;
    if(r<p.sheep_target)return 1;r-=p.sheep_target;
    if(r<p.goose_target)return 2;r-=p.goose_target;
    for(int c=0;c<CROPS;++c){if(r<p.crop_target[c])return 3+c;r-=p.crop_target[c];}
    return 8;
}
EP_HD int task_priority(const Plan&p,uint8_t k){switch(k){case TK_FEED:return p.priority[0];case TK_WATER:return p.priority[1];case TK_CARE:return p.priority[2];case TK_HARVEST:return p.priority[3];case TK_BUILD:case TK_PLACE_ANIMAL:return p.priority[4];case TK_PLANT:return p.priority[5];case TK_DEPOSIT:return 255;case TK_DIG:return 245;case TK_COLLECT_FERT:return 120;default:return 0;}}
struct Candidate {int score=-(1<<29);int x=-1,y=-1;uint8_t kind=TK_NONE,arg=0;};
EP_HD void consider(Candidate&b,const Plan&p,int ux,int uy,int x,int y,uint8_t kind,uint8_t arg){int s=task_priority(p,kind)*256-dist(ux,uy,x,y)*4-spatial_key(p,x,y);if(s>b.score){b={s,x,y,kind,arg};}}
EP_HD bool reserved(const TaskState*assigned,int n,int x,int y,uint8_t kind){for(int i=0;i<n;++i)if(assigned[i].kind==kind&&assigned[i].x==x&&assigned[i].y==y)return true;return false;}
EP_HD int reserved_count(const TaskState*assigned,int n,uint8_t kind,uint8_t arg){int c=0;for(int i=0;i<n;++i)if(assigned[i].kind==kind&&assigned[i].arg==arg)++c;return c;}
EP_HD Candidate choose_task(const Plan&p,const Observation&o,int u,const TaskState*assigned,int n_assigned){
    Candidate b{};const Worker&w=o.workers[u];
    if(carried_total(o,u)>0&&o.hour>=20){int sx,sy;nearest_shed(w.x,w.y,sx,sy);return {255*256-dist(w.x,w.y,sx,sy)*4,sx,sy,TK_DEPOSIT,0};}
    for(int y=0;y<BOARD;++y)for(int x=0;x<BOARD;++x){const Tile&t=o.own[y][x];if(t.kind==T_LOCKED)continue;
        int dc=desired_category(p,o,x,y);
        if(t.kind==T_WEED){if(!reserved(assigned,n_assigned,x,y,TK_DIG))consider(b,p,w.x,w.y,x,y,TK_DIG,0);continue;}
        if(t.kind==T_PLANT){if(t.yield>0&&!reserved(assigned,n_assigned,x,y,TK_HARVEST))consider(b,p,w.x,w.y,x,y,TK_HARVEST,t.what);if(!t.watered&&!reserved(assigned,n_assigned,x,y,TK_WATER))consider(b,p,w.x,w.y,x,y,TK_WATER,t.what);continue;}
        if(t.kind==T_COOP||t.kind==T_PASTURE){if(t.has_animal){int feed_supply=o.shed[WHEAT]+carried_item(o,WHEAT)-reserved_count(assigned,n_assigned,TK_FEED,0);if(!t.fed&&feed_supply>0&&!reserved(assigned,n_assigned,x,y,TK_FEED))consider(b,p,w.x,w.y,x,y,TK_FEED,t.what);if(t.yield>0&&!reserved(assigned,n_assigned,x,y,TK_HARVEST))consider(b,p,w.x,w.y,x,y,TK_HARVEST,t.what);if(!t.cared&&!reserved(assigned,n_assigned,x,y,TK_CARE))consider(b,p,w.x,w.y,x,y,TK_CARE,t.what);if(t.fertilizer&&!reserved(assigned,n_assigned,x,y,TK_COLLECT_FERT))consider(b,p,w.x,w.y,x,y,TK_COLLECT_FERT,FERTILIZER);}else{
            uint8_t want=0;if(t.kind==T_COOP)want=GOOSE;else if(dc==1)want=SHEEP;else want=COW;
            int target=want==GOOSE?p.goose_target:(want==COW?p.cow_target:p.sheep_target);int supply=o.shed[want]+carried_item(o,want)-reserved_count(assigned,n_assigned,TK_PLACE_ANIMAL,want);if(animal_count(o,want)<target&&supply>0&&!reserved(assigned,n_assigned,x,y,TK_PLACE_ANIMAL))consider(b,p,w.x,w.y,x,y,TK_PLACE_ANIMAL,want);
        }continue;}
        if(t.kind==T_EMPTY){if(dc<=2&&!reserved(assigned,n_assigned,x,y,TK_BUILD))consider(b,p,w.x,w.y,x,y,TK_BUILD,(uint8_t)dc);else if(dc>=3&&dc<=7){uint8_t crop=(uint8_t)(dc-3);int available=o.seeds[crop]-reserved_count(assigned,n_assigned,TK_PLANT,crop);if(available>0&&!reserved(assigned,n_assigned,x,y,TK_PLANT))consider(b,p,w.x,w.y,x,y,TK_PLANT,crop);}}
    }return b;
}
EP_HD bool task_complete(const Observation&o,int u,const TaskState&t){
    if(t.kind==TK_NONE)return true;
    if(t.kind==TK_DEPOSIT)return carried_total(o,u)==0;
    if(t.x<0||t.x>=BOARD||t.y<0||t.y>=BOARD)return true;
    const Tile&z=o.own[t.y][t.x];
    switch(t.kind){
        case TK_FEED:return !z.has_animal||z.fed;
        case TK_HARVEST:return z.yield<=0;
        case TK_WATER:return z.kind!=T_PLANT||z.watered;
        case TK_CARE:return !z.has_animal||z.cared;
        case TK_PLACE_ANIMAL:return z.has_animal&&z.what==t.arg;
        case TK_BUILD:return t.arg==2?z.kind==T_COOP:z.kind==T_PASTURE;
        case TK_PLANT:return z.kind==T_PLANT&&z.what==t.arg;
        case TK_DIG:return z.kind==T_EMPTY;
        case TK_COLLECT_FERT:return !z.has_animal||!z.fertilizer;
        default:return true;
    }
}
EP_HD bool task_still_desired(const Plan&p,const Observation&o,const TaskState&t){
    if(t.kind==TK_NONE||t.kind==TK_DEPOSIT||t.kind==TK_FEED||t.kind==TK_HARVEST||t.kind==TK_WATER||t.kind==TK_CARE||t.kind==TK_COLLECT_FERT)return true;
    if(t.x<0||t.x>=BOARD||t.y<0||t.y>=BOARD)return false;
    const int dc=desired_category(p,o,t.x,t.y);
    if(t.kind==TK_BUILD)return (t.arg==0&&dc==0)||(t.arg==1&&dc==1)||(t.arg==2&&dc==2);
    if(t.kind==TK_PLANT)return dc==3+t.arg;
    if(t.kind==TK_PLACE_ANIMAL)return (t.arg==COW&&dc==0)||(t.arg==SHEEP&&dc==1)||(t.arg==GOOSE&&dc==2);
    if(t.kind==TK_DIG)return true;
    return true;
}
EP_HD Candidate candidate_from_task(const TaskState&t){Candidate c{};c.x=t.x;c.y=t.y;c.kind=t.kind;c.arg=t.arg;c.score=1<<28;return c;}
EP_HD void push_order(Action&a,uint8_t op,uint8_t item,int n){if(a.n_orders>=MAX_ORDERS||n<=0)return;a.orders[a.n_orders++]={op,item,n};}
EP_HD int land_cost(int land){return land<=1?1000:land==2?2000:land==3?4000:1<<29;}
EP_HD int hire_cost(int hires){int a=1,b=1;for(int i=0;i<hires;++i){int t=b;b=a+b;a=t;}return a;}
EP_HD void market(const Plan&p,const Observation&o,Action&a){
    a.n_orders=0;const bool terminal=o.step==LAST_STEP;
    int reserve[PRODUCTS]{};reserve[WHEAT]=animal_count(o,GOOSE)+animal_count(o,COW)+animal_count(o,SHEEP);
    if(p.feed_mode==FEED_MAINTAIN)reserve[WHEAT]*=2;
    bool sell_now=p.sales_mode==SELL_ALL||(p.sales_mode==SELL_EXCESS)||(p.sales_mode==SELL_DAY_END&&o.hour>=20)||terminal;
    if(sell_now){for(int it=0;it<PRODUCTS&&a.n_orders<MAX_ORDERS;++it){int q=o.shed[it]-(terminal?0:reserve[it]);if(q>0)push_order(a,M_SELL,it,q);}}
    if(terminal)return;
    int cash=o.own_money;
    if(o.own_land<p.land_target){int cost=land_cost(o.own_land);if(cash-cost>=p.cash_reserve){push_order(a,M_BUY_LAND,0,1);cash-=cost;}}
    int desired_units=min2(MAX_WORKERS,1+(int)p.labour_target);int h=o.hires_today;
    while(o.own_units+(h-o.hires_today)<desired_units&&a.n_orders<MAX_ORDERS){int cost=hire_cost(h);if(cash-cost<p.cash_reserve)break;push_order(a,M_HIRE,0,1);cash-=cost;++h;}
    for(int c=0;c<CROPS&&a.n_orders<MAX_ORDERS;++c){int q=(int)p.crop_target[c]-crop_count(o,c)-o.seeds[c];if(q>0){int unit_cost=c==0?10:c==1?20:c==2?50:c==3?100:80;int afford=max2(0,(cash-p.cash_reserve)/unit_cost);q=min2(q,afford);if(q>0){push_order(a,M_BUY_SEED,c,q);cash-=q*unit_cost;}}}
    const uint8_t items[3]={GOOSE,COW,SHEEP};const int costs[3]={300,400,500};const int targets[3]={p.goose_target,p.cow_target,p.sheep_target};
    for(int j=0;j<3&&a.n_orders<MAX_ORDERS;++j){int q=targets[j]-animal_count(o,items[j])-o.shed[items[j]]-carried_item(o,items[j]);if(q>0){int afford=max2(0,(cash-p.cash_reserve)/costs[j]);q=min2(q,afford);if(q>0){push_order(a,M_BUY_ANIMAL,items[j],q);cash-=q*costs[j];}}}
    int live=animal_count(o,GOOSE)+animal_count(o,COW)+animal_count(o,SHEEP);int feed_need=(p.feed_mode==FEED_MAINTAIN?2:1)*live;int have=o.shed[WHEAT]+carried_item(o,WHEAT);if(feed_need>have&&a.n_orders<MAX_ORDERS)push_order(a,M_BUY_PRODUCT,WHEAT,feed_need-have);
}
EP_HD UnitAction realize_task(const Observation&o,int u,const Candidate&t){
    UnitAction pass{};if(t.kind==TK_NONE)return pass;const Worker&w=o.workers[u];
    int need=-1,qty=1;if(t.kind==TK_FEED){need=WHEAT;qty=10;}else if(t.kind==TK_PLACE_ANIMAL){need=t.arg;qty=1;}
    if(need>=0&&w.inv[need]<=0){int sx,sy;nearest_shed(w.x,w.y,sx,sy);if(shed_cell(w.x,w.y)&&o.shed[need]>0){UnitAction a{};a.op=U_PICKUP;a.arg=need;a.n=min2(qty,(int)o.shed[need]);return a;}return step_toward(w.x,w.y,sx,sy);}
    if(t.kind==TK_DEPOSIT){if(shed_cell(w.x,w.y)){UnitAction a{};a.op=U_DROP;return a;}return step_toward(w.x,w.y,t.x,t.y);}
    if(w.x!=t.x||w.y!=t.y)return step_toward(w.x,w.y,t.x,t.y);
    UnitAction a{};switch(t.kind){case TK_FEED:a.op=U_FEED;break;case TK_HARVEST:a.op=U_HARVEST;break;case TK_WATER:a.op=U_WATER;break;case TK_CARE:a.op=U_CARE;break;case TK_PLACE_ANIMAL:a.op=U_PLACE;a.arg=t.arg;a.n=1;break;case TK_BUILD:a.op=t.arg==2?U_BUILD_COOP:U_BUILD_PASTURE;break;case TK_PLANT:a.op=U_PLANT;a.arg=t.arg;break;case TK_DIG:a.op=U_DIG;break;case TK_COLLECT_FERT:a.op=U_COLLECT_FERTILIZER;break;default:a.op=U_PASS;break;}return a;
}
EP_HD void reset(State&s,const Observation&o){s=State{};s.initialized=1;s.last_step=o.step-1;s.commitment_start=o.step;s.previous_money=o.own_money;s.last_day_money=o.own_money;s.three_day_money=o.own_money;}
EP_HD Action execute_plan(const Plan&p,const Observation&o,State&s){
    if(o.hour==0){if(o.day>0){if(o.day%3==0)s.three_day_money=s.last_day_money;s.last_day_money=o.own_money;}for(int u=0;u<MAX_WORKERS;++u)s.tasks[u]=TaskState{};}
    Action a{};a.n_units=min2(o.own_units,MAX_WORKERS);
    bool keep[MAX_WORKERS]{};TaskState reservations[MAX_WORKERS]{};int nr=0;
    for(int u=0;u<a.n_units;++u){const TaskState&old=s.tasks[u];if(old.kind==TK_NONE)continue;if(task_complete(o,u,old))continue;if(!task_still_desired(p,o,old)){++s.failed_tasks;continue;}if(o.step>old.deadline){++s.timeouts;++s.failed_tasks;continue;}keep[u]=true;reservations[nr++]=old;}
    for(int u=0;u<a.n_units;++u){Candidate c{};TaskState next{};
        if(keep[u]){next=s.tasks[u];c=candidate_from_task(next);}else{c=choose_task(p,o,u,reservations,nr);next.kind=c.kind;next.arg=c.arg;next.x=(int8_t)c.x;next.y=(int8_t)c.y;next.since=(int16_t)o.step;next.deadline=(int16_t)(o.step+24);if(next.kind!=TK_NONE&&nr<MAX_WORKERS)reservations[nr++]=next;}
        s.tasks[u]=next;a.units[u]=realize_task(o,u,c);
    }
    for(int u=a.n_units;u<MAX_WORKERS;++u)s.tasks[u]=TaskState{};
    market(p,o,a);s.previous_money=o.own_money;s.last_step=o.step;return a;
}
EP_HD Action tick(const Plan&p,const Observation&o,State&s){
    if(!s.initialized||o.step<=s.last_step)reset(s,o);
    if(o.step-s.commitment_start>=p.duration){s.commitment_start=o.step;++s.commitment_stage;}
    return execute_plan(p,o,s);
}
EP_HD Action tick(const Bundle&b,const Observation&o,State&s){
    if(!s.initialized||o.step<=s.last_step)reset(s,o);
    int n=b.n_stages<1?1:min2((int)b.n_stages,MAX_PLAN_STAGES);
    int idx=min2(s.commitment_stage,n-1);
    const Plan* p=&b.stages[idx];
    if(idx+1<n && o.step-s.commitment_start>=p->duration){s.commitment_start=o.step;++s.commitment_stage;idx=min2(s.commitment_stage,n-1);p=&b.stages[idx];}
    return execute_plan(*p,o,s);
}

} // namespace economic_plan_preflight
#undef EP_HD
