#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace kagcuda;

__global__ void gp_eval_kernel(Config base, DeviceTables tables,
                               const GPProgram* programs, int n_programs,
                               const Action* routes, const GPRouteDesc* descs, int n_routes,
                               const GPManager* manager_bank, int n_managers,
                               int opponent_route, int candidate_seat,
                               uint64_t seed_start, const uint64_t* seed_values, int n_seeds,
                               Sim* sims, double* rewards) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_programs * n_seeds;
    if (id >= total) return;
    const int pi = id / n_seeds;
    const int si = id - pi * n_seeds;
    Config cfg = base;
    cfg.seed = seed_values ? seed_values[si] : seed_start + static_cast<uint64_t>(si);
    Sim& sim = sims[id];
    sim.init(cfg,&tables);
    const GPProgram& prog = programs[pi];
    const GPRouteDesc od = descs[opponent_route];
    while (!sim.st.done) {
        const int step = sim.st.step;
        const int opp_seat = 1-candidate_seat;
        const ObservationView obs = make_observation(sim,candidate_seat);
        Action cand = gp_act_with_managers(prog,obs,routes,descs,n_routes,manager_bank,n_managers);
        Action opp = gp_copy_action(gp_route_at(routes,od,step),sim.st.farms[opp_seat].n_units);
        if (candidate_seat == 0)
            sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else
            sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    rewards[id*2] = sim.st.farms[0].money;
    rewards[id*2+1] = sim.st.farms[1].money;
}

struct Fitness {
    int wins = 0;
    int ties = 0;
    int cases = 0;
    double floor = DBL_MAX;
    double sum = 0.0;
    // Host-side robustness summaries over opponent x seat cohorts. Each cohort
    // is averaged across the current seed block before these are computed.
    double tail_mean = -DBL_MAX;       // mean of weakest quartile of opponent-seat cohorts
    double worst_cohort = -DBL_MAX;    // single weakest opponent-seat cohort mean
    double seed_tail = -DBL_MAX;       // mean of weakest quartile of seed-regime means
    double worst_seed = -DBL_MAX;       // weakest seed mean across all opponent-seat cases
};

static bool GP_ROBUST_FITNESS = true;

static double mean(const Fitness& f) { return f.cases ? f.sum / f.cases : 0.0; }

static bool fitter(const Fitness& a, const Fitness& b) {
    if (!GP_ROBUST_FITNESS) {
        if (a.wins != b.wins) return a.wins > b.wins;
        if (a.ties != b.ties) return a.ties < b.ties;
        if (a.floor != b.floor) return a.floor > b.floor;
        return a.sum > b.sum;
    }
    // Before the policy is competitive, improve the weakest population slice
    // rather than buying a few wins with catastrophic regressions elsewhere.
    // Once two candidates have comparable lower-tail quality, global margin and
    // then actual win conversion break the tie.
    const double aguard=a.tail_mean<a.seed_tail?a.tail_mean:a.seed_tail;
    const double bguard=b.tail_mean<b.seed_tail?b.tail_mean:b.seed_tail;
    if (aguard != bguard) return aguard > bguard;
    const double abl=a.tail_mean+a.seed_tail, bbl=b.tail_mean+b.seed_tail;
    if (abl != bbl) return abl > bbl;
    const double am=mean(a), bm=mean(b);
    if (am != bm) return am > bm;
    if (a.wins != b.wins) return a.wins > b.wins;
    if (a.worst_cohort != b.worst_cohort) return a.worst_cohort > b.worst_cohort;
    if (a.worst_seed != b.worst_seed) return a.worst_seed > b.worst_seed;
    if (a.floor != b.floor) return a.floor > b.floor;
    return a.ties < b.ties;
}

static std::vector<std::string> read_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open list: "+path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in,line)) {
        const auto hash=line.find('#'); if (hash!=std::string::npos) line.resize(hash);
        const auto a=line.find_first_not_of(" \t\r\n"); if (a==std::string::npos) continue;
        const auto b=line.find_last_not_of(" \t\r\n");
        out.push_back(line.substr(a,b-a+1));
    }
    return out;
}

struct HostRng {
    uint64_t s;
    explicit HostRng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    uint64_t next() {
        uint64_t x=s; x^=x>>12; x^=x<<25; x^=x>>27; s=x;
        return x*2685821657736338717ull;
    }
};

static int ri(HostRng& r, int lo, int hi) {
    if (hi <= lo) return lo;
    const uint64_t span=static_cast<uint64_t>(hi-lo+1);
    return lo+static_cast<int>(r.next()%span);
}

static GPCond random_cond(HostRng& r) {
    GPCond c{};
    const int choice=ri(r,0,10);
    if (choice<=2) {
        c.feature=static_cast<uint8_t>(GPF_FIRST_SHOP+choice);
        c.cmp=GPC_EQ; c.value=ri(r,0,7);
    } else if (choice==3) {
        c.feature=GPF_STEP; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE;
        static const int steps[]={72,96,120,144,168,192,216,240,288,360,480,576,648};
        c.value=steps[ri(r,0,static_cast<int>(sizeof(steps)/sizeof(steps[0]))-1)];
    } else if (choice==4 || choice==5) {
        c.feature=choice==4?GPF_OWN_MONEY:GPF_OPP_MONEY; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE;
        c.value=ri(r,0,20)*500;
    } else if (choice==6 || choice==7) {
        c.feature=choice==6?GPF_PRICE:GPF_MARKET_INV; c.item=ri(r,0,N_PRODUCTS-1);
        c.cmp=ri(r,0,1)?GPC_GE:GPC_LE;
        c.value=choice==6 ? ri(r,1,400) : ri(r,9600,10400);
    } else if (choice==8) {
        c.feature=GPF_SHED; c.item=ri(r,0,N_PRODUCTS-1); c.cmp=GPC_GE; c.value=ri(r,0,60);
    } else if (choice==9) {
        c.feature=GPF_UNITS; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,1,15);
    } else {
        c.feature=GPF_LAND; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,1,4);
    }
    return c;
}

static bool GP_MANAGER_ONLY = false;
static bool GP_ROUTER_ONLY = false;
static int GP_ROUTER_EXPERTS = 0;
static int GP_BOOTSTRAP_ROUTE = -1;
static int GP_BOOTSTRAP_STEP = 0;
static int GP_MANAGER_UNIT_ROUTE = -1;
static bool GP_PLANNED_MARKET = false;

static void set_manager_program(GPProgram& p,const GPManager& m) {
    p=GPProgram{};p.manager=m;
    if(GP_BOOTSTRAP_ROUTE>=0 && GP_BOOTSTRAP_STEP>0){
        p.n_rules=2;
        p.rules[0]=GPRule{};p.rules[0].n_conds=1;
        p.rules[0].conds[0].feature=GPF_STEP;p.rules[0].conds[0].cmp=GPC_LE;p.rules[0].conds[0].value=GP_BOOTSTRAP_STEP-1;
        p.rules[0].leaf={GPL_ROUTE,(uint8_t)GP_BOOTSTRAP_ROUTE,(uint8_t)GP_BOOTSTRAP_ROUTE,0};
        p.rules[1]=GPRule{};
        p.rules[1].leaf=GP_MANAGER_UNIT_ROUTE>=0?GPLeaf{(uint8_t)(GP_PLANNED_MARKET?GPL_ROUTE_UNITS_PLANNED_MARKET:GPL_ROUTE_UNITS_MANAGER_MARKET),(uint8_t)GP_MANAGER_UNIT_ROUTE,(uint8_t)GP_MANAGER_UNIT_ROUTE,0}:GPLeaf{GPL_MANAGER,0,0,0};
    }else{
        p.n_rules=1;p.rules[0]=GPRule{};
        p.rules[0].leaf=GP_MANAGER_UNIT_ROUTE>=0?GPLeaf{(uint8_t)(GP_PLANNED_MARKET?GPL_ROUTE_UNITS_PLANNED_MARKET:GPL_ROUTE_UNITS_MANAGER_MARKET),(uint8_t)GP_MANAGER_UNIT_ROUTE,(uint8_t)GP_MANAGER_UNIT_ROUTE,0}:GPLeaf{GPL_MANAGER,0,0,0};
    }
}

static GPLeaf random_leaf(HostRng& r, int n_routes) {
    GPLeaf l{};
    // Lane B can be isolated so a new executable manager is optimized without
    // mature replay leaves dominating selection before the new meta is viable.
    if (GP_MANAGER_ONLY) l.kind = GPL_MANAGER;
    else {
        const int x=ri(r,0,99);
        l.kind = x<10?GPL_ROUTE:(x<35?GPL_HYBRID:(x<62?GPL_ROUTE_UNITS_MANAGER_MARKET:(x<88?GPL_MANAGER:GPL_MANAGER_UNITS_ROUTE_MARKET)));
    }
    l.unit_route=static_cast<uint8_t>(ri(r,0,n_routes-1));
    l.market_route=static_cast<uint8_t>(ri(r,0,n_routes-1));
    return l;
}

static GPLeaf router_leaf(int expert) {
    const int ur=GP_MANAGER_UNIT_ROUTE>=0?GP_MANAGER_UNIT_ROUTE:0;
    return GPLeaf{GPL_ROUTE_UNITS_PLANNED_MARKET,(uint8_t)ur,(uint8_t)ur,(uint8_t)expert};
}

static void normalize_router(GPProgram& p, HostRng& r) {
    if(p.n_rules<1)p.n_rules=1;
    if(p.n_rules>GP_MAX_RULES)p.n_rules=GP_MAX_RULES;
    for(int i=0;i<p.n_rules;++i){
        p.rules[i].leaf=router_leaf(GP_ROUTER_EXPERTS>0?(p.rules[i].leaf.pad%GP_ROUTER_EXPERTS):0);
        if(i+1<p.n_rules){
            if(p.rules[i].n_conds<1){p.rules[i].n_conds=1;p.rules[i].conds[0]=random_cond(r);}
            if(p.rules[i].n_conds>GP_MAX_CONDS)p.rules[i].n_conds=GP_MAX_CONDS;
        }else p.rules[i].n_conds=0;
    }
}

static GPProgram router_constant(int expert, HostRng& r) {
    GPProgram p{};p.n_rules=1;p.rules[0]=GPRule{};p.rules[0].leaf=router_leaf(expert);normalize_router(p,r);return p;
}

static GPProgram random_router_program(HostRng& r) {
    GPProgram p{};p.n_rules=(uint8_t)ri(r,2,5);
    for(int i=0;i<p.n_rules;++i){
        GPRule& q=p.rules[i];
        q.n_conds=(uint8_t)(i+1==p.n_rules?0:ri(r,1,2));
        for(int c=0;c<q.n_conds;++c)q.conds[c]=random_cond(r);
        q.leaf=router_leaf(ri(r,0,GP_ROUTER_EXPERTS-1));
    }
    normalize_router(p,r);return p;
}

static GPProgram mutate_router(GPProgram p, HostRng& r) {
    const int edits=ri(r,1,3);
    for(int e=0;e<edits;++e){
        const int k=ri(r,0,6);
        if(k==0){
            const int rr=ri(r,0,p.n_rules-1);p.rules[rr].leaf.pad=(uint8_t)ri(r,0,GP_ROUTER_EXPERTS-1);
        }else if(k==1 && p.n_rules>1){
            const int rr=ri(r,0,p.n_rules-2);
            if(p.rules[rr].n_conds<1){p.rules[rr].n_conds=1;p.rules[rr].conds[0]=random_cond(r);}
            else p.rules[rr].conds[ri(r,0,p.rules[rr].n_conds-1)]=random_cond(r);
        }else if(k==2 && p.n_rules<GP_MAX_RULES){
            const int pos=ri(r,0,p.n_rules-1);
            for(int j=p.n_rules;j>pos;--j)p.rules[j]=p.rules[j-1];
            p.rules[pos]=GPRule{};p.rules[pos].n_conds=(uint8_t)ri(r,1,2);
            for(int c=0;c<p.rules[pos].n_conds;++c)p.rules[pos].conds[c]=random_cond(r);
            p.rules[pos].leaf=router_leaf(ri(r,0,GP_ROUTER_EXPERTS-1));++p.n_rules;
        }else if(k==3 && p.n_rules>1){
            const int pos=ri(r,0,p.n_rules-2);for(int j=pos;j+1<p.n_rules;++j)p.rules[j]=p.rules[j+1];--p.n_rules;
        }else if(k==4 && p.n_rules>1){
            const int rr=ri(r,0,p.n_rules-2);
            if(p.rules[rr].n_conds<GP_MAX_CONDS && ri(r,0,1)){p.rules[rr].conds[p.rules[rr].n_conds++]=random_cond(r);}
            else if(p.rules[rr].n_conds>1)--p.rules[rr].n_conds;
            else p.rules[rr].conds[0]=random_cond(r);
        }else if(k==5 && p.n_rules>2){
            const int a=ri(r,0,p.n_rules-2),b=ri(r,0,p.n_rules-2);GPRule q=p.rules[a];p.rules[a]=p.rules[b];p.rules[b]=q;
        }else if(p.n_rules>1){
            const int rr=ri(r,0,p.n_rules-2);p.rules[rr].conds[0]=random_cond(r);
        }
    }
    normalize_router(p,r);return p;
}

static uint32_t host_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}

static bool host_unlocked(int quadrants,int x,int y) {
    const int q=(y>=5?2:0)+(x>=5?1:0);
    return q < quadrants;
}

static uint8_t best_layout_seed(const uint8_t weight[8],int quadrants) {
    int target_total=0;for(int k=0;k<8;++k)target_total+=weight[k];
    const int cells=25*quadrants;
    int best_seed=1,best_err=1<<30;
    for(int seed=1;seed<=255;++seed){
        int got[8]={0};
        for(int y=0;y<10;++y)for(int x=0;x<10;++x){
            if(!host_unlocked(quadrants,x,y))continue;
            int total=target_total;if(total<=0){++got[7];continue;}
            int z=(int)(host_mix32((uint32_t)(x+17*y+131*seed))%(uint32_t)total);
            int cat=7;for(int k=0;k<8;++k){if(z<weight[k]){cat=k;break;}z-=weight[k];}
            ++got[cat];
        }
        int err=0;for(int k=0;k<8;++k){const int target=target_total?((int)weight[k]*cells+target_total/2)/target_total:(k==7?cells:0);const int d=got[k]-target;err+=d*d;}
        if(err<best_err){best_err=err;best_seed=seed;}
    }
    return (uint8_t)best_seed;
}

static void normalize_manager(GPManager& g) {
    // Strictly increasing causal phase boundaries.
    int prev=0;
    for(int i=0;i<GP_PHASES-1;++i){int v=g.phase_step[i];if(v<prev+12)v=prev+12;if(v>708-(GP_PHASES-2-i)*12)v=708-(GP_PHASES-2-i)*12;g.phase_step[i]=(uint16_t)v;prev=v;}
    int land=1,cows=0,sheep=0,geese=0;
    for(int p=0;p<GP_PHASES;++p){
        if(g.phase_target_quadrants[p]<land)g.phase_target_quadrants[p]=(uint8_t)land;
        if(g.phase_target_quadrants[p]>4)g.phase_target_quadrants[p]=4; land=g.phase_target_quadrants[p];
        if(g.phase_cow_target[p]<cows)g.phase_cow_target[p]=(uint8_t)cows;cows=g.phase_cow_target[p];
        if(g.phase_sheep_target[p]<sheep)g.phase_sheep_target[p]=(uint8_t)sheep;sheep=g.phase_sheep_target[p];
        if(g.phase_goose_target[p]<geese)g.phase_goose_target[p]=(uint8_t)geese;geese=g.phase_goose_target[p];
        if(g.phase_hires_per_day[p]>15)g.phase_hires_per_day[p]=15;
        int sum=0;for(int k=0;k<8;++k)sum+=g.phase_layout_weight[p][k];if(sum==0)g.phase_layout_weight[p][7]=1;
    }
}

static GPManager elite_manager(int variant) {
    GPManager g{};
    // Default struct is the sheep-heavy Ueddy/Kosti-style production backbone.
    if(variant==1){ // Kaw-like: 3 quadrants, cow-heavy, more STRAWBERRY.
        for(int p=7;p<GP_PHASES;++p)g.phase_target_quadrants[p]=3;
        const uint8_t cows[GP_PHASES]={2,2,2,3,6,8,8,10,10,10,10};
        const uint8_t sheep[GP_PHASES]={2,2,2,2,4,4,4,4,4,4,4};
        for(int p=0;p<GP_PHASES;++p){g.phase_cow_target[p]=cows[p];g.phase_sheep_target[p]=sheep[p];}
        g.phase_layout_weight[7][5]=35;g.phase_layout_weight[7][7]=20;
        g.phase_layout_weight[8][5]=42;g.phase_layout_weight[8][7]=8;
        g.phase_layout_weight[9][5]=40;g.phase_layout_weight[9][7]=18;
    } else if(variant==2){ // Erfan-like: earlier cash conservation, 3 quadrants, balanced livestock.
        for(int p=0;p<GP_PHASES;++p){if(p>=7)g.phase_target_quadrants[p]=3;g.phase_hires_per_day[p]=(uint8_t)((g.phase_hires_per_day[p]*4+2)/5);}
        const uint8_t cows[GP_PHASES]={1,1,1,2,8,8,8,9,9,9,9};
        const uint8_t sheep[GP_PHASES]={4,4,4,4,4,4,4,4,4,5,5};
        for(int p=0;p<GP_PHASES;++p){g.phase_cow_target[p]=cows[p];g.phase_sheep_target[p]=sheep[p];g.phase_cash_reserve[p]=(uint16_t)(g.phase_cash_reserve[p]+300);}
    }
    normalize_manager(g);
    for(int p=0;p<GP_PHASES;++p)g.phase_layout_seed[p]=best_layout_seed(g.phase_layout_weight[p],g.phase_target_quadrants[p]);
    return g;
}

static GPManager random_manager(HostRng& r) {
    // Random programs are still phase-coherent rather than arbitrary final farms.
    GPManager g=elite_manager(ri(r,0,2));
    for(int p=0;p<GP_PHASES;++p){
        for(int k=0;k<8;++k){int v=(int)g.phase_layout_weight[p][k]+ri(r,-12,12);if(v<0)v=0;if(v>80)v=80;g.phase_layout_weight[p][k]=(uint8_t)v;}
        g.phase_hires_per_day[p]=(uint8_t)ri(r,0,14);
        g.phase_cash_reserve[p]=(uint16_t)(ri(r,0,8)*500);
        if(p>2){g.phase_cow_target[p]=(uint8_t)ri(r,0,14);g.phase_sheep_target[p]=(uint8_t)ri(r,0,16);}
    }
    g.hire_until_hour=(uint8_t)ri(r,0,4);g.max_hires_per_turn=(uint8_t)ri(r,2,8);g.max_sell_orders=(uint8_t)ri(r,1,6);
    g.seed_buy_cap=(uint8_t)ri(r,8,40);g.animal_buy_cap=(uint8_t)ri(r,2,10);g.feed_stock=(uint8_t)ri(r,12,60);g.feed_load=(uint8_t)ri(r,3,16);
    g.drop_load=(uint8_t)ri(r,5,24);g.drop_hour=(uint8_t)ri(r,17,23);g.rebuild_step_div12=(uint8_t)ri(r,16,44);g.dist_penalty=(uint8_t)ri(r,1,10);
    for(int i=0;i<N_PRODUCTS;++i){g.sell_floor_pct[i]=(uint8_t)ri(r,30,160);g.sell_reserve[i]=(uint8_t)ri(r,0,24);}g.sell_batch=(uint8_t)ri(r,12,64);
    for(int i=0;i<10;++i)g.task_priority[i]=(uint8_t)ri(r,100,255);
    normalize_manager(g);for(int p=0;p<GP_PHASES;++p)g.phase_layout_seed[p]=best_layout_seed(g.phase_layout_weight[p],g.phase_target_quadrants[p]);
    return g;
}

static GPProgram random_program(HostRng& r, int n_routes) {
    GPProgram p{};
    p.n_rules=static_cast<uint8_t>(ri(r,1,4));
    for (int i=0;i<p.n_rules;++i) {
        GPRule& rule=p.rules[i];
        rule.n_conds=static_cast<uint8_t>(i==p.n_rules-1?0:ri(r,1,3));
        for (int c=0;c<rule.n_conds;++c) rule.conds[c]=random_cond(r);
        rule.leaf=random_leaf(r,n_routes);
    }
    p.manager=random_manager(r);
    return p;
}

static void mutate_manager(GPManager& g, HostRng& r) {
    const int k=ri(r,0,18), p=ri(r,0,GP_PHASES-1);
    auto bump8=[&](uint8_t& v,int d=10){int x=(int)v+ri(r,-d,d);if(x<0)x=0;if(x>255)x=255;v=(uint8_t)x;};
    if(k==0){bump8(g.phase_layout_weight[p][ri(r,0,7)],12);}
    else if(k==1)g.phase_layout_seed[p]=(uint8_t)ri(r,1,255);
    else if(k==2)g.phase_target_quadrants[p]=(uint8_t)ri(r,1,4);
    else if(k==3)g.phase_hires_per_day[p]=(uint8_t)ri(r,0,15);
    else if(k==4)bump8(g.phase_cow_target[p],3);
    else if(k==5)bump8(g.phase_sheep_target[p],3);
    else if(k==6)bump8(g.phase_goose_target[p],2);
    else if(k==7)g.phase_cash_reserve[p]=(uint16_t)ri(r,0,10)*500;
    else if(k==8&&p>0){int i=p-1;g.phase_step[i]=(uint16_t)((int)g.phase_step[i]+ri(r,-12,12));}
    else if(k==9){g.hire_until_hour=(uint8_t)ri(r,0,5);g.max_hires_per_turn=(uint8_t)ri(r,2,9);}
    else if(k==10)g.max_sell_orders=(uint8_t)ri(r,1,7);
    else if(k==11){g.feed_stock=(uint8_t)ri(r,8,80);g.feed_load=(uint8_t)ri(r,2,20);}
    else if(k==12){g.drop_load=(uint8_t)ri(r,4,32);g.drop_hour=(uint8_t)ri(r,15,23);}
    else if(k==13){int i=ri(r,0,N_PRODUCTS-1);g.sell_floor_pct[i]=(uint8_t)ri(r,0,200);}
    else if(k==14){int i=ri(r,0,N_PRODUCTS-1);g.sell_reserve[i]=(uint8_t)ri(r,0,40);}
    else if(k==15)g.sell_batch=(uint8_t)ri(r,4,90);
    else if(k==16)g.task_priority[ri(r,0,9)]=(uint8_t)ri(r,60,255);
    else if(k==17){g.care_enabled^=1;g.collect_fert_enabled^=1;g.rebuild_step_div12=(uint8_t)ri(r,8,50);}
    else {g.seed_buy_cap=(uint8_t)ri(r,4,50);g.animal_buy_cap=(uint8_t)ri(r,1,12);g.dist_penalty=(uint8_t)ri(r,1,14);g.planned_horizon=(uint8_t)ri(r,8,96);g.planned_seed_horizon=(uint8_t)ri(r,6,72);g.planned_animal_horizon=(uint8_t)ri(r,2,32);g.planned_pickup_horizon=(uint8_t)ri(r,0,12);g.planned_seed_buffer[ri(r,0,N_CROPS-1)]=(uint8_t)ri(r,0,8);g.planned_animal_buffer=(uint8_t)ri(r,0,3);g.planned_feed_buffer=(uint8_t)ri(r,0,12);g.planned_fert_buffer=(uint8_t)ri(r,0,12);}
    normalize_manager(g);
}

static GPProgram mutate(GPProgram p, HostRng& r, int n_routes) {
    if (GP_ROUTER_ONLY) return mutate_router(p,r);
    const int edits=ri(r,1,4);
    if (GP_MANAGER_ONLY) {
        GPManager m=p.manager;for(int e=0;e<edits;++e)mutate_manager(m,r);set_manager_program(p,m);return p;
    }
    for (int e=0;e<edits;++e) {
        const int k=ri(r,0,7);
        if (k<=2) mutate_manager(p.manager,r);
        else if (k==3) {
            const int rr=ri(r,0,p.n_rules-1); p.rules[rr].leaf=random_leaf(r,n_routes);
        } else if (k==4 && p.n_rules<GP_MAX_RULES) {
            const int pos=ri(r,0,p.n_rules-1);
            for(int j=p.n_rules;j>pos;--j)p.rules[j]=p.rules[j-1];
            p.rules[pos]=GPRule{}; p.rules[pos].n_conds=static_cast<uint8_t>(ri(r,1,3));
            for(int c=0;c<p.rules[pos].n_conds;++c)p.rules[pos].conds[c]=random_cond(r);
            p.rules[pos].leaf=random_leaf(r,n_routes); ++p.n_rules;
        } else if (k==5 && p.n_rules>1) {
            const int pos=ri(r,0,p.n_rules-2); for(int j=pos;j+1<p.n_rules;++j)p.rules[j]=p.rules[j+1]; --p.n_rules;
        } else if (k==6 && p.n_rules>1) {
            const int rr=ri(r,0,p.n_rules-2);
            if (p.rules[rr].n_conds == 0) { p.rules[rr].n_conds=1; p.rules[rr].conds[0]=random_cond(r); }
            else p.rules[rr].conds[ri(r,0,p.rules[rr].n_conds-1)]=random_cond(r);
        } else {
            const int rr=ri(r,0,p.n_rules-1);
            p.rules[rr].leaf.unit_route=static_cast<uint8_t>(ri(r,0,n_routes-1));
            p.rules[rr].leaf.market_route=static_cast<uint8_t>(ri(r,0,n_routes-1));
        }
    }
    // Every nonterminal rule must remain a real guarded Sequence. The last
    // rule alone is the unconditional Select fallback.
    for(int i=0;i+1<p.n_rules;++i) if(p.rules[i].n_conds==0){p.rules[i].n_conds=1;p.rules[i].conds[0]=random_cond(r);}
    p.rules[p.n_rules-1].n_conds=0;
    return p;
}

static GPProgram crossover(const GPProgram& a, const GPProgram& b, HostRng& r) {
    GPProgram c=a;
    if (ri(r,0,1)) c.manager=b.manager;
    const int common = a.n_rules < b.n_rules ? a.n_rules : b.n_rules;
    const int cut=ri(r,0,common-1);
    for(int i=cut;i<c.n_rules && i<b.n_rules;++i)c.rules[i]=b.rules[i];
    if(GP_ROUTER_ONLY){normalize_router(c,r);return c;}
    for(int i=0;i+1<c.n_rules;++i) if(c.rules[i].n_conds==0){c.rules[i].n_conds=1;c.rules[i].conds[0]=random_cond(r);}
    c.rules[c.n_rules-1].n_conds=0;
    return c;
}

static void dump_program(std::ostream& os, const GPProgram& p) {
    os << "rules=" << int(p.n_rules) << "\n";
    for(int r=0;r<p.n_rules;++r){
        const auto& q=p.rules[r]; os << " rule "<<r<<" conds="<<int(q.n_conds)<<" leaf(kind="<<int(q.leaf.kind)<<",u="<<int(q.leaf.unit_route)<<",m="<<int(q.leaf.market_route)<<",x="<<int(q.leaf.pad)<<")";
        for(int c=0;c<q.n_conds;++c)os<<" [f="<<int(q.conds[c].feature)<<" cmp="<<int(q.conds[c].cmp)<<" item="<<int(q.conds[c].item)<<" v="<<q.conds[c].value<<"]";
        os << "\n";
    }
    if(GP_ROUTER_ONLY){os<<" router_experts="<<GP_ROUTER_EXPERTS<<"\n";return;}
    const auto& g=p.manager;
    os<<" manager phases";
    for(int ph=0;ph<GP_PHASES;++ph){os<<"\n  p"<<ph<<" step="<<(ph?g.phase_step[ph-1]:0)<<" seed="<<int(g.phase_layout_seed[ph])<<" land="<<int(g.phase_target_quadrants[ph])<<" hires="<<int(g.phase_hires_per_day[ph])<<" animals="<<int(g.phase_cow_target[ph])<<'/'<<int(g.phase_sheep_target[ph])<<'/'<<int(g.phase_goose_target[ph])<<" cash="<<g.phase_cash_reserve[ph]<<" weights=";for(int k=0;k<8;++k)os<<int(g.phase_layout_weight[ph][k])<<(k==7?' ':',');}
    os<<"\n controls hire_until="<<int(g.hire_until_hour)<<" hire_batch="<<int(g.max_hires_per_turn)<<" max_sell_orders="<<int(g.max_sell_orders)
      <<" feed_stock="<<int(g.feed_stock)<<" feed_load="<<int(g.feed_load)<<" drop="<<int(g.drop_load)<<"@"<<int(g.drop_hour)
      <<" sell_batch="<<int(g.sell_batch)<<" care="<<int(g.care_enabled)<<" fert="<<int(g.collect_fert_enabled)<<" rebuild12="<<int(g.rebuild_step_div12)
      <<" planned_h="<<int(g.planned_horizon)<<" seed_h="<<int(g.planned_seed_horizon)<<" animal_h="<<int(g.planned_animal_horizon)<<" pickup_h="<<int(g.planned_pickup_horizon)<<" seed_buf=";for(int i=0;i<N_CROPS;++i)os<<int(g.planned_seed_buffer[i])<<(i+1==N_CROPS?' ':',');os<<"animal_buf="<<int(g.planned_animal_buffer)<<" feed_buf="<<int(g.planned_feed_buffer)<<" fert_buf="<<int(g.planned_fert_buffer)<<"\n";
    os << " sell_floor_pct=";for(int i=0;i<N_PRODUCTS;++i)os<<int(g.sell_floor_pct[i])<<(i+1==N_PRODUCTS?' ':',');
    os << "reserve=";for(int i=0;i<N_PRODUCTS;++i)os<<int(g.sell_reserve[i])<<(i+1==N_PRODUCTS?'\n':',');
    os << " task_priority=";for(int i=0;i<10;++i)os<<int(g.task_priority[i])<<(i==9?'\n':',');
}

int main(int argc,char** argv){
    std::string routes_list, out_path, load_path, seed_program_path, seed_list_arg, fitness_mode="robust";
    std::vector<std::string> router_manager_paths;
    bool manager_only=false,planned_market=false,router_only=false;
    int bootstrap_route=-1,bootstrap_step=0,manager_unit_route=-1;
    int pickup_horizon_override=-1;
    int population=256,generations=20,seeds=8,opponents=8,elite=12,tpb=64;
    uint64_t seed_start=63000, search_seed=340034;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--manager-only"){manager_only=true;continue;}
        if(a=="--planned-market"){planned_market=true;continue;}
        if(a=="--router-only"){router_only=true;continue;}
        auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
        if(a=="--routes")routes_list=need(); else if(a=="--population")population=std::stoi(need());
        else if(a=="--generations")generations=std::stoi(need()); else if(a=="--seeds")seeds=std::stoi(need());
        else if(a=="--seed-start")seed_start=std::stoull(need()); else if(a=="--seed-list")seed_list_arg=need(); else if(a=="--opponents")opponents=std::stoi(need());
        else if(a=="--elite")elite=std::stoi(need()); else if(a=="--tpb")tpb=std::stoi(need());
        else if(a=="--search-seed")search_seed=std::stoull(need()); else if(a=="--out")out_path=need();
        else if(a=="--load")load_path=need();
        else if(a=="--seed-program")seed_program_path=need();
        else if(a=="--router-manager")router_manager_paths.push_back(need());
        else if(a=="--bootstrap-route")bootstrap_route=std::stoi(need());
        else if(a=="--bootstrap-step")bootstrap_step=std::stoi(need());
        else if(a=="--manager-unit-route")manager_unit_route=std::stoi(need());
        else if(a=="--pickup-horizon")pickup_horizon_override=std::stoi(need());
        else if(a=="--fitness")fitness_mode=need();
        else if(a=="--help"){std::cout<<"bonsai_search --routes LIST [--population N --generations N --seeds N --seed-start S --seed-list S1,S2,... --opponents N --elite N --seed-program FILE.bin --router-only --router-manager FILE.bin (repeat) --pickup-horizon N --fitness robust|wins --out PATH]\n";return 0;}
        else throw std::runtime_error("unknown argument: "+a);
    }
    if(routes_list.empty())throw std::runtime_error("--routes LIST is required");
    if(fitness_mode=="robust")GP_ROBUST_FITNESS=true;
    else if(fitness_mode=="wins")GP_ROBUST_FITNESS=false;
    else throw std::runtime_error("--fitness must be robust or wins");
    if(router_only && manager_only)throw std::runtime_error("--router-only and --manager-only are mutually exclusive");
    if(router_only && router_manager_paths.size()<2)throw std::runtime_error("--router-only requires at least two --router-manager FILE.bin arguments");
    if(router_manager_paths.size()>255)throw std::runtime_error("router manager count exceeds uint8 encoding");
    if(router_only && manager_unit_route<0)manager_unit_route=0;
    GP_MANAGER_ONLY=manager_only;GP_ROUTER_ONLY=router_only;GP_ROUTER_EXPERTS=(int)router_manager_paths.size();
    GP_BOOTSTRAP_ROUTE=bootstrap_route;GP_BOOTSTRAP_STEP=bootstrap_step;GP_MANAGER_UNIT_ROUTE=manager_unit_route;GP_PLANNED_MARKET=planned_market||router_only;
    std::vector<uint64_t> h_seed_values;
    if(!seed_list_arg.empty()){
        std::stringstream ss(seed_list_arg);std::string tok;
        while(std::getline(ss,tok,',')){if(tok.empty())continue;h_seed_values.push_back(std::stoull(tok));}
        if(h_seed_values.empty())throw std::runtime_error("--seed-list is empty");
        seeds=(int)h_seed_values.size();
    }
    if(!load_path.empty()){population=1;elite=1;generations=1;}
    auto paths=read_list(routes_list); if(paths.empty())throw std::runtime_error("route list is empty");
    if(paths.size()>255)throw std::runtime_error("route count exceeds uint8 program encoding");
    if(opponents>static_cast<int>(paths.size()))opponents=static_cast<int>(paths.size());
    if(elite<1)elite=1;if(elite>population)elite=population;

    std::vector<NativeTape> nt; nt.reserve(paths.size());
    for(const auto& p:paths)nt.push_back(load_native_tape_json(p));
    std::vector<Action> h_routes; std::vector<GPRouteDesc> h_desc;
    for(const auto& t:nt){int off=h_routes.size();h_routes.insert(h_routes.end(),t.actions.begin(),t.actions.end());h_desc.push_back({off,static_cast<int>(t.actions.size())});}

    std::vector<GPManager> h_manager_bank;
    for(const auto& mp:router_manager_paths){
        GPProgram q{};std::ifstream in(mp,std::ios::binary);if(!in)throw std::runtime_error("cannot open router manager program: "+mp);
        in.read(reinterpret_cast<char*>(&q),sizeof(q));if(in.gcount()!=static_cast<std::streamsize>(sizeof(q)))throw std::runtime_error("invalid router manager program size: "+mp);
        h_manager_bank.push_back(q.manager);
    }
    if(pickup_horizon_override>=0){
        if(pickup_horizon_override>255)throw std::runtime_error("--pickup-horizon must be 0..255");
        for(auto& g:h_manager_bank)g.planned_pickup_horizon=(uint8_t)pickup_horizon_override;
    }

    configure_device(); TablesOwner tables; Config cfg;
    Action* d_routes=nullptr; GPRouteDesc* d_desc=nullptr; GPProgram* d_prog=nullptr; GPManager* d_manager_bank=nullptr; uint64_t* d_seed_values=nullptr; Sim* d_sims=nullptr; double* d_rewards=nullptr;
    const int jobs=population*seeds;
    KAGCUDA_CHECK(cudaMalloc((void**)&d_routes,h_routes.size()*sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_desc,h_desc.size()*sizeof(GPRouteDesc)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_prog,population*sizeof(GPProgram)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_sims,static_cast<size_t>(jobs)*sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_rewards,static_cast<size_t>(jobs)*2*sizeof(double)));
    KAGCUDA_CHECK(cudaMemcpy(d_routes,h_routes.data(),h_routes.size()*sizeof(Action),cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_desc,h_desc.data(),h_desc.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    if(!h_manager_bank.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_manager_bank,h_manager_bank.size()*sizeof(GPManager)));KAGCUDA_CHECK(cudaMemcpy(d_manager_bank,h_manager_bank.data(),h_manager_bank.size()*sizeof(GPManager),cudaMemcpyHostToDevice));}
    if(!h_seed_values.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_seed_values,h_seed_values.size()*sizeof(uint64_t)));KAGCUDA_CHECK(cudaMemcpy(d_seed_values,h_seed_values.data(),h_seed_values.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));}

    HostRng rng(search_seed);
    std::vector<GPProgram> pop(population);
    if(!load_path.empty()){
        std::ifstream in(load_path,std::ios::binary);if(!in)throw std::runtime_error("cannot open program: "+load_path);
        in.read(reinterpret_cast<char*>(&pop[0]),sizeof(GPProgram));if(in.gcount()!=static_cast<std::streamsize>(sizeof(GPProgram)))throw std::runtime_error("invalid program binary size: "+load_path);
        if(router_only)normalize_router(pop[0],rng);
    } else if(router_only) {
        int z=0;
        for(int x=0;x<GP_ROUTER_EXPERTS&&z<population;++x,++z)pop[z]=router_constant(x,rng);
        for(;z<population;++z)pop[z]=random_router_program(rng);
    } else if(manager_only) {
        // Preserve three known-good macro families as priors, then populate a
        // broad evolutionary island around them. They are only starting genes.
        int z=0;
        for(int v=0;v<3&&z<population;++v,++z)set_manager_program(pop[z],elite_manager(v));
        for(;z<population;++z){GPManager m;if(z<population*3/4){m=elite_manager(z%3);for(int e=0;e<ri(rng,2,14);++e)mutate_manager(m,rng);}else m=random_manager(rng);set_manager_program(pop[z],m);}
    } else {
        // Seed known routes and cross-dimensional hybrids, then fill the majority
        // with genuinely new manager/program genomes.
        int z=0;
        for(int r=0;r<static_cast<int>(paths.size()) && z<population;++r,++z){pop[z]=GPProgram{};pop[z].n_rules=1;pop[z].rules[0].leaf={GPL_ROUTE,(uint8_t)r,(uint8_t)r,0};pop[z].manager=random_manager(rng);}
        for(int r=0;r<static_cast<int>(paths.size()) && z<population;++r,++z){pop[z]=GPProgram{};pop[z].n_rules=1;pop[z].rules[0].leaf={GPL_ROUTE_UNITS_MANAGER_MARKET,(uint8_t)r,(uint8_t)r,0};pop[z].manager=random_manager(rng);}
        while(z<population)pop[z++]=random_program(rng,paths.size());
    }
    if(!seed_program_path.empty() && load_path.empty() && !router_only){
        GPProgram seedp{};
        std::ifstream in(seed_program_path,std::ios::binary);
        if(!in)throw std::runtime_error("cannot open seed program: "+seed_program_path);
        in.read(reinterpret_cast<char*>(&seedp),sizeof(seedp));
        if(in.gcount()!=static_cast<std::streamsize>(sizeof(seedp)))throw std::runtime_error("invalid seed program binary size: "+seed_program_path);
        if(manager_only){
            // Keep half the island near the proven champion while retaining the
            // other half as diverse priors/random managers. This prevents a new
            // structural mechanism from being discarded just because the next
            // search restarts from older donor-derived priors.
            const int near=population/2;
            set_manager_program(pop[0],seedp.manager);
            for(int z=1;z<near;++z){
                GPManager m=seedp.manager;
                for(int e=0;e<ri(rng,1,9);++e)mutate_manager(m,rng);
                set_manager_program(pop[z],m);
            }
        } else pop[0]=seedp;
    }
    if(pickup_horizon_override>=0 && !router_only){
        for(auto& p:pop)p.manager.planned_pickup_horizon=(uint8_t)pickup_horizon_override;
    }

    std::vector<Fitness> fit(population); std::vector<double> host(static_cast<size_t>(jobs)*2);
    std::vector<double> cohort_sum(static_cast<size_t>(population)*opponents*2);
    std::vector<double> seed_sum(static_cast<size_t>(population)*seeds);
    auto evaluate=[&](){
        for (int i=0;i<population;++i) fit[i]=Fitness{};
        for(double& x:cohort_sum)x=0.0;
        for(double& x:seed_sum)x=0.0;
        KAGCUDA_CHECK(cudaMemcpy(d_prog,pop.data(),population*sizeof(GPProgram),cudaMemcpyHostToDevice));
        for(int oi=0;oi<opponents;++oi){
            for(int seat=0;seat<2;++seat){
                gp_eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,d_prog,population,d_routes,d_desc,paths.size(),d_manager_bank,(int)h_manager_bank.size(),oi,seat,seed_start,d_seed_values,seeds,d_sims,d_rewards);
                KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());
                KAGCUDA_CHECK(cudaMemcpy(host.data(),d_rewards,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
                for(int pi=0;pi<population;++pi)for(int si=0;si<seeds;++si){
                    const int id=pi*seeds+si; const double r0=host[id*2],r1=host[id*2+1]; const double m=seat==0?r0-r1:r1-r0;
                    Fitness& f=fit[pi];++f.cases;if(m>0)++f.wins;else if(m==0)++f.ties;f.sum+=m;if(m<f.floor)f.floor=m;
                    cohort_sum[(static_cast<size_t>(pi)*opponents+oi)*2+seat]+=m;
                    seed_sum[static_cast<size_t>(pi)*seeds+si]+=m;
                }
            }
        }
        const int ncoh=opponents*2;
        const int ntail=(ncoh+3)/4;
        std::vector<double> cm(ncoh);
        for(int pi=0;pi<population;++pi){
            for(int oi=0;oi<opponents;++oi)for(int seat=0;seat<2;++seat)
                cm[oi*2+seat]=cohort_sum[(static_cast<size_t>(pi)*opponents+oi)*2+seat]/seeds;
            for(int i=1;i<ncoh;++i){double x=cm[i];int j=i-1;while(j>=0&&cm[j]>x){cm[j+1]=cm[j];--j;}cm[j+1]=x;}
            double tail=0.0;for(int i=0;i<ntail;++i)tail+=cm[i];
            fit[pi].tail_mean=tail/ntail;
            fit[pi].worst_cohort=cm[0];
            std::vector<double> sm(seeds);
            for(int si=0;si<seeds;++si)sm[si]=seed_sum[static_cast<size_t>(pi)*seeds+si]/(opponents*2);
            for(int i=1;i<seeds;++i){double x=sm[i];int j=i-1;while(j>=0&&sm[j]>x){sm[j+1]=sm[j];--j;}sm[j+1]=x;}
            const int nstail=(seeds+3)/4;
            double stail=0.0;for(int i=0;i<nstail;++i)stail+=sm[i];
            fit[pi].seed_tail=stail/nstail;
            fit[pi].worst_seed=sm[0];
        }
    };

    GPProgram global_best{}; Fitness global_fit{}; bool have_best=false;
    for(int gen=0;gen<generations;++gen){
        evaluate();
        std::vector<int> order(population);for(int i=0;i<population;++i)order[i]=i;
        for(int i=0;i<population;++i){
            int best=i;
            for(int j=i+1;j<population;++j) if(fitter(fit[order[j]],fit[order[best]])) best=j;
            if(best!=i){const int t=order[i];order[i]=order[best];order[best]=t;}
        }
        const int bi=order[0];
        if(!have_best||fitter(fit[bi],global_fit)){have_best=true;global_best=pop[bi];global_fit=fit[bi];}
        std::cout<<"GEN "<<gen<<" best="<<fit[bi].wins<<"/"<<fit[bi].cases<<" ties="<<fit[bi].ties<<" floor="<<fit[bi].floor<<" mean="<<mean(fit[bi])
                 <<" tail="<<fit[bi].tail_mean<<" worst_cohort="<<fit[bi].worst_cohort<<" seed_tail="<<fit[bi].seed_tail<<" worst_seed="<<fit[bi].worst_seed
                 <<" global="<<global_fit.wins<<"/"<<global_fit.cases<<" gfloor="<<global_fit.floor<<" gmean="<<mean(global_fit)<<" gtail="<<global_fit.tail_mean<<" gworst="<<global_fit.worst_cohort<<" gseed_tail="<<global_fit.seed_tail<<" gworst_seed="<<global_fit.worst_seed<<"\n";
        dump_program(std::cout,pop[bi]); std::cout.flush();
        if(gen+1==generations)break;
        std::vector<GPProgram> next(population);
        for(int i=0;i<elite;++i)next[i]=pop[order[i]];
        auto tournament=[&](){const int cap=(population-1)<63?(population-1):63;int best=order[ri(rng,0,cap)];for(int k=0;k<3;++k){int q=order[ri(rng,0,cap)];if(fitter(fit[q],fit[best]))best=q;}return best;};
        for(int i=elite;i<population;++i){
            const int a=tournament(); GPProgram child=pop[a];
            if(ri(rng,0,99)<25){const int b=tournament();child=crossover(pop[a],pop[b],rng);}
            next[i]=mutate(child,rng,paths.size());
        }
        pop.swap(next);
    }

    std::cout<<"FINAL_BEST wins="<<global_fit.wins<<"/"<<global_fit.cases<<" ties="<<global_fit.ties<<" floor="<<global_fit.floor<<" mean="<<mean(global_fit)<<" tail="<<global_fit.tail_mean<<" worst_cohort="<<global_fit.worst_cohort<<" seed_tail="<<global_fit.seed_tail<<" worst_seed="<<global_fit.worst_seed<<"\n";
    dump_program(std::cout,global_best);
    if(!out_path.empty()){
        std::ofstream out(out_path);out<<"wins="<<global_fit.wins<<" cases="<<global_fit.cases<<" ties="<<global_fit.ties<<" floor="<<global_fit.floor<<" mean="<<mean(global_fit)<<" tail="<<global_fit.tail_mean<<" worst_cohort="<<global_fit.worst_cohort<<" seed_tail="<<global_fit.seed_tail<<" worst_seed="<<global_fit.worst_seed<<" fitness="<<fitness_mode<<"\n";dump_program(out,global_best);
        std::ofstream bout(out_path+".bin",std::ios::binary);bout.write(reinterpret_cast<const char*>(&global_best),sizeof(GPProgram));
    }
    cudaFree(d_routes);cudaFree(d_desc);cudaFree(d_prog);if(d_manager_bank)cudaFree(d_manager_bank);if(d_seed_values)cudaFree(d_seed_values);cudaFree(d_sims);cudaFree(d_rewards);
    return 0;
}
