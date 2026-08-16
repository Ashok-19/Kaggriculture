#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include "bonsai_io.hpp"
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

__device__ __noinline__ Action gp_act_program_index(int e,
                                                     const ObservationView& obs,
                                                     const GPProgram* program_bank, int n_program_bank,
                                                     const Action* routes, const GPRouteDesc* descs, int n_routes) {
    const int be = (e>=0 && e<n_program_bank) ? e : 0;
    return gp_act(program_bank[be],obs,routes,descs,n_routes);
}

__device__ __noinline__ Action gp_dispatch_program_router(const GPProgram& router,
                                                           const ObservationView& obs,
                                                           const GPProgram* program_bank, int n_program_bank,
                                                           const Action* routes, const GPRouteDesc* descs, int n_routes) {
    const GPLeaf top = gp_select_leaf(router,obs);
    return gp_act_program_index(static_cast<int>(top.pad),obs,program_bank,n_program_bank,routes,descs,n_routes);
}

struct GPStateReport {
    int captured = 0;
    int step = 0;
    double own_money = 0.0, opp_money = 0.0;
    int own_units = 0, opp_units = 0;
    int own_land = 0, opp_land = 0;
    int own_pastures = 0, opp_pastures = 0;
    int own_animals[N_ANIMALS] = {0};
    int opp_animals[N_ANIMALS] = {0};
    int own_crops[N_CROPS] = {0};
    int opp_crops[N_CROPS] = {0};
    int own_seeds[N_CROPS] = {0};
    int prices[N_PRODUCTS] = {0};
    int market_inv[N_PRODUCTS] = {0};
};

__device__ __forceinline__ GPStateReport gp_capture_state_report(const ObservationView& obs) {
    GPStateReport z{};
    const int p=obs.player(), q=1-p;
    z.captured=1; z.step=obs.step();
    z.own_money=obs.farm(p).money(); z.opp_money=obs.farm(q).money();
    z.own_units=obs.farm(p).n_units(); z.opp_units=obs.farm(q).n_units();
    z.own_land=obs.farm(p).n_quadrants(); z.opp_land=obs.farm(q).n_quadrants();
    z.own_pastures=gp_public_pastures(obs.farm(p)); z.opp_pastures=gp_public_pastures(obs.farm(q));
    for(int a=0;a<N_ANIMALS;++a){
        z.own_animals[a]=gp_public_animals(obs.farm(p),GOOSE+a);
        z.opp_animals[a]=gp_public_animals(obs.farm(q),GOOSE+a);
    }
    for(int c=0;c<N_CROPS;++c){
        z.own_crops[c]=gp_public_crops(obs.farm(p),c);
        z.opp_crops[c]=gp_public_crops(obs.farm(q),c);
        z.own_seeds[c]=obs.private_farm().seed(c);
    }
    for(int i=0;i<N_PRODUCTS;++i){z.prices[i]=obs.market().price(i);z.market_inv[i]=obs.market().inventory(i);}
    return z;
}

__global__ void gp_eval_kernel(Config base, DeviceTables tables,
                               const GPProgram* programs, int n_programs,
                               const Action* routes, const GPRouteDesc* descs, int n_routes,
                               const GPManager* manager_bank, int n_managers,
                               const GPProgram* program_bank, int n_program_bank,
                               const GPProgram* staged_router_prefix, int staged_router_step,
                               const GPProgram* early_marker_router, int early_marker_step,
                               const GPProgram* reactive_override_router,
                               const int* marker_program_map, int n_marker_programs,
                               int opponent_route, int candidate_seat,
                               uint64_t seed_start, const uint64_t* seed_values, int n_seeds,
                               Sim* sims, double* rewards, int report_state_step, bool report_first_override, bool report_first_override_gap, GPStateReport* state_reports) {
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
    bool ever_reactive_override=false;
    while (!sim.st.done) {
        const int step = sim.st.step;
        const int opp_seat = 1-candidate_seat;
        const ObservationView obs = make_observation(sim,candidate_seat);
        if(state_reports && step==report_state_step) state_reports[id]=gp_capture_state_report(obs);
        Action cand{};
        if (program_bank && n_program_bank > 0) {
            bool override_handled=false;
            if (reactive_override_router) {
                const GPLeaf ov = gp_select_leaf(*reactive_override_router,obs);
                const int oe = static_cast<int>(ov.pad);
                if (oe >= 0 && oe < n_program_bank) {
                    if(report_first_override && state_reports && !state_reports[id].captured) state_reports[id]=gp_capture_state_report(obs);
                    ever_reactive_override=true;
                    cand = gp_act_program_index(oe,obs,program_bank,n_program_bank,routes,descs,n_routes);
                    override_handled=true;
                }
            }
            if(report_first_override_gap && reactive_override_router && ever_reactive_override && !override_handled && state_reports && !state_reports[id].captured) state_reports[id]=gp_capture_state_report(obs);
            if (override_handled) {
                // Reactive causal/safety layer owns this turn.
            } else if (marker_program_map && n_marker_programs > 0) {
                const int marker = obs.private_farm().seed(TOMATO);
                if (marker >= 1 && marker <= n_marker_programs) {
                    const int be = marker_program_map[marker-1];
                    cand = (be>=0 && be<n_program_bank) ? gp_act_program_index(be,obs,program_bank,n_program_bank,routes,descs,n_routes) : gp_dispatch_program_router(*staged_router_prefix,obs,program_bank,n_program_bank,routes,descs,n_routes);
                } else {
                    const GPProgram* decision_router = nullptr;
                    if (early_marker_router && step == early_marker_step) decision_router = early_marker_router;
                    else if (step == staged_router_step) decision_router = &prog;
                    if (decision_router && marker == 0) {
                        const GPLeaf top = gp_select_leaf(*decision_router,obs);
                        const int local = static_cast<int>(top.pad);
                        if (local >= 0 && local < n_marker_programs) {
                            const int be = marker_program_map[local];
                            cand = (be>=0 && be<n_program_bank) ? gp_act_program_index(be,obs,program_bank,n_program_bank,routes,descs,n_routes) : gp_dispatch_program_router(*staged_router_prefix,obs,program_bank,n_program_bank,routes,descs,n_routes);
                            // Externalized commitment memory. Appending the marker order
                            // preserves all existing market-order indices.
                            if (cand.n_orders < MAX_ORDERS) cand.orders[cand.n_orders++] = Order{M_BUY_SEED,TOMATO,local+1};
                        } else {
                            cand = gp_dispatch_program_router(*staged_router_prefix,obs,program_bank,n_program_bank,routes,descs,n_routes);
                        }
                    } else {
                        // Before/between decision stages, or missing/ambiguous marker:
                        // continue the validated base router exactly.
                        cand = gp_dispatch_program_router(*staged_router_prefix,obs,program_bank,n_program_bank,routes,descs,n_routes);
                    }
                }
            } else {
                const GPProgram& router = (staged_router_prefix && step < staged_router_step) ? *staged_router_prefix : prog;
                const GPLeaf top = gp_select_leaf(router,obs);
                const int e = static_cast<int>(top.pad);
                if (e < n_program_bank) {
                    cand = gp_act_program_index(e,obs,program_bank,n_program_bank,routes,descs,n_routes);
                } else if (staged_router_prefix && e == n_program_bank) {
                    cand = gp_dispatch_program_router(*staged_router_prefix,obs,program_bank,n_program_bank,routes,descs,n_routes);
                } else {
                    cand = gp_act_program_index(0,obs,program_bank,n_program_bank,routes,descs,n_routes);
                }
            }
        } else {
            cand = gp_act_with_managers(prog,obs,routes,descs,n_routes,manager_bank,n_managers);
        }
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

enum GPFitnessMode { GPF_WINS = 0, GPF_ROBUST = 1, GPF_BALANCED = 2 };
static int GP_FITNESS_MODE = GPF_ROBUST;

static double mean(const Fitness& f) { return f.cases ? f.sum / f.cases : 0.0; }
static double win_rate(const Fitness& f) { return f.cases ? static_cast<double>(f.wins) / f.cases : 0.0; }
static double balanced_score(const Fitness& f) {
    // Near-competitive stage objective. Money margins and both lower-tail axes
    // stay on their natural simulator scale, while a full 100% win-rate swing
    // is worth 5000 margin units. This prevents ~1k tail noise from discarding
    // dozens of converted games, without allowing catastrophic tails to hide.
    return mean(f) + 0.5*(f.tail_mean + f.seed_tail) + 5000.0*win_rate(f);
}

static bool fitter(const Fitness& a, const Fitness& b) {
    if (GP_FITNESS_MODE == GPF_WINS) {
        if (a.wins != b.wins) return a.wins > b.wins;
        if (a.ties != b.ties) return a.ties < b.ties;
        if (a.floor != b.floor) return a.floor > b.floor;
        return a.sum > b.sum;
    }
    if (GP_FITNESS_MODE == GPF_BALANCED) {
        const double as=balanced_score(a), bs=balanced_score(b);
        if(as!=bs)return as>bs;
        if(a.wins!=b.wins)return a.wins>b.wins;
        const double aguard=a.tail_mean<a.seed_tail?a.tail_mean:a.seed_tail;
        const double bguard=b.tail_mean<b.seed_tail?b.tail_mean:b.seed_tail;
        if(aguard!=bguard)return aguard>bguard;
        if(a.floor!=b.floor)return a.floor>b.floor;
        return a.ties<b.ties;
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
    const int choice=ri(r,0,27);
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
    } else if (choice==10) {
        c.feature=GPF_LAND; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,1,4);
    } else if (choice==11) {
        c.feature=GPF_SHOP_SEEN; c.item=ri(r,0,MAX_SHOP_INSTANCES-1); c.cmp=GPC_GE; c.value=ri(r,1,2);
    } else if (choice==12) {
        c.feature=GPF_OWN_PASTURES; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,2,24);
    } else if (choice==13) {
        c.feature=GPF_OWN_ANIMAL; c.item=ri(r,0,N_ANIMALS-1); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,14);
    } else if (choice==14) {
        c.feature=GPF_OWN_CROP; c.item=ri(r,0,N_CROPS-1); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,30);
    } else if (choice==15) {
        c.feature=GPF_SEED; c.item=ri(r,0,N_CROPS-1); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,30);
    } else if (choice<=17) {
        c.feature=choice==16?GPF_WORKER_X:GPF_WORKER_Y; c.item=ri(r,0,15); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,9);
    } else if (choice==18) {
        c.feature=GPF_STEP_BUCKET8; c.cmp=GPC_EQ; c.value=ri(r,0,89);
    } else if (choice==19) {
        c.feature=GPF_OWN_WEEDS; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,35);
    } else if (choice==20) {
        c.feature=GPF_OPP_UNITS; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,1,15);
    } else if (choice==21) {
        c.feature=GPF_OPP_LAND; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,1,4);
    } else if (choice==22) {
        c.feature=GPF_OPP_PASTURES; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,24);
    } else if (choice==23) {
        c.feature=GPF_OPP_ANIMAL; c.item=ri(r,0,N_ANIMALS-1); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,14);
    } else if (choice==24) {
        c.feature=GPF_OPP_CROP; c.item=ri(r,0,N_CROPS-1); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,30);
    } else if (choice==25) {
        c.feature=GPF_OPP_WEEDS; c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,35);
    } else {
        c.feature=choice==26?GPF_OPP_WORKER_X:GPF_OPP_WORKER_Y; c.item=ri(r,0,15); c.cmp=ri(r,0,1)?GPC_GE:GPC_LE; c.value=ri(r,0,9);
    }
    return c;
}

static bool GP_MANAGER_ONLY = false;
static bool GP_ROUTER_ONLY = false;
static bool GP_ROUTE_ROUTER_ONLY = false;
static bool GP_PROGRAM_ROUTER_ONLY = false;
static bool GP_SUFFIX_SEARCH = false;
static bool GP_SUFFIX_RICH = false;
static int GP_SUFFIX_START = -1;
static int GP_SUFFIX_FREEZE_RULES = 0;
static std::vector<int> GP_SUFFIX_ROUTES;
static int GP_ROUTER_EXPERTS = 0;
static std::vector<int> GP_ROUTER_UNIT_ROUTES;
static int GP_BOOTSTRAP_ROUTE = -1;
static int GP_BOOTSTRAP_STEP = 0;
static int GP_MANAGER_UNIT_ROUTE = -1;
static bool GP_PLANNED_MARKET = false;
static bool GP_SUFFIX_CROSSOVER = false;

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
    int ur=GP_MANAGER_UNIT_ROUTE>=0?GP_MANAGER_UNIT_ROUTE:0;
    if(expert>=0 && expert<(int)GP_ROUTER_UNIT_ROUTES.size())ur=GP_ROUTER_UNIT_ROUTES[expert];
    if(GP_ROUTE_ROUTER_ONLY || GP_PROGRAM_ROUTER_ONLY)
        return GPLeaf{GPL_ROUTE,(uint8_t)ur,(uint8_t)ur,(uint8_t)expert};
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

static bool suffix_route_allowed(int route) {
    for(int x:GP_SUFFIX_ROUTES) if(x==route) return true;
    return false;
}

static GPLeaf suffix_leaf(HostRng& r) {
    GPLeaf l{};
    const int route=GP_SUFFIX_ROUTES[ri(r,0,(int)GP_SUFFIX_ROUTES.size()-1)];
    l.unit_route=(uint8_t)route;l.market_route=(uint8_t)route;l.pad=0;
    if(!GP_SUFFIX_RICH) l.kind=GPL_ROUTE;
    else {
        const int k=ri(r,0,4);
        l.kind=(uint8_t)(k==0?GPL_ROUTE:(k==1?GPL_ROUTE_UNITS_MANAGER_MARKET:(k==2?GPL_MANAGER_UNITS_ROUTE_MARKET:(k==3?GPL_ROUTE_UNITS_PLANNED_MARKET:GPL_MANAGER))));
    }
    return l;
}
static bool suffix_leaf_allowed(const GPLeaf& l) {
    if(!GP_SUFFIX_RICH) return l.kind==GPL_ROUTE && suffix_route_allowed(l.unit_route) && l.unit_route==l.market_route;
    if(l.kind==GPL_MANAGER) return true;
    if(l.kind==GPL_ROUTE || l.kind==GPL_ROUTE_UNITS_MANAGER_MARKET || l.kind==GPL_MANAGER_UNITS_ROUTE_MARKET || l.kind==GPL_ROUTE_UNITS_PLANNED_MARKET)
        return suffix_route_allowed(l.unit_route) && (l.kind!=GPL_ROUTE || l.unit_route==l.market_route);
    return false;
}

static GPCond suffix_guard() {
    GPCond c{};c.feature=GPF_STEP;c.cmp=GPC_GE;c.value=GP_SUFFIX_START;return c;
}

static GPRule random_suffix_rule(HostRng& r) {
    GPRule q{};q.n_conds=(uint8_t)ri(r,2,GP_MAX_CONDS);q.conds[0]=suffix_guard();
    for(int c=1;c<q.n_conds;++c)q.conds[c]=random_cond(r);
    q.leaf=suffix_leaf(r);return q;
}

static void normalize_suffix(GPProgram& p, HostRng& r) {
    if(GP_SUFFIX_FREEZE_RULES<0)GP_SUFFIX_FREEZE_RULES=0;
    if(GP_SUFFIX_FREEZE_RULES>=GP_MAX_RULES)GP_SUFFIX_FREEZE_RULES=GP_MAX_RULES-1;
    if(p.n_rules<GP_SUFFIX_FREEZE_RULES+1)p.n_rules=(uint8_t)(GP_SUFFIX_FREEZE_RULES+1);
    if(p.n_rules>GP_MAX_RULES)p.n_rules=GP_MAX_RULES;
    for(int i=GP_SUFFIX_FREEZE_RULES;i+1<p.n_rules;++i){
        GPRule& q=p.rules[i];
        if(q.n_conds<2){q.n_conds=2;q.conds[1]=random_cond(r);}
        if(q.n_conds>GP_MAX_CONDS)q.n_conds=GP_MAX_CONDS;
        q.conds[0]=suffix_guard();
        if(!suffix_leaf_allowed(q.leaf))q.leaf=suffix_leaf(r);
    }
    GPRule& fallback=p.rules[p.n_rules-1];fallback.n_conds=0;
    if(!suffix_leaf_allowed(fallback.leaf))fallback.leaf=suffix_leaf(r);
}

static GPProgram mutate_suffix(GPProgram p, HostRng& r) {
    const int edits=ri(r,1,4);
    for(int e=0;e<edits;++e){
        const int first=GP_SUFFIX_FREEZE_RULES;
        const int last=(int)p.n_rules-1;
        const int k=ri(r,0,GP_SUFFIX_RICH?8:6);
        if(k==0 && p.n_rules<GP_MAX_RULES){
            const int pos=ri(r,first,last); // insert before fallback
            for(int j=p.n_rules;j>pos;--j)p.rules[j]=p.rules[j-1];
            p.rules[pos]=random_suffix_rule(r);++p.n_rules;
        }else if(k==1 && p.n_rules>first+1){
            const int pos=ri(r,first,(int)p.n_rules-2);
            for(int j=pos;j+1<p.n_rules;++j)p.rules[j]=p.rules[j+1];--p.n_rules;
        }else if(k==2){
            const int rr=ri(r,first,(int)p.n_rules-1);p.rules[rr].leaf=suffix_leaf(r);
        }else if(k==3 && p.n_rules>first+1){
            const int rr=ri(r,first,(int)p.n_rules-2);
            if(p.rules[rr].n_conds<2){p.rules[rr].n_conds=2;p.rules[rr].conds[1]=random_cond(r);}
            else p.rules[rr].conds[ri(r,1,p.rules[rr].n_conds-1)]=random_cond(r);
        }else if(k==4 && p.n_rules>first+1){
            const int rr=ri(r,first,(int)p.n_rules-2);
            if(p.rules[rr].n_conds<GP_MAX_CONDS && ri(r,0,1))p.rules[rr].conds[p.rules[rr].n_conds++]=random_cond(r);
            else if(p.rules[rr].n_conds>2)--p.rules[rr].n_conds;
            else p.rules[rr].conds[1]=random_cond(r);
        }else if(k==5 && p.n_rules>first+2){
            const int a=ri(r,first,(int)p.n_rules-2),b=ri(r,first,(int)p.n_rules-2);
            GPRule q=p.rules[a];p.rules[a]=p.rules[b];p.rules[b]=q;
        }else if(k==6 && GP_SUFFIX_RICH){
            mutate_manager(p.manager,r);
        }else if(k==7 && GP_SUFFIX_RICH){
            const int rr=ri(r,first,(int)p.n_rules-1);p.rules[rr].leaf=suffix_leaf(r);mutate_manager(p.manager,r);
        }else{
            p.rules[p.n_rules-1].leaf=suffix_leaf(r);
        }
    }
    normalize_suffix(p,r);return p;
}

static GPProgram mutate(GPProgram p, HostRng& r, int n_routes) {
    if (GP_SUFFIX_SEARCH) return mutate_suffix(p,r);
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

static GPManager crossover_manager_groups(const GPManager& a, const GPManager& b, HostRng& r) {
    GPManager c=a;
    // Phase genes form coherent macro bundles: timing, layout, land/labor,
    // livestock, and reserve for one phase inherit together.
    for(int ph=0;ph<GP_PHASES;++ph){
        if(!ri(r,0,1))continue;
        c.phase_layout_seed[ph]=b.phase_layout_seed[ph];
        for(int k=0;k<8;++k)c.phase_layout_weight[ph][k]=b.phase_layout_weight[ph][k];
        c.phase_target_quadrants[ph]=b.phase_target_quadrants[ph];
        c.phase_hires_per_day[ph]=b.phase_hires_per_day[ph];
        c.phase_cow_target[ph]=b.phase_cow_target[ph];
        c.phase_sheep_target[ph]=b.phase_sheep_target[ph];
        c.phase_goose_target[ph]=b.phase_goose_target[ph];
        c.phase_cash_reserve[ph]=b.phase_cash_reserve[ph];
        if(ph>0)c.phase_step[ph-1]=b.phase_step[ph-1];
    }
    // Recombine operational control families independently. This is much less
    // destructive than inheriting the complete manager wholesale.
    if(ri(r,0,1)){c.hire_until_hour=b.hire_until_hour;c.max_hires_per_turn=b.max_hires_per_turn;c.max_sell_orders=b.max_sell_orders;c.seed_buy_cap=b.seed_buy_cap;c.animal_buy_cap=b.animal_buy_cap;}
    if(ri(r,0,1)){c.feed_stock=b.feed_stock;c.feed_load=b.feed_load;c.fert_load=b.fert_load;c.drop_load=b.drop_load;c.drop_hour=b.drop_hour;c.care_enabled=b.care_enabled;c.collect_fert_enabled=b.collect_fert_enabled;c.rebuild_step_div12=b.rebuild_step_div12;c.dist_penalty=b.dist_penalty;}
    if(ri(r,0,1)){c.planned_horizon=b.planned_horizon;c.planned_seed_horizon=b.planned_seed_horizon;c.planned_animal_horizon=b.planned_animal_horizon;c.planned_pickup_horizon=b.planned_pickup_horizon;for(int i=0;i<N_CROPS;++i)c.planned_seed_buffer[i]=b.planned_seed_buffer[i];c.planned_animal_buffer=b.planned_animal_buffer;c.planned_feed_buffer=b.planned_feed_buffer;c.planned_fert_buffer=b.planned_fert_buffer;c.copy_route_infrastructure=b.copy_route_infrastructure;}
    if(ri(r,0,1)){for(int i=0;i<N_PRODUCTS;++i){c.sell_floor_pct[i]=b.sell_floor_pct[i];c.sell_reserve[i]=b.sell_reserve[i];}c.sell_batch=b.sell_batch;}
    if(ri(r,0,1))for(int i=0;i<10;++i)c.task_priority[i]=b.task_priority[i];
    normalize_manager(c);
    return c;
}

static GPProgram crossover_suffix(const GPProgram& a, const GPProgram& b, HostRng& r) {
    GPProgram c=a;
    // Keep the frozen causal prefix from parent A; recombine only the suffix.
    if (GP_SUFFIX_RICH) {
        const int mk=ri(r,0,2);
        if(mk==1)c.manager=b.manager;
        else if(mk==2)c.manager=crossover_manager_groups(a.manager,b.manager,r);
    }
    const int common = a.n_rules < b.n_rules ? a.n_rules : b.n_rules;
    const int first = GP_SUFFIX_FREEZE_RULES;
    if (common > first) {
        const int cut = ri(r,first,common-1);
        for(int i=cut;i<c.n_rules && i<b.n_rules;++i)c.rules[i]=b.rules[i];
    }
    for(int i=0;i<first && i<a.n_rules;++i)c.rules[i]=a.rules[i];
    normalize_suffix(c,r);
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
    std::string routes_list, out_path, load_path, seed_program_path, seed_program2_path, seed_list_arg, fitness_mode="robust", staged_router_prefix_path, early_marker_router_path, reactive_override_router_path;
    std::vector<std::string> router_manager_paths, router_program_paths, population_program_paths;
    std::vector<int> router_unit_routes, router_route_ids, suffix_route_ids, marker_program_ids;
    bool manager_only=false,planned_market=false,router_only=false,route_router_only=false,program_router_only=false,suffix_search=false,suffix_rich=false,unit_route_sweep=false,report_seeds=false,report_cases=false,report_first_override=false,report_first_override_gap=false;
    int bootstrap_route=-1,bootstrap_step=0,manager_unit_route=-1,suffix_start=-1,suffix_freeze_override=-1,staged_router_step=-1,early_marker_step=-1;
    int pickup_horizon_override=-1,report_state_step=-1;
    int population=256,generations=20,seeds=8,opponents=8,opponent_start=0,elite=12,tpb=64;
    uint64_t seed_start=63000, search_seed=340034;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--manager-only"){manager_only=true;continue;}
        if(a=="--planned-market"){planned_market=true;continue;}
        if(a=="--router-only"){router_only=true;continue;}
        if(a=="--route-router-only"){route_router_only=true;continue;}
        if(a=="--program-router-only"){program_router_only=true;continue;}
        if(a=="--suffix-search"){suffix_search=true;continue;}
        if(a=="--suffix-rich"){suffix_rich=true;continue;}
        if(a=="--unit-route-sweep"){unit_route_sweep=true;continue;}
        if(a=="--report-seeds"){report_seeds=true;continue;}
        if(a=="--report-cases"){report_cases=true;continue;}
        if(a=="--report-first-override"){report_first_override=true;continue;}
        if(a=="--report-first-override-gap"){report_first_override_gap=true;continue;}
        auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
        if(a=="--routes")routes_list=need(); else if(a=="--population")population=std::stoi(need());
        else if(a=="--generations")generations=std::stoi(need()); else if(a=="--seeds")seeds=std::stoi(need());
        else if(a=="--seed-start")seed_start=std::stoull(need()); else if(a=="--seed-list")seed_list_arg=need(); else if(a=="--opponents")opponents=std::stoi(need()); else if(a=="--opponent-start")opponent_start=std::stoi(need());
        else if(a=="--elite")elite=std::stoi(need()); else if(a=="--tpb")tpb=std::stoi(need());
        else if(a=="--search-seed")search_seed=std::stoull(need()); else if(a=="--out")out_path=need();
        else if(a=="--load")load_path=need();
        else if(a=="--seed-program")seed_program_path=need();
        else if(a=="--seed-program2")seed_program2_path=need();
        else if(a=="--staged-program-router-prefix")staged_router_prefix_path=need();
        else if(a=="--staged-program-router-step")staged_router_step=std::stoi(need());
        else if(a=="--early-marker-router")early_marker_router_path=need();
        else if(a=="--early-marker-step")early_marker_step=std::stoi(need());
        else if(a=="--reactive-override-router")reactive_override_router_path=need();
        else if(a=="--router-manager")router_manager_paths.push_back(need());
        else if(a=="--router-program")router_program_paths.push_back(need());
        else if(a=="--population-program")population_program_paths.push_back(need());
        else if(a=="--router-unit-route")router_unit_routes.push_back(std::stoi(need()));
        else if(a=="--router-route")router_route_ids.push_back(std::stoi(need()));
        else if(a=="--marker-commit-program")marker_program_ids.push_back(std::stoi(need()));
        else if(a=="--suffix-route")suffix_route_ids.push_back(std::stoi(need()));
        else if(a=="--suffix-start")suffix_start=std::stoi(need());
        else if(a=="--suffix-freeze-rules")suffix_freeze_override=std::stoi(need());
        else if(a=="--bootstrap-route")bootstrap_route=std::stoi(need());
        else if(a=="--bootstrap-step")bootstrap_step=std::stoi(need());
        else if(a=="--manager-unit-route")manager_unit_route=std::stoi(need());
        else if(a=="--pickup-horizon")pickup_horizon_override=std::stoi(need());
        else if(a=="--report-state-step")report_state_step=std::stoi(need());
        else if(a=="--fitness")fitness_mode=need();
        else if(a=="--help"){std::cout<<"bonsai_search --routes LIST [--population N --generations N --seeds N --seed-start S --seed-list S1,S2,... --opponents N --elite N --seed-program FILE.bin [--seed-program2 FILE.bin] --router-only --router-manager FILE.bin (repeat) --router-unit-route N (repeat) --route-router-only --router-route N (repeat) --program-router-only --router-program FILE.bin (repeat) --population-program FILE.bin (repeat) --opponent-start N --staged-program-router-prefix FILE.bin --staged-program-router-step N --early-marker-router FILE.bin --early-marker-step N --reactive-override-router FILE.bin --marker-commit-program BANK_INDEX (repeat) --suffix-search --suffix-start N --suffix-freeze-rules N --suffix-route N (repeat) --unit-route-sweep --report-seeds --pickup-horizon N --fitness robust|balanced|wins --out PATH]\n";return 0;}
        else throw std::runtime_error("unknown argument: "+a);
    }
    if(routes_list.empty())throw std::runtime_error("--routes LIST is required");
    if(report_state_step>=720)throw std::runtime_error("--report-state-step must be <720");
    if(fitness_mode=="robust")GP_FITNESS_MODE=GPF_ROBUST;
    else if(fitness_mode=="wins")GP_FITNESS_MODE=GPF_WINS;
    else if(fitness_mode=="balanced")GP_FITNESS_MODE=GPF_BALANCED;
    else throw std::runtime_error("--fitness must be robust, balanced, or wins");
    if((router_only||route_router_only||program_router_only||suffix_search) && manager_only)throw std::runtime_error("router/suffix modes and --manager-only are mutually exclusive");
    if((int)router_only+(int)route_router_only+(int)program_router_only>1)throw std::runtime_error("router modes are mutually exclusive");
    if(suffix_search && (router_only||route_router_only||program_router_only||unit_route_sweep))throw std::runtime_error("--suffix-search cannot be combined with router or sweep modes");
    if(suffix_rich && !suffix_search)throw std::runtime_error("--suffix-rich requires --suffix-search");
    if(suffix_search && seed_program_path.empty())throw std::runtime_error("--suffix-search requires --seed-program FILE.bin");
    if(!seed_program2_path.empty() && !suffix_search)throw std::runtime_error("--seed-program2 is supported only with --suffix-search");
    if(suffix_search && !load_path.empty())throw std::runtime_error("--suffix-search cannot be combined with --load");
    if(suffix_search && (suffix_start<0||suffix_start>=720))throw std::runtime_error("--suffix-search requires --suffix-start in 0..719");
    if(suffix_search && suffix_route_ids.size()<2)throw std::runtime_error("--suffix-search requires at least two --suffix-route N values");
    if(unit_route_sweep && (router_only||route_router_only||program_router_only||manager_only))throw std::runtime_error("--unit-route-sweep cannot be combined with router or manager-only modes");
    if(unit_route_sweep && seed_program_path.empty())throw std::runtime_error("--unit-route-sweep requires --seed-program FILE.bin");
    if(router_only && router_manager_paths.size()<2)throw std::runtime_error("--router-only requires at least two --router-manager FILE.bin arguments");
    if(route_router_only && router_route_ids.size()<2)throw std::runtime_error("--route-router-only requires at least two --router-route N arguments");
    if(program_router_only && router_program_paths.size()<2)throw std::runtime_error("--program-router-only requires at least two --router-program FILE.bin arguments");
    if(!staged_router_prefix_path.empty() && !program_router_only)throw std::runtime_error("--staged-program-router-prefix requires --program-router-only");
    if(staged_router_step>=0 && staged_router_prefix_path.empty())throw std::runtime_error("--staged-program-router-step requires --staged-program-router-prefix");
    if(!staged_router_prefix_path.empty() && (staged_router_step<0 || staged_router_step>=720))throw std::runtime_error("staged program-router step must be in 0..719");
    if(!marker_program_ids.empty() && (!program_router_only || staged_router_prefix_path.empty()))throw std::runtime_error("--marker-commit-program requires staged --program-router-only mode");
    if(!early_marker_router_path.empty() && marker_program_ids.empty())throw std::runtime_error("--early-marker-router requires marker commitment mode");
    if(early_marker_step>=0 && early_marker_router_path.empty())throw std::runtime_error("--early-marker-step requires --early-marker-router");
    if(!early_marker_router_path.empty() && (early_marker_step<0 || early_marker_step>=staged_router_step))throw std::runtime_error("early marker step must be >=0 and earlier than staged marker step");
    if(!reactive_override_router_path.empty() && (!program_router_only || staged_router_prefix_path.empty()))throw std::runtime_error("--reactive-override-router requires staged --program-router-only mode");
    if(router_manager_paths.size()>255 || router_route_ids.size()>255 || router_program_paths.size()>255)throw std::runtime_error("router expert count exceeds uint8 encoding");
    if(!router_unit_routes.empty() && router_unit_routes.size()!=router_manager_paths.size())throw std::runtime_error("repeat --router-unit-route once per --router-manager, or omit all for a shared route");
    if(router_only && manager_unit_route<0)manager_unit_route=0;
    if(router_unit_routes.empty())router_unit_routes.assign(router_manager_paths.size(),manager_unit_route>=0?manager_unit_route:0);
    GP_MANAGER_ONLY=manager_only;GP_ROUTER_ONLY=router_only||route_router_only||program_router_only;GP_ROUTE_ROUTER_ONLY=route_router_only;GP_PROGRAM_ROUTER_ONLY=program_router_only;
    GP_SUFFIX_SEARCH=suffix_search;GP_SUFFIX_RICH=suffix_rich;GP_SUFFIX_START=suffix_start;GP_SUFFIX_ROUTES=suffix_route_ids;GP_SUFFIX_CROSSOVER=suffix_rich&&!seed_program2_path.empty();
    GP_ROUTER_EXPERTS=(int)(program_router_only?(!marker_program_ids.empty()?(marker_program_ids.size()+1):(router_program_paths.size()+(!staged_router_prefix_path.empty()?1:0))):(route_router_only?router_route_ids.size():router_manager_paths.size()));
    GP_ROUTER_UNIT_ROUTES=route_router_only?router_route_ids:router_unit_routes;
    GP_BOOTSTRAP_ROUTE=bootstrap_route;GP_BOOTSTRAP_STEP=bootstrap_step;GP_MANAGER_UNIT_ROUTE=manager_unit_route;GP_PLANNED_MARKET=planned_market||router_only;
    std::vector<uint64_t> h_seed_values;
    if(!seed_list_arg.empty()){
        std::stringstream ss(seed_list_arg);std::string tok;
        while(std::getline(ss,tok,',')){if(tok.empty())continue;h_seed_values.push_back(std::stoull(tok));}
        if(h_seed_values.empty())throw std::runtime_error("--seed-list is empty");
        seeds=(int)h_seed_values.size();
    }
    if(!load_path.empty()){population=1;elite=1;generations=1;}
    if(!population_program_paths.empty()){
        if(!load_path.empty()||!seed_program_path.empty()||!seed_program2_path.empty())throw std::runtime_error("--population-program cannot be combined with --load or seed programs");
        population=(int)population_program_paths.size();elite=population;generations=1;
    }
    auto paths=read_list(routes_list); if(paths.empty())throw std::runtime_error("route list is empty");
    if(paths.size()>255)throw std::runtime_error("route count exceeds uint8 program encoding");
    for(int ur:GP_ROUTER_UNIT_ROUTES)if(ur<0||ur>=(int)paths.size())throw std::runtime_error("router unit route out of range for route list");
    for(int ur:GP_SUFFIX_ROUTES)if(ur<0||ur>=(int)paths.size())throw std::runtime_error("suffix route out of range for route list");
    if(unit_route_sweep){population=(int)paths.size();elite=population;generations=1;GP_PLANNED_MARKET=true;}
    if(opponent_start<0 || opponent_start>=static_cast<int>(paths.size()))throw std::runtime_error("--opponent-start out of range");
    if(opponents>static_cast<int>(paths.size())-opponent_start)opponents=static_cast<int>(paths.size())-opponent_start;
    if(elite<1)elite=1;if(elite>population)elite=population;

    std::vector<NativeTape> nt; nt.reserve(paths.size());
    for(const auto& p:paths)nt.push_back(load_native_tape_json(p));
    std::vector<Action> h_routes; std::vector<GPRouteDesc> h_desc;
    for(const auto& t:nt){int off=h_routes.size();h_routes.insert(h_routes.end(),t.actions.begin(),t.actions.end());h_desc.push_back({off,static_cast<int>(t.actions.size())});}

    std::vector<GPManager> h_manager_bank;
    for(const auto& mp:router_manager_paths){
        GPProgram q=load_gp_program_binary(mp);
        h_manager_bank.push_back(q.manager);
    }
    std::vector<GPProgram> h_program_bank;
    for(const auto& pp:router_program_paths)h_program_bank.push_back(load_gp_program_binary(pp));
    for(int e:marker_program_ids)if(e<0 || e>=(int)h_program_bank.size())throw std::runtime_error("marker commit program index out of range");
    GPProgram h_staged_router_prefix{};
    const bool have_staged_router_prefix=!staged_router_prefix_path.empty();
    if(have_staged_router_prefix)h_staged_router_prefix=load_gp_program_binary(staged_router_prefix_path);
    GPProgram h_early_marker_router{};
    const bool have_early_marker_router=!early_marker_router_path.empty();
    if(have_early_marker_router)h_early_marker_router=load_gp_program_binary(early_marker_router_path);
    GPProgram h_reactive_override_router{};
    const bool have_reactive_override_router=!reactive_override_router_path.empty();
    if(have_reactive_override_router)h_reactive_override_router=load_gp_program_binary(reactive_override_router_path);
    if(pickup_horizon_override>=0){
        if(pickup_horizon_override>255)throw std::runtime_error("--pickup-horizon must be 0..255");
        for(auto& g:h_manager_bank)g.planned_pickup_horizon=(uint8_t)pickup_horizon_override;
    }

    configure_device(); TablesOwner tables; Config cfg;
    Action* d_routes=nullptr; GPRouteDesc* d_desc=nullptr; GPProgram* d_prog=nullptr; GPManager* d_manager_bank=nullptr; GPProgram* d_program_bank=nullptr; GPProgram* d_staged_router_prefix=nullptr; GPProgram* d_early_marker_router=nullptr; GPProgram* d_reactive_override_router=nullptr; int* d_marker_program_map=nullptr; uint64_t* d_seed_values=nullptr; Sim* d_sims=nullptr; double* d_rewards=nullptr; GPStateReport* d_state_reports=nullptr;
    const int jobs=population*seeds;
    KAGCUDA_CHECK(cudaMalloc((void**)&d_routes,h_routes.size()*sizeof(Action)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_desc,h_desc.size()*sizeof(GPRouteDesc)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_prog,population*sizeof(GPProgram)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_sims,static_cast<size_t>(jobs)*sizeof(Sim)));
    KAGCUDA_CHECK(cudaMalloc((void**)&d_rewards,static_cast<size_t>(jobs)*2*sizeof(double)));
    if(report_state_step>=0 || report_first_override || report_first_override_gap)KAGCUDA_CHECK(cudaMalloc((void**)&d_state_reports,static_cast<size_t>(jobs)*sizeof(GPStateReport)));
    KAGCUDA_CHECK(cudaMemcpy(d_routes,h_routes.data(),h_routes.size()*sizeof(Action),cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(d_desc,h_desc.data(),h_desc.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    if(!h_manager_bank.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_manager_bank,h_manager_bank.size()*sizeof(GPManager)));KAGCUDA_CHECK(cudaMemcpy(d_manager_bank,h_manager_bank.data(),h_manager_bank.size()*sizeof(GPManager),cudaMemcpyHostToDevice));}
    if(!h_program_bank.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_program_bank,h_program_bank.size()*sizeof(GPProgram)));KAGCUDA_CHECK(cudaMemcpy(d_program_bank,h_program_bank.data(),h_program_bank.size()*sizeof(GPProgram),cudaMemcpyHostToDevice));}
    if(have_staged_router_prefix){KAGCUDA_CHECK(cudaMalloc((void**)&d_staged_router_prefix,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMemcpy(d_staged_router_prefix,&h_staged_router_prefix,sizeof(GPProgram),cudaMemcpyHostToDevice));}
    if(have_early_marker_router){KAGCUDA_CHECK(cudaMalloc((void**)&d_early_marker_router,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMemcpy(d_early_marker_router,&h_early_marker_router,sizeof(GPProgram),cudaMemcpyHostToDevice));}
    if(have_reactive_override_router){KAGCUDA_CHECK(cudaMalloc((void**)&d_reactive_override_router,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMemcpy(d_reactive_override_router,&h_reactive_override_router,sizeof(GPProgram),cudaMemcpyHostToDevice));}
    if(!marker_program_ids.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_marker_program_map,marker_program_ids.size()*sizeof(int)));KAGCUDA_CHECK(cudaMemcpy(d_marker_program_map,marker_program_ids.data(),marker_program_ids.size()*sizeof(int),cudaMemcpyHostToDevice));}
    if(!h_seed_values.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&d_seed_values,h_seed_values.size()*sizeof(uint64_t)));KAGCUDA_CHECK(cudaMemcpy(d_seed_values,h_seed_values.data(),h_seed_values.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));}

    HostRng rng(search_seed);
    std::vector<GPProgram> pop(population);
    if(!population_program_paths.empty()){
        for(int i=0;i<population;++i){pop[i]=load_gp_program_binary(population_program_paths[i]);if(GP_ROUTER_ONLY)normalize_router(pop[i],rng);}
    } else if(unit_route_sweep){
        GPProgram seedp=load_gp_program_binary(seed_program_path);
        for(int r=0;r<population;++r){pop[r]=GPProgram{};pop[r].manager=seedp.manager;pop[r].n_rules=1;pop[r].rules[0].leaf={GPL_ROUTE_UNITS_PLANNED_MARKET,(uint8_t)r,(uint8_t)r,0};}
    } else if(!load_path.empty()){
        pop[0]=load_gp_program_binary(load_path);
        if(GP_ROUTER_ONLY)normalize_router(pop[0],rng);
    } else if(GP_ROUTER_ONLY) {
        int z=0;
        if(!seed_program_path.empty()){
            GPProgram seedp=load_gp_program_binary(seed_program_path);
            normalize_router(seedp,rng);pop[z++]=seedp;
            const int near=population/2;
            for(;z<near;++z){GPProgram q=seedp;for(int e=0;e<ri(rng,1,6);++e)q=mutate_router(q,rng);pop[z]=q;}
        }
        for(int x=0;x<GP_ROUTER_EXPERTS&&z<population;++x,++z)pop[z]=router_constant(x,rng);
        for(;z<population;++z)pop[z]=random_router_program(rng);
    } else if(suffix_search) {
        GPProgram seedp=load_gp_program_binary(seed_program_path);
        const bool have_seed2=!seed_program2_path.empty();
        GPProgram seedp2{};
        // Legacy single-parent suffix-rich searches start from generic elite managers.
        // Two-parent searches preserve both learned manager genomes for real crossover.
        if(GP_SUFFIX_RICH && !have_seed2) seedp.manager=elite_manager(0);
        if(seedp.n_rules<1)throw std::runtime_error("suffix seed program has no fallback rule");
        GP_SUFFIX_FREEZE_RULES=suffix_freeze_override>=0?suffix_freeze_override:(int)seedp.n_rules-1;
        if(GP_SUFFIX_FREEZE_RULES<0 || GP_SUFFIX_FREEZE_RULES>(int)seedp.n_rules-1)throw std::runtime_error("--suffix-freeze-rules must be in 0..seed_rules-1");
        if(GP_SUFFIX_FREEZE_RULES>=GP_MAX_RULES-1)throw std::runtime_error("suffix seed prefix leaves no search capacity");
        normalize_suffix(seedp,rng);
        if(have_seed2){
            seedp2=load_gp_program_binary(seed_program2_path);
            if(seedp2.n_rules<=GP_SUFFIX_FREEZE_RULES)throw std::runtime_error("--seed-program2 has insufficient rules for suffix freeze");
            normalize_suffix(seedp2,rng);
        }
        int z=0;pop[z++]=seedp;
        if(have_seed2 && z<population)pop[z++]=seedp2;
        // Deterministically include every allowed constant continuation.
        for(int route:GP_SUFFIX_ROUTES){
            if(z>=population)break;
            GPProgram q=seedp;
            q.rules[q.n_rules-1].leaf={GPL_ROUTE,(uint8_t)route,(uint8_t)route,0};
            normalize_suffix(q,rng);pop[z++]=q;
        }
        // Seed explicit two-parent offspring before mutation-only descendants.
        if(have_seed2){
            const int cross_target=population/4;
            while(z<population && z<cross_target){
                GPProgram q=(z&1)?crossover_suffix(seedp,seedp2,rng):crossover_suffix(seedp2,seedp,rng);
                for(int e=0;e<ri(rng,0,4);++e)q=mutate_suffix(q,rng);
                pop[z++]=q;
            }
        }
        for(;z<population;++z){
            GPProgram q=have_seed2&&ri(rng,0,1)?seedp2:seedp;
            if(GP_SUFFIX_RICH && !have_seed2 && (z&3)==0) q.manager=elite_manager((z/4)%3);
            for(int e=0;e<ri(rng,2,10);++e)q=mutate_suffix(q,rng);
            pop[z]=q;
        }
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
    if(!seed_program_path.empty() && load_path.empty() && !GP_ROUTER_ONLY && !unit_route_sweep && !suffix_search){
        GPProgram seedp=load_gp_program_binary(seed_program_path);
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
    if(pickup_horizon_override>=0 && !GP_ROUTER_ONLY){
        for(auto& p:pop)p.manager.planned_pickup_horizon=(uint8_t)pickup_horizon_override;
    }

    std::vector<Fitness> fit(population); std::vector<double> host(static_cast<size_t>(jobs)*2);
    std::vector<GPStateReport> host_state_reports((report_state_step>=0||report_first_override||report_first_override_gap)?static_cast<size_t>(jobs):0);
    std::vector<double> cohort_sum(static_cast<size_t>(population)*opponents*2);
    std::vector<double> seed_sum(static_cast<size_t>(population)*seeds);
    std::vector<int> seed_wins(static_cast<size_t>(population)*seeds);
    auto evaluate=[&](){
        for (int i=0;i<population;++i) fit[i]=Fitness{};
        for(double& x:cohort_sum)x=0.0;
        for(double& x:seed_sum)x=0.0;
        for(int& x:seed_wins)x=0;
        KAGCUDA_CHECK(cudaMemcpy(d_prog,pop.data(),population*sizeof(GPProgram),cudaMemcpyHostToDevice));
        for(int oi=0;oi<opponents;++oi){
            for(int seat=0;seat<2;++seat){
                if(d_state_reports)KAGCUDA_CHECK(cudaMemset(d_state_reports,0,static_cast<size_t>(jobs)*sizeof(GPStateReport)));
                gp_eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,d_prog,population,d_routes,d_desc,paths.size(),d_manager_bank,(int)h_manager_bank.size(),d_program_bank,(int)h_program_bank.size(),d_staged_router_prefix,staged_router_step,d_early_marker_router,early_marker_step,d_reactive_override_router,d_marker_program_map,(int)marker_program_ids.size(),opponent_start+oi,seat,seed_start,d_seed_values,seeds,d_sims,d_rewards,report_state_step,report_first_override,report_first_override_gap,d_state_reports);
                KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());
                KAGCUDA_CHECK(cudaMemcpy(host.data(),d_rewards,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
                if(d_state_reports){
                    KAGCUDA_CHECK(cudaMemcpy(host_state_reports.data(),d_state_reports,static_cast<size_t>(jobs)*sizeof(GPStateReport),cudaMemcpyDeviceToHost));
                    for(int pi=0;pi<population;++pi)for(int si=0;si<seeds;++si){
                        const GPStateReport& z=host_state_reports[static_cast<size_t>(pi)*seeds+si];
                        if(!z.captured)continue;
                        const uint64_t sv=h_seed_values.empty()?seed_start+(uint64_t)si:h_seed_values[si];
                        std::cout<<"STATE_REPORT pi="<<pi<<" opponent="<<oi<<" opponent_route="<<(opponent_start+oi)<<" seat="<<seat<<" seed="<<sv<<" step="<<z.step
                                 <<" own_money="<<z.own_money<<" opp_money="<<z.opp_money<<" own_units="<<z.own_units<<" opp_units="<<z.opp_units<<" own_land="<<z.own_land<<" opp_land="<<z.opp_land
                                 <<" own_past="<<z.own_pastures<<" opp_past="<<z.opp_pastures;
                        for(int a=0;a<N_ANIMALS;++a)std::cout<<" own_animal"<<a<<'='<<z.own_animals[a]<<" opp_animal"<<a<<'='<<z.opp_animals[a];
                        for(int c=0;c<N_CROPS;++c)std::cout<<" own_crop"<<c<<'='<<z.own_crops[c]<<" opp_crop"<<c<<'='<<z.opp_crops[c]<<" seed"<<c<<'='<<z.own_seeds[c];
                        for(int i=0;i<N_PRODUCTS;++i)std::cout<<" price"<<i<<'='<<z.prices[i]<<" market_inv"<<i<<'='<<z.market_inv[i];
                        std::cout<<'\n';
                    }
                }
                for(int pi=0;pi<population;++pi)for(int si=0;si<seeds;++si){
                    const int id=pi*seeds+si; const double r0=host[id*2],r1=host[id*2+1]; const double m=seat==0?r0-r1:r1-r0;
                    Fitness& f=fit[pi];++f.cases;if(m>0)++f.wins;else if(m==0)++f.ties;f.sum+=m;if(m<f.floor)f.floor=m;
                    cohort_sum[(static_cast<size_t>(pi)*opponents+oi)*2+seat]+=m;
                    seed_sum[static_cast<size_t>(pi)*seeds+si]+=m;
                    if(m>0)++seed_wins[static_cast<size_t>(pi)*seeds+si];
                    if(report_cases){const uint64_t sv=h_seed_values.empty()?seed_start+(uint64_t)si:h_seed_values[si];std::cout<<"CASE_REPORT pi="<<pi<<" opponent="<<oi<<" seat="<<seat<<" seed="<<sv<<" margin="<<m<<"\n";}
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
        if(report_seeds){
            const int seed_cases=opponents*2;
            for(int pi=0;pi<population;++pi)for(int si=0;si<seeds;++si){
                const uint64_t sv=h_seed_values.empty()?seed_start+(uint64_t)si:h_seed_values[si];
                std::cout<<"SEED_REPORT pi="<<pi<<" si="<<si<<" seed="<<sv<<" wins="<<seed_wins[(size_t)pi*seeds+si]<<"/"<<seed_cases<<" mean="<<(seed_sum[(size_t)pi*seeds+si]/seed_cases)<<"\n";
            }
        }
        std::vector<int> order(population);for(int i=0;i<population;++i)order[i]=i;
        for(int i=0;i<population;++i){
            int best=i;
            for(int j=i+1;j<population;++j) if(fitter(fit[order[j]],fit[order[best]])) best=j;
            if(best!=i){const int t=order[i];order[i]=order[best];order[best]=t;}
        }
        const int bi=order[0];
        if(!have_best||fitter(fit[bi],global_fit)){have_best=true;global_best=pop[bi];global_fit=fit[bi];}
        std::cout<<"GEN "<<gen<<" best="<<fit[bi].wins<<"/"<<fit[bi].cases<<" ties="<<fit[bi].ties<<" floor="<<fit[bi].floor<<" mean="<<mean(fit[bi])
                 <<" tail="<<fit[bi].tail_mean<<" worst_cohort="<<fit[bi].worst_cohort<<" seed_tail="<<fit[bi].seed_tail<<" worst_seed="<<fit[bi].worst_seed<<" balanced="<<balanced_score(fit[bi])
                 <<" global="<<global_fit.wins<<"/"<<global_fit.cases<<" gfloor="<<global_fit.floor<<" gmean="<<mean(global_fit)<<" gtail="<<global_fit.tail_mean<<" gworst="<<global_fit.worst_cohort<<" gseed_tail="<<global_fit.seed_tail<<" gworst_seed="<<global_fit.worst_seed<<" gbalanced="<<balanced_score(global_fit)<<"\n";
        dump_program(std::cout,pop[bi]); std::cout.flush();
        if(gen+1==generations)break;
        std::vector<GPProgram> next(population);
        for(int i=0;i<elite;++i)next[i]=pop[order[i]];
        auto tournament=[&](){const int cap=(population-1)<63?(population-1):63;int best=order[ri(rng,0,cap)];for(int k=0;k<3;++k){int q=order[ri(rng,0,cap)];if(fitter(fit[q],fit[best]))best=q;}return best;};
        for(int i=elite;i<population;++i){
            const int a=tournament(); GPProgram child=pop[a];
            if(ri(rng,0,99)<25){const int b=tournament();if(!GP_SUFFIX_SEARCH)child=crossover(pop[a],pop[b],rng);else if(GP_SUFFIX_CROSSOVER)child=crossover_suffix(pop[a],pop[b],rng);}
            next[i]=mutate(child,rng,paths.size());
        }
        pop.swap(next);
    }

    std::cout<<"FINAL_BEST wins="<<global_fit.wins<<"/"<<global_fit.cases<<" ties="<<global_fit.ties<<" floor="<<global_fit.floor<<" mean="<<mean(global_fit)<<" tail="<<global_fit.tail_mean<<" worst_cohort="<<global_fit.worst_cohort<<" seed_tail="<<global_fit.seed_tail<<" worst_seed="<<global_fit.worst_seed<<" balanced="<<balanced_score(global_fit)<<"\n";
    dump_program(std::cout,global_best);
    if(!out_path.empty()){
        std::ofstream out(out_path);out<<"wins="<<global_fit.wins<<" cases="<<global_fit.cases<<" ties="<<global_fit.ties<<" floor="<<global_fit.floor<<" mean="<<mean(global_fit)<<" tail="<<global_fit.tail_mean<<" worst_cohort="<<global_fit.worst_cohort<<" seed_tail="<<global_fit.seed_tail<<" worst_seed="<<global_fit.worst_seed<<" fitness="<<fitness_mode<<"\n";dump_program(out,global_best);
        std::ofstream bout(out_path+".bin",std::ios::binary);bout.write(reinterpret_cast<const char*>(&global_best),sizeof(GPProgram));
    }
    cudaFree(d_routes);cudaFree(d_desc);cudaFree(d_prog);if(d_manager_bank)cudaFree(d_manager_bank);if(d_program_bank)cudaFree(d_program_bank);if(d_staged_router_prefix)cudaFree(d_staged_router_prefix);if(d_early_marker_router)cudaFree(d_early_marker_router);if(d_reactive_override_router)cudaFree(d_reactive_override_router);if(d_marker_program_map)cudaFree(d_marker_program_map);if(d_state_reports)cudaFree(d_state_reports);if(d_seed_values)cudaFree(d_seed_values);cudaFree(d_sims);cudaFree(d_rewards);
    return 0;
}
