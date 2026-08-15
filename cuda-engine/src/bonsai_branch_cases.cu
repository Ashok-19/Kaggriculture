#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

constexpr int MAX_BRANCH_EXPERTS = 8;

struct BranchFeatures {
    int step = 0, day = 0, hour = 0;
    int n_shops = 0;
    int shops[MAX_SHOP_INSTANCES] = {0};
    double own_money = 0.0, opp_money = 0.0;
    int own_units = 0, opp_units = 0;
    int own_land = 0, opp_land = 0;
    int own_hires = 0, opp_hires = 0;
    int prices[N_PRODUCTS] = {0};
    int market_inv[N_PRODUCTS] = {0};
    int shed[N_ITEMS] = {0};
    int seeds[N_CROPS] = {0};
    int carried[N_ITEMS] = {0};
    int own_crops[N_CROPS] = {0};
    int opp_crops[N_CROPS] = {0};
    int own_animals[N_ANIMALS] = {0};
    int opp_animals[N_ANIMALS] = {0};
    int own_pastures = 0, opp_pastures = 0;
    int own_coops = 0, opp_coops = 0;
    int own_weeds = 0, opp_weeds = 0;
};

__device__ __forceinline__ void capture_public_farm(const PublicFarmView& f,
                                                    int* crops, int* animals,
                                                    int& pastures, int& coops, int& weeds) {
    for (int y=0;y<BOARD;++y) for (int x=0;x<BOARD;++x) {
        const Tile& t=f.tile(x,y);
        if (t.kind==T_PLANT && t.what<N_CROPS) ++crops[t.what];
        else if (t.kind==T_PASTURE) {
            ++pastures;
            if (t.has_animal && t.what>=GOOSE && t.what<=SHEEP) ++animals[t.what-GOOSE];
        } else if (t.kind==T_COOP) {
            ++coops;
            if (t.has_animal && t.what>=GOOSE && t.what<=SHEEP) ++animals[t.what-GOOSE];
        } else if (t.kind==T_WEED) ++weeds;
    }
}

__device__ __forceinline__ BranchFeatures capture_features(const ObservationView& obs) {
    BranchFeatures z{};
    const int me=obs.player(), opp=1-me;
    const auto own=obs.farm(me), other=obs.farm(opp);
    const auto priv=obs.private_farm();
    const auto market=obs.market();
    const auto town=obs.town();
    z.step=obs.step();z.day=obs.day();z.hour=obs.hour();z.n_shops=town.n_shops();
    for(int i=0;i<z.n_shops && i<MAX_SHOP_INSTANCES;++i)z.shops[i]=town.shop(i);
    z.own_money=own.money();z.opp_money=other.money();
    z.own_units=own.n_units();z.opp_units=other.n_units();
    z.own_land=own.n_quadrants();z.opp_land=other.n_quadrants();
    z.own_hires=own.hires_today();z.opp_hires=other.hires_today();
    for(int i=0;i<N_PRODUCTS;++i){z.prices[i]=market.price(i);z.market_inv[i]=market.inventory(i);}
    for(int i=0;i<N_ITEMS;++i)z.shed[i]=priv.shed(i);
    for(int i=0;i<N_CROPS;++i)z.seeds[i]=priv.seed(i);
    for(int u=0;u<priv.n_units();++u)for(int i=0;i<N_ITEMS;++i)z.carried[i]+=priv.inventory(u,i);
    capture_public_farm(own,z.own_crops,z.own_animals,z.own_pastures,z.own_coops,z.own_weeds);
    capture_public_farm(other,z.opp_crops,z.opp_animals,z.opp_pastures,z.opp_coops,z.opp_weeds);
    return z;
}

__global__ void branch_eval_kernel(Config base, DeviceTables tables,
                                   const GPProgram* prefix,
                                   const GPProgram* experts, int n_experts,
                                   const Action* routes, const GPRouteDesc* descs, int n_routes,
                                   int opponent_route, int candidate_seat, int branch_step,
                                   uint64_t seed_start, const uint64_t* seed_values, int n_seeds,
                                   Sim* sims, double* rewards, BranchFeatures* features) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int total=n_experts*n_seeds;
    if(id>=total)return;
    const int ei=id/n_seeds, si=id-ei*n_seeds;
    Config cfg=base;cfg.seed=seed_values?seed_values[si]:seed_start+(uint64_t)si;
    Sim& sim=sims[id];sim.init(cfg,&tables);
    const GPRouteDesc od=descs[opponent_route];
    bool captured=false;
    while(!sim.st.done){
        const int step=sim.st.step, os=1-candidate_seat;
        const ObservationView obs=make_observation(sim,candidate_seat);
        if(!captured && step>=branch_step){if(ei==0)features[si]=capture_features(obs);captured=true;}
        Action cand = step<branch_step ? gp_act(*prefix,obs,routes,descs,n_routes) : gp_act(experts[ei],obs,routes,descs,n_routes);
        Action opp=gp_copy_action(gp_route_at(routes,od,step),sim.st.farms[os].n_units);
        if(candidate_seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    rewards[id*2]=sim.st.farms[0].money;rewards[id*2+1]=sim.st.farms[1].money;
}

static std::vector<std::string> read_list(const std::string& p){
    std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list: "+p);
    std::vector<std::string> out;std::string l;
    while(std::getline(in,l)){const auto h=l.find('#');if(h!=std::string::npos)l.resize(h);const auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;const auto b=l.find_last_not_of(" \t\r\n");out.push_back(l.substr(a,b-a+1));}
    return out;
}
static GPProgram load_program(const std::string& p){GPProgram q{};std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open program: "+p);in.read((char*)&q,sizeof(q));if(in.gcount()!=(std::streamsize)sizeof(q))throw std::runtime_error("bad program size: "+p);return q;}
static std::vector<uint64_t> parse_seeds(const std::string& s){std::vector<uint64_t> out;std::string x;for(char c:s){if(c==','){if(!x.empty()){out.push_back(std::stoull(x));x.clear();}}else x.push_back(c);}if(!x.empty())out.push_back(std::stoull(x));return out;}

static void csv_feature_header(std::ostream& o){
    o<<"opponent,seat,seed,branch_step,day,hour,n_shops";
    for(int i=0;i<MAX_SHOP_INSTANCES;++i)o<<",shop"<<(i+1);
    o<<",own_money,opp_money,own_units,opp_units,own_land,opp_land,own_hires,opp_hires";
    for(int i=0;i<N_PRODUCTS;++i)o<<",price"<<i;
    for(int i=0;i<N_PRODUCTS;++i)o<<",market_inv"<<i;
    for(int i=0;i<N_ITEMS;++i)o<<",shed"<<i;
    for(int i=0;i<N_CROPS;++i)o<<",seed"<<i;
    for(int i=0;i<N_ITEMS;++i)o<<",carried"<<i;
    for(int i=0;i<N_CROPS;++i)o<<",own_crop"<<i;
    for(int i=0;i<N_CROPS;++i)o<<",opp_crop"<<i;
    for(int i=0;i<N_ANIMALS;++i)o<<",own_animal"<<i;
    for(int i=0;i<N_ANIMALS;++i)o<<",opp_animal"<<i;
    o<<",own_pastures,opp_pastures,own_coops,opp_coops,own_weeds,opp_weeds";
}
static void csv_features(std::ostream& o,const BranchFeatures& z){
    o<<','<<z.day<<','<<z.hour<<','<<z.n_shops;
    for(int i=0;i<MAX_SHOP_INSTANCES;++i)o<<','<<(i<z.n_shops?z.shops[i]:-1);
    o<<','<<z.own_money<<','<<z.opp_money<<','<<z.own_units<<','<<z.opp_units<<','<<z.own_land<<','<<z.opp_land<<','<<z.own_hires<<','<<z.opp_hires;
    for(int i=0;i<N_PRODUCTS;++i)o<<','<<z.prices[i];
    for(int i=0;i<N_PRODUCTS;++i)o<<','<<z.market_inv[i];
    for(int i=0;i<N_ITEMS;++i)o<<','<<z.shed[i];
    for(int i=0;i<N_CROPS;++i)o<<','<<z.seeds[i];
    for(int i=0;i<N_ITEMS;++i)o<<','<<z.carried[i];
    for(int i=0;i<N_CROPS;++i)o<<','<<z.own_crops[i];
    for(int i=0;i<N_CROPS;++i)o<<','<<z.opp_crops[i];
    for(int i=0;i<N_ANIMALS;++i)o<<','<<z.own_animals[i];
    for(int i=0;i<N_ANIMALS;++i)o<<','<<z.opp_animals[i];
    o<<','<<z.own_pastures<<','<<z.opp_pastures<<','<<z.own_coops<<','<<z.opp_coops<<','<<z.own_weeds<<','<<z.opp_weeds;
}

int main(int argc,char**argv){
    std::string routes_path,prefix_path,out_path,seed_list;
    std::vector<std::string> expert_paths;
    int branch_step=-1,seeds=32,opponents=8,tpb=64;uint64_t seed_start=72000;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
        if(a=="--routes")routes_path=need();else if(a=="--prefix")prefix_path=need();else if(a=="--expert")expert_paths.push_back(need());
        else if(a=="--branch-step")branch_step=std::stoi(need());else if(a=="--seeds")seeds=std::stoi(need());else if(a=="--seed-start")seed_start=std::stoull(need());
        else if(a=="--seed-list")seed_list=need();else if(a=="--opponents")opponents=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--out")out_path=need();
        else if(a=="--help"){std::cout<<"bonsai_branch_cases --routes LIST --prefix PROGRAM.bin --expert PROGRAM.bin (repeat) --branch-step N [--seed-list a,b,c | --seeds N --seed-start S] [--opponents N --out CSV]\n";return 0;}
        else throw std::runtime_error("unknown arg: "+a);
    }
    if(routes_path.empty()||prefix_path.empty()||expert_paths.empty()||branch_step<0||out_path.empty())throw std::runtime_error("--routes, --prefix, >=1 --expert, --branch-step, --out required");
    if(expert_paths.size()>MAX_BRANCH_EXPERTS)throw std::runtime_error("too many experts");
    auto paths=read_list(routes_path);if(paths.empty())throw std::runtime_error("empty routes");if(opponents>(int)paths.size())opponents=paths.size();
    auto explicit_seeds=parse_seeds(seed_list);if(!explicit_seeds.empty())seeds=explicit_seeds.size();
    GPProgram prefix=load_program(prefix_path);std::vector<GPProgram> experts;for(auto&p:expert_paths)experts.push_back(load_program(p));
    std::vector<NativeTape> tapes;for(auto&p:paths)tapes.push_back(load_native_tape_json(p));
    std::vector<Action> hr;std::vector<GPRouteDesc> hd;for(auto&t:tapes){int off=hr.size();hr.insert(hr.end(),t.actions.begin(),t.actions.end());hd.push_back({off,(int)t.actions.size()});}
    configure_device();TablesOwner tables;Config cfg;
    Action*dr=nullptr;GPRouteDesc*dd=nullptr;GPProgram*dprefix=nullptr;GPProgram*dexp=nullptr;uint64_t*dseeds=nullptr;Sim*ds=nullptr;double*dreward=nullptr;BranchFeatures*dfeat=nullptr;
    const int jobs=experts.size()*seeds;
    KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));
    KAGCUDA_CHECK(cudaMalloc((void**)&dprefix,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMalloc((void**)&dexp,experts.size()*sizeof(GPProgram)));
    KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dreward,(size_t)jobs*2*sizeof(double)));KAGCUDA_CHECK(cudaMalloc((void**)&dfeat,(size_t)seeds*sizeof(BranchFeatures)));
    KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    KAGCUDA_CHECK(cudaMemcpy(dprefix,&prefix,sizeof(GPProgram),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dexp,experts.data(),experts.size()*sizeof(GPProgram),cudaMemcpyHostToDevice));
    if(!explicit_seeds.empty()){KAGCUDA_CHECK(cudaMalloc((void**)&dseeds,explicit_seeds.size()*sizeof(uint64_t)));KAGCUDA_CHECK(cudaMemcpy(dseeds,explicit_seeds.data(),explicit_seeds.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));}
    std::vector<double> rewards((size_t)jobs*2);std::vector<BranchFeatures> feats(seeds);
    std::ofstream out(out_path);if(!out)throw std::runtime_error("cannot open output: "+out_path);
    csv_feature_header(out);for(size_t e=0;e<experts.size();++e)out<<",expert"<<e<<"_margin";out<<",best_expert,best_margin\n";
    for(int oi=0;oi<opponents;++oi)for(int seat=0;seat<2;++seat){
        KAGCUDA_CHECK(cudaMemset(dfeat,0,(size_t)seeds*sizeof(BranchFeatures)));
        branch_eval_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dprefix,dexp,experts.size(),dr,dd,paths.size(),oi,seat,branch_step,seed_start,dseeds,seeds,ds,dreward,dfeat);
        KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());
        KAGCUDA_CHECK(cudaMemcpy(rewards.data(),dreward,rewards.size()*sizeof(double),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(feats.data(),dfeat,feats.size()*sizeof(BranchFeatures),cudaMemcpyDeviceToHost));
        for(int si=0;si<seeds;++si){
            const uint64_t seed=explicit_seeds.empty()?seed_start+(uint64_t)si:explicit_seeds[si];
            out<<oi<<','<<seat<<','<<seed<<','<<branch_step;csv_features(out,feats[si]);
            int best=0;double bm=-1e300;
            for(int ei=0;ei<(int)experts.size();++ei){const int id=ei*seeds+si;const double r0=rewards[id*2],r1=rewards[id*2+1];const double m=seat==0?r0-r1:r1-r0;out<<','<<m;if(m>bm){bm=m;best=ei;}}
            out<<','<<best<<','<<bm<<'\n';
        }
    }
    cudaFree(dr);cudaFree(dd);cudaFree(dprefix);cudaFree(dexp);if(dseeds)cudaFree(dseeds);cudaFree(ds);cudaFree(dreward);cudaFree(dfeat);
    std::cout<<"WROTE "<<out_path<<" cases="<<(opponents*2*seeds)<<" experts="<<experts.size()<<" branch_step="<<branch_step<<"\n";
    return 0;
}
