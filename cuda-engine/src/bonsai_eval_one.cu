#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace kagcuda;

__global__ void eval_one_kernel(Config cfg, DeviceTables tables, const GPProgram* prog,
                                const Action* routes, const GPRouteDesc* descs, int n_routes,
                                int opponent, int seat, int stop_step, Sim* out_sim, Action* out_action) {
    if(blockIdx.x||threadIdx.x)return;
    Sim& sim=*out_sim; sim.init(cfg,&tables);
    while(!sim.st.done && sim.st.step < stop_step){
        const int t=sim.st.step, os=1-seat;
        Action cand=gp_act(*prog,make_observation(sim,seat),routes,descs,n_routes);
        Action opp=gp_copy_action(gp_route_at(routes,descs[opponent],t),sim.st.farms[os].n_units);
        if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    if(!sim.st.done){*out_action=gp_act(*prog,make_observation(sim,seat),routes,descs,n_routes);}
}

static std::vector<std::string> read_list(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list");std::vector<std::string>o;std::string l;while(std::getline(in,l)){auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");o.push_back(l.substr(a,b-a+1));}return o;}

static void farm_summary(const Farm&f,const char*name){
    int crops[N_CROPS]={0},animals[N_ANIMALS]={0},past=0,coop=0,weeds=0,empty=0;
    for(int y=0;y<BOARD;++y)for(int x=0;x<BOARD;++x){const Tile&t=f.tiles[y][x];if(t.kind==T_PLANT&&t.what<N_CROPS)++crops[t.what];else if(t.kind==T_PASTURE){++past;if(t.has_animal&&t.what>=GOOSE&&t.what<=SHEEP)++animals[t.what-GOOSE];}else if(t.kind==T_COOP){++coop;if(t.has_animal&&t.what>=GOOSE&&t.what<=SHEEP)++animals[t.what-GOOSE];}else if(t.kind==T_WEED)++weeds;else if(t.kind==T_EMPTY)++empty;}
    long long discarded=0;for(int i=0;i<N_ITEMS;++i)discarded+=f.discarded[i];
    std::cout<<name<<" money="<<f.money<<" spend="<<f.total_spend<<" revenue="<<f.sell_revenue<<" land="<<f.n_quadrants<<" shed="<<f.shed_total<<" discarded="<<discarded<<"\n";
    std::cout<<" crops W,C,T,S,M=";for(int i=0;i<N_CROPS;++i)std::cout<<crops[i]<<(i+1==N_CROPS?'\n':',');
    std::cout<<" structures pasture="<<past<<" coop="<<coop<<" animals G,C,S="<<animals[0]<<','<<animals[1]<<','<<animals[2]<<" weeds="<<weeds<<" empty="<<empty<<"\n";
    std::cout<<" produced=";for(int i=0;i<N_PRODUCTS;++i)std::cout<<f.produced[i]<<(i+1==N_PRODUCTS?'\n':',');
    std::cout<<" sold=";for(int i=0;i<N_PRODUCTS;++i)std::cout<<f.sold_units[i]<<(i+1==N_PRODUCTS?'\n':',');
    std::cout<<" shed_items=";for(int i=0;i<N_ITEMS;++i)std::cout<<f.shed[i]<<(i+1==N_ITEMS?'\n':',');
    std::cout<<" seeds=";for(int i=0;i<N_CROPS;++i)std::cout<<f.seeds[i]<<(i+1==N_CROPS?'\n':',');
    int carried[N_ITEMS]={0};for(int u=0;u<f.n_units;++u)for(int i=0;i<N_ITEMS;++i)carried[i]+=f.inv[u][i];
    std::cout<<" carried=";for(int i=0;i<N_ITEMS;++i)std::cout<<carried[i]<<(i+1==N_ITEMS?'\n':',');
    std::cout<<" workers="<<f.n_units<<" hires_today="<<f.hires_today<<" positions=";for(int u=0;u<f.n_units;++u)std::cout<<'('<<int(f.pos_x[u])<<','<<int(f.pos_y[u])<<')'<<(u+1==f.n_units?'\n':' ');
}

static const char* op_name(uint8_t op){
    static const char* names[]={"PASS","NORTH","SOUTH","WEST","EAST","PICKUP","DROP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","INVALID"};
    return op<sizeof(names)/sizeof(names[0])?names[op]:"?";
}
static const char* mop_name(uint8_t op){static const char* names[]={"NONE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};return op<sizeof(names)/sizeof(names[0])?names[op]:"?";}

int main(int argc,char**argv){
    std::string list,program;int opponent=0,seat=0,stop_step=719;uint64_t seed=63400;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("missing value");if(a=="--routes")list=argv[++i];else if(a=="--program")program=argv[++i];else if(a=="--opponent")opponent=std::stoi(argv[++i]);else if(a=="--seat")seat=std::stoi(argv[++i]);else if(a=="--seed")seed=std::stoull(argv[++i]);else if(a=="--stop-step")stop_step=std::stoi(argv[++i]);else throw std::runtime_error("unknown arg "+a);}
    if(list.empty()||program.empty())throw std::runtime_error("--routes and --program required");auto paths=read_list(list);if(opponent<0||opponent>=(int)paths.size())throw std::runtime_error("bad opponent");
    GPProgram hp{};{std::ifstream in(program,std::ios::binary);if(!in)throw std::runtime_error("cannot open program");in.read((char*)&hp,sizeof(hp));if(in.gcount()!=(std::streamsize)sizeof(hp))throw std::runtime_error("bad program size");}
    std::vector<NativeTape>t;for(auto&p:paths)t.push_back(load_native_tape_json(p));std::vector<Action>ha;std::vector<GPRouteDesc>hd;for(auto&x:t){int o=ha.size();ha.insert(ha.end(),x.actions.begin(),x.actions.end());hd.push_back({o,(int)x.actions.size()});}
    configure_device();TablesOwner tables;Config cfg;cfg.seed=seed;GPProgram*dp=nullptr;Action*da=nullptr;GPRouteDesc*dd=nullptr;Sim*ds=nullptr;Action*dnext=nullptr;KAGCUDA_CHECK(cudaMalloc((void**)&dp,sizeof(hp)));KAGCUDA_CHECK(cudaMalloc((void**)&da,ha.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dnext,sizeof(Action)));KAGCUDA_CHECK(cudaMemset(dnext,0,sizeof(Action)));KAGCUDA_CHECK(cudaMemcpy(dp,&hp,sizeof(hp),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(da,ha.data(),ha.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    eval_one_kernel<<<1,1>>>(cfg,tables.dev,dp,da,dd,paths.size(),opponent,seat,stop_step,ds,dnext);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());Sim hs{};KAGCUDA_CHECK(cudaMemcpy(&hs,ds,sizeof(Sim),cudaMemcpyDeviceToHost));
    Action hn{};KAGCUDA_CHECK(cudaMemcpy(&hn,dnext,sizeof(Action),cudaMemcpyDeviceToHost));
    std::cout<<"seed="<<seed<<" candidate_seat="<<seat<<" opponent="<<opponent<<":"<<paths[opponent]<<" final_step="<<hs.st.step<<"\n";farm_summary(hs.st.farms[seat],"candidate");farm_summary(hs.st.farms[1-seat],"opponent");
    if(!hs.st.done){std::cout<<"next_action units="<<int(hn.n_units)<<" ";for(int u=0;u<hn.n_units;++u){std::cout<<u<<':'<<op_name(hn.units[u].op);if(hn.units[u].op==OP_PICKUP||hn.units[u].op==OP_PLACE||hn.units[u].op==OP_PLANT)std::cout<<'('<<int(hn.units[u].arg)<<','<<hn.units[u].n<<')';std::cout<<' ';}std::cout<<" market=";for(int i=0;i<hn.n_orders;++i)std::cout<<mop_name(hn.orders[i].op)<<'('<<int(hn.orders[i].item)<<','<<hn.orders[i].n<<") ";std::cout<<'\n';}
    cudaFree(dp);cudaFree(da);cudaFree(dd);cudaFree(ds);cudaFree(dnext);return 0;
}
