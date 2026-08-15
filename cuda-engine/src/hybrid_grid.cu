#include "runtime.cuh"
#include "tape_io.hpp"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

struct RouteDesc { int offset; int length; };

__device__ __forceinline__ const Action& at(const Action* actions, RouteDesc d, int step) {
    const int i = step < d.length ? step : d.length - 1;
    return actions[d.offset + i];
}

__device__ __forceinline__ Action hybrid_action(const Action* actions, const RouteDesc* descs,
                                                int unit_route, int market_route,
                                                int step, int n_units) {
    const Action& ua = at(actions,descs[unit_route],step);
    const Action& ma = at(actions,descs[market_route],step);
    Action out = ua;
    out.n_units = n_units;
    out.n_orders = ma.n_orders;
    for (int i=0;i<ma.n_orders && i<MAX_ORDERS;++i) out.orders[i]=ma.orders[i];
    return out;
}

__global__ void hybrid_grid_kernel(Config base, DeviceTables tables,
                                   const Action* routes, const RouteDesc* descs, int n_routes,
                                   int opponent_route, int candidate_seat,
                                   uint64_t seed_start, int n_seeds,
                                   Sim* sims, double* rewards) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int n_candidates=n_routes*n_routes;
    const int total=n_candidates*n_seeds;
    if(id>=total)return;
    const int candidate=id/n_seeds;
    const int si=id-candidate*n_seeds;
    const int unit_route=candidate/n_routes;
    const int market_route=candidate-unit_route*n_routes;
    Config cfg=base; cfg.seed=seed_start+static_cast<uint64_t>(si);
    Sim& sim=sims[id]; sim.init(cfg,&tables);
    while(!sim.st.done){
        const int step=sim.st.step;
        const int opp_seat=1-candidate_seat;
        Action cand=hybrid_action(routes,descs,unit_route,market_route,step,sim.st.farms[candidate_seat].n_units);
        Action opp=at(routes,descs[opponent_route],step); opp.n_units=sim.st.farms[opp_seat].n_units;
        if(candidate_seat==0) sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    rewards[id*2]=sim.st.farms[0].money; rewards[id*2+1]=sim.st.farms[1].money;
}

struct Fit {int wins=0,ties=0,cases=0;double floor=DBL_MAX,sum=0;};
static bool better(const Fit&a,const Fit&b){if(a.wins!=b.wins)return a.wins>b.wins;if(a.ties!=b.ties)return a.ties<b.ties;if(a.floor!=b.floor)return a.floor>b.floor;return a.sum>b.sum;}
static double avg(const Fit&f){return f.cases?f.sum/f.cases:0;}

static std::vector<std::string> read_list(const std::string& path){
    std::ifstream in(path);if(!in)throw std::runtime_error("cannot open list: "+path);
    std::vector<std::string> out;std::string line;
    while(std::getline(in,line)){const auto h=line.find('#');if(h!=std::string::npos)line.resize(h);const auto a=line.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;const auto b=line.find_last_not_of(" \t\r\n");out.push_back(line.substr(a,b-a+1));}
    return out;
}

int main(int argc,char**argv){
    std::string list;int seeds=4,opponents=12,tpb=64,topn=30;uint64_t seed_start=63200;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(i+1>=argc&&a!="--help")throw std::runtime_error("missing value for "+a);if(a=="--routes")list=argv[++i];else if(a=="--seeds")seeds=std::stoi(argv[++i]);else if(a=="--opponents")opponents=std::stoi(argv[++i]);else if(a=="--seed-start")seed_start=std::stoull(argv[++i]);else if(a=="--tpb")tpb=std::stoi(argv[++i]);else if(a=="--top")topn=std::stoi(argv[++i]);else if(a=="--help"){std::cout<<"hybrid_grid --routes LIST [--seeds N --opponents N --seed-start S --top N]\n";return 0;}else throw std::runtime_error("unknown arg "+a);}
    if(list.empty())throw std::runtime_error("--routes required");
    auto paths=read_list(list);const int n=paths.size();if(n<=0)throw std::runtime_error("empty route list");if(opponents>n)opponents=n;
    std::vector<NativeTape> tapes;for(const auto&p:paths)tapes.push_back(load_native_tape_json(p));
    std::vector<Action> ha;std::vector<RouteDesc> hd;for(const auto&t:tapes){int off=ha.size();ha.insert(ha.end(),t.actions.begin(),t.actions.end());hd.push_back({off,(int)t.actions.size()});}
    configure_device();TablesOwner tables;Config cfg;Action*da=nullptr;RouteDesc*dd=nullptr;Sim*ds=nullptr;double*dr=nullptr;
    const int candidates=n*n;const int jobs=candidates*seeds;
    cudaDeviceProp prop{};int dev=0;KAGCUDA_CHECK(cudaGetDevice(&dev));KAGCUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    const size_t sim_bytes=(size_t)jobs*sizeof(Sim);if(sim_bytes>(size_t)(prop.totalGlobalMem*0.72))throw std::runtime_error("requested grid exceeds safe GPU memory; reduce --seeds or routes");
    KAGCUDA_CHECK(cudaMalloc((void**)&da,ha.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(RouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,sim_bytes));KAGCUDA_CHECK(cudaMalloc((void**)&dr,(size_t)jobs*2*sizeof(double)));
    KAGCUDA_CHECK(cudaMemcpy(da,ha.data(),ha.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(RouteDesc),cudaMemcpyHostToDevice));
    std::vector<double> host((size_t)jobs*2);std::vector<Fit> fit(candidates);
    for(int oi=0;oi<opponents;++oi)for(int seat=0;seat<2;++seat){
        hybrid_grid_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,da,dd,n,oi,seat,seed_start,seeds,ds,dr);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(host.data(),dr,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(int c=0;c<candidates;++c)for(int s=0;s<seeds;++s){const int id=c*seeds+s;const double r0=host[id*2],r1=host[id*2+1],m=seat==0?r0-r1:r1-r0;Fit&f=fit[c];++f.cases;if(m>0)++f.wins;else if(m==0)++f.ties;f.sum+=m;if(m<f.floor)f.floor=m;}
    }
    std::vector<int> order(candidates);for(int i=0;i<candidates;++i)order[i]=i;
    const int emit=topn<candidates?topn:candidates;
    for(int i=0;i<emit;++i){int bi=i;for(int j=i+1;j<candidates;++j)if(better(fit[order[j]],fit[order[bi]]))bi=j;if(bi!=i){int t=order[i];order[i]=order[bi];order[bi]=t;}const int c=order[i],u=c/n,m=c-u*n;std::cout<<"RANK "<<i+1<<" wins="<<fit[c].wins<<"/"<<fit[c].cases<<" ties="<<fit[c].ties<<" floor="<<fit[c].floor<<" mean="<<avg(fit[c])<<" unit="<<u<<":"<<paths[u]<<" market="<<m<<":"<<paths[m]<<"\n";}
    cudaFree(da);cudaFree(dd);cudaFree(ds);cudaFree(dr);return 0;
}
