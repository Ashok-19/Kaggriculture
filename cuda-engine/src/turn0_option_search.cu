#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include "bonsai_io.hpp"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace kagcuda;

struct CandidateDesc {
    uint8_t farmer_build=0, hires=4, cows=1, sheep=2, wheat_seed=5, melon_seed=5, wheat_buy=6, order_template=0;
};
struct Score { int wins=0,cases=0; double sum=0.0,floor=DBL_MAX; };

static std::vector<std::string> read_list(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list: "+p);std::vector<std::string>o;std::string l;while(std::getline(in,l)){auto h=l.find('#');if(h!=std::string::npos)l.resize(h);auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");o.push_back(l.substr(a,b-a+1));}return o;}
static std::vector<int> parse_ints(const std::string&s){std::vector<int>o;std::stringstream ss(s);std::string x;while(std::getline(ss,x,','))if(!x.empty())o.push_back(std::stoi(x));return o;}
static void push(Action&a,uint8_t op,uint8_t item,int n){if(a.n_orders>=MAX_ORDERS||n<=0)return;a.orders[a.n_orders++]={op,item,n};}
static Action make_first(const CandidateDesc& d){
    Action a{};a.n_units=1;a.units[0].op=d.farmer_build?OP_BUILD_PASTURE:OP_PASS;
    if(d.order_template==0){
        for(int i=0;i<d.hires;++i)push(a,M_HIRE,0,1);
        push(a,M_BUY_ANIMAL,COW,d.cows);push(a,M_BUY_ANIMAL,SHEEP,d.sheep);
        push(a,M_BUY_SEED,WHEAT,d.wheat_seed);push(a,M_BUY_SEED,MELON,d.melon_seed);push(a,M_BUY_PRODUCT,WHEAT,d.wheat_buy);
    }else{
        push(a,M_BUY_PRODUCT,WHEAT,d.wheat_buy);
        for(int i=0;i<d.hires;++i)push(a,M_HIRE,0,1);
        push(a,M_BUY_ANIMAL,COW,d.cows);push(a,M_BUY_ANIMAL,SHEEP,d.sheep);
        push(a,M_BUY_SEED,MELON,d.melon_seed);push(a,M_BUY_SEED,WHEAT,d.wheat_seed);
    }
    return a;
}

__device__ __forceinline__ Action dispatch_router(const GPProgram& router,const ObservationView& obs,const GPProgram* bank,int n_bank,const Action*routes,const GPRouteDesc*descs,int n_routes){
    const GPLeaf top=gp_select_leaf(router,obs);const int e=(int)top.pad;
    if(e>=0&&e<n_bank)return gp_act(bank[e],obs,routes,descs,n_routes);
    return gp_act(bank[0],obs,routes,descs,n_routes);
}

__global__ void option_kernel(Config base,DeviceTables tables,const Action*first,int n_candidates,
                              const GPProgram*router,const GPProgram*bank,int n_bank,
                              const Action*routes,const GPRouteDesc*descs,int n_routes,int anti_route,
                              int opponent,int seat,bool anti_lane,uint64_t seed_start,int n_seeds,
                              Sim*sims,double*rewards){
    int id=blockIdx.x*blockDim.x+threadIdx.x,total=n_candidates*n_seeds;if(id>=total)return;
    int ci=id/n_seeds,si=id-ci*n_seeds;Config cfg=base;cfg.seed=seed_start+(uint64_t)si;Sim&sim=sims[id];sim.init(cfg,&tables);int os=1-seat;
    while(!sim.st.done){int st=sim.st.step;Action cand{};
        if(st==0)cand=gp_copy_action(first[ci],sim.st.farms[seat].n_units);
        else if(anti_lane)cand=gp_copy_action(gp_route_at(routes,descs[anti_route],st),sim.st.farms[seat].n_units);
        else cand=dispatch_router(*router,make_observation(sim,seat),bank,n_bank,routes,descs,n_routes);
        Action opp=gp_copy_action(gp_route_at(routes,descs[opponent],st),sim.st.farms[os].n_units);
        if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    rewards[id*2]=sim.st.farms[0].money;rewards[id*2+1]=sim.st.farms[1].money;
}

int main(int argc,char**argv){
    std::string routes_path,router_path,out_path,main_arg,anti_arg;std::vector<std::string>bank_paths;int anti_route=50,seeds=16,tpb=128;uint64_t seed_start=710000;bool wide=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
        if(a=="--routes")routes_path=need();else if(a=="--router")router_path=need();else if(a=="--program")bank_paths.push_back(need());else if(a=="--main-opponents")main_arg=need();else if(a=="--anti-opponents")anti_arg=need();else if(a=="--anti-route")anti_route=std::stoi(need());else if(a=="--seeds")seeds=std::stoi(need());else if(a=="--seed-start")seed_start=std::stoull(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--out")out_path=need();else if(a=="--wide")wide=true;else if(a=="--help"){std::cout<<"turn0_option_search --routes LIST --router ROUTER.bin --program P.bin (repeat) --main-opponents csv --anti-opponents csv [--anti-route 50 --seeds 16 --seed-start N --wide --out CSV]\n";return 0;}else throw std::runtime_error("unknown arg: "+a);
    }
    if(routes_path.empty()||router_path.empty()||bank_paths.empty()||main_arg.empty()||anti_arg.empty())throw std::runtime_error("missing required arguments");
    auto paths=read_list(routes_path);auto main_opps=parse_ints(main_arg),anti_opps=parse_ints(anti_arg);if(anti_route<0||anti_route>=(int)paths.size())throw std::runtime_error("bad anti route");
    std::vector<NativeTape>tapes;for(auto&p:paths)tapes.push_back(load_native_tape_json(p));std::vector<Action>hr;std::vector<GPRouteDesc>hd;for(auto&t:tapes){int o=hr.size();hr.insert(hr.end(),t.actions.begin(),t.actions.end());hd.push_back({o,(int)t.actions.size()});}
    GPProgram router=load_gp_program_binary(router_path);std::vector<GPProgram>bank;for(auto&p:bank_paths)bank.push_back(load_gp_program_binary(p));
    std::vector<CandidateDesc> desc;std::vector<Action> first;
    const std::vector<int> hires = wide ? std::vector<int>{3,4,5} : std::vector<int>{4,5};
    const std::vector<int> cows = {1,2};
    const std::vector<int> sheep = wide ? std::vector<int>{2,3,4} : std::vector<int>{2,4};
    const std::vector<int> wheat_seed = wide ? std::vector<int>{4,5,6,7,8} : std::vector<int>{5,7};
    const std::vector<int> melon_seed = wide ? std::vector<int>{4,5,8,10,12} : std::vector<int>{5,12};
    const std::vector<int> wheat_buy = wide ? std::vector<int>{4,6,8,10} : std::vector<int>{6,8};
    for(int fb=0;fb<2;++fb)for(int h:hires)for(int c:cows)for(int sh:sheep)for(int ws:wheat_seed)for(int ms:melon_seed)for(int wb:wheat_buy)for(int ot=0;ot<2;++ot){CandidateDesc d{};d.farmer_build=fb;d.hires=h;d.cows=c;d.sheep=sh;d.wheat_seed=ws;d.melon_seed=ms;d.wheat_buy=wb;d.order_template=ot;desc.push_back(d);first.push_back(make_first(d));}
    const int nc=desc.size(),jobs=nc*seeds;configure_device();TablesOwner tables;Config cfg;Action*df=nullptr,*dr=nullptr;GPRouteDesc*dd=nullptr;GPProgram*drouter=nullptr,*dbank=nullptr;Sim*ds=nullptr;double*drew=nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&df,nc*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&drouter,sizeof(GPProgram)));KAGCUDA_CHECK(cudaMalloc((void**)&dbank,bank.size()*sizeof(GPProgram)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&drew,(size_t)jobs*2*sizeof(double)));
    KAGCUDA_CHECK(cudaMemcpy(df,first.data(),nc*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(drouter,&router,sizeof(router),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dbank,bank.data(),bank.size()*sizeof(GPProgram),cudaMemcpyHostToDevice));
    std::vector<Score> anti(nc),main(nc);std::vector<double>host((size_t)jobs*2);
    for(int lane=0;lane<2;++lane){
        const std::vector<int>& opps = lane==0 ? anti_opps : main_opps;
        const bool anti_lane = lane==0;
        std::vector<Score>& score = lane==0 ? anti : main;
        for(int oi:opps)for(int seat=0;seat<2;++seat){
            option_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,df,nc,drouter,dbank,bank.size(),dr,dd,paths.size(),anti_route,oi,seat,anti_lane,seed_start,seeds,ds,drew);
            KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(host.data(),drew,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
            for(int ci=0;ci<nc;++ci)for(int si=0;si<seeds;++si){int id=ci*seeds+si;double m=seat==0?host[id*2]-host[id*2+1]:host[id*2+1]-host[id*2];Score&sc=score[ci];++sc.cases;sc.wins+=m>0;sc.sum+=m;if(m<sc.floor)sc.floor=m;}
        }
    }
    std::vector<int>order(nc);for(int i=0;i<nc;++i)order[i]=i;
    for(int i=1;i<nc;++i){int x=order[i],j=i-1;while(j>=0){int y=order[j];double xaw=(double)anti[x].wins/anti[x].cases,xmw=(double)main[x].wins/main[x].cases,yaw=(double)anti[y].wins/anti[y].cases,ymw=(double)main[y].wins/main[y].cases;double xmin=xaw<xmw?xaw:xmw,ymin=yaw<ymw?yaw:ymw,xavg=(xaw+xmw)/2,yavg=(yaw+ymw)/2;double xs=anti[x].sum/anti[x].cases+main[x].sum/main[x].cases,ys=anti[y].sum/anti[y].cases+main[y].sum/main[y].cases;bool better=(xmin>ymin)||(xmin==ymin&&xavg>yavg)||(xmin==ymin&&xavg==yavg&&xs>ys);if(!better)break;order[j+1]=order[j];--j;}order[j+1]=x;}
    std::ostream*os=&std::cout;std::ofstream fout;if(!out_path.empty()){fout.open(out_path);if(!fout)throw std::runtime_error("cannot open out");os=&fout;}*os<<"rank,id,build,hires,cows,sheep,wseed,mseed,wbuy,order,anti_wins,anti_cases,anti_mean,anti_floor,main_wins,main_cases,main_mean,main_floor,min_wr,avg_wr\n";
    for(int r=0;r<nc;++r){int i=order[r];auto&d=desc[i];double aw=(double)anti[i].wins/anti[i].cases,mw=(double)main[i].wins/main[i].cases;*os<<r<<','<<i<<','<<int(d.farmer_build)<<','<<int(d.hires)<<','<<int(d.cows)<<','<<int(d.sheep)<<','<<int(d.wheat_seed)<<','<<int(d.melon_seed)<<','<<int(d.wheat_buy)<<','<<int(d.order_template)<<','<<anti[i].wins<<','<<anti[i].cases<<','<<anti[i].sum/anti[i].cases<<','<<anti[i].floor<<','<<main[i].wins<<','<<main[i].cases<<','<<main[i].sum/main[i].cases<<','<<main[i].floor<<','<<(aw<mw?aw:mw)<<','<<(aw+mw)/2<<'\n';}
    for(int r=0;r<(20<nc?20:nc);++r){int i=order[r];auto&d=desc[i];std::cout<<"TOP "<<r<<" id="<<i<<" build="<<int(d.farmer_build)<<" h="<<int(d.hires)<<" c="<<int(d.cows)<<" s="<<int(d.sheep)<<" ws="<<int(d.wheat_seed)<<" ms="<<int(d.melon_seed)<<" wb="<<int(d.wheat_buy)<<" ord="<<int(d.order_template)<<" anti="<<anti[i].wins<<'/'<<anti[i].cases<<" mean="<<anti[i].sum/anti[i].cases<<" main="<<main[i].wins<<'/'<<main[i].cases<<" mean="<<main[i].sum/main[i].cases<<" minwr="<<(((double)anti[i].wins/anti[i].cases)<((double)main[i].wins/main[i].cases)?((double)anti[i].wins/anti[i].cases):((double)main[i].wins/main[i].cases))<<"\n";}
    cudaFree(df);cudaFree(dr);cudaFree(dd);cudaFree(drouter);cudaFree(dbank);cudaFree(ds);cudaFree(drew);return 0;
}
