#include "runtime.cuh"
#include "tape_io.hpp"
#include <cfloat>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace kagcuda;

struct RouteDesc { int offset; int length; };
struct TimingGene { uint8_t q[4]={0,0,0,0}; uint8_t wheat_first120=0; };
struct Fit { int wins=0,ties=0,cases=0; double floor=DBL_MAX,sum=0.0; };

__device__ __forceinline__ const Action& at(const Action* a,RouteDesc d,int step){int i=step<d.length?step:d.length-1;return a[d.offset+i];}
__device__ __forceinline__ bool is_wheat_sell(const Order&o){return o.op==M_SELL && o.item==WHEAT;}

__device__ __forceinline__ Action patched_action(const Action* routes,const RouteDesc* descs,int route,int step,int n_units,const TimingGene& g){
    const Action& src=at(routes,descs[route],step);Action out=src;out.n_units=n_units;
    int slot=-1;if(step==116)slot=0;else if(step==120)slot=1;else if(step==121)slot=2;else if(step==122)slot=3;
    if(slot<0)return out;
    Order keep[MAX_ORDERS];int nk=0;for(int i=0;i<src.n_orders && i<MAX_ORDERS;++i)if(!is_wheat_sell(src.orders[i]))keep[nk++]=src.orders[i];
    out.n_orders=0;const int q=g.q[slot];
    auto add_wheat=[&](){if(q>0 && out.n_orders<MAX_ORDERS)out.orders[out.n_orders++]={M_SELL,WHEAT,q};};
    if(step==120 && q>0 && !g.wheat_first120){
        if(nk>0)out.orders[out.n_orders++]=keep[0];add_wheat();for(int i=1;i<nk && out.n_orders<MAX_ORDERS;++i)out.orders[out.n_orders++]=keep[i];
    }else{
        add_wheat();for(int i=0;i<nk && out.n_orders<MAX_ORDERS;++i)out.orders[out.n_orders++]=keep[i];
    }
    return out;
}

__global__ void timing_kernel(Config base,DeviceTables tables,const Action*routes,const RouteDesc*descs,int n_routes,
                              const TimingGene*genes,int n_genes,int unit_route,int opponent,int seat,
                              uint64_t seed_start,int n_seeds,Sim*sims,double*rewards){
    const int id=blockIdx.x*blockDim.x+threadIdx.x;const int total=n_genes*n_seeds;if(id>=total)return;const int gi=id/n_seeds,si=id-gi*n_seeds;
    Config cfg=base;cfg.seed=seed_start+(uint64_t)si;Sim&sim=sims[id];sim.init(cfg,&tables);const int os=1-seat;
    while(!sim.st.done){int step=sim.st.step;Action cand=patched_action(routes,descs,unit_route,step,sim.st.farms[seat].n_units,genes[gi]);Action opp=at(routes,descs[opponent],step);opp.n_units=sim.st.farms[os].n_units;if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);}
    rewards[id*2]=sim.st.farms[0].money;rewards[id*2+1]=sim.st.farms[1].money;
}

static std::vector<std::string> read_list(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list: "+p);std::vector<std::string>o;std::string l;while(std::getline(in,l)){auto h=l.find('#');if(h!=std::string::npos)l.resize(h);auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");o.push_back(l.substr(a,b-a+1));}return o;}
static double mean(const Fit&f){return f.cases?f.sum/f.cases:0.0;}
static bool better(const Fit&a,const Fit&b){if(a.wins!=b.wins)return a.wins>b.wins;if(a.ties!=b.ties)return a.ties<b.ties;if(a.floor!=b.floor)return a.floor>b.floor;return a.sum>b.sum;}

int main(int argc,char**argv){
    std::string list,out_path,gene_spec;int unit_route=1,total_wheat=18,seeds=16,opponents=21,tpb=64,topn=40;uint64_t seed_start=74800;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--routes")list=need();else if(a=="--unit-route")unit_route=std::stoi(need());else if(a=="--total-wheat")total_wheat=std::stoi(need());else if(a=="--seeds")seeds=std::stoi(need());else if(a=="--seed-start")seed_start=std::stoull(need());else if(a=="--opponents")opponents=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--top")topn=std::stoi(need());else if(a=="--out")out_path=need();else if(a=="--gene")gene_spec=need();else if(a=="--help"){std::cout<<"wheat_timing_search --routes LIST [--unit-route 1 --total-wheat 18 --gene q116,q120,q121,q122,first120 --seeds N --opponents N --seed-start S --top N --out CSV]\n";return 0;}else throw std::runtime_error("unknown arg "+a);}
    if(list.empty())throw std::runtime_error("--routes required");if(total_wheat<0||total_wheat>255)throw std::runtime_error("bad total wheat");auto paths=read_list(list);int n=paths.size();if(n<=0||unit_route<0||unit_route>=n)throw std::runtime_error("bad route");if(opponents>n)opponents=n;
    std::vector<NativeTape>t;for(auto&p:paths)t.push_back(load_native_tape_json(p));std::vector<Action>hr;std::vector<RouteDesc>hd;for(auto&x:t){int off=hr.size();hr.insert(hr.end(),x.actions.begin(),x.actions.end());hd.push_back({off,(int)x.actions.size()});}
    std::vector<TimingGene> genes;
    if(!gene_spec.empty()){
        int v[5]={0,0,0,0,0},k=0;std::string x;
        for(char ch:gene_spec){if(ch==','){if(k>=5||x.empty())throw std::runtime_error("bad --gene");v[k++]=std::stoi(x);x.clear();}else x.push_back(ch);}
        if(k!=4||x.empty())throw std::runtime_error("--gene needs q116,q120,q121,q122,first120");v[k]=std::stoi(x);
        TimingGene g{};for(int i=0;i<4;++i){if(v[i]<0||v[i]>255)throw std::runtime_error("bad gene quantity");g.q[i]=(uint8_t)v[i];}if(v[4]<0||v[4]>1)throw std::runtime_error("bad first120");g.wheat_first120=(uint8_t)v[4];genes.push_back(g);
    }else for(int a=0;a<=total_wheat;++a)for(int b=0;b<=total_wheat-a;++b)for(int c=0;c<=total_wheat-a-b;++c){int d=total_wheat-a-b-c;for(int first=0;first<2;++first){TimingGene g{};g.q[0]=a;g.q[1]=b;g.q[2]=c;g.q[3]=d;g.wheat_first120=first;genes.push_back(g);}}
    std::cout<<"genes="<<genes.size()<<" total_wheat="<<total_wheat<<"\n";
    configure_device();TablesOwner tables;Config cfg;Action*dr=nullptr;RouteDesc*dd=nullptr;TimingGene*dg=nullptr;Sim*ds=nullptr;double*dm=nullptr;const int G=genes.size(),jobs=G*seeds;
    cudaDeviceProp prop{};int dev=0;KAGCUDA_CHECK(cudaGetDevice(&dev));KAGCUDA_CHECK(cudaGetDeviceProperties(&prop,dev));size_t sim_bytes=(size_t)jobs*sizeof(Sim);if(sim_bytes>(size_t)(prop.totalGlobalMem*0.72))throw std::runtime_error("search exceeds safe GPU memory; reduce --seeds");
    KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(RouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&dg,G*sizeof(TimingGene)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,sim_bytes));KAGCUDA_CHECK(cudaMalloc((void**)&dm,(size_t)jobs*2*sizeof(double)));KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(RouteDesc),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dg,genes.data(),G*sizeof(TimingGene),cudaMemcpyHostToDevice));
    std::vector<double> m((size_t)jobs*2);std::vector<Fit> fit(G);
    for(int oi=0;oi<opponents;++oi)for(int seat=0;seat<2;++seat){timing_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dr,dd,n,dg,G,unit_route,oi,seat,seed_start,seeds,ds,dm);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(m.data(),dm,m.size()*sizeof(double),cudaMemcpyDeviceToHost));for(int g=0;g<G;++g)for(int s=0;s<seeds;++s){int id=g*seeds+s;double x=seat==0?m[id*2]-m[id*2+1]:m[id*2+1]-m[id*2];Fit&f=fit[g];++f.cases;if(x>0)++f.wins;else if(x==0)++f.ties;f.sum+=x;if(x<f.floor)f.floor=x;}}
    std::vector<int> top;top.reserve(topn+1);for(int g=0;g<G;++g){if((int)top.size()<topn||better(fit[g],fit[top.back()])){top.push_back(g);int j=top.size()-1;while(j>0&&better(fit[top[j]],fit[top[j-1]])){int z=top[j];top[j]=top[j-1];top[j-1]=z;--j;}if((int)top.size()>topn)top.pop_back();}}
    std::ostream*osp=&std::cout;std::ofstream out;if(!out_path.empty()){out.open(out_path);if(!out)throw std::runtime_error("cannot open out");osp=&out;}*osp<<"rank,q116,q120,q121,q122,wheat_first120,wins,cases,ties,floor,mean\n";for(int i=0;i<(int)top.size();++i){int g=top[i];auto&x=genes[g];auto&f=fit[g];*osp<<i+1<<','<<int(x.q[0])<<','<<int(x.q[1])<<','<<int(x.q[2])<<','<<int(x.q[3])<<','<<int(x.wheat_first120)<<','<<f.wins<<','<<f.cases<<','<<f.ties<<','<<f.floor<<','<<mean(f)<<'\n';if(i<10)std::cout<<"RANK "<<i+1<<" q="<<int(x.q[0])<<','<<int(x.q[1])<<','<<int(x.q[2])<<','<<int(x.q[3])<<" first120="<<int(x.wheat_first120)<<" wins="<<f.wins<<'/'<<f.cases<<" ties="<<f.ties<<" floor="<<f.floor<<" mean="<<mean(f)<<'\n';}
    if(out)std::cout<<"WROTE "<<out_path<<"\n";cudaFree(dr);cudaFree(dd);cudaFree(dg);cudaFree(ds);cudaFree(dm);return 0;
}
