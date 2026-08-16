#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace kagcuda;

constexpr int N_PATTERNS=6;
__device__ __constant__ uint8_t D_PATTERN_MASK[N_PATTERNS]={0,5,9,6,10,15};
__device__ __constant__ int D_X[4]={1,2,1,1};
__device__ __constant__ int D_Y[4]={0,0,3,4};

__global__ void geometry_kernel(Config base, DeviceTables tables,
                                const Action* routes,const GPRouteDesc* descs,
                                int route_a,int route_b,int opponent,int seat,int branch_step,
                                uint64_t seed_start,int n_seeds,
                                Sim* sims,Sim* refs,double* margins,int* first_shop){
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int jobs=N_PATTERNS*2*n_seeds;if(id>=jobs)return;
    const int pi=id/(2*n_seeds);const int rem=id-pi*(2*n_seeds);const int ci=rem/n_seeds;const int si=rem-ci*n_seeds;
    Config cfg=base;cfg.seed=seed_start+(uint64_t)si;
    Sim& sim=sims[id];Sim& ref=refs[id];sim.init(cfg,&tables);ref.init(cfg,&tables);
    const int os=1-seat;
    while(!sim.st.done && sim.st.step<branch_step){
        const int step=sim.st.step;
        Action ca=gp_copy_action(gp_route_at(routes,descs[route_a],step),sim.st.farms[seat].n_units);
        Action cb=gp_copy_action(gp_route_at(routes,descs[route_b],step),ref.st.farms[seat].n_units);
        Action oa=gp_copy_action(gp_route_at(routes,descs[opponent],step),sim.st.farms[os].n_units);
        Action ob=gp_copy_action(gp_route_at(routes,descs[opponent],step),ref.st.farms[os].n_units);
        if(seat==0){sim.step_with_unit_counts(ca,sim.st.farms[0].n_units,oa,sim.st.farms[1].n_units);ref.step_with_unit_counts(cb,ref.st.farms[0].n_units,ob,ref.st.farms[1].n_units);}else{sim.step_with_unit_counts(oa,sim.st.farms[0].n_units,ca,sim.st.farms[1].n_units);ref.step_with_unit_counts(ob,ref.st.farms[0].n_units,cb,ref.st.farms[1].n_units);}
    }
    if(pi==0&&ci==0)first_shop[si]=sim.st.n_shops?sim.st.shops[0]:-1;
    const uint8_t mask=D_PATTERN_MASK[pi];
    for(int k=0;k<4;++k)if((mask>>k)&1u)sim.st.farms[seat].tiles[D_Y[k]][D_X[k]]=ref.st.farms[seat].tiles[D_Y[k]][D_X[k]];
    const int cr=ci?route_b:route_a;
    while(!sim.st.done){
        const int step=sim.st.step;
        Action cand=gp_copy_action(gp_route_at(routes,descs[cr],step),sim.st.farms[seat].n_units);
        Action opp=gp_copy_action(gp_route_at(routes,descs[opponent],step),sim.st.farms[os].n_units);
        if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    margins[id]=seat==0?sim.st.farms[0].money-sim.st.farms[1].money:sim.st.farms[1].money-sim.st.farms[0].money;
}

static std::vector<std::string> read_list(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list");std::vector<std::string>o;std::string l;while(std::getline(in,l)){auto h=l.find('#');if(h!=std::string::npos)l.resize(h);auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");o.push_back(l.substr(a,b-a+1));}return o;}
struct Stat{long long wins=0,cases=0;double sum=0,floor=1e300;void add(double m){++cases;if(m>0)++wins;sum+=m;if(m<floor)floor=m;}};
int main(int argc,char**argv){
    std::string list,out_path;int route_a=1,route_b=8,branch_step=72,seeds=64,opponents=21,tpb=64;uint64_t seed_start=74000;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value");return std::string(argv[++i]);};if(a=="--routes")list=need();else if(a=="--route-a")route_a=std::stoi(need());else if(a=="--route-b")route_b=std::stoi(need());else if(a=="--branch-step")branch_step=std::stoi(need());else if(a=="--seeds")seeds=std::stoi(need());else if(a=="--seed-start")seed_start=std::stoull(need());else if(a=="--opponents")opponents=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--out")out_path=need();else throw std::runtime_error("unknown arg "+a);}
    if(list.empty())throw std::runtime_error("--routes required");auto paths=read_list(list);if(opponents>(int)paths.size())opponents=paths.size();
    std::vector<NativeTape>t;for(auto&p:paths)t.push_back(load_native_tape_json(p));std::vector<Action>hr;std::vector<GPRouteDesc>hd;for(auto&x:t){int off=hr.size();hr.insert(hr.end(),x.actions.begin(),x.actions.end());hd.push_back({off,(int)x.actions.size()});}
    configure_device();TablesOwner tables;Config cfg;Action*dr=nullptr;GPRouteDesc*dd=nullptr;Sim*ds=nullptr;Sim*dref=nullptr;double*dm=nullptr;int*dshop=nullptr;const int jobs=N_PATTERNS*2*seeds;
    KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dref,(size_t)jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dm,(size_t)jobs*sizeof(double)));KAGCUDA_CHECK(cudaMalloc((void**)&dshop,(size_t)seeds*sizeof(int)));KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    std::vector<double> m(jobs);std::vector<int> shops(seeds);Stat st[N_PATTERNS][2],oracle[N_PATTERNS];std::ofstream out;if(!out_path.empty()){out.open(out_path);out<<"opponent,seat,seed,shop1,pattern,margin_a,margin_b\n";}
    for(int oi=0;oi<opponents;++oi)for(int seat=0;seat<2;++seat){geometry_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dr,dd,route_a,route_b,oi,seat,branch_step,seed_start,seeds,ds,dref,dm,dshop);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(m.data(),dm,(size_t)jobs*sizeof(double),cudaMemcpyDeviceToHost));KAGCUDA_CHECK(cudaMemcpy(shops.data(),dshop,(size_t)seeds*sizeof(int),cudaMemcpyDeviceToHost));for(int p=0;p<N_PATTERNS;++p)for(int si=0;si<seeds;++si){double a=m[(p*2+0)*seeds+si],b=m[(p*2+1)*seeds+si];st[p][0].add(a);st[p][1].add(b);oracle[p].add(a>b?a:b);if(out)out<<oi<<','<<seat<<','<<(seed_start+si)<<','<<shops[si]<<','<<p<<','<<a<<','<<b<<'\n';}}
    for(int p=0;p<N_PATTERNS;++p){std::cout<<"PATTERN "<<p<<" mask="<<int((uint8_t[]){0,5,9,6,10,15}[p]);for(int c=0;c<2;++c)std::cout<<" cont"<<(c?'B':'A')<<"="<<st[p][c].wins<<'/'<<st[p][c].cases<<" mean="<<(st[p][c].sum/st[p][c].cases)<<" floor="<<st[p][c].floor;std::cout<<" oracle="<<oracle[p].wins<<'/'<<oracle[p].cases<<" mean="<<(oracle[p].sum/oracle[p].cases)<<" floor="<<oracle[p].floor<<'\n';}
    if(out)std::cout<<"WROTE "<<out_path<<'\n';cudaFree(dr);cudaFree(dd);cudaFree(ds);cudaFree(dref);cudaFree(dm);cudaFree(dshop);return 0;
}
