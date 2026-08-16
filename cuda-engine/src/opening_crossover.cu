#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kagcuda;

struct PrefixSummary {
    uint64_t lo = 0, hi = 0;
    int money = 0;
    int shed_total = 0;
    uint8_t crops[N_CROPS] = {0};
    uint8_t animals[N_ANIMALS] = {0};
    uint8_t pastures = 0, coops = 0, weeds = 0, land = 0;
};

struct RankedPrefix { uint64_t mask; int da,db,mx,sum; bool macro; };
static bool ranked_prefix_less(const RankedPrefix& a,const RankedPrefix& b){
    if(a.macro!=b.macro)return a.macro>b.macro;
    if(a.mx!=b.mx)return a.mx<b.mx;
    if(a.sum!=b.sum)return a.sum<b.sum;
    return a.mask<b.mask;
}

__device__ __forceinline__ uint8_t tile_cat(const Tile& t) {
    if (t.kind == T_EMPTY) return 0;
    if (t.kind == T_PLANT) {
        if (t.what == WHEAT) return 1;
        if (t.what == MELON) return 2;
        return 7;
    }
    if (t.kind == T_PASTURE) {
        if (!t.has_animal) return 3;
        if (t.what == COW) return 4;
        if (t.what == SHEEP) return 5;
        return 6;
    }
    return 7;
}

__device__ __forceinline__ PrefixSummary summarize_prefix(const Farm& f) {
    PrefixSummary z{};
    z.money = static_cast<int>(f.money);
    z.shed_total = f.shed_total;
    z.land = static_cast<uint8_t>(f.n_quadrants);
    int k = 0;
    for (int y=0;y<5;++y) for (int x=0;x<5;++x,++k) {
        const Tile& t=f.tiles[y][x];
        const uint8_t c=tile_cat(t);
        if (k < 21) z.lo |= static_cast<uint64_t>(c) << (3*k);
        else z.hi |= static_cast<uint64_t>(c) << (3*(k-21));
        if (t.kind==T_PLANT && t.what<N_CROPS) ++z.crops[t.what];
        else if (t.kind==T_PASTURE) {
            ++z.pastures;
            if (t.has_animal && t.what>=GOOSE && t.what<=SHEEP) ++z.animals[t.what-GOOSE];
        } else if (t.kind==T_COOP) {
            ++z.coops;
            if (t.has_animal && t.what>=GOOSE && t.what<=SHEEP) ++z.animals[t.what-GOOSE];
        } else if (t.kind==T_WEED) ++z.weeds;
    }
    return z;
}

__global__ void prefix_kernel(Config base, DeviceTables tables,
                              const Action* routes, const GPRouteDesc* descs,
                              int route_a, int route_b, int opponent, int seat,
                              int branch_step, const int* diff_index,
                              uint64_t mask_start, int n_masks,
                              Sim* sims, PrefixSummary* out) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n_masks)return;
    const uint64_t mask=mask_start+static_cast<uint64_t>(id);
    Sim& sim=sims[id];sim.init(base,&tables);
    const int os=1-seat;
    while(!sim.st.done && sim.st.step<branch_step){
        const int step=sim.st.step;
        const Action& aa=gp_route_at(routes,descs[route_a],step);
        const Action& bb=gp_route_at(routes,descs[route_b],step);
        Action cand=gp_copy_action(aa,sim.st.farms[seat].n_units);
        const int bit=diff_index[step];
        if(bit>=0 && ((mask>>bit)&1ull)) {
            Action alt=gp_copy_action(bb,sim.st.farms[seat].n_units);
            for(int u=0;u<cand.n_units;++u)cand.units[u]=alt.units[u];
        }
        Action opp=gp_copy_action(gp_route_at(routes,descs[opponent],step),sim.st.farms[os].n_units);
        if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);
        else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    out[id]=summarize_prefix(sim.st.farms[seat]);
}

static std::vector<std::string> read_list(const std::string& p){
    std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list: "+p);
    std::vector<std::string> out;std::string l;
    while(std::getline(in,l)){auto h=l.find('#');if(h!=std::string::npos)l.resize(h);auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");out.push_back(l.substr(a,b-a+1));}
    return out;
}

static bool same_units(const Action& a,const Action& b){
    if(a.n_units!=b.n_units)return false;
    for(int u=0;u<a.n_units;++u){
        if(a.units[u].op!=b.units[u].op || a.units[u].arg!=b.units[u].arg || a.units[u].n!=b.units[u].n)return false;
    }
    return true;
}

static uint8_t get_cat(const PrefixSummary& z,int k){
    if(k<21)return static_cast<uint8_t>((z.lo>>(3*k))&7ull);
    return static_cast<uint8_t>((z.hi>>(3*(k-21)))&7ull);
}
static int layout_dist(const PrefixSummary&a,const PrefixSummary&b){int d=0;for(int k=0;k<25;++k)d+=get_cat(a,k)!=get_cat(b,k);return d;}
static bool same_macro(const PrefixSummary&a,const PrefixSummary&b){
    if(a.money!=b.money||a.shed_total!=b.shed_total||a.pastures!=b.pastures||a.coops!=b.coops||a.weeds!=b.weeds||a.land!=b.land)return false;
    for(int i=0;i<N_CROPS;++i)if(a.crops[i]!=b.crops[i])return false;
    for(int i=0;i<N_ANIMALS;++i)if(a.animals[i]!=b.animals[i])return false;
    return true;
}

int main(int argc,char**argv){
    std::string list,out_path;int route_a=1,route_b=8,opponent=0,seat=0,branch_step=72,tpb=128,batch=32768,topn=64;uint64_t seed=72400;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
        if(a=="--routes")list=need();else if(a=="--route-a")route_a=std::stoi(need());else if(a=="--route-b")route_b=std::stoi(need());else if(a=="--opponent")opponent=std::stoi(need());else if(a=="--seat")seat=std::stoi(need());else if(a=="--branch-step")branch_step=std::stoi(need());else if(a=="--seed")seed=std::stoull(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--batch")batch=std::stoi(need());else if(a=="--top")topn=std::stoi(need());else if(a=="--out")out_path=need();else if(a=="--help"){std::cout<<"opening_crossover --routes LIST [--route-a 1 --route-b 8 --branch-step 72 --seed N --top N --out FILE]\n";return 0;}else throw std::runtime_error("unknown arg: "+a);
    }
    if(list.empty())throw std::runtime_error("--routes required");
    auto paths=read_list(list);if(route_a<0||route_b<0||route_a>=(int)paths.size()||route_b>=(int)paths.size()||opponent<0||opponent>=(int)paths.size())throw std::runtime_error("bad route index");
    std::vector<NativeTape> tapes;for(auto&p:paths)tapes.push_back(load_native_tape_json(p));
    std::vector<Action> hr;std::vector<GPRouteDesc> hd;for(auto&t:tapes){int off=hr.size();hr.insert(hr.end(),t.actions.begin(),t.actions.end());hd.push_back({off,(int)t.actions.size()});}
    std::vector<int> diff_index(branch_step,-1);int n_diff=0;
    for(int s=0;s<branch_step;++s){const auto& a=tapes[route_a].actions[s];const auto& b=tapes[route_b].actions[s];if(!same_units(a,b))diff_index[s]=n_diff++;}
    if(n_diff>=63)throw std::runtime_error("too many differing opening steps");
    const uint64_t total=1ull<<n_diff;
    std::cout<<"diff_steps="<<n_diff<<" total_masks="<<total<<" steps=";for(int s=0;s<branch_step;++s)if(diff_index[s]>=0)std::cout<<s<<',';std::cout<<"\n";

    configure_device();TablesOwner tables;Config cfg;cfg.seed=seed;
    Action*dr=nullptr;GPRouteDesc*dd=nullptr;int*ddi=nullptr;Sim*ds=nullptr;PrefixSummary*dout=nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&ddi,branch_step*sizeof(int)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)batch*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&dout,(size_t)batch*sizeof(PrefixSummary)));
    KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(ddi,diff_index.data(),branch_step*sizeof(int),cudaMemcpyHostToDevice));
    std::vector<PrefixSummary> all(total),buf(batch);
    for(uint64_t start=0;start<total;start+=batch){int n=(int)((total-start)<(uint64_t)batch?(total-start):(uint64_t)batch);prefix_kernel<<<(n+tpb-1)/tpb,tpb>>>(cfg,tables.dev,dr,dd,route_a,route_b,opponent,seat,branch_step,ddi,start,n,ds,dout);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(buf.data(),dout,(size_t)n*sizeof(PrefixSummary),cudaMemcpyDeviceToHost));for(int i=0;i<n;++i)all[start+(uint64_t)i]=buf[i];}
    const PrefixSummary& A=all[0];const PrefixSummary& B=all[total-1];
    std::cout<<"target_dist="<<layout_dist(A,B)<<" target_macro_equal="<<same_macro(A,B)<<" A_money="<<A.money<<" B_money="<<B.money<<"\n";
    std::vector<RankedPrefix> top;top.reserve(topn+1);int macro_count=0;
    for(uint64_t m=0;m<total;++m){
        int da=layout_dist(all[m],A),db=layout_dist(all[m],B);bool macro=same_macro(all[m],A);if(macro)++macro_count;
        RankedPrefix r{m,da,db,da>db?da:db,da+db,macro};
        if((int)top.size()<topn || ranked_prefix_less(r,top.back())){
            top.push_back(r);int j=(int)top.size()-1;while(j>0 && ranked_prefix_less(top[j],top[j-1])){RankedPrefix t=top[j];top[j]=top[j-1];top[j-1]=t;--j;}
            if((int)top.size()>topn)top.pop_back();
        }
    }
    std::cout<<"macro_preserving="<<macro_count<<"/"<<total<<" best_max_dist="<<top[0].mx<<" best_sum_dist="<<top[0].sum<<"\n";
    std::ostream* osp=&std::cout;std::ofstream fout;if(!out_path.empty()){fout.open(out_path);if(!fout)throw std::runtime_error("cannot open out");osp=&fout;}
    *osp<<"rank,mask,dist_a,dist_b,max_dist,sum_dist,macro,bits\n";
    for(int i=0;i<(int)top.size();++i){const auto&r=top[i];*osp<<i<<','<<r.mask<<','<<r.da<<','<<r.db<<','<<r.mx<<','<<r.sum<<','<<r.macro<<',';for(int b=0;b<n_diff;++b)*osp<<((r.mask>>b)&1ull);*osp<<'\n';}
    if(osp!=&std::cout){std::cout<<"WROTE "<<out_path<<" top="<<topn<<"\n";}
    cudaFree(dr);cudaFree(dd);cudaFree(ddi);cudaFree(ds);cudaFree(dout);return 0;
}
