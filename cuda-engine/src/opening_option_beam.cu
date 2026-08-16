#include "runtime.cuh"
#include "bonsai_gp.cuh"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace kagcuda;

constexpr int MAX_OPENING_DEPTH = 24;
struct CandidateDesc { uint8_t farmer_build=0, hires=4, cows=1, sheep=2, wheat_seed=5, melon_seed=5, wheat_buy=6, order_template=0; };
struct PrefixGenome { int root_id=0; uint8_t depth=1; uint8_t donors[MAX_OPENING_DEPTH-1]{}; };
struct Score { int wins=0,cases=0; double sum=0.0,floor=DBL_MAX; };
struct Ranked { int idx=0; double min_wr=0.0,avg_wr=0.0,mean_sum=0.0,min_floor=0.0; };

static std::vector<std::string> read_list(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open list: "+p);std::vector<std::string>o;std::string l;while(std::getline(in,l)){auto h=l.find('#');if(h!=std::string::npos)l.resize(h);auto a=l.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=l.find_last_not_of(" \t\r\n");o.push_back(l.substr(a,b-a+1));}return o;}
static std::vector<int> parse_ints(const std::string&s){std::vector<int>o;std::stringstream ss(s);std::string x;while(std::getline(ss,x,','))if(!x.empty())o.push_back(std::stoi(x));return o;}
static void push(Action&a,uint8_t op,uint8_t item,int n){if(a.n_orders>=MAX_ORDERS||n<=0)return;a.orders[a.n_orders++]={op,item,n};}
static Action make_first(const CandidateDesc& d){Action a{};a.n_units=1;a.units[0].op=d.farmer_build?OP_BUILD_PASTURE:OP_PASS;if(d.order_template==0){for(int i=0;i<d.hires;++i)push(a,M_HIRE,0,1);push(a,M_BUY_ANIMAL,COW,d.cows);push(a,M_BUY_ANIMAL,SHEEP,d.sheep);push(a,M_BUY_SEED,WHEAT,d.wheat_seed);push(a,M_BUY_SEED,MELON,d.melon_seed);push(a,M_BUY_PRODUCT,WHEAT,d.wheat_buy);}else{push(a,M_BUY_PRODUCT,WHEAT,d.wheat_buy);for(int i=0;i<d.hires;++i)push(a,M_HIRE,0,1);push(a,M_BUY_ANIMAL,COW,d.cows);push(a,M_BUY_ANIMAL,SHEEP,d.sheep);push(a,M_BUY_SEED,MELON,d.melon_seed);push(a,M_BUY_SEED,WHEAT,d.wheat_seed);}return a;}
static std::vector<Action> make_wide_first(){std::vector<Action> out;for(int fb=0;fb<2;++fb)for(int h:{3,4,5})for(int c:{1,2})for(int sh:{2,3,4})for(int ws:{4,5,6,7,8})for(int ms:{4,5,8,10,12})for(int wb:{4,6,8,10})for(int ot=0;ot<2;++ot){CandidateDesc d{};d.farmer_build=fb;d.hires=h;d.cows=c;d.sheep=sh;d.wheat_seed=ws;d.melon_seed=ms;d.wheat_buy=wb;d.order_template=ot;out.push_back(make_first(d));}return out;}

__global__ void beam_kernel(Config base,DeviceTables tables,const Action* first_all,int n_first,
                            const PrefixGenome* genomes,int n_genomes,
                            const Action* routes,const GPRouteDesc* descs,int n_routes,
                            int continuation_route,int opponent,int seat,bool hybrid_mode,int hybrid_anti_route,int hybrid_main_route,int hybrid_unit_bits,uint64_t seed_start,int n_seeds,
                            Sim* sims,double* rewards){
    const int id=blockIdx.x*blockDim.x+threadIdx.x,total=n_genomes*n_seeds;if(id>=total)return;
    const int gi=id/n_seeds,si=id-gi*n_seeds;const PrefixGenome& g=genomes[gi];Config cfg=base;cfg.seed=seed_start+(uint64_t)si;Sim& sim=sims[id];sim.init(cfg,&tables);const int os=1-seat;
    while(!sim.st.done){const int st=sim.st.step;Action cand{};
        if(st==0){const int rid=(g.root_id>=0&&g.root_id<n_first)?g.root_id:0;cand=gp_copy_action(first_all[rid],sim.st.farms[seat].n_units);}
        else if(st<(int)g.depth){
            if(hybrid_mode){
                const int mask=(int)g.donors[st-1];
                const Action& aa=gp_route_at(routes,descs[hybrid_anti_route],st);
                const Action& bb=gp_route_at(routes,descs[hybrid_main_route],st);
                cand=gp_copy_action(aa,sim.st.farms[seat].n_units);
                const int lim=sim.st.farms[seat].n_units<hybrid_unit_bits?sim.st.farms[seat].n_units:hybrid_unit_bits;
                for(int u=0;u<lim;++u)if(mask&(1<<u)){
                    if(u<bb.n_units)cand.units[u]=bb.units[u];
                    else cand.units[u]=UnitAction{};
                }
                if(mask&(1<<hybrid_unit_bits)){
                    cand.n_orders=bb.n_orders;
                    for(int k=0;k<bb.n_orders&&k<MAX_ORDERS;++k)cand.orders[k]=bb.orders[k];
                }
            }else{
                const int dr=(int)g.donors[st-1];const int rr=(dr>=0&&dr<n_routes)?dr:continuation_route;cand=gp_copy_action(gp_route_at(routes,descs[rr],st),sim.st.farms[seat].n_units);
            }
        }else cand=gp_copy_action(gp_route_at(routes,descs[continuation_route],st),sim.st.farms[seat].n_units);
        Action opp=gp_copy_action(gp_route_at(routes,descs[opponent],st),sim.st.farms[os].n_units);
        if(seat==0)sim.step_with_unit_counts(cand,sim.st.farms[0].n_units,opp,sim.st.farms[1].n_units);else sim.step_with_unit_counts(opp,sim.st.farms[0].n_units,cand,sim.st.farms[1].n_units);
    }
    rewards[id*2]=sim.st.farms[0].money;rewards[id*2+1]=sim.st.farms[1].money;
}

static bool better_rank(const Ranked&a,const Ranked&b){if(a.min_wr!=b.min_wr)return a.min_wr>b.min_wr;if(a.avg_wr!=b.avg_wr)return a.avg_wr>b.avg_wr;if(a.mean_sum!=b.mean_sum)return a.mean_sum>b.mean_sum;if(a.min_floor!=b.min_floor)return a.min_floor>b.min_floor;return a.idx<b.idx;}
static void insert_ranked(std::vector<Ranked>&v,const Ranked&r,int cap){int pos=(int)v.size();v.push_back(r);while(pos>0&&better_rank(v[pos],v[pos-1])){Ranked t=v[pos];v[pos]=v[pos-1];v[pos-1]=t;--pos;}if((int)v.size()>cap)v.pop_back();}
static std::string genome_text(const PrefixGenome&g){std::ostringstream o;o<<"root="<<g.root_id<<" depth="<<(int)g.depth<<" donors=";for(int s=1;s<(int)g.depth;++s){if(s>1)o<<':' ;o<<(int)g.donors[s-1];}return o.str();}

int main(int argc,char**argv){
    std::string routes_path,root_arg,donor_arg,anti_arg,main_arg,out_path;int anti_route=50,main_route=83,seeds=16,beam=64,max_depth=8,tpb=128,hybrid_main_route=-1,hybrid_unit_bits=6;uint64_t seed_start=872000;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--routes")routes_path=need();else if(a=="--root-ids")root_arg=need();else if(a=="--donor-routes")donor_arg=need();else if(a=="--hybrid-main-route")hybrid_main_route=std::stoi(need());else if(a=="--hybrid-unit-bits")hybrid_unit_bits=std::stoi(need());else if(a=="--anti-opponents")anti_arg=need();else if(a=="--main-opponents")main_arg=need();else if(a=="--anti-route")anti_route=std::stoi(need());else if(a=="--main-route")main_route=std::stoi(need());else if(a=="--seeds")seeds=std::stoi(need());else if(a=="--seed-start")seed_start=std::stoull(need());else if(a=="--beam")beam=std::stoi(need());else if(a=="--max-depth")max_depth=std::stoi(need());else if(a=="--tpb")tpb=std::stoi(need());else if(a=="--out")out_path=need();else if(a=="--help"){std::cout<<"opening_option_beam --routes LIST --root-ids csv --donor-routes csv --anti-opponents csv --main-opponents csv [--anti-route 50 --main-route 83 --donor-routes csv | --hybrid-main-route 83 --hybrid-unit-bits 6 --beam 64 --max-depth 8 --seeds 16 --out CSV]\n";return 0;}else throw std::runtime_error("unknown arg: "+a);}
    const bool hybrid_mode=hybrid_main_route>=0;if(routes_path.empty()||root_arg.empty()||(!hybrid_mode&&donor_arg.empty())||anti_arg.empty()||main_arg.empty())throw std::runtime_error("missing required args");if(max_depth<1||max_depth>MAX_OPENING_DEPTH)throw std::runtime_error("bad --max-depth");if(hybrid_mode&&(hybrid_unit_bits<1||hybrid_unit_bits>7))throw std::runtime_error("bad --hybrid-unit-bits");
    auto paths=read_list(routes_path);auto root_ids=parse_ints(root_arg);auto donors=hybrid_mode?std::vector<int>{}:parse_ints(donor_arg);auto anti_opps=parse_ints(anti_arg);auto main_opps=parse_ints(main_arg);if(root_ids.empty()||(!hybrid_mode&&donors.empty())||anti_opps.empty()||main_opps.empty())throw std::runtime_error("empty search list");if(hybrid_mode){if(hybrid_main_route<0||hybrid_main_route>=(int)paths.size())throw std::runtime_error("bad hybrid main route");const int nmask=1<<(hybrid_unit_bits+1);for(int m=0;m<nmask;++m)donors.push_back(m);}
    std::vector<NativeTape> tapes;for(auto&p:paths)tapes.push_back(load_native_tape_json(p));std::vector<Action>hr;std::vector<GPRouteDesc>hd;for(auto&t:tapes){int o=hr.size();hr.insert(hr.end(),t.actions.begin(),t.actions.end());hd.push_back({o,(int)t.actions.size()});}
    auto first=make_wide_first();for(int x:root_ids)if(x<0||x>=(int)first.size())throw std::runtime_error("bad root id");if(!hybrid_mode)for(int x:donors)if(x<0||x>=(int)paths.size())throw std::runtime_error("bad donor route");
    std::vector<PrefixGenome> current;for(int rid:root_ids){PrefixGenome g{};g.root_id=rid;g.depth=1;current.push_back(g);}const int max_candidates=beam*(int)donors.size()> (int)current.size()?beam*(int)donors.size():(int)current.size();const int max_jobs=max_candidates*seeds;
    configure_device();TablesOwner tables;Config cfg;Action*df=nullptr,*dr=nullptr;GPRouteDesc*dd=nullptr;PrefixGenome*dg=nullptr;Sim*ds=nullptr;double*drew=nullptr;
    KAGCUDA_CHECK(cudaMalloc((void**)&df,first.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dr,hr.size()*sizeof(Action)));KAGCUDA_CHECK(cudaMalloc((void**)&dd,hd.size()*sizeof(GPRouteDesc)));KAGCUDA_CHECK(cudaMalloc((void**)&dg,max_candidates*sizeof(PrefixGenome)));KAGCUDA_CHECK(cudaMalloc((void**)&ds,(size_t)max_jobs*sizeof(Sim)));KAGCUDA_CHECK(cudaMalloc((void**)&drew,(size_t)max_jobs*2*sizeof(double)));KAGCUDA_CHECK(cudaMemcpy(df,first.data(),first.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dr,hr.data(),hr.size()*sizeof(Action),cudaMemcpyHostToDevice));KAGCUDA_CHECK(cudaMemcpy(dd,hd.data(),hd.size()*sizeof(GPRouteDesc),cudaMemcpyHostToDevice));
    std::vector<double>host((size_t)max_jobs*2);std::ofstream fout;if(!out_path.empty()){fout.open(out_path);if(!fout)throw std::runtime_error("cannot open out");fout<<"depth,rank,root_id,donors,anti_wins,anti_cases,anti_mean,anti_floor,main_wins,main_cases,main_mean,main_floor,min_wr,avg_wr\n";}
    auto evaluate=[&](const std::vector<PrefixGenome>&cand,std::vector<Score>&anti,std::vector<Score>&main){const int n=(int)cand.size(),jobs=n*seeds;anti.assign(n,Score{});main.assign(n,Score{});KAGCUDA_CHECK(cudaMemcpy(dg,cand.data(),n*sizeof(PrefixGenome),cudaMemcpyHostToDevice));for(int lane=0;lane<2;++lane){const auto&opps=lane==0?anti_opps:main_opps;const int cont=lane==0?anti_route:main_route;auto&sc=lane==0?anti:main;for(int oi:opps)for(int seat=0;seat<2;++seat){beam_kernel<<<(jobs+tpb-1)/tpb,tpb>>>(cfg,tables.dev,df,(int)first.size(),dg,n,dr,dd,(int)paths.size(),cont,oi,seat,hybrid_mode,anti_route,hybrid_main_route,hybrid_unit_bits,seed_start,seeds,ds,drew);KAGCUDA_CHECK(cudaGetLastError());KAGCUDA_CHECK(cudaDeviceSynchronize());KAGCUDA_CHECK(cudaMemcpy(host.data(),drew,(size_t)jobs*2*sizeof(double),cudaMemcpyDeviceToHost));for(int gi=0;gi<n;++gi)for(int si=0;si<seeds;++si){const int id=gi*seeds+si;const double m=seat==0?host[id*2]-host[id*2+1]:host[id*2+1]-host[id*2];Score&s=sc[gi];++s.cases;s.wins+=m>0;s.sum+=m;if(m<s.floor)s.floor=m;}}}};
    std::cout<<"MODE "<<(hybrid_mode?"hybrid":"donor")<<" branch_choices="<<donors.size()<<" hybrid_main="<<hybrid_main_route<<" unit_bits="<<hybrid_unit_bits<<"\n";
    for(int depth=1;depth<=max_depth;++depth){std::vector<Score>anti,main;evaluate(current,anti,main);std::vector<Ranked>ranked;const int cap=beam<(int)current.size()?beam:(int)current.size();for(int i=0;i<(int)current.size();++i){const double aw=(double)anti[i].wins/anti[i].cases,mw=(double)main[i].wins/main[i].cases;Ranked r{};r.idx=i;r.min_wr=aw<mw?aw:mw;r.avg_wr=(aw+mw)*0.5;r.mean_sum=anti[i].sum/anti[i].cases+main[i].sum/main[i].cases;r.min_floor=anti[i].floor<main[i].floor?anti[i].floor:main[i].floor;insert_ranked(ranked,r,cap);}std::cout<<"DEPTH "<<depth<<" candidates="<<current.size()<<"\n";for(int k=0;k<(20<(int)ranked.size()?20:(int)ranked.size());++k){const int i=ranked[k].idx;std::cout<<"TOP "<<k<<' '<<genome_text(current[i])<<" anti="<<anti[i].wins<<'/'<<anti[i].cases<<" amean="<<anti[i].sum/anti[i].cases<<" main="<<main[i].wins<<'/'<<main[i].cases<<" mmean="<<main[i].sum/main[i].cases<<" minwr="<<ranked[k].min_wr<<"\n";}if(fout)for(int k=0;k<(int)ranked.size();++k){const int i=ranked[k].idx;fout<<depth<<','<<k<<','<<current[i].root_id<<",\"";for(int s=1;s<(int)current[i].depth;++s){if(s>1)fout<<':';fout<<(int)current[i].donors[s-1];}fout<<"\","<<anti[i].wins<<','<<anti[i].cases<<','<<anti[i].sum/anti[i].cases<<','<<anti[i].floor<<','<<main[i].wins<<','<<main[i].cases<<','<<main[i].sum/main[i].cases<<','<<main[i].floor<<','<<ranked[k].min_wr<<','<<ranked[k].avg_wr<<'\n';}
        if(depth==max_depth)break;std::vector<PrefixGenome>next;next.reserve(ranked.size()*donors.size());for(const auto&r:ranked){const PrefixGenome&p=current[r.idx];for(int drt:donors){PrefixGenome c=p;c.donors[depth-1]=(uint8_t)drt;c.depth=(uint8_t)(depth+1);next.push_back(c);}}current.swap(next);
    }
    cudaFree(df);cudaFree(dr);cudaFree(dd);cudaFree(dg);cudaFree(ds);cudaFree(drew);return 0;
}
