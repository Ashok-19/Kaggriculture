// CPU-oracle side of the development-only deep state parity gate.
#include "../../cpp-engine/include/sim.hpp"
#include "logical_hash.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
using namespace kag;
struct TurnHost{Action a[2];};
static void read_config(std::istream&in,Config&cfg){const std::streampos pos=in.tellg();std::string tok;if(!(in>>tok))return;if(tok!="CONFIG"){in.clear();in.seekg(pos);return;}
    in>>cfg.episode_steps>>cfg.board_size>>cfg.starting_money>>cfg.max_orders>>cfg.turns_per_day>>cfg.shed_capacity>>cfg.weed_chance>>cfg.shop_unlock_interval>>cfg.shop_sell_interval>>cfg.center_sell_interval>>cfg.hire_mult;}
static bool load(const char*path,uint64_t&seed,Config&cfg,std::vector<TurnHost>&turns){std::ifstream in(path);if(!in)return false;int n=0;in>>seed>>n;read_config(in,cfg);turns.resize(n);
    for(int t=0;t<n;++t)for(int p=0;p<2;++p){int nu=0,no=0;in>>nu>>no;Action&a=turns[t].a[p];a.n_units=std::min(nu,MAX_UNITS);
        for(int i=0;i<nu;++i){int op,arg,cnt;in>>op>>arg>>cnt;if(i<MAX_UNITS)a.units[i]={static_cast<uint8_t>(op),static_cast<uint8_t>(arg),static_cast<int16_t>(cnt)};}
        a.n_orders=std::min(no,16);for(int i=0;i<no;++i){int op,item,cnt;in>>op>>item>>cnt;if(i<16)a.orders[i]={static_cast<uint8_t>(op),static_cast<uint8_t>(item),cnt};}}return true;}
static uint64_t hash_state(const State&s){KagLogicalHash q;q.add_int(s.step);q.add_int(s.day);q.add_int(s.hour);q.add_bool(s.done);q.add_int(s.n_shops);for(int i=0;i<s.n_shops;++i)q.add_int(s.shops[i]);
    for(int i=0;i<N_PRODUCTS;++i){q.add_int(s.market.inventory[i]);q.add_int(s.market.prices[i]);}
    for(int p=0;p<2;++p){const Farm&f=s.farms[p];q.add_double(f.money);q.add_int(f.n_units);q.add_int(f.n_quadrants);q.add_int(f.hires_today);q.add_int(f.next_decay_step);q.add_int(f.shed_total);q.add_double(f.sell_revenue);q.add_double(f.total_spend);
        for(int i=0;i<N_ITEMS;++i){q.add_int(f.shed[i]);q.add_int(f.discarded[i]);q.add_int(f.produced[i]);q.add_int(f.sold_units[i]);}for(int i=0;i<N_CROPS;++i)q.add_int(f.seeds[i]);
        for(int u=0;u<f.n_units;++u){q.add_int(f.pos_x[u]);q.add_int(f.pos_y[u]);q.add_int(f.inv_nkeys[u]);for(int k=0;k<f.inv_nkeys[u];++k)q.add_int(f.inv_keys[u][k]);for(int i=0;i<N_ITEMS;++i)q.add_int(f.inv[u][i]);}
        for(int y=0;y<BOARD;++y)for(int x=0;x<BOARD;++x){const Tile&t=f.tiles[y][x];q.add_int(t.kind);q.add_int(t.what);q.add_bool(t.has_animal);q.add_bool(t.watered_today);q.add_bool(t.fed_today);q.add_bool(t.cared_today);q.add_bool(t.fertilizer_available);
            q.add_int(t.consecutive_dry);q.add_int(t.yield_units);q.add_int(t.pending_care_bonus);q.add_int(t.planted_day);q.add_int(t.max_lifespan_step);q.add_int(t.fertilized_until_day);}}return q.h;}
int main(int argc,char**argv){if(argc!=2){std::fprintf(stderr,"usage: cpu_state_hash TRACE\n");return 2;}uint64_t seed=0;Config cfg;std::vector<TurnHost>turns;if(!load(argv[1],seed,cfg,turns))return 2;cfg.seed=seed;Sim sim(cfg);
    std::printf("0 %016llx\n",static_cast<unsigned long long>(hash_state(sim.st)));for(size_t t=0;t<turns.size();++t){sim.step(turns[t].a[0],turns[t].a[1]);std::printf("%zu %016llx\n",t+1,static_cast<unsigned long long>(hash_state(sim.st)));}return 0;}
