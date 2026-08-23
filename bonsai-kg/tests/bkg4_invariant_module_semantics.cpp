#include "../include/bkg4_invariant_module.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace {

struct Unit { uint8_t op=0, arg=0; int16_t n=1; };
struct Order { uint8_t op=0, item=0; int32_t n=0; };
struct Action { int n_units=0; Unit units[40]{}; int n_orders=0; Order orders[16]{}; };

struct Obs {
    int money=0;
    int n_units=2;
    int tile0=7, tile1=8;
    int item7=10, item8=-1;
    int shed_milk=0;
    int wheat0=0, wheat1=1;
    int32_t feature(uint16_t id, int32_t a0=0, int32_t a1=0) const {
        using namespace bonsaikg4;
        if (id==F_OWN_UNITS) return n_units;
        if (id==F_OWN_MONEY) return money;
        if (id==F_OWN_WORKER_TILE) return a0==0?tile0:tile1;
        if (id==F_OWN_TILE_HAS_ANIMAL) return (a0==7 && item7==10) || (a0==8 && item8==10);
        if (id==F_OWN_TILE_ITEM) return a0==7?item7:item8;
        if (id==F_OWN_TILE_CONSECUTIVE_UNFED) return 0;
        if (id==F_OWN_TILE_PENDING_CARE_BONUS) return 0;
        if (id==F_OWN_SHED && a0==5) return shed_milk;
        if (id==F_OWN_INVENTORY && a1==0) return a0==0?wheat0:wheat1;
        return 0;
    }
};

using namespace bonsaikg4::invariant;

GuardBin G(uint16_t feature, uint8_t cmp, uint8_t mode, int arg, int value) {
    GuardBin g{}; g.feature_id=feature; g.cmp=cmp; g.arg_mode=mode; g.arg=arg; g.value=value; return g;
}

Candidate feed_candidate() {
    Candidate c{}; std::memcpy(c.header.magic,"B4IM",4); c.header.version=1; c.header.n_rules=1;
    auto& r=c.rules[0]; r.domain=DOMAIN_EACH_OWN_UNIT; r.control=CONTROL_INSTANT; r.action=ACTION_UNIT_REWRITE; r.n_guard=4; r.effect=2;
    r.p0=bonsaikg4::U_FEED; r.p1=bonsaikg4::U_PASS; r.p2=0; r.p3=1;
    r.guards[0]=G(bonsaikg4::F_OWN_TILE_HAS_ANIMAL,bonsaikg4::C_EQ,ARG_BOUND_TILE,0,1);
    r.guards[1]=G(bonsaikg4::F_OWN_TILE_ITEM,bonsaikg4::C_EQ,ARG_BOUND_TILE,0,10);
    r.guards[2]=G(bonsaikg4::F_OWN_TILE_CONSECUTIVE_UNFED,bonsaikg4::C_EQ,ARG_BOUND_TILE,0,0);
    r.guards[3]=G(bonsaikg4::F_OWN_TILE_PENDING_CARE_BONUS,bonsaikg4::C_EQ,ARG_BOUND_TILE,0,0);
    return c;
}

} // namespace

int main() {
    static_assert(sizeof(Candidate)==256, "B4IM1 candidate ABI must be 256 bytes");
    {
        auto c=feed_candidate(); RuntimeState s{}; Obs obs{}; auto d=evaluate(c,s,obs);
        assert(d.activations==1); assert((d.rewrite_masks[0]&1ull)!=0); assert((d.rewrite_masks[0]&2ull)==0);
        Action a{}; a.n_units=2; a.units[0].op=bonsaikg4::U_FEED; a.units[1].op=bonsaikg4::U_FEED;
        apply_post(c,d,a,10); assert(a.units[0].op==bonsaikg4::U_PASS); assert(a.units[1].op==bonsaikg4::U_FEED);
        Obs perm=obs; perm.tile0=8; perm.tile1=7; RuntimeState sp{}; auto dp=evaluate(c,sp,perm);
        assert((dp.rewrite_masks[0]&1ull)==0); assert((dp.rewrite_masks[0]&2ull)!=0);
    }
    {
        Candidate c{}; std::memcpy(c.header.magic,"B4IM",4); c.header.version=1; c.header.n_rules=1;
        auto& r=c.rules[0]; r.domain=DOMAIN_EACH_OWN_UNIT; r.control=CONTROL_INSTANT; r.action=ACTION_UNIT_REWRITE; r.n_guard=1; r.effect=6;
        r.p0=bonsaikg4::U_CARE; r.p1=bonsaikg4::U_FEED; r.p2=0; r.p3=1;
        r.guards[0]=G(bonsaikg4::F_OWN_INVENTORY,bonsaikg4::C_GE,ARG_CONST,0,1);
        RuntimeState s{}; Obs obs{}; auto d=evaluate(c,s,obs);
        assert((d.rewrite_masks[0]&1ull)==0); assert((d.rewrite_masks[0]&2ull)!=0);
    }
    {
        Candidate c{}; std::memcpy(c.header.magic,"B4IM",4); c.header.version=1; c.header.n_rules=1;
        auto& r=c.rules[0]; r.domain=DOMAIN_GLOBAL; r.control=CONTROL_INSTANT; r.action=ACTION_MARKET_INTENT; r.n_guard=1; r.effect=3;
        r.p0=bonsaikg4::M_SELL; r.p1=5; r.p2=2; r.p3=bonsaikg4::AUTH_COMMITTED_STRATEGY; r.p4=20;
        r.guards[0]=G(bonsaikg4::F_OWN_SHED,bonsaikg4::C_GE,ARG_CONST,5,2);
        Obs obs{};obs.shed_milk=3;RuntimeState s{};auto d=evaluate(c,s,obs);Action a{};a.n_orders=1;a.orders[0].op=bonsaikg4::M_HIRE;
        apply_post(c,d,a,10);assert(a.n_orders==2);assert(a.orders[0].op==bonsaikg4::M_SELL);assert(a.orders[0].item==5);assert(a.orders[0].n==2);
    }
    {
        Candidate c{};std::memcpy(c.header.magic,"B4IM",4);c.header.version=1;c.header.n_rules=1;
        auto& r=c.rules[0];r.domain=DOMAIN_GLOBAL;r.control=CONTROL_COMMITTED;r.action=ACTION_COMMITTED_ROUTE;r.n_guard=1;r.n_exit=1;r.effect=4;r.p0=70;
        r.guards[0]=G(bonsaikg4::F_OWN_MONEY,bonsaikg4::C_GE,ARG_CONST,0,6000);
        r.guards[1]=G(bonsaikg4::F_OWN_MONEY,bonsaikg4::C_LE,ARG_CONST,0,2500);
        RuntimeState s{};Obs o{};o.money=7000;auto d=evaluate(c,s,o);assert(d.route_override==70);o.money=4000;d=evaluate(c,s,o);assert(d.route_override==70);o.money=2000;d=evaluate(c,s,o);assert(d.route_override==bonsaikg4::NONE_U16);
    }
    return 0;
}
