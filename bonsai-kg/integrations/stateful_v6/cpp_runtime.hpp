#pragma once

#include "route_pack.hpp"
#include "../../cpp/kag_adapter.hpp"
#include "../../include/bonsai_kg.hpp"
#include <stdexcept>
#include <string>

namespace bonsaikg_v6 {

struct CppCompatState {
    bonsaikg::RuntimeState controller{};
    int last_step = -1;
};

inline kag::UnitAction to_cpp_unit(const PackedUnit& p) {
    kag::UnitAction out{}; out.op=p.op; out.arg=p.arg; out.n=p.n; return out;
}
inline kag::Order to_cpp_order(const PackedOrder& p) {
    kag::Order out{}; out.op=p.op; out.item=p.item; out.n=p.n; return out;
}
inline void materialize_cpp(const PackedAction& src, int n_units, kag::Action& out) {
    if (n_units < 1) n_units = 1;
    if (n_units > kag::MAX_UNITS) n_units = kag::MAX_UNITS;
    out = kag::Action{}; out.n_units=n_units; out.units[0]=to_cpp_unit(src.farmer);
    const int hands=n_units-1;
    for(int i=0;i<hands;++i) out.units[i+1]=i<src.saved_hands?to_cpp_unit(src.hands[i]):kag::UnitAction{};
    out.n_orders=src.n_market;
    for(int i=0;i<out.n_orders;++i) out.orders[i]=to_cpp_order(src.market[i]);
}

class CppCompatAgent {
public:
    CppCompatAgent(const std::string& controller_path, const std::string& pack_path)
        : owned_(bonsaikg::load_compiled(controller_path)), tree_(owned_.view()), pack_(RoutePack::load(pack_path)) {
        if (controller_hash_hex(owned_) != CONTROLLER_SEMANTIC_HASH) throw std::runtime_error("BKG3-v6 controller identity mismatch");
        if (tree_.max_intents != 0) throw std::runtime_error("BKG3-v6 controller unexpectedly emits game intents");
    }
    void reset(CppCompatState& s) const { bonsaikg::reset_runtime(tree_,s.controller); s.last_step=-1; }
    void act(CppCompatState& s, const kag::Sim& sim, int player, kag::Action& out) const {
        const int step=sim.st.step;
        if (step==0 || step<=s.last_step) reset(s);
        s.last_step=step;
        bonsaikg::CppObservation obs(&sim,player); bonsaikg::ActionResult ignored{};
        const bonsaikg::Status st=bonsaikg::tick(tree_,s.controller,obs,0,ignored,nullptr);
        if (s.controller.fault) throw std::runtime_error("BKG3-v6 controller VM fault " + std::to_string(s.controller.fault));
        if (st!=bonsaikg::SUCCESS) throw std::runtime_error("BKG3-v6 controller returned non-success");
        const int route=s.controller.blackboard[1];
        materialize_cpp(pack_.at(static_cast<uint16_t>(route),step),sim.st.farms[player].n_units,out);
    }
    const RoutePack& pack() const { return pack_; }
    const bonsaikg::OwnedPolicy& controller() const { return owned_; }
private:
    bonsaikg::OwnedPolicy owned_;
    bonsaikg::PolicyView tree_{};
    RoutePack pack_;
};

} // namespace bonsaikg_v6
