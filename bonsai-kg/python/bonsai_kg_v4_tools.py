from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Callable,Dict,Iterable,List,Sequence,Set,Tuple
import bonsai_kg_v4 as b

@dataclass
class ProfileAggregate:
    ticks:int=0
    visits:List[int]=field(default_factory=list)
    max_depth:int=0
    max_live_intents:int=0
    feature_evals:int=0
    node_visits:Dict[int,int]=field(default_factory=dict)
    activated:Set[int]=field(default_factory=set)
    def add(self,p:b.TickProfile)->None:
        self.ticks+=1;self.visits.append(p.visits);self.max_depth=max(self.max_depth,p.max_depth);self.max_live_intents=max(self.max_live_intents,p.max_live_intents);self.feature_evals+=p.feature_evals
        for k,v in p.node_visits.items():self.node_visits[k]=self.node_visits.get(k,0)+v;self.activated.add(k)
    def percentile(self,q:float)->float:
        if not self.visits:return 0.0
        xs=sorted(self.visits);pos=(len(xs)-1)*q;lo=int(pos);hi=min(lo+1,len(xs)-1);w=pos-lo;return xs[lo]*(1-w)+xs[hi]*w

def policy_quality_report(cp:b.CompiledPolicy,profiles:ProfileAggregate|None=None)->Dict[str,Any]:
    out={'nodes':len(cp.nodes),'state_slots':cp.state_count,'bb_slots':len(cp.bb),'values':len(cp.values),'feature_args':len(cp.feature_args),'predicates':len(cp.predicates),'lifecycles':len(cp.lifecycles),'effects':len(cp.effects),'max_depth':cp.max_depth,'static_max_visits':cp.analysis[0].max_visits,'max_live_intents':cp.max_live_intents,'module_count':len(cp.manifest.get('modules',[])),'warnings':cp.warnings}
    if profiles:
        cold=len(cp.nodes)-len(profiles.activated);out.update(observed_ticks=profiles.ticks,observed_p50_visits=profiles.percentile(.5),observed_p95_visits=profiles.percentile(.95),observed_p99_visits=profiles.percentile(.99),observed_max_visits=max(profiles.visits,default=0),observed_max_depth=profiles.max_depth,observed_max_live_intents=profiles.max_live_intents,cold_nodes=cold,cold_fraction=cold/len(cp.nodes) if cp.nodes else 0.0,hot_nodes=sorted(profiles.node_visits.items(),key=lambda x:(-x[1],x[0]))[:32])
    return out

def profile_policy(cp:b.CompiledPolicy,observations:Iterable[Any],max_market_orders:int=10)->ProfileAggregate:
    vm=b.VM(cp);agg=ProfileAggregate()
    for o in observations:
        r=vm.tick(o,max_market_orders,profile=True);agg.add(r.profile)
    return agg

def metamorphic_stability(cp:b.CompiledPolicy,base_observation:Any,variants:Sequence[Any],projection:Callable[[b.TickResult],Any]|None=None)->Dict[str,Any]:
    """Compare a fresh-policy decision under base vs nearby observations.

    Strategy-level callers should supply reachable/engine-consistent variants. VM tests may
    use synthetic frames. `projection` defaults to status+action and can instead select a
    branch/module trace for causal-invariance testing.
    """
    proj=projection or (lambda r:(r.status,r.action.compact()))
    def one(o):return proj(b.VM(cp).tick(o,trace=True))
    expected=one(base_observation);m=[]
    for i,o in enumerate(variants):
        got=one(o)
        if got!=expected:m.append({'variant':i,'expected':expected,'got':got})
    return {'variants':len(variants),'mismatches':len(m),'details':m[:16]}
