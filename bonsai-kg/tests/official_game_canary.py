from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg as bkg
from kaggle_environments import make


def line(t,res):
    a=res.action;units=';'.join(f"{op},{arg},{n}" for op,arg,n in a.units);market=';'.join(f"{op},{item},{n}" for op,item,n in a.market);bb=','.join(str(x) for x in res.blackboard);ns=','.join(f"{x}/{y}" for x,y in res.node_state)
    return f"C=0 T={t} S={res.status} F={res.fault} U={a.n_units}:{units} M={len(a.market)}:{market} B={bb} N={ns}"


def main(args):
    source=bkg.load_policy_json(args.policy);cp=bkg.compile_policy(source);vm=bkg.VM(cp);records=[];last_step=-1;seed_visible=[]
    def agent(observation):
        nonlocal last_step
        step=int(observation.get("step",0) or 0)
        if step<=last_step:vm.reset()
        last_step=step
        seed_visible.append("seed" in observation)
        res=vm.tick(observation,max_market_orders=args.max_orders,trace=False);records.append(line(step,res));return bkg.to_official_action(res.action)
    env=make("kaggriculture",configuration={"episodeSteps":args.steps,"seed":args.seed,"maxMarketOrdersPerTurn":args.max_orders},debug=True)
    env.run([agent,"pass"])
    for x in records:print(x)
    final=env.steps[-1]
    rewards=[s.reward for s in final]
    print(json.dumps({"official_seed":env.info.get("seed"),"ticks":len(records),"rewards":rewards,"seed_exposed_to_agent":any(seed_visible)},sort_keys=True),file=sys.stderr)
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("policy");ap.add_argument("--seed",type=int,default=12345);ap.add_argument("--steps",type=int,default=16);ap.add_argument("--max-orders",type=int,default=10);raise SystemExit(main(ap.parse_args()))
