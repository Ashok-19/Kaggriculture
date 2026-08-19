from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'python'))
import bonsai_kg_v4 as bkg
from kaggle_environments import make

def line(t,res):
    a=res.action;units=';'.join(f'{op},{arg},{n}' for op,arg,n in a.units);market=';'.join(f'{op},{item},{n}' for op,item,n in a.market);bb=','.join(str(x) for x in res.blackboard);ns=','.join(f'{x}/{y}' for x,y in res.node_state);return f'C=0 T={t} S={res.status} F={res.fault} U={a.n_units}:{units} M={len(a.market)}:{market} B={bb} N={ns}'

def main(args):
    if args.seat not in (0,1):raise ValueError('seat must be 0/1')
    cp=bkg.compile_policy(bkg.load_policy_json(args.policy));vm=bkg.VM(cp);records=[];last=-1;seed_visible=[]
    def agent(observation):
        nonlocal last
        step=int(observation.get('step',0) or 0)
        if step<=last:vm.reset()
        last=step;seed_visible.append('seed' in observation);r=vm.tick(observation,args.max_orders);records.append(line(step,r));return bkg.to_official_action(r.action)
    env=make('kaggriculture',configuration={'episodeSteps':args.steps,'seed':args.seed,'maxMarketOrdersPerTurn':args.max_orders},debug=True);env.run([agent,'pass'] if args.seat==0 else ['pass',agent]);print('\n'.join(records));final=env.steps[-1];print(json.dumps({'official_seed':env.info.get('seed'),'ticks':len(records),'seat':args.seat,'rewards':[s.reward for s in final],'seed_exposed_to_agent':any(seed_visible)},sort_keys=True),file=sys.stderr);return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('policy');ap.add_argument('--seed',type=int,default=12345);ap.add_argument('--steps',type=int,default=16);ap.add_argument('--max-orders',type=int,default=10);ap.add_argument('--seat',type=int,default=0);raise SystemExit(main(ap.parse_args()))
